import json
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.config import (
    AGNConfig,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    LikelihoodConfig,
    Observation,
    PhotometryData,
    SpectroscopyData,
    fit_config_from_mapping,
)
from jaxsedfit.line_extrapolation import (
    M_I_Z2_ZEROPOINT,
    continuum_ew_broad_line_luminosity,
    load_qsogen_line_ew_table,
    m_i_z2_from_disc_llambda_2500,
    qsogen_emline_type,
    qsogen_line_ew_matrix,
)
from jaxsedfit.model import (
    GRAHSP_PL_BEND_LOC_A,
    GRAHSP_PL_BEND_WIDTH,
    GRAHSP_PL_CUTOFF_A,
    grahsp_photometric_model,
)
from jaxsedfit.preload import build_model_context


REDSHIFT = 0.5
# Rest-frame C IV + He II/O III] band, fully outside the rest 2000-3000 A spectrum.
BAND_REST = (1440.0, 1720.0)


def _ew(name, emline_type=0.0):
    table = load_qsogen_line_ew_table()
    row = table.ew_angstrom[table.names.index(name)]
    return float(np.interp(emline_type, table.emline_type_grid, row))


def test_qsogen_table_matches_composite_sanity_ranges():
    table = load_qsogen_line_ew_table()
    assert table.beslope == pytest.approx(0.183)
    assert table.benorm == pytest.approx(-27.0)
    assert table.emline_type_grid[0] == pytest.approx(-2.0)
    assert table.emline_type_grid[-1] == pytest.approx(3.0)
    # Vanden Berk et al. (2001): Ly-alpha + N V ~ 94 A, C IV 24 A, C III] 21 A, Mg II 32 A.
    assert 70.0 < _ew("Lya+NV") < 100.0
    assert 24.0 < _ew("CIV") < 50.0
    assert 15.0 < _ew("CIII]+AlIII+SiIII]") < 40.0
    assert 25.0 < _ew("MgII") < 55.0
    # H-beta side bands must avoid its red wing and Fe II 4924/5018 (about 60-66 A).
    assert 55.0 < _ew("Hb") < 75.0
    # Narrow lines: [O III] 5007 of order 10 A, 4959 tied at 1/2.98; none in the UV.
    assert 8.0 < _ew("[OIII]5007") < 20.0
    assert _ew("[OIII]4959") == pytest.approx(_ew("[OIII]5007") / 2.98)
    assert {"broad", "narrow"} == set(table.family)
    assert all(lo > 3000.0 for fam, lo in zip(table.family, table.window_lo) if fam == "narrow")
    # Baldwin effect: fainter quasars (larger emline_type) have stronger Ly-alpha and C IV.
    assert _ew("Lya+NV", 1.0) > _ew("Lya+NV", 0.0) > _ew("Lya+NV", -1.0)
    assert _ew("CIV", 1.0) > _ew("CIV", 0.0) > _ew("CIV", -1.0)


def test_m_i_z2_conversion_and_emline_type():
    # L_nu(2500) = 1e31 erg/s/Hz, i.e. L_lambda = 1e31 c / lambda^2 / 1e7 W/A.
    llambda = 1.0e31 * 2.99792458e18 / 2500.0**2 / 1.0e7
    m_i = float(m_i_z2_from_disc_llambda_2500(llambda))
    assert m_i == pytest.approx(M_I_Z2_ZEROPOINT - 2.5 * 31.0)
    assert m_i == pytest.approx(-27.10, abs=0.01)
    assert float(qsogen_emline_type(-27.0)) == pytest.approx(0.0)
    assert float(qsogen_emline_type(-26.0)) == pytest.approx(0.183)
    assert float(qsogen_emline_type(0.0)) == pytest.approx(3.0)
    assert float(qsogen_emline_type(-60.0)) == pytest.approx(-2.0)


def test_line_ew_matrix_maps_default_broad_lines():
    table = np.loadtxt(
        files("jaxsedfit").joinpath("resources/templates/emission_line_table.formatted"),
        comments="#",
        usecols=(1, 2),
    )
    wave, blagn = table[:, 0], table[:, 1]
    ew_matrix, anchored = qsogen_line_ew_matrix(wave, blagn)
    grid = load_qsogen_line_ew_table().emline_type_grid
    mid = int(np.argmin(np.abs(grid)))

    def index(rest):
        return int(np.flatnonzero((np.abs(wave - rest) < 0.5) & (blagn > 0.0))[0])

    lya, nv, civ, niv = index(1215.0), index(1240.0), index(1549.0), index(1486.0)
    # Blends are split by the broad template ratio and conserve the group EW.
    assert ew_matrix[lya, mid] + ew_matrix[nv, mid] == pytest.approx(_ew("Lya+NV"))
    assert ew_matrix[lya, mid] / ew_matrix[nv, mid] == pytest.approx(blagn[lya] / blagn[nv])
    assert ew_matrix[civ, mid] + ew_matrix[niv, mid] == pytest.approx(_ew("CIV"))
    assert anchored[lya] == anchored[civ] == 1.0
    # N III] 1750 and He II 4686 have no QSOGEN window and stay template-scaled.
    assert anchored[index(1750.0)] == 0.0
    assert anchored[index(4686.0)] == 0.0
    assert np.all(ew_matrix[blagn <= 0.0] == 0.0)


def test_continuum_ew_luminosity_rule():
    wave = np.array([1215.0, 1240.0, 1549.0, 1750.0])
    blagn = np.array([10.0, 3.0, 5.0, 0.4])
    disc = jnp.array([2.0e36, 1.9e36, 1.5e36, 1.3e36])
    lumin = np.asarray(continuum_ew_broad_line_luminosity(wave, blagn, disc, 1.0e36, emline_type=0.0))
    ew_lya = _ew("Lya+NV") * 10.0 / 13.0
    ew_nv = _ew("Lya+NV") * 3.0 / 13.0
    ew_civ = _ew("CIV")
    np.testing.assert_allclose(lumin[:3], [ew_lya * 2.0e36, ew_nv * 1.9e36, ew_civ * 1.5e36], rtol=1e-12)
    # Unanchored lines keep their template ratio to the anchored lines.
    assert lumin[3] == pytest.approx(0.4 * lumin[:3].sum() / blagn[:3].sum(), rel=1e-12)
    with pytest.raises(ValueError, match="QSOGEN line window"):
        continuum_ew_broad_line_luminosity(np.array([3500.0]), np.array([1.0]), jnp.ones(1), 1.0)


def test_continuum_ew_luminosity_is_jittable_and_follows_luminosity():
    wave = np.array([1215.0, 1549.0])
    blagn = np.array([10.0, 5.0])

    @jax.jit
    def ew_lya(llambda_2500):
        lumin = continuum_ew_broad_line_luminosity(wave, blagn, jnp.ones(2), llambda_2500)
        return lumin[0]

    faint, bright = float(ew_lya(1.0e33)), float(ew_lya(1.0e37))
    assert faint > bright
    m_faint = float(m_i_z2_from_disc_llambda_2500(1.0e33))
    expected = _ew("Lya+NV", float(qsogen_emline_type(m_faint)))
    assert faint == pytest.approx(expected, rel=1e-10)


def test_agn_config_line_extrapolation_fields_round_trip_and_default_for_old_configs():
    cfg = AGNConfig(line_extrapolation_anchor=" Continuum_EW ", line_extrapolation_emline_type=0.5)
    assert cfg.line_extrapolation_anchor == "continuum_ew"
    assert cfg.line_extrapolation_emline_type == 0.5
    with pytest.raises(ValueError, match="line_extrapolation_anchor"):
        AGNConfig(line_extrapolation_anchor="qsogen")
    with pytest.raises(ValueError, match="line_extrapolation_emline_type"):
        AGNConfig(line_extrapolation_emline_type=4.0)

    full = _joint_config(anchor="continuum_ew", emline_type=-0.5)
    restored = fit_config_from_mapping(json.loads(json.dumps(full.to_dict())))
    assert restored.agn.line_extrapolation_anchor == "continuum_ew"
    assert restored.agn.line_extrapolation_emline_type == -0.5

    legacy = full.to_dict()
    del legacy["agn"]["line_extrapolation_anchor"]
    del legacy["agn"]["line_extrapolation_emline_type"]
    del legacy["agn"]["line_extrapolation_ew_scope"]
    legacy_cfg = fit_config_from_mapping(legacy)
    assert legacy_cfg.agn.line_extrapolation_anchor == "fitted_lines"
    assert legacy_cfg.agn.line_extrapolation_emline_type is None

    mutated = _joint_config()
    mutated.agn.line_extrapolation_anchor = "bogus"
    with pytest.raises(ValueError, match="line_extrapolation_anchor"):
        mutated.validate()


@pytest.fixture
def fake_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())


def _top_hat(name, band_rest):
    band_obs = np.array(band_rest) * (1.0 + REDSHIFT)
    filter_wave = np.linspace(band_obs[0] - 1.0, band_obs[1] + 1.0, 600)
    filter_trans = ((filter_wave >= band_obs[0]) & (filter_wave <= band_obs[1])).astype(float)
    return FilterCurve(name=name, wave=filter_wave.tolist(), transmission=filter_trans.tolist())


# Rest-frame H-alpha band, redward of the rest 2000-3000 A spectrum.
RED_BAND_REST = (6300.0, 6800.0)


def _joint_config(anchor="fitted_lines", emline_type=None, scope=None, red_band=False):
    spec_wave = np.linspace(2000.0, 3000.0, 200) * (1.0 + REDSHIFT)
    curves = [_top_hat("civ_band", BAND_REST)]
    if red_band:
        curves.append(_top_hat("ha_band", RED_BAND_REST))
    scope_kwargs = {} if scope is None else {"line_extrapolation_ew_scope": scope}
    return FitConfig(
        observation=Observation(object_id="obj", redshift=REDSHIFT),
        photometry=PhotometryData(
            filter_names=[curve.name for curve in curves],
            fluxes=[1.0] * len(curves),
            errors=[0.1] * len(curves),
        ),
        filters=FilterSet(curves=curves),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=512, fit_host=False),
        agn=AGNConfig(
            fit_lines=True,
            tied_lines=True,
            fit_feii=False,
            fit_balmer_continuum=False,
            line_flux_scale_mjy=0.1,
            line_extrapolation_anchor=anchor,
            line_extrapolation_emline_type=emline_type,
            **scope_kwargs,
        ),
        likelihood=LikelihoodConfig(variability_uncertainty=False, fit_spectrum_scale=False),
        spectroscopy=SpectroscopyData(
            wave_obs=spec_wave.tolist(),
            fluxes=np.full(spec_wave.size, 0.1).tolist(),
            errors=np.full(spec_wave.size, 0.02).tolist(),
            instrument="sdss",
        ),
        inference=InferenceConfig(map_steps=2),
    )


_PARAMS = {
    "log_agn_amp": np.array(85.0),
    "pl_slope": np.array(-1.5),
    "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
    "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
    "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
    "fcov": np.array(0.1),
    "si": np.array(0.0),
    "cool_lam": np.array(17.0),
    "cool_width": np.array(0.45),
    "hot_lam": np.array(2.0),
    "hot_width": np.array(0.5),
    "log_hot_fcov": np.array(np.log(0.1)),
    "log_ebv_agn": np.array(np.log(1.0e-12)),
}


def _trace(cfg):
    context = build_model_context(cfg)
    model = seed(substitute(grahsp_photometric_model, data=_PARAMS), jax.random.PRNGKey(4))
    return context, trace(model).get_trace(context, include_components=True)


def test_continuum_ew_band_flux_ratio_matches_equivalent_width(fake_ssp, monkeypatch):
    """A known EW on the fitted continuum gives the expected line-to-disc band flux ratio.

    Unit chain: EW [A] x continuum L_lambda [W/A] = line luminosity [W],
    attenuated, redshifted and projected by ``_project_integrated_local_line_filters``
    into mJy; the disc goes through the ordinary SED projection into mJy. In a
    top-hat band the ratio is sum_i(L_i f_i) / integral(L_lambda dlambda), with
    f_i the fraction of line i's Gaussian profile inside the band.
    """
    from math import erf, sqrt

    captured = _spy_extrapolation(monkeypatch)
    cfg = _joint_config(anchor="continuum_ew", emline_type=0.0)
    context, tr = _trace(cfg)
    line_flux = float(np.asarray(tr["spectral_extrapolated_broad_photometry"]["value"])[0])
    disk_flux = float(np.asarray(tr["disk_fluxes"]["value"])[0])
    broad, _, width_kms, _ = (np.asarray(x) for x in captured["out"])
    continuum = np.asarray(captured["kwargs"]["agn_continuum_at_lines"])

    rest = np.asarray(context.rest_wave, dtype=float)
    disk_rest = np.asarray(tr["disk_rest_sed"]["value"], dtype=float)
    wave = np.asarray(context.templates.line_wave)
    blagn = np.asarray(context.templates.line_blagn)
    ew_matrix, _ = qsogen_line_ew_matrix(wave, blagn)
    grid = load_qsogen_line_ew_table().emline_type_grid
    ew = np.array([np.interp(0.0, grid, row) for row in ew_matrix])
    in_band = (wave > BAND_REST[0]) & (wave < BAND_REST[1]) & (blagn > 0.0)
    assert set(np.round(wave[in_band])) == {1486.0, 1549.0, 1640.0, 1663.0}
    np.testing.assert_allclose(broad[in_band], ew[in_band] * continuum[in_band], rtol=1e-10)
    sigma = wave * float(width_kms) / 299792.458 / 2.354820045
    lo, hi = BAND_REST
    fraction = np.array(
        [0.5 * (erf((hi - w) / (sqrt(2.0) * s)) - erf((lo - w) / (sqrt(2.0) * s))) for w, s in zip(wave, sigma)]
    )
    band = np.linspace(*BAND_REST, 2001)
    disc_band = np.trapezoid(np.exp(np.interp(np.log(band), np.log(rest), np.log(disk_rest))), band)
    expected_ratio = np.sum(broad * fraction) / disc_band
    # QSOGEN: C IV + N IV] + He II + O III] ~ 53 A over the band.
    in_band_ratio = np.sum(ew[in_band] * continuum[in_band]) / disc_band
    assert in_band_ratio == pytest.approx((_ew("CIV") + _ew("HeII+OIII]")) / (hi - lo), rel=0.1)
    assert line_flux / disk_flux == pytest.approx(expected_ratio, rel=0.01)


def test_default_anchor_is_bit_identical_to_explicit_fitted_lines(fake_ssp):
    default_cfg = _joint_config()
    explicit_cfg = _joint_config(anchor="fitted_lines")
    assert default_cfg.agn.line_extrapolation_anchor == "fitted_lines"
    _, tr_default = _trace(default_cfg)
    _, tr_explicit = _trace(explicit_cfg)
    for site in ("spectral_extrapolated_broad_photometry", "spectral_extrapolated_narrow_photometry", "pred_fluxes"):
        np.testing.assert_array_equal(np.asarray(tr_default[site]["value"]), np.asarray(tr_explicit[site]["value"]))


def test_continuum_ew_changes_only_out_of_coverage_lines(fake_ssp):
    _, tr_fitted = _trace(_joint_config())
    _, tr_cew = _trace(_joint_config(anchor="continuum_ew"))
    for site in ("spectral_line_photometry", "disk_fluxes"):
        np.testing.assert_array_equal(np.asarray(tr_fitted[site]["value"]), np.asarray(tr_cew[site]["value"]))
    for site in ("spectral_extrapolated_broad_photometry", "spectral_extrapolated_narrow_photometry"):
        assert not np.array_equal(np.asarray(tr_fitted[site]["value"]), np.asarray(tr_cew[site]["value"]))


def _spy_extrapolation(monkeypatch):
    import jaxsedfit.model as model_module

    captured = {}
    original = model_module._spectral_family_extrapolation

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.update(args=args, kwargs=kwargs, out=out)
        return out

    monkeypatch.setattr(model_module, "_spectral_family_extrapolation", spy)
    return captured


def _red_side_config(redshift, spec_rest, band_rest):
    cfg = _joint_config(anchor="continuum_ew", emline_type=0.0)
    cfg.observation.redshift = redshift
    band_obs = np.array(band_rest) * (1.0 + redshift)
    filter_wave = np.linspace(band_obs[0] - 1.0, band_obs[1] + 1.0, 600)
    trans = ((filter_wave >= band_obs[0]) & (filter_wave <= band_obs[1])).astype(float)
    cfg.filters.curves[0] = FilterCurve(name="civ_band", wave=filter_wave.tolist(), transmission=trans.tolist())
    spec_wave = np.linspace(*spec_rest, 200) * (1.0 + redshift)
    cfg.spectroscopy.wave_obs = spec_wave.tolist()
    return cfg


@pytest.mark.parametrize(
    ("redshift", "spec_rest"),
    [(0.6, (3000.0, 6000.0)), (2.0, (1300.0, 3000.0))],
)
def test_continuum_ew_red_side_h_alpha(fake_ssp, monkeypatch, redshift, spec_rest):
    """H-alpha redward of the spectrum (z ~ 0.6 optical, z ~ 2 in the K band) gets EW x (disc + torus)."""
    captured = _spy_extrapolation(monkeypatch)
    cfg = _red_side_config(redshift, spec_rest, (6300.0, 6800.0))
    context, tr = _trace(cfg)
    broad, narrow, _, _ = (np.asarray(x) for x in captured["out"])
    continuum = np.asarray(captured["kwargs"]["agn_continuum_at_lines"])
    wave = np.asarray(context.templates.line_wave)
    ha = int(np.flatnonzero(np.abs(wave - 6563.0) < 1.0)[0])
    assert broad[ha] == pytest.approx(_ew("Ha") * continuum[ha], rel=1e-10)
    # Narrow H-alpha from QSOGEN's narrow-line template (not the Seyfert-2 one).
    table = load_qsogen_line_ew_table()
    ha_n = float(np.interp(0.0, table.emline_type_grid, table.ew_angstrom[table.names.index("Ha_n")]))
    assert narrow[ha] == pytest.approx(ha_n * continuum[ha], rel=1e-10)
    # The continuum at the line includes the torus: at least the disc.
    rest = np.asarray(context.rest_wave, dtype=float)
    disc = np.exp(np.interp(np.log(wave), np.log(rest), np.log(np.asarray(tr["disk_rest_sed"]["value"]))))
    assert continuum[ha] >= disc[ha] * (1.0 - 1e-6)
    assert np.asarray(tr["spectral_extrapolated_broad_photometry"]["value"])[0] > 0.0


def test_continuum_ew_includes_torus_in_the_near_ir(fake_ssp, monkeypatch):
    captured = _spy_extrapolation(monkeypatch)
    context, tr = _trace(_red_side_config(0.6, (3000.0, 6000.0), (18000.0, 19500.0)))
    continuum = np.asarray(captured["kwargs"]["agn_continuum_at_lines"])
    wave = np.asarray(context.templates.line_wave)
    rest = np.asarray(context.rest_wave, dtype=float)
    disc = np.exp(np.interp(np.log(wave), np.log(rest), np.log(np.asarray(tr["disk_rest_sed"]["value"]))))
    paa = int(np.flatnonzero(np.abs(wave - 18750.0) < 1.0)[0])
    lya = int(np.flatnonzero(np.abs(wave - 1215.0) < 1.0)[0])
    assert continuum[paa] > 1.01 * disc[paa]  # the torus (hot dust) contributes at 1.9 micron
    assert continuum[lya] == pytest.approx(disc[lya], rel=1e-3)


def test_continuum_ew_narrow_lines(fake_ssp, monkeypatch):
    captured = _spy_extrapolation(monkeypatch)
    context, _ = _trace(_red_side_config(0.6, (3000.0, 4800.0), (4900.0, 5100.0)))
    _, narrow, _, _ = (np.asarray(x) for x in captured["out"])
    continuum = np.asarray(captured["kwargs"]["agn_continuum_at_lines"])
    wave = np.asarray(context.templates.line_wave)
    o3 = int(np.flatnonzero(np.abs(wave - 5007.0) < 1.0)[0])
    lya = int(np.flatnonzero(np.abs(wave - 1215.0) < 1.0)[0])
    civ = int(np.flatnonzero(np.abs(wave - 1549.0) < 1.0)[0])
    assert 8.0 < narrow[o3] / continuum[o3] < 25.0  # [O III] 5007 rest EW of order 10 A
    # QSOGEN has no narrow UV lines: none are extrapolated.
    assert narrow[lya] == 0.0 and narrow[civ] == 0.0


def test_uv_slope_is_deterministic_without_prior(fake_ssp):
    _, tr = _trace(_joint_config(anchor="continuum_ew", emline_type=0.0))
    assert tr["uv_slope"]["type"] == "deterministic"
    assert float(np.asarray(tr["uv_slope"]["value"])) == 0.0


def test_ew_scope_blueward_keeps_fitted_anchor_redward(fake_ssp):
    sites = {}
    for label, kwargs in {
        "fitted": {},
        "blueward": {"anchor": "continuum_ew", "emline_type": 0.0, "scope": "blueward"},
        "all": {"anchor": "continuum_ew", "emline_type": 0.0},
    }.items():
        _, tr = _trace(_joint_config(red_band=True, **kwargs))
        sites[label] = np.asarray(tr["spectral_extrapolated_broad_photometry"]["value"])
    assert _joint_config(anchor="continuum_ew").agn.line_extrapolation_ew_scope == "all"
    blue, red = 0, 1
    # Blue band: both continuum_ew scopes anchor C IV to the continuum.
    assert sites["blueward"][blue] == sites["all"][blue]
    assert sites["blueward"][blue] != sites["fitted"][blue]
    # Red band (H-alpha): "blueward" keeps the fitted-line anchor, "all" does not.
    assert sites["blueward"][red] == pytest.approx(sites["fitted"][red], rel=1e-12)
    assert sites["all"][red] != pytest.approx(sites["fitted"][red], rel=1e-3)


def test_ew_scope_validation():
    assert AGNConfig(line_extrapolation_ew_scope=" ALL ").line_extrapolation_ew_scope == "all"
    with pytest.raises(ValueError, match="line_extrapolation_ew_scope"):
        AGNConfig(line_extrapolation_ew_scope="redward")


def test_blueward_scope_excludes_lines_inside_the_spectrum(fake_ssp):
    from jaxsedfit.model import _blueward_of_spectral_coverage

    context = build_model_context(_joint_config(anchor="continuum_ew"))
    waves = np.array([1215.0, 1999.0, 2001.0, 2798.0, 3001.0, 4861.0])
    blue = np.asarray(_blueward_of_spectral_coverage(context, context.fit_config, waves, REDSHIFT))
    # Spectrum covers rest 2000-3000 A: only lines blueward of 2000 A qualify.
    assert blue.tolist() == [True, True, False, False, False, False]


def _narrow_out(monkeypatch, scope, mode):
    captured = _spy_extrapolation(monkeypatch)
    cfg = _joint_config(anchor="continuum_ew", emline_type=0.0, scope=scope)
    cfg.agn.narrow_line_extrapolation = mode
    context, _ = _trace(cfg)
    return context, np.asarray(captured["out"][1])


def test_continuum_ew_scope_all_overrides_every_narrow_mode(fake_ssp, monkeypatch):
    outs = {mode: _narrow_out(monkeypatch, "all", mode)[1] for mode in ("template", "optical_only", "none")}
    np.testing.assert_array_equal(outs["optical_only"], outs["template"])
    np.testing.assert_array_equal(outs["none"], outs["template"])
    context, _ = _narrow_out(monkeypatch, "all", "template")
    wave = np.asarray(context.templates.line_wave)
    o3 = int(np.flatnonzero(np.abs(wave - 5007.0) < 1.0)[0])
    assert outs["none"][o3] > 0.0  # QSOGEN [O III] still extrapolated with mode "none"


def test_continuum_ew_blueward_leaves_redward_narrow_lines_to_the_mode(fake_ssp, monkeypatch):
    # Spectrum at rest 2000-3000 A: [O III] 5007 is redward, C IV blueward.
    context, template = _narrow_out(monkeypatch, "blueward", "template")
    _, none = _narrow_out(monkeypatch, "blueward", "none")
    _, optical = _narrow_out(monkeypatch, "blueward", "optical_only")
    wave = np.asarray(context.templates.line_wave)
    o3 = int(np.flatnonzero(np.abs(wave - 5007.0) < 1.0)[0])
    civ = int(np.flatnonzero(np.abs(wave - 1549.0) < 1.0)[0])
    assert template[o3] > 0.0 and none[o3] == 0.0
    assert optical[o3] == template[o3]  # optical line kept by "optical_only"
    # Blueward lines use the EW anchor whatever the mode (no narrow UV in QSOGEN).
    assert template[civ] == none[civ] == optical[civ] == 0.0
