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
BAND_REST = (1450.0, 1700.0)


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


def _joint_config(anchor="fitted_lines", emline_type=None):
    band_obs = np.array(BAND_REST) * (1.0 + REDSHIFT)
    filter_wave = np.linspace(band_obs[0] - 1.0, band_obs[1] + 1.0, 600)
    filter_trans = ((filter_wave >= band_obs[0]) & (filter_wave <= band_obs[1])).astype(float)
    spec_wave = np.linspace(2000.0, 3000.0, 200) * (1.0 + REDSHIFT)
    return FitConfig(
        observation=Observation(object_id="obj", redshift=REDSHIFT),
        photometry=PhotometryData(filter_names=["civ_band"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="civ_band", wave=filter_wave.tolist(), transmission=filter_trans.tolist())]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=512, fit_host=False),
        agn=AGNConfig(
            fit_lines=True,
            tied_lines=True,
            fit_feii=False,
            fit_balmer_continuum=False,
            line_flux_scale_mjy=0.1,
            line_extrapolation_anchor=anchor,
            line_extrapolation_emline_type=emline_type,
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
    "uv_slope": np.array(0.0),
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


def test_continuum_ew_band_flux_ratio_matches_equivalent_width(fake_ssp):
    """A known EW on the fitted disc gives the expected line-to-disc band flux ratio.

    Unit chain: EW [A] x disc L_lambda [W/A] = line luminosity [W], attenuated,
    redshifted and projected by ``_project_integrated_local_line_filters`` into
    mJy; the disc goes through the ordinary SED projection into mJy. In a
    top-hat band over rest 1450-1700 A on a nearly flat continuum the ratio is
    sum(EW_i L_lambda(lambda_i)) / integral(L_lambda dlambda) over the band.
    """
    cfg = _joint_config(anchor="continuum_ew", emline_type=0.0)
    context, tr = _trace(cfg)
    line_flux = float(np.asarray(tr["spectral_extrapolated_broad_photometry"]["value"])[0])
    disk_flux = float(np.asarray(tr["disk_fluxes"]["value"])[0])

    rest = np.asarray(context.rest_wave, dtype=float)
    disk_rest = np.asarray(tr["disk_rest_sed"]["value"], dtype=float)
    wave = np.asarray(context.templates.line_wave)
    blagn = np.asarray(context.templates.line_blagn)
    disc_at_lines = np.exp(np.interp(np.log(wave), np.log(rest), np.log(disk_rest)))
    ew_matrix, _ = qsogen_line_ew_matrix(wave, blagn)
    grid = load_qsogen_line_ew_table().emline_type_grid
    ew = np.array([np.interp(0.0, grid, row) for row in ew_matrix])
    in_band = (wave > BAND_REST[0]) & (wave < BAND_REST[1]) & (blagn > 0.0)
    assert set(np.round(wave[in_band])) == {1486.0, 1549.0, 1640.0, 1663.0}
    band = np.linspace(*BAND_REST, 2001)
    expected_ratio = np.sum(ew[in_band] * disc_at_lines[in_band]) / np.trapezoid(
        np.exp(np.interp(np.log(band), np.log(rest), np.log(disk_rest))), band
    )
    # QSOGEN: C IV + N IV] + He II + O III] ~ 53 A over a 250 A band.
    assert expected_ratio == pytest.approx((_ew("CIV") + _ew("HeII+OIII]")) / 250.0, rel=0.1)
    assert line_flux / disk_flux == pytest.approx(expected_ratio, rel=0.02)


def test_default_anchor_is_bit_identical_to_explicit_fitted_lines(fake_ssp):
    default_cfg = _joint_config()
    explicit_cfg = _joint_config(anchor="fitted_lines")
    assert default_cfg.agn.line_extrapolation_anchor == "fitted_lines"
    _, tr_default = _trace(default_cfg)
    _, tr_explicit = _trace(explicit_cfg)
    for site in ("spectral_extrapolated_broad_photometry", "spectral_extrapolated_narrow_photometry", "pred_fluxes"):
        np.testing.assert_array_equal(np.asarray(tr_default[site]["value"]), np.asarray(tr_explicit[site]["value"]))


def test_continuum_ew_changes_only_out_of_coverage_broad_lines(fake_ssp):
    _, tr_fitted = _trace(_joint_config())
    _, tr_cew = _trace(_joint_config(anchor="continuum_ew"))
    for site in ("spectral_line_photometry", "spectral_extrapolated_narrow_photometry", "disk_fluxes"):
        np.testing.assert_array_equal(np.asarray(tr_fitted[site]["value"]), np.asarray(tr_cew[site]["value"]))
    assert not np.array_equal(
        np.asarray(tr_fitted["spectral_extrapolated_broad_photometry"]["value"]),
        np.asarray(tr_cew["spectral_extrapolated_broad_photometry"]["value"]),
    )


# Recorded with jaxsedfit main 041f51f (before the continuum_ew anchor existed)
# on the fixed input of ``_joint_config()`` / ``_PARAMS``.
_MAIN_041F51F_DEFAULT_SITES = {
    "pred_fluxes": [0.007083976730586054],
    "spectral_extrapolated_broad_photometry": [0.001527605416558358],
    "spectral_extrapolated_narrow_photometry": [7.021637253046015e-05],
    "disk_fluxes": [0.005486005841998523],
}


def test_default_anchor_reproduces_pre_feature_outputs(fake_ssp):
    _, tr = _trace(_joint_config())
    for site, expected in _MAIN_041F51F_DEFAULT_SITES.items():
        np.testing.assert_allclose(np.asarray(tr[site]["value"]), expected, rtol=1e-12, atol=0.0, err_msg=site)
