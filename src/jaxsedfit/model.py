from __future__ import annotations

# Portions of this file are derived from or closely based on CIGALE and
# GRAHSP/pcigale model logic, translated into JAX/NumPyro for jaxsedfit.
# Relevant GRAHSP upstream sources include:
# - pcigale/creation_modules/activate.py
# - pcigale/creation_modules/activategtorus.py
# - pcigale/creation_modules/activatelines.py
# - pcigale/creation_modules/biattenuation.py
# - pcigale/creation_modules/redshifting.py
# - pcigale/creation_modules/galdale2014.py
# Relevant CIGALE v2025.1 sources include:
# - pcigale/sed_modules/sfhdelayed.py
# - pcigale/sed_modules/nebular.py
# - pcigale/sed_modules/redshifting.py
# - pcigale/sed_modules/dale2014.py
# Upstream license: CeCILL v2. See LICENSES/CeCILL-v2.txt and
# LICENSES/THIRD_PARTY_NOTICES.md.

from dataclasses import replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_U_PARAMS, DiffstarUParams, calc_sfh_singlegal, get_bounded_diffstar_params
from diffstar.defaults import FB as DIFFSTAR_FB
from diffstar.defaults import LGT0 as DIFFSTAR_LGT0
from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
from dsps.sed.ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf
import numpyro
import numpyro.distributions as dist

from .preload import ModelContext, _build_fixed_igm_jax as _igm_transmission
from .spectral_results import SpectralSites
from .velocity import shift_and_broaden_lnlam as _fourier_shift_and_broaden_lnlam

C_KMS = 299792.458
C_MS = 2.99792458e8
ERG_PER_WATT = 1.0e7
AGN_BOLOMETRIC_CORRECTION_5100 = 9.26
DSPS_SOLAR_METALLICITY = 0.019
MPC_TO_M = 3.085677581491367e22
DEFAULT_BROAD_LINE_WIDTH_KMS_MIN = 1000.0
DEFAULT_BROAD_LINE_WIDTH_KMS_MAX = 15000.0
DEFAULT_NARROW_LINE_WIDTH_KMS_MIN = 100.0
DEFAULT_NARROW_LINE_WIDTH_KMS_MAX = 1500.0
DEFAULT_BROAD_LINE_WIDTH_KMS = 3000.0
DEFAULT_NARROW_LINE_WIDTH_KMS = 500.0
DEFAULT_NARROW_LINES_STRENGTH = 1.0
DEFAULT_BALMER_CONTINUUM_STRENGTH = 1.0e-3
_SSP_BIN_QUAD_NODES, _SSP_BIN_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(16)
BURST_RISE_GYR = 0.002
GRAHSP_BIATTENUATION_BREAK_A = 11000.0
GRAHSP_EBV_MIN = 0.01
GRAHSP_EBV_MAX = 10.0
# Noll et al. (2009) modified Calzetti curve, optional galaxy attenuation
# (GalaxyConfig.attenuation_curve = "noll2009"). The 2175 A bump strength is tied
# to the slope delta by Kriek & Conroy (2013): E_b = 0.85 - 1.9 delta.
CALZETTI_RV = 4.05
NOLL_BUMP_WAVE_A = 2175.0
NOLL_BUMP_FWHM_A = 350.0
NOLL_V_WAVE_A = 5500.0
NOLL_B_WAVE_A = 4400.0
KRIEK_CONROY_BUMP_INTERCEPT = 0.85
KRIEK_CONROY_BUMP_SLOPE = -1.9
NOLL_DELTA_PRIOR_LOC = -0.2
NOLL_DELTA_PRIOR_SCALE = 0.3
NOLL_DELTA_MIN = -0.8
NOLL_DELTA_MAX = 0.4
GRAHSP_PL_BEND_LOC_A = 1000.0
GRAHSP_PL_BEND_WIDTH = 10.0
GRAHSP_PL_CUTOFF_A = 100000.0
GRAHSP_PL_SLOPE_LOW = -2.7
GRAHSP_PL_SLOPE_HIGH = -1.0
GRAHSP_PL_BEND_LOC_LOW_A = 500.0
GRAHSP_PL_BEND_LOC_HIGH_A = 1500.0
GRAHSP_PL_BEND_WIDTH_LOW = 0.1
GRAHSP_PL_BEND_WIDTH_HIGH = 10.0
GRAHSP_TORUS_NORM_A = 120000.0
GRAHSP_SI_EM_LAM_A = 98410.0
GRAHSP_SI_ABS_LAM_A = 142240.0
GRAHSP_SI_EM_WIDTH_A = 10253.0
GRAHSP_SI_ABS_WIDTH_A = 11635.0


def _np_to_jnp(x):
    """Convert an array-like object to a float64 JAX array.

    Parameters
    ----------
    x : array_like
        Values to convert to a float64 array.
    """
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _bool_to_jnp(x):
    """Convert an array-like object to a boolean JAX array.

    Parameters
    ----------
    x : array_like
        Values to convert to a boolean array.
    """
    return jnp.asarray(np.asarray(x, dtype=bool))


def _flat_lcdm_luminosity_distance_m_jax(redshift, h0: float, om0: float):
    """Flat-LCDM luminosity distance by trapezoid integration (matches astropy to ~1e-5).

    Parameters
    ----------
    redshift : float or jax.Array
        Redshift(s); negative values are clipped to zero.
    h0 : float
        Hubble constant in km/s/Mpc.
    om0 : float
        Present-day matter density parameter.
    """
    z = jnp.maximum(jnp.asarray(redshift, dtype=jnp.float64), 0.0)
    grid = jnp.linspace(0.0, 1.0, 256, dtype=jnp.float64)
    z_grid = z[..., None] * grid
    e_z = jnp.sqrt(float(om0) * (1.0 + z_grid) ** 3 + (1.0 - float(om0)))
    integrand = 1.0 / jnp.maximum(e_z, 1.0e-30)
    dt = 1.0 / (grid.size - 1)
    integral_unit = dt * (
        0.5 * integrand[..., 0] + jnp.sum(integrand[..., 1:-1], axis=-1) + 0.5 * integrand[..., -1]
    )
    comoving_mpc = (C_KMS / float(h0)) * z * integral_unit
    return (1.0 + z) * comoving_mpc * MPC_TO_M


def _luminosity_distance_m_jax(redshift, h0: float, om0: float):
    """Return luminosity distance in meters using a JAX-native flat LCDM path.

    Parameters
    ----------
    redshift : float or jax.Array
        Redshift(s); scalar input returns a scalar.
    h0 : float
        Hubble constant in km/s/Mpc.
    om0 : float
        Present-day matter density parameter.
    """
    # jax_cosmo is not used here: its tabulated background is biased by ~2% at z~0.001
    # and disagrees with the astropy distances used by the fixed-redshift caches.
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    scalar_input = redshift.ndim == 0
    d_l_m = _flat_lcdm_luminosity_distance_m_jax(redshift, h0, om0)
    return jnp.reshape(d_l_m, ()) if scalar_input else d_l_m


def _flat_lcdm_age_gyr_jax(redshift, h0: float, om0: float):
    """Return the cosmic age for a flat matter-plus-Lambda cosmology."""
    z = jnp.maximum(jnp.asarray(redshift, dtype=jnp.float64), 0.0)
    omega_m = jnp.clip(jnp.asarray(om0, dtype=jnp.float64), 1.0e-8, 1.0 - 1.0e-8)
    omega_l = 1.0 - omega_m
    hubble_time_gyr = 977.7922216807892 / jnp.asarray(h0, dtype=jnp.float64)
    argument = jnp.sqrt(omega_l / omega_m) / (1.0 + z) ** 1.5
    return (
        2.0
        * hubble_time_gyr
        / (3.0 * jnp.sqrt(omega_l))
        * jnp.arcsinh(argument)
    )


def _prior_distribution(prior_config: dict[str, Any], key: str, default_distribution):
    """Read a NumPyro distribution-like prior from the flat prior mapping.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    key : str
        Prior key (site name) to look up.
    default_distribution : numpyro distribution
        Distribution returned when ``key`` is absent or unrecognised.
    """
    cfg = prior_config.get(key, None)
    if cfg is None:
        return default_distribution
    if isinstance(cfg, dist.Distribution):
        return cfg
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return dist.Normal(jnp.asarray(cfg[0], dtype=jnp.float64), jnp.maximum(jnp.asarray(cfg[1], dtype=jnp.float64), 1.0e-6))
    if not isinstance(cfg, dict):
        return default_distribution

    default_name = default_distribution.__class__.__name__
    family = str(cfg.get("dist", cfg.get("family", default_name))).lower()
    if family in {"normal", "gaussian"}:
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.Normal(loc, scale)
    if family in {"truncatednormal", "truncated_normal", "truncnormal", "truncnorm"}:
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        low = jnp.asarray(cfg.get("low", -jnp.inf), dtype=jnp.float64)
        high = jnp.asarray(cfg.get("high", jnp.inf), dtype=jnp.float64)
        return dist.TruncatedNormal(loc, scale, low=low, high=high)
    if family in {"lognormal", "log-normal", "log_normal"}:
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.LogNormal(loc, scale)
    if family in {"halfnormal", "half_normal"}:
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.HalfNormal(scale)
    if family in {"student_t", "studentt", "t"}:
        df = jnp.maximum(jnp.asarray(cfg.get("df", 5.0), dtype=jnp.float64), 1.0e-6)
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.StudentT(df=df, loc=loc, scale=scale)
    if family in {"uniform", "flat"}:
        low = jnp.asarray(cfg.get("low", 0.0), dtype=jnp.float64)
        high = jnp.asarray(cfg.get("high", 1.0), dtype=jnp.float64)
        lo = jnp.minimum(low, high)
        hi = jnp.maximum(jnp.maximum(low, high), lo + 1.0e-6)
        return dist.Uniform(lo, hi)
    if family in {"exponential", "exp"}:
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-30)
        return dist.Exponential(1.0 / scale)
    return default_distribution


def _sample_prior(prior_config: dict[str, Any], key: str, default_distribution):
    """Sample a scalar site from a configured distribution or a default.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    key : str
        NumPyro site name, also used as the prior key.
    default_distribution : numpyro distribution
        Distribution used when ``key`` is not configured.
    """
    return numpyro.sample(key, _prior_distribution(prior_config, key, default_distribution))


def _sample_log_positive_from_distribution(
    prior_config: dict[str, Any],
    *,
    value_key: str,
    log_key: str,
    default_distribution,
):
    """Sample a log-parameter from a distribution and expose its physical value.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    value_key : str
        Deterministic site name for the physical (exponentiated) value.
    log_key : str
        Sample site name for the natural-log parameter.
    default_distribution : numpyro distribution
        Default prior on the natural-log parameter.
    """
    log_value = numpyro.sample(log_key, _prior_distribution(prior_config, log_key, default_distribution))
    value = jnp.exp(log_value)
    numpyro.deterministic(value_key, value)
    return value


def _sample_positive_distribution(
    prior_config: dict[str, Any],
    *,
    value_key: str,
    log_key: str,
    default_value_distribution,
    default_log_distribution,
    default_to_log: bool = False,
):
    """Sample a positive parameter, honoring either physical or log prior keys.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    value_key : str
        Site name for the physical positive parameter.
    log_key : str
        Site name for its natural-log parameterization.
    default_value_distribution : numpyro distribution
        Default prior on the physical value.
    default_log_distribution : numpyro distribution
        Default prior on the natural-log value.
    default_to_log : bool
        Sample in natural log when neither key is configured.
    """
    if log_key in prior_config:
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=default_log_distribution,
        )
    if value_key not in prior_config and default_to_log:
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=default_log_distribution,
        )
    return _sample_prior(prior_config, value_key, default_value_distribution)


def _sample_positive(
    prior_config: dict[str, Any],
    *,
    value_key: str,
    log_key: str,
    default_value: float,
    default_log_scale: float,
    default_family: str = "lognormal",
):
    """Sample a positive parameter with an explicit prior family.

    ``prior_config[value_key]["family"]`` or ``["dist"]`` may be one of
    ``"exponential"`` or ``"lognormal"``. A direct ``prior_config[log_key]``
    override always selects the log-normal parameterization.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    value_key : str
        Site name for the physical positive parameter.
    log_key : str
        Site name for its natural-log parameterization.
    default_value : float
        Default median (log-normal) or mean (exponential) of the parameter.
    default_log_scale : float
        Default log-normal width in natural-log units.
    default_family : str
        Default prior family, ``"lognormal"`` or ``"exponential"``.
    """
    cfg = prior_config.get(value_key, None)
    family = default_family
    if isinstance(cfg, dist.Distribution):
        family = cfg.__class__.__name__.lower()
    if isinstance(cfg, dict):
        family = str(cfg.get("family", cfg.get("dist", family))).lower()
    if log_key in prior_config:
        family = "lognormal"

    if family in {"exponential", "exp"}:
        return _sample_prior(prior_config, value_key, dist.Exponential(1.0 / max(default_value, 1.0e-30)))
    if family in {"lognormal", "log-normal", "log_normal", "normal_log"}:
        if isinstance(cfg, (dict, dist.LogNormal)):
            return _sample_prior(
                prior_config,
                value_key,
                dist.LogNormal(np.log(max(default_value, 1.0e-30)), default_log_scale),
            )
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=dist.Normal(np.log(max(default_value, 1.0e-30)), default_log_scale),
        )
    raise ValueError(
        f"prior_config[{value_key!r}] family must be one of: "
        "'exponential', 'lognormal'."
    )

def _sample_bounded_normal(prior_config: dict[str, Any], key: str, default: float, scale: float, low, high):
    """Sample a Normal-like prior truncated to model support.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    key : str
        NumPyro site name, also used as the prior key.
    default : float
        Default prior location.
    scale : float
        Default prior standard deviation.
    low : float or jax.Array
        Lower bound of the model support.
    high : float or jax.Array
        Upper bound of the model support.
    """
    cfg = prior_config.get(key, None)
    loc = default
    prior_scale = scale
    prior_low = low
    prior_high = high
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        loc, prior_scale = cfg[:2]
    elif isinstance(cfg, dist.Distribution) and cfg.__class__.__name__ in {
        "TruncatedNormal", "TwoSidedTruncatedDistribution"
    }:
        base = getattr(cfg, "base_dist", cfg)
        loc, prior_scale = base.loc, base.scale
        prior_low = jnp.maximum(jnp.asarray(low, dtype=jnp.float64), jnp.asarray(cfg.low, dtype=jnp.float64))
        prior_high = jnp.minimum(jnp.asarray(high, dtype=jnp.float64), jnp.asarray(cfg.high, dtype=jnp.float64))
    elif isinstance(cfg, dist.Normal):
        loc, prior_scale = cfg.loc, cfg.scale
    elif isinstance(cfg, dist.Distribution):
        raise ValueError(f"prior_config[{key!r}] must be Normal-like to enforce bounded model support.")
    elif isinstance(cfg, dict):
        family = str(cfg.get("dist", cfg.get("family", "normal"))).lower()
        if family not in {"normal", "gaussian", "truncatednormal", "truncated_normal", "truncnormal", "truncnorm"}:
            raise ValueError(f"prior_config[{key!r}] must be Normal-like to enforce bounded model support.")
        loc = cfg.get("loc", default)
        prior_scale = cfg.get("scale", scale)
        if family not in {"normal", "gaussian"}:
            prior_low = jnp.maximum(jnp.asarray(low, dtype=jnp.float64), jnp.asarray(cfg.get("low", low), dtype=jnp.float64))
            prior_high = jnp.minimum(jnp.asarray(high, dtype=jnp.float64), jnp.asarray(cfg.get("high", high), dtype=jnp.float64))
    prior_low = jnp.asarray(prior_low, dtype=jnp.float64)
    prior_high = jnp.asarray(prior_high, dtype=jnp.float64)
    return numpyro.sample(
        key,
        dist.TruncatedNormal(
            jnp.asarray(loc, dtype=jnp.float64),
            jnp.maximum(jnp.asarray(prior_scale, dtype=jnp.float64), 1.0e-6),
            low=prior_low,
            high=prior_high,
        ),
    )


def _sample_optional_truncnorm(prior_config: dict[str, Any], key: str, default: float, scale: float, low, high):
    """Return a fixed default unless a bounded Normal-like prior is configured."""
    if key not in prior_config:
        return jnp.asarray(default, dtype=jnp.float64)
    return _sample_bounded_normal(prior_config, key, default, scale, low, high)


def _safe_log10(x):
    """Take log10 after clipping to a tiny positive floor.

    Parameters
    ----------
    x : float or jax.Array
        Positive values; clipped to [1e-30, 1e300] before log10.
    """
    return jnp.log10(jnp.clip(jnp.asarray(x, dtype=jnp.float64), 1.0e-30, 1.0e300))


def _sample_log_stellar_mass(prior_config: dict[str, Any]):
    """Sample stellar mass with a less top-heavy default prior.

    By default this uses a heavy-tailed Student-t prior centered lower than the
    original Normal(10.5, 2.5) benchmark default. Existing Normal-like
    overrides with only ``loc`` and ``scale`` are still supported.

    Parameters
    ----------
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    """
    return _sample_prior(prior_config, "log_stellar_mass", dist.StudentT(df=5.0, loc=10.0, scale=2.0))


def _ssp_lgmet_solar_offset(
    ssp_lgmet,
    metallicity_coordinate: str = "absolute_log10_z",
    solar_metallicity: float = DSPS_SOLAR_METALLICITY,
):
    """Return the declared solar-metallicity offset of an SSP grid.

    Parameters
    ----------
    ssp_lgmet : jax.Array
        SSP metallicity grid, shape (n_met,); unused, kept for API compatibility.
    """
    del ssp_lgmet
    coordinate = str(metallicity_coordinate).strip().lower()
    if coordinate == "absolute_log10_z":
        return jnp.log10(jnp.asarray(solar_metallicity, dtype=jnp.float64))
    if coordinate == "log10_z_over_zsun":
        return jnp.asarray(0.0, dtype=jnp.float64)
    raise ValueError(f"Unsupported SSP metallicity coordinate: {metallicity_coordinate!r}.")


def _gal_lgmet_to_absolute_z(
    gal_lgmet,
    ssp_lgmet=None,
    metallicity_coordinate: str = "absolute_log10_z",
    solar_metallicity: float = DSPS_SOLAR_METALLICITY,
):
    """Convert galaxy metallicity from the SSP-grid convention to absolute Z.

    Parameters
    ----------
    gal_lgmet : float or jax.Array
        Galaxy log10 metallicity in the SSP-grid coordinate.
    ssp_lgmet : jax.Array
        SSP metallicity grid, shape (n_met,); unused, kept for API compatibility.
    """
    gal_lgmet = jnp.asarray(gal_lgmet, dtype=jnp.float64)
    del ssp_lgmet
    coordinate = str(metallicity_coordinate).strip().lower()
    if coordinate == "absolute_log10_z":
        absolute_logz = gal_lgmet
    elif coordinate == "log10_z_over_zsun":
        absolute_logz = gal_lgmet + jnp.log10(jnp.asarray(solar_metallicity, dtype=jnp.float64))
    else:
        raise ValueError(f"Unsupported SSP metallicity coordinate: {metallicity_coordinate!r}.")
    return jnp.power(10.0, absolute_logz)


def _absolute_z_to_gal_lgmet(
    metallicity,
    *,
    metallicity_coordinate: str = "absolute_log10_z",
    solar_metallicity: float = DSPS_SOLAR_METALLICITY,
):
    """Convert an absolute metal mass fraction to the SSP-grid coordinate."""
    metallicity = jnp.maximum(jnp.asarray(metallicity, dtype=jnp.float64), 1.0e-30)
    coordinate = str(metallicity_coordinate).strip().lower()
    if coordinate == "absolute_log10_z":
        return jnp.log10(metallicity)
    if coordinate == "log10_z_over_zsun":
        return jnp.log10(metallicity / jnp.asarray(solar_metallicity, dtype=jnp.float64))
    raise ValueError(f"Unsupported SSP metallicity coordinate: {metallicity_coordinate!r}.")


def _resolve_tied_metallicity(context: ModelContext, prior_config: dict[str, Any]):
    """Return one shared absolute Z and SSP-coordinate Z when tying is enabled.

    An explicit stellar-metallicity prior takes precedence as the shared
    parameter. Otherwise an explicit nebular-metallicity prior is sampled.
    With neither prior, the configured nebular value is fixed and propagated
    to the host; if it is ``None``, the configured stellar value is used.
    """
    galaxy_cfg = context.fit_config.galaxy
    if not (galaxy_cfg.fit_host and galaxy_cfg.tie_stellar_nebular_metallicity):
        return None, None

    ssp_lgmet = context.host_basis_jax.ssp_lgmet
    if "gal_lgmet" in prior_config:
        gal_lgmet = _sample_bounded_normal(
            prior_config,
            "gal_lgmet",
            _absolute_z_to_gal_lgmet(
                galaxy_cfg.stellar_metallicity,
                metallicity_coordinate=galaxy_cfg.ssp_metallicity_coordinate,
                solar_metallicity=galaxy_cfg.ssp_solar_metallicity,
            ),
            0.5,
            jnp.min(ssp_lgmet),
            jnp.max(ssp_lgmet),
        )
        shared_z = _gal_lgmet_to_absolute_z(
            gal_lgmet,
            metallicity_coordinate=galaxy_cfg.ssp_metallicity_coordinate,
            solar_metallicity=galaxy_cfg.ssp_solar_metallicity,
        )
        return shared_z, gal_lgmet

    nebular_cfg = context.fit_config.nebular
    default_z = (
        float(nebular_cfg.zgas)
        if nebular_cfg.zgas is not None
        else float(galaxy_cfg.stellar_metallicity)
    )
    if "nebular_zgas" in prior_config:
        templates = context.nebular_templates_jax
        shared_z = _sample_optional_truncnorm(
            prior_config,
            "nebular_zgas",
            default_z,
            0.01,
            templates.z_grid[0],
            templates.z_grid[-1],
        )
    else:
        shared_z = jnp.asarray(default_z, dtype=jnp.float64)
    gal_lgmet = _absolute_z_to_gal_lgmet(
        shared_z,
        metallicity_coordinate=galaxy_cfg.ssp_metallicity_coordinate,
        solar_metallicity=galaxy_cfg.ssp_solar_metallicity,
    )
    return shared_z, gal_lgmet


def _default_gal_lgmet_loc(
    ssp_lgmet,
    metallicity_coordinate: str = "absolute_log10_z",
    solar_metallicity: float = DSPS_SOLAR_METALLICITY,
):
    """Default galaxy metallicity center in the SSP grid's metallicity convention.

    Parameters
    ----------
    ssp_lgmet : jax.Array
        SSP log10 metallicity grid, shape (n_met,); bounds the result.
    """
    ssp_lgmet = jnp.asarray(ssp_lgmet, dtype=jnp.float64)
    loc = _ssp_lgmet_solar_offset(ssp_lgmet, metallicity_coordinate, solar_metallicity) - 0.3
    return jnp.clip(loc, jnp.nanmin(ssp_lgmet), jnp.nanmax(ssp_lgmet))


def _cigale_delayed_sfh_shape(elapsed_gyr, tau_gyr, age_gyr):
    """Return the no-burst CIGALE v2025.1 delayed-tau SFH shape.

    This is the continuous-grid equivalent of ``sfhdelayed.py`` in CIGALE
    v2025.1, where ``SFR(t) = t exp(-t / tau) / tau**2``. CIGALE evaluates
    the expression on a 1 Myr grid; jaxsedfit evaluates the same expression on
    its JAX SFH grid and normalizes it to the requested formed stellar mass.

    Upstream source:
    https://gitlab.lam.fr/cigale/cigale/-/blob/v2025.1/pcigale/sed_modules/sfhdelayed.py

    Parameters
    ----------
    elapsed_gyr : object
        Time since the onset of star formation in Gyr.
    tau_gyr : object
        Delayed-SFH e-folding time in Gyr.
    age_gyr : object
        Maximum elapsed time supported by the model in Gyr.
    """
    elapsed_gyr = jnp.asarray(elapsed_gyr, dtype=jnp.float64)
    tau_gyr = jnp.maximum(jnp.asarray(tau_gyr, dtype=jnp.float64), 1.0e-12)
    age_gyr = jnp.asarray(age_gyr, dtype=jnp.float64)
    shape = elapsed_gyr * jnp.exp(-elapsed_gyr / tau_gyr) / (tau_gyr * tau_gyr)
    return jnp.where((elapsed_gyr > 0.0) & (elapsed_gyr <= age_gyr), shape, 0.0)


def _ssp_log_age_bin_edges(ssp_lg_age_gyr):
    """Return logarithmic bin edges centered on the tabulated SSP ages."""
    lg_age = jnp.asarray(ssp_lg_age_gyr, dtype=jnp.float64)
    interior = 0.5 * (lg_age[:-1] + lg_age[1:])
    first = lg_age[0] - 0.5 * (lg_age[1] - lg_age[0])
    last = lg_age[-1] + 0.5 * (lg_age[-1] - lg_age[-2])
    return jnp.concatenate((first[None], interior, last[None]))


def _delayed_sfh_cumulative_mass(elapsed_gyr, tau_gyr):
    """Integral from zero to ``elapsed_gyr`` of the delayed-tau SFH shape."""
    elapsed_gyr = jnp.maximum(jnp.asarray(elapsed_gyr, dtype=jnp.float64), 0.0)
    tau_gyr = jnp.maximum(jnp.asarray(tau_gyr, dtype=jnp.float64), 1.0e-12)
    x = elapsed_gyr / tau_gyr
    return -jnp.expm1(-x) - x * jnp.exp(-x)


def _analytic_delayed_ssp_weights(
    age_gyr,
    tau_gyr,
    gal_lgmet,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
):
    """Exact delayed-SFH mass weights in the native SSP stellar-age bins.

    The delayed SFH has an analytic antiderivative, so its mass contribution
    to each SSP age bin does not require an auxiliary cosmic-time grid.
    """
    age_weights = _analytic_delayed_age_weights(age_gyr, tau_gyr, ssp_lg_age_gyr)
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        gal_lgmet, gal_lgmet_scatter, ssp_lgmet
    )
    weights = lgmet_weights[:, None] * age_weights[None, :]
    weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-30)
    return weights, lgmet_weights, age_weights


def _analytic_delayed_age_weights(age_gyr, tau_gyr, ssp_lg_age_gyr):
    """Exact normalized delayed-SFH mass weights in SSP stellar-age bins."""
    age_gyr = jnp.maximum(jnp.asarray(age_gyr, dtype=jnp.float64), 0.0)
    age_edges_gyr = 10.0 ** _ssp_log_age_bin_edges(ssp_lg_age_gyr)
    # Stellar age increases in the opposite direction to formation time.
    elapsed_lo = jnp.clip(age_gyr - age_edges_gyr[1:], 0.0, age_gyr)
    elapsed_hi = jnp.clip(age_gyr - age_edges_gyr[:-1], 0.0, age_gyr)
    bin_mass = _delayed_sfh_cumulative_mass(elapsed_hi, tau_gyr) - _delayed_sfh_cumulative_mass(
        elapsed_lo, tau_gyr
    )
    return bin_mass / jnp.maximum(jnp.sum(bin_mass), 1.0e-30)


def _analytic_delayed_burst_age_weights(
    age_gyr,
    tau_gyr,
    burst_fraction,
    burst_age_gyr,
    burst_tau_gyr,
    ssp_lg_age_gyr,
):
    """Smooth SSP age weights for a delayed SFH plus exponential burst.

    The recent burst has a short logistic rise rather than a discontinuous
    Heaviside onset. Its analytic smooth cumulative mass keeps derivatives
    continuous as the burst onset moves across SSP bin edges.
    """
    age_gyr = jnp.maximum(jnp.asarray(age_gyr, dtype=jnp.float64), 0.0)
    burst_fraction = jnp.clip(
        jnp.asarray(burst_fraction, dtype=jnp.float64), 0.0, 1.0 - 1.0e-8
    )
    burst_age_gyr = jnp.clip(
        jnp.asarray(burst_age_gyr, dtype=jnp.float64), 1.0e-12, age_gyr
    )
    burst_tau_gyr = jnp.maximum(
        jnp.asarray(burst_tau_gyr, dtype=jnp.float64), 1.0e-12
    )
    age_edges_gyr = 10.0 ** _ssp_log_age_bin_edges(ssp_lg_age_gyr)
    elapsed_lo = jnp.clip(age_gyr - age_edges_gyr[1:], 0.0, age_gyr)
    elapsed_hi = jnp.clip(age_gyr - age_edges_gyr[:-1], 0.0, age_gyr)

    main_bin_mass = _delayed_sfh_cumulative_mass(
        elapsed_hi, tau_gyr
    ) - _delayed_sfh_cumulative_mass(elapsed_lo, tau_gyr)
    burst_bin_mass = _exponential_burst_cumulative_mass(
        elapsed_hi, age_gyr, burst_age_gyr, burst_tau_gyr
    ) - _exponential_burst_cumulative_mass(
        elapsed_lo, age_gyr, burst_age_gyr, burst_tau_gyr
    )
    main_total = _delayed_sfh_cumulative_mass(age_gyr, tau_gyr)
    burst_total = jnp.sum(burst_bin_mass)
    burst_scale = (
        burst_fraction
        / jnp.maximum(1.0 - burst_fraction, 1.0e-8)
        * main_total
        / jnp.maximum(burst_total, 1.0e-30)
    )
    age_weights = main_bin_mass + burst_scale * burst_bin_mass
    return age_weights / jnp.maximum(jnp.sum(age_weights), 1.0e-30)


def _exponential_burst_cumulative_mass(elapsed_gyr, age_gyr, burst_age_gyr, burst_tau_gyr):
    """Smooth cumulative mass of the unit-amplitude exponential burst."""
    burst_start = age_gyr - burst_age_gyr
    rise_gyr = jnp.asarray(BURST_RISE_GYR, dtype=jnp.float64)
    duration = rise_gyr * (
        jax.nn.softplus((elapsed_gyr - burst_start) / rise_gyr)
        - jax.nn.softplus(-burst_start / rise_gyr)
    )
    return burst_tau_gyr * (-jnp.expm1(-duration / burst_tau_gyr))


def _smooth_exponential_burst_shape(
    elapsed_gyr,
    age_gyr,
    burst_age_gyr,
    burst_tau_gyr,
    rise_gyr=BURST_RISE_GYR,
):
    """Return an exponential burst with a smooth, positive logistic onset."""
    elapsed_gyr = jnp.asarray(elapsed_gyr, dtype=jnp.float64)
    burst_start = jnp.asarray(age_gyr, dtype=jnp.float64) - jnp.asarray(
        burst_age_gyr, dtype=jnp.float64
    )
    burst_tau_gyr = jnp.maximum(jnp.asarray(burst_tau_gyr, dtype=jnp.float64), 1.0e-12)
    rise_gyr = jnp.asarray(rise_gyr, dtype=jnp.float64)
    since_start = elapsed_gyr - burst_start
    smooth_duration = rise_gyr * (
        jax.nn.softplus(since_start / rise_gyr)
        - jax.nn.softplus(-burst_start / rise_gyr)
    )
    log_shape = jax.nn.log_sigmoid(since_start / rise_gyr) - smooth_duration / burst_tau_gyr
    return jnp.exp(log_shape)


def _diffstar_ssp_age_weights(
    bounded_diffstar_params,
    ssp_lg_age_gyr,
    t_obs_gyr,
    t_birth_min_gyr=0.01,
    quad_nodes=_SSP_BIN_QUAD_NODES,
    quad_weights=_SSP_BIN_QUAD_WEIGHTS,
):
    """Integrate a Diffstar SFH directly inside every native SSP age bin."""
    t_obs_gyr = jnp.asarray(t_obs_gyr, dtype=jnp.float64)
    t_birth_min_gyr = jnp.asarray(t_birth_min_gyr, dtype=jnp.float64)
    stellar_age_edges = 10.0 ** _ssp_log_age_bin_edges(ssp_lg_age_gyr)
    birth_lo = jnp.clip(t_obs_gyr - stellar_age_edges[1:], t_birth_min_gyr, t_obs_gyr)
    birth_hi = jnp.clip(t_obs_gyr - stellar_age_edges[:-1], t_birth_min_gyr, t_obs_gyr)
    half_width = 0.5 * jnp.maximum(birth_hi - birth_lo, 0.0)
    midpoint = 0.5 * (birth_hi + birth_lo)
    nodes = midpoint[:, None] + half_width[:, None] * jnp.asarray(
        quad_nodes, dtype=jnp.float64
    )[None, :]
    sfr = calc_sfh_singlegal(
        bounded_diffstar_params,
        DEFAULT_MAH_PARAMS,
        nodes.reshape((-1,)),
        lgt0=DIFFSTAR_LGT0,
        fb=DIFFSTAR_FB,
        return_smh=False,
    ).reshape(nodes.shape)
    bin_mass = half_width * jnp.sum(
        sfr * jnp.asarray(quad_weights, dtype=jnp.float64)[None, :], axis=1
    )
    age_weights = bin_mass / jnp.maximum(jnp.sum(bin_mass), 1.0e-30)
    return age_weights, bin_mass


def _cigale_delayed_burst_sfh_shape(
    elapsed_gyr,
    tau_gyr,
    age_gyr,
    burst_fraction,
    burst_age_gyr,
    burst_tau_gyr,
):
    """Return CIGALE ``sfhdelayed`` with its optional exponential burst.

    ``burst_fraction`` is the fraction of total formed mass in the recent
    component.  CIGALE scales the exponential burst so that its time integral
    relative to the main delayed component is ``f_burst / (1 - f_burst)``.
    The normalization uses the analytic time integrals of both components (the
    same ones used for the formed mass), so it does not depend on how well the
    SFH grid resolves a short main e-folding time.
    """
    elapsed_gyr = jnp.asarray(elapsed_gyr, dtype=jnp.float64)
    age_gyr = jnp.asarray(age_gyr, dtype=jnp.float64)
    burst_fraction = jnp.clip(jnp.asarray(burst_fraction, dtype=jnp.float64), 0.0, 1.0 - 1.0e-8)
    burst_age_gyr = jnp.clip(jnp.asarray(burst_age_gyr, dtype=jnp.float64), 1.0e-6, age_gyr)
    burst_tau_gyr = jnp.maximum(jnp.asarray(burst_tau_gyr, dtype=jnp.float64), 1.0e-12)

    main = _cigale_delayed_sfh_shape(elapsed_gyr, tau_gyr, age_gyr)
    burst = _smooth_exponential_burst_shape(
        elapsed_gyr,
        age_gyr,
        burst_age_gyr,
        burst_tau_gyr,
    )
    main_integral = _delayed_sfh_cumulative_mass(age_gyr, tau_gyr)
    burst_integral = _exponential_burst_cumulative_mass(age_gyr, age_gyr, burst_age_gyr, burst_tau_gyr)
    burst_scale = (
        burst_fraction
        / jnp.maximum(1.0 - burst_fraction, 1.0e-8)
        * main_integral
        / jnp.maximum(burst_integral, 1.0e-30)
    )
    return main + burst_scale * burst


def _shift_and_broaden_single_spectrum_lnlam(lnwave, spectrum, v_kms, sigma_kms):
    """Apply a velocity shift and Gaussian broadening in log-wavelength space.

    Parameters
    ----------
    lnwave : jax.Array
        Natural-log wavelength grid (Angstrom), shape (n_pix,).
    spectrum : jax.Array
        Spectrum sampled on ``lnwave``, shape (n_pix,).
    v_kms : float or jax.Array
        Velocity shift in km/s.
    sigma_kms : float or jax.Array
        Gaussian velocity dispersion in km/s.
    """
    return _fourier_shift_and_broaden_lnlam(
        lnwave,
        spectrum,
        v_kms,
        sigma_kms,
    )


def _powerlaw_jax(wave, norm, lam1, lam2, x0, xbrk, bend_width, cutoff):
    """Evaluate the bent AGN disk power-law continuum.

    Parameters
    ----------
    wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    norm : float or jax.Array
        L_lambda at the pivot ``x0`` in W/Angstrom.
    lam1 : float or jax.Array
        Power-law index blueward of the bend (GRAHSP ``uv_slope``).
    lam2 : float or jax.Array
        Power-law index redward of the bend (GRAHSP ``pl_slope``).
    x0 : float
        Pivot wavelength in Angstrom (5100 in the model).
    xbrk : float or jax.Array
        Bend wavelength in Angstrom.
    bend_width : float or jax.Array
        Bend smoothness parameter (larger is smoother).
    cutoff : float or jax.Array
        Long-wavelength exponential cutoff in Angstrom; disabled if <= 0.
    """
    expo = 1.0 / jnp.maximum(bend_width, 1e-6)
    lamaddexpo = (lam1 + lam2 + 2.0) / 2.0
    lamsubexpo = (lam2 - lam1) / 2.0 * jnp.maximum(bend_width, 1e-6)
    xpivratio = x0 / xbrk
    divisor = 1.0 / (xpivratio**expo + xpivratio**-expo)
    xratio = wave / xbrk
    bbb = norm * (wave / x0) ** lamaddexpo * ((xratio**expo + xratio**-expo) * divisor) ** lamsubexpo * (x0 / wave)
    cutoff_factor = -jnp.expm1(-(jnp.maximum(cutoff, 0.0) / wave))
    return jnp.where(cutoff > 0.0, bbb * cutoff_factor, bbb)


def _torus_component(wave, fcov, si, cool_lam, cool_width, hot_lam, hot_width, hot_fcov, si_ratio, si_em_lam, si_abs_lam, si_em_width, si_abs_width, l_agn):
    """Evaluate the empirical torus component on the rest-frame wavelength grid.

    The torus luminosity follows the GRAHSP-style empirical normalization from
    the AGN luminosity and covering factor proxy. It is not computed from the
    luminosity absorbed by the AGN attenuation curve.

    Parameters
    ----------
    wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    fcov : float or jax.Array
        Torus covering-factor proxy scaling the torus luminosity.
    si : object
        Signed latent controlling the silicate modulation. Positive values
        produce emission near ``si_em_lam`` and negative values produce
        absorption; a hyperbolic-tangent transform bounds the fractional
        modulation so the total torus spectrum remains strictly positive.
    cool_lam : float or jax.Array
        Cool-dust peak wavelength in micron.
    cool_width : float or jax.Array
        Cool-dust log-normal width in dex of wavelength.
    hot_lam : float or jax.Array
        Hot-dust peak wavelength in micron.
    hot_width : float or jax.Array
        Hot-dust log-normal width in dex of wavelength.
    hot_fcov : float or jax.Array
        Hot-to-cool dust amplitude ratio.
    si_ratio : float
        Silicate absorption-to-emission profile amplitude ratio.
    si_em_lam : float
        Silicate emission centre in Angstrom.
    si_abs_lam : float
        Silicate absorption centre in Angstrom.
    si_em_width : float
        Silicate emission Gaussian sigma in Angstrom.
    si_abs_width : float
        Silicate absorption Gaussian sigma in Angstrom.
    l_agn : float or jax.Array
        AGN lambda L_lambda at 5100 Angstrom in W.
    """
    log_wave_um = jnp.log10(wave / 10000.0)
    log_cool = jnp.log10(cool_lam)
    log_hot = jnp.log10(hot_lam)
    cool = jnp.exp(-((log_wave_um - log_cool) / cool_width) ** 2)
    hot = hot_fcov * 10 ** (log_cool - log_hot) * jnp.exp(-((log_wave_um - log_hot) / hot_width) ** 2)
    total = cool + hot
    norm_index = jnp.argmin(jnp.abs(wave - GRAHSP_TORUS_NORM_A))
    l_torus = 2.5 * l_agn * fcov
    torus = l_torus / GRAHSP_TORUS_NORM_A * total / jnp.maximum(total[norm_index], 1e-30)
    si_profile = jnp.exp(-0.5 * ((wave - si_em_lam) / si_em_width) ** 2) - si_ratio * jnp.exp(
        -0.5 * ((wave - si_abs_lam) / si_abs_width) ** 2
    )
    # Keep a small positive margin even when tanh saturates numerically at one.
    # Unlike clipping the additive feature at -torus, this transformation is
    # smooth everywhere and therefore well behaved for gradient-based inference.
    si_fraction = (1.0 - 1.0e-6) * jnp.tanh(si)
    return torus * (1.0 + si_fraction * si_profile)


def _feii_component(wave, template_flux_on_wave, norm, fwhm_kms, shift_frac):
    """Broaden, shift, and normalize the Fe II template contribution.

    Parameters
    ----------
    wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    template_flux_on_wave : jax.Array
        Fe II template interpolated onto ``wave``, shape (n_wave,).
    norm : float or jax.Array
        Fe II amplitude in W/Angstrom multiplying the template.
    fwhm_kms : float or jax.Array
        Fe II broadening FWHM in km/s.
    shift_frac : float or jax.Array
        Fractional velocity shift v/c.
    """
    sigma_kms = jnp.maximum(fwhm_kms / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0))), 10.0)
    return norm * _shift_and_broaden_single_spectrum_lnlam(jnp.log(wave), jnp.maximum(template_flux_on_wave, 0.0), C_KMS * shift_frac, sigma_kms)


def _line_gaussians(wave, line_wave, line_lumin, width_kms):
    """Evaluate a summed Gaussian emission-line template with one shared width.

    Parameters
    ----------
    wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Per-line GRAHSP amplitudes in W/Angstrom (L_lambda-scaled), shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    """
    fwhm_to_sigma_conversion = 1 / (2 * jnp.sqrt(2 * jnp.log(2)))
    width_wave = line_wave * (width_kms * 1000.0) / 299792458.0
    sigma = width_wave * fwhm_to_sigma_conversion
    z = (wave[:, None] - line_wave[None, :]) / jnp.maximum(sigma[None, :], 1e-12)
    norm = 5100.0 / jnp.sqrt(jnp.pi * sigma**2)
    return jnp.sum(line_lumin[None, :] * jnp.exp(-0.5 * z * z) * norm[None, :], axis=1)


def _flux_conserving_line_gaussians(wave, line_wave, line_lumin, width_kms):
    """Evaluate CIGALE-style nebular lines preserving integrated luminosity.

    Parameters
    ----------
    wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Integrated line luminosities in W, shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    """
    fwhm_wave = jnp.maximum(line_wave * width_kms / C_KMS, 1.0e-8)
    sigma = fwhm_wave / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    z = (wave[:, None] - line_wave[None, :]) / jnp.maximum(sigma[None, :], 1.0e-12)
    profile = jnp.exp(-0.5 * z * z) / jnp.maximum(sigma[None, :] * jnp.sqrt(2.0 * jnp.pi), 1.0e-30)
    return jnp.sum(line_lumin[None, :] * profile, axis=1)


def _interp_grid_axis(grid, value, *, log_scale: bool = False):
    """Return bracketing indices and interpolation weight for one template axis.

    Parameters
    ----------
    grid : jax.Array
        Monotonically increasing 1D template axis.
    value : float or jax.Array
        Value to bracket; clipped to the grid range.
    log_scale : bool
        Interpolate in log10 of the axis and value.
    """
    grid = jnp.asarray(grid, dtype=jnp.float64)
    x_grid = jnp.log10(jnp.maximum(grid, 1.0e-300)) if log_scale else grid
    x = jnp.log10(jnp.maximum(value, 1.0e-300)) if log_scale else jnp.asarray(value, dtype=jnp.float64)
    x = jnp.clip(x, x_grid[0], x_grid[-1])
    upper = jnp.clip(jnp.searchsorted(x_grid, x, side="right"), 1, grid.shape[0] - 1)
    lower = upper - 1
    x0 = x_grid[lower]
    x1 = x_grid[upper]
    weight = jnp.clip((x - x0) / jnp.maximum(x1 - x0, 1.0e-300), 0.0, 1.0)
    return lower, upper, weight


def _trilinear_nebular_grid(values, z_grid, logu_grid, ne_grid, zgas, logu, ne):
    """Interpolate a nebular template grid in log Z, log U, and log density.

    Parameters
    ----------
    values : jax.Array
        Template grid with leading axes (n_z, n_logu, n_ne, ...).
    z_grid : jax.Array
        Gas metallicity (absolute Z) grid, shape (n_z,).
    logu_grid : jax.Array
        log10 ionization-parameter grid, shape (n_logu,).
    ne_grid : jax.Array
        Electron-density grid in cm^-3, shape (n_ne,).
    zgas : float or jax.Array
        Gas metallicity (absolute Z); interpolated in log10.
    logu : float or jax.Array
        log10 ionization parameter.
    ne : float or jax.Array
        Electron density in cm^-3; interpolated in log10.
    """
    z0, z1, wz = _interp_grid_axis(z_grid, zgas, log_scale=True)
    u0, u1, wu = _interp_grid_axis(logu_grid, logu, log_scale=False)
    n0, n1, wn = _interp_grid_axis(ne_grid, ne, log_scale=True)

    c000 = values[z0, u0, n0]
    c001 = values[z0, u0, n1]
    c010 = values[z0, u1, n0]
    c011 = values[z0, u1, n1]
    c100 = values[z1, u0, n0]
    c101 = values[z1, u0, n1]
    c110 = values[z1, u1, n0]
    c111 = values[z1, u1, n1]

    c00 = c000 * (1.0 - wn) + c001 * wn
    c01 = c010 * (1.0 - wn) + c011 * wn
    c10 = c100 * (1.0 - wn) + c101 * wn
    c11 = c110 * (1.0 - wn) + c111 * wn
    c0 = c00 * (1.0 - wu) + c01 * wu
    c1 = c10 * (1.0 - wu) + c11 * wu
    return c0 * (1.0 - wz) + c1 * wz


def _cigale_nebular_correction(f_esc, f_dust):
    """CIGALE nebular escape/dust correction factor.

    Parameters
    ----------
    f_esc : float or jax.Array
        Fraction of Lyman-continuum photons escaping the galaxy.
    f_dust : float or jax.Array
        Fraction of Lyman-continuum photons absorbed by dust.
    """
    alpha_b = jnp.asarray(2.58e-19, dtype=jnp.float64)
    alpha_1 = jnp.asarray(1.54e-19, dtype=jnp.float64)
    escaped_or_dust = f_esc + f_dust
    return jnp.clip((1.0 - escaped_or_dust) / (1.0 + alpha_1 / alpha_b * escaped_or_dust), 0.0, 1.0)


def _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel):
    """Evaluate the GRAHSP ``activatelines`` Balmer continuum template.

    The analytic edge broadening follows GRAHSP commit
    ``7d35f5232ac9918a785e8dfe75dff693ab246daf`` after converting nm to
    Angstrom. ``balmer_norm`` is the absolute L-lambda normalization; the
    caller converts GRAHSP's dimensionless ``ABC`` strength into that unit.

    Parameters
    ----------
    wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    balmer_norm : float or jax.Array
        L_lambda at the Balmer edge in W/Angstrom.
    balmer_te : float
        Electron temperature in K.
    balmer_tau : float or jax.Array
        Optical depth at the Balmer edge.
    balmer_vel : float or jax.Array
        Velocity broadening of the edge in km/s (Gaussian sigma).
    """
    lam_be = 3646.0
    h_c_per_k_B = 1.439e8
    bb = (wave**-5) / jnp.expm1(jnp.clip(h_c_per_k_B / (balmer_te * wave), 1e-9, 700.0))
    bb0 = (lam_be**-5) / jnp.expm1(jnp.clip(h_c_per_k_B / (balmer_te * lam_be), 1e-9, 700.0))
    wave_ratio = wave / lam_be
    truncation = -jnp.expm1(-balmer_tau * wave_ratio**3)
    truncation0 = -jnp.expm1(-balmer_tau)

    # GRAHSP analytically convolves a linear approximation to the Balmer edge.
    alpha = 1.8
    beta = -0.8
    sigma = jnp.maximum(balmer_vel / C_KMS, 1.0e-12)
    z = (wave_ratio - 1.0) / (jnp.sqrt(2.0) * sigma)
    term_b = 0.5 * (1.0 - jax.lax.erf(z))
    term_a1 = 0.5 * wave_ratio
    term_a2 = -0.5 * wave_ratio * jax.lax.erf(z)
    term_a3 = -sigma / jnp.sqrt(2.0 * jnp.pi) * jnp.exp(-(z**2))
    convolved = (beta * term_b + alpha * (term_a1 + term_a2 + term_a3)) * truncation0
    truncation_broadened = jnp.where(wave > 2500.0, convolved, truncation)
    bc = balmer_norm * bb / jnp.maximum(bb0, 1.0e-30) * truncation_broadened / jnp.maximum(truncation0, 1.0e-30)
    return jnp.where(wave <= lam_be, bc, 0.0)


def _attenuation_curve(wave_rest, opt_index, nir_index, norm, lam_break):
    """Return the broken power-law attenuation curve in magnitudes.

    Parameters
    ----------
    wave_rest : jax.Array
        Rest-frame wavelengths in Angstrom.
    opt_index : float
        Power-law index blueward of ``lam_break``.
    nir_index : float
        Power-law index redward of ``lam_break``.
    norm : float
        Curve value (mag per unit E(B-V)) at ``lam_break``.
    lam_break : float
        Break wavelength in Angstrom.
    """
    return norm * (wave_rest / lam_break) ** jnp.where(wave_rest < lam_break, opt_index, nir_index)


def _calzetti_k(wave_rest):
    """Return the Calzetti et al. (2000) starburst curve k(lambda) = A(lambda) / E(B-V).

    The published fit spans 0.12-2.2 micron. Shortward it is extended linearly in
    wavelength with the 0.12 micron slope, and longward as a power law with the
    2.2 micron logarithmic slope, so the curve is continuous and positive.
    """
    lam = jnp.clip(jnp.asarray(wave_rest, dtype=jnp.float64) / 1.0e4, 1.0e-3, None)

    def uv(x):
        return 2.659 * (-2.156 + 1.509 / x - 0.198 / x**2 + 0.011 / x**3) + CALZETTI_RV

    def ir(x):
        return 2.659 * (-1.857 + 1.040 / x) + CALZETTI_RV

    uv_slope_012 = 2.659 * (-1.509 / 0.12**2 + 0.396 / 0.12**3 - 0.033 / 0.12**4)
    ir_logslope_22 = -2.659 * 1.040 / 2.2 / ir(2.2)
    k_short = uv(0.12) + uv_slope_012 * (lam - 0.12)
    k_uv = uv(jnp.clip(lam, 0.12, 0.63))
    k_ir = ir(jnp.clip(lam, 0.63, 2.2))
    k_long = ir(2.2) * (jnp.maximum(lam, 2.2) / 2.2) ** ir_logslope_22
    k = jnp.where(lam < 0.12, k_short, jnp.where(lam < 0.63, k_uv, jnp.where(lam <= 2.2, k_ir, k_long)))
    return jnp.clip(k, 0.0, None)


def _noll2009_bump_strength(delta):
    """Return the 2175 A bump amplitude tied to the slope (Kriek & Conroy 2013)."""
    return jnp.clip(KRIEK_CONROY_BUMP_INTERCEPT + KRIEK_CONROY_BUMP_SLOPE * jnp.asarray(delta, dtype=jnp.float64), 0.0, None)


def _noll2009_attenuation_curve(wave_rest, delta):
    """Return the Noll et al. (2009) modified Calzetti curve per unit E(B-V).

    ``[k_Calzetti(lambda) + D(lambda; E_b)] (lambda / 5500 A)^delta`` with a Drude 2175 A
    bump of amplitude ``E_b = 0.85 - 1.9 delta`` (Kriek & Conroy 2013), renormalized so
    that A_B - A_V = 1: the sampled screen amplitude stays a B-V colour excess for any
    slope. ``delta = 0`` is Calzetti plus a Milky-Way-strength (E_b = 0.85) bump.
    """
    delta = jnp.asarray(delta, dtype=jnp.float64)
    bump_strength = _noll2009_bump_strength(delta)

    def raw(wave):
        wave = jnp.asarray(wave, dtype=jnp.float64)
        width_term = (wave * NOLL_BUMP_FWHM_A) ** 2
        drude = bump_strength * width_term / ((wave**2 - NOLL_BUMP_WAVE_A**2) ** 2 + width_term)
        return (_calzetti_k(wave) + drude) * (wave / NOLL_V_WAVE_A) ** delta

    colour_excess = raw(NOLL_B_WAVE_A) - raw(NOLL_V_WAVE_A)
    return raw(wave_rest) / colour_excess


class DustScreen(NamedTuple):
    """Foreground screens seen by one component when the galaxy curve is not GRAHSP's.

    ``ebv_host`` passes through the galaxy curve (Noll et al. 2009 with slope
    ``noll_delta``); ``ebv_nuclear`` is the AGN's own screen and keeps the GRAHSP
    curve. Line projectors also accept a bare E(B-V), which is the default
    single-curve case.
    """

    ebv_host: Any
    ebv_nuclear: Any
    noll_delta: Any


def _screen_magnitudes(screen, wave_rest, grahsp_curve=None):
    """Return the attenuation A(lambda) in magnitudes for a screen or a bare E(B-V).

    ``grahsp_curve`` is the GRAHSP curve already evaluated at ``wave_rest`` (cached
    paths precompute it); it is evaluated here when not supplied.
    """
    if grahsp_curve is None:
        grahsp_curve = _attenuation_curve(wave_rest, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    if not isinstance(screen, DustScreen):
        return jnp.asarray(screen, dtype=jnp.float64) * grahsp_curve
    host_curve = _noll2009_attenuation_curve(wave_rest, screen.noll_delta)
    return (
        jnp.asarray(screen.ebv_host, dtype=jnp.float64) * host_curve
        + jnp.asarray(screen.ebv_nuclear, dtype=jnp.float64) * grahsp_curve
    )


def _absorbed_line_luminosity(line_wave, line_lumin, ebv, opt_index, nir_index, norm, lam_break, noll_delta=None):
    """Return attenuation-absorbed energy from integrated narrow-line luminosities.

    With ``noll_delta`` the host screen uses the Noll et al. (2009) curve instead.
    """
    if noll_delta is None:
        curve = _attenuation_curve(line_wave, opt_index, nir_index, norm, lam_break)
    else:
        curve = _noll2009_attenuation_curve(line_wave, noll_delta)
    transmitted = 10 ** (jnp.asarray(ebv, dtype=jnp.float64) * curve / -2.5)
    return jnp.sum(jnp.asarray(line_lumin, dtype=jnp.float64) * (1.0 - transmitted))


def _apply_biattenuation(
    wave_rest, gal_spec, agn_spec, ebv_gal, ebv_agn, opt_index, nir_index, norm, lam_break, noll_delta=None
):
    """Apply differential attenuation to host and AGN components.

    The returned ``dust_luminosity`` is the host-galaxy luminosity absorbed by
    the galaxy attenuation curve. AGN light is attenuated separately, but its
    absorbed luminosity is not added to the host dust energy-balance budget.
    With ``noll_delta`` the host screen (seen by host and AGN light) uses the
    Noll et al. (2009) curve, and the AGN's own screen keeps the broken power law.

    Parameters
    ----------
    wave_rest : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    gal_spec : jax.Array
        Intrinsic host (stellar + nebular) L_lambda in W/Angstrom, shape (n_wave,).
    agn_spec : jax.Array
        Intrinsic attenuable AGN L_lambda in W/Angstrom, shape (n_wave,).
    ebv_gal : float or jax.Array
        Host-galaxy E(B-V) in magnitudes (also seen by the AGN).
    ebv_agn : float or jax.Array
        Additional nuclear AGN E(B-V) in magnitudes.
    opt_index : float
        Broken power-law index blueward of ``lam_break``.
    nir_index : float
        Broken power-law index redward of ``lam_break``.
    norm : float
        Broken power-law curve value at ``lam_break``.
    lam_break : float
        Break wavelength in Angstrom.
    """
    curve = _attenuation_curve(wave_rest, opt_index, nir_index, norm, lam_break)
    if noll_delta is None:
        gal_att = gal_spec * 10 ** (ebv_gal * curve / -2.5)
        agn_att = agn_spec * 10 ** ((ebv_gal + ebv_agn) * curve / -2.5)
    else:
        host_mag = ebv_gal * _noll2009_attenuation_curve(wave_rest, noll_delta)
        gal_att = gal_spec * 10 ** (host_mag / -2.5)
        agn_att = agn_spec * 10 ** ((host_mag + ebv_agn * curve) / -2.5)
    host_absorbed = jnp.clip(gal_spec - gal_att, 0.0, None)
    dust_luminosity = jnp.clip(jnp.trapezoid(host_absorbed, wave_rest), 0.0, None)
    return gal_att, agn_att, host_absorbed, dust_luminosity


def _attenuation_transmitted_fraction(direct_attenuated, direct_intrinsic):
    """Return the attenuation-only transmitted fraction for direct components.

    This excludes re-emitted host dust and empirical torus emission so the
    attenuation model uncertainty is controlled only by components that pass
    through the attenuation curve.

    Parameters
    ----------
    direct_attenuated : jax.Array
        Attenuated direct (host + AGN) luminosity or flux.
    direct_intrinsic : jax.Array
        Unattenuated direct luminosity or flux, same shape.
    """
    return jnp.clip(
        direct_attenuated / jnp.maximum(direct_intrinsic, 1.0e-30),
        1.0e-4,
        1.0,
    )


def _band_transmitted_fraction(direct_attenuated_flux, direct_intrinsic_flux):
    """Return a dimensionless bandwise attenuated-to-intrinsic flux ratio."""
    attenuated = jnp.asarray(direct_attenuated_flux, dtype=jnp.float64)
    intrinsic = jnp.asarray(direct_intrinsic_flux, dtype=jnp.float64)
    ratio = attenuated / jnp.maximum(intrinsic, 1.0e-30)
    return jnp.where(intrinsic > 0.0, jnp.clip(ratio, 1.0e-4, 1.0), 1.0)


def _apply_extended_capture(total_flux, extended_flux, capture_fraction):
    """Return total flux after aperture capture of extended components only.

    Parameters
    ----------
    total_flux : jax.Array
        Total model band fluxes in mJy, shape (n_filters,).
    extended_flux : jax.Array
        Extended-component part of ``total_flux`` in mJy.
    capture_fraction : float or jax.Array
        Captured fraction of the extended light, in [0, 1].
    """
    total_flux = jnp.asarray(total_flux, dtype=jnp.float64)
    extended_flux = jnp.asarray(extended_flux, dtype=jnp.float64)
    capture_fraction = jnp.asarray(capture_fraction, dtype=jnp.float64)
    return total_flux - extended_flux + capture_fraction * extended_flux


def _redshift_to_obs(rest_wave, rest_lum, obs_wave, redshift, luminosity_distance_m):
    """Project a rest-frame luminosity density to the observed frame.

    Parameters
    ----------
    rest_wave : jax.Array
        Rest-frame wavelengths in Angstrom, shape (n_wave,).
    rest_lum : jax.Array
        Rest-frame L_lambda in W/Angstrom, shape (n_wave,).
    obs_wave : jax.Array
        Observed-frame output wavelengths in Angstrom.
    redshift : float or jax.Array
        Source redshift.
    luminosity_distance_m : float or jax.Array
        Luminosity distance in meters.
    """
    wave_obs = rest_wave * (1.0 + redshift)
    flux_obs = rest_lum / (4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1e-12) ** 2 * jnp.maximum(1.0 + redshift, 1e-8))
    return jnp.interp(obs_wave, wave_obs, flux_obs, left=0.0, right=0.0)


def _project_filters(obs_flux, packed_filters):
    """Project an observed-frame spectrum through prepared filters.

    The final f_lambda-to-mJy conversion uses each filter's pivot wavelength
    squared. Together with the energy-weighted transmission prepared by the
    filter loader, this is algebraically identical to CIGALE's direct F_nu
    integral.

    Parameters
    ----------
    obs_flux : jax.Array
        Observed f_lambda in W m^-2 Angstrom^-1 on the context observed grid.
    packed_filters : PackedFiltersJax
        Packed filter interpolation indices, weights, and transmissions.
    """
    interp_indices = packed_filters.interp_indices
    interp_weight = packed_filters.interp_weight
    transmission = packed_filters.transmission
    work_wave = packed_filters.work_wave
    effective_wavelength = packed_filters.effective_wavelength
    valid_mask = packed_filters.valid_mask

    left = obs_flux[interp_indices]
    right = obs_flux[interp_indices + 1]
    values = left * (1.0 - interp_weight) + right * interp_weight
    values = jnp.where(valid_mask, values, 0.0)
    weighted_trans = jnp.where(valid_mask, transmission, 0.0)
    weighted_wave = work_wave
    numer = jnp.trapezoid(values * weighted_trans, weighted_wave, axis=1)
    denom = jnp.maximum(jnp.trapezoid(weighted_trans, weighted_wave, axis=1), 1e-30)
    f_lambda = numer / denom
    return 1e-10 / 299792458.0 * 1e29 * effective_wavelength * effective_wavelength * f_lambda


def _can_use_fixed_filter_projection(context: ModelContext, cfg) -> bool:
    """Return whether cached fixed-z photometric projection matrices are valid.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    cfg : FitConfig
        Fit configuration (normally ``context.fit_config``).
    """
    return bool(
        cfg.likelihood.use_fast_photometry_projection
        and not cfg.observation.fits_redshift
        and context.fixed_filter_projection_jax is not None
        and context.fixed_scalar_filter_projection_jax is not None
    )


def _project_rest_luminosity_filters(context: ModelContext, rest_lum):
    """Project fixed-redshift rest luminosity density directly into filter mJy.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    rest_lum : jax.Array
        Rest-frame L_lambda in W/Angstrom on ``context.rest_wave_jax``, shape (n_wave,).
    """
    return context.fixed_filter_projection_jax @ rest_lum


def _interp_redshift_projection_matrix(redshift, grid, matrices):
    """Linearly interpolate a redshift-tabulated projection matrix.

    Parameters
    ----------
    redshift : float or jax.Array
        Source redshift.
    grid : jax.Array
        Increasing redshift grid, shape (n_z,).
    matrices : jax.Array
        Projection matrices tabulated on ``grid``, shape (n_z, ...).
    """
    z = jnp.asarray(redshift, dtype=jnp.float64)
    idx_hi = jnp.searchsorted(grid, z, side="right")
    idx_hi = jnp.clip(idx_hi, 1, grid.shape[0] - 1)
    idx_lo = idx_hi - 1
    z_lo = grid[idx_lo]
    z_hi = grid[idx_hi]
    weight = jnp.clip((z - z_lo) / jnp.maximum(z_hi - z_lo, 1.0e-30), 0.0, 1.0)
    return matrices[idx_lo] * (1.0 - weight) + matrices[idx_hi] * weight


def _can_use_redshift_filter_projection(context: ModelContext, cfg) -> bool:
    """Return whether cached variable-redshift photometric matrices are valid.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    cfg : FitConfig
        Fit configuration (normally ``context.fit_config``).
    """
    return bool(
        cfg.likelihood.use_redshift_projection_cache
        and cfg.observation.fits_redshift
        and context.redshift_projection_cache_jax is not None
    )


def _project_redshift_luminosity_filters(context: ModelContext, rest_lum, redshift):
    """Project rest luminosity density through redshift-interpolated matrices.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    rest_lum : jax.Array
        Rest-frame L_lambda in W/Angstrom on ``context.rest_wave_jax``, shape (n_wave,).
    redshift : float or jax.Array
        Source redshift.
    """
    cache = context.redshift_projection_cache_jax
    matrix = _interp_redshift_projection_matrix(redshift, cache.redshift_grid, cache.filter_projection)
    return matrix @ rest_lum


def _local_line_gaussian_grid(line_wave, line_lumin, width_kms, *, n_local: int = 9):
    """Evaluate CIGALE-style line profiles on per-line local grids.

    Parameters
    ----------
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Per-line GRAHSP amplitudes in W/Angstrom (L_lambda-scaled), shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    n_local : int
        Number of local grid points per line (spanning +/-3 FWHM).
    """
    line_wave = jnp.asarray(line_wave, dtype=jnp.float64)
    line_lumin = jnp.asarray(line_lumin, dtype=jnp.float64)
    offsets = jnp.linspace(-3.0, 3.0, int(n_local), dtype=jnp.float64)
    fwhm_wave = jnp.maximum(line_wave * (width_kms * 1000.0) / 299792458.0, 1.0e-8)
    sigma = fwhm_wave / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    wave = line_wave[:, None] + offsets[None, :] * fwhm_wave[:, None]
    z = (wave - line_wave[:, None]) / jnp.maximum(sigma[:, None], 1.0e-12)
    norm = 5100.0 / jnp.sqrt(jnp.pi * sigma * sigma)
    lumin = line_lumin[:, None] * jnp.exp(-0.5 * z * z) * norm[:, None]
    return wave, lumin


def _local_flux_conserving_line_grid(line_wave, line_lumin, width_kms, *, n_local: int = 9):
    """Evaluate integrated-luminosity line profiles on per-line local grids.

    Parameters
    ----------
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Integrated line luminosities in W, shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    n_local : int
        Number of local grid points per line (spanning +/-3 FWHM).
    """
    line_wave = jnp.asarray(line_wave, dtype=jnp.float64)
    line_lumin = jnp.asarray(line_lumin, dtype=jnp.float64)
    offsets = jnp.linspace(-3.0, 3.0, int(n_local), dtype=jnp.float64)
    fwhm_wave = jnp.maximum(line_wave * jnp.asarray(width_kms, dtype=jnp.float64) / C_KMS, 1.0e-8)
    sigma = fwhm_wave / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    wave = jnp.maximum(line_wave[:, None] + offsets[None, :] * fwhm_wave[:, None], 1.0e-6)
    z = (wave - line_wave[:, None]) / jnp.maximum(sigma[:, None], 1.0e-12)
    norm = 1.0 / jnp.maximum(sigma * jnp.sqrt(2.0 * jnp.pi), 1.0e-30)
    lumin = line_lumin[:, None] * jnp.exp(-0.5 * z * z) * norm[:, None]
    return wave, lumin


def _igm_transmission_at_redshift(context: ModelContext, redshift):
    """Return the IGM transmission on the rest grid at a sampled redshift.

    Interpolates the preload table linearly in redshift when it exists, else evaluates
    the exact model.
    """
    table = context.igm_redshift_table_jax
    if table is None:
        return _igm_transmission(context.igm_cache_jax, redshift)
    n_z = table.table.shape[0]
    pos = jnp.clip((jnp.asarray(redshift, dtype=jnp.float64) - table.z0) / table.dz, 0.0, n_z - 1.0)
    idx = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, n_z - 2)
    frac = pos - idx
    values = table.table[idx] * (1.0 - frac) + table.table[idx + 1] * frac
    return jnp.ones_like(context.rest_wave_jax).at[table.columns].set(values)


def _packed_filter_transmission(curves, obs_wave):
    """Return every packed filter's transmission at ``obs_wave``, shape (n_filters,) + obs_wave.shape."""
    obs_wave = jnp.asarray(obs_wave, dtype=jnp.float64)
    if curves.uniform_transmission is None:
        return jax.vmap(lambda w, t: jnp.interp(obs_wave, w, t, left=0.0, right=0.0))(curves.wave, curves.transmission)
    n_grid = curves.uniform_transmission.shape[1]
    pos = (obs_wave.reshape(-1)[None, :] - curves.uniform_wave0[:, None]) / curves.uniform_dwave[:, None]
    idx = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, n_grid - 2)
    frac = pos - idx
    lo = jnp.take_along_axis(curves.uniform_transmission, idx, axis=1)
    hi = jnp.take_along_axis(curves.uniform_transmission, idx + 1, axis=1)
    inside = (pos >= 0.0) & (pos <= curves.uniform_last_index[:, None])
    trans = jnp.where(inside, lo + (hi - lo) * frac, 0.0)
    return trans.reshape((-1,) + obs_wave.shape)


def _project_local_grid_filters(context: ModelContext, obs_line_wave, flux_lambda):
    """Project local line grids (n_lines, n_local) of f_lambda through every packed filter, in mJy."""
    curves = context.packed_filter_curves_jax
    trans = _packed_filter_transmission(curves, obs_line_wave)
    numer = jnp.sum(jnp.trapezoid(trans * flux_lambda[None], obs_line_wave[None], axis=-1), axis=-1)
    f_lambda = numer / jnp.maximum(curves.denom, 1.0e-30)
    eff_wave = context.filter_effective_wavelength_jax
    return 1.0e-10 / 299792458.0 * 1.0e29 * eff_wave * eff_wave * f_lambda


def _project_local_line_filters(
    context: ModelContext,
    line_wave,
    line_lumin,
    width_kms,
    ebv_total,
    redshift,
    luminosity_distance_m,
    igm,
):
    """Project Gaussian line emission through filters using local line grids.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Per-line GRAHSP amplitudes in W/Angstrom (L_lambda-scaled), shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    ebv_total : float, jax.Array or DustScreen
        Total E(B-V) in magnitudes seen by the lines, or a ``DustScreen`` for the Noll
        host curve.
    redshift : float or jax.Array
        Source redshift.
    luminosity_distance_m : float or jax.Array
        Luminosity distance in meters.
    igm : jax.Array
        IGM transmission on ``context.rest_wave_jax``, shape (n_wave,).
    """
    line_lumin = jnp.asarray(line_lumin, dtype=jnp.float64)
    rest_line_wave, rest_lumin = _local_line_gaussian_grid(line_wave, line_lumin, width_kms)
    rest_line_wave = jnp.maximum(rest_line_wave, 1.0e-6)
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    luminosity_distance_m = jnp.asarray(luminosity_distance_m, dtype=jnp.float64)
    distance_scale = 4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1.0e-12) ** 2 * jnp.maximum(1.0 + redshift, 1.0e-8)
    obs_line_wave = rest_line_wave * (1.0 + redshift)
    igm = jnp.interp(rest_line_wave, context.rest_wave_jax, igm, left=0.0, right=0.0)
    attenuation_factor = 10 ** (_screen_magnitudes(ebv_total, rest_line_wave) / -2.5)
    flux_lambda = rest_lumin * attenuation_factor * igm / distance_scale
    return _project_local_grid_filters(context, obs_line_wave, flux_lambda)


def _project_integrated_local_line_filters(
    context: ModelContext,
    line_wave,
    line_lumin,
    width_kms,
    ebv_total,
    redshift,
    luminosity_distance_m,
    igm,
    *,
    spectral_state=None,
    spectral_component_cfg=None,
):
    """Project integrated line luminosities through filters on local grids.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Integrated line luminosities in W, shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    ebv_total : float, jax.Array or DustScreen
        Total E(B-V) in magnitudes seen by the lines, or a ``DustScreen`` for the Noll
        host curve.
    redshift : float or jax.Array
        Source redshift.
    luminosity_distance_m : float or jax.Array
        Luminosity distance in meters.
    igm : jax.Array
        IGM transmission on ``context.rest_wave_jax``, shape (n_wave,).
    """
    rest_line_wave, rest_lumin = _local_flux_conserving_line_grid(line_wave, line_lumin, width_kms)
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    luminosity_distance_m = jnp.asarray(luminosity_distance_m, dtype=jnp.float64)
    distance_scale = 4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1.0e-12) ** 2 * jnp.maximum(1.0 + redshift, 1.0e-8)
    obs_line_wave = rest_line_wave * (1.0 + redshift)
    igm_local = jnp.interp(rest_line_wave, context.rest_wave_jax, igm, left=0.0, right=0.0)
    attenuation_factor = 10 ** (_screen_magnitudes(ebv_total, rest_line_wave) / -2.5)
    flux_lambda = rest_lumin * attenuation_factor * igm_local / distance_scale
    if spectral_state is not None and spectral_component_cfg is not None:
        from .spectroscopy import render_joint_feature_state

        bal_only_cfg = replace(
            spectral_component_cfg,
            use_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            custom_line_components=(),
        )
        flux_lambda = flux_lambda * render_joint_feature_state(
            obs_line_wave,
            redshift,
            spectral_state,
            config=bal_only_cfg,
        )["bal_transmission"]
    return _project_local_grid_filters(context, obs_line_wave, flux_lambda)


def _local_integrated_line_obs_sed(
    context: ModelContext,
    line_wave,
    line_lumin,
    width_kms,
    ebv_total,
    redshift,
    luminosity_distance_m,
    igm,
    *,
    spectral_state=None,
    spectral_component_cfg=None,
):
    """Return observed-frame local-grid wavelengths and F_lambda for lines.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    line_lumin : jax.Array
        Integrated line luminosities in W, shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    ebv_total : float, jax.Array or DustScreen
        Total E(B-V) in magnitudes seen by the lines, or a ``DustScreen`` for the Noll
        host curve.
    redshift : float or jax.Array
        Source redshift.
    luminosity_distance_m : float or jax.Array
        Luminosity distance in meters.
    igm : jax.Array
        IGM transmission on ``context.rest_wave_jax``, shape (n_wave,).
    """
    rest_line_wave, rest_lumin = _local_flux_conserving_line_grid(line_wave, line_lumin, width_kms)
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    luminosity_distance_m = jnp.asarray(luminosity_distance_m, dtype=jnp.float64)
    distance_scale = 4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1.0e-12) ** 2 * jnp.maximum(1.0 + redshift, 1.0e-8)
    obs_line_wave = rest_line_wave * (1.0 + redshift)
    igm_local = jnp.interp(rest_line_wave, context.rest_wave_jax, igm, left=0.0, right=0.0)
    attenuation_factor = 10 ** (_screen_magnitudes(ebv_total, rest_line_wave) / -2.5)
    flux_lambda = rest_lumin * attenuation_factor * igm_local / distance_scale
    if spectral_state is not None and spectral_component_cfg is not None:
        from .spectroscopy import render_joint_feature_state

        bal_only_cfg = replace(
            spectral_component_cfg,
            use_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            custom_line_components=(),
        )
        flux_lambda = flux_lambda * render_joint_feature_state(
            obs_line_wave,
            redshift,
            spectral_state,
            config=bal_only_cfg,
        )["bal_transmission"]
    separator = jnp.full((obs_line_wave.shape[0], 1), jnp.nan, dtype=jnp.float64)
    obs_plot = jnp.concatenate([obs_line_wave, separator], axis=1)
    flux_plot = jnp.concatenate([flux_lambda, separator], axis=1)
    return jnp.ravel(obs_plot), jnp.ravel(flux_plot)


def _interp_fixed_local_line_terms(width_kms, cache):
    """Interpolate cached fixed-z local line projection terms in log width.

    Parameters
    ----------
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    cache : FixedLocalLineProjectionCacheJax
        Fixed-z local line terms tabulated over log width (AGN or nebular).
    """
    log_width = jnp.log(jnp.maximum(jnp.asarray(width_kms, dtype=jnp.float64), 1.0e-12))
    grid = cache.log_width_grid
    idx_hi = jnp.searchsorted(grid, log_width, side="right")
    idx_hi = jnp.clip(idx_hi, 1, grid.shape[0] - 1)
    idx_lo = idx_hi - 1
    w = jnp.clip((log_width - grid[idx_lo]) / jnp.maximum(grid[idx_hi] - grid[idx_lo], 1.0e-30), 0.0, 1.0)
    profile_norm = cache.profile_norm[idx_lo] * (1.0 - w) + cache.profile_norm[idx_hi] * w
    attenuation_curve = cache.attenuation_curve[idx_lo] * (1.0 - w) + cache.attenuation_curve[idx_hi] * w
    projection_weight = cache.projection_weight[idx_lo] * (1.0 - w) + cache.projection_weight[idx_hi] * w
    return profile_norm, attenuation_curve, projection_weight


def _interp_fixed_local_line_rest_wave(width_kms, cache):
    """Interpolate the cached local line rest wavelengths in log width (same weights as the terms)."""
    log_width = jnp.log(jnp.maximum(jnp.asarray(width_kms, dtype=jnp.float64), 1.0e-12))
    grid = cache.log_width_grid
    idx_hi = jnp.clip(jnp.searchsorted(grid, log_width, side="right"), 1, grid.shape[0] - 1)
    idx_lo = idx_hi - 1
    w = jnp.clip((log_width - grid[idx_lo]) / jnp.maximum(grid[idx_hi] - grid[idx_lo], 1.0e-30), 0.0, 1.0)
    return cache.rest_line_wave[idx_lo] * (1.0 - w) + cache.rest_line_wave[idx_hi] * w


def _project_fixed_cached_local_line_filters(context: ModelContext, line_lumin, width_kms, ebv_total):
    """Project local AGN lines with fixed-z cached filter overlap terms.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    line_lumin : jax.Array
        Per-line GRAHSP amplitudes in W/Angstrom (L_lambda-scaled), shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    ebv_total : float, jax.Array or DustScreen
        Total E(B-V) in magnitudes seen by the lines, or a ``DustScreen`` for the Noll
        host curve.
    """
    cache = context.fixed_local_line_projection_cache_jax
    profile_norm, attenuation_curve, projection_weight = _interp_fixed_local_line_terms(width_kms, cache)
    rest_line_wave = _interp_fixed_local_line_rest_wave(width_kms, cache) if isinstance(ebv_total, DustScreen) else None
    attenuation_factor = 10 ** (_screen_magnitudes(ebv_total, rest_line_wave, attenuation_curve) / -2.5)
    line_flux_density = jnp.asarray(line_lumin, dtype=jnp.float64)[:, None] * profile_norm * attenuation_factor
    return jnp.sum(projection_weight * line_flux_density[None, :, :], axis=(1, 2))


def _project_fixed_cached_local_nebular_line_filters(context: ModelContext, line_lumin, width_kms, ebv_total):
    """Project local nebular lines with fixed-z cached filter overlap terms.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    line_lumin : jax.Array
        Integrated line luminosities in W, shape (n_lines,).
    width_kms : float or jax.Array
        Line FWHM in km/s shared by all lines.
    ebv_total : float, jax.Array or DustScreen
        Total E(B-V) in magnitudes seen by the lines, or a ``DustScreen`` for the Noll
        host curve.
    """
    cache = context.fixed_local_nebular_line_projection_cache_jax
    profile_norm, attenuation_curve, projection_weight = _interp_fixed_local_line_terms(width_kms, cache)
    rest_line_wave = _interp_fixed_local_line_rest_wave(width_kms, cache) if isinstance(ebv_total, DustScreen) else None
    attenuation_factor = 10 ** (_screen_magnitudes(ebv_total, rest_line_wave, attenuation_curve) / -2.5)
    line_flux_density = jnp.asarray(line_lumin, dtype=jnp.float64)[:, None] * profile_norm * attenuation_factor
    return jnp.sum(projection_weight * line_flux_density[None, :, :], axis=(1, 2))


def _interp_dale_template(alpha, alpha_grid, dust_lumin_grid):
    """Interpolate the Dale 2014 host-dust grid in alpha.

    Parameters
    ----------
    alpha : float or jax.Array
        Dale (2014) alpha; clipped to the grid range.
    alpha_grid : jax.Array
        Tabulated alpha values, shape (n_alpha,).
    dust_lumin_grid : jax.Array
        Unit-luminosity dust templates, shape (n_alpha, n_wave).
    """
    alpha = jnp.clip(alpha, jnp.min(alpha_grid), jnp.max(alpha_grid))
    return jax.vmap(lambda row: jnp.interp(alpha, alpha_grid, row))(dust_lumin_grid.T)


def _host_dust_emission(context: ModelContext, dust_luminosity, dust_alpha):
    """Convert absorbed host luminosity into a Dale-template dust SED.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    dust_luminosity : float or jax.Array
        Absorbed host luminosity re-emitted by dust, in W.
    dust_alpha : float or jax.Array
        Dale (2014) alpha selecting the template.
    """
    dust_template = _interp_dale_template(dust_alpha, context.dust_alpha_grid_jax, context.dust_lumin_rest_jax)
    dust_rest_native = jnp.clip(dust_luminosity, 0.0, None) * jnp.clip(dust_template, 0.0, None)
    return dust_rest_native


def _host_dl07_emission(context: ModelContext, dust_luminosity, dust_umin):
    """Return fixed-Prospector-shape DL07 emission with free ``U_min`` only."""
    u_grid = context.dl07_umin_grid_jax
    q_grid = context.dl07_qpah_grid_jax
    u = jnp.clip(dust_umin, u_grid[0], u_grid[-1])
    q = jnp.clip(jnp.asarray(2.5, dtype=jnp.float64), q_grid[0], q_grid[-1])
    iu = jnp.clip(jnp.searchsorted(u_grid, u) - 1, 0, u_grid.size - 2)
    iq = jnp.clip(jnp.searchsorted(q_grid, q) - 1, 0, q_grid.size - 2)
    fu = (u - u_grid[iu]) / (u_grid[iu + 1] - u_grid[iu])
    fq = (q - q_grid[iq]) / (q_grid[iq + 1] - q_grid[iq])

    def bilinear(grid):
        return (
            (1.0 - fq) * (1.0 - fu) * grid[iq, iu]
            + (1.0 - fq) * fu * grid[iq, iu + 1]
            + fq * (1.0 - fu) * grid[iq + 1, iu]
            + fq * fu * grid[iq + 1, iu + 1]
        )

    # FSPS/Prospector DL07 constants: alpha=2, Umax=1e6, gamma=0.01,
    # qPAH=2.5%. Gamma is a dust-mass fraction, so restore the relative
    # luminosity per mass of the power-law-heated component (DL07 Eq. 33).
    umax = jnp.asarray(1.0e6, dtype=jnp.float64)
    gamma = jnp.asarray(0.01, dtype=jnp.float64)
    pdr_weight = umax * jnp.log(umax / u) / (umax - u)
    shape = (1.0 - gamma) * bilinear(context.dl07_single_u_rest_jax)
    shape = shape + gamma * pdr_weight * bilinear(context.dl07_powerlaw_rest_jax)
    c_angstrom_s = jnp.asarray(2.99792458e18, dtype=jnp.float64)
    wave = context.rest_wave_jax
    frequency = c_angstrom_s / wave
    shape_lnu = shape * wave**2 / c_angstrom_s
    norm = -jnp.trapezoid(shape_lnu, frequency)
    scaled_lnu = jnp.clip(dust_luminosity, 0.0, None) * shape_lnu / norm
    return jnp.where(
        norm > 0.0,
        jnp.clip(scaled_lnu * c_angstrom_s / wave**2, 0.0, None),
        jnp.zeros_like(shape),
    )


def _host_metallicity_parameters(
    context: ModelContext,
    prior_config: dict[str, Any],
    shared_gal_lgmet=None,
):
    """Return host mean metallicity and MDF scatter under the configured policy."""
    galaxy_cfg = context.fit_config.galaxy
    ssp_lgmet = context.host_basis_jax.ssp_lgmet
    if shared_gal_lgmet is not None:
        gal_lgmet = shared_gal_lgmet
    elif "gal_lgmet" in prior_config:
        gal_lgmet = _sample_bounded_normal(
            prior_config,
            "gal_lgmet",
            _absolute_z_to_gal_lgmet(
                galaxy_cfg.stellar_metallicity,
                metallicity_coordinate=galaxy_cfg.ssp_metallicity_coordinate,
                solar_metallicity=galaxy_cfg.ssp_solar_metallicity,
            ),
            0.5,
            jnp.min(ssp_lgmet),
            jnp.max(ssp_lgmet),
        )
    else:
        gal_lgmet = _absolute_z_to_gal_lgmet(
            galaxy_cfg.stellar_metallicity,
            metallicity_coordinate=galaxy_cfg.ssp_metallicity_coordinate,
            solar_metallicity=galaxy_cfg.ssp_solar_metallicity,
        )

    if "gal_lgmet_scatter" in prior_config or "log_gal_lgmet_scatter" in prior_config:
        gal_lgmet_scatter = _sample_positive_distribution(
            prior_config,
            value_key="gal_lgmet_scatter",
            log_key="log_gal_lgmet_scatter",
            default_value_distribution=dist.LogNormal(np.log(0.15), 0.8),
            default_log_distribution=dist.Normal(np.log(0.15), 0.8),
            default_to_log=True,
        )
    else:
        gal_lgmet_scatter = jnp.asarray(galaxy_cfg.stellar_metallicity_scatter, dtype=jnp.float64)
    return gal_lgmet, gal_lgmet_scatter


def _fixed_metallicity_host_basis(context: ModelContext, prior_config: dict[str, Any]):
    """Return the host SSP basis pre-summed over metallicity when no metallicity is sampled.

    With the configured (unsampled) metallicity and MDF scatter the metallicity weights are
    constants, so contracting the (n_met, n_age, n_wave) basis on every evaluation is wasted
    work. Returns the normalized-weight sum over metallicity, shape (n_age, n_wave), evaluated
    once at trace time, or ``None`` when any metallicity parameter is sampled.
    """
    sampled = {"gal_lgmet", "gal_lgmet_scatter", "log_gal_lgmet_scatter"}
    galaxy_cfg = context.fit_config.galaxy
    tied = bool(getattr(galaxy_cfg, "fit_host", True) and getattr(galaxy_cfg, "tie_stellar_nebular_metallicity", False))
    if tied:
        sampled.add("nebular_zgas")
    if any(key in prior_config for key in sampled):
        return None
    with jax.ensure_compile_time_eval():
        shared_gal_lgmet = _resolve_tied_metallicity(context, prior_config)[1] if tied else None
        gal_lgmet, gal_lgmet_scatter = _host_metallicity_parameters(context, prior_config, shared_gal_lgmet)
        lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
            gal_lgmet, gal_lgmet_scatter, context.host_basis_jax.ssp_lgmet
        )
        lgmet_weights = lgmet_weights / jnp.maximum(jnp.sum(lgmet_weights), 1.0e-30)
        return jnp.tensordot(lgmet_weights, context.host_basis_jax.rest_llambda, axes=(0, 0))


def _contract_host_basis(host_weights, age_weights, host_basis_rest, fixed_basis):
    """Return sum_{met, age} w * basis, using the metallicity-collapsed basis when available.

    ``host_weights`` is the normalized outer product of the metallicity and age weights, so
    with constant metallicity weights it equals the normalized age weights against the
    pre-summed basis.
    """
    if fixed_basis is None:
        return jnp.tensordot(host_weights, host_basis_rest, axes=((0, 1), (0, 1)))
    age_norm = age_weights / jnp.maximum(jnp.sum(age_weights), 1.0e-30)
    return age_norm @ fixed_basis


def _build_diffstar_host(
    context: ModelContext,
    prior_config: dict[str, Any],
    *,
    full_output: bool = True,
    shared_gal_lgmet=None,
    redshift=None,
):
    """Build the host-galaxy SED from Diffstar SFH and a precomputed SSP basis.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    full_output : bool
        If True, also return SSP weights and the scaled SFR/stellar-mass history tables.
    """
    ssp_lgmet = context.host_basis_jax.ssp_lgmet
    ssp_lg_age_gyr = context.host_basis_jax.ssp_lg_age_gyr
    host_basis_rest = context.host_basis_jax.rest_llambda
    surviving_frac_by_age = context.host_basis_jax.surviving_frac_by_age
    galaxy_cfg = context.fit_config.galaxy
    if redshift is None:
        redshift = jnp.asarray(context.fit_config.observation.redshift, dtype=jnp.float64)
    if bool(getattr(context.fit_config.observation, "fits_redshift", False)):
        t_obs_gyr = _flat_lcdm_age_gyr_jax(
            redshift,
            galaxy_cfg.cosmology_h0,
            galaxy_cfg.cosmology_om0,
        )
        gal_t_table = jnp.geomspace(
            jnp.asarray(galaxy_cfg.sfh_t_min_gyr, dtype=jnp.float64),
            jnp.maximum(t_obs_gyr, galaxy_cfg.sfh_t_min_gyr * 1.01),
            context.host_basis_jax.gal_t_table.size,
        )
    else:
        gal_t_table = context.host_basis_jax.gal_t_table
        t_obs_gyr = jnp.asarray(context.t_obs_gyr, dtype=jnp.float64)

    log_stellar_mass = _sample_log_stellar_mass(prior_config)

    u_params = {}
    for key in DEFAULT_DIFFSTAR_U_PARAMS._fields:
        default_loc = float(np.asarray(getattr(DEFAULT_DIFFSTAR_U_PARAMS, key)))
        u_params[key] = _sample_prior(prior_config, key, dist.Normal(default_loc, 1.0))
    bounded = get_bounded_diffstar_params(DiffstarUParams(**u_params))
    base_history = calc_sfh_singlegal(
        bounded,
        DEFAULT_MAH_PARAMS,
        gal_t_table,
        lgt0=DIFFSTAR_LGT0,
        fb=DIFFSTAR_FB,
        return_smh=True,
    )

    gal_lgmet, gal_lgmet_scatter = _host_metallicity_parameters(
        context,
        prior_config,
        shared_gal_lgmet,
    )
    age_weights, _ = _diffstar_ssp_age_weights(
        bounded,
        ssp_lg_age_gyr,
        t_obs_gyr,
        t_birth_min_gyr=gal_t_table[0],
    )
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        gal_lgmet, gal_lgmet_scatter, ssp_lgmet
    )
    host_weights = lgmet_weights[:, None] * age_weights[None, :]
    host_weights = host_weights / jnp.maximum(jnp.sum(host_weights), 1.0e-30)
    surviving_mass_fraction = jnp.clip(jnp.sum(age_weights * surviving_frac_by_age), 1e-12, 1.0)
    target_surviving_mass = 10.0**log_stellar_mass
    target_formed_mass = target_surviving_mass / surviving_mass_fraction
    # The original history remains the reported/scaled SFH diagnostic; only
    # its projection into SSP age bins uses the higher-accuracy quadrature.
    base_formed_mass = jnp.clip(base_history.smh[-1], 1e-30, 1.0e40)
    sfh_scale = target_formed_mass / base_formed_mass
    host_rest = target_formed_mass * _contract_host_basis(
        host_weights, age_weights, host_basis_rest, _fixed_metallicity_host_basis(context, prior_config)
    )
    state = {
        "host_rest": host_rest,
        "log_stellar_mass": log_stellar_mass,
        "surviving_mass_fraction": surviving_mass_fraction,
        "formed_mass": target_formed_mass,
        "sfh_scale": sfh_scale,
        "gal_lgmet": gal_lgmet,
        "gal_lgmet_scatter": gal_lgmet_scatter,
        "host_age_weights": age_weights,
        "host_lgmet_weights": lgmet_weights,
        "host_ssp_weights": host_weights,
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
        "sfh_age_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_tau_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "current_sfr": base_history.sfh[-1] * sfh_scale,
    }
    if not full_output:
        return state
    scaled_sfh = base_history.sfh * sfh_scale
    scaled_smh = base_history.smh * sfh_scale
    state.update(
        {
            "host_age_weights": age_weights,
            "host_lgmet_weights": lgmet_weights,
            "host_ssp_weights": host_weights,
            "gal_sfr_table": scaled_sfh,
            "gal_smh_table": scaled_smh,
        }
    )
    return state


def _delayed_ssfr_age_product(x):
    """sSFR x age for a delayed-tau SFH at ``x = age / tau`` (sSFR per formed mass).

    Equals ``x**2 exp(-x) / (1 - exp(-x) (1 + x))``; it decreases monotonically from 2
    (``x -> 0``, constant SFR) to 0 (``x -> infinity``, quenched).
    """
    x = jnp.maximum(jnp.asarray(x, dtype=jnp.float64), 1.0e-12)
    small = x < 1.0e-3
    xs = jnp.where(small, 1.0, x)
    formed = -jnp.expm1(-xs) - xs * jnp.exp(-xs)
    return jnp.where(small, 2.0 - 2.0 * x / 3.0, xs * xs * jnp.exp(-xs) / jnp.maximum(formed, 1.0e-300))


def _solve_delayed_x(target_product, n_bisect=60):
    """``x = age / tau`` with ``_delayed_ssfr_age_product(x) = target_product`` (0 < target < 2).

    Bisection in log x (without gradients) followed by one Newton step, so the result carries
    the exact implicit derivative with respect to ``target_product`` for gradient-based samplers.
    """
    target = jnp.clip(jnp.asarray(target_product, dtype=jnp.float64), 1.0e-300, 2.0 - 1.0e-9)
    t0 = jax.lax.stop_gradient(target)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        above = _delayed_ssfr_age_product(jnp.exp(mid)) > t0
        return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (jnp.log(1.0e-6), jnp.log(700.0)))
    x0 = jax.lax.stop_gradient(jnp.exp(0.5 * (lo + hi)))
    h0, dh = jax.jvp(_delayed_ssfr_age_product, (x0,), (jnp.ones_like(x0),))
    dh = jnp.where(jnp.abs(dh) > 1.0e-300, dh, -1.0e-300)
    return jnp.maximum(x0 - (h0 - target) / dh, 1.0e-6)


def _burst_ssfr_per_yr(age_gyr, burst_age_gyr, burst_tau_gyr):
    """Current SFR of the smooth exponential burst divided by the mass it formed (yr^-1)."""
    current = _smooth_exponential_burst_shape(age_gyr, age_gyr, burst_age_gyr, burst_tau_gyr)
    formed = _exponential_burst_cumulative_mass(age_gyr, age_gyr, burst_age_gyr, burst_tau_gyr)
    return current / jnp.maximum(formed * 1.0e9, 1.0e-300)


def _sample_delayed_burst_parameters(prior_config, age_gyr):
    """Sample the burst fraction, age and e-folding time of the delayed + burst SFH."""
    log_burst_fraction = numpyro.sample(
        "log_sfh_burst_fraction",
        _prior_distribution(
            prior_config,
            "log_sfh_burst_fraction",
            dist.Uniform(np.log(1.0e-4), np.log(0.2)),
        ),
    )
    burst_age_upper = jnp.minimum(age_gyr, 0.5)
    log_burst_age_gyr = numpyro.sample(
        "log_sfh_burst_age_gyr",
        _prior_distribution(
            prior_config,
            "log_sfh_burst_age_gyr",
            dist.Uniform(np.log(0.01), jnp.log(burst_age_upper)),
        ),
    )
    log_burst_tau_gyr = numpyro.sample(
        "log_sfh_burst_tau_gyr",
        _prior_distribution(
            prior_config,
            "log_sfh_burst_tau_gyr",
            dist.Uniform(np.log(0.01), np.log(0.2)),
        ),
    )
    return jnp.exp(log_burst_fraction), jnp.exp(log_burst_age_gyr), jnp.exp(log_burst_tau_gyr)


def _sample_ssfr_uniform_sfh(prior_config, cfg, physical_min_age, physical_max_age, use_burst):
    """Optional SFH prior uniform in log10 sSFR (``GalaxyConfig.sfh_prior = "ssfr_uniform"``).

    Samples ``log_sfh_ssfr`` (log10 of the current SFR per formed stellar mass, yr^-1) uniformly
    in ``GalaxyConfig.ssfr_prior_range`` (or a user prior under that name), then the SFH age
    log-uniformly between the default minimum age and the largest age that can still reach that
    sSFR (a delayed-tau SFH has sSFR x age < 2), then solves for tau. With the delayed + burst
    SFH the burst parameters are sampled as usual and tau is chosen so that the total sSFR,
    ``(1 - f_burst) sSFR_main + f_burst sSFR_burst``, equals the sampled value; when the burst
    alone already exceeds it (or the main component cannot reach it) the main sSFR is clipped
    to the feasible range, so the burst-mode prior is uniform only approximately there.
    Returns ``(log_age_gyr, log_tau_gyr, burst_params)``.
    """
    for key in ("log_sfh_age_gyr", "log_sfh_tau_gyr", "log_sfh_tau_over_age"):
        if key in prior_config:
            raise ValueError(
                f"galaxy.sfh_prior='ssfr_uniform' sets the {key} prior itself; "
                "set a prior on log_sfh_ssfr instead, or use sfh_prior='age_tau'."
            )
    default_max_age = jnp.minimum(10.0, physical_max_age)
    default_min_age = jnp.minimum(max(10.0**-0.8, physical_min_age), default_max_age / 1.01)
    lo, hi = (float(v) for v in cfg.ssfr_prior_range)
    # a delayed-tau SFH reaches at most sSFR = 2 / age; leave 5% margin at the minimum age
    hi_eff = jnp.minimum(hi, jnp.log10(1.9 / (default_min_age * 1.0e9)))
    lo_eff = jnp.minimum(lo, hi_eff - 1.0e-3)
    log_ssfr = numpyro.sample(
        "log_sfh_ssfr",
        _prior_distribution(prior_config, "log_sfh_ssfr", dist.Uniform(lo_eff, hi_eff)),
    )
    ssfr = 10.0**log_ssfr
    age_cap = jnp.clip(1.9 / (ssfr * 1.0e9), default_min_age * 1.0001, default_max_age)
    log_age_gyr = numpyro.sample(
        "log_sfh_age_gyr",
        dist.Uniform(jnp.log(default_min_age), jnp.log(age_cap)),
    )
    age_gyr = jnp.exp(log_age_gyr)
    burst = (None, None, None)
    ssfr_main = ssfr
    if use_burst:
        burst = _sample_delayed_burst_parameters(prior_config, age_gyr)
        burst_fraction, burst_age_gyr, burst_tau_gyr = burst
        ssfr_burst = _burst_ssfr_per_yr(age_gyr, burst_age_gyr, burst_tau_gyr)
        ssfr_main = (ssfr - burst_fraction * ssfr_burst) / jnp.maximum(1.0 - burst_fraction, 1.0e-8)
    product = jnp.clip(ssfr_main * age_gyr * 1.0e9, 1.0e-300, 1.999)
    x = _solve_delayed_x(product)
    log_tau_gyr = numpyro.deterministic("log_sfh_tau_gyr", log_age_gyr - jnp.log(x))
    return log_age_gyr, log_tau_gyr, burst


def _build_delayed_host(
    context: ModelContext,
    prior_config: dict[str, Any],
    *,
    full_output: bool = True,
    shared_gal_lgmet=None,
    redshift=None,
):
    """Build the host-galaxy SED from a CIGALE-like delayed-tau SFH.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    full_output : bool
        If True, also return SSP weights and the scaled SFR/stellar-mass history tables.
    """
    ssp_lgmet = context.host_basis_jax.ssp_lgmet
    ssp_lg_age_gyr = context.host_basis_jax.ssp_lg_age_gyr
    host_basis_rest = context.host_basis_jax.rest_llambda
    surviving_frac_by_age = context.host_basis_jax.surviving_frac_by_age
    cfg = context.fit_config.galaxy
    if redshift is None:
        redshift = jnp.asarray(context.fit_config.observation.redshift, dtype=jnp.float64)
    if bool(getattr(context.fit_config.observation, "fits_redshift", False)):
        t_obs_gyr = _flat_lcdm_age_gyr_jax(
            redshift,
            cfg.cosmology_h0,
            cfg.cosmology_om0,
        )
        gal_t_table = jnp.geomspace(
            jnp.asarray(cfg.sfh_t_min_gyr, dtype=jnp.float64),
            jnp.maximum(t_obs_gyr, cfg.sfh_t_min_gyr * 1.01),
            context.host_basis_jax.gal_t_table.size,
        )
    else:
        gal_t_table = context.host_basis_jax.gal_t_table
        t_obs_gyr = jnp.asarray(context.t_obs_gyr, dtype=jnp.float64)

    log_stellar_mass = _sample_log_stellar_mass(prior_config)
    physical_min_age = max(float(cfg.sfh_t_min_gyr), 1.0e-3)
    physical_max_age = jnp.maximum(t_obs_gyr, physical_min_age * 1.01)
    model_name = str(cfg.host_sfh_model).lower()
    use_burst = model_name in {"delayed_burst", "sfhdelayed_burst", "delayed-burst"}
    burst_params = None
    if str(getattr(cfg, "sfh_prior", "age_tau")).lower() == "ssfr_uniform":
        log_age_gyr, log_tau_gyr, burst_params = _sample_ssfr_uniform_sfh(
            prior_config, cfg, physical_min_age, physical_max_age, use_burst
        )
        log_tau_over_age = numpyro.deterministic("log_sfh_tau_over_age", log_tau_gyr - log_age_gyr)
    elif "log_sfh_age_gyr" in prior_config:
        log_age_gyr = _sample_bounded_normal(
            prior_config,
            "log_sfh_age_gyr",
            jnp.log(jnp.minimum(3.0, physical_max_age)),
            1.0,
            np.log(physical_min_age),
            jnp.log(physical_max_age),
        )
    else:
        # Continuous equivalent of the GRAHSP age_main grid: log-uniform
        # from 10^2.2 Myr to 10 Gyr, capped by the age of the Universe.
        default_max_age = jnp.minimum(10.0, physical_max_age)
        default_min_age = jnp.minimum(
            max(10.0**-0.8, physical_min_age),
            default_max_age / 1.01,
        )
        log_age_gyr = numpyro.sample(
            "log_sfh_age_gyr",
            dist.Uniform(jnp.log(default_min_age), jnp.log(default_max_age)),
        )
    age_gyr = jnp.exp(log_age_gyr)
    if burst_params is not None:
        pass  # sSFR-uniform prior: tau already solved from the sampled sSFR
    elif "log_sfh_tau_over_age" in prior_config:
        log_tau_over_age = _sample_bounded_normal(
            prior_config,
            "log_sfh_tau_over_age",
            0.0,
            float(cfg.tau_host_prior_scale),
            jnp.log(0.03 / age_gyr),
            jnp.log(30.0 / age_gyr),
        )
        log_tau_gyr = numpyro.deterministic("log_sfh_tau_gyr", log_age_gyr + log_tau_over_age)
    elif "log_sfh_tau_gyr" in prior_config:
        log_tau_gyr = _sample_bounded_normal(
            prior_config,
            "log_sfh_tau_gyr",
            np.log(1.0),
            float(cfg.tau_host_prior_scale),
            np.log(0.03),
            np.log(30.0),
        )
        log_tau_over_age = numpyro.deterministic("log_sfh_tau_over_age", log_tau_gyr - log_age_gyr)
    else:
        # Continuous equivalent of GRAHSP's independent tau_main grid.
        log_tau_gyr = numpyro.sample(
            "log_sfh_tau_gyr",
            dist.Uniform(np.log(0.1), np.log(10.0)),
        )
        log_tau_over_age = numpyro.deterministic("log_sfh_tau_over_age", log_tau_gyr - log_age_gyr)
    tau_gyr = jnp.exp(log_tau_gyr)
    stellar_age_gyr = jnp.maximum(t_obs_gyr - gal_t_table, 0.0)
    sfh_age_gyr = age_gyr - stellar_age_gyr
    if use_burst:
        if burst_params is not None and burst_params[0] is not None:
            burst_fraction, burst_age_gyr, burst_tau_gyr = burst_params
        else:
            burst_fraction, burst_age_gyr, burst_tau_gyr = _sample_delayed_burst_parameters(prior_config, age_gyr)
        base_sfh = _cigale_delayed_burst_sfh_shape(
            sfh_age_gyr,
            tau_gyr,
            age_gyr,
            burst_fraction,
            burst_age_gyr,
            burst_tau_gyr,
        )
    else:
        burst_fraction = jnp.asarray(0.0, dtype=jnp.float64)
        burst_age_gyr = jnp.asarray(0.0, dtype=jnp.float64)
        burst_tau_gyr = jnp.asarray(0.0, dtype=jnp.float64)
        base_sfh = _cigale_delayed_sfh_shape(sfh_age_gyr, tau_gyr, age_gyr)
    elapsed_history = jnp.clip(sfh_age_gyr, 0.0, age_gyr)
    main_cumulative = _delayed_sfh_cumulative_mass(elapsed_history, tau_gyr)
    main_total = _delayed_sfh_cumulative_mass(age_gyr, tau_gyr)
    if use_burst:
        burst_cumulative = _exponential_burst_cumulative_mass(
            elapsed_history, age_gyr, burst_age_gyr, burst_tau_gyr
        )
        burst_total = _exponential_burst_cumulative_mass(
            age_gyr, age_gyr, burst_age_gyr, burst_tau_gyr
        )
        burst_scale = (
            burst_fraction
            / jnp.maximum(1.0 - burst_fraction, 1.0e-8)
            * main_total
            / jnp.maximum(burst_total, 1.0e-30)
        )
        base_smh = (main_cumulative + burst_scale * burst_cumulative) * 1.0e9
        base_formed_mass = jnp.maximum(
            main_total / jnp.maximum(1.0 - burst_fraction, 1.0e-8) * 1.0e9,
            1.0e-30,
        )
    else:
        base_smh = main_cumulative * 1.0e9
        base_formed_mass = jnp.maximum(main_total * 1.0e9, 1.0e-30)

    gal_lgmet, gal_lgmet_scatter = _host_metallicity_parameters(
        context,
        prior_config,
        shared_gal_lgmet,
    )
    if use_burst:
        age_weights = _analytic_delayed_burst_age_weights(
            age_gyr,
            tau_gyr,
            burst_fraction,
            burst_age_gyr,
            burst_tau_gyr,
            ssp_lg_age_gyr,
        )
    else:
        age_weights = _analytic_delayed_age_weights(age_gyr, tau_gyr, ssp_lg_age_gyr)
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        gal_lgmet, gal_lgmet_scatter, ssp_lgmet
    )
    host_weights = lgmet_weights[:, None] * age_weights[None, :]
    host_weights = host_weights / jnp.maximum(jnp.sum(host_weights), 1.0e-30)
    surviving_mass_fraction = jnp.clip(jnp.sum(age_weights * surviving_frac_by_age), 1e-12, 1.0)
    target_surviving_mass = 10.0**log_stellar_mass
    target_formed_mass = target_surviving_mass / surviving_mass_fraction
    sfh_scale = target_formed_mass / base_formed_mass
    host_rest = target_formed_mass * _contract_host_basis(
        host_weights, age_weights, host_basis_rest, _fixed_metallicity_host_basis(context, prior_config)
    )
    state = {
        "host_rest": host_rest,
        "log_stellar_mass": log_stellar_mass,
        "surviving_mass_fraction": surviving_mass_fraction,
        "formed_mass": target_formed_mass,
        "sfh_scale": sfh_scale,
        "gal_lgmet": gal_lgmet,
        "gal_lgmet_scatter": gal_lgmet_scatter,
        "host_age_weights": age_weights,
        "host_lgmet_weights": lgmet_weights,
        "host_ssp_weights": host_weights,
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
        "sfh_age_gyr": age_gyr,
        "sfh_tau_gyr": tau_gyr,
        "sfh_burst_fraction": burst_fraction,
        "sfh_burst_age_gyr": burst_age_gyr,
        "sfh_burst_tau_gyr": burst_tau_gyr,
        "current_sfr": base_sfh[-1] * sfh_scale,
    }
    if not full_output:
        return state
    state.update(
        {
            "host_age_weights": age_weights,
            "host_lgmet_weights": lgmet_weights,
            "host_ssp_weights": host_weights,
            "gal_sfr_table": base_sfh * sfh_scale,
            "gal_smh_table": base_smh * sfh_scale,
        }
    )
    return state


def _build_host_state(
    context: ModelContext,
    prior_config: dict[str, Any],
    *,
    full_output: bool = True,
    shared_gal_lgmet=None,
    redshift=None,
):
    """Dispatch to the configured host SFH model.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    full_output : bool
        If True, also return SSP weights and the scaled SFR/stellar-mass history tables.
    """
    model_name = str(context.fit_config.galaxy.host_sfh_model).lower()
    if model_name in {
        "delayed",
        "sfhdelayed",
        "delayed_tau",
        "delayed-tau",
        "delayed_burst",
        "sfhdelayed_burst",
        "delayed-burst",
    }:
        return _build_delayed_host(
            context,
            prior_config,
            full_output=full_output,
            shared_gal_lgmet=shared_gal_lgmet,
            redshift=redshift,
        )
    if model_name in {"diffstar", "dsps_diffstar"}:
        return _build_diffstar_host(
            context,
            prior_config,
            full_output=full_output,
            shared_gal_lgmet=shared_gal_lgmet,
            redshift=redshift,
        )
    raise ValueError("galaxy.host_sfh_model must be one of: 'delayed', 'delayed_burst', 'diffstar'.")


def _host_rest_on_basis(host_state: dict[str, Any], host_basis_jax):
    """Evaluate the sampled SSP mixture on an alternate wavelength basis.

    Parameters
    ----------
    host_state : dict[str, Any]
        Host state with ``formed_mass`` and ``host_ssp_weights`` (n_met, n_age).
    host_basis_jax : HostBasisJax
        SSP basis whose ``rest_llambda`` (n_met, n_age, n_wave) is in W/Angstrom per
        Msun.
    """
    return host_state["formed_mass"] * jnp.tensordot(
        host_state["host_ssp_weights"],
        host_basis_jax.rest_llambda,
        axes=((0, 1), (0, 1)),
    )


def _empty_host_state(context: ModelContext):
    """Return zero-valued host placeholders for AGN-only fits.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    """
    rest_wave = _np_to_jnp(context.rest_wave)
    ssp_lgmet = _np_to_jnp(context.ssp_data.ssp_lgmet)
    ssp_lg_age_gyr = _np_to_jnp(context.ssp_data.ssp_lg_age_gyr)
    gal_t_table = _np_to_jnp(context.gal_t_table)
    zero_host_weights = jnp.zeros((ssp_lgmet.shape[0], ssp_lg_age_gyr.shape[0]), dtype=jnp.float64)
    return {
        "host_rest": jnp.zeros_like(rest_wave),
        "host_age_weights": jnp.zeros_like(ssp_lg_age_gyr),
        "host_lgmet_weights": jnp.zeros_like(ssp_lgmet),
        "host_ssp_weights": zero_host_weights,
        "surviving_mass_fraction": jnp.asarray(0.0, dtype=jnp.float64),
        "formed_mass": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_scale": jnp.asarray(0.0, dtype=jnp.float64),
        "gal_lgmet": jnp.asarray(0.0, dtype=jnp.float64),
        "gal_lgmet_scatter": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_age_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_tau_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "gal_sfr_table": jnp.zeros_like(gal_t_table),
        "gal_smh_table": jnp.zeros_like(gal_t_table),
        "current_sfr": jnp.asarray(0.0, dtype=jnp.float64),
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
    }


def _build_nebular_components(
    context: ModelContext,
    host_state: dict[str, Any],
    host_rest,
    prior_config: dict[str, Any],
    *,
    build_line_sed: bool = True,
    shared_zgas=None,
):
    """Build CIGALE/GRAHSP-style host nebular absorption, lines, and continuum.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    host_state : dict[str, Any]
        Host state from ``_build_host_state`` (formed mass and SSP weights).
    host_rest : jax.Array
        Intrinsic stellar host L_lambda in W/Angstrom, shape (n_wave,).
    prior_config : dict[str, Any]
        Flat prior mapping keyed by NumPyro site name.
    """
    rest_wave = context.rest_wave_jax
    cfg = context.fit_config.nebular
    zeros = jnp.zeros_like(rest_wave)
    if not (context.fit_config.galaxy.fit_host and cfg.enabled):
        return {
            "absorption_rest": zeros,
            "lines_rest": zeros,
            "continuum_rest": zeros,
            "emission_rest": zeros,
            "dust_luminosity": jnp.asarray(0.0, dtype=jnp.float64),
            "n_ly_young": jnp.asarray(0.0, dtype=jnp.float64),
            "n_ly_old": jnp.asarray(0.0, dtype=jnp.float64),
            "ly_lum_young": jnp.asarray(0.0, dtype=jnp.float64),
            "ly_lum_old": jnp.asarray(0.0, dtype=jnp.float64),
            "logU": jnp.asarray(float(cfg.logU), dtype=jnp.float64),
            "zgas": (
                jnp.asarray(shared_zgas, dtype=jnp.float64)
                if shared_zgas is not None
                else jnp.asarray(
                    float(cfg.zgas)
                    if cfg.zgas is not None
                    else float(context.fit_config.galaxy.stellar_metallicity),
                    dtype=jnp.float64,
                )
            ),
            "ne": jnp.asarray(float(cfg.ne), dtype=jnp.float64),
            "f_esc": jnp.asarray(float(cfg.f_esc), dtype=jnp.float64),
            "f_dust": jnp.asarray(float(cfg.f_dust), dtype=jnp.float64),
            "lines_width": jnp.asarray(float(cfg.lines_width), dtype=jnp.float64),
            "line_scale": jnp.asarray(1.0, dtype=jnp.float64),
            "corr": jnp.asarray(0.0, dtype=jnp.float64),
            "line_lumin": jnp.zeros((1,), dtype=jnp.float64),
        }

    templates = context.nebular_templates_jax
    logu = _sample_optional_truncnorm(
        prior_config,
        "nebular_logU",
        float(cfg.logU),
        0.3,
        templates.logu_grid[0],
        templates.logu_grid[-1],
    )
    if shared_zgas is not None:
        zgas = shared_zgas
    else:
        default_zgas = float(cfg.zgas) if cfg.zgas is not None else None
        galaxy_cfg = context.fit_config.galaxy
        host_zgas = _gal_lgmet_to_absolute_z(
            host_state["gal_lgmet"],
            metallicity_coordinate=galaxy_cfg.ssp_metallicity_coordinate,
            solar_metallicity=galaxy_cfg.ssp_solar_metallicity,
        )
        zgas_default = host_zgas if default_zgas is None else jnp.asarray(default_zgas, dtype=jnp.float64)
        zgas = _sample_optional_truncnorm(
            prior_config,
            "nebular_zgas",
            float(default_zgas) if default_zgas is not None else float(context.fit_config.galaxy.stellar_metallicity),
            0.01,
            templates.z_grid[0],
            templates.z_grid[-1],
        )
        zgas = jnp.where(default_zgas is None and "nebular_zgas" not in prior_config, zgas_default, zgas)
    ne = _sample_optional_truncnorm(
        prior_config,
        "nebular_ne",
        float(cfg.ne),
        100.0,
        templates.ne_grid[0],
        templates.ne_grid[-1],
    )
    f_esc = _sample_optional_truncnorm(prior_config, "nebular_f_esc", float(cfg.f_esc), 0.1, 0.0, 1.0)
    default_remaining = max(1.0 - float(cfg.f_esc), 1.0e-12)
    default_f_dust_fraction = float(np.clip(float(cfg.f_dust) / default_remaining, 0.0, 1.0))
    f_dust_fraction = _sample_optional_truncnorm(
        prior_config,
        "nebular_f_dust_fraction",
        default_f_dust_fraction,
        0.1,
        0.0,
        1.0,
    )
    f_dust = jnp.maximum(1.0 - f_esc, 0.0) * f_dust_fraction
    lines_width = _sample_optional_truncnorm(
        prior_config,
        "nebular_lines_width",
        float(cfg.lines_width),
        100.0,
        1.0,
        1.0e5,
    )
    if "log_nebular_line_scale" in prior_config:
        line_scale = _sample_log_positive_from_distribution(
            prior_config,
            value_key="nebular_line_scale",
            log_key="log_nebular_line_scale",
            default_distribution=dist.Normal(0.0, 0.5),
        )
    else:
        line_scale = jnp.asarray(1.0, dtype=jnp.float64)

    weights = host_state["formed_mass"] * host_state["host_ssp_weights"]
    young_mask = (jnp.power(10.0, context.host_basis_jax.ssp_lg_age_gyr) * 1.0e3) <= float(cfg.young_age_cut_myr)
    young_mask_2d = young_mask[None, :]
    old_mask_2d = ~young_mask_2d
    n_basis = context.host_basis_jax.n_ly_per_msun
    l_basis = context.host_basis_jax.ly_lum_per_msun
    n_ly_young = jnp.sum(weights * n_basis * young_mask_2d)
    n_ly_old = jnp.sum(weights * n_basis * old_mask_2d)
    ly_lum_young = jnp.sum(weights * l_basis * young_mask_2d)
    ly_lum_old = jnp.sum(weights * l_basis * old_mask_2d)
    n_ly_total = jnp.clip(n_ly_young + n_ly_old, 0.0, 1.0e300)
    ly_lum_total = jnp.clip(ly_lum_young + ly_lum_old, 0.0, 1.0e300)

    corr = _cigale_nebular_correction(f_esc, f_dust)
    rest_templates = context.nebular_rest_templates_jax
    continuum_per_photon = _trilinear_nebular_grid(
        rest_templates.continuum_lumin_per_a_per_photon,
        templates.z_grid,
        templates.logu_grid,
        templates.ne_grid,
        zgas,
        logu,
        ne,
    )
    line_lumin_per_photon = _trilinear_nebular_grid(
        templates.line_lumin_per_photon,
        templates.z_grid,
        templates.logu_grid,
        templates.ne_grid,
        zgas,
        logu,
        ne,
    )
    continuum_rest = continuum_per_photon * n_ly_total * corr
    line_lumin = line_lumin_per_photon * n_ly_total * corr
    if not build_line_sed:
        lines_rest = zeros
    elif context.fixed_nebular_line_profile_jax is not None:
        lines_rest = context.fixed_nebular_line_profile_jax * n_ly_total * corr
    elif rest_templates.line_profile_per_photon is not None:
        line_profile_per_photon = _trilinear_nebular_grid(
            rest_templates.line_profile_per_photon,
            templates.z_grid,
            templates.logu_grid,
            templates.ne_grid,
            zgas,
            logu,
            ne,
        )
        lines_rest = line_profile_per_photon * n_ly_total * corr
    else:
        lines_rest = _flux_conserving_line_gaussians(rest_wave, templates.line_wave_a, line_lumin, lines_width)
    emission_scale = jnp.asarray(1.0 if cfg.emission else 0.0, dtype=jnp.float64)
    line_lumin = line_lumin * emission_scale * line_scale
    lines_rest = lines_rest * emission_scale * line_scale
    continuum_rest = continuum_rest * emission_scale
    absorption_rest = jnp.where(rest_wave < 912.0, -host_rest * (1.0 - f_esc), 0.0)
    return {
        "absorption_rest": absorption_rest,
        "lines_rest": lines_rest,
        "continuum_rest": continuum_rest,
        "emission_rest": lines_rest + continuum_rest,
        "dust_luminosity": ly_lum_total * f_dust,
        "n_ly_young": n_ly_young,
        "n_ly_old": n_ly_old,
        "ly_lum_young": ly_lum_young,
        "ly_lum_old": ly_lum_old,
        "logU": logu,
        "zgas": zgas,
        "ne": ne,
        "f_esc": f_esc,
        "f_dust": f_dust,
        "f_dust_fraction": f_dust_fraction,
        "lines_width": lines_width,
        "line_scale": line_scale,
        "corr": corr,
        "line_lumin": line_lumin,
    }


def _estimate_log_agn_amp_prior_loc(context: ModelContext, redshift: float) -> float:
    """Estimate a rough log(lambda L_lambda, 5100 A) prior center from the photometry.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    redshift : float
        Source redshift used to pick the band nearest rest 5100 Angstrom.
    """
    obs_fluxes_mjy = np.asarray(context.fluxes, dtype=float)
    mask = np.asarray(context.positive_detected_mask, dtype=bool) & np.isfinite(obs_fluxes_mjy) & (obs_fluxes_mjy > 0.0)
    if not np.any(mask):
        return float(np.log(1.0e36))
    filter_wavelength = np.array([f.effective_wavelength for f in context.filters], dtype=float)
    target_obs_wave = 5100.0 * (1.0 + max(float(redshift), 0.0))
    valid_indices = np.flatnonzero(mask)
    best_index = valid_indices[int(np.argmin(np.abs(filter_wavelength[valid_indices] - target_obs_wave)))]
    flux_w_m2_hz = obs_fluxes_mjy[best_index] * 1.0e-29
    nu_obs_hz = C_MS / max(filter_wavelength[best_index] * 1.0e-10, 1.0e-30)
    agn_amp_w = 4.0 * np.pi * float(context.luminosity_distance_m) ** 2 * nu_obs_hz * flux_w_m2_hz
    return float(np.log(np.clip(agn_amp_w, 1.0e30, 1.0e50)))


def _default_log_agn_amp_prior(context: ModelContext, redshift: float):
    """Return the default AGN luminosity prior.

    The photometry-seeded luminosity is an upper-envelope estimate because the
    filter flux includes host light. Center the default below that value so
    fit_agn=True allows weak/no AGN solutions instead of expecting the AGN to
    explain the full optical continuum.
    """
    photometry_seeded_loc = _estimate_log_agn_amp_prior_loc(context, redshift)
    return dist.Normal(photometry_seeded_loc - 4.0, 3.0)


def _sample_redshift(context: ModelContext, prior_config: dict[str, Any], cfg) -> jnp.ndarray:
    """Sample redshift from either the legacy Gaussian prior or a tabulated p(z).

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    prior_config : dict[str, Any]
        Flat prior mapping; an optional ``redshift_pdf`` entry supplies a tabulated
        p(z).
    cfg : FitConfig
        Fit configuration (normally ``context.fit_config``).
    """
    redshift_pdf = prior_config.get("redshift_pdf")
    if redshift_pdf is None:
        return numpyro.sample(
            "redshift",
            dist.TruncatedNormal(
                cfg.observation.redshift,
                max(cfg.observation.redshift_err, 1e-3),
                low=1.0e-8,
            ),
        )

    z_grid = np.asarray(redshift_pdf["z_grid"], dtype=float)
    pdf = np.asarray(redshift_pdf["pdf"], dtype=float)
    pdf_norm = pdf / max(float(np.trapezoid(pdf, z_grid)), 1.0e-300)
    z_grid_jnp = _np_to_jnp(z_grid)
    pdf_jnp = _np_to_jnp(pdf_norm)
    redshift = numpyro.sample(
        "redshift",
        dist.Uniform(
            low=float(z_grid[0]),
            high=float(z_grid[-1]),
        ),
    )
    pz_val = jnp.interp(redshift, z_grid_jnp, pdf_jnp, left=0.0, right=0.0)
    numpyro.factor("redshift_pdf_prior", jnp.log(jnp.clip(pz_val, 1.0e-300, None)))
    return redshift


def _chi2_upper_limit(obs_fluxes, model_fluxes, total_variance):
    """Return the one-sided chi-square contribution for upper limits.

    Parameters
    ----------
    obs_fluxes : jax.Array
        Upper-limit flux values in mJy.
    model_fluxes : jax.Array
        Model fluxes in mJy.
    total_variance : jax.Array
        Total variance in mJy^2.
    """
    z = (obs_fluxes - model_fluxes) / jnp.sqrt(2.0 * jnp.maximum(total_variance, 1e-60))
    return -2.0 * jnp.log(0.5 * (1.0 + jax.scipy.special.erf(z)) + 1e-300)


def _agn_variability_nev(agn_bol_lum_w, max_nev):
    """Return the Simm+2016-inspired fractional variability variance cap.

    Parameters
    ----------
    agn_bol_lum_w : float or jax.Array
        AGN bolometric luminosity in W.
    max_nev : float or jax.Array
        Upper cap on the normalized excess variance.
    """
    agn_bol_lum_w = jnp.maximum(jnp.asarray(agn_bol_lum_w, dtype=jnp.float64), 1.0e-30)
    max_nev = jnp.maximum(jnp.asarray(max_nev, dtype=jnp.float64), 1.0e-6)
    log_lbol_erg_s = jnp.log10(agn_bol_lum_w * ERG_PER_WATT)
    l45 = log_lbol_erg_s - 45.0
    simm_nev = 10.0 ** (-1.43 - 0.74 * l45)
    # Smooth generalized minimum: close to the smaller input without the kink
    # in the likelihood geometry at simm_nev == max_nev.
    smoothness = jnp.asarray(8.0, dtype=jnp.float64)
    return jnp.exp(
        -jnp.logaddexp(
            -smoothness * jnp.log(max_nev),
            -smoothness * jnp.log(simm_nev),
        )
        / smoothness
    )


def _host_capture_fraction(effective_radius_arcsec, host_size_arcsec):
    """Map an effective measurement radius to the captured host-light fraction.

    The fixed-slope relation ``r_eff**2 / (r_eff**2 + r_host**2)`` uses one
    inferred host-size parameter. Aperture radii and Gaussian-PSF effective
    radii are converted to this common coordinate while building the context.
    Invalid radii represent measurements without usable spatial metadata and
    therefore default to total capture.

    Parameters
    ----------
    effective_radius_arcsec : object
        Effective measurement radius in arcseconds.
    host_size_arcsec : object
        Host turnover radius in arcseconds (the 50-percent capture radius).
    """
    effective_radius_arcsec = jnp.asarray(effective_radius_arcsec, dtype=jnp.float64)
    valid = jnp.isfinite(effective_radius_arcsec) & (effective_radius_arcsec > 0.0)
    host_size_arcsec = jnp.maximum(jnp.asarray(host_size_arcsec, dtype=jnp.float64), 1.0e-3)
    safe_radius = jnp.where(valid, jnp.clip(effective_radius_arcsec, 1.0e-3, 1.0e6), host_size_arcsec)
    frac = safe_radius**2 / (safe_radius**2 + host_size_arcsec**2)
    return jnp.where(valid, frac, 1.0)


def photometric_loglike(
    pred_fluxes,
    obs_fluxes,
    obs_errors,
    upper_limits,
    data_mask,
    systematics_width,
    likelihood_family,
    student_t_df,
    agn_component,
    agn_bol_lum_w,
    agn_nev,
    variability_uncertainty,
    attenuation_model_uncertainty,
    transmitted_fraction,
    lyman_break_uncertainty,
    filter_wavelength,
    redshift,
    nebular_line_component=None,
    local_nebular_line_uncertainty_dex=0.0,
    agn_systematics_width=0.0,
    return_diagnostics=False,
):
    """Evaluate the broadband photometric log-likelihood for one model state.

    Parameters
    ----------
    pred_fluxes : jax.Array
        Model band fluxes in mJy, shape (n_filters,).
    obs_fluxes : jax.Array
        Observed fluxes (or upper-limit values) in mJy, shape (n_filters,).
    obs_errors : jax.Array
        1-sigma flux uncertainties in mJy, shape (n_filters,).
    upper_limits : jax.Array
        Boolean mask of censored (upper-limit) bands, shape (n_filters,).
    data_mask : jax.Array
        Boolean mask of detected bands used in the likelihood, shape (n_filters,).
    systematics_width : float or jax.Array
        Fractional systematic error, scaled by the observed flux.
    likelihood_family : str
        Sampling distribution: ``"gaussian"`` or ``"student_t"``.
    student_t_df : float or jax.Array
        Student-t degrees of freedom.
    agn_component : jax.Array
        AGN model band fluxes in mJy for variability/AGN systematics, shape
        (n_filters,).
    agn_bol_lum_w : float or jax.Array
        AGN bolometric luminosity in W.
    agn_nev : float or jax.Array
        Cap on the AGN normalized excess variance.
    variability_uncertainty : bool
        Add the AGN variability variance term.
    attenuation_model_uncertainty : bool
        Add the attenuation-model variance term.
    transmitted_fraction : jax.Array
        Bandwise attenuated-to-intrinsic flux ratio, shape (n_filters,).
    lyman_break_uncertainty : bool
        Inflate bands with rest effective wavelength < 1500 Angstrom.
    filter_wavelength : jax.Array
        Observed-frame filter effective wavelengths in Angstrom, shape (n_filters,).
    redshift : float or jax.Array
        Source redshift.
    nebular_line_component : jax.Array or None
        Local nebular-line band fluxes in mJy, or None to skip that term.
    local_nebular_line_uncertainty_dex : float
        Fractional nebular-line flux uncertainty in dex.
    agn_systematics_width : object
        Fractional AGN model-mismatch coefficient in the GRAHSP systematic
        error term.
    return_diagnostics : bool, optional
        If True, also return standardized-residual chi-square diagnostics.
        These diagnostics exclude censored upper limits and use the effective
        variance constructed by this likelihood.
    """
    pred_fluxes = jnp.asarray(pred_fluxes, dtype=jnp.float64)
    obs_fluxes = jnp.asarray(obs_fluxes, dtype=jnp.float64)
    obs_errors = jnp.asarray(obs_errors, dtype=jnp.float64)
    agn_component = jnp.asarray(agn_component, dtype=jnp.float64)
    transmitted_fraction = jnp.asarray(transmitted_fraction, dtype=jnp.float64)
    active = jnp.asarray(data_mask, dtype=bool) | jnp.asarray(upper_limits, dtype=bool)
    input_valid = (
        jnp.isfinite(pred_fluxes)
        & jnp.isfinite(obs_fluxes)
        & jnp.isfinite(obs_errors)
        & (obs_errors > 0.0)
    )
    auxiliary_valid = jnp.ones_like(input_valid)
    if variability_uncertainty:
        auxiliary_valid = auxiliary_valid & jnp.isfinite(agn_component)
    if attenuation_model_uncertainty:
        auxiliary_valid = auxiliary_valid & jnp.isfinite(transmitted_fraction)
    if nebular_line_component is not None:
        auxiliary_valid = auxiliary_valid & jnp.isfinite(jnp.asarray(nebular_line_component, dtype=jnp.float64))

    pred_fluxes = jnp.nan_to_num(pred_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    obs_fluxes = jnp.nan_to_num(obs_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    obs_errors = jnp.nan_to_num(obs_errors, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    agn_component = jnp.nan_to_num(agn_component, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    transmitted_fraction = jnp.nan_to_num(transmitted_fraction, nan=1.0e-4, posinf=1.0, neginf=1.0e-4)
    obs_variance = obs_errors**2
    variability_nev = _agn_variability_nev(agn_bol_lum_w, agn_nev)
    var_variance = jnp.where(variability_uncertainty, variability_nev * agn_component**2, 0.0)
    # Model-uncertainty terms scale with the model flux, as in GRAHSP
    # (dualsampler.chi2_with_norm: sys_variance = (sys_error * model_fluxes)**2).
    # The AGN-template mismatch is an independent contribution.
    sys_variance = (systematics_width * pred_fluxes) ** 2 + (agn_systematics_width * agn_component) ** 2
    if nebular_line_component is not None:
        nebular_line_component = jnp.nan_to_num(nebular_line_component, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
        nebular_line_sigma = jnp.expm1(
            jnp.log(10.0) * jnp.maximum(jnp.asarray(local_nebular_line_uncertainty_dex, dtype=jnp.float64), 0.0)
        )
        sys_variance = sys_variance + (nebular_line_sigma * nebular_line_component) ** 2
    if attenuation_model_uncertainty:
        tf = jnp.clip(transmitted_fraction, 1e-4, 1.0)
        neg_log = -jnp.log10(tf + 1e-4)
        log_unc_frac = jnp.minimum(-4.0 + 2.0 * neg_log, -1.0)
        att_unc = 10 ** log_unc_frac / tf
        sys_variance = sys_variance + (att_unc * pred_fluxes) ** 2
    if lyman_break_uncertainty:
        ly_unc = jnp.where(filter_wavelength / (1.0 + redshift) < 1500.0, 1.0e8, 0.0)
        sys_variance = sys_variance + (ly_unc * pred_fluxes) ** 2
    raw_total_variance = obs_variance + sys_variance + var_variance
    variance_valid = jnp.isfinite(raw_total_variance) & (raw_total_variance > 0.0)
    total_variance = jnp.nan_to_num(raw_total_variance, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    scale = jnp.sqrt(jnp.clip(total_variance, 1e-30, 1.0e60))
    family = str(likelihood_family).lower()
    if family in {"gaussian", "normal"}:
        data_dist = dist.Normal(loc=pred_fluxes, scale=scale)
    elif family in {"student_t", "student-t", "studentt", "t"}:
        data_dist = dist.StudentT(df=student_t_df, loc=pred_fluxes, scale=scale)
    else:
        raise ValueError("likelihood.likelihood_family must be one of: 'gaussian', 'student_t'.")
    # Mask distribution inputs before evaluating them. JAX differentiates
    # both branches of ``where``; masking only the returned log probabilities
    # allows an unstable inactive tail (notably Student-t CDF gradients) to
    # poison the full gradient with NaNs.
    detection_obs = jnp.where(data_mask, obs_fluxes, pred_fluxes)
    logl_data = jnp.sum(jnp.where(data_mask, data_dist.log_prob(detection_obs), 0.0))
    # Censored measurements must use the same sampling family as detections.
    # In particular, the default Student-t likelihood has substantially
    # heavier upper-tail probability than a Gaussian at fixed scale.
    # Do not let detection-only entries enter the CDF computational graph.
    # StudentT.cdf can have a non-finite derivative in its extreme tails, and
    # masking the resulting log-CDF afterward is too late for reverse-mode AD.
    # Use a completely parameter-independent standard distribution at inactive
    # entries; active upper limits retain the configured distribution exactly.
    cdf_loc = jnp.where(upper_limits, pred_fluxes, 0.0)
    cdf_scale = jnp.where(upper_limits, scale, 1.0)
    # Evaluate inactive entries at +1 standard scale rather than exactly at
    # the location. This keeps the placeholder finite and away from the
    # Student-t CDF's problematic zero-offset derivative.
    cdf_fluxes = jnp.where(upper_limits, obs_fluxes, 1.0)
    if family in {"gaussian", "normal"}:
        limit_dist = dist.Normal(loc=cdf_loc, scale=cdf_scale)
    else:
        limit_dist = dist.StudentT(df=student_t_df, loc=cdf_loc, scale=cdf_scale)
    log_cdf = jnp.log(jnp.clip(limit_dist.cdf(cdf_fluxes), 1.0e-300, 1.0))
    logl_lim = jnp.sum(jnp.where(upper_limits, log_cdf, 0.0))
    invalid = active & ~(input_valid & auxiliary_valid & variance_valid)
    penalty = -1.0e6 * jnp.sum(invalid.astype(jnp.float64))
    loglike = logl_data + logl_lim + penalty
    if not return_diagnostics:
        return loglike

    chi2_mask = jnp.asarray(data_mask, dtype=bool) & input_valid & auxiliary_valid & variance_valid
    n_eff = jnp.sum(chi2_mask.astype(jnp.float64))
    chi2_sum = jnp.sum(
        jnp.where(
            chi2_mask,
            (obs_fluxes - pred_fluxes) ** 2 / jnp.maximum(total_variance, 1.0e-60),
            0.0,
        )
    )
    chi2 = jnp.where(n_eff > 0.0, chi2_sum, jnp.asarray(jnp.nan, dtype=jnp.float64))
    reduced_chi2 = jnp.where(
        n_eff > 0.0,
        chi2_sum / jnp.maximum(n_eff, 1.0),
        jnp.asarray(jnp.nan, dtype=jnp.float64),
    )
    return loglike, {
        "chi2": chi2,
        "n_eff": n_eff,
        "reduced_chi2": reduced_chi2,
    }


def spectroscopic_loglike(
    pred_fluxes,
    obs_fluxes,
    obs_errors,
    mask,
    systematics_width,
    student_t_df,
    return_diagnostics=False,
):
    """Evaluate the observed-frame spectral log-likelihood.

    Parameters
    ----------
    pred_fluxes : jax.Array
        Model spectral fluxes in mJy, shape (n_pix,).
    obs_fluxes : jax.Array
        Observed spectral fluxes in mJy, shape (n_pix,).
    obs_errors : jax.Array
        1-sigma spectral uncertainties in mJy, shape (n_pix,).
    mask : jax.Array
        Boolean mask of pixels used in the likelihood, shape (n_pix,).
    systematics_width : float or jax.Array
        Fractional systematic error, scaled by the model flux.
    student_t_df : float or jax.Array
        Student-t degrees of freedom.
    return_diagnostics : bool, optional
        If True, also return standardized-residual chi-square diagnostics
        using the same effective variance as the spectral likelihood.
    """
    pred_fluxes = jnp.asarray(pred_fluxes, dtype=jnp.float64)
    obs_fluxes = jnp.asarray(obs_fluxes, dtype=jnp.float64)
    obs_errors = jnp.asarray(obs_errors, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    input_valid = (
        jnp.isfinite(pred_fluxes)
        & jnp.isfinite(obs_fluxes)
        & jnp.isfinite(obs_errors)
        & (obs_errors > 0.0)
    )
    pred_fluxes = jnp.nan_to_num(pred_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    obs_fluxes = jnp.nan_to_num(obs_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    obs_errors = jnp.nan_to_num(obs_errors, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    variance = obs_errors**2 + (jnp.maximum(systematics_width, 0.0) * pred_fluxes) ** 2
    variance_valid = jnp.isfinite(variance) & (variance > 0.0)
    scale = jnp.sqrt(jnp.clip(variance, 1e-30, 1.0e60))
    student = dist.StudentT(df=student_t_df, loc=pred_fluxes, scale=scale)
    valid = mask & input_valid & variance_valid & jnp.isfinite(scale)
    penalty = -1.0e6 * jnp.sum((mask & ~valid).astype(jnp.float64))
    loglike = jnp.sum(jnp.where(valid, student.log_prob(obs_fluxes), 0.0)) + penalty
    if not return_diagnostics:
        return loglike

    n_eff = jnp.sum(valid.astype(jnp.float64))
    chi2_sum = jnp.sum(
        jnp.where(
            valid,
            (obs_fluxes - pred_fluxes) ** 2 / jnp.maximum(variance, 1.0e-60),
            0.0,
        )
    )
    chi2 = jnp.where(n_eff > 0.0, chi2_sum, jnp.asarray(jnp.nan, dtype=jnp.float64))
    reduced_chi2 = jnp.where(
        n_eff > 0.0,
        chi2_sum / jnp.maximum(n_eff, 1.0),
        jnp.asarray(jnp.nan, dtype=jnp.float64),
    )
    return loglike, {
        "chi2": chi2,
        "n_eff": n_eff,
        "reduced_chi2": reduced_chi2,
    }


def _spectrum_continuum_log_pivot(
    continuum_fluxes,
    observed_fluxes,
    mask,
    spectrum_index,
    n_spectra,
):
    """Return a smooth per-spectrum log continuum-to-data RMS ratio.

    The ratio is evaluated before the instrumental spectrum scale is applied.
    Reparameterizing ``log_spectrum_scale`` by this offset makes the scaled
    continuum RMS a direct NUTS coordinate while preserving the original prior
    and likelihood exactly. Squared broad averages avoid medians, extrema,
    parameter-dependent masks, and positivity clips.
    """
    continuum_fluxes = jnp.nan_to_num(
        jnp.asarray(continuum_fluxes, dtype=jnp.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    observed_fluxes = jnp.nan_to_num(
        jnp.asarray(observed_fluxes, dtype=jnp.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mask = jnp.asarray(mask, dtype=bool)
    spectrum_index = jnp.asarray(spectrum_index, dtype=jnp.int32)
    offsets = []
    for index in range(int(n_spectra)):
        use = mask & (spectrum_index == index)
        count = jnp.maximum(jnp.sum(use), 1)
        model_mean_square = jnp.sum(
            jnp.where(use, continuum_fluxes**2, 0.0)
        ) / count
        observed_mean_square = jnp.sum(
            jnp.where(use, observed_fluxes**2, 0.0)
        ) / count
        # This data-scaled floor only handles an all-zero edge case; in an
        # ordinary fit it is many orders of magnitude below either RMS.
        floor_square = jnp.maximum(observed_mean_square * 1.0e-24, 1.0e-60)
        offsets.append(
            0.5
            * (
                jnp.log(model_mean_square + floor_square)
                - jnp.log(observed_mean_square + floor_square)
            )
        )
    result = jnp.stack(offsets)
    return result[0] if int(n_spectra) == 1 else result


def _flambda_to_mjy(wave_obs, flux_lambda):
    """Convert internal f_lambda values on an observed wavelength grid to mJy.

    Parameters
    ----------
    wave_obs : jax.Array
        Observed-frame wavelengths in Angstrom.
    flux_lambda : jax.Array
        f_lambda in W m^-2 Angstrom^-1 on ``wave_obs``.
    """
    return 1.0e-10 / 299792458.0 * 1.0e29 * wave_obs * wave_obs * flux_lambda


def _fixed_spectral_line_coverage_rest(context: ModelContext, cfg: FitConfig) -> tuple[float, float] | None:
    """Return fixed-redshift rest coverage for the shared tied-line table.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    cfg : FitConfig
        Fit configuration (normally ``context.fit_config``).
    """
    if cfg.observation.fits_redshift:
        return None
    spec_wave = np.asarray(context.spec_wave_obs, dtype=float)
    spec_mask = np.asarray(context.spec_mask, dtype=bool)
    valid = spec_mask & np.isfinite(spec_wave) & (spec_wave > 0.0)
    if not np.any(valid):
        return None
    redshift = max(float(cfg.observation.redshift), 0.0)
    margin = max(float(cfg.agn.line_coverage_margin_kms), 0.0) / C_KMS
    rest_min = float(np.min(spec_wave[valid]) / (1.0 + redshift))
    rest_max = float(np.max(spec_wave[valid]) / (1.0 + redshift))
    return (rest_min * (1.0 - margin), rest_max * (1.0 + margin))


def _photometric_agn_line_mask(context: ModelContext, cfg: FitConfig, line_wave, redshift):
    """Mask native AGN SED lines to those not covered by spectroscopy.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    cfg : FitConfig
        Fit configuration (normally ``context.fit_config``).
    line_wave : jax.Array
        Rest-frame line centre wavelengths in Angstrom, shape (n_lines,).
    redshift : float or jax.Array
        Source redshift.
    """
    spectral_cfg = cfg.agn
    if (
        not bool(cfg.spectroscopy_list)
        or not bool(spectral_cfg.fit_lines)
        or len(context.spec_wave_obs) == 0
        or not np.any(context.spec_mask)
    ):
        return jnp.ones_like(line_wave)

    spec_wave = jnp.asarray(context.spec_wave_obs, dtype=jnp.float64)
    spec_mask = jnp.asarray(context.spec_mask, dtype=bool)
    valid = spec_mask & jnp.isfinite(spec_wave) & (spec_wave > 0.0)
    spec_min = jnp.min(jnp.where(valid, spec_wave, jnp.inf))
    spec_max = jnp.max(jnp.where(valid, spec_wave, -jnp.inf))
    margin = jnp.asarray(max(float(spectral_cfg.line_coverage_margin_kms), 0.0) / C_KMS, dtype=jnp.float64)
    line_obs = jnp.asarray(line_wave, dtype=jnp.float64) * jnp.maximum(1.0 + redshift, 1.0e-8)
    coverage_min = spec_min * (1.0 - margin)
    coverage_max = spec_max * (1.0 + margin)
    if not cfg.observation.fits_redshift:
        covered = (line_obs >= coverage_min) & (line_obs <= coverage_max)
        return jnp.where(covered, 0.0, 1.0)

    transition = jnp.maximum(margin * 0.5 * (spec_min + spec_max) / 6.0, 1.0e-3)
    covered_weight = jax.nn.sigmoid((line_obs - coverage_min) / transition) * jax.nn.sigmoid(
        (coverage_max - line_obs) / transition
    )
    return 1.0 - covered_weight


def _spectral_instrumental_sigma_ln(context):
    """Return each spectral pixel's instrumental sigma in ln(wavelength), or None.

    Built from the per-spectrum ``SpectroscopyData.resolving_power``; pixels of
    spectra without a resolving power get zero (no convolution).
    """
    resolving_power = np.asarray(context.spec_resolving_power, dtype=float)
    spectrum_index = np.asarray(context.spec_spectrum_index, dtype=int)
    if resolving_power.size == 0 or spectrum_index.size == 0:
        return None
    pixel_r = resolving_power[np.clip(spectrum_index, 0, resolving_power.size - 1)]
    valid = np.isfinite(pixel_r) & (pixel_r > 0.0)
    if not np.any(valid):
        return None
    return np.where(valid, 1.0 / (2.354820045 * np.where(valid, pixel_r, 1.0)), 0.0)


def _broaden_spectral_host(context, spec_rest_wave, spec_host_intrinsic, gal_v_kms, gal_sigma_kms):
    """Apply stellar kinematics, instrument, and SSP template resolution to the spectral host.

    Each spectrum with a resolving power gets its own kernel from
    ``host_kernel_sigma_kms``; spectra without one keep the fitted dispersion as an
    effective broadening. ``gal_sigma_kms`` itself stays the intrinsic dispersion.
    """
    from .spectral_model import host_kernel_sigma_kms

    lnwave = jnp.log(spec_rest_wave)
    sigma_inst_ln = _spectral_instrumental_sigma_ln(context)
    if sigma_inst_ln is None:
        return _shift_and_broaden_single_spectrum_lnlam(lnwave, spec_host_intrinsic, gal_v_kms, gal_sigma_kms)
    template_fwhm = context.fit_config.galaxy.ssp_template_fwhm_angstrom
    spectrum_index = np.asarray(context.spec_spectrum_index, dtype=int)
    rest_wave_np = np.asarray(spec_rest_wave, dtype=float)
    broadened = jnp.zeros_like(spec_host_intrinsic)
    for index in np.unique(spectrum_index):
        pixels = spectrum_index == index
        sigma_inst_kms = C_KMS * float(sigma_inst_ln[pixels][0])
        kernel_sigma = host_kernel_sigma_kms(
            gal_sigma_kms, sigma_inst_kms, template_fwhm, float(np.median(rest_wave_np[pixels]))
        )
        broadened = jnp.where(
            jnp.asarray(pixels),
            _shift_and_broaden_single_spectrum_lnlam(lnwave, spec_host_intrinsic, gal_v_kms, kernel_sigma),
            broadened,
        )
    return broadened


def _evaluate_spectral_components(
    wave_obs,
    redshift,
    continuum_mjy,
    cfg,
    line_prior_config,
    rest_wave,
    feii_template_flux,
    line_coverage_rest=None,
    fixed_narrow_fwhm_kms=None,
    fixed_narrow_amp_scale=None,
    feature_amplitude_scale=1.0,
    include_lines=True,
    instrumental_sigma_ln=None,
    bal_continuum_mjy=None,
    include_bal_components=True,
):
    """Evaluate the built-in detailed spectral components.

    Parameters
    ----------
    wave_obs : jax.Array
        Observed-frame spectral pixel wavelengths in Angstrom, shape (n_pix,).
    redshift : float or jax.Array
        Source redshift.
    continuum_mjy : jax.Array
        Model continuum at the spectral pixels in mJy, shape (n_pix,).
    cfg : FitConfig
        Fit configuration (normally ``context.fit_config``).
    line_prior_config : dict[str, Any]
        Prior overrides for the spectral line/feature sites.
    rest_wave : jax.Array
        Native Fe II template rest-frame wavelengths in Angstrom.
    feii_template_flux : jax.Array
        Native Fe II template luminosity on ``rest_wave``.
    line_coverage_rest : tuple[float, float] or None
        Rest-frame wavelength range (Angstrom) covered by spectra, or None.
    fixed_narrow_fwhm_kms : float or None
        If set, fixed FWHM in km/s for all narrow lines.
    fixed_narrow_amp_scale : float or None
        If set, multiplicative scale applied to narrow-line amplitudes.
    feature_amplitude_scale : object
        Calibration scale defining the observed-spectrum coordinate system for
        sampled line, Fe II, and Balmer amplitudes.
    instrumental_sigma_ln : object
        Optional static per-pixel instrumental sigma in ln(wavelength).
    bal_continuum_mjy : object, optional
        Compact AGN disk continuum subject to BAL absorption.
    include_bal_components : bool, optional
        Whether BAL absorption custom components are active. Other custom
        continuum and line components are unaffected.
    """
    try:
        from .spectral_custom_components import is_bal_absorption_component
        from .spectroscopy import (
            SpectralComponentConfig,
            evaluate_joint_spectral_components,
        )
    except Exception as exc:  # pragma: no cover - exercised only without optional dependency
        raise ImportError(
            "Unable to load the built-in detailed spectral backend."
        ) from exc

    spectral_cfg = cfg.agn

    def _active_components(components):
        components = tuple(components or ())
        if include_bal_components:
            return components
        return tuple(
            component
            for component in components
            if not is_bal_absorption_component(component)
        )

    component_cfg = SpectralComponentConfig(
        use_lines=bool(spectral_cfg.fit_lines and include_lines),
        tied_lines=bool(spectral_cfg.tied_lines),
        use_feii=bool(spectral_cfg.fit_feii),
        use_balmer_continuum=bool(spectral_cfg.fit_balmer_continuum),
        multiplicative_tilt=bool(spectral_cfg.fit_spectrum_tilt),
        line_table=spectral_cfg.line_table,
        line_prior_config=line_prior_config,
        line_flux_scale_mjy=float(spectral_cfg.line_flux_scale_mjy),
        line_coverage_rest=line_coverage_rest,
        include_elg_narrow_lines=bool(spectral_cfg.include_elg_narrow_lines),
        include_high_ionization_lines=bool(spectral_cfg.include_high_ionization_lines),
        broad_fwhm_kms_default=DEFAULT_BROAD_LINE_WIDTH_KMS,
        feii_fwhm_kms_default=DEFAULT_BROAD_LINE_WIDTH_KMS,
        feii_fnu_pivot_rest=4575.0,
        balmer_velocity_kms_default=DEFAULT_BROAD_LINE_WIDTH_KMS,
        broadening_convolution=spectral_cfg.broadening_convolution,
        fixed_narrow_fwhm_kms=fixed_narrow_fwhm_kms,
        fixed_narrow_amp_scale=fixed_narrow_amp_scale,
        custom_components=_active_components(spectral_cfg.custom_components),
        custom_line_components=_active_components(spectral_cfg.custom_line_components),
        components=_active_components(getattr(spectral_cfg, "components", ())),
    )
    result = evaluate_joint_spectral_components(
        wave_obs,
        redshift,
        continuum_mjy,
        config=component_cfg,
        feii_template_wave_rest=rest_wave,
        feii_template_flux=feii_template_flux,
        site_prefix="spectral",
        feature_amplitude_scale=feature_amplitude_scale,
        instrumental_sigma_ln=instrumental_sigma_ln,
        bal_continuum_mjy=bal_continuum_mjy,
    )
    result["component_config"] = component_cfg
    return result


def _simple_feii_fnu_shape(wave_obs, redshift, template_wave_rest, template_flux, pivot_rest=4575.0):
    """Return the unbroadened SED Fe II template as a fixed-pivot f-nu shape."""
    wave_rest = jnp.asarray(wave_obs, dtype=jnp.float64) / jnp.maximum(1.0 + redshift, 1.0e-8)
    flambda_shape = jnp.interp(
        wave_rest,
        jnp.asarray(template_wave_rest, dtype=jnp.float64),
        jnp.asarray(template_flux, dtype=jnp.float64),
        left=0.0,
        right=0.0,
    )
    return flambda_shape * (wave_rest / float(pivot_rest)) ** 2


def _project_spectral_smooth_state_filters(
    context,
    state,
    redshift,
    component_cfg,
    feii_template_wave,
    feii_template_flux,
    coverage_rest,
    sed_feii_scale,
):
    """Project smooth spectral features through filters from one shared log grid.

    Fe II broadening and Balmer-edge smoothing are the expensive operations.
    Rendering them independently on every packed filter curve creates a large
    batched FFT graph.  A single log-wavelength rendering is sufficiently fine
    for these broad components; its result is interpolated onto the native
    filter curves before exact filter quadrature.
    """
    from .spectroscopy import render_joint_feature_state

    curves = context.packed_filter_curves_jax
    valid_filter_waves = [
        np.asarray(filt.wave, dtype=float)
        for filt in context.filters
        if np.asarray(filt.wave).size > 0
    ]
    if not valid_filter_waves:
        empty = jnp.zeros_like(context.filter_effective_wavelength_jax)
        return empty, empty, empty
    wave_min = min(float(np.min(wave)) for wave in valid_filter_waves)
    wave_max = max(float(np.max(wave)) for wave in valid_filter_waves)
    n_grid = int(context.fit_config.agn.feature_grid_size)
    feature_wave = jnp.geomspace(wave_min, wave_max, n_grid)
    # Lines have their own analytic, flux-conserving filter projection. Avoid
    # constructing their many-component profiles in this smooth-feature pass.
    smooth_component_cfg = replace(component_cfg, use_lines=False)
    rendered = render_joint_feature_state(
        feature_wave,
        redshift,
        state,
        config=smooth_component_cfg,
        feii_template_wave_rest=feii_template_wave,
        feii_template_flux=feii_template_flux,
    )
    sed_feii = sed_feii_scale * _simple_feii_fnu_shape(
        feature_wave, redshift, feii_template_wave, feii_template_flux
    )
    if coverage_rest is None and context.fit_config.observation.fits_redshift:
        spec_wave = jnp.asarray(context.spec_wave_obs, dtype=jnp.float64)
        spec_mask = jnp.asarray(context.spec_mask, dtype=bool)
        valid = spec_mask & jnp.isfinite(spec_wave) & (spec_wave > 0.0)
        spec_min = jnp.min(jnp.where(valid, spec_wave, jnp.inf))
        spec_max = jnp.max(jnp.where(valid, spec_wave, -jnp.inf))
        margin = max(
            float(context.fit_config.agn.line_coverage_margin_kms),
            0.0,
        ) / C_KMS
        coverage_min = spec_min * (1.0 - margin)
        coverage_max = spec_max * (1.0 + margin)
        transition = jnp.maximum(margin * 0.5 * (spec_min + spec_max) / 6.0, 1.0e-3)
        covered = jax.nn.sigmoid((feature_wave - coverage_min) / transition) * jax.nn.sigmoid(
            (coverage_max - feature_wave) / transition
        )
    elif coverage_rest is None:
        covered = jnp.ones_like(feature_wave)
    else:
        feature_wave_rest = feature_wave / jnp.maximum(1.0 + redshift, 1.0e-8)
        covered = (
            (feature_wave_rest >= coverage_rest[0])
            & (feature_wave_rest <= coverage_rest[1])
        ).astype(jnp.float64)
    feii_grid = covered * rendered["feii"]
    extrapolated_feii_grid = (1.0 - covered) * sed_feii * rendered["bal_transmission"]

    def _interpolate_one(wave):
        return (
            jnp.interp(wave, feature_wave, feii_grid, left=0.0, right=0.0),
            jnp.interp(wave, feature_wave, extrapolated_feii_grid, left=0.0, right=0.0),
            jnp.interp(wave, feature_wave, rendered["balmer"], left=0.0, right=0.0),
        )

    feii_fnu, extrapolated_feii_fnu, balmer_fnu = jax.vmap(_interpolate_one)(curves.wave)
    return (
        _project_fnu_mjy_filter_curves(context, curves.wave, feii_fnu),
        _project_fnu_mjy_filter_curves(context, curves.wave, extrapolated_feii_fnu),
        _project_fnu_mjy_filter_curves(context, curves.wave, balmer_fnu),
    )


def _project_fnu_mjy_filter_curves(context: ModelContext, wave_obs, fnu_mjy):
    """Project f-nu samples on packed native filter curves into band fluxes."""
    curves = context.packed_filter_curves_jax
    wave_obs = jnp.asarray(wave_obs, dtype=jnp.float64)
    fnu_mjy = jnp.asarray(fnu_mjy, dtype=jnp.float64)
    conversion = 1.0e-10 / 299792458.0 * 1.0e29
    flambda = fnu_mjy / jnp.maximum(conversion * wave_obs * wave_obs, 1.0e-30)
    weighted = jnp.where(curves.valid_mask, flambda * curves.transmission, 0.0)
    mean_flambda = jnp.trapezoid(weighted, wave_obs, axis=1) / jnp.maximum(curves.denom, 1.0e-30)
    return conversion * context.filter_effective_wavelength_jax**2 * mean_flambda


def _project_spectral_custom_state_filters(
    context,
    state,
    redshift,
    component_cfg,
    *,
    bal_continuum_mjy=None,
):
    """Render sampled custom components on native filter grids and integrate them."""
    from .spectroscopy import render_joint_feature_state

    curves = context.packed_filter_curves_jax
    rendered = render_joint_feature_state(
        curves.wave,
        redshift,
        state,
        config=replace(component_cfg, use_lines=False, use_feii=False, use_balmer_continuum=False),
        bal_continuum_mjy=bal_continuum_mjy,
    )
    custom = rendered["custom"]
    zero = jnp.zeros_like(curves.wave)
    continuum = rendered["custom_continuum"]
    broad = sum(
        (
            custom[component.output_name]
            for component in component_cfg.custom_line_components
            if component.line_kind == "broad"
        ),
        zero,
    )
    narrow = sum(
        (
            custom[component.output_name]
            for component in component_cfg.custom_line_components
            if component.line_kind == "narrow"
        ),
        zero,
    )
    return tuple(
        _project_fnu_mjy_filter_curves(context, curves.wave, value)
        for value in (continuum, broad, narrow)
    )


def _project_spectral_line_state_filters(
    context: ModelContext,
    state,
    redshift,
    broad_only=None,
    component_cfg=None,
):
    """Project sampled log-wavelength Gaussians using flux-conserving local grids."""
    amps = jnp.asarray(state.get("line_amp_per_component", jnp.zeros(0)), dtype=jnp.float64)
    n_filters = context.packed_filter_curves_jax.wave.shape[0]
    if not amps.size:
        return jnp.zeros((n_filters,), dtype=jnp.float64)
    mus = jnp.asarray(state["line_mu_per_component"], dtype=jnp.float64)
    sigs = jnp.asarray(state["line_sig_per_component"], dtype=jnp.float64)
    broad_mask = jnp.asarray(state["line_broad_mask_per_component"], dtype=jnp.float64)
    if broad_only is True:
        amps = amps * broad_mask
    elif broad_only is False:
        amps = amps * (1.0 - broad_mask)
    nodes = jnp.linspace(-7.0, 7.0, 33, dtype=jnp.float64)
    rest_wave = jnp.exp(mus[:, None] + sigs[:, None] * nodes[None, :])
    obs_line_wave = rest_wave * (1.0 + redshift)
    fnu = amps[:, None] * jnp.exp(-0.5 * nodes[None, :] ** 2)
    if component_cfg is not None and broad_only is not False:
        from .spectroscopy import render_joint_feature_state

        bal_only_cfg = replace(
            component_cfg,
            use_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            custom_line_components=(),
        )
        bal_transmission = render_joint_feature_state(
            obs_line_wave,
            redshift,
            state,
            config=bal_only_cfg,
        )["bal_transmission"]
        if broad_only is True:
            fnu = fnu * bal_transmission
        else:
            fnu = fnu * (
                broad_mask[:, None] * bal_transmission
                + (1.0 - broad_mask[:, None])
            )
    conversion = 1.0e-10 / 299792458.0 * 1.0e29
    flambda = fnu / jnp.maximum(conversion * obs_line_wave**2, 1.0e-30)
    return _project_local_grid_filters(context, obs_line_wave, flambda)


def _spectral_family_extrapolation(
    context: ModelContext,
    cfg: FitConfig,
    state,
    redshift,
    luminosity_distance_m,
    ebv_total,
    igm,
    line_wave,
    line_blagn,
    line_narrow_template,
):
    """Anchor fixed-ratio out-of-coverage lines to fitted spectral family fluxes."""
    amps = jnp.asarray(state.get("line_amp_per_component", jnp.zeros(0)), dtype=jnp.float64)
    if not amps.size:
        zeros = jnp.zeros_like(line_wave)
        return zeros, zeros, jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS), jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS)
    mus = jnp.asarray(state["line_mu_per_component"], dtype=jnp.float64)
    sigs = jnp.asarray(state["line_sig_per_component"], dtype=jnp.float64)
    broad_mask = jnp.asarray(state["line_broad_mask_per_component"], dtype=jnp.float64)
    component_wave = jnp.exp(mus)
    component_obs_wave = component_wave * (1.0 + redshift)
    conversion = 1.0e-10 / 299792458.0 * 1.0e29
    integrated_flux = amps * jnp.sqrt(2.0 * jnp.pi) * sigs / jnp.maximum(conversion * component_obs_wave, 1.0e-30)
    attenuation = 10 ** (_screen_magnitudes(ebv_total, component_wave) / -2.5)
    intrinsic_lumin = integrated_flux * 4.0 * jnp.pi * luminosity_distance_m**2 / jnp.maximum(attenuation, 1.0e-12)
    outside_mask = _photometric_agn_line_mask(context, cfg, line_wave, redshift)
    covered_mask = 1.0 - outside_mask
    broad_template_anchor = jnp.sum(line_blagn * covered_mask)
    narrow_template_anchor = jnp.sum(line_narrow_template * covered_mask)
    broad_norm = jnp.where(
        broad_template_anchor > 0.0,
        jnp.sum(intrinsic_lumin * broad_mask) / jnp.maximum(broad_template_anchor, 1.0e-30),
        0.0,
    )
    narrow_norm = jnp.where(
        narrow_template_anchor > 0.0,
        jnp.sum(intrinsic_lumin * (1.0 - broad_mask)) / jnp.maximum(narrow_template_anchor, 1.0e-30),
        0.0,
    )
    broad_lumin = broad_norm * line_blagn * outside_mask
    narrow_lumin = narrow_norm * line_narrow_template * outside_mask
    positive = jnp.clip(intrinsic_lumin, 0.0, None)
    broad_weight = positive * broad_mask
    narrow_weight = positive * (1.0 - broad_mask)
    component_fwhm = 299792.458 * 2.354820045 * sigs
    broad_width = jnp.exp(jnp.sum(broad_weight * jnp.log(jnp.maximum(component_fwhm, 1.0))) / jnp.maximum(jnp.sum(broad_weight), 1.0e-30))
    narrow_width = jnp.exp(jnp.sum(narrow_weight * jnp.log(jnp.maximum(component_fwhm, 1.0))) / jnp.maximum(jnp.sum(narrow_weight), 1.0e-30))
    broad_width = jnp.where(jnp.sum(broad_weight) > 0.0, broad_width, DEFAULT_BROAD_LINE_WIDTH_KMS)
    narrow_width = jnp.where(jnp.sum(narrow_weight) > 0.0, narrow_width, DEFAULT_NARROW_LINE_WIDTH_KMS)
    return broad_lumin, narrow_lumin, broad_width, narrow_width


def evaluate_photometric_state(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
    add_likelihood: bool = True,
    return_state: bool = True,
    force_component_fluxes: bool = False,
    include_spectral_lines: bool = True,
    include_spectral_bal: bool = True,
):
    """Evaluate one jaxsedfit photometric model state inside a NumPyro trace.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    include_components : bool
        If True, compute per-component observed SEDs and band fluxes for diagnostics.
    include_sed_agn_features : bool
        Whether the native SED AGN features (emission lines, Fe II, Balmer continuum)
        are included.
    include_spectral_features : bool
        Whether detailed spectral-engine features (Fe II, Balmer continuum, custom
        components) are evaluated.
    include_spectral_lines : object
        Whether detailed broad and narrow emission lines are active. Smooth
        spectral features such as Fe II and Balmer continuum remain controlled
        by ``include_spectral_features``.
    include_spectral_bal : bool
        Whether BAL absorption custom components are active. Ordinary custom
        components remain controlled by ``include_spectral_features``.
    add_likelihood : bool
        If True, add the photometric and spectroscopic log-likelihood factors to the
        trace.
    return_state : bool
        If True, return the dictionary of model products; otherwise return None.
    force_component_fluxes : bool
        Compute component band fluxes even when the likelihood does not need them.
    """
    cfg = context.fit_config
    prior_config = cfg.prior_config.to_mapping()
    rest_wave = context.rest_wave_jax
    obs_wave = context.obs_wave_jax
    feii_template_on_rest = context.feii_template_on_rest_jax
    feii_template_wave_native = _np_to_jnp(context.templates.feii_wave)
    feii_template_flux_native = _np_to_jnp(context.templates.feii_lumin)
    line_wave = _np_to_jnp(context.templates.line_wave)
    line_blagn = _np_to_jnp(context.templates.line_blagn)
    line_sy2 = _np_to_jnp(context.templates.line_sy2)
    line_liner = _np_to_jnp(context.templates.line_liner)
    filter_wavelength = context.filter_effective_wavelength_jax
    obs_fluxes = _np_to_jnp(context.fluxes)
    obs_errors = _np_to_jnp(context.errors)
    upper_limits = _bool_to_jnp(context.upper_limits)
    data_mask = _bool_to_jnp(context.data_mask)
    spec_wave_obs = _np_to_jnp(context.spec_wave_obs)
    spec_fluxes = _np_to_jnp(context.spec_fluxes)
    spec_errors = _np_to_jnp(context.spec_errors)
    spec_mask = _bool_to_jnp(context.spec_mask)
    spec_spectrum_index = jnp.asarray(context.spec_spectrum_index, dtype=jnp.int32)
    spec_spatial_scale_arcsec = _np_to_jnp(context.spec_effective_spatial_scale_arcsec)
    dust_alpha_grid = np.asarray(context.templates.dust_alpha_grid, dtype=float)
    fit_host = bool(cfg.galaxy.fit_host)
    fit_agn = bool(cfg.agn.fit_agn)
    spatial_scale_arcsec = _np_to_jnp(context.effective_spatial_scale_arcsec)
    photometry_total_capture = _bool_to_jnp(context.photometry_total_capture)
    spectroscopy_enabled = bool(
        cfg.spectroscopy is not None
        and bool(cfg.spectroscopy_list)
        and len(context.spec_wave_obs) > 0
        and np.any(context.spec_mask)
    )
    spectral_cfg = cfg.agn
    spectral_engine_enabled = spectroscopy_enabled
    configured_spectral_lines = bool(
        spectral_engine_enabled
        and spectral_cfg.fit_lines
    )
    shared_spectral_lines = bool(
        spectral_engine_enabled
        and spectral_cfg.fit_lines
    )
    active_spectral_lines = bool(
        configured_spectral_lines
        and include_spectral_features
        and include_spectral_lines
    )
    use_native_feii = bool(
        include_sed_agn_features
        and not (spectral_engine_enabled and bool(spectral_cfg.fit_feii))
    )
    use_native_balmer = bool(
        include_sed_agn_features
        and not (spectral_engine_enabled and bool(spectral_cfg.fit_balmer_continuum))
    )
    spectral_line_coverage_rest = _fixed_spectral_line_coverage_rest(context, cfg) if spectroscopy_enabled else None
    has_phot_spatial_scale = bool(
        np.any(np.isfinite(context.effective_spatial_scale_arcsec) & (np.asarray(context.effective_spatial_scale_arcsec, dtype=float) > 0.0))
    )
    missing_partial_photometry_scale = (
        ~np.asarray(context.photometry_total_capture, dtype=bool)
        & ~(
            np.isfinite(context.effective_spatial_scale_arcsec)
            & (np.asarray(context.effective_spatial_scale_arcsec, dtype=float) > 0.0)
        )
    )
    has_missing_partial_photometry_scale = bool(np.any(missing_partial_photometry_scale))
    has_spec_spatial_scale = bool(
        spectroscopy_enabled
        and np.any(np.isfinite(context.spec_effective_spatial_scale_arcsec) & (np.asarray(context.spec_effective_spatial_scale_arcsec, dtype=float) > 0.0))
    )
    host_capture_enabled = bool(
        fit_host
        and cfg.likelihood.use_host_capture_model
        and (
            has_phot_spatial_scale
            or has_spec_spatial_scale
            or has_missing_partial_photometry_scale
        )
    )
    skip_coarse_agn_line_grid = bool(
        fit_agn
        and include_sed_agn_features
        and cfg.likelihood.use_local_line_photometry
        and not include_components
        and not spectroscopy_enabled
        and not cfg.likelihood.attenuation_model_uncertainty
    )
    skip_coarse_nebular_line_grid = bool(
        fit_host
        and cfg.nebular.enabled
        and cfg.nebular.emission
        and cfg.likelihood.use_local_line_photometry
        and not include_components
        and not spectroscopy_enabled
        and not cfg.likelihood.attenuation_model_uncertainty
    )
    if cfg.observation.fits_redshift:
        redshift = _sample_redshift(context, prior_config, cfg)
        luminosity_distance_m = _luminosity_distance_m_jax(
            redshift,
            cfg.galaxy.cosmology_h0,
            cfg.galaxy.cosmology_om0,
        )
        igm = _igm_transmission_at_redshift(context, redshift)
    else:
        redshift = context.fixed_redshift_jax
        luminosity_distance_m = context.fixed_luminosity_distance_m_jax
        igm = context.fixed_igm_jax
    needs_spec_host_basis = bool(
        spectroscopy_enabled
        and context.spec_host_basis_jax is not None
        and context.spec_rest_wave_jax.size == context.spec_wave_obs.size
        and not cfg.observation.fits_redshift
    )
    shared_zgas, shared_gal_lgmet = _resolve_tied_metallicity(context, prior_config)
    host_state = (
        _build_host_state(
            context,
            prior_config,
            full_output=include_components or needs_spec_host_basis,
            shared_gal_lgmet=shared_gal_lgmet,
            redshift=redshift,
        )
        if fit_host
        else _empty_host_state(context)
    )
    host_rest = host_state["host_rest"]
    host_kinematics_enabled = bool(fit_host and cfg.galaxy.fit_host_kinematics and spectroscopy_enabled)
    if host_kinematics_enabled:
        gal_v_kms = _sample_prior(prior_config, "gal_v_kms", dist.Normal(0.0, 150.0))
        gal_sigma_kms = _sample_prior(prior_config, "gal_sigma_kms", dist.HalfNormal(150.0))
        # Fixed-redshift joint fits have a dense spectroscopic host basis below.
        # Apply LOS kinematics there only: the global SED grid is intentionally
        # coarse over several wavelength decades and cannot resolve a
        # ~100-km/s convolution without severe Fourier ringing.
        if not needs_spec_host_basis:
            host_rest = _shift_and_broaden_single_spectrum_lnlam(
                jnp.log(rest_wave),
                host_rest,
                gal_v_kms,
                gal_sigma_kms,
            )
    else:
        gal_v_kms = jnp.asarray(0.0, dtype=jnp.float64)
        gal_sigma_kms = jnp.asarray(0.0, dtype=jnp.float64)

    host_llambda_5100 = jnp.interp(5100.0, rest_wave, host_rest, left=0.0, right=0.0) if fit_host else jnp.asarray(0.0, dtype=jnp.float64)
    if fit_agn:
        # Infer the AGN normalization directly from the photometry rather than
        # forcing the host 5100 A continuum to set the AGN amplitude.
        fracagn_5100 = jnp.asarray(0.999, dtype=jnp.float64)
        log_agn_amp = _sample_prior(
            prior_config,
            "log_agn_amp",
            _default_log_agn_amp_prior(context, cfg.observation.redshift),
        )
        agn_amp = jnp.exp(log_agn_amp)
        if fit_host:
            agn_llambda_5100 = agn_amp / 5100.0
            fracagn_5100 = jnp.clip(
                agn_llambda_5100 / jnp.maximum(agn_llambda_5100 + jnp.clip(host_llambda_5100, 0.0, 1.0e60), 1.0e-30),
                1.0e-4,
                0.999,
            )
    else:
        fracagn_5100 = jnp.asarray(1.0e-4, dtype=jnp.float64)
        agn_amp = jnp.asarray(0.0, dtype=jnp.float64)
        log_agn_amp = jnp.log(jnp.clip(agn_amp, 1.0e-30, 1.0e80))

    agn_type = int(cfg.agn.agn_type)
    if fit_agn:
        pl_slope = _sample_prior(
            prior_config,
            "pl_slope",
            dist.TruncatedNormal(-1.85, 0.6, low=GRAHSP_PL_SLOPE_LOW, high=GRAHSP_PL_SLOPE_HIGH),
        )
        # GRAHSP fixes the short-wavelength slope to zero. Keeping this
        # deterministic by default also removes a weakly identified UV-shape
        # degree of freedom from broadband-only fits: with the default bend
        # width it also reshapes the optical/NIR continuum. Supply a
        # ``uv_slope`` prior to fit it, e.g. TruncatedNormal(-0.6, 0.4,
        # low=-1.5, high=0.3) for the EUV composite alpha_nu ~= -1.4.
        if prior_config.get("uv_slope") is not None:
            uv_slope = _sample_prior(prior_config, "uv_slope", dist.Normal(0.0, 1.0))
        else:
            uv_slope = jnp.asarray(0.0, dtype=jnp.float64)
            numpyro.deterministic("uv_slope", uv_slope)
        pl_bend_loc = _sample_positive_distribution(
            prior_config,
            value_key="pl_bend_loc",
            log_key="log_pl_bend_loc",
            default_value_distribution=dist.TruncatedNormal(
                GRAHSP_PL_BEND_LOC_A,
                500.0,
                low=GRAHSP_PL_BEND_LOC_LOW_A,
                high=GRAHSP_PL_BEND_LOC_HIGH_A,
            ),
            default_log_distribution=dist.Normal(np.log(GRAHSP_PL_BEND_LOC_A), 0.2),
        )
        pl_bend_width = _sample_positive_distribution(
            prior_config,
            value_key="pl_bend_width",
            log_key="log_pl_bend_width",
            default_value_distribution=dist.TruncatedNormal(
                5.05,
                4.95,
                low=GRAHSP_PL_BEND_WIDTH_LOW,
                high=GRAHSP_PL_BEND_WIDTH_HIGH,
            ),
            default_log_distribution=dist.Normal(np.log(GRAHSP_PL_BEND_WIDTH), 0.4),
        )
        pl_cutoff = jnp.asarray(GRAHSP_PL_CUTOFF_A, dtype=jnp.float64)
        numpyro.deterministic("pl_cutoff", pl_cutoff)
        disk_rest = _powerlaw_jax(rest_wave, agn_amp / 5100.0, uv_slope, pl_slope, 5100.0, pl_bend_loc, pl_bend_width, pl_cutoff)

        fcov = _sample_positive_distribution(
            prior_config,
            value_key="fcov",
            log_key="log_fcov",
            default_value_distribution=dist.Uniform(0.05, 0.95),
            default_log_distribution=dist.Uniform(np.log(0.05), np.log(0.95)),
        )
        si = _sample_prior(prior_config, "si", dist.Normal(0.0, 1.0))
        cool_lam = _sample_positive_distribution(
            prior_config,
            value_key="cool_lam",
            log_key="log_cool_lam",
            default_value_distribution=dist.Uniform(10.0, 30.0),
            default_log_distribution=dist.TransformedDistribution(
                dist.Uniform(10.0, 30.0),
                dist.transforms.ExpTransform().inv,
            ),
        )
        cool_width = _sample_positive_distribution(
            prior_config,
            value_key="cool_width",
            log_key="log_cool_width",
            default_value_distribution=dist.Uniform(0.2, 0.65),
            default_log_distribution=dist.TransformedDistribution(
                dist.Uniform(0.2, 0.65),
                dist.transforms.ExpTransform().inv,
            ),
        )
        hot_lam = _sample_positive_distribution(
            prior_config,
            value_key="hot_lam",
            log_key="log_hot_lam",
            default_value_distribution=dist.Uniform(1.0, 5.5),
            default_log_distribution=dist.TransformedDistribution(
                dist.Uniform(1.0, 5.5),
                dist.transforms.ExpTransform().inv,
            ),
        )
        hot_width = _sample_positive_distribution(
            prior_config,
            value_key="hot_width",
            log_key="log_hot_width",
            default_value_distribution=dist.Uniform(0.2, 0.65),
            default_log_distribution=dist.TransformedDistribution(
                dist.Uniform(0.2, 0.65),
                dist.transforms.ExpTransform().inv,
            ),
        )
        hot_fcov = _sample_positive_distribution(
            prior_config,
            value_key="hot_fcov",
            log_key="log_hot_fcov",
            default_value_distribution=dist.LogUniform(0.04, 10.0),
            default_log_distribution=dist.Uniform(np.log(0.04), np.log(10.0)),
            default_to_log=True,
        )
        torus_rest = _torus_component(
            rest_wave,
            fcov,
            si,
            cool_lam,
            cool_width,
            hot_lam,
            hot_width,
            hot_fcov,
            0.29,
            GRAHSP_SI_EM_LAM_A,
            GRAHSP_SI_ABS_LAM_A,
            GRAHSP_SI_EM_WIDTH_A,
            GRAHSP_SI_ABS_WIDTH_A,
            agn_amp,
        )

        if include_sed_agn_features:
            if shared_spectral_lines or (
                configured_spectral_lines and not active_spectral_lines
            ):
                broad_lines_strength = jnp.asarray(1.0, dtype=jnp.float64)
                narrow_lines_strength = jnp.asarray(DEFAULT_NARROW_LINES_STRENGTH, dtype=jnp.float64)
                broad_line_width = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
                narrow_line_width = jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS, dtype=jnp.float64)
            else:
                broad_lines_strength = _sample_positive_distribution(
                    prior_config,
                    value_key="broad_lines_strength",
                    log_key="log_broad_lines_strength",
                    default_value_distribution=dist.LogUniform(0.3, 20.0),
                    default_log_distribution=dist.Uniform(np.log(0.3), np.log(20.0)),
                )
                narrow_lines_strength = _sample_positive_distribution(
                    prior_config,
                    value_key="narrow_lines_strength",
                    log_key="log_narrow_lines_strength",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_NARROW_LINES_STRENGTH), 0.5),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_NARROW_LINES_STRENGTH), 0.5),
                )
                broad_line_width = _sample_log_positive_from_distribution(
                    prior_config,
                    value_key="broad_line_width_kms",
                    log_key="log_broad_line_width_kms",
                    default_distribution=dist.TruncatedNormal(
                        np.log(DEFAULT_BROAD_LINE_WIDTH_KMS),
                        0.4,
                        low=np.log(DEFAULT_BROAD_LINE_WIDTH_KMS_MIN),
                        high=np.log(DEFAULT_BROAD_LINE_WIDTH_KMS_MAX),
                    ),
                )
                narrow_line_width = _sample_log_positive_from_distribution(
                    prior_config,
                    value_key="narrow_line_width_kms",
                    log_key="log_narrow_line_width_kms",
                    default_distribution=dist.TruncatedNormal(
                        np.log(DEFAULT_NARROW_LINE_WIDTH_KMS),
                        0.3,
                        low=np.log(DEFAULT_NARROW_LINE_WIDTH_KMS_MIN),
                        high=np.log(DEFAULT_NARROW_LINE_WIDTH_KMS_MAX),
                    ),
                )
            balmer_enabled = bool(use_native_balmer and cfg.agn.fit_balmer_continuum and agn_type == 1)
            if balmer_enabled:
                balmer_norm = _sample_positive_distribution(
                    prior_config,
                    value_key="balmer_norm",
                    log_key="log_balmer_norm",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_BALMER_CONTINUUM_STRENGTH), 1.0),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_BALMER_CONTINUUM_STRENGTH), 1.0),
                )
                balmer_tau = _sample_positive_distribution(
                    prior_config,
                    value_key="balmer_tau",
                    log_key="log_balmer_tau",
                    default_value_distribution=dist.LogNormal(np.log(1.0), 0.5),
                    default_log_distribution=dist.Normal(np.log(1.0), 0.5),
                )
                balmer_vel = _sample_positive_distribution(
                    prior_config,
                    value_key="balmer_vel",
                    log_key="log_balmer_vel",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.4),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.4),
                )
            else:
                balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
                balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
                balmer_vel = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        else:
            broad_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
            narrow_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
            broad_line_width = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
            narrow_line_width = jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS, dtype=jnp.float64)
            balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
            balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
            balmer_vel = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
            balmer_enabled = False
        l_agn_lambda_5100 = agn_amp / 5100.0
        agn_bol_luminosity = agn_amp * AGN_BOLOMETRIC_CORRECTION_5100
        l_broadlines = 0.02 * l_agn_lambda_5100 * broad_lines_strength
        l_narrowlines = 0.002 * l_agn_lambda_5100 * narrow_lines_strength

        suppress_joint_stage_lines = bool(
            configured_spectral_lines and not active_spectral_lines
        )
        if skip_coarse_agn_line_grid or suppress_joint_stage_lines:
            line_bl_rest = jnp.zeros_like(rest_wave)
            line_nl_rest = jnp.zeros_like(rest_wave)
            line_liner_rest = jnp.zeros_like(rest_wave)
            line_rest = jnp.zeros_like(rest_wave)
        else:
            line_bl_rest = jnp.where(
                agn_type == 1,
                _line_gaussians(rest_wave, line_wave, l_broadlines * line_blagn, broad_line_width),
                jnp.zeros_like(rest_wave),
            )
            line_nl_rest = jnp.where(
                agn_type in (1, 2),
                _line_gaussians(rest_wave, line_wave, l_narrowlines * line_sy2, narrow_line_width),
                jnp.zeros_like(rest_wave),
            )
            line_liner_rest = jnp.where(
                agn_type == 3,
                _line_gaussians(rest_wave, line_wave, l_narrowlines * line_liner, narrow_line_width),
                jnp.zeros_like(rest_wave),
            )
            line_rest = line_bl_rest + line_nl_rest + line_liner_rest

        if use_native_feii and agn_type == 1:
            feii_norm = _sample_positive_distribution(
                prior_config,
                value_key="feii_norm",
                log_key="log_feii_norm",
                default_value_distribution=dist.LogUniform(0.63, 31.6),
                default_log_distribution=dist.Uniform(np.log(0.63), np.log(31.6)),
            )
            l_feii = feii_norm * l_broadlines
            if cfg.agn.fit_feii_broadening:
                feii_fwhm = _sample_positive_distribution(
                    prior_config,
                    value_key="feii_fwhm",
                    log_key="log_feii_fwhm",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.3),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.3),
                )
                feii_shift = _sample_prior(prior_config, "feii_shift", dist.Normal(0.0, 0.01))
                feii_rest = _feii_component(rest_wave, feii_template_on_rest, l_feii, feii_fwhm, feii_shift)
            else:
                feii_fwhm = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
                feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
                feii_rest = l_feii * jnp.maximum(feii_template_on_rest, 0.0)
        else:
            feii_norm = jnp.asarray(0.0, dtype=jnp.float64)
            feii_fwhm = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
            feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
            feii_rest = jnp.zeros_like(rest_wave)
        balmer_rest = (
            _balmer_continuum_jax(
                rest_wave,
                l_agn_lambda_5100 * balmer_norm,
                15000.0,
                balmer_tau,
                balmer_vel,
            )
            if balmer_enabled
            else jnp.zeros_like(rest_wave)
        )
    else:
        uv_slope = jnp.asarray(0.0, dtype=jnp.float64)
        pl_slope = jnp.asarray(-1.0, dtype=jnp.float64)
        pl_bend_loc = jnp.asarray(GRAHSP_PL_BEND_LOC_A, dtype=jnp.float64)
        pl_bend_width = jnp.asarray(GRAHSP_PL_BEND_WIDTH, dtype=jnp.float64)
        pl_cutoff = jnp.asarray(GRAHSP_PL_CUTOFF_A, dtype=jnp.float64)
        disk_rest = jnp.zeros_like(rest_wave)
        fcov = jnp.asarray(0.0, dtype=jnp.float64)
        si = jnp.asarray(0.0, dtype=jnp.float64)
        cool_lam = jnp.asarray(17.0, dtype=jnp.float64)
        cool_width = jnp.asarray(0.45, dtype=jnp.float64)
        hot_lam = jnp.asarray(2.0, dtype=jnp.float64)
        hot_width = jnp.asarray(0.5, dtype=jnp.float64)
        hot_fcov = jnp.asarray(0.0, dtype=jnp.float64)
        torus_rest = jnp.zeros_like(rest_wave)
        broad_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
        narrow_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
        broad_line_width = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        narrow_line_width = jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS, dtype=jnp.float64)
        balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
        balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
        balmer_vel = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        l_agn_lambda_5100 = jnp.asarray(0.0, dtype=jnp.float64)
        agn_bol_luminosity = jnp.asarray(0.0, dtype=jnp.float64)
        line_bl_rest = jnp.zeros_like(rest_wave)
        line_nl_rest = jnp.zeros_like(rest_wave)
        line_liner_rest = jnp.zeros_like(rest_wave)
        line_rest = jnp.zeros_like(rest_wave)
        feii_norm = jnp.asarray(0.0, dtype=jnp.float64)
        feii_fwhm = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
        feii_rest = jnp.zeros_like(rest_wave)
        balmer_rest = jnp.zeros_like(rest_wave)

    ebv_gal = (
        _sample_positive_distribution(
            prior_config,
            value_key="ebv_gal",
            log_key="log_ebv_gal",
            default_value_distribution=dist.TransformedDistribution(
                dist.Uniform(np.log(GRAHSP_EBV_MIN), np.log(GRAHSP_EBV_MAX)),
                dist.transforms.ExpTransform(),
            ),
            default_log_distribution=dist.Uniform(np.log(GRAHSP_EBV_MIN), np.log(GRAHSP_EBV_MAX)),
            default_to_log=True,
        )
        if fit_host
        else jnp.asarray(0.0, dtype=jnp.float64)
    )
    ebv_agn = (
        _sample_positive_distribution(
            prior_config,
            value_key="ebv_agn",
            log_key="log_ebv_agn",
            default_value_distribution=dist.TransformedDistribution(
                dist.Uniform(np.log(GRAHSP_EBV_MIN), np.log(GRAHSP_EBV_MAX)),
                dist.transforms.ExpTransform(),
            ),
            default_log_distribution=dist.Uniform(np.log(GRAHSP_EBV_MIN), np.log(GRAHSP_EBV_MAX)),
            default_to_log=True,
        )
        if fit_agn
        else jnp.asarray(0.0, dtype=jnp.float64)
    )
    noll_delta = None
    if fit_host and cfg.galaxy.attenuation_curve == "noll2009":
        noll_delta = _sample_bounded_normal(
            prior_config, "dust_delta", NOLL_DELTA_PRIOR_LOC, NOLL_DELTA_PRIOR_SCALE, NOLL_DELTA_MIN, NOLL_DELTA_MAX
        )
        numpyro.deterministic("dust_bump_strength", _noll2009_bump_strength(noll_delta))
    # Screens handed to the line projectors: a bare E(B-V) on the default single
    # curve, or the host (Noll) plus AGN (GRAHSP) screens.
    if noll_delta is None:
        host_screen = ebv_gal
        agn_screen = ebv_gal + ebv_agn
    else:
        host_screen = DustScreen(ebv_gal, 0.0, noll_delta)
        agn_screen = DustScreen(ebv_gal, ebv_agn, noll_delta)
    if cfg.galaxy.use_energy_balance and fit_host and cfg.galaxy.dust_model == "dale2014":
        dust_alpha_low = max(0.75, float(np.min(dust_alpha_grid)))
        dust_alpha_high = min(2.75, float(np.max(dust_alpha_grid)))
        if "dust_alpha" in prior_config:
            dust_alpha = _sample_bounded_normal(
                prior_config,
                "dust_alpha",
                cfg.galaxy.dust_alpha,
                0.4,
                float(np.min(dust_alpha_grid)),
                float(np.max(dust_alpha_grid)),
            )
        else:
            # Continuous equivalent of the GRAHSP/CIGALE galdale grid, which
            # assigns equal prior weight over alpha=0.75--2.75.
            dust_alpha = numpyro.sample(
                "dust_alpha",
                dist.Uniform(dust_alpha_low, dust_alpha_high),
            )
    else:
        dust_alpha = jnp.asarray(float(cfg.galaxy.dust_alpha), dtype=jnp.float64)
    if cfg.galaxy.use_energy_balance and fit_host and cfg.galaxy.dust_model == "dl07":
        if "dust_umin" in prior_config:
            dust_umin = _sample_bounded_normal(
                prior_config,
                "dust_umin",
                cfg.galaxy.dust_umin,
                2.0,
                0.1,
                25.0,
            )
        else:
            # Prospector's one-dimensional DL07 shape prior.
            dust_umin = numpyro.sample("dust_umin", dist.Uniform(0.1, 25.0))
    else:
        dust_umin = jnp.asarray(float(cfg.galaxy.dust_umin), dtype=jnp.float64)
    if cfg.likelihood.fit_systematics_width:
        if "systematics_width" in prior_config or "log_systematics_width" in prior_config:
            systematics_width = _sample_positive(
                prior_config,
                value_key="systematics_width",
                log_key="log_systematics_width",
                default_value=float(cfg.likelihood.systematics_width_prior_scale),
                default_log_scale=1.0,
                default_family="exponential",
            )
        else:
            systematics_width = _sample_log_positive_from_distribution(
                prior_config,
                value_key="systematics_width",
                log_key="log_systematics_width",
                default_distribution=dist.TruncatedNormal(
                    np.log(0.10),
                    0.05,
                    low=np.log(0.07),
                    high=np.log(0.15),
                ),
            )
    else:
        systematics_width = jnp.asarray(float(cfg.likelihood.systematics_width), dtype=jnp.float64)
    if cfg.likelihood.fit_agn_systematics_width and fit_agn:
        agn_systematics_width = _sample_positive(
            prior_config,
            value_key="agn_systematics_width",
            log_key="log_agn_systematics_width",
            default_value=float(cfg.likelihood.agn_systematics_width_prior_scale),
            default_log_scale=0.5,
            default_family="lognormal",
        )
    else:
        agn_systematics_width = jnp.asarray(float(cfg.likelihood.agn_systematics_width), dtype=jnp.float64)
    nebular = _build_nebular_components(
        context,
        host_state,
        host_rest,
        prior_config,
        build_line_sed=not skip_coarse_nebular_line_grid,
        shared_zgas=shared_zgas,
    )
    host_with_nebular_rest = host_rest + nebular["absorption_rest"] + nebular["emission_rest"]
    agn_attenuated_input_rest = disk_rest + feii_rest + line_rest + balmer_rest
    gal_att_rest, agn_attenuated_rest, host_absorbed_rest, dust_luminosity = _apply_biattenuation(
        rest_wave,
        host_with_nebular_rest,
        agn_attenuated_input_rest,
        ebv_gal,
        ebv_agn,
        -1.2,
        -3.0,
        1.2,
        GRAHSP_BIATTENUATION_BREAK_A,
        noll_delta=noll_delta,
    )
    if skip_coarse_nebular_line_grid:
        # Narrow lines are generally unresolved by the broadband rest-wave
        # grid.  Use their conserved integrated luminosities for energy
        # balance instead of integrating undersampled Gaussian profiles.
        dust_luminosity = dust_luminosity + _absorbed_line_luminosity(
            context.nebular_templates_jax.line_wave_a,
            nebular["line_lumin"],
            ebv_gal,
            -1.2,
            -3.0,
            1.2,
            GRAHSP_BIATTENUATION_BREAK_A,
            noll_delta=noll_delta,
        )
    dust_luminosity = dust_luminosity + nebular["dust_luminosity"]
    attenuation_curve = _attenuation_curve(rest_wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    gal_att_factor = 10 ** (_screen_magnitudes(host_screen, rest_wave, attenuation_curve) / -2.5)
    agn_att_factor = 10 ** (_screen_magnitudes(agn_screen, rest_wave, attenuation_curve) / -2.5)
    host_stellar_att_rest = (host_rest + nebular["absorption_rest"]) * gal_att_factor
    nebular_att_rest = nebular["emission_rest"] * gal_att_factor
    nebular_lines_att_rest = nebular["lines_rest"] * gal_att_factor
    nebular_continuum_att_rest = nebular["continuum_rest"] * gal_att_factor
    disk_att_rest = disk_rest * agn_att_factor
    # The compact torus is seen through the same foreground host screen as
    # the rest of the source. Its own nuclear obscuration is already encoded
    # by the torus template, so do not apply the additional AGN E(B-V).
    torus_att_rest = torus_rest * gal_att_factor
    feii_att_rest = feii_rest * agn_att_factor
    line_bl_att_rest = line_bl_rest * agn_att_factor
    line_nl_att_rest = line_nl_rest * agn_att_factor
    line_liner_att_rest = line_liner_rest * agn_att_factor
    line_att_rest = line_rest * agn_att_factor
    balmer_att_rest = balmer_rest * agn_att_factor
    if cfg.galaxy.use_energy_balance and fit_host:
        if cfg.galaxy.dust_model == "dl07":
            dust_rest = _host_dl07_emission(context, dust_luminosity, dust_umin)
        else:
            dust_rest = _host_dust_emission(context, dust_luminosity, dust_alpha)
    else:
        dust_rest = jnp.zeros_like(rest_wave)
    agn_rest = agn_attenuated_rest + torus_att_rest
    total_rest = gal_att_rest + dust_rest + agn_rest
    direct_intrinsic_rest = host_with_nebular_rest + agn_attenuated_input_rest
    direct_attenuated_rest = gal_att_rest + agn_attenuated_rest
    fast_projection_enabled = _can_use_fixed_filter_projection(context, cfg)
    redshift_projection_enabled = (
        _can_use_redshift_filter_projection(context, cfg)
        and not include_components
        and not spectroscopy_enabled
    )
    needs_obs_sed = bool(include_components or spectroscopy_enabled)

    def band_fluxes(rest_lumin):
        """Return band fluxes in mJy of a rest-frame luminosity density on ``rest_wave``."""
        if fast_projection_enabled:
            return _project_rest_luminosity_filters(context, rest_lumin)
        if redshift_projection_enabled:
            return _project_redshift_luminosity_filters(context, rest_lumin, redshift)
        obs = _redshift_to_obs(rest_wave, rest_lumin * igm, obs_wave, redshift, luminosity_distance_m)
        return _project_filters(obs, context.packed_filters_jax)

    if fast_projection_enabled:
        total_obs = _redshift_to_obs(rest_wave, total_rest * igm, obs_wave, redshift, luminosity_distance_m) if needs_obs_sed else jnp.zeros_like(obs_wave)
        pred_fluxes_raw = _project_rest_luminosity_filters(context, total_rest)
    elif redshift_projection_enabled:
        total_obs = jnp.zeros_like(obs_wave)
        pred_fluxes_raw = _project_redshift_luminosity_filters(context, total_rest, redshift)
    else:
        total_obs = _redshift_to_obs(rest_wave, total_rest * igm, obs_wave, redshift, luminosity_distance_m)
        pred_fluxes_raw = _project_filters(total_obs, context.packed_filters_jax)
    local_agn_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    local_broad_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    local_narrow_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    coarse_agn_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    correct_agn_line_photometry = (
        fit_agn
        and include_sed_agn_features
        and not (configured_spectral_lines and not active_spectral_lines)
        and bool(cfg.likelihood.use_local_line_photometry)
    )
    if correct_agn_line_photometry:
        photometric_agn_line_mask = _photometric_agn_line_mask(context, cfg, line_wave, redshift)
        if agn_type == 1:
            local_broad_line_lumin = l_broadlines * line_blagn
            local_narrow_line_lumin = l_narrowlines * line_sy2
        elif agn_type == 2:
            local_broad_line_lumin = jnp.zeros_like(line_wave)
            local_narrow_line_lumin = l_narrowlines * line_sy2
        elif agn_type == 3:
            local_broad_line_lumin = jnp.zeros_like(line_wave)
            local_narrow_line_lumin = l_narrowlines * line_liner
        else:
            local_broad_line_lumin = jnp.zeros_like(line_wave)
            local_narrow_line_lumin = jnp.zeros_like(line_wave)
        local_broad_line_lumin = local_broad_line_lumin * photometric_agn_line_mask
        local_narrow_line_lumin = local_narrow_line_lumin * photometric_agn_line_mask
        if context.fixed_local_line_projection_cache_jax is not None and not cfg.observation.fits_redshift:
            local_broad_line_fluxes = _project_fixed_cached_local_line_filters(
                context,
                local_broad_line_lumin,
                broad_line_width,
                agn_screen,
            )
            local_narrow_line_fluxes = _project_fixed_cached_local_line_filters(
                context,
                local_narrow_line_lumin,
                narrow_line_width,
                agn_screen,
            )
        else:
            local_broad_line_fluxes = _project_local_line_filters(
                context,
                line_wave,
                local_broad_line_lumin,
                broad_line_width,
                agn_screen,
                redshift,
                luminosity_distance_m,
                igm,
            )
            local_narrow_line_fluxes = _project_local_line_filters(
                context,
                line_wave,
                local_narrow_line_lumin,
                narrow_line_width,
                agn_screen,
                redshift,
                luminosity_distance_m,
                igm,
            )
        local_agn_line_fluxes = local_broad_line_fluxes + local_narrow_line_fluxes
        coarse_agn_line_fluxes = band_fluxes(line_att_rest)
        pred_fluxes_raw = pred_fluxes_raw - coarse_agn_line_fluxes + local_agn_line_fluxes
    local_nebular_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    coarse_nebular_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    correct_nebular_line_photometry = (
        fit_host
        and bool(cfg.nebular.enabled)
        and bool(cfg.nebular.emission)
        and bool(cfg.likelihood.use_local_line_photometry)
    )
    if correct_nebular_line_photometry:
        if context.fixed_local_nebular_line_projection_cache_jax is not None and not cfg.observation.fits_redshift:
            local_nebular_line_fluxes = _project_fixed_cached_local_nebular_line_filters(
                context,
                nebular["line_lumin"],
                nebular["lines_width"],
                host_screen,
            )
        else:
            local_nebular_line_fluxes = _project_integrated_local_line_filters(
                context,
                context.nebular_templates_jax.line_wave_a,
                nebular["line_lumin"],
                nebular["lines_width"],
                host_screen,
                redshift,
                luminosity_distance_m,
                igm,
            )
        coarse_nebular_line_fluxes = band_fluxes(nebular_lines_att_rest)
        pred_fluxes_raw = pred_fluxes_raw - coarse_nebular_line_fluxes + local_nebular_line_fluxes
    host_dust_fluxes_total = jnp.zeros_like(pred_fluxes_raw)
    if host_capture_enabled or include_components or spectroscopy_enabled:
        host_obs = (
            jnp.zeros_like(obs_wave)
            if redshift_projection_enabled and not (include_components or spectroscopy_enabled)
            else _redshift_to_obs(rest_wave, gal_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        )
        if fast_projection_enabled:
            host_fluxes_total = _project_rest_luminosity_filters(context, gal_att_rest)
        elif redshift_projection_enabled:
            host_fluxes_total = _project_redshift_luminosity_filters(context, gal_att_rest, redshift)
        else:
            host_fluxes_total = _project_filters(host_obs, context.packed_filters_jax)
        if correct_nebular_line_photometry:
            host_fluxes_total = host_fluxes_total - coarse_nebular_line_fluxes + local_nebular_line_fluxes
        if host_capture_enabled:
            host_dust_fluxes_total = band_fluxes(dust_rest)
    else:
        host_obs = jnp.zeros_like(total_obs)
        host_fluxes_total = jnp.zeros_like(pred_fluxes_raw)
    if host_capture_enabled:
        log_host_capture_scale_arcsec = _sample_prior(
            prior_config,
            "log_host_capture_scale_arcsec",
            dist.Normal(np.log(3.0), 1.0),
        )
        host_capture_slope = jnp.asarray(2.0, dtype=jnp.float64)
        phot_capture_raw = _host_capture_fraction(
            spatial_scale_arcsec,
            jnp.exp(log_host_capture_scale_arcsec),
        )
        phot_scale_valid = jnp.isfinite(spatial_scale_arcsec) & (spatial_scale_arcsec > 0.0)
        if has_missing_partial_photometry_scale:
            missing_indices = np.flatnonzero(missing_partial_photometry_scale)
            missing_psf_capture_values = numpyro.sample(
                "missing_psf_host_capture_fraction",
                dist.Uniform(0.0, 1.0)
                .expand((len(missing_indices),))
                .to_event(1),
            )
            missing_psf_capture = jnp.ones_like(phot_capture_raw).at[
                jnp.asarray(missing_indices, dtype=jnp.int32)
            ].set(missing_psf_capture_values)
        else:
            missing_psf_capture = jnp.ones_like(phot_capture_raw)
        host_capture_fraction = jnp.where(
            photometry_total_capture,
            1.0,
            jnp.where(phot_scale_valid, phot_capture_raw, missing_psf_capture),
        )
        spec_capture_raw = _host_capture_fraction(
            spec_spatial_scale_arcsec,
            jnp.exp(log_host_capture_scale_arcsec),
        )
        spec_scale_valid = jnp.isfinite(spec_spatial_scale_arcsec) & (spec_spatial_scale_arcsec > 0.0)
        spec_host_capture_fraction_by_spectrum = jnp.where(spec_scale_valid, spec_capture_raw, 1.0)
    else:
        log_host_capture_scale_arcsec = jnp.asarray(np.log(3.0), dtype=jnp.float64)
        host_capture_slope = jnp.asarray(2.0, dtype=jnp.float64)
        host_capture_fraction = jnp.ones_like(pred_fluxes_raw)
        spec_host_capture_fraction_by_spectrum = jnp.ones_like(spec_spatial_scale_arcsec)
    host_capture_source_fluxes = host_fluxes_total + host_dust_fluxes_total
    agn_narrow_line_fluxes_total = jnp.zeros_like(pred_fluxes_raw)
    if host_capture_enabled and fit_agn and include_sed_agn_features:
        if correct_agn_line_photometry:
            agn_narrow_line_fluxes_total = local_narrow_line_fluxes
        else:
            agn_narrow_line_fluxes_total = band_fluxes(line_nl_att_rest)
    extended_capture_source_fluxes = host_capture_source_fluxes + agn_narrow_line_fluxes_total
    host_fluxes = host_fluxes_total * host_capture_fraction
    captured_host_dust_fluxes = host_dust_fluxes_total * host_capture_fraction
    captured_agn_narrow_line_fluxes = agn_narrow_line_fluxes_total * host_capture_fraction
    captured_extended_source_fluxes = extended_capture_source_fluxes * host_capture_fraction
    pred_fluxes = (
        pred_fluxes_raw
        if not host_capture_enabled
        else _apply_extended_capture(pred_fluxes_raw, extended_capture_source_fluxes, host_capture_fraction)
    )

    spec_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_host_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_disk_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_torus_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_continuum_model_fluxes = jnp.zeros_like(spec_fluxes)
    spectrum_scale = jnp.asarray(1.0, dtype=jnp.float64)
    log_spectrum_scale = jnp.asarray(0.0, dtype=jnp.float64)
    feature_amplitude_scale = jnp.asarray(1.0, dtype=jnp.float64)
    spec_chi2 = jnp.asarray(jnp.nan, dtype=jnp.float64)
    spec_n_eff = jnp.asarray(0.0, dtype=jnp.float64)
    spec_reduced_chi2 = jnp.asarray(jnp.nan, dtype=jnp.float64)
    spectral_line_photometry = jnp.zeros_like(pred_fluxes)
    spectral_feii_photometry = jnp.zeros_like(pred_fluxes)
    spectral_extrapolated_feii_photometry = jnp.zeros_like(pred_fluxes)
    spectral_balmer_photometry = jnp.zeros_like(pred_fluxes)
    spectral_extrapolated_broad_photometry = jnp.zeros_like(pred_fluxes)
    spectral_extrapolated_narrow_photometry = jnp.zeros_like(pred_fluxes)
    spectral_extrapolated_broad_lumin = jnp.zeros_like(line_wave)
    spectral_extrapolated_narrow_lumin = jnp.zeros_like(line_wave)
    spectral_extrapolated_broad_width = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
    spectral_extrapolated_narrow_width = jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS, dtype=jnp.float64)
    spectral_photometry_adjustment = jnp.zeros_like(pred_fluxes)
    spectral_custom_continuum_photometry = jnp.zeros_like(pred_fluxes)
    spectral_line_obs_sed = jnp.zeros_like(obs_wave)
    spectral_feii_obs_sed = jnp.zeros_like(obs_wave)
    spectral_balmer_obs_sed = jnp.zeros_like(obs_wave)
    spectral_custom_continuum_obs_sed = jnp.zeros_like(obs_wave)
    spectral_bal_decrement_obs_sed = jnp.zeros_like(obs_wave)
    active_bal_component_config = None
    disk_obs_for_bal = (
        _redshift_to_obs(
            rest_wave,
            disk_att_rest * igm,
            obs_wave,
            redshift,
            luminosity_distance_m,
        )
        if spectroscopy_enabled and include_spectral_features
        else jnp.zeros_like(obs_wave)
    )
    if spectroscopy_enabled:
        use_spec_resolution_continuum = bool(
            context.spec_host_basis_jax is not None
            and context.spec_rest_wave_jax.size == context.spec_wave_obs.size
            and not cfg.observation.fits_redshift
        )
        if use_spec_resolution_continuum:
            spec_rest_wave = context.spec_rest_wave_jax
            spec_igm = jnp.interp(spec_rest_wave, rest_wave, igm, left=0.0, right=0.0)
            spec_att_curve = _attenuation_curve(
                spec_rest_wave,
                -1.2,
                -3.0,
                1.2,
                GRAHSP_BIATTENUATION_BREAK_A,
            )
            spec_gal_att_factor = 10 ** (_screen_magnitudes(host_screen, spec_rest_wave, spec_att_curve) / -2.5)
            spec_agn_att_factor = 10 ** (_screen_magnitudes(agn_screen, spec_rest_wave, spec_att_curve) / -2.5)
            spec_nebular_absorption_rest = jnp.interp(
                spec_rest_wave,
                rest_wave,
                nebular["absorption_rest"],
                left=0.0,
                right=0.0,
            )
            spec_host_intrinsic = _host_rest_on_basis(host_state, context.spec_host_basis_jax)
            if host_kinematics_enabled:
                spec_host_intrinsic = _broaden_spectral_host(
                    context, spec_rest_wave, spec_host_intrinsic, gal_v_kms, gal_sigma_kms
                )
            spec_host_rest = (spec_host_intrinsic + spec_nebular_absorption_rest) * spec_gal_att_factor
            spec_disk_rest = (
                _powerlaw_jax(
                    spec_rest_wave,
                    agn_amp / 5100.0,
                    uv_slope,
                    pl_slope,
                    5100.0,
                    pl_bend_loc,
                    pl_bend_width,
                    pl_cutoff,
                )
                * spec_agn_att_factor
            )
            spec_torus_rest = _torus_component(
                spec_rest_wave,
                fcov,
                si,
                cool_lam,
                cool_width,
                hot_lam,
                hot_width,
                hot_fcov,
                0.29,
                GRAHSP_SI_EM_LAM_A,
                GRAHSP_SI_ABS_LAM_A,
                GRAHSP_SI_EM_WIDTH_A,
                GRAHSP_SI_ABS_WIDTH_A,
                agn_amp,
            )
            spec_denom = (
                4.0
                * jnp.pi
                * jnp.maximum(luminosity_distance_m, 1e-12) ** 2
                * jnp.maximum(1.0 + redshift, 1e-8)
            )
            spec_host_lambda = spec_host_rest * spec_igm / spec_denom
            spec_disk_lambda = spec_disk_rest * spec_igm / spec_denom
            spec_torus_lambda = spec_torus_rest * spec_igm / spec_denom
            if host_capture_enabled:
                spec_capture_at_pixel = spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                spec_host_lambda = spec_capture_at_pixel * spec_host_lambda
            spec_host_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_host_lambda)
            spec_disk_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_disk_lambda)
            spec_torus_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_torus_lambda)
            spec_model_fluxes = spec_host_model_fluxes + spec_disk_model_fluxes + spec_torus_model_fluxes
        else:
            disk_obs_for_spectrum = _redshift_to_obs(
                rest_wave,
                disk_att_rest * igm,
                obs_wave,
                redshift,
                luminosity_distance_m,
            )
            torus_obs_for_spectrum = _redshift_to_obs(
                rest_wave,
                torus_att_rest * igm,
                obs_wave,
                redshift,
                luminosity_distance_m,
            )
            spec_source_obs = host_obs + disk_obs_for_spectrum + torus_obs_for_spectrum
            spec_model_lambda = jnp.interp(spec_wave_obs, obs_wave, spec_source_obs, left=0.0, right=0.0)
            spec_host_lambda = jnp.interp(spec_wave_obs, obs_wave, host_obs, left=0.0, right=0.0)
            spec_disk_lambda = jnp.interp(
                spec_wave_obs, obs_wave, disk_obs_for_spectrum, left=0.0, right=0.0
            )
            spec_torus_lambda = jnp.interp(
                spec_wave_obs, obs_wave, torus_obs_for_spectrum, left=0.0, right=0.0
            )
            if host_capture_enabled:
                spec_capture_at_pixel = spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                spec_model_lambda = spec_model_lambda - spec_host_lambda + spec_capture_at_pixel * spec_host_lambda
                spec_host_lambda = spec_capture_at_pixel * spec_host_lambda
            spec_host_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_host_lambda)
            spec_disk_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_disk_lambda)
            spec_torus_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_torus_lambda)
            spec_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_model_lambda)
        spec_continuum_model_fluxes = spec_model_fluxes
        fit_spectrum_scale = (
            cfg.photometry is not None
            if cfg.likelihood.fit_spectrum_scale is None
            else bool(cfg.likelihood.fit_spectrum_scale)
        )
        if fit_spectrum_scale:
            scale_prior = _prior_distribution(
                prior_config,
                "log_spectrum_scale",
                dist.Normal(
                    0.0,
                    np.log(10.0)
                    * cfg.likelihood.spectrum_scale_prior_sigma_dex,
                ),
            )
            n_spectra = len(context.spec_instruments)
            pivot_offset = _spectrum_continuum_log_pivot(
                spec_continuum_model_fluxes,
                spec_fluxes,
                spec_mask,
                spec_spectrum_index,
                n_spectra,
            )
            scale_distribution = (
                scale_prior.expand([n_spectra]).to_event(1)
                if n_spectra > 1
                else scale_prior
            )
            log_spectrum_scale = numpyro.sample(
                "log_spectrum_scale",
                scale_distribution,
                infer={
                    "jaxsedfit_additive_pivot": {
                        "offset": pivot_offset,
                        "auxiliary_name": "log_spectrum_continuum_pivot",
                    }
                },
            )
            spectrum_scale = jnp.exp(log_spectrum_scale)
        feature_amplitude_scale = (
            spectrum_scale[0]
            if jnp.ndim(spectrum_scale) > 0
            else spectrum_scale
        )
        if include_spectral_features:
            spectral_components = _evaluate_spectral_components(
                spec_wave_obs,
                redshift,
                spec_model_fluxes,
                cfg,
                context.spectral_prior_config,
                feii_template_wave_native,
                feii_template_flux_native,
                line_coverage_rest=spectral_line_coverage_rest,
                feature_amplitude_scale=feature_amplitude_scale,
                include_lines=include_spectral_lines,
                instrumental_sigma_ln=_spectral_instrumental_sigma_ln(context),
                bal_continuum_mjy=spec_disk_model_fluxes,
                include_bal_components=include_spectral_bal,
            )
            spectral_cfg = cfg.agn
            if host_capture_enabled and bool(
                spectral_cfg.fit_lines and include_spectral_lines
            ):
                spec_capture_at_pixel = spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                spectral_line_model_aperture = _apply_extended_capture(
                    spectral_components["lines"],
                    spectral_components["line_narrow"],
                    spec_capture_at_pixel,
                )
                spectral_total_aperture = (
                    spectral_components["continuum"]
                    + spectral_components["feii"]
                    + spectral_components["balmer"]
                    + spectral_line_model_aperture
                )
                spectral_components = {
                    **spectral_components,
                    "total": spectral_total_aperture,
                    "lines": spectral_line_model_aperture,
                    "line_narrow": spec_capture_at_pixel * spectral_components["line_narrow"],
                }
                numpyro.deterministic("spectral_line_model_aperture", spectral_line_model_aperture)
                numpyro.deterministic("spectral_line_model_narrow_aperture", spectral_components["line_narrow"])
            spectral_state = spectral_components["state"]
            # Render the sampled spectral-feature state on the SED grid as
            # well. These are the same posterior line, Fe II, and Balmer
            # models used by the spectrum, not rescaled native templates.
            from .spectroscopy import render_joint_feature_state

            spectral_sed_rendered = render_joint_feature_state(
                obs_wave,
                redshift,
                spectral_state,
                config=spectral_components["component_config"],
                bal_continuum_mjy=_flambda_to_mjy(obs_wave, disk_obs_for_bal),
            )
            spectral_sed_lines_mjy = spectral_sed_rendered["lines"]
            sed_mjy_per_flambda = jnp.maximum(
                1.0e-10 / 299792458.0 * 1.0e29 * obs_wave**2,
                1.0e-30,
            )
            spectral_line_obs_sed = spectral_sed_lines_mjy / sed_mjy_per_flambda
            spectral_custom_continuum_obs_sed = (
                spectral_sed_rendered["custom_continuum"] / sed_mjy_per_flambda
            )
            spectral_bal_decrement_obs_sed = (
                spectral_sed_rendered["bal_decrement"] / sed_mjy_per_flambda
            )
            # Fe II and Balmer broadening are comparatively expensive FFTs.
            # Render them only for component predictions used by plotting, not
            # during every likelihood evaluation in MAP or NUTS.
            if include_components and (
                "feii_norm" in spectral_state or "balmer_norm" in spectral_state
            ):
                spectral_sed_smooth_mjy = render_joint_feature_state(
                    obs_wave,
                    redshift,
                    spectral_state,
                    config=spectral_components["component_config"],
                    feii_template_wave_rest=feii_template_wave_native,
                    feii_template_flux=feii_template_flux_native,
                )
                spectral_feii_obs_sed = spectral_sed_smooth_mjy["feii"] / sed_mjy_per_flambda
                spectral_balmer_obs_sed = spectral_sed_smooth_mjy["balmer"] / sed_mjy_per_flambda
            component_config = spectral_components["component_config"]
            if any(
                str(getattr(component, "metadata", {}).get("component_type", ""))
                == "bal_absorption"
                for component in getattr(component_config, "custom_components", ())
            ):
                active_bal_component_config = component_config
            spectral_broad_photometry = _project_spectral_line_state_filters(
                context,
                spectral_state,
                redshift,
                broad_only=True,
                component_cfg=active_bal_component_config,
            )
            spectral_narrow_photometry_total = _project_spectral_line_state_filters(
                context, spectral_state, redshift, broad_only=False
            )
            spectral_narrow_photometry = spectral_narrow_photometry_total * host_capture_fraction
            spectral_line_photometry = spectral_broad_photometry + spectral_narrow_photometry
            spectral_custom_broad_photometry = jnp.zeros_like(pred_fluxes)
            spectral_custom_narrow_photometry_total = jnp.zeros_like(pred_fluxes)
            if (
                getattr(component_config, "custom_components", ())
                or getattr(component_config, "custom_line_components", ())
            ):
                (
                    spectral_custom_continuum_photometry,
                    spectral_custom_broad_photometry,
                    spectral_custom_narrow_photometry_total,
                ) = _project_spectral_custom_state_filters(
                    context,
                    spectral_state,
                    redshift,
                    component_config,
                    bal_continuum_mjy=_flambda_to_mjy(
                        context.packed_filter_curves_jax.wave,
                        jax.vmap(
                            lambda wave: jnp.interp(
                                wave,
                                obs_wave,
                                disk_obs_for_bal,
                                left=0.0,
                                right=0.0,
                            )
                        )(context.packed_filter_curves_jax.wave),
                    ),
                )
            spectral_custom_narrow_photometry = (
                spectral_custom_narrow_photometry_total * host_capture_fraction
            )
            simple_spec_feii_shape = _simple_feii_fnu_shape(
                spec_wave_obs,
                redshift,
                feii_template_wave_native,
                feii_template_flux_native,
            )
            if spectral_line_coverage_rest is None:
                spec_feii_anchor_mask = spec_mask
            else:
                spec_rest_for_feii = spec_wave_obs / jnp.maximum(1.0 + redshift, 1.0e-8)
                spec_feii_anchor_mask = (
                    spec_mask
                    & (spec_rest_for_feii >= spectral_line_coverage_rest[0])
                    & (spec_rest_for_feii <= spectral_line_coverage_rest[1])
                )
            anchor_shape = jnp.where(spec_feii_anchor_mask, simple_spec_feii_shape, 0.0)
            anchor_spectral = jnp.where(spec_feii_anchor_mask, spectral_components["feii"], 0.0)
            sed_feii_scale = jnp.sum(anchor_shape * anchor_spectral) / jnp.maximum(
                jnp.sum(anchor_shape * anchor_shape), 1.0e-30
            )
            spectral_feii_photometry, spectral_extrapolated_feii_photometry, spectral_balmer_photometry = _project_spectral_smooth_state_filters(
                context,
                spectral_state,
                redshift,
                spectral_components["component_config"],
                feii_template_wave_native,
                feii_template_flux_native,
                spectral_line_coverage_rest,
                sed_feii_scale,
            )
            if bool(
                spectral_cfg.fit_lines and include_spectral_lines
            ):
                line_narrow_template = jnp.where(agn_type == 3, line_liner, line_sy2)
                spectral_extrapolated_broad_lumin, spectral_extrapolated_narrow_lumin, spectral_extrapolated_broad_width, spectral_extrapolated_narrow_width = (
                    _spectral_family_extrapolation(
                        context,
                        cfg,
                        spectral_state,
                        redshift,
                        luminosity_distance_m,
                        agn_screen,
                        igm,
                        line_wave,
                        line_blagn,
                        line_narrow_template,
                    )
                )
                # ``_spectral_family_extrapolation`` returns integrated line
                # luminosities.  The native AGN projector expects the legacy
                # CIGALE amplitude convention (lambda L_lambda / 5100 A),
                # which would multiply these fluxes by ~5100*sqrt(2).
                # Use the flux-conserving integrated-luminosity projector.
                spectral_extrapolated_broad_photometry = _project_integrated_local_line_filters(
                    context,
                    line_wave,
                    spectral_extrapolated_broad_lumin,
                    spectral_extrapolated_broad_width,
                    agn_screen,
                    redshift,
                    luminosity_distance_m,
                    igm,
                    spectral_state=spectral_state,
                    spectral_component_cfg=active_bal_component_config,
                )
                spectral_extrapolated_narrow_total = _project_integrated_local_line_filters(
                    context,
                    line_wave,
                    spectral_extrapolated_narrow_lumin,
                    spectral_extrapolated_narrow_width,
                    agn_screen,
                    redshift,
                    luminosity_distance_m,
                    igm,
                )
                spectral_extrapolated_narrow_photometry = spectral_extrapolated_narrow_total * host_capture_fraction
            shared_spectral_photometry = (
                spectral_line_photometry
                + spectral_custom_continuum_photometry
                + spectral_custom_broad_photometry
                + spectral_custom_narrow_photometry
                + spectral_feii_photometry
                + spectral_extrapolated_feii_photometry
                + spectral_balmer_photometry
                + spectral_extrapolated_broad_photometry
                + spectral_extrapolated_narrow_photometry
            )
            native_replaced_photometry = jnp.where(
                bool(spectral_cfg.fit_lines and include_spectral_lines),
                local_agn_line_fluxes,
                jnp.zeros_like(local_agn_line_fluxes),
            )
            spectral_photometry_adjustment = shared_spectral_photometry - native_replaced_photometry
            pred_fluxes = pred_fluxes + spectral_photometry_adjustment
            spec_model_fluxes = spectral_components["total"]
        if jnp.ndim(spectrum_scale) > 0:
            spec_model_fluxes = spectrum_scale[spec_spectrum_index] * spec_model_fluxes
        else:
            spec_model_fluxes = spectrum_scale * spec_model_fluxes
        spec_logl, spec_chi2_diagnostics = spectroscopic_loglike(
            spec_model_fluxes,
            spec_fluxes,
            spec_errors,
            spec_mask,
            cfg.likelihood.spectrum_systematics_width,
            cfg.likelihood.spectrum_student_t_df,
            return_diagnostics=True,
        )
        spec_chi2 = spec_chi2_diagnostics["chi2"]
        spec_n_eff = spec_chi2_diagnostics["n_eff"]
        spec_reduced_chi2 = jnp.where(
            spec_n_eff > 0.0,
            spec_chi2 / jnp.maximum(spec_n_eff, 1.0e-30),
            jnp.asarray(jnp.nan, dtype=jnp.float64),
        )
        if add_likelihood:
            numpyro.factor("spectroscopy_loglike_factor", spec_logl)
    else:
        log_spectrum_scale = jnp.asarray(0.0, dtype=jnp.float64)
        spec_logl = jnp.asarray(0.0, dtype=jnp.float64)
    need_agn_fluxes = fit_agn and (
        include_components
        or force_component_fluxes
        or cfg.likelihood.variability_uncertainty
        or cfg.likelihood.fit_agn_systematics_width
        or cfg.likelihood.agn_systematics_width > 0.0
    )
    need_trans_fluxes = include_components or cfg.likelihood.attenuation_model_uncertainty
    if include_components:
        agn_obs = _redshift_to_obs(rest_wave, agn_rest * igm, obs_wave, redshift, luminosity_distance_m)
        host_stellar_obs = _redshift_to_obs(rest_wave, host_stellar_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        dust_obs = _redshift_to_obs(rest_wave, dust_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_obs = _redshift_to_obs(rest_wave, nebular_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_lines_obs = _redshift_to_obs(rest_wave, nebular_lines_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_continuum_obs = _redshift_to_obs(rest_wave, nebular_continuum_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_lines_local_obs_wave, nebular_lines_local_obs = _local_integrated_line_obs_sed(
            context,
            context.nebular_templates_jax.line_wave_a,
            nebular["line_lumin"],
            nebular["lines_width"],
            host_screen,
            redshift,
            luminosity_distance_m,
            igm,
        )
        total_local_obs = (
            jnp.interp(nebular_lines_local_obs_wave, obs_wave, total_obs, left=0.0, right=0.0)
            - jnp.interp(nebular_lines_local_obs_wave, obs_wave, nebular_lines_obs, left=0.0, right=0.0)
            + nebular_lines_local_obs
        )
        disk_obs = _redshift_to_obs(rest_wave, disk_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        torus_obs = _redshift_to_obs(rest_wave, torus_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        feii_obs = _redshift_to_obs(rest_wave, feii_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_obs = _redshift_to_obs(rest_wave, line_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_bl_obs = _redshift_to_obs(rest_wave, line_bl_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_nl_obs = _redshift_to_obs(rest_wave, line_nl_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_liner_obs = _redshift_to_obs(rest_wave, line_liner_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        balmer_obs = _redshift_to_obs(rest_wave, balmer_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        joint_line_rendering = bool(
            spectroscopy_enabled
            and include_spectral_features
            and include_spectral_lines
            and (
                cfg.agn.fit_lines
                or cfg.agn.custom_line_components
            )
        )
        joint_feii_rendering = bool(
            spectroscopy_enabled
            and True
            and include_spectral_features
            and cfg.agn.fit_feii
        )
        joint_balmer_rendering = bool(
            spectroscopy_enabled
            and True
            and include_spectral_features
            and cfg.agn.fit_balmer_continuum
        )
        plotted_total_obs = total_obs
        plotted_agn_obs = agn_obs
        if spectroscopy_enabled and True:
            plotted_total_obs = plotted_total_obs + spectral_custom_continuum_obs_sed
            plotted_agn_obs = plotted_agn_obs + spectral_custom_continuum_obs_sed
        if joint_line_rendering:
            # The photometric likelihood replaces the native GRAHSP AGN-line
            # template with the fitted spectral state.  Apply the same replacement
            # to the plotted continuous SED instead of leaving invisible
            # native UV lines in the black and AGN-total curves.
            replaced_line_obs = jnp.where(
                bool(cfg.agn.fit_lines),
                line_obs,
                jnp.zeros_like(line_obs),
            )
            plotted_total_obs = plotted_total_obs - replaced_line_obs + spectral_line_obs_sed
            plotted_agn_obs = plotted_agn_obs - replaced_line_obs + spectral_line_obs_sed
        if joint_feii_rendering:
            # Joint fitting disables the native SED Fe II component. Add the
            # fitted JAXQSOFit state back to the continuous plotted totals,
            # matching the component curve and photometric likelihood.
            plotted_total_obs = plotted_total_obs - feii_obs + spectral_feii_obs_sed
            plotted_agn_obs = plotted_agn_obs - feii_obs + spectral_feii_obs_sed
        if joint_balmer_rendering:
            # As above, use the fitted spectral Balmer continuum in the total
            # curves. This remains active in continuum-only warm-up stages.
            plotted_total_obs = plotted_total_obs - balmer_obs + spectral_balmer_obs_sed
            plotted_agn_obs = plotted_agn_obs - balmer_obs + spectral_balmer_obs_sed
        if joint_line_rendering and correct_nebular_line_photometry:
            # Locally rendered nebular profiles are plotted below.  Remove the
            # undersampled coarse-grid version from the continuous total first.
            plotted_total_obs = plotted_total_obs - nebular_lines_obs

        agn_broad_local_wave, agn_broad_local_obs = _local_integrated_line_obs_sed(
            context,
            line_wave,
            spectral_extrapolated_broad_lumin,
            spectral_extrapolated_broad_width,
            agn_screen,
            redshift,
            luminosity_distance_m,
            igm,
            spectral_state=spectral_state if active_bal_component_config is not None else None,
            spectral_component_cfg=active_bal_component_config,
        )
        agn_narrow_local_wave, agn_narrow_local_obs = _local_integrated_line_obs_sed(
            context,
            line_wave,
            spectral_extrapolated_narrow_lumin,
            spectral_extrapolated_narrow_width,
            agn_screen,
            redshift,
            luminosity_distance_m,
            igm,
        )
        agn_lines_local_obs_wave = jnp.concatenate([agn_broad_local_wave, agn_narrow_local_wave])
        agn_lines_local_obs = jnp.concatenate([agn_broad_local_obs, agn_narrow_local_obs])
        total_agn_lines_local_obs = jnp.concatenate(
            [
                jnp.where(
                    agn_broad_local_obs > 0.0,
                    jnp.interp(agn_broad_local_wave, obs_wave, plotted_total_obs, left=0.0, right=0.0)
                    + agn_broad_local_obs,
                    jnp.nan,
                ),
                jnp.where(
                    agn_narrow_local_obs > 0.0,
                    jnp.interp(agn_narrow_local_wave, obs_wave, plotted_total_obs, left=0.0, right=0.0)
                    + agn_narrow_local_obs,
                    jnp.nan,
                ),
            ]
        )
        total_local_obs = (
            jnp.interp(nebular_lines_local_obs_wave, obs_wave, plotted_total_obs, left=0.0, right=0.0)
            + nebular_lines_local_obs
        )
        if fast_projection_enabled:
            agn_fluxes = _project_rest_luminosity_filters(context, agn_rest)
            dust_fluxes = _project_rest_luminosity_filters(context, dust_rest)
            nebular_fluxes = _project_rest_luminosity_filters(context, nebular_att_rest)
            nebular_lines_fluxes = _project_rest_luminosity_filters(context, nebular_lines_att_rest)
            nebular_continuum_fluxes = _project_rest_luminosity_filters(context, nebular_continuum_att_rest)
            disk_fluxes = _project_rest_luminosity_filters(context, disk_att_rest)
            torus_fluxes = _project_rest_luminosity_filters(context, torus_att_rest)
            feii_fluxes = _project_rest_luminosity_filters(context, feii_att_rest)
            line_fluxes = _project_rest_luminosity_filters(context, line_att_rest)
            line_bl_fluxes = _project_rest_luminosity_filters(context, line_bl_att_rest)
            line_nl_fluxes = _project_rest_luminosity_filters(context, line_nl_att_rest)
            line_liner_fluxes = _project_rest_luminosity_filters(context, line_liner_att_rest)
            balmer_fluxes = _project_rest_luminosity_filters(context, balmer_att_rest)
            direct_attenuated_fluxes = _project_rest_luminosity_filters(context, direct_attenuated_rest)
            direct_intrinsic_fluxes = _project_rest_luminosity_filters(context, direct_intrinsic_rest)
        else:
            agn_fluxes = _project_filters(agn_obs, context.packed_filters_jax)
            dust_fluxes = _project_filters(dust_obs, context.packed_filters_jax)
            nebular_fluxes = _project_filters(nebular_obs, context.packed_filters_jax)
            nebular_lines_fluxes = _project_filters(nebular_lines_obs, context.packed_filters_jax)
            nebular_continuum_fluxes = _project_filters(nebular_continuum_obs, context.packed_filters_jax)
            disk_fluxes = _project_filters(disk_obs, context.packed_filters_jax)
            torus_fluxes = _project_filters(torus_obs, context.packed_filters_jax)
            feii_fluxes = _project_filters(feii_obs, context.packed_filters_jax)
            line_fluxes = _project_filters(line_obs, context.packed_filters_jax)
            line_bl_fluxes = _project_filters(line_bl_obs, context.packed_filters_jax)
            line_nl_fluxes = _project_filters(line_nl_obs, context.packed_filters_jax)
            line_liner_fluxes = _project_filters(line_liner_obs, context.packed_filters_jax)
            balmer_fluxes = _project_filters(balmer_obs, context.packed_filters_jax)
            direct_attenuated_obs = _redshift_to_obs(
                rest_wave, direct_attenuated_rest * igm, obs_wave, redshift, luminosity_distance_m
            )
            direct_intrinsic_obs = _redshift_to_obs(
                rest_wave, direct_intrinsic_rest * igm, obs_wave, redshift, luminosity_distance_m
            )
            direct_attenuated_fluxes = _project_filters(direct_attenuated_obs, context.packed_filters_jax)
            direct_intrinsic_fluxes = _project_filters(direct_intrinsic_obs, context.packed_filters_jax)
        trans_fluxes = _band_transmitted_fraction(direct_attenuated_fluxes, direct_intrinsic_fluxes)
        if correct_nebular_line_photometry:
            nebular_lines_fluxes = local_nebular_line_fluxes
            nebular_fluxes = nebular_continuum_fluxes + local_nebular_line_fluxes
        if correct_agn_line_photometry:
            agn_fluxes = agn_fluxes - coarse_agn_line_fluxes + local_agn_line_fluxes
            line_bl_fluxes = local_broad_line_fluxes
            line_nl_fluxes = local_narrow_line_fluxes
            line_fluxes = line_bl_fluxes + line_nl_fluxes
        if host_capture_enabled and fit_agn and include_sed_agn_features:
            agn_fluxes = _apply_extended_capture(agn_fluxes, agn_narrow_line_fluxes_total, host_capture_fraction)
            line_nl_fluxes = captured_agn_narrow_line_fluxes
            line_fluxes = line_bl_fluxes + line_nl_fluxes
    else:
        nebular_lines_local_obs_wave = jnp.zeros((1,), dtype=jnp.float64)
        nebular_lines_local_obs = jnp.zeros((1,), dtype=jnp.float64)
        total_local_obs = jnp.zeros((1,), dtype=jnp.float64)
        if need_agn_fluxes:
            agn_fluxes = band_fluxes(agn_rest)
            if correct_agn_line_photometry:
                agn_fluxes = agn_fluxes - coarse_agn_line_fluxes + local_agn_line_fluxes
        else:
            agn_fluxes = jnp.zeros_like(pred_fluxes)
        if host_capture_enabled and fit_agn and include_sed_agn_features and need_agn_fluxes:
            agn_fluxes = _apply_extended_capture(agn_fluxes, agn_narrow_line_fluxes_total, host_capture_fraction)
        if need_trans_fluxes:
            direct_attenuated_fluxes = band_fluxes(direct_attenuated_rest)
            direct_intrinsic_fluxes = band_fluxes(direct_intrinsic_rest)
            trans_fluxes = _band_transmitted_fraction(direct_attenuated_fluxes, direct_intrinsic_fluxes)
        else:
            trans_fluxes = jnp.ones_like(pred_fluxes)

    if spectral_engine_enabled and include_spectral_features:
        agn_fluxes = agn_fluxes + spectral_photometry_adjustment
        constant_agn_fluxes = (
            spectral_narrow_photometry
            + spectral_custom_narrow_photometry
            + spectral_extrapolated_narrow_photometry
        )
    else:
        constant_agn_fluxes = (
            captured_agn_narrow_line_fluxes
            if host_capture_enabled
            else agn_narrow_line_fluxes_total
        )
    variable_agn_fluxes = jnp.where(
        need_agn_fluxes,
        agn_fluxes - constant_agn_fluxes,
        jnp.zeros_like(pred_fluxes),
    )
    logl, sed_chi2_diagnostics = photometric_loglike(
        pred_fluxes=pred_fluxes,
        obs_fluxes=obs_fluxes,
        obs_errors=obs_errors,
        upper_limits=upper_limits,
        data_mask=data_mask,
        systematics_width=systematics_width,
        agn_systematics_width=agn_systematics_width,
        likelihood_family=cfg.likelihood.likelihood_family,
        student_t_df=cfg.likelihood.student_t_df,
        agn_component=agn_fluxes,
        agn_bol_lum_w=agn_bol_luminosity,
        agn_nev=cfg.likelihood.agn_nev,
        variability_uncertainty=cfg.likelihood.variability_uncertainty,
        attenuation_model_uncertainty=cfg.likelihood.attenuation_model_uncertainty,
        transmitted_fraction=trans_fluxes,
        lyman_break_uncertainty=cfg.likelihood.lyman_break_uncertainty,
        filter_wavelength=filter_wavelength,
        redshift=redshift,
        nebular_line_component=local_nebular_line_fluxes,
        local_nebular_line_uncertainty_dex=cfg.likelihood.local_nebular_line_uncertainty_dex,
        return_diagnostics=True,
    )
    sed_chi2 = sed_chi2_diagnostics["chi2"]
    sed_n_eff = sed_chi2_diagnostics["n_eff"]
    sed_reduced_chi2 = sed_chi2_diagnostics["reduced_chi2"]
    joint_n_eff = sed_n_eff + spec_n_eff
    joint_chi2_sum = jnp.where(sed_n_eff > 0.0, sed_chi2, 0.0) + jnp.where(
        spec_n_eff > 0.0,
        spec_chi2,
        0.0,
    )
    joint_chi2 = jnp.where(
        joint_n_eff > 0.0,
        joint_chi2_sum,
        jnp.asarray(jnp.nan, dtype=jnp.float64),
    )
    joint_reduced_chi2 = jnp.where(
        joint_n_eff > 0.0,
        joint_chi2_sum / jnp.maximum(joint_n_eff, 1.0e-30),
        jnp.asarray(jnp.nan, dtype=jnp.float64),
    )
    if add_likelihood:
        numpyro.factor("photometry_loglike", logl)
    numpyro.deterministic("pred_fluxes", pred_fluxes)
    numpyro.deterministic(SpectralSites.MODEL_FLUX, spec_model_fluxes)
    numpyro.deterministic("variable_agn_fluxes", variable_agn_fluxes)
    numpyro.deterministic("constant_agn_fluxes", constant_agn_fluxes)
    numpyro.deterministic("spectral_line_photometry", spectral_line_photometry)
    numpyro.deterministic("spectral_feii_photometry", spectral_feii_photometry)
    numpyro.deterministic("spectral_extrapolated_feii_photometry", spectral_extrapolated_feii_photometry)
    numpyro.deterministic("spectral_balmer_photometry", spectral_balmer_photometry)
    numpyro.deterministic("spectral_extrapolated_broad_photometry", spectral_extrapolated_broad_photometry)
    numpyro.deterministic("spectral_extrapolated_narrow_photometry", spectral_extrapolated_narrow_photometry)
    numpyro.deterministic("spectral_custom_continuum_photometry", spectral_custom_continuum_photometry)
    numpyro.deterministic("spectral_photometry_adjustment", spectral_photometry_adjustment)
    numpyro.deterministic("spectral_line_obs_sed", spectral_line_obs_sed)
    numpyro.deterministic("spectral_feii_obs_sed", spectral_feii_obs_sed)
    numpyro.deterministic("spectral_balmer_obs_sed", spectral_balmer_obs_sed)
    numpyro.deterministic("spectral_custom_continuum_obs_sed", spectral_custom_continuum_obs_sed)
    numpyro.deterministic("spectral_bal_decrement_obs_sed", spectral_bal_decrement_obs_sed)
    numpyro.deterministic(SpectralSites.CONTINUUM_FLUX, spec_continuum_model_fluxes)
    numpyro.deterministic(SpectralSites.HOST_FLUX, spec_host_model_fluxes)
    numpyro.deterministic(SpectralSites.DISK_FLUX, spec_disk_model_fluxes)
    numpyro.deterministic(SpectralSites.TORUS_FLUX, spec_torus_model_fluxes)
    numpyro.deterministic(SpectralSites.SCALE, spectrum_scale)
    numpyro.deterministic("log_spectrum_scale_fit", log_spectrum_scale)
    numpyro.deterministic("spectral_feature_amplitude_scale", feature_amplitude_scale)
    numpyro.deterministic("spectrum_host_capture_fraction", spec_host_capture_fraction_by_spectrum)
    numpyro.deterministic("spectroscopy_loglike", spec_logl)
    numpyro.deterministic("sed_chi2", sed_chi2)
    numpyro.deterministic("sed_n_eff", sed_n_eff)
    numpyro.deterministic("sed_reduced_chi2", sed_reduced_chi2)
    numpyro.deterministic("spectroscopy_chi2", spec_chi2)
    numpyro.deterministic("spectroscopy_n_eff", spec_n_eff)
    numpyro.deterministic("spectroscopy_reduced_chi2", spec_reduced_chi2)
    numpyro.deterministic("joint_chi2", joint_chi2)
    numpyro.deterministic("joint_n_eff", joint_n_eff)
    numpyro.deterministic("joint_reduced_chi2", joint_reduced_chi2)
    numpyro.deterministic("fracAGN_5100_fit", fracagn_5100)
    numpyro.deterministic("log_agn_amp_fit", log_agn_amp)
    numpyro.deterministic("log_disk_luminosity_fit", _safe_log10(l_agn_lambda_5100))
    numpyro.deterministic("log_agn_bol_luminosity_fit", _safe_log10(agn_bol_luminosity))
    numpyro.deterministic("agn_variability_nev", _agn_variability_nev(agn_bol_luminosity, cfg.likelihood.agn_nev))
    numpyro.deterministic("transmitted_fraction_fluxes", trans_fluxes)
    numpyro.deterministic("host_total_fluxes", host_fluxes_total)
    numpyro.deterministic("host_capture_source_fluxes", host_capture_source_fluxes)
    numpyro.deterministic("captured_host_dust_fluxes", captured_host_dust_fluxes)
    numpyro.deterministic("agn_narrow_line_fluxes_total", agn_narrow_line_fluxes_total)
    numpyro.deterministic("captured_agn_narrow_line_fluxes", captured_agn_narrow_line_fluxes)
    numpyro.deterministic("extended_capture_source_fluxes", extended_capture_source_fluxes)
    numpyro.deterministic("captured_extended_source_fluxes", captured_extended_source_fluxes)
    numpyro.deterministic("host_capture_fraction_fluxes", host_capture_fraction)
    numpyro.deterministic("log_host_capture_scale_arcsec_fit", log_host_capture_scale_arcsec)
    numpyro.deterministic("host_capture_slope_fit", host_capture_slope)
    numpyro.deterministic("formed_stellar_mass", host_state["formed_mass"])
    numpyro.deterministic("surviving_mass_fraction", host_state["surviving_mass_fraction"])
    numpyro.deterministic("gal_lgmet_fit", host_state["gal_lgmet"])
    numpyro.deterministic("gal_lgmet_scatter_fit", host_state["gal_lgmet_scatter"])
    numpyro.deterministic("sfh_age_gyr_fit", host_state["sfh_age_gyr"])
    numpyro.deterministic("sfh_tau_gyr_fit", host_state["sfh_tau_gyr"])
    numpyro.deterministic("log_sfr_fit", _safe_log10(host_state["current_sfr"]))
    numpyro.deterministic("sfh_burst_fraction_fit", host_state.get("sfh_burst_fraction", 0.0))
    numpyro.deterministic("sfh_burst_age_gyr_fit", host_state.get("sfh_burst_age_gyr", 0.0))
    numpyro.deterministic("sfh_burst_tau_gyr_fit", host_state.get("sfh_burst_tau_gyr", 0.0))
    numpyro.deterministic("log_dust_luminosity_fit", _safe_log10(dust_luminosity))
    numpyro.deterministic("dust_alpha_fit", dust_alpha)
    numpyro.deterministic("dust_umin_fit", dust_umin)
    numpyro.deterministic("nebular_logU_fit", nebular["logU"])
    numpyro.deterministic("nebular_zgas_fit", nebular["zgas"])
    numpyro.deterministic("nebular_ne_fit", nebular["ne"])
    numpyro.deterministic("nebular_f_esc_fit", nebular["f_esc"])
    numpyro.deterministic("nebular_f_dust_fit", nebular["f_dust"])
    numpyro.deterministic("nebular_f_dust_fraction_fit", nebular.get("f_dust_fraction", nebular["f_dust"]))
    numpyro.deterministic("nebular_lines_width_fit", nebular["lines_width"])
    numpyro.deterministic("nebular_line_scale_fit", nebular["line_scale"])
    numpyro.deterministic("nebular_corr_fit", nebular["corr"])
    numpyro.deterministic("nebular_n_ly_young_fit", nebular["n_ly_young"])
    numpyro.deterministic("nebular_n_ly_old_fit", nebular["n_ly_old"])
    numpyro.deterministic("rest_wave", rest_wave)
    numpyro.deterministic("obs_wave", obs_wave)
    numpyro.deterministic(SpectralSites.WAVELENGTH_OBS, spec_wave_obs)
    numpyro.deterministic(SpectralSites.SPECTRUM_INDEX, spec_spectrum_index)
    numpyro.deterministic(SpectralSites.REDSHIFT, redshift)
    if include_components:
        numpyro.deterministic("host_age_weights", host_state["host_age_weights"])
        numpyro.deterministic("host_lgmet_weights", host_state["host_lgmet_weights"])
        numpyro.deterministic("host_ssp_weights", host_state["host_ssp_weights"])
        numpyro.deterministic("gal_sfr_table", host_state["gal_sfr_table"])
        numpyro.deterministic("gal_smh_table", host_state["gal_smh_table"])
        numpyro.deterministic("total_rest_sed", total_rest)
        numpyro.deterministic("agn_rest_sed", agn_rest)
        numpyro.deterministic("host_rest_sed", host_stellar_att_rest)
        numpyro.deterministic("host_total_rest_sed", gal_att_rest)
        numpyro.deterministic("host_absorbed_rest_sed", host_absorbed_rest)
        numpyro.deterministic("dust_rest_sed", dust_rest)
        numpyro.deterministic("nebular_rest_sed", nebular_att_rest)
        numpyro.deterministic("nebular_lines_rest_sed", nebular_lines_att_rest)
        numpyro.deterministic("nebular_continuum_rest_sed", nebular_continuum_att_rest)
        numpyro.deterministic("nebular_absorption_rest_sed", nebular["absorption_rest"])
        numpyro.deterministic("disk_rest_sed", disk_att_rest)
        numpyro.deterministic("torus_rest_sed", torus_att_rest)
        numpyro.deterministic("feii_rest_sed", feii_att_rest)
        numpyro.deterministic("line_rest_sed", line_att_rest)
        numpyro.deterministic("line_bl_rest_sed", line_bl_att_rest)
        numpyro.deterministic("line_nl_rest_sed", line_nl_att_rest)
        numpyro.deterministic("line_liner_rest_sed", line_liner_att_rest)
        numpyro.deterministic("balmer_rest_sed", balmer_att_rest)
        numpyro.deterministic("agn_fluxes", agn_fluxes)
        numpyro.deterministic("host_fluxes", host_fluxes)
        numpyro.deterministic("dust_fluxes", dust_fluxes)
        numpyro.deterministic("nebular_fluxes", nebular_fluxes)
        numpyro.deterministic("nebular_lines_fluxes", nebular_lines_fluxes)
        numpyro.deterministic("nebular_continuum_fluxes", nebular_continuum_fluxes)
        numpyro.deterministic("disk_fluxes", disk_fluxes)
        numpyro.deterministic("torus_fluxes", torus_fluxes)
        numpyro.deterministic("feii_fluxes", feii_fluxes)
        numpyro.deterministic("line_fluxes", line_fluxes)
        numpyro.deterministic("line_bl_fluxes", line_bl_fluxes)
        numpyro.deterministic("line_nl_fluxes", line_nl_fluxes)
        numpyro.deterministic("line_liner_fluxes", line_liner_fluxes)
        numpyro.deterministic("balmer_fluxes", balmer_fluxes)
        numpyro.deterministic("total_obs_sed", plotted_total_obs)
        numpyro.deterministic("total_local_lines_obs_wave", nebular_lines_local_obs_wave)
        numpyro.deterministic("total_local_lines_obs_sed", total_local_obs)
        numpyro.deterministic("total_agn_lines_local_obs_wave", agn_lines_local_obs_wave)
        numpyro.deterministic("total_agn_lines_local_obs_sed", total_agn_lines_local_obs)
        numpyro.deterministic("agn_lines_local_obs_wave", agn_lines_local_obs_wave)
        numpyro.deterministic("agn_lines_local_obs_sed", agn_lines_local_obs)
        numpyro.deterministic("agn_obs_sed", plotted_agn_obs)
        numpyro.deterministic("host_obs_sed", host_stellar_obs)
        numpyro.deterministic("host_total_obs_sed", host_obs)
        numpyro.deterministic("dust_obs_sed", dust_obs)
        numpyro.deterministic("nebular_obs_sed", nebular_obs)
        numpyro.deterministic("nebular_lines_obs_sed", nebular_lines_obs)
        numpyro.deterministic("nebular_lines_local_obs_wave", nebular_lines_local_obs_wave)
        numpyro.deterministic("nebular_lines_local_obs_sed", nebular_lines_local_obs)
        numpyro.deterministic("nebular_continuum_obs_sed", nebular_continuum_obs)
        numpyro.deterministic("disk_obs_sed", disk_obs)
        numpyro.deterministic("torus_obs_sed", torus_obs)
        numpyro.deterministic("feii_obs_sed", feii_obs)
        numpyro.deterministic("line_obs_sed", line_obs)
        numpyro.deterministic("line_bl_obs_sed", line_bl_obs)
        numpyro.deterministic("line_nl_obs_sed", line_nl_obs)
        numpyro.deterministic("line_liner_obs_sed", line_liner_obs)
        numpyro.deterministic("balmer_obs_sed", balmer_obs)

    if not return_state:
        return None

    state = {
        "pred_fluxes": pred_fluxes,
        "pred_spectrum_fluxes": spec_model_fluxes,
        "spec_continuum_model_fluxes": spec_continuum_model_fluxes,
        "spec_host_model_fluxes": spec_host_model_fluxes,
        "spec_disk_model_fluxes": spec_disk_model_fluxes,
        "spec_torus_model_fluxes": spec_torus_model_fluxes,
        "agn_fluxes": agn_fluxes,
        "host_fluxes": host_fluxes,
        "host_total_fluxes": host_fluxes_total,
        "host_capture_source_fluxes": host_capture_source_fluxes,
        "captured_host_dust_fluxes": captured_host_dust_fluxes,
        "dust_fluxes": dust_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "nebular_fluxes": nebular_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "nebular_lines_fluxes": nebular_lines_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "nebular_continuum_fluxes": nebular_continuum_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "rest_wave": rest_wave,
        "obs_wave": obs_wave,
        "redshift_fit": redshift,
        "photometry_loglike": logl,
        "spectroscopy_loglike": spec_logl,
        "sed_chi2": sed_chi2,
        "sed_n_eff": sed_n_eff,
        "sed_reduced_chi2": sed_reduced_chi2,
        "spectroscopy_chi2": spec_chi2,
        "spectroscopy_n_eff": spec_n_eff,
        "spectroscopy_reduced_chi2": spec_reduced_chi2,
        "joint_chi2": joint_chi2,
        "joint_n_eff": joint_n_eff,
        "joint_reduced_chi2": joint_reduced_chi2,
    }
    if include_components:
        state.update(
            {
                "total_rest_sed": total_rest,
                "agn_rest_sed": agn_rest,
                "host_rest_sed": host_stellar_att_rest,
                "host_total_rest_sed": gal_att_rest,
                "dust_rest_sed": dust_rest,
                "nebular_rest_sed": nebular_att_rest,
                "total_obs_sed": plotted_total_obs,
                "total_local_lines_obs_wave": nebular_lines_local_obs_wave,
                "total_local_lines_obs_sed": total_local_obs,
                "total_agn_lines_local_obs_wave": agn_lines_local_obs_wave,
                "total_agn_lines_local_obs_sed": total_agn_lines_local_obs,
                "agn_lines_local_obs_wave": agn_lines_local_obs_wave,
                "agn_lines_local_obs_sed": agn_lines_local_obs,
                "agn_obs_sed": plotted_agn_obs,
                "host_obs_sed": host_stellar_obs,
                "host_total_obs_sed": host_obs,
                "dust_obs_sed": dust_obs,
                "nebular_obs_sed": nebular_obs,
                "nebular_lines_obs_sed": nebular_lines_obs,
                "nebular_lines_local_obs_wave": nebular_lines_local_obs_wave,
                "nebular_lines_local_obs_sed": nebular_lines_local_obs,
                "nebular_continuum_obs_sed": nebular_continuum_obs,
                "disk_obs_sed": disk_obs,
                "torus_obs_sed": torus_obs,
                "feii_obs_sed": feii_obs,
                "line_obs_sed": line_obs,
                "line_bl_obs_sed": line_bl_obs,
                "line_nl_obs_sed": line_nl_obs,
                "line_liner_obs_sed": line_liner_obs,
                "balmer_obs_sed": balmer_obs,
            }
        )
    return state


def grahsp_photometric_model(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
    include_spectral_lines: bool = True,
    force_component_fluxes: bool = False,
    include_spectral_bal: bool = True,
):
    """NumPyro model for one jaxsedfit photometric fit or predictive expansion.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    include_components : bool
        If True, compute per-component observed SEDs and band fluxes for diagnostics.
    include_sed_agn_features : bool
        Whether the native SED AGN features (emission lines, Fe II, Balmer continuum)
        are included.
    include_spectral_features : bool
        Whether detailed spectral-engine features (Fe II, Balmer continuum, custom
        components) are evaluated.
    include_spectral_lines : bool
        Whether detailed broad and narrow spectral emission lines are active.
    force_component_fluxes : object
        Compute component band fluxes even when the likelihood does not need
        them. Used by lightweight posterior prediction products.
    include_spectral_bal : bool
        Whether BAL absorption custom components are active.
    """
    return evaluate_photometric_state(
        context,
        include_components=include_components,
        include_sed_agn_features=include_sed_agn_features,
        include_spectral_features=include_spectral_features,
        include_spectral_lines=include_spectral_lines,
        include_spectral_bal=include_spectral_bal,
        force_component_fluxes=force_component_fluxes,
        add_likelihood=True,
        return_state=False,
    )


def sed_numpyro_model(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
    include_spectral_lines: bool = True,
    include_spectral_bal: bool = True,
):
    """NumPyro SED model for one configured ``jaxsedfit`` target.

    This is the preferred public name for the low-level NumPyro model. The
    historical name :func:`grahsp_photometric_model` remains available as a
    compatibility alias.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    include_components : bool
        If True, compute per-component observed SEDs and band fluxes for diagnostics.
    include_sed_agn_features : bool
        Whether the native SED AGN features (emission lines, Fe II, Balmer continuum)
        are included.
    include_spectral_features : bool
        Whether detailed spectral-engine features (Fe II, Balmer continuum, custom
        components) are evaluated.
    include_spectral_lines : bool
        Whether detailed broad and narrow spectral emission lines are active.
    include_spectral_bal : bool
        Whether BAL absorption custom components are active.
    """
    return grahsp_photometric_model(
        context,
        include_components=include_components,
        include_sed_agn_features=include_sed_agn_features,
        include_spectral_features=include_spectral_features,
        include_spectral_lines=include_spectral_lines,
        include_spectral_bal=include_spectral_bal,
    )


def evaluate_sed_model(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
    add_likelihood: bool = True,
    return_state: bool = True,
    force_component_fluxes: bool = False,
    include_spectral_lines: bool = True,
    include_spectral_bal: bool = True,
):
    """Evaluate the SED model state for a configured target.

    This is the preferred public name for deterministic model evaluation. It
    delegates to :func:`evaluate_photometric_state`, which is kept for
    compatibility with earlier releases.

    Parameters
    ----------
    context : ModelContext
        Preloaded model context (data, templates, caches, and fit config).
    include_components : bool
        If True, compute per-component observed SEDs and band fluxes for diagnostics.
    include_sed_agn_features : bool
        Whether the native SED AGN features (emission lines, Fe II, Balmer continuum)
        are included.
    include_spectral_features : bool
        Whether detailed spectral-engine features (Fe II, Balmer continuum, custom
        components) are evaluated.
    include_spectral_lines : bool
        Whether detailed broad and narrow spectral emission lines are active.
    include_spectral_bal : bool
        Whether BAL absorption custom components are active.
    add_likelihood : bool
        If True, add the photometric and spectroscopic log-likelihood factors to the
        trace.
    return_state : bool
        If True, return the dictionary of model products; otherwise return None.
    force_component_fluxes : bool
        Compute component band fluxes even when the likelihood does not need them.
    """
    return evaluate_photometric_state(
        context,
        include_components=include_components,
        include_sed_agn_features=include_sed_agn_features,
        include_spectral_features=include_spectral_features,
        include_spectral_lines=include_spectral_lines,
        include_spectral_bal=include_spectral_bal,
        add_likelihood=add_likelihood,
        return_state=return_state,
        force_component_fluxes=force_component_fluxes,
    )


def photometric_log_likelihood(*args, **kwargs):
    """Return the photometric log likelihood for one model/data comparison.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return photometric_loglike(*args, **kwargs)


def spectroscopic_log_likelihood(*args, **kwargs):
    """Return the spectroscopic log likelihood for one model/data comparison.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return spectroscopic_loglike(*args, **kwargs)
