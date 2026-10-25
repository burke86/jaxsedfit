"""Batched forward simulation of broadband photometry.

:func:`simulate` draws model photometry, and optionally the per-component fluxes, for many
parameter sets at once. Parameters you pass are held fixed; every other model parameter is
drawn from the priors configured in the :class:`~jaxsedfit.config.FitConfig`. Uses include
mock catalogues, population studies, prior predictive checks and training data for
simulation-based (amortized) inference.

The observed photometry in the configuration is not used as data: only its filter names,
PSF/aperture metadata and photometry methods define what is simulated. The fluxes may be
placeholders. One caveat: the default AGN luminosity prior is seeded from the configured
fluxes, so set ``config.prior_config.agn.log_amp`` to a population prior when simulating
with an AGN component.

Example
-------
>>> import numpy as np
>>> from jaxsedfit.simulation import simulate, simulation_config
>>> cfg = simulation_config([f"sloan.sdss.{b}" for b in "ugriz"],
...                         redshift=0.5, dsps_ssp_fn="tempdata.h5")
>>> res = simulate(cfg, {"log_stellar_mass": np.full(100, 10.0)}, seed=1)
>>> res.fluxes.shape     # (100, 5), mJy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from .config import FitConfig, Observation, PhotometryData
from .model import evaluate_sed_model
from .preload import ModelContext, build_model_context

__all__ = ["SimulationResult", "make_simulator", "simulate", "simulation_config"]


@dataclass
class SimulationResult:
    """Simulated photometry for ``n`` parameter sets.

    Attributes
    ----------
    filter_names : list of str
        Filters, in the order of the flux columns.
    fluxes : numpy.ndarray
        Total model fluxes, shape ``(n, n_filters)``, in mJy.
    components : dict of str to numpy.ndarray
        Per-component band fluxes (e.g. ``host_fluxes``, ``agn_fluxes``,
        ``variable_agn_fluxes``), each ``(n, n_filters)`` in mJy. Empty when
        components were not requested.
    params : dict of str to numpy.ndarray
        Every sampled model parameter (fixed or drawn), leading dimension ``n``.
    """

    filter_names: list[str]
    fluxes: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)
    params: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.fluxes.shape[0])

    def to_dict(self) -> dict[str, np.ndarray]:
        """Flat columns: scalar parameters, ``flux_<filter>`` and ``<component>_<filter>``.

        ``pandas.DataFrame(result.to_dict())`` gives a table.
        """
        cols: dict[str, np.ndarray] = {}
        for name, value in self.params.items():
            if value.ndim == 1:
                cols[name] = value
        for j, f in enumerate(self.filter_names):
            cols[f"flux_{f}"] = self.fluxes[:, j]
        for comp, value in self.components.items():
            for j, f in enumerate(self.filter_names):
                cols[f"{comp}_{f}"] = value[:, j]
        return cols



def simulation_config(
    filter_names,
    *,
    redshift: float = 1.0,
    redshift_range: tuple[float, float] | None = None,
    n_redshift_grid: int = 256,
    dsps_ssp_fn: str | None = None,
    **fit_config_kwargs,
) -> FitConfig:
    """Build a :class:`FitConfig` for simulation, with placeholder photometry.

    Parameters
    ----------
    filter_names : sequence of str
        Filters to simulate (registry names or names defined in ``filters=``).
    redshift : float
        Fixed redshift, used when ``redshift_range`` is None.
    redshift_range : (float, float), optional
        If given, redshift becomes a sampled parameter with a flat prior over this range,
        and the filter projection is cached on ``n_redshift_grid`` points across it. Pass
        explicit ``redshift`` values to :func:`simulate` to impose another distribution.
    dsps_ssp_fn : str, optional
        DSPS SSP template file (``GalaxyConfig.dsps_ssp_fn``).
    **fit_config_kwargs
        Passed to :class:`FitConfig` (e.g. ``agn=``, ``galaxy=``, ``prior_config=``,
        ``filters=``).
    """
    names = list(filter_names)
    n = len(names)
    phot = PhotometryData(filter_names=names, fluxes=[1.0] * n, errors=[0.1] * n, is_upper_limit=[False] * n)
    if redshift_range is None:
        obs = Observation(redshift=float(redshift))
    else:
        lo, hi = map(float, redshift_range)
        if not 0.0 < lo < hi:
            raise ValueError("redshift_range must satisfy 0 < low < high")
        obs = Observation(redshift=0.5 * (lo + hi), redshift_mode="fit", redshift_err=hi - lo)
    cfg = FitConfig(observation=obs, photometry=phot, **fit_config_kwargs)
    if dsps_ssp_fn is not None:
        cfg.galaxy.dsps_ssp_fn = str(dsps_ssp_fn)
    if redshift_range is not None:
        grid = np.linspace(lo, hi, int(n_redshift_grid))
        cfg.prior_config.redshift.z_grid = list(grid)
        cfg.prior_config.redshift.pdf = list(np.ones_like(grid))
        cfg.likelihood.redshift_projection_n_grid = int(n_redshift_grid)
    return cfg


def _seeded_agn_prior_in_use(config: FitConfig) -> bool:
    return bool(config.agn.fit_agn) and config.prior_config.agn.log_amp is None


def make_simulator(
    context: ModelContext,
    *,
    include_components: bool = True,
    include_spectral_lines: bool = True,
) -> Callable[[jax.Array, Mapping[str, jax.Array]], tuple[jax.Array, dict, dict]]:
    """Return a jitted, batched simulator ``f(keys, params) -> (fluxes, components, params)``.

    ``keys`` has shape ``(n, 2)`` (``jax.random.split``); ``params`` maps site names to arrays
    with leading dimension ``n``. The function re-traces when the set of fixed parameter names
    or the batch size changes.
    """
    model = partial(
        evaluate_sed_model,
        context,
        include_components=include_components,
        force_component_fluxes=include_components,
        include_spectral_lines=include_spectral_lines,
        add_likelihood=False,
        return_state=False,
    )

    def one(key, params):
        tr = handlers.trace(handlers.substitute(handlers.seed(model, key), data=params)).get_trace()
        fluxes = tr["pred_fluxes"]["value"]
        comps = {
            name: site["value"]
            for name, site in tr.items()
            if site["type"] == "deterministic" and name.endswith("_fluxes") and name != "pred_fluxes"
        } if include_components else {}
        latents = {
            name: site["value"]
            for name, site in tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        return fluxes, comps, latents

    return jax.jit(jax.vmap(one))


def simulate(
    config: FitConfig,
    params: Mapping[str, Any] | None = None,
    n: int | None = None,
    *,
    seed: int = 0,
    batch_size: int = 4096,
    include_components: bool = True,
    include_spectral_lines: bool = True,
    context: ModelContext | None = None,
) -> SimulationResult:
    """Simulate photometry for ``n`` parameter sets.

    Parameters
    ----------
    config : FitConfig
        Model configuration; see :func:`simulation_config`. Its photometry fluxes are not
        used as data.
    params : mapping of str to array-like, optional
        Fixed values for model sites (e.g. ``redshift``, ``log_stellar_mass``,
        ``log_agn_amp``), each with leading dimension ``n``. Sites not given are drawn from
        the configured priors.
    n : int, optional
        Number of simulations; inferred from ``params`` when omitted.
    seed : int
        Random seed for the drawn parameters.
    batch_size : int
        Simulations per compiled batch; the last batch is padded, so one compilation
        serves the whole run.
    include_components : bool
        Also return per-component band fluxes.
    include_spectral_lines : bool
        Include emission-line contributions to the band fluxes.
    context : ModelContext, optional
        Reuse a context from :func:`jaxsedfit.preload.build_model_context`.

    Returns
    -------
    SimulationResult
    """
    params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in (params or {}).items()}
    sizes = {k: int(v.shape[0]) for k, v in params.items() if v.ndim >= 1}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"fixed parameters have different lengths: {sizes}")
    if n is None:
        if not sizes:
            raise ValueError("give n, or fixed parameters with a leading dimension")
        n = next(iter(sizes.values()))
    elif sizes and next(iter(sizes.values())) != n:
        raise ValueError(f"n={n} does not match the fixed parameters' length {next(iter(sizes.values()))}")
    if any(v.ndim == 0 for v in params.values()):
        params = {k: (jnp.full((n,), v) if v.ndim == 0 else v) for k, v in params.items()}
    if "log_agn_amp" not in params and _seeded_agn_prior_in_use(config):
        warnings.warn(
            "simulate: the default AGN luminosity prior is seeded from the configured (placeholder) "
            "photometry; set config.prior_config.agn.log_amp or pass log_agn_amp for a population prior",
            stacklevel=2,
        )

    ctx = context if context is not None else build_model_context(config)
    # Reuse the jitted simulator across calls on the same context: a fresh closure would miss
    # JAX's compilation cache and re-trace and re-compile the whole model on every call.
    simulators = ctx.__dict__.setdefault("_simulator_cache", {})
    sim_key = (bool(include_components), bool(include_spectral_lines))
    if sim_key not in simulators:
        simulators[sim_key] = make_simulator(
            ctx, include_components=include_components, include_spectral_lines=include_spectral_lines
        )
    sim = simulators[sim_key]
    bs = int(min(batch_size, n))
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), n)
    out_f, out_c, out_p = [], {}, {}
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        m = stop - start
        idx = np.arange(start, start + bs) % n if m < bs else np.arange(start, stop)  # pad the last batch
        f, c, p = sim(keys[idx], {k: v[idx] for k, v in params.items()})
        out_f.append(np.asarray(f)[:m])
        for k, v in c.items():
            out_c.setdefault(k, []).append(np.asarray(v)[:m])
        for k, v in p.items():
            out_p.setdefault(k, []).append(np.asarray(v)[:m])
    n_filt = len(config.photometry.filter_names)
    components = {k: np.concatenate(v) for k, v in out_c.items()}
    # keep band-flux components only; components unused by this configuration are empty
    components = {k: v for k, v in components.items() if v.ndim == 2 and v.shape[1] == n_filt}
    return SimulationResult(
        filter_names=list(config.photometry.filter_names),
        fluxes=np.concatenate(out_f),
        components=components,
        params={k: np.concatenate(v) for k, v in out_p.items()},
    )
