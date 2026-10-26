"""Batched inference for many objects that share one model configuration.

``JAXSEDFit`` compiles and fits one object at a time. When many objects share a
model configuration (same filters, templates, priors and settings) and differ only
in their data and redshift, the per-object ``ModelContext`` arrays can be stacked
and the model evaluated under ``jax.vmap``, so one compiled program fits a whole
chunk of objects. :func:`fit_batch` is the public entry point; see the "Fitting
many objects" docs page.

Spectra change the model structure in two ways, both removed before stacking:
pixel counts differ (spectra are padded with masked pixels), and the tied-line
table depends on rest-frame coverage. Objects are grouped into redshift windows
that share one line table (``AGNConfig.line_activation_rest``), and lines a
spectrum does not cover are masked to zero amplitude.

Supported methods are ``"map"``, ``"laplace"`` and ``"svi"``. The MAP mirrors
:meth:`jaxsedfit.JAXSEDFit.fit_map` (staged continuum-only then full model, Adam on
the unconstrained AutoDelta parameters, loss ``-log p(x, z)``), and the Laplace and
SVI steps use the same functions as the single-object fits
(:mod:`jaxsedfit.approximate`).
"""
from __future__ import annotations

import copy
import ctypes
import dataclasses
import gc
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import log_density

from .approximate import gaussian_vi, laplace_draws, latent_sites, site_transforms
from .config import FitConfig, serialize_config
from .model import grahsp_photometric_model
from .preload import (
    ModelContext,
    build_model_context,
    clear_per_object_caches,
    spectral_coverage_rest,
    usable_spectral_pixels,
)

_NUMERIC = (np.ndarray, jax.Array, float, np.floating)
# Zero-padded with a valid_mask whose widths depend on redshift; padding further is exact.
_PADDED_FIELDS = {"packed_filters", "packed_filters_jax", "packed_filter_curves_jax"}
_CONTINUUM_MODEL_KWARGS = dict(
    include_sed_agn_features=True,
    include_spectral_features=True,
    include_spectral_lines=False,
    include_spectral_bal=False,
)


def _leaves(obj: Any) -> list[Any]:
    """Flatten nested dataclasses, sequences, dicts and distributions into a list of leaves."""
    if isinstance(obj, dist.Distribution):
        # Data-scaled spectral priors differ between objects.
        return jax.tree_util.tree_leaves(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [leaf for f in dataclasses.fields(obj) for leaf in _leaves(getattr(obj, f.name))]
    if isinstance(obj, (list, tuple)):
        return [leaf for item in obj for leaf in _leaves(item)]
    if isinstance(obj, dict):
        return [leaf for key in sorted(obj, key=str) for leaf in _leaves(obj[key])]
    return [obj]


def _rebuild(obj: Any, leaves: Any) -> Any:
    """Inverse of :func:`_leaves`: rebuild ``obj`` taking leaves from an iterator."""
    if isinstance(obj, dist.Distribution):
        treedef = jax.tree_util.tree_structure(obj)
        return jax.tree_util.tree_unflatten(treedef, [next(leaves) for _ in range(treedef.num_leaves)])
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        new = copy.copy(obj)
        for f in dataclasses.fields(obj):
            object.__setattr__(new, f.name, _rebuild(getattr(obj, f.name), leaves))
        return new
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*[_rebuild(item, leaves) for item in obj])
    if isinstance(obj, (list, tuple)):
        return type(obj)(_rebuild(item, leaves) for item in obj)
    if isinstance(obj, dict):
        rebuilt = {key: _rebuild(obj[key], leaves) for key in sorted(obj, key=str)}
        return {key: rebuilt[key] for key in obj}
    return next(leaves)


def _leaf_equal(a: Any, b: Any) -> bool:
    if isinstance(a, _NUMERIC):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return True  # non-numeric leaves (distributions, names, ...) come from the template


class BatchedContext:
    """Layout for stacking per-object ``ModelContext`` arrays.

    The template context and the set of object-dependent leaves are fixed from a
    calibration set of contexts, which should span the redshift range of the
    sample. :meth:`stack` then batches any chunk of contexts against that layout,
    so one compiled program serves every chunk; it raises if a leaf that was
    constant in the calibration set differs.

    Parameters
    ----------
    contexts : sequence of ModelContext
        Calibration contexts built from the same configuration.
    pad_slack : int
        Extra padding columns for the redshift-dependent packed filter arrays.
    """

    def __init__(self, contexts: Sequence[ModelContext], pad_slack: int = 8):
        self.template = copy.copy(contexts[0])
        # ``filters`` holds observed-frame curves trimmed to each object's model grid. The
        # model reads only their effective wavelengths and the min/max wavelength (the range
        # of the AGN feature-rendering grid), so the template carries the union range.
        union = []
        for i, filt in enumerate(self.template.filters):
            waves = [np.asarray(c.filters[i].wave) for c in contexts if np.size(c.filters[i].wave)]
            new = copy.copy(filt)
            if waves:
                object.__setattr__(new, "wave", np.array([min(w.min() for w in waves), max(w.max() for w in waves)]))
            union.append(new)
        object.__setattr__(self.template, "filters", union)
        # Redshift (in fit_config) is batched explicitly; see :meth:`context`.
        self.fields = [f.name for f in dataclasses.fields(self.template) if f.name not in ("fit_config", "filters")]
        self._template_leaves = [_leaves(getattr(self.template, name)) for name in self.fields]
        per_ctx = [[_leaves(getattr(c, name)) for name in self.fields] for c in contexts]
        self.varying: list[tuple[int, int]] = []
        self.widths: list[int | None] = []
        for fi, name in enumerate(self.fields):
            for li, leaf in enumerate(per_ctx[0][fi]):
                if not isinstance(leaf, _NUMERIC):
                    continue
                # Only (filter, sample) arrays are padded; per-filter vectors are not.
                padded = name in _PADDED_FIELDS and np.ndim(leaf) >= 2
                if not padded and all(_leaf_equal(leaf, pc[fi][li]) for pc in per_ctx[1:]):
                    continue
                shapes = {np.shape(pc[fi][li]) for pc in per_ctx}
                width = None
                if padded:
                    if len({s[:-1] for s in shapes}) != 1:
                        raise ValueError(f"{name}[{li}] has object-dependent shape {shapes}")
                    width = max(s[-1] for s in shapes) + pad_slack
                elif len(shapes) != 1:
                    raise ValueError(f"{name}[{li}] has object-dependent shape {shapes}")
                self.varying.append((fi, li))
                self.widths.append(width)
        self._varying_set = set(self.varying)

    def stack(self, contexts: Sequence[ModelContext]) -> tuple[list[jnp.ndarray], jnp.ndarray]:
        """Stack the object-dependent arrays of ``contexts``.

        Returns
        -------
        arrays : list of jax.Array
            One array per varying leaf, with a leading object axis.
        redshift : jax.Array
            Object redshifts.
        """
        per_ctx = [[_leaves(getattr(c, name)) for name in self.fields] for c in contexts]
        for fi, name in enumerate(self.fields):
            for li, leaf in enumerate(self._template_leaves[fi]):
                if (fi, li) in self._varying_set or not isinstance(leaf, _NUMERIC):
                    continue
                if not all(_leaf_equal(leaf, pc[fi][li]) for pc in per_ctx):
                    raise ValueError(f"{name}[{li}] varies between objects but was constant in the calibration set")
        arrays = []
        for (fi, li), width in zip(self.varying, self.widths):
            values = [np.asarray(pc[fi][li]) for pc in per_ctx]
            if width is not None:
                if max(v.shape[-1] for v in values) > width:
                    raise ValueError(f"{self.fields[fi]}[{li}] is wider than the calibrated padding ({width})")
                values = [np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, width - v.shape[-1])]) for v in values]
            arrays.append(jnp.stack([jnp.asarray(v) for v in values]))
        redshift = jnp.asarray([c.fit_config.observation.redshift for c in contexts], dtype=jnp.float64)
        return arrays, redshift

    def context(self, arrays: Sequence[Any], redshift: Any) -> ModelContext:
        """Rebuild one object's ``ModelContext`` from its (possibly traced) arrays."""
        replace: dict[int, dict[int, Any]] = {}
        for (fi, li), arr in zip(self.varying, arrays):
            replace.setdefault(fi, {})[li] = arr
        ctx = copy.copy(self.template)
        for fi, name in enumerate(self.fields):
            if fi not in replace:
                continue
            leaves = list(self._template_leaves[fi])
            for li, arr in replace[fi].items():
                leaves[li] = arr
            object.__setattr__(ctx, name, _rebuild(getattr(self.template, name), iter(leaves)))
        cfg = copy.copy(self.template.fit_config)
        cfg.observation = copy.copy(cfg.observation)
        object.__setattr__(cfg.observation, "redshift", redshift)
        object.__setattr__(ctx, "fit_config", cfg)
        return ctx

    def model(self, **model_kwargs) -> Callable:
        """NumPyro model ``model(arrays, redshift)`` for one object of this layout."""

        def model(arrays, redshift):
            return grahsp_photometric_model(self.context(arrays, redshift), include_components=False, **model_kwargs)

        return model


class BatchedInputs:
    """Layout for stacking per-object keyword inputs of an arbitrary NumPyro model.

    The counterpart of :class:`BatchedContext` for models called as
    ``model(**inputs)``, such as :func:`jaxsedfit.spectral_model.qso_fsps_joint_model`.
    All objects must share one input structure: pad spectra to a common length
    beforehand. Numeric leaves that differ between the calibration inputs are
    stacked; all other values come from the first object.

    Parameters
    ----------
    model : callable
        NumPyro model taking keyword arguments.
    inputs : sequence of mapping
        Calibration inputs, one mapping of keyword arguments per object.
    redshift_key : str
        Key holding each object's redshift, which is stacked explicitly.
    """

    def __init__(self, model: Callable, inputs: Sequence[dict[str, Any]], redshift_key: str = "z_qso"):
        self._model = model
        self.redshift_key = redshift_key
        self.template = dict(inputs[0])
        self._template_leaves = _leaves(self.template)
        per = [_leaves(dict(x)) for x in inputs]
        self.varying = []
        for li, leaf in enumerate(self._template_leaves):
            if not isinstance(leaf, _NUMERIC) or all(_leaf_equal(leaf, p[li]) for p in per[1:]):
                continue
            shapes = {np.shape(p[li]) for p in per}
            if len(shapes) != 1:
                raise ValueError(f"Input leaf {li} has object-dependent shape {shapes}; pad the inputs first.")
            self.varying.append(li)
        self._varying_set = set(self.varying)

    def stack(self, inputs: Sequence[dict[str, Any]]) -> tuple[list[jnp.ndarray], jnp.ndarray]:
        """Stack the object-dependent leaves of ``inputs``; see :meth:`BatchedContext.stack`."""
        per = [_leaves(dict(x)) for x in inputs]
        for li, leaf in enumerate(self._template_leaves):
            if li in self._varying_set or not isinstance(leaf, _NUMERIC):
                continue
            if not all(_leaf_equal(leaf, p[li]) for p in per):
                raise ValueError(f"Input leaf {li} varies between objects but was constant in the calibration set")
        arrays = [jnp.stack([jnp.asarray(p[li]) for p in per]) for li in self.varying]
        redshift = jnp.asarray([float(x[self.redshift_key]) for x in inputs], dtype=jnp.float64)
        return arrays, redshift

    def inputs(self, arrays: Sequence[Any], redshift: Any) -> dict[str, Any]:
        """Rebuild one object's keyword inputs from its (possibly traced) arrays."""
        leaves = list(self._template_leaves)
        for li, arr in zip(self.varying, arrays):
            leaves[li] = arr
        kwargs = _rebuild(self.template, iter(leaves))
        kwargs[self.redshift_key] = redshift
        return kwargs

    def model(self, **model_kwargs) -> Callable:
        """NumPyro model ``model(arrays, redshift)``; ``model_kwargs`` override inputs."""

        def model(arrays, redshift):
            return self._model(**{**self.inputs(arrays, redshift), **model_kwargs})

        return model


_BATCH_METHODS = {"map": "map", "optax": "map", "laplace": "laplace", "svi": "svi"}


def _release_memory() -> None:
    """Free a fitted chunk's model contexts and hand the memory back to the OS.

    Contexts contain reference cycles, so they are only freed by the cyclic garbage
    collector, which runs based on object counts rather than bytes. On Linux, glibc
    also keeps freed memory unless asked to trim it; without this, memory grew by
    about a chunk's worth of contexts per chunk on CI runners.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def make_batched_fitter(
    bctx: BatchedContext | BatchedInputs,
    *,
    method: str = "map",
    steps: int = 2000,
    learning_rate: float = 5e-3,
    staged: bool = True,
    staged_steps: int | None = None,
    num_samples: int = 400,
    svi_steps: int = 2000,
    svi_learning_rate: float | None = None,
    svi_num_particles: int = 4,
    svi_full_rank: bool = True,
    derived_sites: Sequence[str] = (),
    continuum_model_kwargs: dict[str, Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a batched fitter for one :class:`BatchedContext` or :class:`BatchedInputs` layout.

    Parameters mirror :class:`jaxsedfit.config.InferenceConfig`; ``staged_steps``
    defaults to ``steps // 3`` as in :meth:`jaxsedfit.JAXSEDFit.fit_map`.
    ``derived_sites`` names deterministic model sites (e.g. ``"log_sfr_fit"``) to
    evaluate at the MAP and at every draw; they are added to ``map`` and ``samples``.
    ``continuum_model_kwargs`` are passed to ``bctx.model`` for the staged
    continuum-only MAP; the default switches off the jaxsedfit line features.

    Returns
    -------
    callable
        ``fit(arrays, redshift, keys, init_values=None)`` taking the output of
        :meth:`BatchedContext.stack`, one PRNG key per object and optional initial
        values (constrained, with a leading object axis) for the first MAP stage;
        later stages start from them updated with the earlier stage. It returns a dict
        with ``samples`` (constrained values per site with leading object and draw
        axes; one draw for ``"map"``), ``map`` (constrained MAP values), ``loss``
        (final ``-log p(x, z)`` at the MAP) and, for ``"laplace"``,
        ``hessian_min_eigenvalue`` or, for ``"svi"``, ``elbo_loss`` (final negative
        ELBO). The program is compiled on the first call and reused for chunks of the
        same size.
    """
    method = _BATCH_METHODS[method]
    derived_sites = tuple(derived_sites)
    full = bctx.model()
    continuum = bctx.model(**(_CONTINUUM_MODEL_KWARGS if continuum_model_kwargs is None else continuum_model_kwargs))
    optimizer = optax.adam(learning_rate)
    if staged_steps is None:
        staged_steps = max(1, steps // 3)

    def map_stage(model, transforms, n_steps):
        names = list(transforms)

        def init_one(key, arrays, redshift, init_values):
            def substitute(site):
                if site["name"] in init_values:
                    return init_values[site["name"]]
                return init_to_median(site, num_samples=15)

            seeded = handlers.seed(model, key)
            trace = handlers.trace(handlers.substitute(seeded, substitute_fn=substitute)).get_trace(arrays, redshift)
            return {k: transforms[k].inv(trace[k]["value"]) for k in names}

        def loss_one(u, arrays, redshift):
            params = {k: transforms[k](u[k]) for k in names}
            return -log_density(model, (arrays, redshift), {}, params)[0]

        def run(keys, arrays, redshift, init_values):
            u0 = jax.vmap(init_one)(keys, arrays, redshift, init_values)

            def total(u):
                # Parameters are disjoint between objects and Adam is elementwise, so
                # optimizing the sum is an independent Adam run per object.
                return jnp.sum(jax.vmap(loss_one)(u, arrays, redshift))

            def step(carry, _):
                u, state = carry
                grads = jax.grad(total)(u)
                updates, state = optimizer.update(grads, state, u)
                return (optax.apply_updates(u, updates), state), None

            (u, _), _ = jax.lax.scan(step, (u0, optimizer.init(u0)), None, length=n_steps)
            values = {k: transforms[k](u[k]) for k in names}
            return values, jax.vmap(loss_one)(u, arrays, redshift)

        return run

    compiled: dict[str, Any] = {}

    def build(arrays, redshift):
        first = [a[0] for a in arrays]
        transforms = site_transforms(latent_sites(full, (first, redshift[0])))
        transforms_c = site_transforms(latent_sites(continuum, (first, redshift[0])))
        run_c = map_stage(continuum, transforms_c, staged_steps)
        run_f = map_stage(full, transforms, steps)
        seeded_full = handlers.seed(full, rng_seed=0)  # all latent sites are substituted

        def derived_one(params, a, z):
            tr = handlers.trace(handlers.substitute(seeded_full, data=params)).get_trace(a, z)
            missing = [k for k in derived_sites if k not in tr]
            if missing:
                raise KeyError(f"derived_sites not in the model: {missing}")
            return {k: tr[k]["value"] for k in derived_sites}

        derived_objects = jax.vmap(derived_one)

        def derived_draws(samples, arrays, redshift):
            # Loop over draws and vmap over objects: vmapping over both would hold every
            # draw's full SED evaluation in memory at once.
            by_draw = {k: jnp.swapaxes(v, 0, 1) for k, v in samples.items()}
            out = jax.lax.map(lambda s: derived_objects(s, arrays, redshift), by_draw)
            return {k: jnp.swapaxes(v, 0, 1) for k, v in out.items()}

        @jax.jit
        def program(keys, arrays, redshift, given):
            map_keys, post_keys = jax.vmap(lambda k: jax.random.split(k))(keys).transpose(1, 0, 2)
            init_values = {k: v for k, v in given.items() if k in transforms}
            if staged:
                values_c, _ = run_c(map_keys, arrays, redshift, {k: v for k, v in given.items() if k in transforms_c})
                init_values.update({k: v for k, v in values_c.items() if k in transforms})
            values, loss = run_f(map_keys, arrays, redshift, init_values)
            out = {"map": values, "loss": loss}
            if method == "map":
                out["samples"] = {k: v[:, None] for k, v in values.items()}
            elif method == "laplace":
                draws, min_eig = jax.vmap(
                    lambda v, a, z, k: laplace_draws(full, (a, z), v, transforms, num_samples, k)
                )(values, arrays, redshift, post_keys)
                out["samples"], out["hessian_min_eigenvalue"] = draws, min_eig
            else:
                draws, losses = jax.vmap(
                    lambda v, a, z, k: gaussian_vi(
                        full, (a, z), v, transforms,
                        steps=svi_steps,
                        learning_rate=learning_rate if svi_learning_rate is None else svi_learning_rate,
                        num_particles=svi_num_particles,
                        full_rank=svi_full_rank,
                        num_samples=num_samples,
                        rng_key=k,
                    )
                )(values, arrays, redshift, post_keys)
                out["samples"], out["elbo_loss"] = draws, losses[:, -1]
            if derived_sites:
                out["map"] = {**values, **derived_objects(values, arrays, redshift)}
                out["samples"] = {**out["samples"], **derived_draws(out["samples"], arrays, redshift)}
            return out

        return program

    def fit(arrays, redshift, keys, init_values=None) -> dict[str, Any]:
        if "program" not in compiled:
            compiled["program"] = build(arrays, redshift)
        given = {k: jnp.asarray(v) for k, v in (init_values or {}).items()}
        out = compiled["program"](keys, arrays, redshift, given)
        return jax.tree_util.tree_map(np.asarray, out)

    return fit


def _batch_signature(cfg: FitConfig) -> str:
    """Everything in a config except per-object data, which a batch must share."""
    stripped = copy.copy(cfg)
    # Spectral pixels and resolving powers are per-object data (stacked after padding).
    stripped.spectroscopy = [
        dataclasses.replace(spectrum, wave_obs=(), fluxes=(), errors=(), mask=None, epoch_mjd=None,
                            resolving_power=spectrum.resolving_power is not None)
        for spectrum in cfg.spectroscopy_list
    ]
    data = serialize_config(stripped)
    for key in ("object_id", "redshift", "ra", "dec"):
        data["observation"].pop(key, None)
    for key in ("fluxes", "errors", "is_upper_limit"):
        (data["photometry"] or {}).pop(key, None)
    data.pop("output", None)
    return json.dumps(data, sort_keys=True, default=repr)


def _uses_line_table(cfg: FitConfig) -> bool:
    """Whether the config fits a tied-line table chosen from spectral coverage."""
    return bool(cfg.spectroscopy_list and cfg.agn.fit_lines and cfg.agn.tied_lines
                and not cfg.observation.fits_redshift)


def _share_line_tables(configs: list[FitConfig], redshift_window: float | None) -> list[FitConfig]:
    """Give spectroscopic configs in the same redshift window one fixed tied-line table.

    Objects are sorted by redshift within each configuration group and split into
    windows no wider than ``redshift_window`` in ln(1 + z). Each window's table spans
    the union of its members' line coverage (``AGNConfig.line_activation_rest``);
    configs that already fix the table are left alone. Returns shallow copies.
    """
    out = list(configs)
    pending: dict[str, list[int]] = {}
    for i, cfg in enumerate(configs):
        if _uses_line_table(cfg) and cfg.agn.line_activation_rest is None:
            pending.setdefault(_batch_signature(cfg), []).append(i)
    for indices in pending.values():
        coverage = {}
        for i in indices:
            cfg = configs[i]
            waves, masks = zip(*[(np.asarray(sp.wave_obs, dtype=float), usable_spectral_pixels(sp))
                                 for sp in cfg.spectroscopy_list])
            cov = spectral_coverage_rest(np.concatenate(waves), np.concatenate(masks),
                                         cfg.observation.redshift, cfg.agn.line_coverage_margin_kms)
            if cov is None:
                raise ValueError(f"Object {cfg.observation.object_id} has no usable spectral pixels.")
            coverage[i] = cov
        lnz = {i: np.log1p(max(float(configs[i].observation.redshift), 0.0)) for i in indices}
        order = sorted(indices, key=lambda i: lnz[i])
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and (
                redshift_window is None or lnz[order[stop]] - lnz[order[start]] <= redshift_window
            ):
                stop += 1
            window = order[start:stop]
            table = (min(coverage[i][0] for i in window), max(coverage[i][1] for i in window))
            for i in window:
                cfg = copy.copy(configs[i])
                cfg.agn = dataclasses.replace(cfg.agn, line_activation_rest=table)
                out[i] = cfg
            start = stop
    return out


@dataclasses.dataclass
class BatchFitResult:
    """Results of :func:`fit_batch`, one entry per input configuration.

    Parameters
    ----------
    configs : list of FitConfig
        Input configurations, in input order.
    method : str
        Inference method used.
    samples : list of dict
        Per-object posterior draws (``"laplace"``/``"svi"``) or the MAP as a single
        draw (``"map"``), keyed by site, each with a leading draw axis. Includes the
        requested ``derived_sites``.
    map_values : list of dict
        Per-object constrained MAP value of every latent site and requested derived
        site (all methods start from the MAP).
    diagnostics : list of dict
        Per-object scalars: ``loss`` (``-log p(x, z)`` at the MAP), plus
        ``hessian_min_eigenvalue`` (Laplace) or ``elbo_loss`` (SVI).
    """

    configs: list[FitConfig]
    method: str
    samples: list[dict[str, np.ndarray]]
    map_values: list[dict[str, np.ndarray]]
    diagnostics: list[dict[str, float]]

    def __len__(self) -> int:
        return len(self.configs)

    @property
    def object_ids(self) -> list[str]:
        """Object ids, in input order."""
        return [str(cfg.observation.object_id) for cfg in self.configs]

    def __getitem__(self, index: int):
        """Return a :class:`jaxsedfit.FitResult` for one object.

        The object's ``JAXSEDFit`` is rebuilt (its model context is not kept by the
        batch) and loaded with the batch draws, so predictions, plots, summaries and
        ``save`` work as for a single fit.
        """
        from .core import JAXSEDFit

        fitter = JAXSEDFit(self.configs[index])
        fitter.samples = {k: np.asarray(v) for k, v in self.samples[index].items()}
        fitter._ensure_fit_state().approx_result = {"method": self.method, **self.diagnostics[index]}
        return fitter._make_result(method=self.method)

    def summary_table(self, sites: Sequence[str] | None = None, percentiles: Sequence[float] = (16.0, 50.0, 84.0)):
        """Per-object percentiles of scalar sites as an :class:`astropy.table.Table`.

        Parameters
        ----------
        sites : sequence of str, optional
            Sites to include; defaults to every scalar site in ``samples``.
        percentiles : sequence of float
            Percentiles to tabulate (all equal to the MAP for ``"map"``).
        """
        from astropy.table import Table

        if sites is None:
            sites = [k for k, v in self.samples[0].items() if np.ndim(v) == 1]
        table = Table({"object_id": self.object_ids})
        for site in sites:
            values = np.array([np.percentile(np.asarray(s[site], dtype=float), percentiles) for s in self.samples])
            for j, q in enumerate(percentiles):
                table[f"{site}_p{q:g}"] = values[:, j]
        for key in self.diagnostics[0]:
            table[key] = [d[key] for d in self.diagnostics]
        return table


def fit_batch(
    configs: Sequence[FitConfig],
    method: str | None = None,
    *,
    chunk_size: int = 32,
    seed: int | None = None,
    progress_bar: bool = True,
    derived_sites: Sequence[str] = (),
    redshift_window: float | None = 0.25,
) -> BatchFitResult:
    """Fit many objects at once with one compiled program per compatible group.

    Objects are grouped by everything in their configuration except per-object data
    (object id, redshift, sky position, photometric fluxes, errors and upper-limit
    flags, spectral pixels and resolving powers), so a batch may mix filter sets or
    priors; each group compiles once. Within a group, objects are fitted
    ``chunk_size`` at a time under ``jax.vmap``, and the model contexts for the next
    chunk are built while the current chunk runs.

    Photometric, joint (photometry + spectra) and spectrum-only fits are supported.
    Spectroscopic objects are additionally split into redshift windows that share
    one tied-line table (``AGNConfig.line_activation_rest``): lines outside an
    object's spectral coverage stay in the model with zero amplitude, constrained by
    their priors only. Spectra in a group are padded with masked pixels to a common
    length. Fixing the table can change a fit slightly from ``JAXSEDFit`` (tied
    groups that straddle the coverage edge keep all their members); the returned
    configs carry the table, so ``JAXSEDFit`` on them fits the same model.

    Parameters
    ----------
    configs : sequence of FitConfig
        One configuration per object, e.g. built from a catalog row by row.
    method : {"map", "laplace", "svi"}, optional
        Inference method (``"optax"`` is accepted for ``"map"``); defaults to the
        first config's ``inference.method``. MAP, Laplace and SVI settings come from
        each group's ``inference`` config.
    chunk_size : int
        Objects per compiled batch. Larger chunks amortize overhead but use more
        memory (roughly 40 MB of model context per object for the default
        configuration, plus the optimizer state).
    seed : int, optional
        Base random seed; defaults to the first config's ``inference.seed``. Each
        object's key is derived from the seed and its position in ``configs``, so
        results do not depend on ``chunk_size``.
    progress_bar : bool
        Show a progress bar over chunks.
    derived_sites : sequence of str
        Deterministic model sites to evaluate at the MAP and at every draw, e.g.
        ``"log_sfr_fit"`` or ``"log_agn_bol_luminosity_fit"``. By default only latent
        sites are returned.
    redshift_window : float or None
        Widest ln(1 + z) spread of spectroscopic objects sharing a line table. Wider
        windows mean fewer compiled programs but more masked lines per object; None
        gives each configuration group a single table.

    Returns
    -------
    BatchFitResult
        Per-object draws and diagnostics, in input order, with the configs as
        fitted (including any fixed line table).
    """
    configs = list(configs)
    if not configs:
        raise ValueError("fit_batch needs at least one configuration.")
    method = str(configs[0].inference.method if method is None else method).lower()
    if method not in _BATCH_METHODS:
        raise ValueError("fit_batch supports method 'map' (or 'optax'), 'laplace' and 'svi'.")
    if any(cfg.spectroscopy_list and cfg.observation.fits_redshift for cfg in configs):
        raise NotImplementedError("fit_batch does not fit redshifts of spectroscopic objects; fit them with JAXSEDFit.")
    configs = _share_line_tables(configs, redshift_window)
    base_key = jax.random.PRNGKey(int(configs[0].inference.seed if seed is None else seed))

    groups: dict[str, list[int]] = {}
    for i, cfg in enumerate(configs):
        groups.setdefault(_batch_signature(cfg), []).append(i)

    samples: list[Any] = [None] * len(configs)
    map_values: list[Any] = [None] * len(configs)
    diagnostics: list[Any] = [None] * len(configs)
    n_chunks = sum(-(-len(idx) // chunk_size) for idx in groups.values())
    progress = None
    if progress_bar:
        from tqdm.auto import tqdm

        progress = tqdm(total=n_chunks, desc=f"fit_batch ({method})", unit="chunk")

    for indices in groups.values():
        inference = configs[indices[0]].inference
        settings = dict(
            method=method,
            steps=inference.map_steps,
            learning_rate=inference.learning_rate,
            staged=inference.staged_map,
            staged_steps=inference.staged_steps,
            num_samples=inference.num_samples,
            svi_steps=inference.svi_steps,
            svi_learning_rate=inference.svi_learning_rate,
            svi_num_particles=inference.svi_num_particles,
            svi_full_rank=inference.svi_full_rank,
            derived_sites=derived_sites,
        )
        z = np.array([float(configs[i].observation.redshift) for i in indices])
        # Pad every spectrum slot to the group's longest, so all chunks share one shape.
        pad_to = None
        if configs[indices[0]].spectroscopy_list:
            pad_to = [max(len(configs[i].spectroscopy_list[k].wave_obs) for i in indices)
                      for k in range(len(configs[indices[0]].spectroscopy_list))]
        rng = np.random.default_rng(0)
        # Calibrate the stacking layout on objects spanning the redshift range.
        calib = sorted({int(np.argmin(z)), int(np.argmax(z)),
                        *rng.choice(len(indices), min(14, len(indices)), replace=False).tolist()})
        calib_contexts = [build_model_context(configs[indices[j]], spec_pad_to=pad_to) for j in calib]
        clear_per_object_caches()
        bctx = BatchedContext(calib_contexts)
        fit = make_batched_fitter(bctx, **settings)
        chunks = [indices[s:s + chunk_size] for s in range(0, len(indices), chunk_size)]

        def prepare(chunk):
            padded = list(chunk) + [chunk[-1]] * (chunk_size - len(chunk))
            contexts = [build_model_context(configs[i], spec_pad_to=pad_to) for i in padded]
            # Redshift-keyed caches would otherwise grow by tens of MB per object.
            clear_per_object_caches()
            return padded, contexts

        with ThreadPoolExecutor(max_workers=1) as pool:  # build the next chunk while XLA runs
            future = pool.submit(prepare, chunks[0])
            for k, chunk in enumerate(chunks):
                padded, contexts = future.result()
                if k + 1 < len(chunks):
                    future = pool.submit(prepare, chunks[k + 1])
                try:
                    arrays, redshift = bctx.stack(contexts)
                except ValueError:
                    # A leaf that was constant in the calibration set varies here (or a padded
                    # array is wider): recalibrate including this chunk and recompile.
                    bctx = BatchedContext(calib_contexts + contexts)
                    fit = make_batched_fitter(bctx, **settings)
                    arrays, redshift = bctx.stack(contexts)
                keys = jnp.stack([jax.random.fold_in(base_key, i) for i in padded])
                out = fit(arrays, redshift, keys)
                for j, i in enumerate(chunk):
                    samples[i] = {site: v[j] for site, v in out["samples"].items()}
                    map_values[i] = {site: v[j] for site, v in out["map"].items()}
                    diag = {"loss": float(out["loss"][j])}
                    for key in ("hessian_min_eigenvalue", "elbo_loss"):
                        if key in out:
                            diag[key] = float(out[key][j])
                    diagnostics[i] = diag
                del out, arrays, redshift, contexts
                _release_memory()
                if progress is not None:
                    progress.update(1)
    if progress is not None:
        progress.close()
    return BatchFitResult(configs=configs, method=method, samples=samples, map_values=map_values, diagnostics=diagnostics)
