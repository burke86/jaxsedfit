"""Exact NumPyro coordinate transformations shared by spectral fitters."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import ndtr, ndtri
from numpyro.infer.reparam import Reparam


NORMAL_LOGNORMAL_STANDARDIZATION = (
    "spectral_normal_lognormal_standardization"
)


class TruncatedLogNormal(dist.TransformedDistribution):
    """LogNormal truncated above at ``high``: ``log x ~ Normal(loc, scale)``, ``x <= high``.

    Supports the same exact standard-Normal coordinate as ``LogNormal`` (through
    the truncated inverse CDF), so bounded widths keep the NUTS reparameterization.
    """

    arg_constraints = {}

    def __init__(self, loc, scale, high, *, validate_args=None):
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)
        self.high = jnp.asarray(high)
        base = dist.TruncatedNormal(self.loc, self.scale, high=jnp.log(self.high))
        super().__init__(base, dist.transforms.ExpTransform(), validate_args=validate_args)

    @property
    def support(self):
        return dist.constraints.interval(0.0, self.high)

    def tree_flatten(self):
        return (self.loc, self.scale, self.high), None

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params)

    def _upper_cdf(self):
        return ndtr((jnp.log(self.high) - self.loc) / self.scale)


def _is_two_sided_truncated_normal(fn) -> bool:
    return isinstance(fn, dist.TwoSidedTruncatedDistribution) and isinstance(fn.base_dist, dist.Normal)


def standardized_prior_site(site_name: str) -> str:
    """Return the auxiliary standard-Normal site for one physical prior."""
    return f"{str(site_name)}_std"


def _scoped_auxiliary_names(
    site_name: str,
    auxiliary_name: str,
) -> tuple[str, str]:
    """Return public and local auxiliary names under a NumPyro scope handler."""
    site_name = str(site_name)
    auxiliary_name = str(auxiliary_name)
    if "/" not in site_name:
        return auxiliary_name, auxiliary_name
    scope_prefix = site_name.rsplit("/", 1)[0]
    local_name = auxiliary_name.rsplit("/", 1)[-1]
    public_name = (
        auxiliary_name
        if "/" in auxiliary_name
        else f"{scope_prefix}/{auxiliary_name}"
    )
    return public_name, local_name


class NormalLogNormalStandardizeReparam(Reparam):
    """Sample a scalar Normal or LogNormal prior on a standard-Normal axis.

    The original physical sample site is retained as a deterministic value.
    This is an exact coordinate transformation: it changes neither the prior
    nor any downstream likelihood or prediction.
    """

    def __init__(
        self,
        auxiliary_name: str,
        sampling_name: str | None = None,
    ):
        self.auxiliary_name = str(auxiliary_name)
        self.sampling_name = str(
            self.auxiliary_name if sampling_name is None else sampling_name
        )

    @staticmethod
    def _validate(fn):
        if fn.batch_shape or fn.event_shape:
            raise ValueError(
                "Normal/LogNormal standardization currently supports only "
                "scalar sites."
            )
        if not (isinstance(fn, (dist.Normal, dist.LogNormal, TruncatedLogNormal)) or _is_two_sided_truncated_normal(fn)):
            raise ValueError(
                "Normal/LogNormal standardization requires a scalar Normal, "
                f"LogNormal, TruncatedLogNormal or two-sided TruncatedNormal prior, got {type(fn).__name__}."
            )

    def __call__(self, name, fn, obs):
        if obs is not None:
            raise ValueError(
                "Normal/LogNormal standardization requires a latent site."
            )
        self._validate(fn)
        loc_scale = fn.base_dist if _is_two_sided_truncated_normal(fn) else fn
        dtype = jnp.result_type(loc_scale.loc, loc_scale.scale)
        standardized = numpyro.sample(
            self.sampling_name,
            dist.Normal(
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
            ),
        )
        if _is_two_sided_truncated_normal(fn):
            # Inverse-CDF map of the two-sided truncated Normal.
            base = fn.base_dist
            cdf_low = ndtr((fn.low - base.loc) / base.scale)
            cdf_high = ndtr((fn.high - base.loc) / base.scale)
            return None, base.loc + base.scale * ndtri(cdf_low + ndtr(standardized) * (cdf_high - cdf_low))
        if isinstance(fn, TruncatedLogNormal):
            # Inverse-CDF map of the truncated Normal in log space.
            unconstrained = fn.loc + fn.scale * ndtri(ndtr(standardized) * fn._upper_cdf())
            return None, jnp.exp(unconstrained)
        unconstrained = fn.loc + fn.scale * standardized
        value = (
            jnp.exp(unconstrained)
            if isinstance(fn, dist.LogNormal)
            else unconstrained
        )
        return None, value

    def transform_initial_value(self, fn, value):
        """Map a physical initial value to its standard-Normal coordinate."""
        self._validate(fn)
        value = jnp.asarray(value)
        if _is_two_sided_truncated_normal(fn):
            base = fn.base_dist
            cdf_low = ndtr((fn.low - base.loc) / base.scale)
            cdf_high = ndtr((fn.high - base.loc) / base.scale)
            return ndtri((ndtr((value - base.loc) / base.scale) - cdf_low) / (cdf_high - cdf_low))
        if isinstance(fn, TruncatedLogNormal):
            return ndtri(ndtr((jnp.log(value) - fn.loc) / fn.scale) / fn._upper_cdf())
        unconstrained = (
            jnp.log(value) if isinstance(fn, dist.LogNormal) else value
        )
        return (unconstrained - fn.loc) / fn.scale


def normal_lognormal_standardization_reparam(site):
    """Resolve a model-provided scalar prior standardization."""
    metadata = (site.get("infer") or {}).get(
        NORMAL_LOGNORMAL_STANDARDIZATION
    )
    if metadata is None:
        return None
    auxiliary_name, sampling_name = _scoped_auxiliary_names(
        site["name"],
        metadata["auxiliary_name"],
    )
    return NormalLogNormalStandardizeReparam(
        auxiliary_name,
        sampling_name,
    )
