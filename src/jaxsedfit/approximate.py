"""Gaussian posterior approximations around a MAP solution: Laplace and variational.

Both approximations are Gaussian in NumPyro's *unconstrained* parameter space, so
bounded parameters (e.g. a Uniform prior) map back to draws that respect their
support. The functions here are pure JAX (no Python branching on data), so the
same code serves single-object fits (:meth:`jaxsedfit.JAXSEDFit.fit_laplace`,
:meth:`jaxsedfit.JAXSEDFit.fit_svi`) and batched fits under ``jax.vmap``
(:func:`jaxsedfit.fit_batch`).

The log density used is the posterior in unconstrained coordinates,
``log p(x, T(u)) + log|det dT/du|``.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import log_density

# Lower bound on the Hessian eigenvalues used by laplace_draws (unit width in unconstrained space).
_MIN_EIGENVALUE = 1.0


def latent_sites(model: Callable, model_args: Sequence[Any] = ()) -> dict[str, dict]:
    """Return the latent (unobserved, non-auxiliary) sample sites of ``model``."""
    trace = handlers.trace(handlers.seed(model, 0)).get_trace(*model_args)
    return {
        name: site
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"] and not site.get("infer", {}).get("is_auxiliary")
    }


def site_transforms(sites: Mapping[str, dict]) -> dict[str, Any]:
    """Bijections from unconstrained space to each site's support."""
    return {name: biject_to(site["fn"].support) for name, site in sites.items()}


def _unconstrained_potential(model, model_args, transforms, unravel):
    def potential(uflat):
        u = unravel(uflat)
        params = {k: t(u[k]) for k, t in transforms.items()}
        logdet = sum(jnp.sum(t.log_abs_det_jacobian(u[k], params[k])) for k, t in transforms.items())
        return -(log_density(model, tuple(model_args), {}, params)[0] + logdet)

    return potential


def _finite_difference_hessian(potential: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """Symmetrized central-difference Hessian of ``potential`` from its gradient."""
    grad = jax.grad(potential)
    step = 1e-5 * jnp.maximum(1.0, jnp.abs(x))
    shifts = jnp.eye(x.size) * step[:, None]
    plus = jax.vmap(lambda e: grad(x + e))(shifts)
    minus = jax.vmap(lambda e: grad(x - e))(shifts)
    hess = (plus - minus) / (2.0 * step[:, None])
    return 0.5 * (hess + hess.T)


def laplace_draws(
    model: Callable,
    model_args: Sequence[Any],
    map_values: Mapping[str, Any],
    transforms: Mapping[str, Any],
    num_samples: int,
    rng_key: jax.Array,
    hessian: str = "exact",
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Draw from the Laplace approximation at a MAP point.

    The Gaussian is centred on the MAP in unconstrained space with covariance equal
    to the inverse Hessian of the unconstrained negative log posterior. Hessian
    eigenvalues below one (flat or non-convex directions, or an unconverged MAP) are
    raised to one, i.e. a unit-width Gaussian in the unconstrained coordinates, which
    is roughly the prior scale. A much smaller floor would make the draws so wide
    along those directions that every parameter with even a small component in
    them is pushed to the edges of its support.

    Parameters
    ----------
    model : callable
        NumPyro model.
    model_args : sequence
        Positional arguments for ``model``.
    map_values : mapping
        Constrained MAP value of every latent site.
    transforms : mapping
        Output of :func:`site_transforms`.
    num_samples : int
        Number of draws.
    rng_key : jax.Array
        PRNG key.
    hessian : {"exact", "finite_difference"}
        ``"exact"`` uses ``jax.hessian``. ``"finite_difference"`` takes central
        differences of the gradient along each unconstrained coordinate (steps of
        1e-5, relative for coordinates above one). It compiles faster and uses less
        memory; on the jaxqsofit spectral model it matched the exact Hessian to about
        1e-8 relative, and the implied posterior widths to 1e-5.

    Returns
    -------
    draws : dict
        Constrained draws per site, with a leading ``num_samples`` axis.
    min_eigenvalue : jax.Array
        Smallest Hessian eigenvalue; below one means the approximation was regularized.
    """
    u_map = {k: t.inv(jnp.asarray(map_values[k])) for k, t in transforms.items()}
    uflat, unravel = ravel_pytree(u_map)
    potential = _unconstrained_potential(model, model_args, transforms, unravel)
    if hessian == "exact":
        hess = jax.hessian(potential)(uflat)
    elif hessian == "finite_difference":
        hess = _finite_difference_hessian(potential, uflat)
    else:
        raise ValueError("hessian must be 'exact' or 'finite_difference'.")
    eigval, eigvec = jnp.linalg.eigh(0.5 * (hess + hess.T))
    eps = jax.random.normal(rng_key, (num_samples, uflat.size))
    flat_draws = uflat + (eps / jnp.sqrt(jnp.maximum(eigval, _MIN_EIGENVALUE))) @ eigvec.T
    draws = jax.vmap(lambda d: {k: t(unravel(d)[k]) for k, t in transforms.items()})(flat_draws)
    return draws, eigval[0]


def gaussian_vi(
    model: Callable,
    model_args: Sequence[Any],
    init_values: Mapping[str, Any],
    transforms: Mapping[str, Any],
    *,
    steps: int,
    learning_rate: float,
    num_particles: int,
    full_rank: bool,
    num_samples: int,
    rng_key: jax.Array,
    init_scale: float = 0.1,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Fit a Gaussian variational posterior and draw from it.

    Maximizes the evidence lower bound with reparameterized Monte Carlo gradients
    (``num_particles`` draws per step) and Adam. ``full_rank=True`` fits a full
    covariance (like NumPyro's ``AutoMultivariateNormal``); ``False`` a diagonal one
    (like ``AutoNormal``).

    Parameters
    ----------
    model, model_args, transforms
        As in :func:`laplace_draws`.
    init_values : mapping
        Constrained starting point (normally the MAP).
    steps : int
        Optimization steps.
    learning_rate : float
        Adam learning rate.
    num_particles : int
        Monte Carlo draws per ELBO evaluation.
    full_rank : bool
        Full or diagonal covariance.
    num_samples : int
        Number of returned draws.
    rng_key : jax.Array
        PRNG key.
    init_scale : float
        Initial standard deviation in unconstrained space.

    Returns
    -------
    draws : dict
        Constrained draws per site, with a leading ``num_samples`` axis.
    losses : jax.Array
        Negative ELBO at each step.
    """
    u0 = {k: t.inv(jnp.asarray(init_values[k])) for k, t in transforms.items()}
    loc0, unravel = ravel_pytree(u0)
    dim = loc0.size
    potential = _unconstrained_potential(model, model_args, transforms, unravel)
    tril = jnp.tril_indices(dim, -1)

    def scale_tril(params):
        diag = jax.nn.softplus(params["raw_diag"])
        L = jnp.diag(diag)
        if full_rank:
            L = L.at[tril].set(params["off_diag"])
        return L

    params = {"loc": loc0, "raw_diag": jnp.full(dim, jnp.log(jnp.expm1(init_scale)))}
    if full_rank:
        params["off_diag"] = jnp.zeros(tril[0].size)

    def neg_elbo(p, key):
        L = scale_tril(p)
        eps = jax.random.normal(key, (num_particles, dim))
        u = p["loc"] + eps @ L.T
        # Gaussian entropy up to a constant.
        entropy = jnp.sum(jnp.log(jnp.diag(L)))
        return jnp.mean(jax.vmap(potential)(u)) - entropy

    optimizer = optax.adam(learning_rate)
    fit_key, draw_key = jax.random.split(rng_key)

    def step(carry, key):
        p, state = carry
        loss, grads = jax.value_and_grad(neg_elbo)(p, key)
        updates, state = optimizer.update(grads, state, p)
        return (optax.apply_updates(p, updates), state), loss

    (params, _), losses = jax.lax.scan(step, (params, optimizer.init(params)), jax.random.split(fit_key, steps))
    eps = jax.random.normal(draw_key, (num_samples, dim))
    flat_draws = params["loc"] + eps @ scale_tril(params).T
    draws = jax.vmap(lambda d: {k: t(unravel(d)[k]) for k, t in transforms.items()})(flat_draws)
    return draws, losses
