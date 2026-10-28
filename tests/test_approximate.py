import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from jaxsedfit.approximate import gaussian_vi, laplace_draws, latent_sites, site_transforms

# Conjugate model: mu ~ N(0, 2), y_i ~ N(mu, 0.5). The posterior of mu is exactly Gaussian.
Y = jnp.array([1.2, 0.8, 1.1, 0.9, 1.4])
PRIOR_SD, NOISE_SD = 2.0, 0.5
POST_VAR = 1.0 / (1.0 / PRIOR_SD**2 + Y.size / NOISE_SD**2)
POST_MEAN = POST_VAR * Y.sum() / NOISE_SD**2


def conjugate_model():
    mu = numpyro.sample("mu", dist.Normal(0.0, PRIOR_SD))
    numpyro.sample("y", dist.Normal(mu, NOISE_SD), obs=Y)


def bounded_model():
    # A positive parameter: draws must stay in its support.
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 0.3))
    numpyro.sample("y", dist.Normal(0.0, scale), obs=Y - Y.mean())


def test_laplace_is_exact_for_a_gaussian_posterior():
    transforms = site_transforms(latent_sites(conjugate_model))
    draws, min_eig = laplace_draws(conjugate_model, (), {"mu": POST_MEAN}, transforms, 20000, jax.random.PRNGKey(0))
    assert np.isclose(float(min_eig), 1.0 / POST_VAR, rtol=1e-6)
    assert np.isclose(np.mean(draws["mu"]), POST_MEAN, atol=0.01)
    assert np.isclose(np.std(draws["mu"]), np.sqrt(POST_VAR), rtol=0.02)


def saddle_model():
    # Negative curvature along a direction that is mostly x but mixes in the tightly
    # constrained y: the Hessian of -log p at (0, 0) is [[1, 20], [20, 100]].
    x = numpyro.sample("x", dist.Normal(0.0, 100.0))
    y = numpyro.sample("y", dist.Normal(0.0, 100.0))
    numpyro.factor("coupling", -0.5 * (x**2 + 100.0 * y**2 + 40.0 * x * y))


def test_laplace_keeps_non_convex_directions_local():
    transforms = site_transforms(latent_sites(saddle_model))
    draws, min_eig = laplace_draws(saddle_model, (), {"x": 0.0, "y": 0.0}, transforms, 4000, jax.random.PRNGKey(4))
    assert float(min_eig) < 0.0
    # The non-convex direction gets unit width, so y (which has a ~4% share of it)
    # stays near its well-constrained value instead of being thrown far away.
    assert np.std(draws["x"]) < 2.0
    assert np.std(draws["y"]) < 0.3


def test_gaussian_vi_recovers_a_gaussian_posterior():
    transforms = site_transforms(latent_sites(conjugate_model))
    draws, losses = gaussian_vi(
        conjugate_model, (), {"mu": 0.0}, transforms,
        steps=1500, learning_rate=0.02, num_particles=8, full_rank=True, num_samples=20000,
        rng_key=jax.random.PRNGKey(1),
    )
    assert losses[-100:].mean() < losses[:100].mean()
    assert np.isclose(np.mean(draws["mu"]), POST_MEAN, atol=0.03)
    assert np.isclose(np.std(draws["mu"]), np.sqrt(POST_VAR), rtol=0.1)


def test_approximations_respect_bounded_supports():
    transforms = site_transforms(latent_sites(bounded_model))
    laplace, _ = laplace_draws(bounded_model, (), {"scale": 0.3}, transforms, 2000, jax.random.PRNGKey(2))
    vi, _ = gaussian_vi(
        bounded_model, (), {"scale": 0.3}, transforms,
        steps=300, learning_rate=0.02, num_particles=4, full_rank=False, num_samples=2000,
        rng_key=jax.random.PRNGKey(3),
    )
    assert np.all(np.asarray(laplace["scale"]) > 0)
    assert np.all(np.asarray(vi["scale"]) > 0)


def test_finite_difference_hessian_matches_exact():
    for model, point in ((saddle_model, {"x": 0.3, "y": -0.2}), (bounded_model, {"scale": 0.7})):
        transforms = site_transforms(latent_sites(model))
        key = jax.random.PRNGKey(1)
        exact, eig_exact = laplace_draws(model, (), point, transforms, 500, key)
        fd, eig_fd = laplace_draws(model, (), point, transforms, 500, key, hessian="finite_difference")
        assert np.isclose(float(eig_fd), float(eig_exact), rtol=1e-6, atol=1e-6)
        for site in exact:
            assert np.allclose(fd[site], exact[site], rtol=1e-5, atol=1e-6)
