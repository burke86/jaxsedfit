import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.util import log_density

from jaxsedfit.spectral_reparameterization import TruncatedLogNormal
from jaxsedfit.spectroscopy import (
    NORMAL_LOGNORMAL_STANDARDIZATION,
    NormalLogNormalStandardizeReparam,
    normal_lognormal_standardization_reparam,
    standardized_prior_site,
)


def _standardizable_prior_model(prior):
    return numpyro.sample(
        "physical",
        prior,
        infer={
            NORMAL_LOGNORMAL_STANDARDIZATION: {
                "auxiliary_name": standardized_prior_site("physical"),
            }
        },
    )


def test_normal_lognormal_standardization_is_exact_and_maps_initial_values():
    wrapped = reparam(
        _standardizable_prior_model,
        config=normal_lognormal_standardization_reparam,
    )
    standardized = np.asarray(-0.4)
    priors = (
        dist.Normal(2.0, 3.0),
        dist.LogNormal(np.log(2.0), 0.7),
    )

    for prior in priors:
        unconstrained = prior.loc + prior.scale * standardized
        physical = (
            np.exp(np.asarray(unconstrained))
            if isinstance(prior, dist.LogNormal)
            else np.asarray(unconstrained)
        )
        original_density, _ = log_density(
            _standardizable_prior_model,
            (),
            {"prior": prior},
            {"physical": physical},
        )
        standardized_density, standardized_trace = log_density(
            wrapped,
            (),
            {"prior": prior},
            {"physical_std": standardized},
        )
        log_abs_det = np.log(float(prior.scale))
        if isinstance(prior, dist.LogNormal):
            log_abs_det += np.log(float(physical))
        np.testing.assert_allclose(
            standardized_density,
            original_density + log_abs_det,
            rtol=1.0e-12,
        )
        assert standardized_trace["physical"]["type"] == "deterministic"
        np.testing.assert_allclose(
            standardized_trace["physical"]["value"],
            physical,
            rtol=1.0e-12,
        )
        reparameterizer = NormalLogNormalStandardizeReparam("physical_std")
        np.testing.assert_allclose(
            reparameterizer.transform_initial_value(prior, physical),
            standardized,
            rtol=1.0e-12,
        )


def test_truncated_lognormal_standardization_is_exact_and_bounded():
    wrapped = reparam(
        _standardizable_prior_model,
        config=normal_lognormal_standardization_reparam,
    )
    prior = TruncatedLogNormal(np.log(3000.0), 0.4, 10000.0)
    reparameterizer = NormalLogNormalStandardizeReparam("physical_std")

    def physical_at(z):
        _, tr = log_density(wrapped, (), {"prior": prior}, {"physical_std": np.asarray(z)})
        return float(tr["physical"]["value"])

    for z in (-2.0, 0.0, 1.5):
        physical = physical_at(z)
        original_density, _ = log_density(
            _standardizable_prior_model, (), {"prior": prior}, {"physical": np.asarray(physical)}
        )
        standardized_density, _ = log_density(
            wrapped, (), {"prior": prior}, {"physical_std": np.asarray(z)}
        )
        # Change of variables: the standard-Normal density equals the prior density times |dx/dz|.
        eps = 1.0e-5
        log_abs_det = np.log((physical_at(z + eps) - physical_at(z - eps)) / (2.0 * eps))
        np.testing.assert_allclose(standardized_density, original_density + log_abs_det, rtol=1.0e-6)
        np.testing.assert_allclose(
            reparameterizer.transform_initial_value(prior, physical), z, rtol=1.0e-6, atol=1.0e-8
        )
    # The standard-Normal axis never maps past the upper limit, and neither does the
    # unconstrained transform used by MAP and NUTS without the reparameterization.
    assert physical_at(8.0) <= 10000.0
    assert float(dist.biject_to(prior.support)(np.asarray(50.0))) <= 10000.0


def test_two_sided_truncated_normal_standardization_is_exact_and_bounded():
    wrapped = reparam(
        _standardizable_prior_model,
        config=normal_lognormal_standardization_reparam,
    )
    prior = dist.TruncatedNormal(0.0, 0.01, low=-0.01, high=0.01)
    reparameterizer = NormalLogNormalStandardizeReparam("physical_std")

    def physical_at(z):
        _, tr = log_density(wrapped, (), {"prior": prior}, {"physical_std": np.asarray(z)})
        return float(tr["physical"]["value"])

    for z in (-1.0, 0.0, 0.7):
        physical = physical_at(z)
        original_density, _ = log_density(
            _standardizable_prior_model, (), {"prior": prior}, {"physical": np.asarray(physical)}
        )
        standardized_density, _ = log_density(
            wrapped, (), {"prior": prior}, {"physical_std": np.asarray(z)}
        )
        eps = 1.0e-5
        log_abs_det = np.log((physical_at(z + eps) - physical_at(z - eps)) / (2.0 * eps))
        np.testing.assert_allclose(standardized_density, original_density + log_abs_det, rtol=1.0e-6)
        np.testing.assert_allclose(
            reparameterizer.transform_initial_value(prior, physical), z, rtol=1.0e-6, atol=1.0e-8
        )
    assert -0.01 <= physical_at(-8.0) and physical_at(8.0) <= 0.01
