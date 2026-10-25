"""Cached Balmer-continuum and Fe II template inputs must not change the model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsedfit.spectral_model import _balmer_continuum_jax, _balmer_static_terms_jax, _fe_template_component

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("wave", [jnp.linspace(2500.0, 4500.0, 512), jnp.geomspace(2000.0, 6000.0, 3000)])
def test_balmer_continuum_cached_static_terms_match_uncached(wave):
    kwargs = dict(balmer_norm=2.3, balmer_te=15000.0, balmer_tau=0.7, balmer_vel=3200.0, convolution_method="fft")
    uncached = _balmer_continuum_jax(wave, **kwargs)
    cached = _balmer_continuum_jax(wave, balmer_static_terms=_balmer_static_terms_jax(wave, balmer_te=15000.0), **kwargs)
    np.testing.assert_allclose(np.asarray(cached), np.asarray(uncached), rtol=1e-12, atol=1e-12)


def test_fe_template_cached_template_on_wave_matches_uncached():
    wave = jnp.linspace(1900.0, 3100.0, 256)
    wave_template = jnp.linspace(1800.0, 3200.0, 192)
    flux_template = jnp.exp(-0.5 * ((wave_template - 2500.0) / 95.0) ** 2)
    kwargs = dict(norm=1.7, fwhm_kms=3500.0, shift_frac=0.002, base_fwhm_kms=900.0, convolution_method="fft")
    uncached = _fe_template_component(wave, wave_template, flux_template, **kwargs)
    cached = _fe_template_component(
        wave, wave_template, flux_template,
        template_on_wave=jnp.interp(wave, wave_template, flux_template, left=0.0, right=0.0), **kwargs,
    )
    np.testing.assert_allclose(np.asarray(cached), np.asarray(uncached), rtol=1e-12, atol=1e-12)
