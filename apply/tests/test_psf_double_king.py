from __future__ import annotations

import numpy as np

from apply.stages.psf_double_king import (
    conditional_double_king_bin_probabilities,
    conditional_double_king_cdf,
    fit_double_king_profile,
    king_cdf,
    rayleigh_cdf,
)


def test_king_cdf_is_normalized_and_monotonic() -> None:
    radius = np.linspace(0.0, 1000.0, 10001)
    cdf = king_cdf(radius, sigma_deg=0.35, gamma=2.5)
    assert cdf[0] == 0.0
    assert np.all(np.diff(cdf) >= 0.0)
    assert cdf[-1] > 0.999


def test_king_converges_to_rayleigh_at_large_gamma() -> None:
    radius = np.linspace(0.0, 3.0, 1001)
    king = king_cdf(radius, sigma_deg=0.4, gamma=1.0e6)
    rayleigh = rayleigh_cdf(radius, sigma_deg=0.4)
    assert np.max(np.abs(king - rayleigh)) < 1.0e-6


def test_double_king_bin_probabilities_sum_to_one() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    probability = conditional_double_king_bin_probabilities(
        edges,
        conditional_core_fraction=0.93,
        sigma_core_deg=0.31,
        gamma_core=3.8,
        sigma_tail_deg=1.2,
        gamma_tail=1.4,
    )
    assert np.all(probability > 0.0)
    assert np.isclose(np.sum(probability), 1.0, rtol=0.0, atol=1.0e-13)


def test_synthetic_double_king_shape_fit_and_containment() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    widths = np.diff(edges)
    target_probability = conditional_double_king_bin_probabilities(
        edges,
        conditional_core_fraction=0.92,
        sigma_core_deg=0.30,
        gamma_core=4.0,
        sigma_tail_deg=1.1,
        gamma_tail=1.6,
    )
    fit, model_density = fit_double_king_profile(
        target_probability / widths,
        edges,
        target_containment=0.7129790300890827,
        gamma_min=1.05,
        random_seed=17,
        random_starts=4,
    )
    assert fit.kl_divergence < 1.0e-8
    assert np.isclose(np.sum(model_density * widths), 1.0, rtol=0.0, atol=1.0e-12)
    fitted_containment = conditional_double_king_cdf(
        fit.conditional_r_target_deg,
        max_radius_deg=edges[-1],
        conditional_core_fraction=fit.conditional_core_fraction,
        sigma_core_deg=fit.sigma_core_deg,
        gamma_core=fit.gamma_core,
        sigma_tail_deg=fit.sigma_tail_deg,
        gamma_tail=fit.gamma_tail,
    )
    assert np.isclose(fitted_containment, 0.7129790300890827, rtol=0.0, atol=1.0e-10)
