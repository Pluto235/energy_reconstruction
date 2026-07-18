from __future__ import annotations

import numpy as np

from apply.stages.psf_rayleigh_king import (
    fit_rayleigh_king_counts,
    rayleigh_king_bin_probabilities,
    rayleigh_king_cdf,
    spherical_king_bin_probabilities,
    truncated_rayleigh_bin_probabilities,
)


def test_component_and_mixture_probabilities_cover_full_sphere() -> None:
    edges = np.concatenate(
        (
            np.linspace(0.0, 5.0, 101),
            np.linspace(5.25, 20.0, 60),
            np.linspace(21.0, 180.0, 160),
        )
    )
    core = truncated_rayleigh_bin_probabilities(edges, sigma_deg=0.35)
    tail = spherical_king_bin_probabilities(edges, sigma_deg=9.0, gamma=1.7)
    mixture = rayleigh_king_bin_probabilities(
        edges,
        core_fraction=0.72,
        sigma_rayleigh_deg=0.35,
        sigma_king_deg=9.0,
        gamma_king=1.7,
    )
    for probability in (core, tail, mixture):
        assert np.all(probability > 0.0)
        assert np.isclose(np.sum(probability), 1.0, rtol=0.0, atol=1.0e-12)


def test_synthetic_rayleigh_king_fit_recovers_shape() -> None:
    edges = np.concatenate(
        (
            np.linspace(0.0, 5.0, 101),
            np.linspace(5.25, 20.0, 60),
            np.linspace(21.0, 180.0, 160),
        )
    )
    target = rayleigh_king_bin_probabilities(
        edges,
        core_fraction=0.68,
        sigma_rayleigh_deg=0.32,
        sigma_king_deg=11.0,
        gamma_king=1.8,
    )
    fit, model, _, _ = fit_rayleigh_king_counts(
        target * 1.0e6,
        edges,
        random_seed=23,
        random_starts=4,
    )
    assert fit.kl_divergence < 1.0e-8
    assert np.max(np.abs(model - target)) < 1.0e-5
    assert np.isclose(fit.core_fraction, 0.68, atol=5.0e-3)
    assert np.isclose(fit.sigma_rayleigh_deg, 0.32, rtol=2.0e-2)


def test_rayleigh_king_survival_is_normalized_and_monotonic() -> None:
    radius = np.linspace(0.0, 180.0, 1001)
    cdf = rayleigh_king_cdf(
        radius,
        core_fraction=0.74,
        sigma_rayleigh_deg=0.38,
        sigma_king_deg=14.0,
        gamma_king=1.4,
    )
    survival = 1.0 - cdf
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert survival[0] == 1.0
    assert survival[-1] == 0.0
    assert np.all(np.diff(cdf) >= -1.0e-14)
    assert np.all(np.diff(survival) <= 1.0e-14)
