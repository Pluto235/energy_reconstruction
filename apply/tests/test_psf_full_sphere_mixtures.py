from __future__ import annotations

import numpy as np

from apply.stages.psf_full_sphere_mixtures import (
    double_rayleigh_bin_probabilities,
    double_rayleigh_cdf,
    double_spherical_king_bin_probabilities,
    double_spherical_king_cdf,
    fit_double_rayleigh_counts,
    fit_double_spherical_king_counts,
)
from apply.stages.psf_rayleigh_king import spherical_king_bin_probabilities


def full_sphere_edges() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(0.0, 5.0, 101),
            np.linspace(5.25, 20.0, 60),
            np.linspace(21.0, 180.0, 160),
        )
    )


def test_full_sphere_mixture_probabilities_are_positive_and_normalized() -> None:
    edges = full_sphere_edges()
    double_rayleigh = double_rayleigh_bin_probabilities(edges, 0.72, 0.35, 18.0)
    double_king = double_spherical_king_bin_probabilities(edges, 0.70, 0.28, 8.0, 9.0, 1.5)
    for probability in (double_rayleigh, double_king):
        assert np.all(probability > 0.0)
        assert np.isclose(np.sum(probability), 1.0, rtol=0.0, atol=1.0e-12)


def test_synthetic_double_rayleigh_fit_recovers_shape() -> None:
    edges = full_sphere_edges()
    target = double_rayleigh_bin_probabilities(edges, 0.68, 0.32, 14.0)
    fit, model = fit_double_rayleigh_counts(
        target * 1.0e6,
        edges,
        random_seed=31,
        random_starts=4,
    )
    assert fit.kl_divergence < 1.0e-9
    assert np.max(np.abs(model - target)) < 1.0e-5


def test_synthetic_double_spherical_king_fit_recovers_shape() -> None:
    edges = full_sphere_edges()
    target = double_spherical_king_bin_probabilities(edges, 0.66, 0.30, 8.0, 10.0, 1.7)
    fit, model = fit_double_spherical_king_counts(
        target * 1.0e6,
        edges,
        random_seed=41,
        random_starts=6,
    )
    assert fit.kl_divergence < 1.0e-7
    assert np.max(np.abs(model - target)) < 2.0e-5


def test_full_sphere_survival_functions_are_normalized_and_monotonic() -> None:
    radius = np.linspace(0.0, 180.0, 1001)
    cdfs = (
        double_rayleigh_cdf(radius, 0.72, 0.35, 18.0),
        double_spherical_king_cdf(radius, 0.70, 0.28, 8.0, 9.0, 1.5),
    )
    for cdf in cdfs:
        survival = 1.0 - cdf
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0
        assert survival[0] == 1.0
        assert survival[-1] == 0.0
        assert np.all(np.diff(cdf) >= -1.0e-14)
        assert np.all(np.diff(survival) <= 1.0e-14)


def test_weighted_double_king_components_sum_to_total() -> None:
    edges = full_sphere_edges()
    core_fraction = 0.73
    core = spherical_king_bin_probabilities(edges, 0.31, 3.2)
    tail = spherical_king_bin_probabilities(edges, 16.0, 4.5)
    components = core_fraction * core + (1.0 - core_fraction) * tail
    total = double_spherical_king_bin_probabilities(edges, core_fraction, 0.31, 3.2, 16.0, 4.5)
    assert np.allclose(components, total, rtol=0.0, atol=1.0e-13)
