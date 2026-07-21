from __future__ import annotations

import numpy as np

from apply.report.compare_v6_nhit_folded_rayleigh_models import (
    conditional_double_rayleigh_probability,
    fit_conditional_double_rayleigh,
)


def test_conditional_double_rayleigh_is_positive_and_normalized() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    probability = conditional_double_rayleigh_probability(edges, 0.82, 0.24, 1.3)
    assert np.all(probability > 0.0)
    assert np.isclose(np.sum(probability), 1.0, rtol=0.0, atol=1.0e-14)


def test_synthetic_conditional_double_rayleigh_fit_recovers_shape() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    target = conditional_double_rayleigh_probability(edges, 0.76, 0.21, 1.15)
    fit, model = fit_conditional_double_rayleigh(
        target,
        edges,
        effective_events=2.0e5,
        one_rayleigh_sigma_deg=0.5,
        random_seed=17,
        random_starts=12,
    )
    assert fit.optimizer_success
    assert not fit.boundary_flags
    assert fit.kl_divergence < 1.0e-10
    assert np.max(np.abs(model - target)) < 1.0e-6
    assert np.isclose(fit.conditional_core_fraction, 0.76, atol=2.0e-3)
    assert np.isclose(fit.sigma_core_deg, 0.21, rtol=5.0e-3)
    assert np.isclose(fit.sigma_tail_deg, 1.15, rtol=5.0e-3)


def test_double_rayleigh_tail_bins_remain_finite() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    probability = conditional_double_rayleigh_probability(edges, 0.995, 0.08, 0.25)
    assert np.all(np.isfinite(probability))
    assert np.all(probability > 0.0)
    assert probability[-1] > 0.0
