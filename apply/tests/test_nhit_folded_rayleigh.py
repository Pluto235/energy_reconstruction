from __future__ import annotations

import numpy as np

from apply.report.fit_v6_nhit_folded_rayleigh import (
    conditional_rayleigh_probability,
    fit_rayleigh_profile,
    fold_profiles_by_nhit,
)


def test_conditional_rayleigh_fit_recovers_synthetic_sigma() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    target = conditional_rayleigh_probability(edges, sigma_deg=0.43)
    fit, model = fit_rayleigh_profile(target, edges, effective_events=2.0e5)
    assert np.isclose(fit.sigma_deg, 0.43, rtol=0.0, atol=1.0e-7)
    assert fit.kl_divergence < 1.0e-12
    assert fit.multinomial_deviance < 1.0e-6
    assert np.max(np.abs(model - target)) < 1.0e-10
    assert fit.optimizer_success
    assert not fit.boundary_flag


def test_conditional_rayleigh_tail_probabilities_do_not_cancel_to_zero() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    probability = conditional_rayleigh_probability(edges, sigma_deg=0.2)
    assert np.all(probability > 0.0)
    assert probability[-1] < 1.0e-130
    assert np.isclose(np.sum(probability), 1.0, rtol=0.0, atol=1.0e-14)


def test_fold_uses_unequal_baseline_weights_and_excludes_empty_profiles() -> None:
    edges = np.asarray([0.0, 1.0, 2.0])
    density = np.asarray(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.0, 0.0],
            [0.3, 0.7],
        ]
    )
    folded = fold_profiles_by_nhit(
        cell_id=np.asarray([1, 2, 3, 4]),
        nhit_bin=np.asarray(["[100,200)", "[100,200)", "[100,200)", "[200,300)"]),
        profile_density=density,
        profile_edges_deg=edges,
        sumw_baseline=np.asarray([1.0, 3.0, 9.0, 2.0]),
        effective_events=np.asarray([10.0, 30.0, 90.0, 20.0]),
    )
    assert len(folded) == 2
    assert folded[0].used_cell_ids == (1, 2)
    assert folded[0].excluded_cell_ids == (3,)
    assert np.allclose(folded[0].probability, np.asarray([0.275, 0.725]))
    assert np.isclose(folded[0].effective_events, 40.0)
    assert np.isclose(np.sum(folded[1].probability), 1.0)


def test_folded_probability_and_survival_are_normalized_and_monotonic() -> None:
    edges = np.linspace(0.0, 5.0, 101)
    p1 = conditional_rayleigh_probability(edges, sigma_deg=0.3)
    p2 = conditional_rayleigh_probability(edges, sigma_deg=0.9)
    folded = fold_profiles_by_nhit(
        cell_id=np.asarray([1, 2]),
        nhit_bin=np.asarray(["[100,200)", "[100,200)"]),
        profile_density=np.vstack((p1, p2)) / np.diff(edges),
        profile_edges_deg=edges,
        sumw_baseline=np.asarray([2.0, 1.0]),
        effective_events=np.asarray([1000.0, 500.0]),
    )[0]
    survival = 1.0 - np.concatenate(([0.0], np.cumsum(folded.probability)))
    assert np.isclose(np.sum(folded.probability), 1.0)
    assert np.isclose(survival[0], 1.0)
    assert np.isclose(survival[-1], 0.0, atol=1.0e-14)
    assert np.all(np.diff(survival) <= 1.0e-14)
