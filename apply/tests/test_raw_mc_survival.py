from __future__ import annotations

import numpy as np

from apply.report.plot_v6_cell123_raw_mc_survival import (
    empirical_survival,
    weighted_quantiles,
)


def test_empirical_survival_uses_strict_greater_than_and_all_event_denominator() -> None:
    radius = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    weight = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
    grid = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    survival = empirical_survival(radius, weight, grid)
    assert np.allclose(survival, [1.0, 0.75, 0.25, 0.0, 0.0])


def test_weighted_quantiles_follow_weighted_empirical_cdf() -> None:
    radius = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    weight = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
    quantiles = weighted_quantiles(radius, weight, np.asarray([0.25, 0.50, 0.75, 1.0]))
    assert np.allclose(quantiles, [1.0, 1.5, 2.0, 3.0])
