from __future__ import annotations

import numpy as np

from apply.stages.empirical_cdf_asimov import (
    asimov_significance,
    integrate_centered_disk_density,
    radius_grid,
    select_smallest_near_maximum,
    signal_counts_from_numerator,
)


def test_radius_grid_inclusive() -> None:
    values = radius_grid(0.2, 2.0, 0.01)
    assert values.size == 181
    assert values[0] == 0.2
    assert values[-1] == 2.0


def test_asimov_significance_matches_weak_signal_limit() -> None:
    signal = np.asarray([1.0])
    background = np.asarray([1.0e6])
    value = asimov_significance(signal, background)[0]
    assert np.isclose(value, signal[0] / np.sqrt(background[0]), rtol=2.0e-6)


def test_centered_disk_integral_removes_odd_terms() -> None:
    coefficients = np.asarray([2.0, 9.0, -4.0, 0.3, 7.0, -0.1])
    radius = np.asarray([0.5, 1.0])
    expected = np.pi * radius**2 * 2.0 + np.pi * radius**4 * 0.2 / 4.0
    assert np.allclose(integrate_centered_disk_density(coefficients, radius), expected)


def test_smallest_radius_on_99_percent_plateau() -> None:
    radii = np.asarray([0.2, 0.3, 0.4, 0.5])
    objective = np.asarray([7.0, 9.91, 10.0, 9.95])
    adopted, exact = select_smallest_near_maximum(radii, objective, 0.99)
    assert adopted == 1
    assert exact == 2


def test_signal_counts_from_numerator() -> None:
    numerator = np.asarray([[[1.0]], [[2.0]]])
    denominator = np.asarray([[4.0]])
    counts = signal_counts_from_numerator(
        numerator,
        denominator,
        np.asarray([0.0, 1.0]),
        4.0,
        np.asarray([2.0]),
        np.asarray([3.0]),
    )
    cosine = np.cos(np.radians(0.5))
    assert np.allclose(counts, 1.0e4 * np.asarray([6.0, 12.0]) * cosine)
