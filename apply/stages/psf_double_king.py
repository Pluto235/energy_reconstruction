#!/usr/bin/env python3
"""Fermi-style double-King PSF math and binned profile fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize


EPSILON = 1.0e-300


@dataclass(frozen=True)
class DoubleKingFit:
    conditional_core_fraction: float
    physical_core_fraction: float
    sigma_core_deg: float
    gamma_core: float
    sigma_tail_deg: float
    gamma_tail: float
    conditional_r_target_deg: float
    kl_divergence: float
    optimizer_success: bool
    optimizer_message: str
    boundary_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def king_cdf(radius_deg: np.ndarray | float, sigma_deg: float, gamma: float) -> np.ndarray:
    """Circular King containment, normalized over the full plane."""
    radius = np.asarray(radius_deg, dtype=np.float64)
    if sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be positive")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")
    base = 1.0 + radius**2 / (2.0 * gamma * sigma_deg**2)
    return 1.0 - base ** (1.0 - gamma)


def rayleigh_cdf(radius_deg: np.ndarray | float, sigma_deg: float) -> np.ndarray:
    radius = np.asarray(radius_deg, dtype=np.float64)
    if sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be positive")
    return 1.0 - np.exp(-0.5 * (radius / sigma_deg) ** 2)


def normalized_bin_probabilities(cdf_at_edges: np.ndarray) -> np.ndarray:
    probability = np.diff(np.asarray(cdf_at_edges, dtype=np.float64))
    probability = np.clip(probability, EPSILON, None)
    total = float(np.sum(probability))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("CDF differences do not define a finite distribution")
    return probability / total


def king_bin_probabilities(edges_deg: np.ndarray, sigma_deg: float, gamma: float) -> np.ndarray:
    return normalized_bin_probabilities(king_cdf(edges_deg, sigma_deg, gamma))


def rayleigh_bin_probabilities(edges_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    return normalized_bin_probabilities(rayleigh_cdf(edges_deg, sigma_deg))


def conditional_double_king_bin_probabilities(
    edges_deg: np.ndarray,
    conditional_core_fraction: float,
    sigma_core_deg: float,
    gamma_core: float,
    sigma_tail_deg: float,
    gamma_tail: float,
) -> np.ndarray:
    """Bin probabilities conditional on the supplied radial fit window."""
    if not 0.0 < conditional_core_fraction < 1.0:
        raise ValueError("conditional_core_fraction must lie in (0, 1)")
    if not sigma_core_deg < sigma_tail_deg:
        raise ValueError("sigma_core_deg must be smaller than sigma_tail_deg")
    core = king_bin_probabilities(edges_deg, sigma_core_deg, gamma_core)
    tail = king_bin_probabilities(edges_deg, sigma_tail_deg, gamma_tail)
    return conditional_core_fraction * core + (1.0 - conditional_core_fraction) * tail


def conditional_double_king_cdf(
    radius_deg: np.ndarray | float,
    max_radius_deg: float,
    conditional_core_fraction: float,
    sigma_core_deg: float,
    gamma_core: float,
    sigma_tail_deg: float,
    gamma_tail: float,
) -> np.ndarray:
    """Double-King containment conditional on 0 <= r <= max_radius_deg."""
    radius = np.asarray(radius_deg, dtype=np.float64)
    core_norm = float(king_cdf(max_radius_deg, sigma_core_deg, gamma_core))
    tail_norm = float(king_cdf(max_radius_deg, sigma_tail_deg, gamma_tail))
    if core_norm <= 0.0 or tail_norm <= 0.0:
        raise ValueError("King components have zero mass in the fit window")
    core = king_cdf(radius, sigma_core_deg, gamma_core) / core_norm
    tail = king_cdf(radius, sigma_tail_deg, gamma_tail) / tail_norm
    return conditional_core_fraction * core + (1.0 - conditional_core_fraction) * tail


def physical_core_fraction_from_conditional(
    conditional_core_fraction: float,
    max_radius_deg: float,
    sigma_core_deg: float,
    gamma_core: float,
    sigma_tail_deg: float,
    gamma_tail: float,
) -> float:
    """Map the fit-window mixture weight to the full-plane Fermi mixture weight."""
    core_mass = float(king_cdf(max_radius_deg, sigma_core_deg, gamma_core))
    tail_mass = float(king_cdf(max_radius_deg, sigma_tail_deg, gamma_tail))
    numerator = conditional_core_fraction * tail_mass
    denominator = numerator + (1.0 - conditional_core_fraction) * core_mass
    if denominator <= 0.0:
        raise ValueError("Cannot map conditional mixture weight to the full plane")
    return numerator / denominator


def profile_probability(profile_density: np.ndarray, edges_deg: np.ndarray) -> np.ndarray:
    density = np.asarray(profile_density, dtype=np.float64)
    edges = np.asarray(edges_deg, dtype=np.float64)
    if density.shape != (edges.size - 1,):
        raise ValueError("profile_density and edges_deg have incompatible shapes")
    probability = np.where(np.isfinite(density) & (density > 0.0), density, 0.0) * np.diff(edges)
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("profile_density has no positive probability mass")
    return probability / total


def kl_divergence(data_probability: np.ndarray, model_probability: np.ndarray) -> float:
    data = np.asarray(data_probability, dtype=np.float64)
    model = np.asarray(model_probability, dtype=np.float64)
    mask = data > 0.0
    return float(np.sum(data[mask] * np.log(data[mask] / np.clip(model[mask], EPSILON, None))))


def _cross_entropy(data_probability: np.ndarray, model_probability: np.ndarray) -> float:
    return float(-np.sum(data_probability * np.log(np.clip(model_probability, EPSILON, None))))


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _logistic(value: float) -> float:
    if value >= 0.0:
        exp_minus = math.exp(-value)
        return 1.0 / (1.0 + exp_minus)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _best_minimize(
    objective: Callable[[np.ndarray], float],
    starts: Sequence[np.ndarray],
    bounds: Sequence[tuple[float, float]],
) -> object:
    best = None
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=list(bounds),
            options={"maxiter": 4000, "ftol": 1.0e-14, "gtol": 1.0e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Double-King optimization did not produce a finite result")
    return best


def fit_double_king_profile(
    profile_density: np.ndarray,
    edges_deg: np.ndarray,
    *,
    target_containment: float,
    gamma_min: float = 1.05,
    random_seed: int = 0,
    random_starts: int = 16,
) -> tuple[DoubleKingFit, np.ndarray]:
    """Fit a two-component King mixture to a normalized radial profile.

    The likelihood is conditional on the finite radial range in ``edges_deg``.
    This makes the plotted shape well-defined without pretending that the input
    NPZ retained the probability mass outside its 0-5 degree profile window.
    """
    edges = np.asarray(edges_deg, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("edges_deg must be a strictly increasing one-dimensional array")
    if not 0.0 < target_containment < 1.0:
        raise ValueError("target_containment must lie in (0, 1)")
    if not 1.0 < gamma_min < 10.0:
        raise ValueError("gamma_min must be between 1 and 10")

    data_probability = profile_probability(profile_density, edges)

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float, float]:
        conditional_core_fraction = _logistic(float(parameters[0]))
        sigma_core_deg = float(np.exp(parameters[1]))
        sigma_tail_deg = float(sigma_core_deg * np.exp(parameters[2]))
        gamma_core = float(1.0 + np.exp(parameters[3]))
        gamma_tail = float(1.0 + np.exp(parameters[4]))
        return (
            conditional_core_fraction,
            sigma_core_deg,
            gamma_core,
            sigma_tail_deg,
            gamma_tail,
        )

    def model_probability(parameters: np.ndarray) -> np.ndarray:
        return conditional_double_king_bin_probabilities(edges, *unpack(parameters))

    def objective(parameters: np.ndarray) -> float:
        return _cross_entropy(data_probability, model_probability(parameters))

    starts: list[np.ndarray] = []
    for conditional_core_fraction, sigma_core_deg, width_ratio, gamma_core, gamma_tail in (
        (0.80, 0.18, 2.5, 3.0, 1.5),
        (0.90, 0.25, 3.0, 4.0, 1.8),
        (0.95, 0.35, 4.0, 3.0, 1.3),
        (0.98, 0.45, 6.0, 6.0, 2.0),
        (0.70, 0.12, 2.0, 2.0, 1.2),
        (0.90, 0.55, 2.0, 8.0, 3.0),
    ):
        starts.append(
            np.asarray(
                [
                    _logit(conditional_core_fraction),
                    math.log(sigma_core_deg),
                    math.log(width_ratio),
                    math.log(gamma_core - 1.0),
                    math.log(gamma_tail - 1.0),
                ],
                dtype=np.float64,
            )
        )

    rng = np.random.default_rng(random_seed)
    for _ in range(max(0, int(random_starts))):
        conditional_core_fraction = float(rng.uniform(0.20, 0.995))
        starts.append(
            np.asarray(
                [
                    _logit(conditional_core_fraction),
                    rng.uniform(math.log(0.03), math.log(1.0)),
                    rng.uniform(math.log(1.02), math.log(20.0)),
                    rng.uniform(math.log(gamma_min - 1.0), math.log(30.0)),
                    rng.uniform(math.log(gamma_min - 1.0), math.log(15.0)),
                ],
                dtype=np.float64,
            )
        )

    sigma_core_bounds = (0.01, 2.0)
    width_ratio_bounds = (1.01, 50.0)
    gamma_max = 100.0
    bounds = [
        (-7.0, 7.0),
        (math.log(sigma_core_bounds[0]), math.log(sigma_core_bounds[1])),
        (math.log(width_ratio_bounds[0]), math.log(width_ratio_bounds[1])),
        (math.log(gamma_min - 1.0), math.log(gamma_max - 1.0)),
        (math.log(gamma_min - 1.0), math.log(gamma_max - 1.0)),
    ]
    result = _best_minimize(objective, starts, bounds)
    (
        conditional_core_fraction,
        sigma_core_deg,
        gamma_core,
        sigma_tail_deg,
        gamma_tail,
    ) = unpack(result.x)
    fitted_probability = model_probability(result.x)
    max_radius_deg = float(edges[-1])

    physical_core_fraction = physical_core_fraction_from_conditional(
        conditional_core_fraction,
        max_radius_deg,
        sigma_core_deg,
        gamma_core,
        sigma_tail_deg,
        gamma_tail,
    )
    containment_function = lambda radius: float(
        conditional_double_king_cdf(
            radius,
            max_radius_deg,
            conditional_core_fraction,
            sigma_core_deg,
            gamma_core,
            sigma_tail_deg,
            gamma_tail,
        )
        - target_containment
    )
    conditional_r_target_deg = float(brentq(containment_function, 0.0, max_radius_deg))

    width_ratio = sigma_tail_deg / sigma_core_deg
    boundary_flags: list[str] = []
    if conditional_core_fraction < 0.002 or conditional_core_fraction > 0.998:
        boundary_flags.append("conditional_core_fraction")
    if sigma_core_deg <= 1.001 * sigma_core_bounds[0] or sigma_core_deg >= 0.999 * sigma_core_bounds[1]:
        boundary_flags.append("sigma_core")
    if width_ratio <= 1.001 * width_ratio_bounds[0] or width_ratio >= 0.999 * width_ratio_bounds[1]:
        boundary_flags.append("sigma_ratio")
    if gamma_core <= gamma_min + 1.0e-3 or gamma_core >= gamma_max - 1.0e-2:
        boundary_flags.append("gamma_core")
    if gamma_tail <= gamma_min + 1.0e-3 or gamma_tail >= gamma_max - 1.0e-2:
        boundary_flags.append("gamma_tail")

    fit = DoubleKingFit(
        conditional_core_fraction=conditional_core_fraction,
        physical_core_fraction=physical_core_fraction,
        sigma_core_deg=sigma_core_deg,
        gamma_core=gamma_core,
        sigma_tail_deg=sigma_tail_deg,
        gamma_tail=gamma_tail,
        conditional_r_target_deg=conditional_r_target_deg,
        kl_divergence=kl_divergence(data_probability, fitted_probability),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        boundary_flags=tuple(boundary_flags),
    )
    return fit, fitted_probability / np.diff(edges)
