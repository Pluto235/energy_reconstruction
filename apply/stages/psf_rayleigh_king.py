#!/usr/bin/env python3
"""Full-sphere Rayleigh-core plus spherical-King-tail PSF fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize


EPSILON = 1.0e-300
SPHERE_MAX_DEG = 180.0


@dataclass(frozen=True)
class RayleighKingFit:
    core_fraction: float
    sigma_rayleigh_deg: float
    sigma_king_deg: float
    gamma_king: float
    r50_deg: float
    r68_deg: float
    r712979_deg: float
    r90_deg: float
    kl_divergence: float
    optimizer_success: bool
    optimizer_message: str
    boundary_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_sphere_edges(edges_deg: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges_deg, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("edges_deg must be a strictly increasing one-dimensional array")
    if not np.isclose(edges[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Full-sphere radial edges must start at 0 degrees")
    if not np.isclose(edges[-1], SPHERE_MAX_DEG, rtol=0.0, atol=1.0e-12):
        raise ValueError("Full-sphere radial edges must end at 180 degrees")
    return edges


def _normalized_probability(mass: np.ndarray) -> np.ndarray:
    probability = np.asarray(mass, dtype=np.float64)
    if probability.ndim != 1 or not np.all(np.isfinite(probability)):
        raise ValueError("Bin masses must be a finite one-dimensional array")
    probability = np.clip(probability, EPSILON, None)
    total = float(np.sum(probability))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Bin masses do not define a finite distribution")
    return probability / total


def _gauss_legendre_sphere_bins(
    edges_deg: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = _validate_sphere_edges(edges_deg)
    if int(quadrature_order) < 4:
        raise ValueError("quadrature_order must be at least 4")
    edges_rad = np.radians(edges)
    roots, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    low = edges_rad[:-1, None]
    high = edges_rad[1:, None]
    nodes_rad = 0.5 * (high - low) * roots[None, :] + 0.5 * (high + low)
    integration_weights = 0.5 * (high - low) * weights[None, :] * np.sin(nodes_rad)
    return nodes_rad, integration_weights


def truncated_rayleigh_cdf(
    radius_deg: np.ndarray | float,
    sigma_deg: float,
) -> np.ndarray:
    """Rayleigh radial CDF conditioned on the physical interval [0, 180 deg]."""
    if sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be positive")
    radius = np.clip(np.asarray(radius_deg, dtype=np.float64), 0.0, SPHERE_MAX_DEG)
    numerator = -np.expm1(-0.5 * (radius / sigma_deg) ** 2)
    denominator = -math.expm1(-0.5 * (SPHERE_MAX_DEG / sigma_deg) ** 2)
    return np.clip(numerator / denominator, 0.0, 1.0)


def truncated_rayleigh_bin_probabilities(
    edges_deg: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    edges = _validate_sphere_edges(edges_deg)
    return _normalized_probability(np.diff(truncated_rayleigh_cdf(edges, sigma_deg)))


def _spherical_king_probability_from_quadrature(
    nodes_rad: np.ndarray,
    integration_weights: np.ndarray,
    sigma_deg: float,
    gamma: float,
) -> np.ndarray:
    if sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be positive")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")
    sigma_rad = math.radians(sigma_deg)
    kernel = (1.0 + nodes_rad**2 / (2.0 * gamma * sigma_rad**2)) ** (-gamma)
    return _normalized_probability(np.sum(integration_weights * kernel, axis=1))


def spherical_king_bin_probabilities(
    edges_deg: np.ndarray,
    sigma_deg: float,
    gamma: float,
    *,
    quadrature_order: int = 20,
) -> np.ndarray:
    """Integrate sin(theta) times the King radial kernel over full-sphere bins."""
    nodes_rad, integration_weights = _gauss_legendre_sphere_bins(edges_deg, quadrature_order)
    return _spherical_king_probability_from_quadrature(
        nodes_rad,
        integration_weights,
        sigma_deg,
        gamma,
    )


def spherical_king_cdf(
    radius_deg: np.ndarray | float,
    sigma_deg: float,
    gamma: float,
    *,
    quadrature_order: int = 20,
) -> np.ndarray:
    """Numerically normalized spherical-King radial CDF on [0, 180 deg]."""
    radius = np.asarray(radius_deg, dtype=np.float64)
    clipped = np.clip(radius, 0.0, SPHERE_MAX_DEG)
    unique_radius = np.unique(np.concatenate(([0.0], clipped.ravel(), [SPHERE_MAX_DEG])))
    probability = spherical_king_bin_probabilities(
        unique_radius,
        sigma_deg,
        gamma,
        quadrature_order=quadrature_order,
    )
    cdf_at_edges = np.concatenate(([0.0], np.cumsum(probability, dtype=np.float64)))
    cdf_at_edges[-1] = 1.0
    indices = np.searchsorted(unique_radius, clipped.ravel(), side="left")
    return cdf_at_edges[indices].reshape(radius.shape)


def rayleigh_king_bin_probabilities(
    edges_deg: np.ndarray,
    core_fraction: float,
    sigma_rayleigh_deg: float,
    sigma_king_deg: float,
    gamma_king: float,
    *,
    quadrature_order: int = 20,
) -> np.ndarray:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_fraction must lie in (0, 1)")
    if not sigma_rayleigh_deg < sigma_king_deg:
        raise ValueError("sigma_rayleigh_deg must be smaller than sigma_king_deg")
    core = truncated_rayleigh_bin_probabilities(edges_deg, sigma_rayleigh_deg)
    tail = spherical_king_bin_probabilities(
        edges_deg,
        sigma_king_deg,
        gamma_king,
        quadrature_order=quadrature_order,
    )
    return core_fraction * core + (1.0 - core_fraction) * tail


def rayleigh_king_cdf(
    radius_deg: np.ndarray | float,
    core_fraction: float,
    sigma_rayleigh_deg: float,
    sigma_king_deg: float,
    gamma_king: float,
    *,
    quadrature_order: int = 20,
) -> np.ndarray:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_fraction must lie in (0, 1)")
    if not sigma_rayleigh_deg < sigma_king_deg:
        raise ValueError("sigma_rayleigh_deg must be smaller than sigma_king_deg")
    core = truncated_rayleigh_cdf(radius_deg, sigma_rayleigh_deg)
    tail = spherical_king_cdf(
        radius_deg,
        sigma_king_deg,
        gamma_king,
        quadrature_order=quadrature_order,
    )
    return np.clip(core_fraction * core + (1.0 - core_fraction) * tail, 0.0, 1.0)


def profile_probability(weighted_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(weighted_counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("weighted_counts must be one-dimensional")
    probability = np.where(np.isfinite(counts) & (counts > 0.0), counts, 0.0)
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("weighted_counts has no positive probability mass")
    return probability / total


def kl_divergence(data_probability: np.ndarray, model_probability: np.ndarray) -> float:
    data = np.asarray(data_probability, dtype=np.float64)
    model = np.asarray(model_probability, dtype=np.float64)
    mask = data > 0.0
    return float(np.sum(data[mask] * np.log(data[mask] / np.clip(model[mask], EPSILON, None))))


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
            options={"maxiter": 3000, "ftol": 1.0e-14, "gtol": 1.0e-9},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Rayleigh+King optimization did not produce a finite result")
    return best


def fit_rayleigh_king_counts(
    weighted_counts: np.ndarray,
    edges_deg: np.ndarray,
    *,
    random_seed: int = 0,
    random_starts: int = 20,
    quadrature_order: int = 20,
) -> tuple[RayleighKingFit, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a full-sphere Rayleigh core plus spherical-King tail mixture."""
    edges = _validate_sphere_edges(edges_deg)
    counts = np.asarray(weighted_counts, dtype=np.float64)
    if counts.shape != (edges.size - 1,):
        raise ValueError("weighted_counts and edges_deg have incompatible shapes")
    data_probability = profile_probability(counts)
    nodes_rad, integration_weights = _gauss_legendre_sphere_bins(edges, quadrature_order)

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float]:
        core_fraction = _logistic(float(parameters[0]))
        sigma_rayleigh_deg = float(np.exp(parameters[1]))
        sigma_king_deg = float(sigma_rayleigh_deg + np.exp(parameters[2]))
        gamma_king = float(1.0 + np.exp(parameters[3]))
        return core_fraction, sigma_rayleigh_deg, sigma_king_deg, gamma_king

    def components(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        core_fraction, sigma_rayleigh_deg, sigma_king_deg, gamma_king = unpack(parameters)
        core = truncated_rayleigh_bin_probabilities(edges, sigma_rayleigh_deg)
        tail = _spherical_king_probability_from_quadrature(
            nodes_rad,
            integration_weights,
            sigma_king_deg,
            gamma_king,
        )
        return core_fraction * core + (1.0 - core_fraction) * tail, core, tail

    def objective(parameters: np.ndarray) -> float:
        model_probability, _, _ = components(parameters)
        return float(-np.sum(data_probability * np.log(np.clip(model_probability, EPSILON, None))))

    starts: list[np.ndarray] = []
    for core_fraction, sigma_rayleigh_deg, sigma_king_deg, gamma_king in (
        (0.55, 0.20, 3.0, 1.2),
        (0.65, 0.30, 8.0, 1.5),
        (0.75, 0.40, 15.0, 2.0),
        (0.85, 0.55, 30.0, 3.0),
        (0.45, 0.15, 50.0, 1.05),
        (0.90, 0.75, 5.0, 6.0),
    ):
        starts.append(
            np.asarray(
                [
                    _logit(core_fraction),
                    math.log(sigma_rayleigh_deg),
                    math.log(sigma_king_deg - sigma_rayleigh_deg),
                    math.log(gamma_king - 1.0),
                ],
                dtype=np.float64,
            )
        )

    rng = np.random.default_rng(random_seed)
    for _ in range(max(0, int(random_starts))):
        sigma_rayleigh_deg = float(np.exp(rng.uniform(math.log(0.03), math.log(2.0))))
        sigma_king_deg = float(np.exp(rng.uniform(math.log(max(0.15, 1.02 * sigma_rayleigh_deg)), math.log(100.0))))
        starts.append(
            np.asarray(
                [
                    _logit(float(rng.uniform(0.15, 0.98))),
                    math.log(sigma_rayleigh_deg),
                    math.log(sigma_king_deg - sigma_rayleigh_deg),
                    rng.uniform(math.log(1.0e-3), math.log(49.0)),
                ],
                dtype=np.float64,
            )
        )

    sigma_rayleigh_bounds = (0.005, 10.0)
    sigma_separation_bounds = (0.002, 180.0)
    gamma_excess_bounds = (1.0e-3, 99.0)
    bounds = [
        (-8.0, 8.0),
        tuple(math.log(value) for value in sigma_rayleigh_bounds),
        tuple(math.log(value) for value in sigma_separation_bounds),
        tuple(math.log(value) for value in gamma_excess_bounds),
    ]
    result = _best_minimize(objective, starts, bounds)
    core_fraction, sigma_rayleigh_deg, sigma_king_deg, gamma_king = unpack(result.x)
    model_probability, core_probability, tail_probability = components(result.x)

    def containment(radius_deg: float) -> float:
        return float(
            rayleigh_king_cdf(
                radius_deg,
                core_fraction,
                sigma_rayleigh_deg,
                sigma_king_deg,
                gamma_king,
                quadrature_order=quadrature_order,
            )
        )

    def quantile(probability: float) -> float:
        return float(brentq(lambda radius: containment(radius) - probability, 0.0, SPHERE_MAX_DEG))

    boundary_flags: list[str] = []
    physical_parameters = (
        ("core_fraction", core_fraction, _logistic(bounds[0][0]), _logistic(bounds[0][1])),
        ("sigma_rayleigh_deg", sigma_rayleigh_deg, *sigma_rayleigh_bounds),
        (
            "sigma_separation_deg",
            sigma_king_deg - sigma_rayleigh_deg,
            *sigma_separation_bounds,
        ),
        ("gamma_excess", gamma_king - 1.0, *gamma_excess_bounds),
    )
    for name, value, lower, upper in physical_parameters:
        scale = max(abs(upper - lower), 1.0)
        if value - lower < 1.0e-4 * scale:
            boundary_flags.append(f"{name}:lower")
        if upper - value < 1.0e-4 * scale:
            boundary_flags.append(f"{name}:upper")

    fit = RayleighKingFit(
        core_fraction=core_fraction,
        sigma_rayleigh_deg=sigma_rayleigh_deg,
        sigma_king_deg=sigma_king_deg,
        gamma_king=gamma_king,
        r50_deg=quantile(0.50),
        r68_deg=quantile(0.68),
        r712979_deg=quantile(1.0 - math.exp(-0.5 * 1.58**2)),
        r90_deg=quantile(0.90),
        kl_divergence=kl_divergence(data_probability, model_probability),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        boundary_flags=tuple(boundary_flags),
    )
    return fit, model_probability, core_probability, tail_probability
