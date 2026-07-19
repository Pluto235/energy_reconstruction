#!/usr/bin/env python3
"""Full-sphere double-Rayleigh and double-spherical-King PSF mixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from apply.stages.psf_rayleigh_king import (
    EPSILON,
    SPHERE_MAX_DEG,
    kl_divergence,
    profile_probability,
    spherical_king_cdf,
    truncated_rayleigh_bin_probabilities,
    truncated_rayleigh_cdf,
)


@dataclass(frozen=True)
class DoubleRayleighFit:
    core_fraction: float
    sigma_core_deg: float
    sigma_tail_deg: float
    kl_divergence: float
    optimizer_success: bool
    optimizer_message: str
    boundary_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DoubleSphericalKingFit:
    core_fraction: float
    sigma_core_deg: float
    gamma_core: float
    sigma_tail_deg: float
    gamma_tail: float
    kl_divergence: float
    optimizer_success: bool
    optimizer_message: str
    boundary_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_edges(edges_deg: np.ndarray) -> np.ndarray:
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
    probability = np.clip(probability, EPSILON, None)
    total = float(np.sum(probability))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Bin masses do not define a finite distribution")
    return probability / total


def _gauss_legendre_sphere_bins(
    edges_deg: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = _validate_edges(edges_deg)
    if int(quadrature_order) < 4:
        raise ValueError("quadrature_order must be at least 4")
    edges_rad = np.radians(edges)
    roots, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    low = edges_rad[:-1, None]
    high = edges_rad[1:, None]
    nodes_rad = 0.5 * (high - low) * roots[None, :] + 0.5 * (high + low)
    integration_weights = 0.5 * (high - low) * weights[None, :] * np.sin(nodes_rad)
    return nodes_rad, integration_weights


def _spherical_king_probability(
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


def double_rayleigh_bin_probabilities(
    edges_deg: np.ndarray,
    core_fraction: float,
    sigma_core_deg: float,
    sigma_tail_deg: float,
) -> np.ndarray:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_fraction must lie in (0, 1)")
    if not 0.0 < sigma_core_deg < sigma_tail_deg:
        raise ValueError("Require 0 < sigma_core_deg < sigma_tail_deg")
    core = truncated_rayleigh_bin_probabilities(edges_deg, sigma_core_deg)
    tail = truncated_rayleigh_bin_probabilities(edges_deg, sigma_tail_deg)
    return core_fraction * core + (1.0 - core_fraction) * tail


def double_rayleigh_cdf(
    radius_deg: np.ndarray | float,
    core_fraction: float,
    sigma_core_deg: float,
    sigma_tail_deg: float,
) -> np.ndarray:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_fraction must lie in (0, 1)")
    if not 0.0 < sigma_core_deg < sigma_tail_deg:
        raise ValueError("Require 0 < sigma_core_deg < sigma_tail_deg")
    core = truncated_rayleigh_cdf(radius_deg, sigma_core_deg)
    tail = truncated_rayleigh_cdf(radius_deg, sigma_tail_deg)
    return np.clip(core_fraction * core + (1.0 - core_fraction) * tail, 0.0, 1.0)


def double_spherical_king_bin_probabilities(
    edges_deg: np.ndarray,
    core_fraction: float,
    sigma_core_deg: float,
    gamma_core: float,
    sigma_tail_deg: float,
    gamma_tail: float,
    *,
    quadrature_order: int = 20,
) -> np.ndarray:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_fraction must lie in (0, 1)")
    if not 0.0 < sigma_core_deg < sigma_tail_deg:
        raise ValueError("Require 0 < sigma_core_deg < sigma_tail_deg")
    nodes_rad, integration_weights = _gauss_legendre_sphere_bins(edges_deg, quadrature_order)
    core = _spherical_king_probability(
        nodes_rad,
        integration_weights,
        sigma_core_deg,
        gamma_core,
    )
    tail = _spherical_king_probability(
        nodes_rad,
        integration_weights,
        sigma_tail_deg,
        gamma_tail,
    )
    return core_fraction * core + (1.0 - core_fraction) * tail


def double_spherical_king_cdf(
    radius_deg: np.ndarray | float,
    core_fraction: float,
    sigma_core_deg: float,
    gamma_core: float,
    sigma_tail_deg: float,
    gamma_tail: float,
    *,
    quadrature_order: int = 20,
) -> np.ndarray:
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_fraction must lie in (0, 1)")
    if not 0.0 < sigma_core_deg < sigma_tail_deg:
        raise ValueError("Require 0 < sigma_core_deg < sigma_tail_deg")
    core = spherical_king_cdf(
        radius_deg,
        sigma_core_deg,
        gamma_core,
        quadrature_order=quadrature_order,
    )
    tail = spherical_king_cdf(
        radius_deg,
        sigma_tail_deg,
        gamma_tail,
        quadrature_order=quadrature_order,
    )
    return np.clip(core_fraction * core + (1.0 - core_fraction) * tail, 0.0, 1.0)


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
            options={"maxiter": 4000, "ftol": 1.0e-14, "gtol": 1.0e-9},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Mixture optimization did not produce a finite result")
    return best


def _boundary_flags(
    physical_parameters: Sequence[tuple[str, float, float, float]],
) -> tuple[str, ...]:
    flags: list[str] = []
    for name, value, lower, upper in physical_parameters:
        scale = max(abs(upper - lower), 1.0)
        if value - lower < 1.0e-4 * scale:
            flags.append(f"{name}:lower")
        if upper - value < 1.0e-4 * scale:
            flags.append(f"{name}:upper")
    return tuple(flags)


def fit_double_rayleigh_counts(
    weighted_counts: np.ndarray,
    edges_deg: np.ndarray,
    *,
    random_seed: int = 0,
    random_starts: int = 20,
) -> tuple[DoubleRayleighFit, np.ndarray]:
    edges = _validate_edges(edges_deg)
    counts = np.asarray(weighted_counts, dtype=np.float64)
    if counts.shape != (edges.size - 1,):
        raise ValueError("weighted_counts and edges_deg have incompatible shapes")
    data_probability = profile_probability(counts)

    def unpack(parameters: np.ndarray) -> tuple[float, float, float]:
        core_fraction = _logistic(float(parameters[0]))
        sigma_core_deg = float(np.exp(parameters[1]))
        sigma_tail_deg = float(sigma_core_deg + np.exp(parameters[2]))
        return core_fraction, sigma_core_deg, sigma_tail_deg

    def model(parameters: np.ndarray) -> np.ndarray:
        return double_rayleigh_bin_probabilities(edges, *unpack(parameters))

    def objective(parameters: np.ndarray) -> float:
        probability = model(parameters)
        return float(-np.sum(data_probability * np.log(np.clip(probability, EPSILON, None))))

    starts: list[np.ndarray] = []
    for core_fraction, sigma_core_deg, sigma_tail_deg in (
        (0.50, 0.20, 3.0),
        (0.65, 0.30, 8.0),
        (0.75, 0.40, 20.0),
        (0.85, 0.55, 40.0),
        (0.90, 0.75, 70.0),
    ):
        starts.append(
            np.asarray(
                [
                    _logit(core_fraction),
                    math.log(sigma_core_deg),
                    math.log(sigma_tail_deg - sigma_core_deg),
                ],
                dtype=np.float64,
            )
        )
    rng = np.random.default_rng(random_seed)
    for _ in range(max(0, int(random_starts))):
        sigma_core_deg = float(np.exp(rng.uniform(math.log(0.03), math.log(2.0))))
        sigma_tail_deg = float(
            np.exp(rng.uniform(math.log(max(0.15, 1.02 * sigma_core_deg)), math.log(120.0)))
        )
        starts.append(
            np.asarray(
                [
                    _logit(float(rng.uniform(0.10, 0.98))),
                    math.log(sigma_core_deg),
                    math.log(sigma_tail_deg - sigma_core_deg),
                ],
                dtype=np.float64,
            )
        )

    sigma_core_bounds = (0.005, 10.0)
    sigma_separation_bounds = (0.002, 180.0)
    bounds = [
        (-8.0, 8.0),
        tuple(math.log(value) for value in sigma_core_bounds),
        tuple(math.log(value) for value in sigma_separation_bounds),
    ]
    result = _best_minimize(objective, starts, bounds)
    core_fraction, sigma_core_deg, sigma_tail_deg = unpack(result.x)
    model_probability = model(result.x)
    fit = DoubleRayleighFit(
        core_fraction=core_fraction,
        sigma_core_deg=sigma_core_deg,
        sigma_tail_deg=sigma_tail_deg,
        kl_divergence=kl_divergence(data_probability, model_probability),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        boundary_flags=_boundary_flags(
            (
                ("core_fraction", core_fraction, _logistic(bounds[0][0]), _logistic(bounds[0][1])),
                ("sigma_core_deg", sigma_core_deg, *sigma_core_bounds),
                (
                    "sigma_separation_deg",
                    sigma_tail_deg - sigma_core_deg,
                    *sigma_separation_bounds,
                ),
            )
        ),
    )
    return fit, model_probability


def fit_double_spherical_king_counts(
    weighted_counts: np.ndarray,
    edges_deg: np.ndarray,
    *,
    random_seed: int = 0,
    random_starts: int = 28,
    quadrature_order: int = 20,
) -> tuple[DoubleSphericalKingFit, np.ndarray]:
    edges = _validate_edges(edges_deg)
    counts = np.asarray(weighted_counts, dtype=np.float64)
    if counts.shape != (edges.size - 1,):
        raise ValueError("weighted_counts and edges_deg have incompatible shapes")
    data_probability = profile_probability(counts)
    nodes_rad, integration_weights = _gauss_legendre_sphere_bins(edges, quadrature_order)

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float, float]:
        core_fraction = _logistic(float(parameters[0]))
        sigma_core_deg = float(np.exp(parameters[1]))
        sigma_tail_deg = float(sigma_core_deg + np.exp(parameters[2]))
        gamma_core = float(1.0 + np.exp(parameters[3]))
        gamma_tail = float(1.0 + np.exp(parameters[4]))
        return core_fraction, sigma_core_deg, gamma_core, sigma_tail_deg, gamma_tail

    def model(parameters: np.ndarray) -> np.ndarray:
        core_fraction, sigma_core_deg, gamma_core, sigma_tail_deg, gamma_tail = unpack(parameters)
        core = _spherical_king_probability(
            nodes_rad,
            integration_weights,
            sigma_core_deg,
            gamma_core,
        )
        tail = _spherical_king_probability(
            nodes_rad,
            integration_weights,
            sigma_tail_deg,
            gamma_tail,
        )
        return core_fraction * core + (1.0 - core_fraction) * tail

    def objective(parameters: np.ndarray) -> float:
        probability = model(parameters)
        return float(-np.sum(data_probability * np.log(np.clip(probability, EPSILON, None))))

    starts: list[np.ndarray] = []
    for core_fraction, sigma_core_deg, gamma_core, sigma_tail_deg, gamma_tail in (
        (0.55, 0.15, 4.0, 3.0, 1.2),
        (0.65, 0.25, 20.0, 8.0, 1.5),
        (0.75, 0.35, 100.0, 20.0, 2.0),
        (0.85, 0.50, 5.0, 40.0, 5.0),
        (0.90, 0.70, 300.0, 5.0, 1.05),
        (0.50, 0.20, 1.2, 60.0, 100.0),
    ):
        starts.append(
            np.asarray(
                [
                    _logit(core_fraction),
                    math.log(sigma_core_deg),
                    math.log(sigma_tail_deg - sigma_core_deg),
                    math.log(gamma_core - 1.0),
                    math.log(gamma_tail - 1.0),
                ],
                dtype=np.float64,
            )
        )
    rng = np.random.default_rng(random_seed)
    for _ in range(max(0, int(random_starts))):
        sigma_core_deg = float(np.exp(rng.uniform(math.log(0.03), math.log(2.0))))
        sigma_tail_deg = float(
            np.exp(rng.uniform(math.log(max(0.15, 1.02 * sigma_core_deg)), math.log(120.0)))
        )
        starts.append(
            np.asarray(
                [
                    _logit(float(rng.uniform(0.10, 0.98))),
                    math.log(sigma_core_deg),
                    math.log(sigma_tail_deg - sigma_core_deg),
                    rng.uniform(math.log(1.0e-3), math.log(999.0)),
                    rng.uniform(math.log(1.0e-3), math.log(999.0)),
                ],
                dtype=np.float64,
            )
        )

    sigma_core_bounds = (0.005, 10.0)
    sigma_separation_bounds = (0.002, 180.0)
    gamma_excess_bounds = (1.0e-3, 999.0)
    bounds = [
        (-8.0, 8.0),
        tuple(math.log(value) for value in sigma_core_bounds),
        tuple(math.log(value) for value in sigma_separation_bounds),
        tuple(math.log(value) for value in gamma_excess_bounds),
        tuple(math.log(value) for value in gamma_excess_bounds),
    ]
    result = _best_minimize(objective, starts, bounds)
    core_fraction, sigma_core_deg, gamma_core, sigma_tail_deg, gamma_tail = unpack(result.x)
    model_probability = model(result.x)
    fit = DoubleSphericalKingFit(
        core_fraction=core_fraction,
        sigma_core_deg=sigma_core_deg,
        gamma_core=gamma_core,
        sigma_tail_deg=sigma_tail_deg,
        gamma_tail=gamma_tail,
        kl_divergence=kl_divergence(data_probability, model_probability),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        boundary_flags=_boundary_flags(
            (
                ("core_fraction", core_fraction, _logistic(bounds[0][0]), _logistic(bounds[0][1])),
                ("sigma_core_deg", sigma_core_deg, *sigma_core_bounds),
                (
                    "sigma_separation_deg",
                    sigma_tail_deg - sigma_core_deg,
                    *sigma_separation_bounds,
                ),
                ("gamma_core_excess", gamma_core - 1.0, *gamma_excess_bounds),
                ("gamma_tail_excess", gamma_tail - 1.0, *gamma_excess_bounds),
            )
        ),
    )
    return fit, model_probability
