#!/usr/bin/env python3
"""Numerical helpers for empirical-CDF aperture optimization."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


M2_TO_CM2 = 1.0e4


def radius_grid(min_deg: float, max_deg: float, step_deg: float) -> np.ndarray:
    if not math.isfinite(min_deg) or not math.isfinite(max_deg) or not math.isfinite(step_deg):
        raise ValueError("Radius-grid values must be finite")
    if min_deg <= 0.0 or max_deg < min_deg or step_deg <= 0.0:
        raise ValueError("Require 0 < min_deg <= max_deg and step_deg > 0")
    count = int(round((max_deg - min_deg) / step_deg))
    grid = min_deg + step_deg * np.arange(count + 1, dtype=np.float64)
    if not np.isclose(grid[-1], max_deg, rtol=0.0, atol=1.0e-10):
        raise ValueError("Radius step does not land on max_deg")
    grid[-1] = max_deg
    return grid


def asimov_significance(signal: np.ndarray, background: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if signal.shape != background.shape:
        raise ValueError("signal and background must have identical shapes")
    out = np.full(signal.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(signal) & np.isfinite(background) & (signal >= 0.0) & (background > 0.0)
    if np.any(valid):
        s = signal[valid]
        b = background[valid]
        term = (s + b) * np.log1p(s / b) - s
        out[valid] = np.sqrt(np.maximum(2.0 * term, 0.0))
    return out


def integrate_centered_disk_density(coefficients: np.ndarray, radius_deg: np.ndarray) -> np.ndarray:
    """Integrate c0+cx*x+cy*y+cxx*x^2+cxy*x*y+cyy*y^2 over a centered disk."""
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    radius = np.asarray(radius_deg, dtype=np.float64)
    if coefficients.size != 6 or not np.all(np.isfinite(coefficients)):
        raise ValueError("Expected six finite polynomial density coefficients")
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("Radii must be finite and non-negative")
    return (
        math.pi * radius**2 * coefficients[0]
        + math.pi * radius**4 * (coefficients[3] + coefficients[5]) / 4.0
    )


def select_smallest_near_maximum(
    radii_deg: np.ndarray,
    objective: np.ndarray,
    fraction: float = 0.99,
) -> tuple[int, int]:
    """Return (adopted index, exact-maximum index)."""
    radii = np.asarray(radii_deg, dtype=np.float64)
    values = np.asarray(objective, dtype=np.float64)
    if radii.ndim != 1 or values.shape != radii.shape or not np.all(np.diff(radii) > 0.0):
        raise ValueError("radii/objective must be matching one-dimensional ordered arrays")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("objective has no finite values")
    masked = np.where(finite, values, -np.inf)
    exact = int(np.argmax(masked))
    threshold = float(fraction) * float(masked[exact])
    eligible = np.flatnonzero(finite & (values >= threshold))
    if eligible.size == 0:
        raise RuntimeError("No radius meets the near-maximum threshold")
    return int(eligible[0]), exact


def logpar_flux_tev(
    energy_tev: np.ndarray,
    *,
    phi0: float,
    alpha: float,
    beta: float,
    pivot_tev: float,
) -> np.ndarray:
    ratio = np.asarray(energy_tev, dtype=np.float64) / float(pivot_tev)
    log_ratio = np.log(ratio)
    return float(phi0) * np.exp((-float(alpha) - float(beta) * log_ratio) * log_ratio)


def integrate_logpar_flux_bins(
    loge_edges: np.ndarray,
    parameters: Mapping[str, float],
    *,
    pivot_tev: float,
    quadrature_points: int = 64,
) -> np.ndarray:
    edges = np.asarray(loge_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("loge_edges must be strictly increasing")
    if quadrature_points <= 1:
        raise ValueError("quadrature_points must be greater than one")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    out = np.zeros(edges.size - 1, dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        x = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        energy_tev = np.power(10.0, x) / 1000.0
        flux = logpar_flux_tev(
            energy_tev,
            phi0=float(parameters["phi0"]),
            alpha=float(parameters["alpha"]),
            beta=float(parameters["beta"]),
            pivot_tev=float(pivot_tev),
        )
        integrand = flux * math.log(10.0) * energy_tev
        out[index] = 0.5 * (hi - lo) * float(np.sum(weights * integrand))
    return out


def signal_counts_from_numerator(
    numerator_sumw: np.ndarray,
    denominator_sumw: np.ndarray,
    theta_edges_deg: np.ndarray,
    s0_m2: float,
    flux_integral: np.ndarray,
    theta_exposure_sec: np.ndarray,
) -> np.ndarray:
    """Forward-fold one cell's cumulative numerator, shaped (radius, energy, theta)."""
    numerator = np.asarray(numerator_sumw, dtype=np.float64)
    denominator = np.asarray(denominator_sumw, dtype=np.float64)
    theta_edges = np.asarray(theta_edges_deg, dtype=np.float64)
    flux = np.asarray(flux_integral, dtype=np.float64)
    exposure = np.asarray(theta_exposure_sec, dtype=np.float64)
    expected_shape = (denominator.shape[0], denominator.shape[1])
    if numerator.ndim != 3 or numerator.shape[1:] != expected_shape:
        raise ValueError("numerator_sumw has incompatible dimensions")
    if theta_edges.size != denominator.shape[1] + 1:
        raise ValueError("theta edge count does not match denominator")
    if flux.shape != (denominator.shape[0],) or exposure.shape != (denominator.shape[1],):
        raise ValueError("Flux/exposure dimensions do not match the response")
    eta = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator[None, :, :], out=eta, where=denominator[None, :, :] > 0.0)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    a_eff = float(s0_m2) * eta * np.cos(np.radians(theta_centers))[None, None, :]
    return M2_TO_CM2 * np.einsum("ret,e,t->r", a_eff, flux, exposure)

