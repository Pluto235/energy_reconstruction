"""Unit tests for the unbinned profiled-Poisson background surface fit.

These are self-contained (synthetic events, no ROOT / no Stage C data) so they run
locally in the ``py310`` env. They validate the mathematical core of
``fit_profiled_poisson_surface_unbinned`` against the existing binned fitter and against
injected truth.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import check_grad

MODULE_PATH = Path(__file__).resolve().parents[1] / "stages/poisson_roi_background.py"
SPEC = importlib.util.spec_from_file_location("poisson_roi_background", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PRB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRB
SPEC.loader.exec_module(PRB)


FIDUCIAL = 6.0
INNER = 1.0
OUTER = 2.0


def _q(coeffs, x, y):
    c = coeffs
    return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


def _sample_annulus_events(rng, coeffs, n, inner=INNER, outer=OUTER):
    """Rejection-sample events from the positive quadratic density over a centered annulus."""
    # Bound the density on the annulus with a dense evaluation.
    ang = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    rad = np.linspace(inner, outer, 200)
    rr, aa = np.meshgrid(rad, ang)
    qx = _q(coeffs, rr * np.cos(aa), rr * np.sin(aa))
    if np.min(qx) <= 0.0:
        raise ValueError("Test density is not positive on the annulus")
    q_max = float(np.max(qx)) * 1.02
    out = np.empty((0, 2), dtype=np.float64)
    while out.shape[0] < n:
        batch = int((n - out.shape[0]) * 2.5) + 1024
        u = rng.random(batch)
        rho = np.sqrt(inner * inner + u * (outer * outer - inner * inner))
        theta = rng.random(batch) * 2.0 * math.pi
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        accept = rng.random(batch) < (_q(coeffs, x, y) / q_max)
        out = np.vstack([out, np.column_stack([x[accept], y[accept]])])
    return out[:n]


def _binned_inputs(events, step, inner=INNER, outer=OUTER):
    """Build (counts, pixel_basis, mask) for a single cell, mirroring the Stage D binning."""
    lo = -(outer + 3.0 * step)
    hi = outer + 3.0 * step
    edges = np.arange(lo, hi + 0.5 * step, step)
    nx = ny = edges.size - 1
    counts2d, _, _ = np.histogram2d(events[:, 1], events[:, 0], bins=[edges, edges])  # [iy, ix]
    xc = 0.5 * (edges[:-1] + edges[1:])
    basis = np.empty((ny * nx, 6), dtype=np.float64)
    mask = np.zeros(ny * nx, dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            k = iy * nx + ix
            basis[k] = PRB.quadratic_rectangle_basis_integrals(edges[ix], edges[ix + 1], edges[iy], edges[iy + 1])
            rho = math.hypot(xc[ix], xc[iy])
            mask[k] = (rho >= inner) and (rho < outer) and (rho < FIDUCIAL)
    return counts2d.ravel().astype(np.float64), basis, mask


def _tau(fit):
    c = fit.shape_coefficients
    return float(c[3] + c[5])


def _bon_ratio(fit, r_opt, inner=INNER, outer=OUTER):
    """B_on / N_ann for a source-centered aperture, from the analytic centered integrals."""
    c = fit.shape_coefficients
    disk = math.pi * r_opt * r_opt * c[0] + 0.25 * math.pi * r_opt**4 * (c[3] + c[5])
    ann = PRB._centered_annulus_integral(c, inner, outer)
    return disk / ann


def test_t2_injected_tau_recovery():
    rng = np.random.default_rng(2)
    true = np.array([1.0, 0.0, 0.0, 0.05, 0.0, 0.05])  # tau* = 0.10, isotropic
    tau_star = true[3] + true[5]
    events = _sample_annulus_events(rng, true, 200_000)
    fit = PRB.fit_profiled_poisson_surface_unbinned(
        {7: events}, {7: (INNER, OUTER)}, [7], 2, FIDUCIAL, {7: True}
    )
    assert fit.positive_minimum > 0.0
    assert abs(_tau(fit) - tau_star) < 0.15 * tau_star + 0.005


def test_t1_binned_converges_to_unbinned_as_grid_refines():
    # The unbinned fit is the h -> 0 limit of the binned fit: refining the grid must
    # monotonically drive tau_binned and the B_on ratio toward the unbinned values.
    rng = np.random.default_rng(1)
    true = np.array([1.0, 0.0, 0.0, 0.05, 0.0, 0.05])
    events = _sample_annulus_events(rng, true, 120_000)
    unb = PRB.fit_profiled_poisson_surface_unbinned(
        {7: events}, {7: (INNER, OUTER)}, [7], 2, FIDUCIAL, {7: True}
    )
    tau_u = _tau(unb)
    bon_u = _bon_ratio(unb, 0.83)
    steps = [0.10, 0.05, 0.02]
    d_tau = []
    d_bon = []
    for step in steps:
        counts, basis, mask = _binned_inputs(events, step)
        binned = PRB.fit_profiled_poisson_surface(
            {7: counts}, {7: basis}, {7: mask}, [7], 2, FIDUCIAL, {7: True}
        )
        d_tau.append(abs(_tau(binned) - tau_u))
        d_bon.append(abs(_bon_ratio(binned, 0.83) - bon_u))
    # Monotone convergence of both tau and the B_on ratio toward the unbinned limit.
    assert d_tau[0] > d_tau[1] > d_tau[2], f"tau not converging: {d_tau}"
    assert d_bon[0] > d_bon[1] > d_bon[2], f"B_on ratio not converging: {d_bon}"
    # Finest grid is close to the unbinned limit.
    assert d_tau[-1] < 0.02
    # The production 0.1-deg grid is badly biased (motivates the whole change): its tau
    # is far from both the truth and the unbinned estimate.
    assert d_tau[0] > 0.03


def test_t3_anisotropy_nuisance_protection():
    rng = np.random.default_rng(3)
    # Strong azimuthal asymmetry (c1,c2,c4 != 0) at fixed radial trace tau* = 0.10.
    true = np.array([1.0, 0.02, -0.02, 0.06, 0.03, 0.04])
    tau_star = true[3] + true[5]
    events = _sample_annulus_events(rng, true, 250_000)
    fit = PRB.fit_profiled_poisson_surface_unbinned(
        {7: events}, {7: (INNER, OUTER)}, [7], 2, FIDUCIAL, {7: True}
    )
    # Full 2D unbinned fit recovers tau* despite the anisotropy nuisance.
    assert abs(_tau(fit) - tau_star) < 0.15 * tau_star + 0.005


def test_t4_gradient_matches_objective():
    rng = np.random.default_rng(4)
    true = np.array([1.0, 0.01, 0.0, 0.04, 0.01, 0.05])
    events = _sample_annulus_events(rng, true, 20_000)
    # Rebuild the internal objective through a tiny driver by monkeypatching is awkward;
    # instead reconstruct the objective the same way the fitter does.
    active = PRB._active_coefficient_indices(2)
    phi = PRB._basis_at_points(events)
    k1 = math.pi * (OUTER**2 - INNER**2)
    k2 = math.pi * (OUTER**4 - INNER**4) / 4.0
    n_b = float(events.shape[0])
    scale = n_b

    def f(params):
        c = PRB._coefficients_from_parameters(params, active)
        q = phi @ c
        a = c[0] * k1 + (c[3] + c[5]) * k2
        return (-np.sum(np.log(q)) + n_b * math.log(a)) / scale

    def g(params):
        c = PRB._coefficients_from_parameters(params, active)
        q = phi @ c
        a = c[0] * k1 + (c[3] + c[5]) * k2
        grad_full = -(phi.T @ (1.0 / q))
        da = np.zeros(6)
        da[0] = k1
        da[3] = k2
        da[5] = k2
        grad_full += (n_b / a) * da
        return grad_full[active] / scale

    p0 = np.array([0.01, 0.0, 0.04, 0.01, 0.05])  # active = [1,2,3,4,5]
    assert check_grad(f, g, p0) < 1e-5


def test_t5_positivity_holds_without_pixels():
    rng = np.random.default_rng(5)
    # Mild dome (negative radial trace) that is still positive on the annulus.
    true = np.array([1.0, 0.0, 0.0, -0.02, 0.0, -0.02])
    events = _sample_annulus_events(rng, true, 150_000)
    fit = PRB.fit_profiled_poisson_surface_unbinned(
        {7: events}, {7: (INNER, OUTER)}, [7], 2, FIDUCIAL, {7: True}
    )
    # Positive minimum over the whole 6-deg fiducial disk, enforced without training pixels.
    assert fit.positive_minimum > 0.0
    assert math.hypot(*fit.positive_minimum_xy) <= FIDUCIAL * (1.0 + 1e-9)


def test_t6_normalization_equals_event_count():
    rng = np.random.default_rng(6)
    true = np.array([1.0, 0.0, 0.0, 0.03, 0.0, 0.03])
    events = _sample_annulus_events(rng, true, 40_000)
    fit = PRB.fit_profiled_poisson_surface_unbinned(
        {7: events}, {7: (INNER, OUTER)}, [7], 2, FIDUCIAL, {7: True}
    )
    assert fit.annulus_normalizations[7] == float(events.shape[0])


def test_t7_binned_path_unaffected():
    # The pre-existing binned fitter still runs and returns a positive surface
    # (the unbinned addition is purely additive).
    rng = np.random.default_rng(7)
    true = np.array([1.0, 0.0, 0.0, 0.04, 0.0, 0.04])
    events = _sample_annulus_events(rng, true, 60_000)
    counts, basis, mask = _binned_inputs(events, 0.05)
    binned = PRB.fit_profiled_poisson_surface(
        {7: counts}, {7: basis}, {7: mask}, [7], 2, FIDUCIAL, {7: True}
    )
    assert binned.positive_minimum > 0.0
    assert math.isfinite(binned.poisson_deviance)
