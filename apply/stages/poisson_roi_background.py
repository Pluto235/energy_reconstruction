from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import LinearConstraint, minimize


_COEFFICIENT_COUNT = 6
_POSITIVITY_FLOOR = 1.0e-12
_CONSTRAINT_POSITIVITY_FLOOR = 1.0e-8
_SUPPORT_GRID_STEP_DEG = 0.25
_MAX_CUTTING_PLANE_ITERATIONS = 8


class PoissonSurfaceFitError(RuntimeError):
    """Raised when a positive profiled-Poisson surface cannot be fitted."""


@dataclass(frozen=True)
class SurfaceFit:
    order: int
    shape_coefficients: np.ndarray
    donor_cell_ids: tuple[int, ...]
    annulus_normalizations: dict[int, float]
    poisson_deviance: float
    ndof: int
    positive_minimum: float
    positive_minimum_xy: tuple[float, float]
    optimizer_status: dict[str, object]


def quadratic_rectangle_basis_integrals(
    x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    """Integrate [1, x, y, x^2, xy, y^2] over one rectangle."""
    bounds = np.asarray([x0, x1, y0, y1], dtype=np.float64)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Rectangle bounds must be finite")
    if not x1 > x0 or not y1 > y0:
        raise ValueError("Rectangle upper bounds must exceed lower bounds")
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    int_x = 0.5 * float(x1**2 - x0**2)
    int_y = 0.5 * float(y1**2 - y0**2)
    return np.asarray(
        [
            dx * dy,
            int_x * dy,
            dx * int_y,
            float(x1**3 - x0**3) * dy / 3.0,
            int_x * int_y,
            dx * float(y1**3 - y0**3) / 3.0,
        ],
        dtype=np.float64,
    )


def _basis_at_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack([np.ones(x.size), x, y, x * x, x * y, y * y])


def _active_coefficient_indices(order: int) -> np.ndarray:
    if int(order) == 0:
        return np.empty(0, dtype=np.int64)
    if int(order) == 1:
        return np.asarray([1, 2], dtype=np.int64)
    if int(order) == 2:
        return np.arange(1, _COEFFICIENT_COUNT, dtype=np.int64)
    raise PoissonSurfaceFitError(f"Unsupported surface order: {order}; expected 0, 1, or 2")


def _coefficients_from_parameters(parameters: np.ndarray, active: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
    coefficients[0] = 1.0
    coefficients[active] = np.asarray(parameters, dtype=np.float64)
    return coefficients


def _rectangle_corners(pixel_basis: np.ndarray) -> np.ndarray:
    """Recover axis-aligned rectangle corners from its six exact moments."""
    basis = np.asarray(pixel_basis, dtype=np.float64)
    area = basis[:, 0]
    center_x = basis[:, 1] / area
    center_y = basis[:, 2] / area
    width_sq = 12.0 * (basis[:, 3] / area - center_x * center_x)
    height_sq = 12.0 * (basis[:, 5] / area - center_y * center_y)
    scale = np.maximum(1.0, np.maximum(center_x * center_x, center_y * center_y))
    tolerance = 2.0e-11 * scale
    if np.any(width_sq < -tolerance) or np.any(height_sq < -tolerance):
        raise PoissonSurfaceFitError("Pixel basis does not describe axis-aligned rectangles")
    width = np.sqrt(np.maximum(width_sq, 0.0))
    height = np.sqrt(np.maximum(height_sq, 0.0))
    if np.any(width <= 0.0) or np.any(height <= 0.0):
        raise PoissonSurfaceFitError("Pixel rectangles must have positive width and height")
    expected_xy = area * center_x * center_y
    if not np.allclose(basis[:, 4], expected_xy, rtol=2.0e-10, atol=2.0e-12):
        raise PoissonSurfaceFitError("Pixel xy moments are inconsistent with rectangles")
    offsets = np.asarray([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    corners = np.empty((basis.shape[0], 4, 2), dtype=np.float64)
    corners[:, :, 0] = center_x[:, None] + width[:, None] * offsets[None, :, 0]
    corners[:, :, 1] = center_y[:, None] + height[:, None] * offsets[None, :, 1]
    return corners.reshape(-1, 2)


def _fixed_disk_support(radius: float) -> np.ndarray:
    limit = int(math.ceil(radius / _SUPPORT_GRID_STEP_DEG))
    coordinates = np.arange(-limit, limit + 1, dtype=np.float64) * _SUPPORT_GRID_STEP_DEG
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    interior = points[np.sum(points * points, axis=1) <= radius * radius * (1.0 + 1.0e-14)]
    angles = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
    boundary = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([interior, boundary])


def _evaluate(coefficients: np.ndarray, x: float, y: float) -> float:
    c = coefficients
    return float(c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y)


def _minimum_quadratic_on_centered_disk(
    coefficients: np.ndarray, radius: float
) -> tuple[float, tuple[float, float]]:
    """Return the global quadratic minimum using all interior and boundary stationary points."""
    c = np.asarray(coefficients, dtype=np.float64)
    candidates: list[tuple[float, float]] = [(0.0, 0.0), (radius, 0.0), (-radius, 0.0)]

    hessian = np.asarray([[2.0 * c[3], c[4]], [c[4], 2.0 * c[5]]])
    gradient = np.asarray([c[1], c[2]])
    stationary, _, _, _ = np.linalg.lstsq(hessian, -gradient, rcond=None)
    residual = np.linalg.norm(hessian @ stationary + gradient)
    if residual <= 1.0e-10 * max(1.0, np.linalg.norm(gradient)):
        if np.linalg.norm(stationary) <= radius * (1.0 + 1.0e-12):
            candidates.append((float(stationary[0]), float(stationary[1])))

    r = radius
    derivative_descending = np.asarray(
        [
            -r * c[2] + r * r * c[4],
            -2.0 * r * c[1] - 4.0 * r * r * (c[5] - c[3]),
            -6.0 * r * r * c[4],
            -2.0 * r * c[1] + 4.0 * r * r * (c[5] - c[3]),
            r * c[2] + r * r * c[4],
        ]
    )
    nonzero = np.flatnonzero(np.abs(derivative_descending) > 1.0e-14)
    if nonzero.size:
        for root in np.roots(derivative_descending[nonzero[0] :]):
            if abs(float(root.imag)) <= 1.0e-9 * max(1.0, abs(float(root.real))):
                theta = 2.0 * math.atan(float(root.real))
                candidates.append((r * math.cos(theta), r * math.sin(theta)))

    values = np.asarray([_evaluate(c, x, y) for x, y in candidates])
    index = int(np.argmin(values))
    return float(values[index]), candidates[index]


def _validated_fit_inputs(
    counts_by_cell: dict[int, np.ndarray],
    pixel_basis_by_cell: dict[int, np.ndarray],
    annulus_mask_by_cell: dict[int, np.ndarray],
    donor_cell_ids: Sequence[int],
    shape_contributor_by_cell: dict[int, bool],
) -> tuple[
    tuple[int, ...],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, float],
]:
    donors = tuple(int(cell_id) for cell_id in donor_cell_ids)
    if not donors or len(set(donors)) != len(donors):
        raise PoissonSurfaceFitError("donor_cell_ids must be non-empty and unique")
    masked_counts: dict[int, np.ndarray] = {}
    masked_basis: dict[int, np.ndarray] = {}
    full_basis: dict[int, np.ndarray] = {}
    normalizations: dict[int, float] = {}
    for cell_id in donors:
        try:
            counts = np.asarray(counts_by_cell[cell_id], dtype=np.float64)
            basis = np.asarray(pixel_basis_by_cell[cell_id], dtype=np.float64)
            mask = np.asarray(annulus_mask_by_cell[cell_id], dtype=bool)
            bool(shape_contributor_by_cell[cell_id])
        except KeyError as exc:
            raise PoissonSurfaceFitError(f"Missing fit input for donor cell {cell_id}") from exc
        if counts.ndim != 1 or basis.ndim != 2 or basis.shape != (counts.size, _COEFFICIENT_COUNT):
            raise PoissonSurfaceFitError(f"Invalid count or pixel-basis shape for donor cell {cell_id}")
        if mask.shape != counts.shape or not np.any(mask):
            raise PoissonSurfaceFitError(f"Invalid or empty annulus mask for donor cell {cell_id}")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
            raise PoissonSurfaceFitError(f"Counts must be finite and non-negative for donor cell {cell_id}")
        if not np.all(np.isfinite(basis)) or np.any(basis[:, 0] <= 0.0):
            raise PoissonSurfaceFitError(f"Pixel basis must be finite with positive areas for donor cell {cell_id}")
        masked_counts[cell_id] = counts[mask]
        masked_basis[cell_id] = basis[mask]
        full_basis[cell_id] = basis
        normalizations[cell_id] = float(np.sum(counts[mask]))
    return donors, masked_counts, masked_basis, full_basis, normalizations


def fit_profiled_poisson_surface(
    counts_by_cell: dict[int, np.ndarray],
    pixel_basis_by_cell: dict[int, np.ndarray],
    annulus_mask_by_cell: dict[int, np.ndarray],
    donor_cell_ids: Sequence[int],
    order: int,
    positivity_radius_deg: float,
    shape_contributor_by_cell: dict[int, bool],
) -> SurfaceFit:
    """Fit a shared positive polynomial shape with per-donor profiled normalizations."""
    radius = float(positivity_radius_deg)
    if not math.isfinite(radius) or radius <= 0.0:
        raise PoissonSurfaceFitError("positivity_radius_deg must be positive and finite")
    active = _active_coefficient_indices(int(order))
    donors, counts, basis, full_basis, normalizations = _validated_fit_inputs(
        counts_by_cell,
        pixel_basis_by_cell,
        annulus_mask_by_cell,
        donor_cell_ids,
        shape_contributor_by_cell,
    )
    contributors = tuple(cell_id for cell_id in donors if bool(shape_contributor_by_cell[cell_id]))
    if active.size and not contributors:
        raise PoissonSurfaceFitError("At least one donor must contribute shape information")
    for cell_id in contributors:
        if normalizations[cell_id] <= 0.0:
            raise PoissonSurfaceFitError(f"Shape contributor {cell_id} has zero annulus counts")
    likelihood_scale = max(1.0, sum(normalizations[cell_id] for cell_id in contributors))

    training_basis = np.vstack([basis[cell_id] for cell_id in donors])
    unique_full_basis = np.unique(np.vstack([full_basis[cell_id] for cell_id in donors]), axis=0)
    full_centers = np.column_stack(
        [
            unique_full_basis[:, 1] / unique_full_basis[:, 0],
            unique_full_basis[:, 2] / unique_full_basis[:, 0],
        ]
    )
    full_corners = _rectangle_corners(unique_full_basis).reshape(-1, 4, 2)
    centered_in_disk = np.sum(full_centers * full_centers, axis=1) <= radius * radius * (1.0 + 1.0e-14)
    crosses_disk_boundary = np.max(np.sum(full_corners * full_corners, axis=2), axis=1) > radius * radius
    boundary_pixel_basis = unique_full_basis[centered_in_disk & crosses_disk_boundary]
    point_constraint_pixel_basis = np.unique(training_basis, axis=0)
    constraint_pixel_basis = np.unique(
        np.vstack([training_basis, boundary_pixel_basis]), axis=0
    )
    training_corners = _rectangle_corners(point_constraint_pixel_basis)
    constraint_points = np.unique(
        np.vstack([training_corners, _fixed_disk_support(radius)]), axis=0
    )
    point_basis = _basis_at_points(constraint_points)

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = _coefficients_from_parameters(parameters, active)
        nll = 0.0
        gradient = np.zeros(active.size, dtype=np.float64)
        for cell_id in contributors:
            cell_basis = basis[cell_id]
            pixel_shape = cell_basis @ coefficients
            shape_sum = float(np.sum(pixel_shape))
            if np.any(pixel_shape <= 0.0) or not math.isfinite(shape_sum) or shape_sum <= 0.0:
                return 1.0e100, np.zeros(active.size, dtype=np.float64)
            cell_counts = counts[cell_id]
            total = normalizations[cell_id]
            nll += -float(np.dot(cell_counts, np.log(pixel_shape))) + total * math.log(shape_sum)
            active_basis = cell_basis[:, active]
            gradient += -(cell_counts / pixel_shape) @ active_basis
            gradient += (total / shape_sum) * np.sum(active_basis, axis=0)
        return nll / likelihood_scale, gradient / likelihood_scale

    def linear_constraint(extra_points: list[tuple[float, float]]) -> LinearConstraint:
        all_point_basis = point_basis
        if extra_points:
            all_point_basis = np.vstack([all_point_basis, _basis_at_points(np.asarray(extra_points))])
        point_matrix = all_point_basis[:, active]
        point_lower = _CONSTRAINT_POSITIVITY_FLOOR - all_point_basis[:, 0]
        # These domain constraints keep every likelihood probability strictly positive.
        integral_matrix = constraint_pixel_basis[:, active]
        integral_lower = (
            _CONSTRAINT_POSITIVITY_FLOOR * constraint_pixel_basis[:, 0]
            - constraint_pixel_basis[:, 0]
        )
        matrix = np.vstack([point_matrix, integral_matrix])
        lower = np.concatenate([point_lower, integral_lower])
        return LinearConstraint(matrix, lower, np.full(lower.shape, np.inf))

    parameters = np.zeros(active.size, dtype=np.float64)
    result = None
    extra_points: list[tuple[float, float]] = []
    minimum = 1.0
    minimum_xy = (0.0, 0.0)
    if active.size:
        for cutting_iteration in range(_MAX_CUTTING_PLANE_ITERATIONS + 1):
            result = minimize(
                lambda value: objective_and_gradient(value)[0],
                parameters,
                jac=lambda value: objective_and_gradient(value)[1],
                constraints=[linear_constraint(extra_points)],
                method="SLSQP",
                options={"ftol": 1.0e-11, "maxiter": 2000, "disp": False},
            )
            if not bool(result.success) or not np.all(np.isfinite(result.x)):
                message = str(getattr(result, "message", "unknown optimizer failure"))
                raise PoissonSurfaceFitError(f"Profiled-Poisson optimization failed: {message}")
            parameters = np.asarray(result.x, dtype=np.float64)
            coefficients = _coefficients_from_parameters(parameters, active)
            minimum, minimum_xy = _minimum_quadratic_on_centered_disk(coefficients, radius)
            if minimum >= _POSITIVITY_FLOOR * coefficients[0]:
                break
            if cutting_iteration >= _MAX_CUTTING_PLANE_ITERATIONS:
                raise PoissonSurfaceFitError(
                    "Positive surface cutting-plane loop failed after eight iterations: "
                    f"minimum={minimum:.17g} at {minimum_xy}, parameters={parameters.tolist()}"
                )
            extra_points.append(minimum_xy)
        else:  # pragma: no cover - loop is explicitly bounded above
            raise PoissonSurfaceFitError("Positive surface cutting-plane loop did not terminate")
    else:
        coefficients = _coefficients_from_parameters(parameters, active)

    if not math.isfinite(minimum) or minimum <= 0.0:
        raise PoissonSurfaceFitError(f"Fitted surface is not positive on the disk: minimum={minimum}")

    deviance = 0.0
    observed_categories = 0
    for cell_id in contributors:
        pixel_shape = basis[cell_id] @ coefficients
        probabilities = pixel_shape / float(np.sum(pixel_shape))
        expected = normalizations[cell_id] * probabilities
        positive = counts[cell_id] > 0.0
        deviance += 2.0 * float(
            np.sum(counts[cell_id][positive] * np.log(counts[cell_id][positive] / expected[positive]))
        )
        observed_categories += int(counts[cell_id].size)
    ndof = max(0, observed_categories - len(contributors) - int(active.size))
    final_nll, _ = objective_and_gradient(parameters)
    final_linear_constraint = linear_constraint(extra_points)
    constrained_values = np.asarray(final_linear_constraint.A @ parameters, dtype=np.float64)
    constraint_violation = np.maximum(
        np.asarray(final_linear_constraint.lb, dtype=np.float64) - constrained_values,
        0.0,
    )
    status: dict[str, object] = {
        "success": True,
        "method": "SLSQP",
        "message": "constant surface has no free shape parameters" if result is None else str(result.message),
        "status": 0 if result is None else int(result.status),
        "iterations": 0 if result is None else int(result.nit),
        "function_evaluations": 1 if result is None else int(result.nfev),
        "gradient_evaluations": 1 if result is None else int(result.njev),
        "cutting_plane_constraints_added": len(extra_points),
        "constraint_count": int(
            point_basis.shape[0] + constraint_pixel_basis.shape[0] + len(extra_points)
        ),
        "boundary_pixel_integral_constraint_count": int(boundary_pixel_basis.shape[0]),
        "max_linear_constraint_violation": float(np.max(constraint_violation, initial=0.0)),
        "profiled_nll": float(final_nll * likelihood_scale),
    }
    return SurfaceFit(
        order=int(order),
        shape_coefficients=coefficients,
        donor_cell_ids=donors,
        annulus_normalizations=normalizations,
        poisson_deviance=float(deviance),
        ndof=int(ndof),
        positive_minimum=float(minimum),
        positive_minimum_xy=(float(minimum_xy[0]), float(minimum_xy[1])),
        optimizer_status=status,
    )


_UNBINNED_CONSTRAINT_POINT_CAP = 30000


def _centered_annulus_integral(coefficients: np.ndarray, inner: float, outer: float) -> float:
    """Analytic integral of the polynomial density over a centered annulus [inner, outer].

    Mirrors ``integrate_centered_annulus_density`` in ``04_background.py``. Only the
    intercept and the radial trace survive; linear and anisotropic terms integrate to
    zero over a centered region.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    return float(
        math.pi * (outer * outer - inner * inner) * c[0]
        + math.pi * (outer**4 - inner**4) * (c[3] + c[5]) / 4.0
    )


def _validated_unbinned_fit_inputs(
    event_xy_by_cell: dict[int, np.ndarray],
    annulus_bounds_by_cell: dict[int, tuple[float, float]],
    donor_cell_ids: Sequence[int],
    shape_contributor_by_cell: dict[int, bool],
) -> tuple[tuple[int, ...], dict[int, np.ndarray], dict[int, tuple[float, float]], dict[int, float]]:
    donors = tuple(int(cell_id) for cell_id in donor_cell_ids)
    if not donors or len(set(donors)) != len(donors):
        raise PoissonSurfaceFitError("donor_cell_ids must be non-empty and unique")
    events: dict[int, np.ndarray] = {}
    bounds: dict[int, tuple[float, float]] = {}
    normalizations: dict[int, float] = {}
    for cell_id in donors:
        try:
            xy = np.asarray(event_xy_by_cell[cell_id], dtype=np.float64)
            inner, outer = annulus_bounds_by_cell[cell_id]
            bool(shape_contributor_by_cell[cell_id])
        except KeyError as exc:
            raise PoissonSurfaceFitError(f"Missing unbinned fit input for donor cell {cell_id}") from exc
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise PoissonSurfaceFitError(f"Event coordinates for donor cell {cell_id} must have shape (M, 2)")
        if xy.size and not np.all(np.isfinite(xy)):
            raise PoissonSurfaceFitError(f"Event coordinates must be finite for donor cell {cell_id}")
        inner = float(inner)
        outer = float(outer)
        if not (math.isfinite(inner) and math.isfinite(outer)) or inner < 0.0 or outer <= inner:
            raise PoissonSurfaceFitError(f"Annulus bounds must satisfy 0 <= inner < outer for donor cell {cell_id}")
        events[cell_id] = xy
        bounds[cell_id] = (inner, outer)
        normalizations[cell_id] = float(xy.shape[0])
    return donors, events, bounds, normalizations


def fit_profiled_poisson_surface_unbinned(
    event_xy_by_cell: dict[int, np.ndarray],
    annulus_bounds_by_cell: dict[int, tuple[float, float]],
    donor_cell_ids: Sequence[int],
    order: int,
    positivity_radius_deg: float,
    shape_contributor_by_cell: dict[int, bool],
    *,
    constraint_point_cap: int = _UNBINNED_CONSTRAINT_POINT_CAP,
) -> SurfaceFit:
    """Fit a shared positive polynomial shape to continuous OFF-annulus events (unbinned).

    Grid-free analogue of :func:`fit_profiled_poisson_surface`. The profiled multinomial
    negative log-likelihood for donor cell ``b`` is

        -log L_b = -sum_e log q(x_e, y_e) + N_ann,b * log(integral_annulus_b(q))

    where the numerator is evaluated at the exact continuous event coordinates (no pixels)
    and the denominator is the analytic centered-annulus integral. The per-cell
    normalization ``N_ann,b`` (the OFF event count) is profiled out; the shape is shared
    across pooled donors. B_on downstream uses the same analytic centered integrals, so it
    is independent of any grid.
    """
    radius = float(positivity_radius_deg)
    if not math.isfinite(radius) or radius <= 0.0:
        raise PoissonSurfaceFitError("positivity_radius_deg must be positive and finite")
    active = _active_coefficient_indices(int(order))
    donors, events, bounds, normalizations = _validated_unbinned_fit_inputs(
        event_xy_by_cell,
        annulus_bounds_by_cell,
        donor_cell_ids,
        shape_contributor_by_cell,
    )
    contributors = tuple(cell_id for cell_id in donors if bool(shape_contributor_by_cell[cell_id]))
    if active.size and not contributors:
        raise PoissonSurfaceFitError("At least one donor must contribute shape information")
    for cell_id in contributors:
        if normalizations[cell_id] <= 0.0:
            raise PoissonSurfaceFitError(f"Shape contributor {cell_id} has zero annulus events")
    likelihood_scale = max(1.0, sum(normalizations[cell_id] for cell_id in contributors))

    event_basis = {cell_id: _basis_at_points(events[cell_id]) for cell_id in contributors}
    annulus_const: dict[int, tuple[float, float]] = {}
    for cell_id in contributors:
        inner, outer = bounds[cell_id]
        annulus_const[cell_id] = (
            math.pi * (outer * outer - inner * inner),
            math.pi * (outer**4 - inner**4) / 4.0,
        )

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = _coefficients_from_parameters(parameters, active)
        nll = 0.0
        gradient = np.zeros(active.size, dtype=np.float64)
        for cell_id in contributors:
            phi = event_basis[cell_id]
            shape_at_events = phi @ coefficients
            k1, k2 = annulus_const[cell_id]
            annulus_integral = coefficients[0] * k1 + (coefficients[3] + coefficients[5]) * k2
            total = normalizations[cell_id]
            if np.any(shape_at_events <= 0.0) or not math.isfinite(annulus_integral) or annulus_integral <= 0.0:
                return 1.0e100, np.zeros(active.size, dtype=np.float64)
            nll += -float(np.sum(np.log(shape_at_events))) + total * math.log(annulus_integral)
            gradient_full = -(phi.T @ (1.0 / shape_at_events))
            annulus_gradient = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
            annulus_gradient[0] = k1
            annulus_gradient[3] = k2
            annulus_gradient[5] = k2
            gradient_full += (total / annulus_integral) * annulus_gradient
            gradient += gradient_full[active]
        return nll / likelihood_scale, gradient / likelihood_scale

    support_points = _fixed_disk_support(radius)
    if contributors:
        all_events = np.vstack([events[cell_id] for cell_id in contributors])
        event_points = np.unique(all_events, axis=0)
        if event_points.shape[0] > int(constraint_point_cap):
            stride = int(math.ceil(event_points.shape[0] / float(constraint_point_cap)))
            event_points = event_points[::stride]
        base_constraint_points = np.unique(np.vstack([support_points, event_points]), axis=0)
    else:
        base_constraint_points = support_points
    base_point_basis = _basis_at_points(base_constraint_points)

    def linear_constraint(extra_points: list[tuple[float, float]]) -> LinearConstraint:
        point_basis = base_point_basis
        if extra_points:
            point_basis = np.vstack([point_basis, _basis_at_points(np.asarray(extra_points))])
        matrix = point_basis[:, active]
        lower = _CONSTRAINT_POSITIVITY_FLOOR - point_basis[:, 0]
        return LinearConstraint(matrix, lower, np.full(lower.shape[0], np.inf))

    parameters = np.zeros(active.size, dtype=np.float64)
    result = None
    extra_points: list[tuple[float, float]] = []
    minimum = 1.0
    minimum_xy = (0.0, 0.0)
    if active.size:
        for cutting_iteration in range(_MAX_CUTTING_PLANE_ITERATIONS + 1):
            result = minimize(
                lambda value: objective_and_gradient(value)[0],
                parameters,
                jac=lambda value: objective_and_gradient(value)[1],
                constraints=[linear_constraint(extra_points)],
                method="SLSQP",
                options={"ftol": 1.0e-11, "maxiter": 2000, "disp": False},
            )
            if not bool(result.success) or not np.all(np.isfinite(result.x)):
                message = str(getattr(result, "message", "unknown optimizer failure"))
                raise PoissonSurfaceFitError(f"Profiled-Poisson unbinned optimization failed: {message}")
            parameters = np.asarray(result.x, dtype=np.float64)
            coefficients = _coefficients_from_parameters(parameters, active)
            minimum, minimum_xy = _minimum_quadratic_on_centered_disk(coefficients, radius)
            if minimum >= _POSITIVITY_FLOOR * coefficients[0]:
                break
            if cutting_iteration >= _MAX_CUTTING_PLANE_ITERATIONS:
                raise PoissonSurfaceFitError(
                    "Unbinned positive surface cutting-plane loop failed after eight iterations: "
                    f"minimum={minimum:.17g} at {minimum_xy}, parameters={parameters.tolist()}"
                )
            extra_points.append(minimum_xy)
        else:  # pragma: no cover - loop is explicitly bounded above
            raise PoissonSurfaceFitError("Unbinned positive surface cutting-plane loop did not terminate")
    else:
        coefficients = _coefficients_from_parameters(parameters, active)

    if not math.isfinite(minimum) or minimum <= 0.0:
        raise PoissonSurfaceFitError(f"Fitted unbinned surface is not positive on the disk: minimum={minimum}")

    observed_events = int(sum(int(normalizations[cell_id]) for cell_id in contributors))
    ndof = max(0, observed_events - len(contributors) - int(active.size))
    final_nll, _ = objective_and_gradient(parameters)
    final_linear_constraint = linear_constraint(extra_points)
    constrained_values = np.asarray(final_linear_constraint.A @ parameters, dtype=np.float64)
    constraint_violation = np.maximum(
        np.asarray(final_linear_constraint.lb, dtype=np.float64) - constrained_values,
        0.0,
    )
    status: dict[str, object] = {
        "success": True,
        "method": "SLSQP",
        "fit_kind": "unbinned",
        "message": "constant surface has no free shape parameters" if result is None else str(result.message),
        "status": 0 if result is None else int(result.status),
        "iterations": 0 if result is None else int(result.nit),
        "function_evaluations": 1 if result is None else int(result.nfev),
        "gradient_evaluations": 1 if result is None else int(result.njev),
        "cutting_plane_constraints_added": len(extra_points),
        "constraint_count": int(base_point_basis.shape[0] + len(extra_points)),
        "event_count": observed_events,
        "max_linear_constraint_violation": float(np.max(constraint_violation, initial=0.0)),
        "profiled_nll": float(final_nll * likelihood_scale),
    }
    return SurfaceFit(
        order=int(order),
        shape_coefficients=coefficients,
        donor_cell_ids=donors,
        annulus_normalizations=normalizations,
        poisson_deviance=float("nan"),
        ndof=int(ndof),
        positive_minimum=float(minimum),
        positive_minimum_xy=(float(minimum_xy[0]), float(minimum_xy[1])),
        optimizer_status=status,
    )
