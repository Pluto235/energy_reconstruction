"""Weighted-LS quadratic constrained on a continuous disk; legacy fit is unchanged."""
import importlib
import numpy as np
from scipy.optimize import minimize
from .poisson_roi_background import _minimum_quadratic_on_centered_disk


def fit_nonnegative(design, counts, radius):
    a = np.asarray(design, dtype=float)
    z = np.asarray(counts, dtype=float)
    if (a.ndim != 2 or a.shape != (z.size, 6) or z.ndim != 1 or z.size < 6
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(z))
            or np.any(z < 0) or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Expected finite nonnegative counts and six-column quadratic design')
    w = 1 / np.sqrt(np.maximum(z, 1.))
    aw, zw = a * w[:, None], z * w
    if np.linalg.matrix_rank(aw) != 6 or np.linalg.cond(aw) > 1e8:
        raise ValueError('Invalid weighted-LS rank or condition')
    old = np.linalg.lstsq(aw, zw, rcond=None)[0]
    old_obj = float(np.sum((aw @ old - zw)**2))
    # Scale coordinates to the unit disk and density by observed annulus mean.
    density = max(float(np.mean(z)), np.finfo(float).tiny)
    units = np.array([1., radius, radius, radius**2, radius**2, radius**2])
    tolerance = 1e-10 * density  # counts per grid pixel; 1e-10 of mean annulus density.
    minimum, _ = _minimum_quadratic_on_centered_disk(old * units / density, 1.)
    active = minimum < 0.
    coeff, status, iterations = old, 'unconstrained feasible optimum', 0
    if active:
        m = aw / units * density
        h, rhs = m.T @ m, m.T @ zw
        objective_scale = max(float(np.sum(z)), 1.)
        def fun(p):
            delta = p - old * units / density
            return float(delta @ h @ delta / objective_scale)
        def jac(p):
            return 2 * (h @ p - rhs) / objective_scale
        def constraint(p):
            return _minimum_quadratic_on_centered_disk(p, 1.)[0]
        def constraint_jac(p):
            _, (x,y) = _minimum_quadratic_on_centered_disk(p, 1.)
            return np.array([1.,x,y,x*x,x*y,y*y])
        start = np.array([1.,0,0,0,0,0])
        result = minimize(fun, start, jac=jac, method='SLSQP',
            constraints={'type':'ineq','fun':constraint,'jac':constraint_jac},
            options={'ftol':1e-12,'maxiter':1000})
        if not result.success:
            raise RuntimeError(f'Constrained weighted-LS failed: {result.message}')
        coeff = result.x * density / units
        status, iterations = str(result.message), int(result.nit)
    # Normalize before root finding to avoid absolute polynomial-root thresholds at low density.
    final_min, xy = _minimum_quadratic_on_centered_disk(coeff * units / density, 1.)
    final_min *= density
    obj = float(np.sum((aw @ coeff - zw)**2))
    if not np.all(np.isfinite(coeff)) or final_min < -tolerance or obj < old_obj - 1e-8*max(old_obj,1.):
        raise RuntimeError('Continuous-disk or objective validation failed')
    return dict(coefficients=coeff, unconstrained_coefficients=old, objective=obj,
        unconstrained_objective=old_obj, minimum=final_min,
        minimum_xy=np.array(xy)*radius, tolerance=tolerance, active=active,
        optimizer_status=status, iterations=iterations, constraint_violation=max(0.,-final_min))


def normalized_background(fit, design, counts, radius, step):
    d = importlib.import_module('apply.stages.04_background')
    coeff = fit['coefficients']
    # Preserve the original sum(max(q,0)) annulus denominator exactly.
    denominator = float(np.maximum(np.asarray(design) @ coeff, 0.).sum())
    total = float(np.sum(counts))
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(total) or total <= 0:
        raise ValueError('Invalid annulus normalization')
    scale = total / denominator
    b = d.integrate_centered_disk_polynomial(coeff, radius, step, scale)
    if not np.isfinite(b) or b <= 0:
        raise ValueError('Invalid positive analytic B_on')
    return b, scale
