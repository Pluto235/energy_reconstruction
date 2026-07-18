#!/usr/bin/env python
"""Registered time-split acceptance gate for the unbinned pooled-Poisson background.

The unbinned B_on is grid-invariant by construction, so grid phase/resolution is no longer an
independent test. This gate instead splits the livetime into two halves (at the median event
MJD, equal statistics) and checks that the **exposure-free shape factor** g(tau)=B_on/N_ann is
consistent between halves for the order-2 cells (curvature is the only quantity that carries a
grid/time-systematic risk). Raw B_on differs between halves only through exposure and is not
tested here.

Gate: for every order-2 target cell, |g_h1 - g_h2| <= k * sqrt(sigma_g,h1^2 + sigma_g,h2^2),
with sigma_g,half = sqrt(2) * sigma_g,nominal (half statistics), sigma_g,nominal from the
nominal unbinned bootstrap covariance diagonal, and default k=1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_cells(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    cid = np.asarray(d["cell_id"], dtype=np.int64)
    order = np.asarray(d["surface_order"], dtype=np.int64)
    b_on = np.asarray(d["B_on"], dtype=np.float64)
    n_ann = np.asarray(d["continuous_annulus_counts"], dtype=np.float64)
    return {int(c): {"order": int(o), "b_on": float(b), "n_ann": float(n)}
            for c, o, b, n in zip(cid, order, b_on, n_ann)}


def _sigma_bon_from_cov(cov_npz: str):
    d = np.load(cov_npz, allow_pickle=False)
    cid = np.asarray(d["cell_id"], dtype=np.int64)
    var = np.diag(np.asarray(d["B_on_covariance"], dtype=np.float64))
    return {int(c): float(math.sqrt(max(v, 0.0))) for c, v in zip(cid, var)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nominal-npz", required=True, help="Nominal unbinned Stage-D NPZ.")
    p.add_argument("--half1-npz", required=True)
    p.add_argument("--half2-npz", required=True)
    p.add_argument("--bootstrap-cov-npz", required=True, help="Unbinned bootstrap covariance NPZ (B_on_covariance).")
    p.add_argument("--k", type=float, default=1.0, help="Tolerance multiple (default 1).")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    nominal = _load_cells(args.nominal_npz)
    h1 = _load_cells(args.half1_npz)
    h2 = _load_cells(args.half2_npz)
    sigma_bon = _sigma_bon_from_cov(args.bootstrap_cov_npz)

    # Only order-2 TARGET cells (present in the bootstrap covariance / 44-cell SED fit) are
    # gated; order-2 donor-only cells (e.g. 7-12) are not targets and carry no B_on covariance.
    order2 = sorted(c for c, v in nominal.items() if v["order"] == 2 and c in sigma_bon)
    checks = []
    worst = 0.0
    passed = True
    for c in order2:
        g_h1 = h1[c]["b_on"] / h1[c]["n_ann"] if h1[c]["n_ann"] > 0 else float("nan")
        g_h2 = h2[c]["b_on"] / h2[c]["n_ann"] if h2[c]["n_ann"] > 0 else float("nan")
        g_nom = nominal[c]["b_on"] / nominal[c]["n_ann"] if nominal[c]["n_ann"] > 0 else float("nan")
        sigma_g_nom = sigma_bon.get(c, float("nan")) / nominal[c]["n_ann"] if nominal[c]["n_ann"] > 0 else float("nan")
        # half sigma ~ sqrt(2) * nominal; difference sigma = sqrt(sig_h1^2 + sig_h2^2) = 2 * sigma_g_nom
        sigma_diff = 2.0 * sigma_g_nom
        delta = abs(g_h1 - g_h2)
        ratio = delta / sigma_diff if sigma_diff > 0 else float("inf")
        ok = bool(ratio <= args.k)
        passed = passed and ok
        worst = max(worst, ratio)
        checks.append({
            "cell_id": c, "g_h1": g_h1, "g_h2": g_h2, "g_nominal": g_nom,
            "abs_delta_g": delta, "sigma_g_nominal": sigma_g_nom, "sigma_diff": sigma_diff,
            "delta_over_sigma": ratio, "passed": ok,
        })

    gate = {
        "name": "time_split_shape_consistency",
        "passed": bool(passed),
        "observed_max_delta_over_sigma": worst,
        "limit": {"operator": "<=", "value": float(args.k)},
        "n_order2_cells": len(order2),
        "median_mjd_note": "halves are split at the median OFF-event MJD (equal statistics)",
    }
    payload = {
        "schema_version": 1,
        "gate": gate,
        "checks": checks,
        "inputs": {
            "nominal_npz": args.nominal_npz, "half1_npz": args.half1_npz,
            "half2_npz": args.half2_npz, "bootstrap_cov_npz": args.bootstrap_cov_npz,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("time_split_shape_consistency: %s (max delta/sigma = %.3f, k=%.2f)"
          % ("PASS" if passed else "FAIL", worst, args.k))
    print("%5s %10s %10s %12s %12s" % ("cell", "g_h1", "g_h2", "|dg|", "dg/sigma"))
    for ch in checks:
        print("%5d %10.6f %10.6f %12.3e %12.3f%s"
              % (ch["cell_id"], ch["g_h1"], ch["g_h2"], ch["abs_delta_g"],
                 ch["delta_over_sigma"], "" if ch["passed"] else "  <-- FAIL"))
    print(f"Wrote {args.output_json}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
