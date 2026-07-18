#!/usr/bin/env python
"""Stage-1 prototype comparison: binned vs unbinned quadratic B_on.

Reads Stage-D output NPZs (each produced by ``apply/stages/04_background.py`` with the
same inputs but a different ``--poisson-quadratic-fit`` mode / grid / MJD window) and
reports, for the order-2 target cells, how the radial trace tau = c_xx + c_yy and the
analytic B_on change between the binned pixel fit and the grid-free unbinned fit.

It does not run Stage D, does not promote anything, and does not touch the frozen
experiment. It only reads NPZs and emits a JSON + a table.

Key facts it exploits (established in review): for a source-centered aperture
B_on = N_ann * (pi r^2 + 0.25 pi r^4 tau) / (pi(o^2-i^2) + 0.25 pi(o^4-i^4) tau), so the
whole quadratic-cell B_on is a function of the single scalar tau. The unbinned B_on is
grid-independent by construction; the ``--unbinned-grid-npz`` inputs demonstrate that its
B_on envelope across grid steps is ~0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_TARGET_CELLS = "1,2,4,5,6,19,20"


def _load(npz_path: str) -> dict[int, dict[str, float]]:
    data = np.load(npz_path, allow_pickle=False)
    cell_id = np.asarray(data["cell_id"], dtype=np.int64)
    shape = np.asarray(data["surface_shape_coefficients"], dtype=np.float64)
    b_on = np.asarray(data["B_on"], dtype=np.float64)
    order = np.asarray(data["surface_order"], dtype=np.int64)
    n_ann = np.asarray(data["continuous_annulus_counts"], dtype=np.float64)
    out: dict[int, dict[str, float]] = {}
    for i, cid in enumerate(cell_id):
        out[int(cid)] = {
            "tau": float(shape[i, 3] + shape[i, 5]),
            "b_on": float(b_on[i]),
            "order": int(order[i]),
            "n_ann": float(n_ann[i]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binned-npz", required=True, help="Stage-D NPZ, --poisson-quadratic-fit binned (nominal grid).")
    parser.add_argument("--unbinned-npz", required=True, help="Stage-D NPZ, --poisson-quadratic-fit unbinned (nominal grid).")
    parser.add_argument("--unbinned-grid-npz", action="append", default=[], help="Additional unbinned Stage-D NPZ at other grid steps (repeatable) to show B_on grid-invariance.")
    parser.add_argument("--half1-npz", default=None, help="Unbinned Stage-D NPZ, first MJD half.")
    parser.add_argument("--half2-npz", default=None, help="Unbinned Stage-D NPZ, second MJD half.")
    parser.add_argument("--bootstrap-sigma-json", default=None, help="Optional JSON mapping cell_id -> sigma(B_on) from the bootstrap covariance diagonal, for acceptance.")
    parser.add_argument("--target-cells", default=DEFAULT_TARGET_CELLS)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    targets = [int(v) for v in str(args.target_cells).split(",") if v.strip()]
    binned = _load(args.binned_npz)
    unbinned = _load(args.unbinned_npz)
    grids = [_load(p) for p in args.unbinned_grid_npz]
    half1 = _load(args.half1_npz) if args.half1_npz else None
    half2 = _load(args.half2_npz) if args.half2_npz else None
    sigma_bon = None
    if args.bootstrap_sigma_json:
        raw = json.loads(Path(args.bootstrap_sigma_json).read_text())
        sigma_bon = {int(k): float(v) for k, v in raw.items()}

    rows = []
    for cid in targets:
        b = binned[cid]
        u = unbinned[cid]
        # Unbinned B_on across all provided grids (nominal + extra) -> envelope.
        u_bons = [u["b_on"]] + [g[cid]["b_on"] for g in grids if cid in g]
        u_env = (max(u_bons) - min(u_bons)) if len(u_bons) > 1 else 0.0
        d_bon_rel = (u["b_on"] - b["b_on"]) / b["b_on"] if b["b_on"] else float("nan")
        row = {
            "cell_id": cid,
            "order_binned": b["order"],
            "order_unbinned": u["order"],
            "n_ann": u["n_ann"],
            "tau_binned": b["tau"],
            "tau_unbinned": u["tau"],
            "d_tau": u["tau"] - b["tau"],
            "b_on_binned": b["b_on"],
            "b_on_unbinned": u["b_on"],
            "d_b_on_rel": d_bon_rel,
            "unbinned_grid_bon_envelope": u_env,
            "unbinned_grid_bon_envelope_rel": (u_env / u["b_on"]) if u["b_on"] else float("nan"),
        }
        if half1 and half2:
            row["tau_h1"] = half1[cid]["tau"]
            row["tau_h2"] = half2[cid]["tau"]
            row["b_on_h1"] = half1[cid]["b_on"]
            row["b_on_h2"] = half2[cid]["b_on"]
            row["d_tau_split"] = half1[cid]["tau"] - half2[cid]["tau"]
            row["d_b_on_split_rel"] = (
                (half1[cid]["b_on"] - half2[cid]["b_on"]) / u["b_on"] if u["b_on"] else float("nan")
            )
        if sigma_bon and cid in sigma_bon and sigma_bon[cid] > 0:
            row["d_b_on_over_sigma"] = (u["b_on"] - b["b_on"]) / sigma_bon[cid]
        rows.append(row)

    # Table
    hdr = ["cell", "N_ann", "tau_bin", "tau_unb", "d_tau", "B_bin", "B_unb", "dB%", "unb_env%"]
    if half1 and half2:
        hdr += ["dtau_split", "dB_split%"]
    if sigma_bon:
        hdr += ["dB/sig"]
    widths = [6, 12, 10, 10, 10, 12, 12, 8, 9, 11, 10, 8]
    print("".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        cells = [
            str(r["cell_id"]),
            f"{r['n_ann']:.0f}",
            f"{r['tau_binned']:.5f}",
            f"{r['tau_unbinned']:.5f}",
            f"{r['d_tau']:+.5f}",
            f"{r['b_on_binned']:.1f}",
            f"{r['b_on_unbinned']:.1f}",
            f"{100*r['d_b_on_rel']:+.2f}",
            f"{100*r['unbinned_grid_bon_envelope_rel']:.4f}",
        ]
        if half1 and half2:
            cells += [f"{r['d_tau_split']:+.5f}", f"{100*r['d_b_on_split_rel']:+.2f}"]
        if sigma_bon:
            cells += [f"{r.get('d_b_on_over_sigma', float('nan')):+.3f}"]
        print("".join(c.rjust(w) for c, w in zip(cells, widths)))

    payload = {
        "target_cells": targets,
        "binned_npz": args.binned_npz,
        "unbinned_npz": args.unbinned_npz,
        "unbinned_grid_npz": args.unbinned_grid_npz,
        "half1_npz": args.half1_npz,
        "half2_npz": args.half2_npz,
        "rows": rows,
        "summary": {
            "max_unbinned_grid_bon_envelope_rel": max((r["unbinned_grid_bon_envelope_rel"] for r in rows), default=0.0),
            "max_abs_d_b_on_rel": max((abs(r["d_b_on_rel"]) for r in rows), default=0.0),
            "max_abs_d_tau_split": max((abs(r.get("d_tau_split", 0.0)) for r in rows), default=0.0) if half1 and half2 else None,
        },
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
