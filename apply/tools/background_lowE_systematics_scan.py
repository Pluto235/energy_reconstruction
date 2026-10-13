#!/usr/bin/env python
"""Check 3: low-Nhit background-systematics scan (UN-FROZEN WLS annulus-quadratic).

This is a diagnostic driver + aggregator. It does NOT run any pipeline stage, does
NOT promote anything, and never writes into a frozen path. Everything it emits lives
under a ``_diag/bkgscan/`` namespace so the frozen ``poisson_pooled`` /
``poisson_unbinned`` outputs, manifests, and reports are untouched.

Purpose
-------
Test whether the low-Nhit background ``B_on`` (and hence the low-energy SED point)
is sensitive to background-model choices, using the *un-frozen* weighted-least-squares
(``--roi-fit-statistic weighted-ls``) annulus-quadratic Stage-D path. The frozen
pooled-Poisson path is deliberately avoided here: it enforces a frozen pooling
manifest with order/pooling + analytic-B_on baseline-SHA checks, so it is not a
knob one can freely vary for a systematics scan.

Variant grid (baseline + one-knob-at-a-time; NOT a full cross product)
---------------------------------------------------------------------
Off a common WLS baseline (inner=1.5, width=2.0, order=2, mask-min=1.5,
mask-r-opt-factor=2.0), one knob is varied at a time:

* ``--annulus-default-inner-deg`` in {1.0, 1.5, 2.0, 2.5}
* ``--annulus-width-deg``          in {1.5, 2.0, 2.5}
* ``--roi-surface-order``          in {1, 2}
* a widened source-mask variant (``--annulus-source-mask-min-deg`` and
  ``--annulus-source-mask-r-opt-factor`` each one step larger)

The baseline itself is one variant; the other grid points reuse the baseline for
every other knob. Duplicate variants (e.g. the baseline inner/width/order values)
are de-duplicated so the baseline is emitted once.

Modes
-----
``--emit-commands``  Print the exact ``stages/04_background.py`` variant commands
                     (one per line, into a ``_diag/bkgscan/<variant>`` namespace)
                     so the Slurm array sbatch can consume them. Only the Stage-D
                     command is knob-specific; Stage E/F/G are wired by the sbatch.
``--aggregate``      After the Stage-D->G runs exist, load each variant's Stage-D
                     NPZ + Stage-G SED summary, restrict to the two lowest Nhit
                     bins ``[100,200)`` and ``[200,300)``, tabulate per variant, and
                     emit a JSON + a summary PNG.
``--self-test``      Fabricate tiny synthetic Stage-D NPZs + Stage-G CSVs and run
                     the aggregation without any cluster data.

The Stage-D diagnostic NPZ keys read here (all written by
``stages/04_background.py`` in the WLS annulus-quadratic path):
``cell_id``, ``nhit_bin``, ``predE_bin``, ``B_on`` (with ``B_on_analytic`` as a
fallback), ``annulus_residual_rms``, ``annulus_count_residual_final``, and
``surface_order``. ``continuous_annulus_counts`` is read best-effort (it is a
top-level NPZ key only in the Poisson path; in the WLS path the per-cell OFF-annulus
count is taken from ``annulus_off_counts`` when present).

The Stage-G SED summary-CSV columns read here (written by
``stages/07_sed_points.py``): ``grouping`` (``nhit``/``predE``), ``group_label``
(the Nhit interval string), ``effective_energy_tev``, ``E2_dnde``, ``E2_dnde_err``,
and ``ratio_to_full_array_pl_ref`` (the v6-vs-Pass5 ratio).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Diagnostic namespace. Every path produced by this scan lives under here so
# nothing frozen (poisson_pooled / poisson_unbinned outputs, manifests, reports)
# is ever written.
# ---------------------------------------------------------------------------
DIAG_NAMESPACE = "_diag/bkgscan"

# The two lowest Nhit bins, per the fit selector interval strings.
LOW_NHIT_BINS: Tuple[str, ...] = ("[100,200)", "[200,300)")

# ---------------------------------------------------------------------------
# Common WLS baseline for the un-frozen annulus-quadratic Stage-D path.
# These are Stage-D --roi-* / --annulus-* knob values; every variant inherits
# them and overrides exactly one knob.
# ---------------------------------------------------------------------------
BASELINE: Dict[str, object] = {
    "annulus_default_inner_deg": 1.5,
    "annulus_width_deg": 2.0,
    "roi_surface_order": 2,
    "annulus_source_mask_min_deg": 1.5,
    "annulus_source_mask_r_opt_factor": 2.0,
}

# One-knob-at-a-time grid values.
INNER_GRID = (1.0, 1.5, 2.0, 2.5)
WIDTH_GRID = (1.5, 2.0, 2.5)
ORDER_GRID = (1, 2)
# Widened source mask: min +0.5 deg, r-opt-factor +0.5 (one step larger).
WIDE_MASK = {
    "annulus_source_mask_min_deg": 2.0,
    "annulus_source_mask_r_opt_factor": 2.5,
}


@dataclass(frozen=True)
class Variant:
    """A single Stage-D knob configuration for the scan."""

    name: str
    knob: str  # which knob was varied off the baseline
    overrides: Dict[str, object] = field(default_factory=dict)

    def config(self) -> Dict[str, object]:
        cfg = dict(BASELINE)
        cfg.update(self.overrides)
        return cfg


def _fmt(value: object) -> str:
    """Compact, filesystem-safe token for a knob value (2.0 -> ``2p0``)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return ("%g" % value).replace(".", "p").replace("-", "m")
    return str(value)


def build_variants() -> List[Variant]:
    """Baseline + one-knob-at-a-time variants, de-duplicated by name."""
    variants: List[Variant] = []
    seen: set = set()

    def _add(v: Variant) -> None:
        if v.name in seen:
            return
        seen.add(v.name)
        variants.append(v)

    # Baseline first (canonical name). Grid points that reproduce the baseline
    # knob value are collapsed into it (skipped here) so no redundant Stage-D run
    # is emitted; see _dedup_note for the record of the collapse.
    _add(Variant(name="baseline", knob="baseline", overrides={}))

    for inner in INNER_GRID:
        if float(inner) == float(BASELINE["annulus_default_inner_deg"]):
            continue
        _add(
            Variant(
                name=f"inner_{_fmt(inner)}",
                knob="annulus_default_inner_deg",
                overrides={"annulus_default_inner_deg": float(inner)},
            )
        )
    for width in WIDTH_GRID:
        if float(width) == float(BASELINE["annulus_width_deg"]):
            continue
        _add(
            Variant(
                name=f"width_{_fmt(width)}",
                knob="annulus_width_deg",
                overrides={"annulus_width_deg": float(width)},
            )
        )
    for order in ORDER_GRID:
        if int(order) == int(BASELINE["roi_surface_order"]):
            continue
        _add(
            Variant(
                name=f"order_{_fmt(order)}",
                knob="roi_surface_order",
                overrides={"roi_surface_order": int(order)},
            )
        )
    _add(
        Variant(
            name="widemask",
            knob="source_mask",
            overrides=dict(WIDE_MASK),
        )
    )
    return variants


def _dedup_note(variants: Sequence[Variant]) -> Dict[str, str]:
    """Map baseline-equivalent grid points to the canonical ``baseline`` name.

    ``inner_1p5``, ``width_2p0``, ``order_2`` all reproduce the baseline knob
    values; they are represented once as ``baseline`` in ``build_variants`` and
    listed here so the report is explicit about the collapse.
    """
    note: Dict[str, str] = {}
    if float(BASELINE["annulus_default_inner_deg"]) in INNER_GRID:
        note[f"inner_{_fmt(BASELINE['annulus_default_inner_deg'])}"] = "baseline"
    if float(BASELINE["annulus_width_deg"]) in WIDTH_GRID:
        note[f"width_{_fmt(BASELINE['annulus_width_deg'])}"] = "baseline"
    if int(BASELINE["roi_surface_order"]) in ORDER_GRID:
        note[f"order_{_fmt(BASELINE['roi_surface_order'])}"] = "baseline"
    return note


# ---------------------------------------------------------------------------
# --emit-commands
# ---------------------------------------------------------------------------
def stage_d_command(
    variant: Variant,
    *,
    python: str,
    output_root: str,
    run_id: str,
    stage_c_dir: str,
    psf_npz: str,
    cell_selection_csv: str,
    roi_fiducial_deg: float,
    roi_grid_step_deg: float,
    annulus_max_inner_deg: float,
    npz_name: str,
    extra: Sequence[str] = (),
) -> List[str]:
    """Build the argv for one un-frozen WLS annulus-quadratic Stage-D run.

    Crucially: ``--roi-fit-statistic weighted-ls`` and NO ``--pooling-manifest``,
    so the frozen manifest / baseline-SHA guards are never triggered.
    """
    cfg = variant.config()
    argv = [
        python,
        "stages/04_background.py",
        "--stage-c-dir", stage_c_dir,
        "--psf-npz", psf_npz,
        "--cell-selection-csv", cell_selection_csv,
        "--output-dir", output_root,
        "--run-id", run_id,
        "--no-promote-current",
        "--background-mode", "crab_roi_local",
        "--roi-background-method", "annulus-quadratic",
        "--roi-fit-statistic", "weighted-ls",
        "--roi-fiducial-deg", "%g" % float(roi_fiducial_deg),
        "--roi-grid-step-deg", "%g" % float(roi_grid_step_deg),
        "--annulus-default-inner-deg", "%g" % float(cfg["annulus_default_inner_deg"]),
        "--annulus-width-deg", "%g" % float(cfg["annulus_width_deg"]),
        "--annulus-max-inner-deg", "%g" % float(annulus_max_inner_deg),
        "--annulus-source-mask-min-deg", "%g" % float(cfg["annulus_source_mask_min_deg"]),
        "--annulus-source-mask-r-opt-factor", "%g" % float(cfg["annulus_source_mask_r_opt_factor"]),
        "--roi-surface-order", "%d" % int(cfg["roi_surface_order"]),
        "--npz-name", npz_name,
        "--metadata-name", npz_name.replace(".npz", "_metadata.json"),
        "--summary-csv-name", npz_name.replace(".npz", "_summary.csv"),
        "--summary-md-name", npz_name.replace(".npz", "_summary.md"),
    ]
    argv.extend(extra)
    return argv


def emit_commands(args: argparse.Namespace) -> None:
    """Print one Stage-D command per variant (one per line) for the sbatch array."""
    variants = build_variants()
    output_base = f"{args.output_prefix}/{DIAG_NAMESPACE}"
    for variant in variants:
        run_id = f"bkgscan_{variant.name}_stage_d"
        output_root = f"{output_base}/{variant.name}/stage_d"
        npz_name = f"background_bkgscan_{variant.name}.npz"
        argv = stage_d_command(
            variant,
            python=args.python,
            output_root=output_root,
            run_id=run_id,
            stage_c_dir=args.stage_c_dir,
            psf_npz=args.psf_npz,
            cell_selection_csv=args.cell_selection_csv,
            roi_fiducial_deg=args.roi_fiducial_deg,
            roi_grid_step_deg=args.roi_grid_step_deg,
            annulus_max_inner_deg=args.annulus_max_inner_deg,
            npz_name=npz_name,
        )
        print(" ".join(shlex.quote(tok) for tok in argv))


def emit_variant_names(args: argparse.Namespace) -> None:
    """Print the variant names, one per line (for the sbatch array mapping)."""
    for variant in build_variants():
        print(variant.name)


# ---------------------------------------------------------------------------
# --aggregate
# ---------------------------------------------------------------------------
def _parse_interval_low(label: str) -> float:
    """Lower bound of an interval string ``[low,high)``; +inf for open labels."""
    label = str(label).strip()
    if label.startswith("[") and "," in label:
        try:
            return float(label[1:].split(",", 1)[0])
        except ValueError:
            return math.inf
    return math.inf


def _stage_d_npz_path(base: Path, variant_name: str) -> Path:
    run_id = f"bkgscan_{variant_name}_stage_d"
    return (
        base
        / variant_name
        / "stage_d"
        / "runs"
        / run_id
        / f"background_bkgscan_{variant_name}.npz"
    )


def _stage_g_csv_path(base: Path, variant_name: str) -> Path:
    run_id = f"bkgscan_{variant_name}_stage_g"
    return (
        base
        / variant_name
        / "stage_g"
        / "runs"
        / run_id
        / f"sed_points_bkgscan_{variant_name}_summary.csv"
    )


def load_stage_d_lowE(npz_path: Path) -> List[Dict[str, object]]:
    """Read the two lowest Nhit bins from a Stage-D WLS NPZ."""
    data = np.load(npz_path, allow_pickle=False)
    cell_id = np.asarray(data["cell_id"]).astype(np.int64)
    nhit_bin = np.asarray(data["nhit_bin"]).astype(str)
    predE_bin = np.asarray(data["predE_bin"]).astype(str)
    if "B_on" in data.files:
        b_on = np.asarray(data["B_on"], dtype=np.float64)
    else:  # analytic fallback (present in WLS diagnostics)
        b_on = np.asarray(data["B_on_analytic"], dtype=np.float64)
    resid_rms = np.asarray(data["annulus_residual_rms"], dtype=np.float64)
    resid_final = np.asarray(data["annulus_count_residual_final"], dtype=np.float64)
    if "surface_order" in data.files:
        order = np.asarray(data["surface_order"], dtype=np.int64)
    else:
        order = np.full(cell_id.shape, -1, dtype=np.int64)
    # continuous_annulus_counts is a top-level key only in the Poisson path; in the
    # WLS path fall back to annulus_off_counts (per-cell OFF-annulus counts).
    if "continuous_annulus_counts" in data.files:
        ann_counts = np.asarray(data["continuous_annulus_counts"], dtype=np.float64)
    elif "annulus_off_counts" in data.files:
        ann_counts = np.asarray(data["annulus_off_counts"], dtype=np.float64)
    else:
        ann_counts = np.full(cell_id.shape, float("nan"), dtype=np.float64)

    rows: List[Dict[str, object]] = []
    for i in range(cell_id.shape[0]):
        if nhit_bin[i] not in LOW_NHIT_BINS:
            continue
        rows.append(
            {
                "cell_id": int(cell_id[i]),
                "nhit_bin": str(nhit_bin[i]),
                "predE_bin": str(predE_bin[i]),
                "B_on": float(b_on[i]),
                "annulus_residual_rms": float(resid_rms[i]),
                "annulus_count_residual_final": float(resid_final[i]),
                "surface_order": int(order[i]),
                "continuous_annulus_counts": float(ann_counts[i]),
            }
        )
    rows.sort(key=lambda r: (_parse_interval_low(r["nhit_bin"]), r["cell_id"]))
    return rows


def load_stage_g_lowE(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Read the low-Nhit ``grouping=nhit`` SED points from a Stage-G summary CSV.

    Returns a mapping ``nhit_bin -> {effective_energy_tev, E2_dnde, E2_dnde_err,
    ratio_to_full_array_pl_ref}`` for the two lowest Nhit bins.
    """
    out: Dict[str, Dict[str, float]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("grouping", "")).strip() != "nhit":
                continue
            label = str(row.get("group_label", "")).strip()
            if label not in LOW_NHIT_BINS:
                continue

            def _f(key: str) -> float:
                raw = str(row.get(key, "")).strip()
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return float("nan")

            out[label] = {
                "effective_energy_tev": _f("effective_energy_tev"),
                "E2_dnde": _f("E2_dnde"),
                "E2_dnde_err": _f("E2_dnde_err"),
                "ratio_to_full_array_pl_ref": _f("ratio_to_full_array_pl_ref"),
            }
    return out


def aggregate(args: argparse.Namespace) -> Dict[str, object]:
    """Load each variant's Stage-D + Stage-G products, tabulate, emit JSON + PNG."""
    base = Path(args.scan_dir)
    if not base.is_absolute():
        base = (REPO_ROOT / base).resolve()

    variants = build_variants()
    baseline_name = "baseline"

    per_variant: Dict[str, object] = {}
    for variant in variants:
        d_npz = _stage_d_npz_path(base, variant.name)
        g_csv = _stage_g_csv_path(base, variant.name)
        entry: Dict[str, object] = {
            "knob": variant.knob,
            "config": variant.config(),
            "stage_d_npz": str(d_npz),
            "stage_g_csv": str(g_csv),
            "present": d_npz.exists(),
        }
        if d_npz.exists():
            entry["stage_d_lowE"] = load_stage_d_lowE(d_npz)
        else:
            entry["stage_d_lowE"] = []
        entry["stage_g_lowE"] = load_stage_g_lowE(g_csv)
        per_variant[variant.name] = entry

    # Reference (baseline) B_on / SED per (nhit_bin, cell_id) and per nhit_bin.
    def _bon_by_cell(entry: Dict[str, object]) -> Dict[int, float]:
        return {int(r["cell_id"]): float(r["B_on"]) for r in entry.get("stage_d_lowE", [])}

    def _sed_by_bin(entry: Dict[str, object]) -> Dict[str, Dict[str, float]]:
        return entry.get("stage_g_lowE", {})  # type: ignore[return-value]

    base_entry = per_variant.get(baseline_name, {})
    base_bon = _bon_by_cell(base_entry) if base_entry.get("present") else {}
    base_sed = _sed_by_bin(base_entry)

    # Per-variant shifts vs baseline.
    for name, entry in per_variant.items():
        bon = _bon_by_cell(entry)
        b_shifts: List[float] = []
        for cid, val in bon.items():
            ref = base_bon.get(cid)
            if ref and math.isfinite(ref) and math.isfinite(val):
                b_shifts.append((val - ref) / ref)
        entry["b_on_rel_shift_vs_baseline"] = {
            "max_abs": max((abs(s) for s in b_shifts), default=0.0),
            "mean": float(np.mean(b_shifts)) if b_shifts else 0.0,
            "n_cells": len(b_shifts),
        }
        sed = _sed_by_bin(entry)
        sed_shifts: Dict[str, float] = {}
        for nb in LOW_NHIT_BINS:
            ref = base_sed.get(nb, {}).get("E2_dnde")
            cur = sed.get(nb, {}).get("E2_dnde")
            if ref and math.isfinite(ref) and cur is not None and math.isfinite(cur):
                sed_shifts[nb] = (cur - ref) / ref
        entry["sed_e2dnde_rel_shift_vs_baseline"] = sed_shifts

    payload: Dict[str, object] = {
        "check": "3_low_nhit_background_systematics_wls_annulus_quadratic",
        "diag_namespace": DIAG_NAMESPACE,
        "scan_dir": str(base),
        "low_nhit_bins": list(LOW_NHIT_BINS),
        "baseline": BASELINE,
        "baseline_name": baseline_name,
        "dedup_collapsed_to_baseline": _dedup_note(variants),
        "variants": [v.name for v in variants],
        "per_variant": per_variant,
        "summary": _scan_summary(per_variant),
    }

    out_json = Path(args.out_json) if args.out_json else base / "bkgscan_lowE_systematics.json"
    if not out_json.is_absolute():
        out_json = (REPO_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"Wrote {out_json}")

    out_png = Path(args.out_png) if args.out_png else base / "bkgscan_lowE_systematics.png"
    if not out_png.is_absolute():
        out_png = (REPO_ROOT / out_png).resolve()
    try:
        _plot_scan(per_variant, out_png)
        print(f"Wrote {out_png}")
    except Exception as exc:  # plotting is best-effort; never fail the aggregation
        print(f"Skipped PNG ({out_png}): {exc}")

    _print_table(per_variant)
    return payload


def _scan_summary(per_variant: Dict[str, object]) -> Dict[str, object]:
    max_bon = 0.0
    max_sed = 0.0
    present = 0
    for entry in per_variant.values():
        if entry.get("present"):
            present += 1
        max_bon = max(max_bon, float(entry.get("b_on_rel_shift_vs_baseline", {}).get("max_abs", 0.0)))
        for val in entry.get("sed_e2dnde_rel_shift_vs_baseline", {}).values():
            max_sed = max(max_sed, abs(float(val)))
    return {
        "n_variants": len(per_variant),
        "n_present": present,
        "max_abs_b_on_rel_shift": max_bon,
        "max_abs_sed_e2dnde_rel_shift": max_sed,
    }


def _print_table(per_variant: Dict[str, object]) -> None:
    hdr = ["variant", "knob", "present", "maxdB_on%", "dSED[100,200)%", "dSED[200,300)%"]
    widths = [16, 28, 8, 11, 15, 15]
    print("".join(h.ljust(w) for h, w in zip(hdr, widths)))
    for name, entry in per_variant.items():
        s100 = entry.get("sed_e2dnde_rel_shift_vs_baseline", {}).get("[100,200)")
        s200 = entry.get("sed_e2dnde_rel_shift_vs_baseline", {}).get("[200,300)")
        cells = [
            name,
            str(entry.get("knob", "")),
            "yes" if entry.get("present") else "no",
            f"{100 * float(entry.get('b_on_rel_shift_vs_baseline', {}).get('max_abs', 0.0)):+.2f}",
            f"{100 * s100:+.2f}" if isinstance(s100, (int, float)) and math.isfinite(s100) else "-",
            f"{100 * s200:+.2f}" if isinstance(s200, (int, float)) and math.isfinite(s200) else "-",
        ]
        print("".join(c.ljust(w) for c, w in zip(cells, widths)))


def _plot_scan(per_variant: Dict[str, object], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_variant.keys())
    x = np.arange(len(names))
    bon = [100.0 * float(per_variant[n].get("b_on_rel_shift_vs_baseline", {}).get("max_abs", 0.0)) for n in names]
    sed100 = [
        100.0 * float(per_variant[n].get("sed_e2dnde_rel_shift_vs_baseline", {}).get("[100,200)", float("nan")))
        for n in names
    ]
    sed200 = [
        100.0 * float(per_variant[n].get("sed_e2dnde_rel_shift_vs_baseline", {}).get("[200,300)", float("nan")))
        for n in names
    ]

    fig, axes = plt.subplots(2, 1, figsize=(max(6.0, 0.7 * len(names)), 6.4), sharex=True)
    axes[0].bar(x, bon, color="#3b6ea5")
    axes[0].axhline(0.0, color="k", linewidth=0.6)
    axes[0].set_ylabel("max |dB_on| vs baseline [%]")
    axes[0].set_title("Check 3: low-Nhit WLS annulus-quadratic background systematics")
    axes[0].grid(alpha=0.25, linewidth=0.4)

    w = 0.4
    axes[1].bar(x - w / 2, sed100, width=w, label="[100,200)", color="#a53b6e")
    axes[1].bar(x + w / 2, sed200, width=w, label="[200,300)", color="#6ea53b")
    axes[1].axhline(0.0, color="k", linewidth=0.6)
    axes[1].set_ylabel("low-E SED-point shift vs baseline [%]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# --self-test
# ---------------------------------------------------------------------------
def _write_synthetic_stage_d(npz_path: Path, *, b_on_scale: float) -> None:
    """Fabricate a tiny Stage-D WLS NPZ with the two lowest Nhit bins."""
    cell_id = np.array([1, 2, 3, 4], dtype=np.int32)
    nhit_bin = np.array(["[100,200)", "[100,200)", "[200,300)", "[200,300)"], dtype="U32")
    predE_bin = np.array(["[2,2.5)", "[2.5,3)", "[2,2.5)", "[2.5,3)"], dtype="U32")
    b_on = np.array([1000.0, 1500.0, 800.0, 1200.0], dtype=np.float64) * b_on_scale
    np.savez_compressed(
        npz_path,
        cell_id=cell_id,
        nhit_bin=nhit_bin,
        predE_bin=predE_bin,
        B_on=b_on,
        B_on_analytic=b_on.copy(),
        annulus_residual_rms=np.array([0.9, 1.1, 0.8, 1.0], dtype=np.float64),
        annulus_count_residual_final=np.array([0.5, -0.4, 0.3, -0.2], dtype=np.float64),
        surface_order=np.array([2, 2, 2, 2], dtype=np.int64),
        annulus_off_counts=np.array([5000.0, 5200.0, 4800.0, 5100.0], dtype=np.float64),
    )


def _write_synthetic_stage_g(csv_path: Path, *, sed_scale: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grouping",
        "group_label",
        "cell_ids",
        "effective_energy_tev",
        "E2_dnde",
        "E2_dnde_err",
        "ratio_to_full_array_pl_ref",
    ]
    rows = [
        {
            "grouping": "nhit",
            "group_label": "[100,200)",
            "cell_ids": "1;2",
            "effective_energy_tev": 0.5,
            "E2_dnde": 4.0e-11 * sed_scale,
            "E2_dnde_err": 0.4e-11,
            "ratio_to_full_array_pl_ref": 1.02 * sed_scale,
        },
        {
            "grouping": "nhit",
            "group_label": "[200,300)",
            "cell_ids": "3;4",
            "effective_energy_tev": 0.9,
            "E2_dnde": 3.0e-11 * sed_scale,
            "E2_dnde_err": 0.3e-11,
            "ratio_to_full_array_pl_ref": 0.98 * sed_scale,
        },
        {
            "grouping": "predE",
            "group_label": "[2,2.5)",
            "cell_ids": "1;3",
            "effective_energy_tev": 0.6,
            "E2_dnde": 3.5e-11,
            "E2_dnde_err": 0.35e-11,
            "ratio_to_full_array_pl_ref": 1.0,
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def self_test() -> None:
    """Fabricate synthetic Stage-D/Stage-G products and exercise the aggregator."""
    variants = build_variants()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / DIAG_NAMESPACE
        # Give each variant a small, distinct perturbation so shifts are non-trivial.
        for i, variant in enumerate(variants):
            scale = 1.0 + 0.01 * i
            d_npz = _stage_d_npz_path(base, variant.name)
            d_npz.parent.mkdir(parents=True, exist_ok=True)
            _write_synthetic_stage_d(d_npz, b_on_scale=scale)
            g_csv = _stage_g_csv_path(base, variant.name)
            _write_synthetic_stage_g(g_csv, sed_scale=scale)

        ns = argparse.Namespace(
            scan_dir=str(base),
            out_json=str(base / "bkgscan_lowE_systematics.json"),
            out_png=str(base / "bkgscan_lowE_systematics.png"),
        )
        payload = aggregate(ns)

        # --- assertions ---
        assert payload["summary"]["n_variants"] == len(variants), "variant count mismatch"
        assert payload["summary"]["n_present"] == len(variants), "not all synthetic variants loaded"
        # Baseline must have zero self-shift.
        base_entry = payload["per_variant"]["baseline"]
        assert base_entry["b_on_rel_shift_vs_baseline"]["max_abs"] == 0.0, "baseline B_on self-shift != 0"
        for nb, val in base_entry["sed_e2dnde_rel_shift_vs_baseline"].items():
            assert abs(val) < 1e-12, f"baseline SED self-shift != 0 for {nb}"
        # Every variant must have exactly the two lowest Nhit bins in Stage-D.
        for name, entry in payload["per_variant"].items():
            bins = {r["nhit_bin"] for r in entry["stage_d_lowE"]}
            assert bins == set(LOW_NHIT_BINS), f"{name} Stage-D low-Nhit bins wrong: {bins}"
            assert set(entry["stage_g_lowE"].keys()) == set(LOW_NHIT_BINS), f"{name} Stage-G bins wrong"
        # Interval lower-bound parsing.
        assert _parse_interval_low("[100,200)") == 100.0
        assert _parse_interval_low("[200,300)") == 200.0
        assert math.isinf(_parse_interval_low("all"))
        # emit-commands sanity: WLS, no pooling manifest, into _diag/bkgscan.
        argv = stage_d_command(
            variants[1],
            python="python",
            output_root=f"apply/output/x/{DIAG_NAMESPACE}/v/stage_d",
            run_id="r",
            stage_c_dir="sc",
            psf_npz="psf.npz",
            cell_selection_csv="sel.csv",
            roi_fiducial_deg=6.0,
            roi_grid_step_deg=0.1,
            annulus_max_inner_deg=4.5,
            npz_name="background_bkgscan_v.npz",
        )
        joined = " ".join(argv)
        assert "--roi-fit-statistic weighted-ls" in joined, "self-test: not WLS"
        assert "--pooling-manifest" not in joined, "self-test: pooling manifest leaked in"
        assert DIAG_NAMESPACE in joined, "self-test: not under _diag/bkgscan namespace"
    print("SELF-TEST PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-commands", action="store_true", help="Print Stage-D variant commands for the sbatch array.")
    mode.add_argument("--emit-variant-names", action="store_true", help="Print variant names (one per line).")
    mode.add_argument("--aggregate", action="store_true", help="Load Stage-D + Stage-G products and tabulate.")
    mode.add_argument("--self-test", action="store_true", help="Run on synthetic NPZs/CSVs; no cluster data.")

    # --emit-commands inputs (baseline paths; the sbatch overrides via env).
    parser.add_argument("--python", type=str, default="python", help="Python interpreter for the emitted Stage-D commands.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="apply/output/v6_64748_nhit100_reselect44_split56_miss030",
        help="Output prefix; the scan namespace <prefix>/_diag/bkgscan/<variant> lives under it.",
    )
    parser.add_argument("--stage-c-dir", type=str, default="")
    parser.add_argument("--psf-npz", type=str, default="")
    parser.add_argument("--cell-selection-csv", type=str, default="")
    parser.add_argument("--roi-fiducial-deg", type=float, default=6.0)
    parser.add_argument("--roi-grid-step-deg", type=float, default=0.1)
    parser.add_argument("--annulus-max-inner-deg", type=float, default=4.5)

    # --aggregate inputs.
    parser.add_argument("--scan-dir", type=str, default="", help="The <prefix>/_diag/bkgscan directory to aggregate.")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--out-png", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.emit_variant_names:
        emit_variant_names(args)
        return
    if args.emit_commands:
        emit_commands(args)
        return
    if args.aggregate:
        if not args.scan_dir:
            raise SystemExit("--aggregate requires --scan-dir")
        aggregate(args)
        return


if __name__ == "__main__":
    main()
