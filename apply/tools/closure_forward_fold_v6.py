#!/usr/bin/env python
"""Forward-fold self-consistency closure for the Stage F/G pipeline.

Given a Stage A 2D response and a real Stage E signal NPZ (used only to copy
B_on, cell order, and the error model), this tool forward-folds an *injected*
analytic spectrum (PL or LogPar) through the SAME ``model_counts()`` math that
``stages/06_fit.py`` uses, and writes a MOCK Stage E signal NPZ whose per-cell
``excess`` (and derived counts) are the forward-folded expectation. The mock
NPZ matches the schema ``stages/05_signal.py`` produces, so the real
``06_fit.py`` + ``07_sed_points.py`` can be run on it later (by an sbatch the
user writes) and should recover the injected spectrum -- a closure test.

Optionally ``--true-energy-scale-delta`` relabels the true-energy axis used to
build the flux integral (``logE_true += log10(1+delta)``), matching the guarded
``--energy-scale-delta`` flag added to ``06_fit.py``. Injecting a spectrum with
a non-zero delta here and re-fitting with the matching delta in ``06_fit.py``
forms an energy-scale "bridge" test.

This tool does NOT run Stage F or Stage G; it only produces the mock Stage E
NPZ and its metadata JSON. A ``--self-test`` mode fabricates a tiny synthetic
response so the forward-fold and NPZ round-trip can be smoked locally without
cluster data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGE_06_FIT = REPO_ROOT / "apply" / "stages" / "06_fit.py"
M2_TO_CM2 = 1.0e4


def load_stage_06_module():
    """Load ``stages/06_fit.py`` as a module so its helpers can be reused."""
    spec = importlib.util.spec_from_file_location("stage_06_fit", STAGE_06_FIT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Stage F module from {STAGE_06_FIT}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/typing forward-ref resolution can find it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward-fold an injected Crab spectrum into a mock Stage E signal NPZ for closure tests."
    )
    parser.add_argument("--response-npz", type=str, default=None)
    parser.add_argument("--response-metadata", type=str, default=None)
    parser.add_argument(
        "--signal-npz",
        type=str,
        default=None,
        help="Real Stage E signal NPZ; B_on, cell order, and containment are copied from it.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--npz-name", type=str, default="signal_closure_v6.npz")
    parser.add_argument("--metadata-name", type=str, default="signal_closure_v6_metadata.json")

    parser.add_argument("--pivot-tev", type=float, default=3.0)
    parser.add_argument("--model", choices=["logpar", "powerlaw"], default="logpar")
    parser.add_argument("--phi0", type=float, default=2.114e-12, help="Normalization at pivot [TeV^-1 cm^-2 s^-1].")
    parser.add_argument("--alpha", type=float, default=2.69, help="LogPar alpha (spectral index at pivot).")
    parser.add_argument("--gamma", type=float, default=2.69, help="Powerlaw spectral index.")
    parser.add_argument("--beta", type=float, default=0.0, help="LogPar curvature.")
    parser.add_argument(
        "--true-energy-scale-delta",
        type=float,
        default=0.0,
        help=(
            "Inject a global true-energy-scale offset: relabel the response true-energy axis "
            "used for the flux integral by logE_true += log10(1+delta) before integrating."
        ),
    )

    # Exposure geometry (Crab defaults, mirrors 06_fit.py).
    parser.add_argument("--source-files-csv", type=str, default=None)
    parser.add_argument("--stage-c-dir", type=str, default=None)
    parser.add_argument("--source-ra-deg", type=float, default=83.63)
    parser.add_argument("--source-dec-deg", type=float, default=22.01)
    parser.add_argument("--lhaaso-lat-deg", type=float, default=29.45)
    parser.add_argument("--lhaaso-lon-deg", type=float, default=100.14)
    parser.add_argument("--exposure-sample-step-sec", type=float, default=60.0)
    parser.add_argument("--energy-quadrature-points", type=int, default=64)

    parser.add_argument(
        "--self-test",
        action="store_true",
        default=False,
        help="Fabricate a tiny synthetic response + signal, run the forward-fold, and check the NPZ round-trips.",
    )
    return parser.parse_args()


def spectrum_spec(args: argparse.Namespace) -> Tuple[str, Dict[str, float]]:
    """Map the CLI spectrum flags to (model_name, params) accepted by model_counts()."""
    if args.model == "powerlaw":
        return "pl", {"phi0": float(args.phi0), "gamma": float(args.gamma)}
    return "logpar", {"phi0": float(args.phi0), "alpha": float(args.alpha), "beta": float(args.beta)}


def shifted_loge_edges(loge_edges: np.ndarray, delta: float) -> np.ndarray:
    """Relabel true-energy bin edges by logE_true += log10(1+delta).

    Shifting the true-energy axis used to build the flux integral answers the
    energy-scale bridge question: if each reco cell's true energy is globally
    off by ``delta``, how does the forward-folded expectation move.
    """
    if float(delta) == 0.0:
        return np.asarray(loge_edges, dtype=np.float64)
    if float(delta) <= -1.0:
        raise ValueError("--true-energy-scale-delta must be > -1 (1+delta must stay positive)")
    return np.asarray(loge_edges, dtype=np.float64) + math.log10(1.0 + float(delta))


def forward_fold_excess(
    stage06,
    *,
    a_eff_m2: np.ndarray,
    containment: np.ndarray,
    theta_exposure_sec: np.ndarray,
    loge_edges: np.ndarray,
    model_name: str,
    params: Dict[str, float],
    pivot_tev: float,
    quadrature_points: int,
    energy_scale_delta: float,
) -> np.ndarray:
    """Per-cell forward-folded expected excess via the same model_counts() math."""
    loge_for_integral = shifted_loge_edges(loge_edges, energy_scale_delta)
    return stage06.model_counts(
        a_eff_m2,
        containment,
        theta_exposure_sec,
        loge_for_integral,
        model_name=model_name,
        params=params,
        pivot_tev=float(pivot_tev),
        quadrature_points=int(quadrature_points),
    )


def build_mock_signal_arrays(
    real_signal: Dict[str, np.ndarray],
    predicted_excess: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assemble a mock Stage E NPZ payload matching 05_signal.py's schema.

    Counts are made self-consistent with the forward-folded excess:
        excess = predicted (forward-fold), rounded N_on = round(excess + B_on),
        excess (final) = N_on - B_on, so the downstream chi2 is well-posed.
    B_on / containment / cell labels / off-count fields are copied from the
    real Stage E NPZ so the error model and Stage F contract still hold.
    """
    n_cells = int(np.asarray(real_signal["cell_id"]).shape[0])
    b_on = np.asarray(real_signal["B_on"], dtype=np.float64)
    if b_on.shape[0] != n_cells:
        raise ValueError("Real Stage E B_on length does not match cell_id length")

    # Poissonize the forward-fold to integer on-counts so N_on stays integral
    # like the real Stage E scan, then recompute excess = N_on - B_on.
    n_on_float = np.asarray(predicted_excess, dtype=np.float64) + b_on
    n_on = np.rint(np.maximum(n_on_float, 0.0)).astype(np.int64)
    excess = n_on.astype(np.float64) - b_on
    excess_err_stat = np.sqrt(np.maximum(n_on.astype(np.float64), 0.0))
    excess_err_conservative = np.sqrt(np.maximum(n_on.astype(np.float64) + b_on, 0.0))

    def copy_or_default(name: str, default_value, dtype) -> np.ndarray:
        if name in real_signal:
            return np.asarray(real_signal[name], dtype=dtype)
        return np.full(n_cells, default_value, dtype=dtype)

    # known-background Poisson significance for parity with the real schema.
    stat_sigma = np.zeros(n_cells, dtype=np.float64)
    for i in range(n_cells):
        b = float(b_on[i])
        n = float(n_on[i])
        if b > 0.0 and n > 0.0:
            term = n * math.log(n / b) - (n - b)
        elif b > 0.0:
            term = b
        else:
            term = 0.0
        s = math.sqrt(max(0.0, 2.0 * term))
        stat_sigma[i] = -s if n < b else s

    payload: Dict[str, np.ndarray] = {
        "cell_id": np.asarray(real_signal["cell_id"], dtype=np.int32),
        "nhit_bin": np.asarray(real_signal["nhit_bin"]).astype("U32"),
        "predE_bin": np.asarray(real_signal["predE_bin"]).astype("U32"),
        "r_opt_deg": copy_or_default("r_opt_deg", np.nan, np.float32),
        "sigma_deg": copy_or_default("sigma_deg", np.nan, np.float32),
        "containment_r_opt": np.asarray(real_signal["containment_r_opt"], dtype=np.float32),
        "N_on": n_on,
        "B_on": b_on.astype(np.float64),
        "N_off": copy_or_default("N_off", np.nan, np.float64),
        "alpha": copy_or_default("alpha", np.nan, np.float64),
        "excess": excess.astype(np.float64),
        "excess_err_stat": excess_err_stat.astype(np.float64),
        "excess_err_conservative": excess_err_conservative.astype(np.float64),
        "known_b_sigma": stat_sigma.astype(np.float64),
        "li_ma_sigma": np.full(n_cells, np.nan, dtype=np.float64),
        "formal_sigma": stat_sigma.astype(np.float64),
        "background_mode": copy_or_default("background_mode", "closure_mock", "U64"),
        "background_form": copy_or_default("background_form", "direct_expectation", "U64"),
        "statistic_kind": copy_or_default("statistic_kind", "known_background_poisson", "U64"),
    }
    if "roi_fiducial_deg" in real_signal:
        payload["roi_fiducial_deg"] = np.asarray(real_signal["roi_fiducial_deg"], dtype=np.float64)
    return payload


def make_metadata(
    *,
    args: argparse.Namespace,
    model_name: str,
    params: Dict[str, float],
    response_npz: Optional[Path],
    signal_npz: Optional[Path],
    source_files_csv: Optional[Path],
    exposure_meta: Optional[Dict[str, object]],
    predicted_excess: np.ndarray,
    n_cells: int,
) -> Dict[str, object]:
    """Emit a Stage-E-shaped metadata JSON that Stage F/G validators accept.

    Includes ``quality_gate.promotable``, ``stage_d_contract`` and
    ``inputs.stage_c_dir`` so ``06_fit.py`` can resolve exposure and pass
    ``validate_inputs``.
    """
    return {
        "description": "MOCK Stage E signal produced by tools/closure_forward_fold_v6.py (forward-fold closure test).",
        "run_id": "closure_forward_fold_v6",
        "closure_forward_fold": {
            "tool": "tools/closure_forward_fold_v6.py",
            "injected_model": model_name,
            "injected_parameters": {str(k): float(v) for k, v in params.items()},
            "pivot_tev": float(args.pivot_tev),
            "true_energy_scale_delta": float(args.true_energy_scale_delta),
            "energy_quadrature_points": int(args.energy_quadrature_points),
            "counts_formula": "containment * 1e4 * sum_E,theta(A_eff_m2 * theta_exposure_sec * integral_flux_TeV)",
            "n_cells": int(n_cells),
            "total_predicted_excess": float(np.sum(predicted_excess)),
            "response_npz": str(response_npz) if response_npz is not None else None,
        },
        "inputs": {
            "response_npz": str(response_npz) if response_npz is not None else None,
            "real_signal_npz": str(signal_npz) if signal_npz is not None else None,
            "stage_c_dir": str(source_files_csv.parent) if source_files_csv is not None else args.stage_c_dir,
            "source_files_csv": str(source_files_csv) if source_files_csv is not None else None,
        },
        "source": {
            "name": "Crab",
            "ra_deg": float(args.source_ra_deg),
            "dec_deg": float(args.source_dec_deg),
        },
        "stage_d_contract": {
            "background_mode": "closure_mock",
            "background_form": "direct_expectation",
            "statistic_kind": "known_background_poisson",
            "strict_contract": False,
            "promotable_contract": False,
            "diagnostic_reasons": ["Mock Stage E from forward-fold closure tool; not a real background contract."],
            "warnings": [],
            "metadata_summary": {"method": "closure_forward_fold_mock"},
        },
        "quality_gate": {
            "status": "closure_mock",
            "reason": "Forward-fold closure mock; excess is analytic, quality gate is informational only.",
            "promotable": True,
        },
        "statistic": {
            "formal_significance": "known_background_poisson_mock",
            "excess_formula": "N_on - B_on",
        },
        "exposure": exposure_meta,
    }


def run(args: argparse.Namespace, stage06, *, self_test: bool = False) -> Dict[str, object]:
    model_name, params = spectrum_spec(args)

    if self_test:
        response, real_signal, loge_edges, theta_edges, theta_exposure, exposure_meta = _fabricate_synthetic(stage06)
        response_npz = None
        signal_npz = None
        source_files_csv = None
    else:
        if not args.response_npz or not args.signal_npz or not args.output_dir:
            raise ValueError("--response-npz, --signal-npz, and --output-dir are required outside --self-test")
        response_npz = Path(args.response_npz).resolve()
        signal_npz = Path(args.signal_npz).resolve()
        response_metadata_path = Path(args.response_metadata).resolve() if args.response_metadata else None
        signal_metadata_path = signal_npz.with_name(
            signal_npz.stem.replace(".npz", "") + "_metadata.json"
        )
        response, real_signal = stage06.load_arrays(response_npz, signal_npz)
        loge_edges = np.asarray(response["logE_true_edges"], dtype=np.float64)
        theta_edges = np.asarray(response["theta_true_edges_deg"], dtype=np.float64)

        # Resolve the exposure the same way 06_fit.py does.
        signal_metadata: Dict[str, object] = {}
        for candidate in (
            Path(str(signal_npz).replace(".npz", "_metadata.json")),
            signal_npz.with_name("signal_v1_metadata.json"),
        ):
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as f:
                    signal_metadata = json.load(f)
                break
        source_files_csv = stage06.resolve_source_files_csv(args, signal_metadata)
        theta_exposure, exposure_meta = stage06.compute_theta_exposure(
            source_files_csv,
            theta_edges,
            source_ra_deg=float(args.source_ra_deg),
            source_dec_deg=float(args.source_dec_deg),
            latitude_deg=float(args.lhaaso_lat_deg),
            longitude_east_deg=float(args.lhaaso_lon_deg),
            sample_step_sec=float(args.exposure_sample_step_sec),
        )

    a_eff = np.asarray(response["a_eff"], dtype=np.float64)
    containment = np.asarray(real_signal["containment_r_opt"], dtype=np.float64)

    predicted_excess = forward_fold_excess(
        stage06,
        a_eff_m2=a_eff,
        containment=containment,
        theta_exposure_sec=theta_exposure,
        loge_edges=loge_edges,
        model_name=model_name,
        params=params,
        pivot_tev=float(args.pivot_tev),
        quadrature_points=int(args.energy_quadrature_points),
        energy_scale_delta=float(args.true_energy_scale_delta),
    )

    mock = build_mock_signal_arrays(real_signal, predicted_excess)
    n_cells = int(mock["cell_id"].shape[0])

    metadata = make_metadata(
        args=args,
        model_name=model_name,
        params=params,
        response_npz=response_npz if not self_test else None,
        signal_npz=signal_npz if not self_test else None,
        source_files_csv=source_files_csv if not self_test else None,
        exposure_meta=exposure_meta,
        predicted_excess=predicted_excess,
        n_cells=n_cells,
    )

    result: Dict[str, object] = {
        "model_name": model_name,
        "params": params,
        "n_cells": n_cells,
        "total_predicted_excess": float(np.sum(predicted_excess)),
        "total_mock_N_on": int(np.sum(mock["N_on"])),
        "mock_payload": mock,
        "metadata": metadata,
        "predicted_excess": predicted_excess,
    }

    if not self_test:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / args.npz_name
        metadata_path = output_dir / args.metadata_name
        np.savez_compressed(npz_path, **mock)
        write_json(metadata_path, metadata)
        result["npz_path"] = npz_path
        result["metadata_path"] = metadata_path

    return result


def _fabricate_synthetic(stage06):
    """Tiny synthetic response + Stage E signal for --self-test (3 cells x 5 logE x 2 theta)."""
    n_cells, n_loge, n_theta = 3, 5, 2
    rng = np.random.default_rng(20260718)
    # Effective area in m^2, monotone-ish increasing with true energy.
    a_eff = np.abs(rng.uniform(50.0, 5000.0, size=(n_cells, n_loge, n_theta)))
    # log10(E_true / GeV) edges spanning ~1-100 TeV (10^3.5 .. 10^5 GeV).
    loge_edges = np.linspace(3.5, 5.0, n_loge + 1)
    theta_edges = np.array([0.0, 20.0, 40.0], dtype=np.float64)
    theta_exposure = np.array([3.0e6, 1.0e6], dtype=np.float64)  # seconds
    response = {
        "a_eff": a_eff.astype(np.float64),
        "logE_true_edges": loge_edges.astype(np.float64),
        "theta_true_edges_deg": theta_edges.astype(np.float64),
        "cell_id": np.arange(1, n_cells + 1, dtype=np.int32),
        "nhit_bin": np.asarray([f"[{i},{i + 1})" for i in range(n_cells)], dtype="U32"),
        "predE_bin": np.asarray([f"[{i},{i + 1})" for i in range(n_cells)], dtype="U32"),
    }
    real_signal = {
        "cell_id": np.arange(1, n_cells + 1, dtype=np.int32),
        "nhit_bin": response["nhit_bin"].copy(),
        "predE_bin": response["predE_bin"].copy(),
        "containment_r_opt": np.ones(n_cells, dtype=np.float64),
        "B_on": np.array([500.0, 120.0, 30.0], dtype=np.float64),
        "N_on": np.array([600, 150, 40], dtype=np.int64),
        "excess": np.array([100.0, 30.0, 10.0], dtype=np.float64),
        "excess_err_stat": np.sqrt(np.array([600.0, 150.0, 40.0])),
        "excess_err_conservative": np.sqrt(np.array([1100.0, 270.0, 70.0])),
        "r_opt_deg": np.array([1.0, 0.8, 0.6], dtype=np.float32),
        "sigma_deg": np.array([0.6, 0.5, 0.4], dtype=np.float32),
    }
    exposure_meta = {
        "source_visible_live_seconds": float(np.sum(theta_exposure)),
        "source_visible_live_days": float(np.sum(theta_exposure) / 86400.0),
        "method": "synthetic self-test exposure",
    }
    return response, real_signal, loge_edges, theta_edges, theta_exposure, exposure_meta


def self_test(stage06) -> int:
    print("[self-test] fabricating tiny synthetic response (3 cells x 5 logE x 2 theta) ...", flush=True)
    ns = argparse.Namespace(
        response_npz=None, response_metadata=None, signal_npz=None, output_dir=None,
        npz_name="signal_closure_v6.npz", metadata_name="signal_closure_v6_metadata.json",
        pivot_tev=3.0, model="logpar", phi0=2.114e-12, alpha=2.69, gamma=2.69, beta=0.05,
        true_energy_scale_delta=0.0, source_files_csv=None, stage_c_dir=None,
        source_ra_deg=83.63, source_dec_deg=22.01, lhaaso_lat_deg=29.45, lhaaso_lon_deg=100.14,
        exposure_sample_step_sec=60.0, energy_quadrature_points=32, self_test=True,
    )

    result = run(ns, stage06, self_test=True)
    predicted = np.asarray(result["predicted_excess"], dtype=np.float64)
    mock = result["mock_payload"]

    assert predicted.shape[0] == 3, "expected 3 cells"
    assert np.all(np.isfinite(predicted)), "forward-fold produced non-finite excess"
    assert np.all(predicted > 0.0), "expected positive predicted excess for a physical spectrum"
    print(f"[self-test] forward-fold predicted excess per cell: {predicted.tolist()}", flush=True)

    # NPZ round-trip through a temp file.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "mock.npz"
        np.savez_compressed(npz_path, **mock)
        with np.load(npz_path, allow_pickle=False) as data:
            roundtrip = {name: data[name].copy() for name in data.files}

    # 06_fit.py's validate_inputs requires exactly these signal arrays.
    required_signal = {
        "cell_id", "nhit_bin", "predE_bin", "containment_r_opt",
        "N_on", "B_on", "excess", "excess_err_stat", "excess_err_conservative",
    }
    missing = required_signal - set(roundtrip)
    assert not missing, f"mock NPZ missing Stage F-required arrays: {sorted(missing)}"

    # Counts self-consistency: excess == N_on - B_on.
    n_on = np.asarray(roundtrip["N_on"], dtype=np.float64)
    b_on = np.asarray(roundtrip["B_on"], dtype=np.float64)
    excess = np.asarray(roundtrip["excess"], dtype=np.float64)
    assert np.allclose(excess, n_on - b_on, atol=1e-9), "excess must equal N_on - B_on"
    assert np.allclose(
        roundtrip["excess_err_conservative"], np.sqrt(n_on + b_on), atol=1e-9
    ), "conservative error must be sqrt(N_on + B_on)"

    # Energy-scale delta sanity: a positive delta shifts flux to higher true E,
    # changing predicted counts (should differ from the delta=0 fold).
    ns_shift = argparse.Namespace(**{**vars(ns), "true_energy_scale_delta": 0.15})
    shifted = np.asarray(run(ns_shift, stage06, self_test=True)["predicted_excess"], dtype=np.float64)
    assert not np.allclose(shifted, predicted), "non-zero energy-scale delta should move the fold"
    print(f"[self-test] delta=0.15 predicted excess per cell: {shifted.tolist()}", flush=True)

    print("[self-test] PASS: forward-fold runs, mock NPZ round-trips, schema + counts consistent.", flush=True)
    return 0


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    stage06 = load_stage_06_module()

    if args.self_test:
        raise SystemExit(self_test(stage06))

    if args.pivot_tev <= 0.0:
        raise ValueError("--pivot-tev must be positive")
    if args.phi0 <= 0.0:
        raise ValueError("--phi0 must be positive")

    result = run(args, stage06, self_test=False)
    elapsed = time.perf_counter() - start

    print("Forward-fold closure summary:", flush=True)
    print(f"  injected model     : {result['model_name']}", flush=True)
    print(f"  injected params    : {result['params']}", flush=True)
    print(f"  pivot_tev          : {args.pivot_tev}", flush=True)
    print(f"  energy_scale_delta : {args.true_energy_scale_delta}", flush=True)
    print(f"  n_cells            : {result['n_cells']}", flush=True)
    print(f"  total predicted    : {result['total_predicted_excess']:.6g} excess counts", flush=True)
    print(f"  total mock N_on    : {result['total_mock_N_on']:,}", flush=True)
    print(f"  wrote NPZ          : {result['npz_path']}", flush=True)
    print(f"  wrote metadata     : {result['metadata_path']}", flush=True)
    print(f"  elapsed            : {elapsed:.3f} s", flush=True)


if __name__ == "__main__":
    main()
