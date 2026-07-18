#!/usr/bin/env python3
"""Recover a marginalized covariance from profile-likelihood samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v6-live-days", type=float, default=125.67024571878525)
    parser.add_argument("--pass5-live-days", type=float, default=166.451442205)
    return parser.parse_args()


def label_for(signs: list[int], names: list[str]) -> str:
    nonzero = [index for index, sign in enumerate(signs) if sign]
    if not nonzero:
        return "center"
    parts = [f"{names[index]}_{'p' if signs[index] > 0 else 'm'}" for index in nonzero]
    return "__".join(parts)


def main() -> None:
    args = parse_args()
    with (args.run_dir / "profile_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    names = list(manifest["parameter_names"])
    center = np.asarray(manifest["center"], dtype=float)
    steps = np.asarray(manifest["steps"], dtype=float)

    nll: dict[str, float] = {}
    fit_status: dict[str, int] = {}
    for point in manifest["points"]:
        label = point["label"]
        output = args.run_dir / "profiles" / f"output_{label}.yaml"
        with output.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        options = data["output_option"]["gtlike"]
        nll[label] = float(options["negative_loglike"])
        fit_status[label] = int(options["Error_status"])

    hessian = np.zeros((3, 3), dtype=float)
    f0 = nll["center"]
    for index in range(3):
        minus = [0, 0, 0]
        plus = [0, 0, 0]
        minus[index] = -1
        plus[index] = 1
        hessian[index, index] = (
            nll[label_for(plus, names)] - 2.0 * f0 + nll[label_for(minus, names)]
        ) / steps[index] ** 2
    for first in range(3):
        for second in range(first + 1, 3):
            values = {}
            for sign_first in (-1, 1):
                for sign_second in (-1, 1):
                    signs = [0, 0, 0]
                    signs[first] = sign_first
                    signs[second] = sign_second
                    values[(sign_first, sign_second)] = nll[label_for(signs, names)]
            mixed = (
                values[(1, 1)]
                - values[(1, -1)]
                - values[(-1, 1)]
                + values[(-1, -1)]
            ) / (4.0 * steps[first] * steps[second])
            hessian[first, second] = mixed
            hessian[second, first] = mixed

    covariance = np.linalg.inv(hessian)
    errors = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(errors, errors)
    eigenvalues = np.linalg.eigvalsh(hessian)
    exposure_scale = args.pass5_live_days / args.v6_live_days
    covariance_125d = covariance * exposure_scale

    central_hesse = np.asarray(manifest["central_hesse_errors"], dtype=float)
    relative_diagonal_difference = errors / central_hesse - 1.0
    result = {
        "method": "profile-likelihood finite-difference Hessian",
        "parameter_names": names,
        "parameterization": "Pass5 YAML units: norm at 10 TeV, index1, index2",
        "center": center.tolist(),
        "steps": steps.tolist(),
        "central_hesse_errors_from_yaml": central_hesse.tolist(),
        "profile_hessian_errors": errors.tolist(),
        "relative_diagonal_difference": relative_diagonal_difference.tolist(),
        "hessian": hessian.tolist(),
        "hessian_eigenvalues": eigenvalues.tolist(),
        "covariance_observed_166d": covariance.tolist(),
        "correlation_observed_166d": correlation.tolist(),
        "pass5_live_days": args.pass5_live_days,
        "v6_rough_live_days": args.v6_live_days,
        "statistics_only_exposure_scale": exposure_scale,
        "covariance_expected_125d": covariance_125d.tolist(),
        "profile_fit_status": fit_status,
        "profile_nll": nll,
        "all_status_acceptable": all(status in (2, 3) for status in fit_status.values()),
        "positive_definite_hessian": bool(np.all(eigenvalues > 0.0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

