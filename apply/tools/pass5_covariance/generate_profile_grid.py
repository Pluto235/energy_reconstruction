#!/usr/bin/env python3
"""Generate fixed-LogPar profile-likelihood configurations."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml


PARAMETERS = ("norm", "index1", "index2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--step-scale", type=float, default=0.75)
    return parser.parse_args()


def fixed_parameter(value: float, original: list[float]) -> list[float]:
    result = [float(value), 0.0, 0.0]
    if len(original) == 4:
        result.append(float(original[3]))
    return result


def main() -> None:
    args = parse_args()
    with args.fit_yaml.open(encoding="utf-8") as handle:
        fit = yaml.safe_load(handle)

    source = fit["source_dict"]["J0534+2200"]["sed_model"]
    center = [float(source[name][0]) for name in PARAMETERS]
    fallback = (0.02, 0.004, 0.006)
    errors = []
    for name, default in zip(PARAMETERS, fallback):
        values = source[name]
        candidates = [abs(float(value)) for value in values[1:3] if float(value) != 0.0]
        errors.append(max(candidates) if candidates else default)
    steps = [max(error * args.step_scale, 1e-8) for error in errors]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, object]] = []
    deltas = [("center", [0, 0, 0])]
    for index, name in enumerate(PARAMETERS):
        for sign, suffix in ((-1, "m"), (1, "p")):
            delta = [0, 0, 0]
            delta[index] = sign
            deltas.append((f"{name}_{suffix}", delta))
    for first in range(3):
        for second in range(first + 1, 3):
            for sign_first, suffix_first in ((-1, "m"), (1, "p")):
                for sign_second, suffix_second in ((-1, "m"), (1, "p")):
                    delta = [0, 0, 0]
                    delta[first] = sign_first
                    delta[second] = sign_second
                    label = f"{PARAMETERS[first]}_{suffix_first}__{PARAMETERS[second]}_{suffix_second}"
                    deltas.append((label, delta))

    for label, signs in deltas:
        profile = copy.deepcopy(fit)
        path = args.output_dir / f"input_{label}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(profile, handle, sort_keys=False)
        subprocess.run(
            ["tune_yaml", str(path), "set_source", "free_all_norm"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with path.open(encoding="utf-8") as handle:
            profile = yaml.safe_load(handle)
        target = profile["source_dict"]["J0534+2200"]["sed_model"]
        values = [center[i] + signs[i] * steps[i] for i in range(3)]
        for name, value in zip(PARAMETERS, values):
            target[name] = fixed_parameter(value, list(source[name]))
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(profile, handle, sort_keys=False)
        points.append({"label": label, "signs": signs, "values": values, "input": str(path)})

    manifest = {
        "parameter_names": list(PARAMETERS),
        "center": center,
        "central_hesse_errors": errors,
        "steps": steps,
        "step_scale": args.step_scale,
        "points": points,
    }
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

