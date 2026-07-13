#!/usr/bin/env python3
"""Render the controlled v6 1D-versus-2D LogPar uncertainty table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d-json", type=Path, required=True)
    parser.add_argument("--two-d-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def logpar_errors(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = (payload.get("fits") or {}).get("logpar_conservative") or {}
    errors = fit.get("errors") or {}
    required = ("phi0", "alpha", "beta")
    missing = [name for name in required if name not in errors]
    if missing:
        raise ValueError(f"{path} is missing LogPar errors: {missing}")
    return {name: float(errors[name]) for name in required}


def draw_cell(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    textcolor: str,
    fontsize: float,
    weight: str = "normal",
    align: str = "center",
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor="#CBD5E1",
            linewidth=0.9,
        )
    )
    padding = 0.018
    text_x = x + width / 2.0 if align == "center" else x + padding
    ax.text(
        text_x,
        y + height / 2.0,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=weight,
    )


def main() -> None:
    args = parse_args()
    one_d = logpar_errors(args.one_d_json)
    two_d = logpar_errors(args.two_d_json)

    rows = [
        (
            r"Normalization  $\sigma_{\phi_0}$",
            rf"${one_d['phi0'] / 1.0e-14:.3f}\times10^{{-14}}$",
            rf"${two_d['phi0'] / 1.0e-14:.3f}\times10^{{-14}}$",
            "phi0",
        ),
        (r"Spectral index  $\sigma_{\alpha}$", f"{one_d['alpha']:.5f}", f"{two_d['alpha']:.5f}", "alpha"),
        (r"Curvature  $\sigma_{\beta}$", f"{one_d['beta']:.5f}", f"{two_d['beta']:.5f}", "beta"),
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 4.15))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.04,
        0.925,
        "Crab LogPar parameter uncertainties",
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.04,
        0.858,
        r"Controlled comparison: same 44 selected cells; 1D collapses $E_{\mathrm{pred}}$ within each $N_{\mathrm{hit}}$ bin",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#4B5563",
    )

    left = 0.04
    bottom = 0.19
    table_width = 0.92
    row_height = 0.145
    column_fractions = (0.32, 0.22, 0.26, 0.20)
    column_widths = [table_width * fraction for fraction in column_fractions]
    headers = (
        "Parameter uncertainty",
        r"1D  $N_{\mathrm{hit}}$",
        r"2D  $N_{\mathrm{hit}}\times E_{\mathrm{pred}}$",
        "Reduction",
    )

    x_positions = [left]
    for width in column_widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    header_y = bottom + len(rows) * row_height
    for x, width, header in zip(x_positions, column_widths, headers):
        draw_cell(
            ax,
            x=x,
            y=header_y,
            width=width,
            height=row_height,
            text=header,
            facecolor="#1F2937",
            textcolor="white",
            fontsize=10.5,
            weight="bold",
        )

    for display_index, (label, one_text, two_text, key) in enumerate(rows):
        y = bottom + (len(rows) - 1 - display_index) * row_height
        base = "#FFFFFF" if display_index % 2 == 0 else "#F8FAFC"
        reduction = 100.0 * (1.0 - two_d[key] / one_d[key])
        values = (label, one_text, two_text, f"{reduction:.1f}%")
        fills = (base, base, "#EFF6FF", "#ECFDF5")
        colors = ("#111827", "#374151", "#1D4ED8", "#047857")
        weights = ("normal", "normal", "bold", "bold")
        aligns = ("left", "center", "center", "center")
        for x, width, value, fill, color, weight, align in zip(
            x_positions,
            column_widths,
            values,
            fills,
            colors,
            weights,
            aligns,
        ):
            draw_cell(
                ax,
                x=x,
                y=y,
                width=width,
                height=row_height,
                text=value,
                facecolor=fill,
                textcolor=color,
                fontsize=11.5,
                weight=weight,
                align=align,
            )

    ax.text(
        left,
        0.105,
        r"Formal HESSE $1\sigma$ uncertainties at pivot energy $E_0=3$ TeV.  $\phi_0$ uncertainty unit: TeV$^{-1}$ cm$^{-2}$ s$^{-1}$.",
        ha="left",
        va="center",
        fontsize=9.4,
        color="#4B5563",
    )
    ax.text(
        left,
        0.052,
        "Lower uncertainty is better; this table quantifies formal statistical precision, not bias or systematic uncertainty.",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#6B7280",
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
