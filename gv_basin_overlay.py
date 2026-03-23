#!/usr/bin/env python3
"""
gv_basin_overlay.py

Reads the basin detector CSV output and creates an overlay plot showing:
- raw series
- smoothed series
- stability score
- state-colored markers

Expected input:
    results/gv_basin_scan.csv

Usage:
    python gv_basin_overlay.py
    python gv_basin_overlay.py --input results/gv_basin_scan.csv
    python gv_basin_overlay.py --input results/gv_basin_scan.csv --output figures/gv_basin_overlay.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


STATE_STYLE = {
    "stable-basin": {"marker": "o", "label": "Stable basin"},
    "boundary-proximate": {"marker": "s", "label": "Boundary proximate"},
    "transition-zone": {"marker": "^", "label": "Transition zone"},
    "collapse-risk": {"marker": "x", "label": "Collapse risk"},
}


def load_scan(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "t",
            "value",
            "smooth",
            "d1",
            "d2",
            "variance",
            "stability_score",
            "state",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "t": int(float(row["t"])),
                    "value": float(row["value"]),
                    "smooth": float(row["smooth"]),
                    "d1": float(row["d1"]),
                    "d2": float(row["d2"]),
                    "variance": float(row["variance"]),
                    "stability_score": float(row["stability_score"]),
                    "state": row["state"].strip(),
                }
            )
    return rows


def make_overlay(rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t = [r["t"] for r in rows]
    raw = [r["value"] for r in rows]
    smooth = [r["smooth"] for r in rows]
    stability = [r["stability_score"] for r in rows]

    plt.figure(figsize=(12, 7))
    plt.plot(t, raw, linewidth=1.5, label="Raw series")
    plt.plot(t, smooth, linewidth=2.0, label="Smoothed series")
    plt.plot(t, stability, linewidth=1.5, linestyle="--", label="Stability score")

    seen_labels = set()
    for state, style in STATE_STYLE.items():
        xs = [r["t"] for r in rows if r["state"] == state]
        ys = [r["smooth"] for r in rows if r["state"] == state]
        if not xs:
            continue

        label = style["label"] if style["label"] not in seen_labels else None
        seen_labels.add(style["label"])

        plt.scatter(
            xs,
            ys,
            marker=style["marker"],
            s=70,
            label=label,
        )

    plt.xlabel("Time index")
    plt.ylabel("Value / score")
    plt.title("GV Basin Overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def summarize(rows: List[Dict[str, object]]) -> str:
    counts: Dict[str, int] = {}
    for r in rows:
        state = str(r["state"])
        counts[state] = counts.get(state, 0) + 1

    ordered = ["stable-basin", "boundary-proximate", "transition-zone", "collapse-risk"]
    lines = ["GV Basin Overlay Summary", "------------------------"]
    for state in ordered:
        lines.append(f"{state:20s} {counts.get(state, 0)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create overlay plot from basin detector CSV.")
    parser.add_argument(
        "--input",
        default="results/gv_basin_scan.csv",
        help="Input basin scan CSV",
    )
    parser.add_argument(
        "--output",
        default="figures/gv_basin_overlay.png",
        help="Output overlay PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_scan(args.input)
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows in basin scan CSV to make overlay plot.")

    print(summarize(rows))
    make_overlay(rows, args.output)
    print(f"\nSaved overlay: {args.output}")


if __name__ == "__main__":
    main()
