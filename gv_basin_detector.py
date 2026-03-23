#!/usr/bin/env python3
"""
gv_basin_detector.py

Detects basin membership, boundary proximity, and collapse-vs-recovery behavior
from a 1D time series (for example, GV over time).

What it does
------------
- Loads a numeric series from CSV or plain text
- Smooths the signal
- Computes:
    * first derivative
    * second derivative
    * rolling variance
    * local stability score
- Detects:
    * stable basin residence
    * boundary proximity
    * transition zones
    * collapse risk
- Produces:
    * console summary
    * optional CSV output
    * optional PNG plot

Usage examples
--------------
python gv_basin_detector.py --input data/gv_series.csv --column gv
python gv_basin_detector.py --input data/gv_series.csv --column gv --plot
python gv_basin_detector.py --input data/gv_series.csv --column gv --out results/gv_basin_scan.csv --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class BasinPoint:
    t: int
    value: float
    smooth: float
    d1: float
    d2: float
    variance: float
    stability_score: float
    state: str


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < 2:
        return values[:]

    out: List[float] = []
    half = window // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        chunk = values[start:end]
        out.append(sum(chunk) / len(chunk))

    return out


def derivative(values: List[float]) -> List[float]:
    if len(values) < 2:
        return [0.0 for _ in values]

    out = [0.0]
    for i in range(1, len(values)):
        out.append(values[i] - values[i - 1])
    return out


def rolling_variance(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return [0.0 for _ in values]

    out: List[float] = []
    half = window // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        chunk = values[start:end]
        mean = sum(chunk) / len(chunk)
        var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
        out.append(var)

    return out


def normalize(values: List[float], eps: float = 1e-9) -> List[float]:
    if not values:
        return []

    vmax = max(abs(v) for v in values)
    if vmax < eps:
        return [0.0 for _ in values]
    return [v / vmax for v in values]


def load_series(path: str, column: Optional[str]) -> Tuple[List[int], List[float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    _, ext = os.path.splitext(path.lower())

    if ext == ".csv":
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            if not fieldnames:
                raise ValueError("CSV appears to have no header row.")

            target = column if column else fieldnames[0]
            if target not in fieldnames:
                raise ValueError(
                    f"Column '{target}' not found. Available columns: {fieldnames}"
                )

            values: List[float] = []
            for row in reader:
                raw = row.get(target, "")
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue

            return list(range(len(values))), values

    # Plain text / one-number-per-line fallback
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                values.append(float(s))
            except ValueError:
                continue

    return list(range(len(values))), values


def classify_point(
    value: float,
    stability_score: float,
    d1_n: float,
    d2_n: float,
    var_n: float,
    low_value_threshold: float,
    boundary_threshold: float,
    collapse_threshold: float,
) -> str:
    abs_d1 = abs(d1_n)
    abs_d2 = abs(d2_n)

    if value <= low_value_threshold and stability_score <= collapse_threshold:
        return "collapse-risk"

    if stability_score <= collapse_threshold:
        return "transition-zone"

    if stability_score <= boundary_threshold or abs_d1 > 0.35 or abs_d2 > 0.35 or var_n > 0.35:
        return "boundary-proximate"

    return "stable-basin"


def analyze_series(
    x: List[int],
    y: List[float],
    smooth_window: int = 7,
    variance_window: int = 9,
    low_value_threshold: float = 0.35,
    boundary_threshold: float = 0.45,
    collapse_threshold: float = 0.20,
) -> List[BasinPoint]:
    smooth = moving_average(y, smooth_window)
    d1 = derivative(smooth)
    d2 = derivative(d1)
    var = rolling_variance(smooth, variance_window)

    d1_n = normalize(d1)
    d2_n = normalize(d2)
    var_n = normalize(var)

    points: List[BasinPoint] = []

    for i in range(len(y)):
        # Higher slope/curvature/variance => lower stability
        instability = (
            0.45 * abs(d1_n[i]) +
            0.30 * abs(d2_n[i]) +
            0.25 * abs(var_n[i])
        )
        stability_score = max(0.0, 1.0 - instability)

        state = classify_point(
            value=smooth[i],
            stability_score=stability_score,
            d1_n=d1_n[i],
            d2_n=d2_n[i],
            var_n=var_n[i],
            low_value_threshold=low_value_threshold,
            boundary_threshold=boundary_threshold,
            collapse_threshold=collapse_threshold,
        )

        points.append(
            BasinPoint(
                t=x[i],
                value=y[i],
                smooth=smooth[i],
                d1=d1[i],
                d2=d2[i],
                variance=var[i],
                stability_score=stability_score,
                state=state,
            )
        )

    return points


def summarize(points: List[BasinPoint]) -> str:
    if not points:
        return "No points analyzed."

    counts = {
        "stable-basin": 0,
        "boundary-proximate": 0,
        "transition-zone": 0,
        "collapse-risk": 0,
    }

    for p in points:
        counts[p.state] = counts.get(p.state, 0) + 1

    latest = points[-1]

    transitions = 0
    for i in range(1, len(points)):
        if points[i].state != points[i - 1].state:
            transitions += 1

    return "\n".join([
        "GV Basin Detector Summary",
        "-------------------------",
        f"Total points:           {len(points)}",
        f"Stable basin:           {counts['stable-basin']}",
        f"Boundary proximate:     {counts['boundary-proximate']}",
        f"Transition zone:        {counts['transition-zone']}",
        f"Collapse risk:          {counts['collapse-risk']}",
        f"State transitions:      {transitions}",
        "",
        "Latest point",
        "------------",
        f"t:                     {latest.t}",
        f"raw value:             {latest.value:.6f}",
        f"smooth value:          {latest.smooth:.6f}",
        f"stability score:       {latest.stability_score:.6f}",
        f"classification:        {latest.state}",
    ])


def save_csv(points: List[BasinPoint], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "value",
            "smooth",
            "d1",
            "d2",
            "variance",
            "stability_score",
            "state",
        ])
        for p in points:
            writer.writerow([
                p.t,
                p.value,
                p.smooth,
                p.d1,
                p.d2,
                p.variance,
                p.stability_score,
                p.state,
            ])


def make_plot(points: List[BasinPoint], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    t = [p.t for p in points]
    raw = [p.value for p in points]
    smooth = [p.smooth for p in points]
    stability = [p.stability_score for p in points]

    plt.figure(figsize=(12, 7))
    plt.plot(t, raw, label="Raw series")
    plt.plot(t, smooth, label="Smoothed series")
    plt.plot(t, stability, label="Stability score")

    for p in points:
        if p.state == "collapse-risk":
            plt.axvline(p.t, alpha=0.10)
        elif p.state == "transition-zone":
            plt.axvline(p.t, alpha=0.06)

    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.title("GV Basin Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect GV basin behavior from time series.")
    parser.add_argument("--input", required=True, help="Path to CSV or plain text series file.")
    parser.add_argument("--column", default=None, help="CSV column name to use.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Moving average window.")
    parser.add_argument("--variance-window", type=int, default=9, help="Rolling variance window.")
    parser.add_argument("--low-value-threshold", type=float, default=0.35, help="Low-value collapse threshold.")
    parser.add_argument("--boundary-threshold", type=float, default=0.45, help="Boundary threshold.")
    parser.add_argument("--collapse-threshold", type=float, default=0.20, help="Collapse threshold.")
    parser.add_argument("--out", default="results/gv_basin_scan.csv", help="Output CSV path.")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plot.")
    parser.add_argument("--plot-path", default="figures/gv_basin_detector.png", help="Output plot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x, y = load_series(args.input, args.column)

    if len(y) < 5:
        raise ValueError("Need at least 5 numeric points to analyze basin behavior.")

    points = analyze_series(
        x=x,
        y=y,
        smooth_window=args.smooth_window,
        variance_window=args.variance_window,
        low_value_threshold=args.low_value_threshold,
        boundary_threshold=args.boundary_threshold,
        collapse_threshold=args.collapse_threshold,
    )

    print(summarize(points))
    save_csv(points, args.out)

    if args.plot:
        make_plot(points, args.plot_path)
        print(f"\nSaved plot: {args.plot_path}")

    print(f"Saved CSV: {args.out}")


if __name__ == "__main__":
    main()
