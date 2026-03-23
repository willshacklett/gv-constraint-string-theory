#!/usr/bin/env python3

import argparse
import csv
import os
import matplotlib.pyplot as plt

STATE_COLORS = {
    "stable-basin": "green",
    "boundary-proximate": "gold",
    "transition-zone": "orange",
    "collapse-risk": "red",
}

def load(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "t": int(float(r["t"])),
                "value": float(r["value"]),
                "smooth": float(r["smooth"]),
                "stability": float(r["stability_score"]),
                "state": r["state"],
            })
    return rows

def plot(rows, out):
    t = [r["t"] for r in rows]
    smooth = [r["smooth"] for r in rows]
    stability = [r["stability"] for r in rows]

    plt.figure(figsize=(12,7))
    plt.plot(t, smooth, linewidth=2, label="Smooth")
    plt.plot(t, stability, linestyle="--", label="Stability")

    for r in rows:
        plt.scatter(
            r["t"],
            r["smooth"],
            color=STATE_COLORS.get(r["state"], "black"),
            s=80
        )

    plt.title("GV Basin Overlay (State Colored)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/gv_basin_scan.csv")
    parser.add_argument("--output", default="figures/gv_basin_overlay.png")
    args = parser.parse_args()

    rows = load(args.input)
    plot(rows, args.output)

    print("Saved:", args.output)

if __name__ == "__main__":
    main()
