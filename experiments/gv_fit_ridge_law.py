# experiments/gv_fit_ridge_law.py

import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

print("\n🚀 RUNNING GV RIDGE LAW FIT\n")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RIDGE_CSV = os.path.join(DATA_DIR, "gv_ridge_points.csv")
OUT_CSV = os.path.join(DATA_DIR, "gv_ridge_law_fits.csv")

RATIO_FIT_PNG = os.path.join(FIG_DIR, "gv_ridge_ratio_fit.png")
TAILMEAN_FIT_PNG = os.path.join(FIG_DIR, "gv_ridge_tailmean_fit.png")


def load_ridge_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok":
                continue
            rows.append(
                {
                    "C": float(row["C"]),
                    "ratio": float(row["ratio"]),
                    "injection": float(row["injection"]),
                    "final_gv": float(row["final_gv"]),
                    "tail_mean_gv": float(row["tail_mean_gv"]),
                    "tail_span": float(row["tail_span"]),
                    "tail_stable": str(row["tail_stable"]).lower() == "true",
                }
            )
    return rows


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def fit_constant(x, y):
    c = float(np.mean(y))
    yhat = np.full_like(np.asarray(y, dtype=float), c, dtype=float)
    return {
        "name": "constant",
        "params": {"c": c},
        "yhat": yhat,
        "mse": mse(y, yhat),
        "r2": r2_score(y, yhat),
        "equation": f"y = {c:.6f}",
    }


def fit_linear(x, y):
    coeffs = np.polyfit(x, y, 1)
    a, b = [float(v) for v in coeffs]
    yhat = a * np.asarray(x) + b
    return {
        "name": "linear",
        "params": {"a": a, "b": b},
        "yhat": yhat,
        "mse": mse(y, yhat),
        "r2": r2_score(y, yhat),
        "equation": f"y = {a:.6f} * C + {b:.6f}",
    }


def fit_quadratic(x, y):
    coeffs = np.polyfit(x, y, 2)
    a, b, c = [float(v) for v in coeffs]
    x_arr = np.asarray(x)
    yhat = a * x_arr**2 + b * x_arr + c
    return {
        "name": "quadratic",
        "params": {"a": a, "b": b, "c": c},
        "yhat": yhat,
        "mse": mse(y, yhat),
        "r2": r2_score(y, yhat),
        "equation": f"y = {a:.6f} * C^2 + {b:.6f} * C + {c:.6f}",
    }


def fit_inverse(x, y):
    x_arr = np.asarray(x, dtype=float)
    z = 1.0 / x_arr
    coeffs = np.polyfit(z, y, 1)
    a, b = [float(v) for v in coeffs]
    yhat = a * z + b
    return {
        "name": "inverse",
        "params": {"a": a, "b": b},
        "yhat": yhat,
        "mse": mse(y, yhat),
        "r2": r2_score(y, yhat),
        "equation": f"y = {a:.6f} / C + {b:.6f}",
    }


def choose_best_fit(fits):
    return min(fits, key=lambda d: d["mse"])


def save_fit_summary(path, sections):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "model", "equation", "mse", "r2"])

        for target_name, fits in sections.items():
            for fit in fits:
                writer.writerow(
                    [
                        target_name,
                        fit["name"],
                        fit["equation"],
                        f"{fit['mse']:.10f}",
                        f"{fit['r2']:.10f}",
                    ]
                )


def plot_ratio_fit(C, ratio, best_fit, outpath):
    plt.figure(figsize=(10, 6))
    plt.scatter(C, ratio, label="ridge data")

    xline = np.linspace(min(C), max(C), 300)
    if best_fit["name"] == "constant":
        yline = np.full_like(xline, best_fit["params"]["c"])
    elif best_fit["name"] == "linear":
        yline = best_fit["params"]["a"] * xline + best_fit["params"]["b"]
    elif best_fit["name"] == "quadratic":
        yline = (
            best_fit["params"]["a"] * xline**2
            + best_fit["params"]["b"] * xline
            + best_fit["params"]["c"]
        )
    elif best_fit["name"] == "inverse":
        yline = best_fit["params"]["a"] / xline + best_fit["params"]["b"]
    else:
        yline = np.full_like(xline, np.nan)

    plt.plot(xline, yline, label=f"best fit: {best_fit['name']}")
    plt.xlabel("Constraint C")
    plt.ylabel("Ridge ratio* (Injection / Entropy)")
    plt.title("GV Ridge Ratio Law Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_tailmean_fit(C, tail_mean, best_fit, outpath):
    plt.figure(figsize=(10, 6))
    plt.scatter(C, tail_mean, label="ridge data")

    xline = np.linspace(min(C), max(C), 300)
    if best_fit["name"] == "constant":
        yline = np.full_like(xline, best_fit["params"]["c"])
    elif best_fit["name"] == "linear":
        yline = best_fit["params"]["a"] * xline + best_fit["params"]["b"]
    elif best_fit["name"] == "quadratic":
        yline = (
            best_fit["params"]["a"] * xline**2
            + best_fit["params"]["b"] * xline
            + best_fit["params"]["c"]
        )
    elif best_fit["name"] == "inverse":
        yline = best_fit["params"]["a"] / xline + best_fit["params"]["b"]
    else:
        yline = np.full_like(xline, np.nan)

    plt.plot(xline, yline, label=f"best fit: {best_fit['name']}")
    plt.xlabel("Constraint C")
    plt.ylabel("Ridge tail_mean_gv")
    plt.title("GV Ridge Band-Center Law Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def print_fit_block(title, fits):
    print(title)
    print("-" * len(title))
    for fit in sorted(fits, key=lambda d: d["mse"]):
        print(
            f"{fit['name']:>10} | mse={fit['mse']:.8f} | r2={fit['r2']:.6f} | {fit['equation']}"
        )
    print()


def main():
    if not os.path.exists(RIDGE_CSV):
        raise FileNotFoundError(
            f"Could not find ridge CSV at: {RIDGE_CSV}\n"
            "Run python experiments/gv_ridge_extractor.py first."
        )

    rows = load_ridge_rows(RIDGE_CSV)
    if not rows:
        raise RuntimeError("No valid ridge rows found in gv_ridge_points.csv")

    C = np.array([row["C"] for row in rows], dtype=float)
    ratio = np.array([row["ratio"] for row in rows], dtype=float)
    tail_mean = np.array([row["tail_mean_gv"] for row in rows], dtype=float)

    ratio_fits = [
        fit_constant(C, ratio),
        fit_linear(C, ratio),
        fit_quadratic(C, ratio),
        fit_inverse(C, ratio),
    ]

    tailmean_fits = [
        fit_constant(C, tail_mean),
        fit_linear(C, tail_mean),
        fit_quadratic(C, tail_mean),
        fit_inverse(C, tail_mean),
    ]

    best_ratio = choose_best_fit(ratio_fits)
    best_tailmean = choose_best_fit(tailmean_fits)

    save_fit_summary(
        OUT_CSV,
        {
            "ridge_ratio": ratio_fits,
            "ridge_tail_mean_gv": tailmean_fits,
        },
    )

    plot_ratio_fit(C, ratio, best_ratio, RATIO_FIT_PNG)
    plot_tailmean_fit(C, tail_mean, best_tailmean, TAILMEAN_FIT_PNG)

    print("✅ DONE\n")
    print(f"Ridge points loaded: {len(rows)}\n")

    print_fit_block("RIDGE RATIO LAW CANDIDATES", ratio_fits)
    print_fit_block("RIDGE BAND-CENTER LAW CANDIDATES", tailmean_fits)

    print("BEST LAW CANDIDATES")
    print("-------------------")
    print(f"ratio*(C):     {best_ratio['equation']}")
    print(f"band_center(C): {best_tailmean['equation']}\n")

    print(f"Saved fit summary CSV: {OUT_CSV}")
    print(f"Saved ridge ratio fit plot: {RATIO_FIT_PNG}")
    print(f"Saved band-center fit plot: {TAILMEAN_FIT_PNG}\n")


if __name__ == "__main__":
    main()
