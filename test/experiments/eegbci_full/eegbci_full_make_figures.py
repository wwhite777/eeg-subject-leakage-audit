#!/usr/bin/env python3
"""Figures for the EEGBCI full-corpus leakage audit, drawn from the exact CSV.

Fig 1: per-task accuracy under the three evaluation protocols (one panel per
        model), with the chance line.
Fig 2: the two derived quantities per task/model - calibration gap
        (within - cross) and leakage inflation (pooled_random - cross).
No modelling; reads results/eegbci_full/BrainFuse_eegbci_full_leakage_summary_*.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_ORDER = ["imagery_left_right_fist", "real_left_right_fist", "imagery_fists_feet", "real_fists_feet"]
TASK_SHORT = {
    "imagery_left_right_fist": "MI L/R fist",
    "real_left_right_fist": "Real L/R fist",
    "imagery_fists_feet": "MI fists/feet",
    "real_fists_feet": "Real fists/feet",
}
MODELS = ["csp_lda", "riemann_ts_lr"]
MODEL_LABEL = {"csp_lda": "CSP + LDA (low capacity)", "riemann_ts_lr": "Riemannian TS + LR (higher capacity)"}
PROTO_ORDER = ["within_subject", "cross_subject", "pooled_random"]
PROTO_LABEL = {
    "within_subject": "within-subject (legit)",
    "cross_subject": "cross-subject (honest)",
    "pooled_random": "pooled-random (leaky)",
}
PROTO_COLOR = {"within_subject": "#2c7fb8", "cross_subject": "#238b45", "pooled_random": "#d95f0e"}


def load(summary_path: Path):
    data = defaultdict(dict)
    ci = defaultdict(dict)
    for r in csv.DictReader(summary_path.open()):
        data[(r["task"], r["model"])][r["protocol"]] = float(r["mean_acc"])
        ci[(r["task"], r["model"])][r["protocol"]] = (float(r["ci_low"]), float(r["ci_high"]))
    return data, ci


def fig_protocols(data, ci, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.26
    for ax, model in zip(axes, MODELS):
        for i, proto in enumerate(PROTO_ORDER):
            means = [data[(t, model)][proto] for t in TASK_ORDER]
            los = [data[(t, model)][proto] - ci[(t, model)][proto][0] for t in TASK_ORDER]
            his = [ci[(t, model)][proto][1] - data[(t, model)][proto] for t in TASK_ORDER]
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                yerr=[los, his],
                capsize=3,
                color=PROTO_COLOR[proto],
                label=PROTO_LABEL[proto],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.text(len(TASK_ORDER) - 0.5, 0.505, "chance", color="gray", fontsize=8, ha="right")
        ax.set_title(MODEL_LABEL[model], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT[t] for t in TASK_ORDER], rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.45, 0.85)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mean per-subject accuracy", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "PhysioNet MMI (109 subjects): accuracy by evaluation protocol\n"
        "within-subject > cross-subject; pooled-random inflates only the higher-capacity model",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_derived(data, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.36
    metrics = [
        ("calibration gap  (within - cross)", lambda t, m: data[(t, m)]["within_subject"] - data[(t, m)]["cross_subject"]),
        ("leakage inflation  (pooled - cross)", lambda t, m: data[(t, m)]["pooled_random"] - data[(t, m)]["cross_subject"]),
    ]
    colors = {"csp_lda": "#7570b3", "riemann_ts_lr": "#e7298a"}
    for ax, (title, fn) in zip(axes, metrics):
        for j, model in enumerate(MODELS):
            vals = [fn(t, model) for t in TASK_ORDER]
            ax.bar(x + (j - 0.5) * width, vals, width, color=colors[model], label=MODEL_LABEL[model], edgecolor="black", linewidth=0.4)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT[t] for t in TASK_ORDER], rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy difference", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Derived measures across 109 subjects: the calibration gap is large for all methods;\n"
        "leakage inflation appears only for the higher-capacity model (CSP+LDA is near zero)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/eegbci_full/BrainFuse_eegbci_full_leakage_summary_20260708.csv")
    parser.add_argument("--out-dir", default="result/figure/exp1")
    args = parser.parse_args()
    data, ci = load(Path(args.summary))
    out_dir = Path(args.out_dir)
    fig_protocols(data, ci, out_dir / "eegbci_full_protocol_accuracy.png")
    fig_derived(data, out_dir / "eegbci_full_calibration_and_leakage.png")

    # derived CSV for the paper table
    derived_path = Path("results/eegbci_full/BrainFuse_eegbci_full_derived_20260708.csv")
    with derived_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "within", "cross", "pooled", "calibration_gap", "leakage_inflation"])
        for t in TASK_ORDER:
            for m in MODELS:
                p = data[(t, m)]
                w.writerow([
                    t, m,
                    f"{p['within_subject']:.4f}", f"{p['cross_subject']:.4f}", f"{p['pooled_random']:.4f}",
                    f"{p['within_subject']-p['cross_subject']:.4f}", f"{p['pooled_random']-p['cross_subject']:.4f}",
                ])
    print(f"wrote figures to {out_dir} and {derived_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
