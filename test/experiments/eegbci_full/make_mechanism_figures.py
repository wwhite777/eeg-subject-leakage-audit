#!/usr/bin/env python3
"""Figures for the mechanism study, drawn from the exact result CSVs.

Panel 1: reference ablation (inductive vs transductive) on both datasets -> the
         tangent-space reference is NOT the leak channel (bars equal).
Panel 2: classifier regularization sweep -> inflation is controlled by the LR
         penalty (vanishes at strong regularization). The mechanism is classifier
         overfitting to subject-correlated tangent features.
Panel 3: cross-dataset method comparison of pooled-vs-cross leakage inflation.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return list(csv.DictReader(open(path)))


def main():
    eeg_ref = read("results/eegbci_full/BrainFuse_eegbci_reference_leakage_20260709.csv")
    bci_ref = read("results/bci2a/BrainFuse_bci2a_reference_leakage_20260709.csv")
    mech = read("results/eegbci_full/BrainFuse_eegbci_classifier_mechanism_20260709.csv")
    ladder = read("results/eegbci_full/BrainFuse_eegbci_full_capacity_ladder_20260709.csv")
    bci_proto = read("results/bci2a/BrainFuse_bci2a_protocols_20260709.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel 1: reference ablation
    eeg_ind = np.mean([float(r["acc_inductive"]) for r in eeg_ref])
    eeg_tr = np.mean([float(r["acc_transductive"]) for r in eeg_ref])
    bci_ind = float(bci_ref[0]["acc_inductive"])
    bci_tr = float(bci_ref[0]["acc_transductive"])
    x = np.arange(2)
    w = 0.36
    axes[0].bar(x - w / 2, [eeg_ind, bci_ind], w, label="inductive reference (train-only)", color="#238b45", edgecolor="black", linewidth=0.4)
    axes[0].bar(x + w / 2, [eeg_tr, bci_tr], w, label="transductive reference (train+test)", color="#d95f0e", edgecolor="black", linewidth=0.4)
    for xi, (a, b) in zip(x, [(eeg_ind, eeg_tr), (bci_ind, bci_tr)]):
        axes[0].annotate(f"Δ={b-a:+.4f}", (xi, max(a, b)), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["EEGBCI-109", "BCI-IV-2a"], fontsize=9)
    axes[0].set_ylim(0.5, 0.75)
    axes[0].set_ylabel("cross-subject accuracy", fontsize=9)
    axes[0].set_title("(1) Reference is NOT the leak\n(transductive = inductive)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: classifier regularization sweep
    by_c = defaultdict(list)
    for r in mech:
        by_c[float(r["C"])].append(float(r["leakage_inflation"]))
    cs = sorted(by_c)
    means = [np.mean(by_c[c]) for c in cs]
    tasks = sorted({r["task"] for r in mech})
    for t in tasks:
        pts = [(float(r["C"]), float(r["leakage_inflation"])) for r in mech if r["task"] == t]
        pts.sort()
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=0.8, alpha=0.45)
    axes[1].plot(cs, means, marker="s", ms=7, lw=2.4, color="black", label="mean over tasks")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("logistic-regression penalty C  (← stronger reg | weaker reg →)", fontsize=9)
    axes[1].set_ylabel("leakage inflation (pooled − cross)", fontsize=9)
    axes[1].set_title("(2) Inflation is a CLASSIFIER effect\n(vanishes under strong regularization)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel 3: cross-dataset method comparison
    lad = defaultdict(dict)
    for r in ladder:
        lad[(r["task"], r["model"])][r["protocol"]] = float(r["mean_acc"])
    eeg_models = ["csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"]
    eeg_tasks = sorted({t for (t, m) in lad})
    eeg_infl = {m: np.mean([lad[(t, m)]["pooled_random"] - lad[(t, m)]["cross_subject"] for t in eeg_tasks if "pooled_random" in lad[(t, m)]]) for m in eeg_models}
    bp = defaultdict(dict)
    for r in bci_proto:
        bp[r["model"]][r["protocol"]] = float(r["mean_acc"])
    bci_infl = {m: bp[m]["pooled_random"] - bp[m]["cross_subject"] for m in ("csp_lda", "riemann_ts_lr")}
    labels = {"csp_lda": "CSP+LDA", "riemann_ts_lr": "Riemann\nTS+LR", "shallow_fbcsp": "Shallow\nFBCSP", "eegnet_v4": "EEGNet"}
    order = ["csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"]
    xx = np.arange(len(order))
    eeg_vals = [eeg_infl[m] for m in order]
    bci_vals = [bci_infl.get(m, np.nan) for m in order]
    axes[2].bar(xx - 0.2, eeg_vals, 0.38, label="EEGBCI-109 subj", color="#2c7fb8", edgecolor="black", linewidth=0.4)
    axes[2].bar(xx + 0.2, bci_vals, 0.38, label="BCI-IV-2a  9 subj", color="#e7298a", edgecolor="black", linewidth=0.4)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels([labels[m] for m in order], fontsize=8)
    axes[2].set_ylabel("leakage inflation (pooled − cross)", fontsize=9)
    axes[2].set_title("(3) Not universal: severity grows\nwith high-dim features AND few subjects", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("EEG pooled-CV subject leakage: not the tangent-space reference, but classifier overfitting — replicated on two datasets", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path("result/figure/exp3")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "eegbci_mechanism_and_replication.png", dpi=150)
    plt.close(fig)
    print("wrote", out / "eegbci_mechanism_and_replication.png")


if __name__ == "__main__":
    main()
