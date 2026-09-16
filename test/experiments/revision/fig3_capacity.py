#!/usr/bin/env python3
"""Figure 3 — width sweeps (E7): Δ(P0 − P2) and accuracies versus trainable-parameter count for EEGNet
and ShallowFBCSPNet; per-subject points of the difference; train–test gaps from the per-fold table.
Usage: fig3_capacity.py --e7 <eegnet_per_subject.csv,shallow_per_subject.csv> --tag rev1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import fig_common as F  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e7", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    files = [Path(p) for p in args.e7.split(",")]
    ps, pf = [], []
    for p in files:
        ps += C.read_csv(p)
        pf += C.read_csv(Path(str(p).replace("_per_subject.csv", "_per_fold.csv")))
    subj = F.seed_avg_subject(ps, ("task", "model", "protocol", "width"))
    pm = F.paradigm_mean(subj, 0)
    params, gap = {}, defaultdict(list)
    for r in pf:
        params[(r["model"], r["width"])] = int(r["n_params"])
        gap[(r["model"], r["width"], r["protocol"])].append(float(r["train_acc"]) - float(r["test_acc"]))
    models = [m for m in ("eegnet_v4", "shallow_fbcsp") if any(k[1] == m for k in pm)]
    fig, axes = plt.subplots(2, len(models), figsize=(7.4, 4.6), squeeze=False)
    rng = np.random.default_rng(C.SEED)
    cap = []
    for j, m in enumerate(models):
        widths = sorted({int(k[3]) for k in pm if k[1] == m})
        xs = [params[(m, str(w))] for w in widths]
        ax = axes[0, j]
        ds, los, his = [], [], []
        for w, x in zip(widths, xs):
            A = pm[("paradigm_mean", m, "trial_random", str(w))]
            B = pm[("paradigm_mean", m, "subject_disjoint", str(w))]
            subs = sorted(set(A) & set(B))
            d = np.array([A[s] - B[s] for s in subs])
            lo, hi = C.bootstrap_ci(d)
            ds.append(d.mean()); los.append(lo); his.append(hi)
            ax.scatter(x * np.exp(rng.uniform(-0.08, 0.08, len(d))), d, s=4, color=F.MODEL_COLOR[m], alpha=0.2, linewidths=0, rasterized=True)
            cap.append(f"{F.MODEL_LABEL[m]} width {w} ({x} params): Δ(P0−P2) {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
        ax.plot(xs, ds, marker=F.MODEL_MARK[m], ms=4, color="black", zorder=5, label="mean ± 95 % CI")
        ax.fill_between(xs, los, his, color="black", alpha=0.15, linewidth=0, zorder=4)
        ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_ylabel("Δ(P0 − P2) (fraction correct)"); ax.set_title(f"({'ab'[j]}) {F.MODEL_LABEL[m]}: trial-random minus subject-disjoint")
        ax.set_xticks(xs); ax.set_xticklabels([f"{w}\n{x:,}" for w, x in zip(widths, xs)], fontsize=6.5)
        ax.set_xlabel("width parameter (top) and trainable parameters (bottom)")
        ax.legend(loc="upper left", frameon=False)
        ax = axes[1, j]
        for p in ("trial_random", "subject_disjoint"):
            means, lo_, hi_ = [], [], []
            for w in widths:
                v = np.array(list(pm[("paradigm_mean", m, p, str(w))].values()))
                lo, hi = C.bootstrap_ci(v)
                means.append(v.mean()); lo_.append(lo); hi_.append(hi)
            ax.plot(xs, means, marker="o", ms=3.5, color=F.PROTO_COLOR[p], label=F.PROTO_LABEL[p] + " test")
            ax.fill_between(xs, lo_, hi_, color=F.PROTO_COLOR[p], alpha=0.18, linewidth=0)
            g = [np.mean(gap[(m, str(w), p)]) for w in widths]
            ax.plot(xs, np.array(means) + np.array(g), ls=":", lw=1.1, color=F.PROTO_COLOR[p], label=F.PROTO_LABEL[p] + " training (dotted)")
            cap.append(f"{F.MODEL_LABEL[m]} {p}: " + ", ".join(f"w{w}: test {mm:.3f}, train–test gap {gg:+.3f}" for w, mm, gg in zip(widths, means, g)))
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--"); ax.set_xscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs], fontsize=6.5)
        ax.set_xlabel("trainable parameters (count)"); ax.set_ylabel("Accuracy (fraction correct)"); ax.set_title(f"({'cd'[j]}) {F.MODEL_LABEL[m]}: test and training accuracy")
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    caption = ("Figure 3. Network-width sweeps on EEGBCI (paradigm mean, 109 subjects, one seed). Top: per-subject difference between the trial-random (P0) and "
               "subject-disjoint (P2) protocols versus the trainable-parameter count (points = subjects, line = mean, band = subject-bootstrap 95 % CI; dashed = 0). "
               "Bottom: test accuracy (solid, mean with 95 % CI band) and training accuracy (dotted; test mean plus the mean train–test gap over folds) under both protocols. "
               "EEGNet width = F1 (D = 2, F2 = 2 F1); ShallowFBCSPNet width = number of temporal = spatial filters. Values: " + " | ".join(cap))
    F.save(fig, f"fig3_width_{args.tag}", caption, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
