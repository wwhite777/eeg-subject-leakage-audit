#!/usr/bin/env python3
"""Figure 4 — regularization sweep v2 (E2): both accuracy curves, training accuracy, ||w||, and the
protocol differences with per-subject points. Paradigm mean over the four EEGBCI paradigms.
Usage: fig4_sweep.py --e2 <per_subject.csv> --tag rev1  (the per-fold CSV is found by name)
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

PROTOS = ["trial_random", "run_disjoint", "subject_disjoint"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    ps = C.read_csv(Path(args.e2))
    pf = C.read_csv(Path(args.e2.replace("_per_subject.csv", "_per_fold.csv")))
    for r in ps:
        r["model"] = "riemann_ts_lr"
        r["C"] = f"{float(r['C']):g}"
    subj = F.seed_avg_subject(ps, ("task", "model", "protocol", "C"))
    pm = F.paradigm_mean(subj, 0)
    Cs = sorted({float(k[3]) for k in pm})

    def vec(p, c):
        return np.array(list(pm[("paradigm_mean", "riemann_ts_lr", p, f"{c:g}")].values()))
    # training accuracy and weight norm per (protocol, C): mean over tasks, seeds, folds
    tr_acc, wn = defaultdict(list), defaultdict(list)
    for r in pf:
        tr_acc[(r["protocol"], float(r["C"]))].append(float(r["train_acc"]))
        wn[(r["protocol"], float(r["C"]))].append(float(r["w_norm"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    rng = np.random.default_rng(C.SEED)
    cap = []
    ax = axes[0]
    for p in PROTOS:
        means, los, his = [], [], []
        for c in Cs:
            v = vec(p, c)
            m = v.mean()
            lo, hi = C.bootstrap_ci(v)
            means.append(m); los.append(lo); his.append(hi)
        ax.plot(Cs, means, marker="o", ms=3.5, color=F.PROTO_COLOR[p], label=F.PROTO_LABEL[p])
        ax.fill_between(Cs, los, his, color=F.PROTO_COLOR[p], alpha=0.18, linewidth=0)
        ax.plot(Cs, [np.mean(tr_acc[(p, c)]) for c in Cs], ls=":", lw=1.1, color=F.PROTO_COLOR[p])
        cap.append(f"{p}: " + ", ".join(f"C={c:g}: {m:.3f}" for c, m in zip(Cs, means)))
    ax.plot([], [], ls=":", color="black", label="training accuracy (dotted)")
    ax.set_xscale("log"); ax.set_xlabel("Inverse penalty C (dimensionless)"); ax.set_ylabel("Accuracy (fraction correct)")
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--"); ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=True, fontsize=6); ax.set_title("(a) test (solid) and training (dotted) accuracy")
    ax = axes[1]
    for pa, pb, col, lab in (("run_disjoint", "subject_disjoint", "#E69F00", "Δ(P1 − P2)"), ("trial_random", "subject_disjoint", "#D55E00", "Δ(P0 − P2)")):
        means, los, his = [], [], []
        for i, c in enumerate(Cs):
            A = pm[("paradigm_mean", "riemann_ts_lr", pa, f"{c:g}")]
            B = pm[("paradigm_mean", "riemann_ts_lr", pb, f"{c:g}")]
            subs = sorted(set(A) & set(B))
            d = np.array([A[s] - B[s] for s in subs])
            lo, hi = C.bootstrap_ci(d)
            means.append(d.mean()); los.append(lo); his.append(hi)
            xj = c * np.exp(rng.uniform(-0.12, 0.12, len(d)) + (0.16 if pa == "trial_random" else -0.16))
            ax.scatter(xj, d, s=4, color=col, alpha=0.18, linewidths=0, rasterized=True)
        ax.plot(Cs, means, marker="o", ms=3.5, color=col, label=lab, zorder=5)
        ax.fill_between(Cs, los, his, color=col, alpha=0.25, linewidth=0, zorder=4)
        cap.append(f"{lab}: " + ", ".join(f"C={c:g}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]" for c, m, lo, hi in zip(Cs, means, los, his)))
    ax.axhline(0, color="#888888", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xlabel("Inverse penalty C (dimensionless)"); ax.set_ylabel("Accuracy difference (fraction correct)")
    ax.legend(loc="upper left", frameon=True, fontsize=6); ax.set_title("(b) protocol differences (points = subjects)")
    ax = axes[2]
    for p in PROTOS:
        ax.plot(Cs, [np.mean(wn[(p, c)]) for c in Cs], marker="o", ms=3.5, color=F.PROTO_COLOR[p], label=F.PROTO_LABEL[p])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Inverse penalty C (dimensionless)"); ax.set_ylabel("‖w‖₂ (dimensionless)")
    ax.legend(loc="upper left", frameon=True, fontsize=6); ax.set_title("(c) weight norm")
    fig.tight_layout(w_pad=0.8)
    caption = ("Figure 4. Regularization sweep of the tangent-space logistic regression on EEGBCI (paradigm mean over four paradigms, 109 subjects). "
               "(a) Test accuracy under P0 trial-random, P1 run-disjoint and P2 subject-disjoint splits (solid, mean with subject-bootstrap 95 % CI band) and the "
               "corresponding training accuracy (dotted, mean over folds) as a function of the inverse penalty C (larger C = weaker regularization). "
               "(b) Per-subject differences Δ(P1 − P2) and Δ(P0 − P2) (points = subjects, jittered; line = mean; band = 95 % CI). (c) l2 norm of the fitted weight vector (mean over folds). "
               "Values: " + " | ".join(cap))
    F.save(fig, f"fig4_sweep_{args.tag}", caption, [Path(args.e2)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
