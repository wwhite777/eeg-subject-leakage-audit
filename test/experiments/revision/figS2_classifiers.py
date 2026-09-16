#!/usr/bin/env python3
"""Figure S2 — several classifiers on identical tangent features (E5): Δ(P1 − P2) and Δ(P0 − P2) per classifier
with per-subject points, plus subject-disjoint accuracy. Usage: figS2_classifiers.py --e5 <per_subject.csv> --tag rev1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import fig_common as F  # noqa: E402

ORDER = ["lr", "slda", "linsvm", "ridge", "knn", "mdm"]
LABEL = {"lr": "LR (l2, C=1)", "slda": "shrinkage LDA", "linsvm": "linear SVM", "ridge": "ridge", "knn": "k-NN (k=5)", "mdm": "MDM (covariances)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    rows = C.read_csv(Path(args.e5))
    subj = F.seed_avg_subject(rows, ("task", "clf", "protocol"))
    pm = F.paradigm_mean(subj, 0)
    clfs = [c for c in ORDER if any(k[1] == c for k in pm)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    rng = np.random.default_rng(C.SEED)
    cap = []
    ax = axes[0]
    for xi, c in enumerate(clfs):
        for pa, pb, col, off in (("run_disjoint", "subject_disjoint", "#E69F00", -0.18), ("trial_random", "subject_disjoint", "#D55E00", 0.18)):
            A, B = pm[("paradigm_mean", c, pa)], pm[("paradigm_mean", c, pb)]
            subs = sorted(set(A) & set(B)); d = np.array([A[s] - B[s] for s in subs])
            F.strip_points(ax, xi + off, d, col, rng, width=0.12)
            m, lo, hi = F.mean_ci(ax, xi + off, d, col, width=0.15)
            cap.append(f"{LABEL[c]} Δ({'P1' if pa == 'run_disjoint' else 'P0'}−P2) {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    ax.scatter([], [], color="#E69F00", label="Δ(P1 − P2)"); ax.scatter([], [], color="#D55E00", label="Δ(P0 − P2)")
    ax.axhline(0, color="#888888", lw=0.8, ls="--"); ax.set_xticks(range(len(clfs))); ax.set_xticklabels([LABEL[c] for c in clfs], fontsize=6, rotation=20, ha="right")
    ax.set_ylabel("Accuracy difference (fraction correct)"); ax.legend(loc="upper right", frameon=True, fontsize=6.5); ax.set_title("(a) protocol differences per classifier (points = subjects)")
    ax = axes[1]
    for xi, c in enumerate(clfs):
        for p, off in (("trial_random", -0.22), ("run_disjoint", 0.0), ("subject_disjoint", 0.22)):
            v = np.array(list(pm[("paradigm_mean", c, p)].values()))
            F.strip_points(ax, xi + off, v, F.PROTO_COLOR[p], rng, width=0.08, size=5)
            m, lo, hi = F.mean_ci(ax, xi + off, v, F.PROTO_COLOR[p], width=0.1)
            cap.append(f"{LABEL[c]} {p} {m:.3f} [{lo:.3f},{hi:.3f}]")
    for p in ("trial_random", "run_disjoint", "subject_disjoint"):
        ax.scatter([], [], color=F.PROTO_COLOR[p], label=F.PROTO_LABEL[p])
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--"); ax.set_xticks(range(len(clfs))); ax.set_xticklabels([LABEL[c] for c in clfs], fontsize=6, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (fraction correct)"); ax.legend(loc="lower right", frameon=True, fontsize=6.5); ax.set_title("(b) accuracy per protocol")
    fig.tight_layout(w_pad=0.8)
    caption = ("Figure S2. Six classifiers applied to identical inductive tangent-space features of EEGBCI (paradigm mean, 109 subjects; MDM operates on the covariance matrices directly). "
               "(a) Per-subject differences Δ(P1 − P2) and Δ(P0 − P2) (points = subjects; bar = mean; whisker = subject-bootstrap 95 % CI; dashed = 0). (b) Accuracy under the three ladder protocols. Values: " + " | ".join(cap))
    F.save(fig, f"figS2_classifiers_{args.tag}", caption, [Path(args.e5)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
