#!/usr/bin/env python3
"""Figure 5 — subject information in tangent features (E3): (a) variance decomposition, (b) subject-ID
decoding, (c) Δ(P1 − P2) before/after subject-mean removal and Riemannian recentering (per-subject points),
(d) alignment of weight vectors with subject-specific class directions (per fold).
Usage: fig5_mechanism.py --e3a <csv> --e3b <csv> --e3c <per_subject.csv> --e3d <csv> --e1 <e1 eegbci per_subject.csv> --tag rev1
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
    ap.add_argument("--e3a", required=True)
    ap.add_argument("--e3b", required=True)
    ap.add_argument("--e3c", required=True)
    ap.add_argument("--e3d", required=True)
    ap.add_argument("--e1", required=True, help="e1 eegbci per-subject (original TS+LR ladder)")
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    files = [Path(args.e3a), Path(args.e3b), Path(args.e3c), Path(args.e3d), Path(args.e1)]
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    cap = []
    # (a) variance decomposition
    ax = axes[0, 0]
    b = [r for r in C.read_csv(Path(args.e3b)) if r["space"] == "common_tangent"]
    comps = [("frac_subject", "subject", "#0072B2"), ("frac_class", "class", "#D55E00"), ("frac_subject_x_class", "subject × class", "#E69F00"), ("frac_residual", "residual", "#BBBBBB")]
    for i, r in enumerate(b):
        bottom = 0.0
        for key, lab, col in comps:
            v = float(r[key])
            ax.bar(i, v, bottom=bottom, color=col, width=0.7, label=lab if i == 0 else None, linewidth=0)
            if v > 0.06:
                ax.text(i, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if key != "frac_residual" else "black")
            bottom += v
        cap.append(f"{C.TASK_SHORT[r['task']]}: subject {float(r['frac_subject']):.3f}, class {float(r['frac_class']):.3f}, subject×class {float(r['frac_subject_x_class']):.3f}, residual {float(r['frac_residual']):.3f}")
    ax.set_xticks(range(len(b))); ax.set_xticklabels([C.TASK_SHORT[r["task"]] for r in b], fontsize=6.5)
    ax.set_ylabel("Fraction of total sum of squares (dimensionless)"); ax.set_ylim(0, 1.18); ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("(a) variance decomposition of tangent features", pad=14)
    # (b) subject-ID decoding
    ax = axes[0, 1]
    a = C.read_csv(Path(args.e3a))
    conds = [("trial_random", "False"), ("trial_random", "True"), ("run_disjoint", "False"), ("run_disjoint", "True")]
    labels = ["trial-random\nraw", "trial-random\nsubject-centred", "run-disjoint\nraw", "run-disjoint\nsubject-centred"]
    tasks = sorted({r["task"] for r in a}, key=C.TASKS.index)
    w = 0.8 / max(1, len(tasks))
    for ti, t in enumerate(tasks):
        vals = []
        for split, cen in conds:
            rr = [r for r in a if r["task"] == t and r["split"] == split and r["centered"] == cen]
            vals.append(float(rr[0]["subject_id_accuracy"]) if rr else np.nan)
        ax.bar(np.arange(4) + (ti - (len(tasks) - 1) / 2) * w, vals, width=w * 0.95, label=C.TASK_SHORT[t], color=["#0072B2", "#56B4E9", "#009E73", "#CC79A7"][ti % 4], linewidth=0)
        cap.append(f"subject-ID {C.TASK_SHORT[t]}: " + ", ".join(f"{l.replace(chr(10), ' ')} {v:.3f}" for l, v in zip(labels, vals)))
    chance = float(a[0]["chance"])
    ax.axhline(chance, color="#888888", lw=0.8, ls="--"); ax.text(3.45, chance + 0.01, f"chance {chance:.3f}", fontsize=6, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.5); ax.set_ylabel("Subject-identity accuracy (fraction correct)"); ax.set_ylim(0, 1)
    ax.legend(loc="center right", frameon=True, fontsize=6.5); ax.set_title("(b) decoding subject identity from tangent features")
    # (c) mean removal / recentering
    ax = axes[1, 0]
    e1 = [r for r in C.read_csv(Path(args.e1)) if r["model"] == "riemann_ts_lr"]
    for r in e1:
        r["variant"] = "original"
    c3 = C.read_csv(Path(args.e3c))
    subj = F.seed_avg_subject(e1 + c3, ("task", "variant", "protocol"))
    pm = F.paradigm_mean(subj, 0)
    variants = [("original", "original"), ("feature_centered", "subject-mean\nremoved"), ("riemann_recentered", "Riemannian\nrecentred")]
    for xi, (v, lab) in enumerate(variants):
        for k, (pa, pb, col, off) in enumerate((("run_disjoint", "subject_disjoint", "#E69F00", -0.18), ("trial_random", "subject_disjoint", "#D55E00", 0.18))):
            A, B = pm.get(("paradigm_mean", v, pa), {}), pm.get(("paradigm_mean", v, pb), {})
            subs = sorted(set(A) & set(B))
            if not subs:
                continue
            d = np.array([A[s] - B[s] for s in subs])
            F.strip_points(ax, xi + off, d, col, rng, width=0.12)
            m, lo, hi = F.mean_ci(ax, xi + off, d, col, width=0.15)
            cap.append(f"{v} Δ({'P1' if pa == 'run_disjoint' else 'P0'}−P2) {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(d)}")
    ax.scatter([], [], color="#E69F00", label="Δ(P1 − P2)"); ax.scatter([], [], color="#D55E00", label="Δ(P0 − P2)")
    ax.axhline(0, color="#888888", lw=0.8, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in variants], fontsize=6.5)
    ax.set_ylabel("Accuracy difference (fraction correct)"); ax.legend(loc="upper right", frameon=True, fontsize=6.5); ax.set_title("(c) TS+LR after removing subject means")
    # (d) weight alignment
    ax = axes[1, 1]
    d3 = C.read_csv(Path(args.e3d))
    metrics = [("mean_abs_cos_w_d_s_train_subjects", "mean |cos(w, d_s)|\n(training subjects)"), ("fraction_w_in_span_train_d_s", "fraction of ‖w‖²\nin span{d_s}"), ("abs_cos_w_d_global", "|cos(w, d_global)|")]
    for xi, (met, lab) in enumerate(metrics):
        for pa, col, off in (("trial_random", "#D55E00", -0.18), ("subject_disjoint", "#0072B2", 0.18)):
            v = np.array([float(r["value"]) for r in d3 if r["metric"] == met and r["protocol"] == pa])
            if len(v) == 0:
                continue
            F.strip_points(ax, xi + off, v, col, rng, width=0.1, size=9, alpha=0.6)
            m, lo, hi = F.mean_ci(ax, xi + off, v, col, width=0.15)
            cap.append(f"{met} {pa}: {m:.3f} [{lo:.3f},{hi:.3f}] n={len(v)} folds")
    base = [float(r["value"]) for r in d3 if r["metric"] == "random_direction_fraction_in_span"]
    if base:
        ax.hlines(np.mean(base), 0.6, 1.4, color="#888888", ls="--", lw=0.8); ax.text(1.42, np.mean(base), f"random\n{np.mean(base):.3f}", fontsize=6, va="center")
    pw = [float(r["value"]) for r in d3 if r["metric"] == "mean_pairwise_abs_cos_d_s"]
    if pw:
        cap.append(f"mean pairwise |cos(d_s, d_s')| over paradigms = {np.mean(pw):.3f}")
    ax.scatter([], [], color="#D55E00", label="w trained under P0 (trial-random)"); ax.scatter([], [], color="#0072B2", label="w trained under P2 (subject-disjoint)")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in metrics], fontsize=6.5); ax.set_ylabel("Alignment (dimensionless)")
    ax.legend(loc="upper right", frameon=True, fontsize=6.5); ax.set_title(f"(d) weight alignment (points = folds × seeds)")
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    caption = ("Figure 5. Subject information in the tangent-space features (EEGBCI, 109 subjects). (a) Fraction of the total sum of squares of the tangent features "
               "attributable to subject, class, subject × class (cell means) and residual, in a common tangent space (reference = Fréchet mean of all trials; descriptive only). "
               "(b) Accuracy of a multinomial logistic regression predicting the subject (109 classes) from tangent features under trial-random and run-disjoint splits, "
               "with and without per-subject mean removal; dashed = chance. (c) Per-subject differences Δ(P1 − P2) and Δ(P0 − P2) for TS+LR (C = 1) with the original features, "
               "after subtracting each subject's feature mean, and after Riemannian recentering (points = subjects; bar = mean; whisker = 95 % CI). "
               "(d) Alignment of the fitted weight vector w with the subject-specific class-separation directions d_s and with the global direction (points = folds × seeds). Values: " + " | ".join(cap))
    F.save(fig, f"fig5_mechanism_{args.tag}", caption, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
