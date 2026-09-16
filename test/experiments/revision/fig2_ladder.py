#!/usr/bin/env python3
"""Figure 2 — EEGBCI protocol ladder, per-subject points, four decoders.

Inputs (per-subject CSVs): e1 eegbci (CSP+LDA, TS+LR; within/trial/run/subject), e9 t12+t34 (networks;
trial/run/subject), e11 t12+t34 (networks within-subject). Each subject's accuracy is averaged over
seeds and then over the four paradigms (paradigm mean). Mean and subject-bootstrap 95 % CI per protocol.
Usage: fig2_ladder.py --e1 <csv> --e9 <csv,csv> --e11 <csv,csv> --tag rev1
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

ORDER = ["within_subject", "trial_random", "run_disjoint", "subject_disjoint"]
SHORT = {"within_subject": "within", "trial_random": "P0", "run_disjoint": "P1", "subject_disjoint": "P2"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e9", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    rows = []
    files = []
    for p in [args.e1] + args.e9.split(",") + args.e11.split(","):
        files.append(Path(p))
        for r in C.read_csv(Path(p)):
            if r.get("width", "default") not in ("default", "") or r.get("preproc", "ztrial") == "global":
                continue
            rows.append({"task": r["task"], "model": r["model"], "protocol": r["protocol"], "subject": r["subject"], "accuracy": r["accuracy"]})
    subj = F.seed_avg_subject(rows, ("task", "model", "protocol"))
    pm = F.paradigm_mean(subj, 0)
    models = ["csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True)
    rng = np.random.default_rng(C.SEED)
    caption_bits = []
    for ax, m in zip(axes, models):
        for xi, p in enumerate(ORDER):
            d = pm.get(("paradigm_mean", m, p), {})
            if not d:
                ax.text(xi, 0.52, "not\nevaluated", ha="center", va="center", fontsize=6.5, color="#666666")
                continue
            v = np.array([d[s] for s in sorted(d)])
            F.strip_points(ax, xi, v, F.PROTO_COLOR[p], rng)
            mean, lo, hi = F.mean_ci(ax, xi, v, F.PROTO_COLOR[p])
            ax.text(xi, 0.985, f"{mean:.3f}", ha="center", va="top", fontsize=6.5)
            caption_bits.append(f"{F.MODEL_LABEL[m]} {p}: {mean:.3f} [{lo:.3f}, {hi:.3f}] (n={len(v)})")
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--", zorder=1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([SHORT[p] for p in ORDER], fontsize=7)
        ax.set_title(F.MODEL_LABEL[m])
        ax.set_ylim(0.3, 1.0)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
    axes[0].set_ylabel("Accuracy (fraction correct, paradigm mean)")
    fig.text(0.5, -0.03, "Protocol: within = leave-one-run-out inside each subject; P0 = trial-random 3-fold; P1 = run-disjoint, subject-overlapping 3-fold; P2 = subject-disjoint 3-fold\n(EEGBCI, 109 subjects; each point = one subject; bar = mean, whisker = subject-bootstrap 95 % CI; dashed = chance 0.5)", ha="center", fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    caption = ("Figure 2. Accuracy of the four decoders on EEGBCI under the protocol ladder. Each point is one subject's accuracy "
               "(fraction of its test trials predicted correctly, averaged over partition seeds and over the four paradigms); the horizontal bar is the "
               "mean over 109 subjects and the whisker its 2000-resample subject-bootstrap 95 % CI; the dashed line is chance (0.5). "
               "within-subject = leave-one-run-out inside each subject; P0 = trial-random 3-fold (subject- and run-overlapping); P1 = run-disjoint, subject-overlapping "
               "3-fold; P2 = subject-disjoint 3-fold; all three ladder protocols train on two thirds of the trials. "
               "Values: " + "; ".join(caption_bits))
    F.save(fig, f"fig2_ladder_{args.tag}", caption, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
