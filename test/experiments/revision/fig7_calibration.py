#!/usr/bin/env python3
"""Figure 7 — labelled calibration-size curve for unseen subjects (E12) with within-subject and subject-disjoint
reference values; per-subject points. Usage: fig7_calibration.py --e12 <csv,csv> --e1 <e1 eegbci per_subject.csv> --e11 <csv,csv> --tag rev1
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e12", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    files = [Path(p) for p in args.e12.split(",")] + [Path(args.e1)] + [Path(p) for p in args.e11.split(",")]
    rows = []
    for p in args.e12.split(","):
        rows += C.read_csv(Path(p))
    tasks = sorted({r["task"] for r in rows}, key=C.TASKS.index)  # the paradigms actually run (plan §D fallback: imagery only)
    subj = F.seed_avg_subject(rows, ("task", "model", "k"), acc_key="accuracy")  # averages draws
    pm = F.paradigm_mean(subj, 0, tasks=tasks)
    ref_rows = [r for r in C.read_csv(Path(args.e1)) if r["protocol"] == "within_subject" and r["task"] in tasks]
    for p in args.e11.split(","):
        ref_rows += [r for r in C.read_csv(Path(p)) if r["task"] in tasks]
    ref = F.paradigm_mean(F.seed_avg_subject(ref_rows, ("task", "model", "protocol")), 0, tasks=tasks)
    models = [m for m in ("csp_lda", "riemann_ts_lr", "eegnet_v4") if any(k[1] == m for k in pm)]
    ks = sorted({int(k[2]) for k in pm})
    fig, axes = plt.subplots(1, len(models), figsize=(7.4, 2.7), sharey=True, squeeze=False)
    rng = np.random.default_rng(C.SEED)
    cap = []
    for ax, m in zip(axes[0], models):
        means, los, his = [], [], []
        for i, k in enumerate(ks):
            v = np.array(list(pm[("paradigm_mean", m, str(k))].values()))
            lo, hi = C.bootstrap_ci(v); means.append(v.mean()); los.append(lo); his.append(hi)
            F.strip_points(ax, i, v, F.MODEL_COLOR[m], rng, width=0.14)
        ax.plot(range(len(ks)), means, marker=F.MODEL_MARK[m], ms=4, color="black", zorder=5, label="mean ± 95 % CI")
        ax.fill_between(range(len(ks)), los, his, color="black", alpha=0.15, linewidth=0, zorder=4)
        w = ref.get(("paradigm_mean", m, "within_subject"))
        if w:
            wv = np.array(list(w.values()))
            ax.axhline(wv.mean(), color="#009E73", lw=1.2, ls="--", label=f"within-subject (leave-one-run-out) {wv.mean():.3f}")
        ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_xlabel("Labelled calibration trials (count)")
        ax.set_title(F.MODEL_LABEL[m]); ax.set_ylim(0.3, 1.12); ax.legend(loc="upper center", frameon=True, fontsize=6)
        cap.append(f"{F.MODEL_LABEL[m]}: " + ", ".join(f"k={k}: {mm:.3f} [{lo:.3f},{hi:.3f}]" for k, mm, lo, hi in zip(ks, means, los, his)) + (f"; within-subject {wv.mean():.3f}" if w else ""))
    axes[0, 0].set_ylabel("Accuracy on the subject's held-out run (fraction correct)")
    fig.tight_layout(w_pad=0.6)
    caption = (f"Figure 7. Calibration-size curve for unseen subjects on EEGBCI ({' and '.join(C.TASK_SHORT[t] for t in tasks)}; mean over these paradigms; 109 subjects). A subject-disjoint model (P2 training subjects) receives k labelled "
               "trials of the test subject (drawn from its first two runs, 3 draws averaged) and is evaluated on the subject's third run; CSP+LDA and TS+LR are refitted with the k trials appended, "
               "EEGNet is fine-tuned for 20 epochs. Points = subjects; line = mean; band = subject-bootstrap 95 % CI; dashed green = the within-subject (leave-one-run-out) mean of the same decoder; dotted = chance. "
               "Values: " + " | ".join(cap))
    F.save(fig, f"fig7_calibration_{args.tag}", caption, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
