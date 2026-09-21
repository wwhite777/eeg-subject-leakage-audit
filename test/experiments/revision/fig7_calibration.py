#!/usr/bin/env python3
"""Figure 7 — labelled calibration-size curve for unseen subjects (E12) with the within-subject reference
value; per-subject points.

Codex r002 finding F-r002-5: k is a CAP (nominal k). The availability table is read so that the caption
states how many subject-paradigm cells fell short of the nominal k, and the within-subject line is labelled
as an UNMATCHED reference (trained on all three runs under a different protocol; orientation only).
Usage: fig7_calibration.py --e12 <csv,csv> --e1 <e1 eegbci per_subject.csv> --e11 <csv,csv> [--avail <csv>] --tag rev1
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
from matplotlib.lines import Line2D  # noqa: E402


def availability_note(path: Path, ks: list[int]) -> str:
    """Exact-data sentence about the nominal-vs-actual calibration size (no hand-typed counts)."""
    if not path.exists():
        return ""
    rows = C.read_csv(path)
    bits = []
    for k in ks:
        col = f"actual_k{k}"
        if k == 0 or col not in rows[0]:
            continue
        vals = [int(r[col]) for r in rows]
        short = [v for v in vals if v < k]
        if short:
            bits.append(f"at the nominal k = {k}, {len(short)} of {len(vals)} subject-paradigm cells had only {min(short)}–{max(short)} trials available")
    if not bits:
        return f"every subject-paradigm cell supplied the full nominal k (checked over {len(rows)} cells)."
    return "; ".join(bits) + f"; the remaining cells supplied the full nominal k (checked over {len(rows)} cells)."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e12", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--avail", default=str(C.RESULTS / "e12_rev1_availability.csv"))
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
    fig, axes = plt.subplots(1, len(models), figsize=(F.FIG_W_IN, 3.3), sharey=True, squeeze=False)
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
            ax.axhline(wv.mean(), color="#009E73", lw=1.2, ls="--", zorder=3)
            ax.text(len(ks) - 1, wv.mean() + 0.015, f"{wv.mean():.3f}", ha="right", va="bottom", fontsize=8, color="#00654a")
        ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
        ax.set_xlabel("Labelled calibration\ntrials (nominal cap k)")
        ax.set_title(F.MODEL_LABEL[m]); ax.set_ylim(0.3, 1.12)
        cap.append(f"{F.MODEL_LABEL[m]}: " + ", ".join(f"k={k}: {mm:.3f} [{lo:.3f},{hi:.3f}]" for k, mm, lo, hi in zip(ks, means, los, his)) + (f"; within-subject reference {wv.mean():.3f}" if w else ""))
    axes[0, 0].set_ylabel("Accuracy on the subject's\nheld-out run (fraction correct)")
    handles = [Line2D([], [], color="black", lw=1.4, marker="o", ms=4), Line2D([], [], color="#009E73", lw=1.2, ls="--"), Line2D([], [], color="#888888", lw=0.8, ls=":")]
    labels = ["mean ± 95 % CI (points = subjects)", "within-subject reference (unmatched)", "chance"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, columnspacing=1.6, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.085, 1, 1), w_pad=1.0)
    note = availability_note(Path(args.avail), ks)
    caption = (f"Figure 7. Calibration-size curve for unseen subjects on EEGBCI ({' and '.join(C.TASK_SHORT[t] for t in tasks)}; mean over these paradigms; 109 subjects). A subject-disjoint model (P2 training subjects) receives "
               "up to k labelled trials of the test subject (k is a nominal cap drawn from the subject's first two runs, 3 draws averaged) and is evaluated on the subject's third run; CSP+LDA and TS+LR are refitted with the k trials appended, "
               "EEGNet is fine-tuned for 20 epochs. Nominal versus actually available calibration trials: " + note + " "
               "Points = subjects; line = mean; band = subject-bootstrap 95 % CI; dotted = chance. The dashed green line is the within-subject (leave-one-run-out) mean of the same decoder: it is trained on all three runs "
               "of the test subject under a different protocol, so it is an UNMATCHED reference shown for orientation only and is not a paired comparison with the k-trial curve. "
               "Values: " + " | ".join(cap))
    F.save(fig, f"fig7_calibration_{args.tag}", caption, files + [Path(args.avail)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
