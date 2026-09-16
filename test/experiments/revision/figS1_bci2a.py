#!/usr/bin/env python3
"""Figure S1 (also used in the main text as Figure 8 if space allows) — BCI-IV-2a protocol ladder for the four
decoders with the nine subjects as connected points. Usage: figS1_bci2a.py --e1 <e1 bci2a per_subject.csv> --e10 <e10 per_subject.csv> --tag rev1
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

ORDER = ["within_session", "trial_random_6", "run_disjoint", "trial_random_2", "session_disjoint", "loso"]
SHORT = {"within_session": "within", "trial_random_6": "P0a", "run_disjoint": "P1", "trial_random_2": "P0b", "session_disjoint": "P1s", "loso": "P2"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e10", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    rows = C.read_csv(Path(args.e1)) + C.read_csv(Path(args.e10))
    subj = F.seed_avg_subject(rows, ("model", "protocol"))
    models = ["csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=True)
    cap = []
    for ax, m in zip(axes, models):
        subs = sorted(set().union(*[set(subj.get((m, p), {})) for p in ORDER]))
        for s in subs:
            ys = [subj.get((m, p), {}).get(s, np.nan) for p in ORDER]
            ax.plot(range(len(ORDER)), ys, color="#BBBBBB", lw=0.6, zorder=1)
            ax.scatter(range(len(ORDER)), ys, s=9, color=[F.PROTO_COLOR[p] for p in ORDER], zorder=2, linewidths=0)
        for xi, p in enumerate(ORDER):
            d = subj.get((m, p), {})
            if not d:
                continue
            v = np.array(list(d.values()))
            mean, lo, hi = F.mean_ci(ax, xi, v, F.PROTO_COLOR[p], width=0.3)
            cap.append(f"{F.MODEL_LABEL[m]} {p}: {mean:.3f} [{lo:.3f},{hi:.3f}] (n={len(v)})")
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--"); ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([SHORT[p] for p in ORDER], fontsize=7)
        ax.set_title(F.MODEL_LABEL[m]); ax.set_ylim(0.35, 1.02); ax.set_xlim(-0.6, len(ORDER) - 0.4)
    axes[0].set_ylabel("Accuracy (fraction correct)")
    fig.text(0.5, -0.03, "Protocol: within = session-to-session inside each subject; P0a = trial-random 6-fold; P1 = run-disjoint 6-fold; P0b = trial-random 2-fold; P1s = session-disjoint 2-fold; P2 = leave-one-subject-out\n(BCI-IV-2a, 9 subjects; grey lines connect the same subject; bar = mean, whisker = subject-bootstrap 95 % CI; dashed = chance)", ha="center", fontsize=6.5)
    fig.tight_layout(w_pad=0.5)
    caption = ("Figure S1. BCI-IV-2a left/right motor imagery under the protocol ladder for the four decoders (networks: mean over three seeds). Each grey line follows one of the nine subjects; "
               "the bar is the mean and the whisker the 2000-resample subject-bootstrap 95 % CI; dashed = chance. Training fractions: within-subject 1/2 (one session), trial-random 6-fold 5/6, run-disjoint 5/6, "
               "trial-random 2-fold 1/2, session-disjoint 1/2, LOSO 8/9. Values: " + "; ".join(cap))
    F.save(fig, f"figS1_bci2a_{args.tag}", caption, [Path(args.e1), Path(args.e10)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
