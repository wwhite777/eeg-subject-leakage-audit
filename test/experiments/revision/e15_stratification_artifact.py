#!/usr/bin/env python3
"""E15 (EXPLORATORY, post-hoc; added 2026-09-16 after seeing E1/E2): why is the trial-random split (P0) LESS
accurate than the run-disjoint split (P1) for weak classifiers (CSP+LDA, strongly regularized TS+LR)?

Hypothesis: under a globally stratified trial-random split, the class proportion of a subject's TRAINING trials
is anti-correlated with the class proportion of its TEST trials (a subject's surplus of class-1 trials in the
test fold is a deficit in the training fold). A classifier that partly follows per-subject class priors is then
pushed towards the class that is under-represented in that subject's test fold. No model fitting is needed to
measure the effect: a "subject-prior" predictor that outputs the majority class of the subject's training trials
scores below 0.5 under P0 and exactly at its expectation under P1.

Outputs: e15_<tag>.csv with, per task/protocol/seed/subject: train class-1 fraction, test class-1 fraction,
subject-prior accuracy; and summary rows with the correlation between train and test fractions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    started = C.utc_now()
    tag = args.tag or C.stamp()
    rows, summ = [], []
    for task in [t for t in args.tasks.split(",") if t]:
        data = C.load_eegbci_task(task)
        y, subject = data["y"], data["subject"]
        for protocol in ("trial_random", "run_disjoint", "subject_disjoint"):
            seeds = [C.SEED + r for r in range(args.n_seeds)] if protocol != "run_disjoint" else [C.SEED]
            for seed in seeds:
                splits = C.ladder_splits(data, protocol, seed, 3)
                pred = np.full(len(y), -1)
                tr_frac = np.full(len(y), np.nan)
                for tr, te, fold in splits:
                    for s in np.unique(subject[te]):
                        m_tr = tr[subject[tr] == s]
                        m_te = te[subject[te] == s]
                        if len(m_tr) == 0:  # subject-disjoint: no training trials of the test subject -> global prior
                            p1 = float(np.mean(y[tr]))
                        else:
                            p1 = float(np.mean(y[m_tr]))
                        pred[m_te] = 1 if p1 > 0.5 else (0 if p1 < 0.5 else int(seed % 2))
                        tr_frac[m_te] = p1
                for s in np.unique(subject):
                    m = subject == s
                    rows.append({"task": task, "protocol": protocol, "seed": seed, "subject": int(s), "train_class1_fraction": C.fmt(np.nanmean(tr_frac[m])),
                                 "test_class1_fraction": C.fmt(np.mean(y[m])), "subject_prior_accuracy": C.fmt(np.mean(pred[m] == y[m]))})
                sub_rows = [r for r in rows if r["task"] == task and r["protocol"] == protocol and r["seed"] == seed]
                a = np.array([float(r["subject_prior_accuracy"]) for r in sub_rows])
                # per-fold anti-correlation: correlate train fraction with test fraction across (subject, fold) cells
                cells_tr, cells_te = [], []
                for tr, te, fold in splits:
                    for s in np.unique(subject[te]):
                        m_tr = tr[subject[tr] == s]
                        m_te = te[subject[te] == s]
                        if len(m_tr) and len(m_te):
                            cells_tr.append(np.mean(y[m_tr])); cells_te.append(np.mean(y[m_te]))
                corr = float(np.corrcoef(cells_tr, cells_te)[0, 1]) if len(cells_tr) > 2 else float("nan")
                lo, hi = C.bootstrap_ci(a)
                summ.append({"task": task, "protocol": protocol, "seed": seed, "n_subjects": len(a), "subject_prior_accuracy_mean": C.fmt(a.mean()), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi),
                             "corr_train_test_class_fraction": C.fmt(corr), "n_cells": len(cells_tr)})
                print(f"[{task}] {protocol:16s} seed={seed} subject-prior acc={a.mean():.4f} [{lo:.4f},{hi:.4f}]  corr(train frac, test frac)={corr:+.3f}", flush=True)
    p1 = C.write_csv(C.RESULTS / f"e15_{tag}_per_subject.csv", rows)
    p2 = C.write_csv(C.RESULTS / f"e15_{tag}_summary.csv", summ)
    C.write_manifest(C.RESULTS / f"e15_{tag}_manifest.json", f"e15_{tag}", started, [p1, p2], {"args": vars(args), "class": "exploratory_post_hoc"})
    print(f"DONE e15 tag={tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
