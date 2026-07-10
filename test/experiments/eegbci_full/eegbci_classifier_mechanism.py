#!/usr/bin/env python3
"""Test whether the Riemannian pooled-split inflation is CLASSIFIER overfitting.

The reference-mean ablation already ruled out the tangent-space reference as the
leak channel. The remaining hypothesis: the high-dimensional tangent features let
the logistic-regression classifier fit subject-correlated directions, which pays
off only when the same subject is in train and test (pooled_random). If so, the
inflation should SHRINK as we regularize the classifier harder (smaller C).

For each task we compute tangent features once per fold (inductive reference), then
sweep the LR penalty C under pooled_random and cross_subject. Output: inflation vs C.
CPU; reads the npz cache.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eegbci_full_leakage_experiment import TASKS, discover_subjects, load_task  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 20260709


def per_subject_mean_acc(y, pred, groups):
    accs = [accuracy_score(y[groups == s], pred[groups == s]) for s in np.unique(groups)]
    return float(np.mean(accs))


def run_protocol(covs, y, groups, splits):
    """Return {C: mean_per_subject_acc} using inductive per-fold tangent features."""
    # tangent features per fold, reused across C
    fold_feats = []
    for tr, te in splits:
        ref = mean_covariance(covs[tr], metric="riemann")
        xtr = tangent_space(covs[tr], ref, metric="riemann")
        xte = tangent_space(covs[te], ref, metric="riemann")
        fold_feats.append((tr, te, xtr, xte))
    out = {}
    for C in C_GRID:
        pred = np.empty_like(y)
        for tr, te, xtr, xte in fold_feats:
            clf = LogisticRegression(max_iter=3000, C=C, random_state=SEED)
            clf.fit(xtr, y[tr])
            pred[te] = clf.predict(xte)
        out[C] = per_subject_mean_acc(y, pred, groups)
    return out


def main() -> int:
    cache_dir = Path("data_public/eegbci_full/cache")
    rows = []
    for task in TASKS:
        subjects = discover_subjects(cache_dir, task)
        if len(subjects) < 5:
            continue
        x, y, groups, runs = load_task(cache_dir, task, subjects)
        covs = Covariances(estimator="oas").transform(x)
        pooled_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(covs, y))
        cross_splits = list(GroupKFold(n_splits=5).split(covs, y, groups))
        pooled = run_protocol(covs, y, groups, pooled_splits)
        cross = run_protocol(covs, y, groups, cross_splits)
        for C in C_GRID:
            infl = pooled[C] - cross[C]
            rows.append({"task": task, "C": C, "pooled_random": f"{pooled[C]:.6g}",
                         "cross_subject": f"{cross[C]:.6g}", "leakage_inflation": f"{infl:.6g}"})
            print(f"[{task}] C={C:<7g} pooled={pooled[C]:.4f} cross={cross[C]:.4f} inflation={infl:+.4f}", flush=True)
    out = Path("results/eegbci_full/BrainFuse_eegbci_classifier_mechanism_20260709.csv")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary: mean inflation vs C
    print("\nMEAN inflation by C (averaged over tasks):", flush=True)
    for C in C_GRID:
        vals = [float(r["leakage_inflation"]) for r in rows if float(r["C"]) == C]
        print(f"  C={C:<7g} mean_inflation={np.mean(vals):+.4f}", flush=True)
    print(f"wrote {out}\nDONE classifier mechanism", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
