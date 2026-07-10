#!/usr/bin/env python3
"""Isolate the tangent-space REFERENCE as a subject-leakage channel.

Riemannian tangent-space decoding maps each trial covariance C_i into the tangent
plane at a reference mean covariance G (the Frechet mean). G is unsupervised, so
practitioners routinely estimate it on ALL available covariances -- including the
(unlabeled) test trials. This "transductive reference" feels safe but lets test-
subject geometry influence the whitening. Here we measure that specific leak.

Protocol: honest cross-subject GroupKFold (the CLASSIFIER only ever trains on
training-subject labels). For each fold we compare two references, everything
else identical:
  inductive     - G estimated on TRAINING-subject covariances only  (leakage-safe)
  transductive  - G estimated on train + test covariances           (common, leaky)
reference_leakage = acc(transductive) - acc(inductive), per subject then averaged.

Riemannian only; reuses the npz cache. CPU, fast.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eegbci_full_leakage_experiment import TASKS, bootstrap_ci, discover_subjects, load_task  # noqa: E402


def fit_predict(cov_tr, y_tr, cov_te, ref, seed):
    x_tr = tangent_space(cov_tr, ref, metric="riemann")
    x_te = tangent_space(cov_te, ref, metric="riemann")
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(x_tr, y_tr)
    return clf.predict(x_te)


def per_subject_acc(y, pred, groups):
    out = {}
    for s in np.unique(groups):
        m = groups == s
        out[int(s)] = float(accuracy_score(y[m], pred[m]))
    return out


def reference_ablation_from_arrays(x, y, groups, seed):
    """Inductive vs transductive tangent-space reference under honest cross-subject CV.

    Returns (acc_inductive_by_subject, acc_transductive_by_subject). The classifier
    only ever trains on training-subject labels; the two runs differ solely in
    whether the reference mean is estimated on train-only or on train+test covs.
    """
    covs = Covariances(estimator="oas").transform(x)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred_ind = np.empty_like(y)
    pred_trans = np.empty_like(y)
    for tr, te in gkf.split(covs, y, groups):
        ref_ind = mean_covariance(covs[tr], metric="riemann")
        ref_trans = mean_covariance(np.concatenate([covs[tr], covs[te]], axis=0), metric="riemann")
        pred_ind[te] = fit_predict(covs[tr], y[tr], covs[te], ref_ind, seed)
        pred_trans[te] = fit_predict(covs[tr], y[tr], covs[te], ref_trans, seed)
    return per_subject_acc(y, pred_ind, groups), per_subject_acc(y, pred_trans, groups)


def run_task(cache_dir, task, subjects, seed):
    x, y, groups, runs = load_task(cache_dir, task, subjects)
    return reference_ablation_from_arrays(x, y, groups, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_public/eegbci_full/cache")
    ap.add_argument("--out", default="results/eegbci_full/BrainFuse_eegbci_reference_leakage_20260709.csv")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seed", type=int, default=20260709)
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    rows = []
    for task in [t for t in args.tasks.split(",") if t]:
        subjects = discover_subjects(cache_dir, task)
        if len(subjects) < 5:
            continue
        acc_ind, acc_trans = run_task(cache_dir, task, subjects, args.seed)
        subs = sorted(acc_ind)
        ind = np.array([acc_ind[s] for s in subs])
        trans = np.array([acc_trans[s] for s in subs])
        delta = trans - ind
        lo_i, hi_i = bootstrap_ci(ind, args.seed)
        lo_t, hi_t = bootstrap_ci(trans, args.seed)
        lo_d, hi_d = bootstrap_ci(delta, args.seed)
        rows.append({
            "task": task, "n_subjects": len(subs),
            "acc_inductive": f"{ind.mean():.6g}", "ind_ci_low": f"{lo_i:.6g}", "ind_ci_high": f"{hi_i:.6g}",
            "acc_transductive": f"{trans.mean():.6g}", "trans_ci_low": f"{lo_t:.6g}", "trans_ci_high": f"{hi_t:.6g}",
            "reference_leakage": f"{delta.mean():.6g}", "leak_ci_low": f"{lo_d:.6g}", "leak_ci_high": f"{hi_d:.6g}",
        })
        print(f"[{task}] inductive={ind.mean():.4f} transductive={trans.mean():.4f} "
              f"reference_leakage={delta.mean():+.4f} (95%CI {lo_d:+.4f},{hi_d:+.4f})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    m = np.mean([float(r["reference_leakage"]) for r in rows])
    print(f"\nMEAN reference_leakage across tasks: {m:+.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
