#!/usr/bin/env python3
"""E14 transductive-reference test with per-subject deltas (plan B12/B16), both datasets.

Under the subject-disjoint protocol (EEGBCI: 3-fold, seed 20260916; BCI-IV-2a: LOSO) the tangent-space reference
is estimated either from the training-subject covariances only (inductive) or from training + test covariances
(transductive, label-free); the classifier (l2 LR, C = 1) trains on the same training trials in both cases.
delta(s) = acc_transductive(s) - acc_inductive(s) per test subject; assessed with TOST (+-0.01) in e_stats.py.

Outputs: e14_<tag>_per_subject.csv (dataset, task, protocol, subject, n_test, acc_inductive, acc_transductive, delta), e14_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def fold_job(covs, y, tr, te, seed):
    from sklearn.linear_model import LogisticRegression

    ref_ind = C.riemann_mean(covs[tr])
    ref_tr = C.riemann_mean(np.concatenate([covs[tr], covs[te]], axis=0), init=ref_ind)
    out = {}
    for name, ref in (("inductive", ref_ind), ("transductive", ref_tr)):
        xtr, xte = C.tangent(covs[tr], ref), C.tangent(covs[te], ref)
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(xtr, y[tr])
        out[name] = clf.predict(xte)
    return {"te": te, **out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    rows = []
    jobs = []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        jobs.append(("eegbci", task, "subject_disjoint", data, C.split_subject_disjoint(data["subject"], 3, C.SEED)))
    d2 = C.load_bci2a()
    jobs.append(("bci2a", "bci2a_left_right", "loso", d2, C.split_loso(d2["subject"])))
    for dataset, task, protocol, data, splits in jobs:
        y, subject = data["y"], data["subject"]
        covs = C.covariances(data["x"])
        res = Parallel(n_jobs=min(args.n_jobs, len(splits)), backend="loky")(delayed(fold_job)(covs, y, tr, te, C.SEED) for tr, te, _ in splits)
        pred_i, pred_t = np.full(len(y), -1), np.full(len(y), -1)
        for r in res:
            pred_i[r["te"]] = r["inductive"]
            pred_t[r["te"]] = r["transductive"]
        acc_i, acc_t = C.per_subject_acc(y, pred_i, subject), C.per_subject_acc(y, pred_t, subject)
        for s in sorted(acc_i):
            rows.append({"dataset": dataset, "task": task, "protocol": protocol, "subject": s, "n_test": int(np.sum(subject == s)),
                         "acc_inductive": C.fmt(acc_i[s]), "acc_transductive": C.fmt(acc_t[s]), "delta": C.fmt(acc_t[s] - acc_i[s])})
        d = np.array([acc_t[s] - acc_i[s] for s in acc_i])
        lo, hi = C.bootstrap_ci(d)
        print(f"[{dataset} {task}] inductive={np.mean(list(acc_i.values())):.4f} transductive={np.mean(list(acc_t.values())):.4f} delta={d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] n={len(d)}", flush=True)
    out = C.write_csv(C.RESULTS / f"e14_{tag}_per_subject.csv", rows)
    C.validate_csv(out, ["dataset", "task", "protocol", "subject", "acc_inductive", "acc_transductive", "delta"], 1)
    C.write_manifest(C.RESULTS / f"e14_{tag}_manifest.json", f"e14_{tag}", started, [out], {"args": vars(args), "smoke": bool(args.max_subjects)})
    print(f"DONE e14 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
