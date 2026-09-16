#!/usr/bin/env python3
"""E6 target-subject-in-reference test with the classifier training data fixed (plan B9).

Protocol P0 (trial-random 3-fold, one seed). For each fold and each test subject s:
  R_full : Riemannian mean of ALL training-fold covariances (includes s's training trials)
  R_-s   : Riemannian mean of the training-fold covariances of all subjects except s
           (warm-started from R_full)
The classifier (l2 LR, C = 1) is trained on the SAME training trials in both cases; only the
tangent-space reference differs (features are recomputed under each reference). It is evaluated on
subject s's test trials. Delta_ref(s) = acc_s(R_full) - acc_s(R_-s).

Outputs: e6_<tag>_per_subject.csv (task, fold, subject, n_test, acc_full, acc_minus, delta), e6_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def full_fold(covs, y, tr, te, seed):
    from sklearn.linear_model import LogisticRegression

    ref = C.riemann_mean(covs[tr])
    xtr, xte = C.tangent(covs[tr], ref), C.tangent(covs[te], ref)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(xtr, y[tr])
    return ref, clf.predict(xte)


def minus_subject(covs, y, subject, tr, te, s, ref_full, seed):
    from sklearn.linear_model import LogisticRegression

    keep = subject[tr] != s
    ref = C.riemann_mean(covs[tr][keep], init=ref_full)
    xtr = C.tangent(covs[tr], ref)  # same training trials, new reference
    te_s = te[subject[te] == s]
    xte = C.tangent(covs[te_s], ref)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(xtr, y[tr])
    pred = clf.predict(xte)
    dist = float(np.linalg.norm(C.tangent(ref[None], ref_full)[0]))  # Riemannian distance between references
    return {"subject": int(s), "te_s": te_s, "pred": pred, "ref_dist": dist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    rows = []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        y, subject = data["y"], data["subject"]
        covs = C.covariances(data["x"])
        splits = C.split_trial_random(y, 3, C.SEED)
        fulls = Parallel(n_jobs=min(args.n_jobs, 3), backend="loky")(delayed(full_fold)(covs, y, tr, te, C.SEED) for tr, te, _ in splits)
        jobs = []
        for (tr, te, fold), (ref_full, pred_full) in zip(splits, fulls):
            for s in np.unique(subject[te]):
                jobs.append((tr, te, fold, s, ref_full, pred_full))
        print(f"[{task}] {len(jobs)} (fold, subject) jobs", flush=True)
        res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(minus_subject)(covs, y, subject, tr, te, s, ref_full, C.SEED) for (tr, te, fold, s, ref_full, pred_full) in jobs)
        for (tr, te, fold, s, ref_full, pred_full), r in zip(jobs, res):
            te_s = r["te_s"]
            pos = np.searchsorted(te, te_s)  # te is sorted (StratifiedKFold returns sorted test indices)
            acc_full = float(np.mean(pred_full[pos] == y[te_s]))
            acc_minus = float(np.mean(r["pred"] == y[te_s]))
            rows.append({"task": task, "fold": fold, "subject": int(s), "n_test": len(te_s), "acc_full": C.fmt(acc_full), "acc_minus": C.fmt(acc_minus),
                         "delta": C.fmt(acc_full - acc_minus), "ref_distance": C.fmt(r["ref_dist"])})
        d = np.array([float(r["delta"]) for r in rows if r["task"] == task])
        lo, hi = C.bootstrap_ci(d)
        print(f"    mean delta(full - minus) = {d.mean():+.5f} 95%CI [{lo:+.5f},{hi:+.5f}] n={len(d)}", flush=True)
    ps = C.write_csv(C.RESULTS / f"e6_{tag}_per_subject.csv", rows)
    C.validate_csv(ps, ["task", "fold", "subject", "acc_full", "acc_minus", "delta"], 1)
    C.write_manifest(C.RESULTS / f"e6_{tag}_manifest.json", f"e6_{tag}", started, [ps], {"args": vars(args), "smoke": bool(args.max_subjects)})
    print(f"DONE e6 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
