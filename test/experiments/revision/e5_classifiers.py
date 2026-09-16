#!/usr/bin/env python3
"""E5 several classifiers on identical tangent features (plan B8), EEGBCI.

lr (l2 LR, C = 1) | slda (LDA, Ledoit-Wolf shrinkage) | linsvm (LinearSVC, C = 1)
ridge (RidgeClassifier, alpha = 1) | knn (k = 5) | mdm (minimum distance to
Riemannian mean on covariances; a classifier with no fitted weights beyond the
two class means -- the zero-capacity Riemannian control).
Protocols: trial_random (3-fold x seeds), run_disjoint, subject_disjoint (3-fold x seeds).

Outputs: e5_<tag>_per_subject.csv, e5_<tag>_per_fold.csv, e5_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

CLFS = ["lr", "slda", "linsvm", "ridge", "knn", "mdm"]


def make_clf(name: str, seed: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    if name == "lr":
        return LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    if name == "slda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "linsvm":
        return LinearSVC(C=1.0, max_iter=10000, random_state=seed)
    if name == "ridge":
        return RidgeClassifier(alpha=1.0)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(name)


def run_fold(covs, y, tr, te, seed, protocol, fold, clfs):
    from pyriemann.classification import MDM

    out = []
    need_ts = any(c != "mdm" for c in clfs)
    if need_ts:
        xtr, xte, _ = C.fold_tangent_features(covs, tr, te)
    for name in clfs:
        if name == "mdm":
            m = MDM(metric="riemann").fit(covs[tr], y[tr])
            pred, tra = m.predict(covs[te]), float(np.mean(m.predict(covs[tr]) == y[tr]))
        else:
            m = make_clf(name, seed).fit(xtr, y[tr])
            pred, tra = m.predict(xte), float(np.mean(m.predict(xtr) == y[tr]))
        out.append({"clf": name, "pred": pred, "train_acc": tra})
    return {"protocol": protocol, "seed": seed, "fold": fold, "tr": tr, "te": te, "fits": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--clfs", default=",".join(CLFS))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    seeds = [C.SEED + r for r in range(args.n_seeds)]
    clfs = [c for c in args.clfs.split(",") if c]
    ps_rows, pf_rows = [], []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        y, subject = data["y"], data["subject"]
        covs = C.covariances(data["x"])
        jobs = []
        for protocol in ("trial_random", "run_disjoint", "subject_disjoint"):
            for seed in (seeds if protocol != "run_disjoint" else [C.SEED]):
                for tr, te, fold in C.ladder_splits(data, protocol, seed, 3):
                    jobs.append((tr, te, seed, protocol, fold))
        print(f"[{task}] {len(jobs)} folds x {len(clfs)} classifiers", flush=True)
        res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(run_fold)(covs, y, tr, te, seed, p, f, clfs) for (tr, te, seed, p, f) in jobs)
        store, mask = {}, {}
        for r in res:
            for fit in r["fits"]:
                key = (fit["clf"], r["protocol"], r["seed"])
                store.setdefault(key, np.full(len(y), -1))[r["te"]] = fit["pred"]
                mask.setdefault(key, np.zeros(len(y), dtype=bool))[r["te"]] = True
                pf_rows.append({"task": task, "clf": fit["clf"], "protocol": r["protocol"], "seed": r["seed"], "fold": r["fold"], "n_train": len(r["tr"]), "n_test": len(r["te"]),
                                "train_acc": C.fmt(fit["train_acc"]), "test_acc": C.fmt(np.mean(fit["pred"] == y[r["te"]]))})
        for (clf, protocol, seed), pred in sorted(store.items()):
            accs = C.per_subject_acc(y, pred, subject, mask[(clf, protocol, seed)])
            for s, a in sorted(accs.items()):
                ps_rows.append({"task": task, "clf": clf, "protocol": protocol, "seed": seed, "subject": s, "accuracy": C.fmt(a)})
        for clf in clfs:
            m = {p: np.mean([float(r["accuracy"]) for r in ps_rows if r["task"] == task and r["clf"] == clf and r["protocol"] == p]) for p in ("trial_random", "run_disjoint", "subject_disjoint")}
            print(f"    {clf:7s} P0={m['trial_random']:.4f} P1={m['run_disjoint']:.4f} P2={m['subject_disjoint']:.4f} d12={m['run_disjoint']-m['subject_disjoint']:+.4f} d02={m['trial_random']-m['subject_disjoint']:+.4f}", flush=True)
    ps = C.write_csv(C.RESULTS / f"e5_{tag}_per_subject.csv", ps_rows)
    pf = C.write_csv(C.RESULTS / f"e5_{tag}_per_fold.csv", pf_rows)
    C.validate_csv(ps, ["task", "clf", "protocol", "seed", "subject", "accuracy"], 1)
    C.write_manifest(C.RESULTS / f"e5_{tag}_manifest.json", f"e5_{tag}", started, [ps, pf], {"args": vars(args), "smoke": bool(args.max_subjects)})
    print(f"DONE e5 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
