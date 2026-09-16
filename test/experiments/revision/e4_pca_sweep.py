#!/usr/bin/env python3
"""E4 PCA-dimension sweep of the tangent features (plan B7), EEGBCI, LR (C = 1).

Per fold: inductive tangent features -> PCA fitted on the training fold ->
first d components -> l2 logistic regression. d = 2080 means no PCA.
Protocols: trial_random (3-fold x seeds), run_disjoint, subject_disjoint (3-fold x seeds).

Outputs: e4_<tag>_per_subject.csv, e4_<tag>_per_fold.csv, e4_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

D_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2080]


def run_fold(covs, y, tr, te, seed, protocol, fold, d_grid):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    xtr, xte, _ = C.fold_tangent_features(covs, tr, te)
    full_d = xtr.shape[1]
    max_pca = min(max(d for d in d_grid if d < full_d), len(tr) - 1, full_d)
    pca = PCA(n_components=max_pca, svd_solver="randomized", random_state=seed).fit(xtr)
    ztr, zte = pca.transform(xtr), pca.transform(xte)
    out = []
    for d in d_grid:
        if d >= full_d:
            a, b, dd = xtr, xte, full_d
        elif d > max_pca:
            continue
        else:
            a, b, dd = ztr[:, :d], zte[:, :d], d
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(a, y[tr])
        out.append({"d": dd, "pred": clf.predict(b), "train_acc": float(np.mean(clf.predict(a) == y[tr])), "w_norm": float(np.linalg.norm(clf.coef_)),
                    "explained_var": float(np.sum(pca.explained_variance_ratio_[:d])) if d < full_d else 1.0})
    return {"protocol": protocol, "seed": seed, "fold": fold, "tr": tr, "te": te, "fits": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    seeds = [C.SEED + r for r in range(args.n_seeds)]
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
        print(f"[{task}] {len(jobs)} folds x {len(D_GRID)} dimensions", flush=True)
        res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(run_fold)(covs, y, tr, te, seed, p, f, D_GRID) for (tr, te, seed, p, f) in jobs)
        store, mask = {}, {}
        for r in res:
            for fit in r["fits"]:
                key = (r["protocol"], r["seed"], fit["d"])
                store.setdefault(key, np.full(len(y), -1))[r["te"]] = fit["pred"]
                mask.setdefault(key, np.zeros(len(y), dtype=bool))[r["te"]] = True
                pf_rows.append({"task": task, "protocol": r["protocol"], "seed": r["seed"], "fold": r["fold"], "d": fit["d"], "n_train": len(r["tr"]), "n_test": len(r["te"]),
                                "train_acc": C.fmt(fit["train_acc"]), "test_acc": C.fmt(np.mean(fit["pred"] == y[r["te"]])), "w_norm": C.fmt(fit["w_norm"]), "explained_var": C.fmt(fit["explained_var"])})
        for (protocol, seed, d), pred in sorted(store.items()):
            accs = C.per_subject_acc(y, pred, subject, mask[(protocol, seed, d)])
            for s, a in sorted(accs.items()):
                ps_rows.append({"task": task, "protocol": protocol, "seed": seed, "d": d, "subject": s, "accuracy": C.fmt(a)})
        for d in sorted({r["d"] for r in ps_rows if r["task"] == task}):
            m = {p: np.mean([float(r["accuracy"]) for r in ps_rows if r["task"] == task and r["protocol"] == p and r["d"] == d]) for p in ("trial_random", "run_disjoint", "subject_disjoint")}
            print(f"    d={d:<5d} P0={m['trial_random']:.4f} P1={m['run_disjoint']:.4f} P2={m['subject_disjoint']:.4f} d12={m['run_disjoint']-m['subject_disjoint']:+.4f}", flush=True)
    ps = C.write_csv(C.RESULTS / f"e4_{tag}_per_subject.csv", ps_rows)
    pf = C.write_csv(C.RESULTS / f"e4_{tag}_per_fold.csv", pf_rows)
    C.validate_csv(ps, ["task", "protocol", "seed", "d", "subject", "accuracy"], 1)
    C.write_manifest(C.RESULTS / f"e4_{tag}_manifest.json", f"e4_{tag}", started, [ps, pf], {"args": vars(args), "d_grid": D_GRID, "smoke": bool(args.max_subjects)})
    print(f"DONE e4 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
