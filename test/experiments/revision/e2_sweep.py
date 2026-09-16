#!/usr/bin/env python3
"""E2 regularization sweep v2 (plan B5): EEGBCI, TS+LR.

Per fold of each ladder protocol (trial_random 3-fold x 5 seeds, run_disjoint,
subject_disjoint 3-fold x 5 seeds) the inductive tangent features are computed
once, then l2 logistic regression is fitted at every C in the grid. Recorded:
per-subject test accuracy, training accuracy, ||w||_2, number of lbfgs
iterations, so that both accuracy curves, the training curves and the weight
norms can be shown (Referee 1, point 2; Referee 2, point 6).

Outputs: e2_<tag>_per_subject.csv, e2_<tag>_per_fold.csv, e2_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def run_fold(covs, y, tr, te, seed, protocol, fold):
    from sklearn.linear_model import LogisticRegression

    xtr, xte, _ = C.fold_tangent_features(covs, tr, te)
    out = []
    for c in C_GRID:
        clf = LogisticRegression(max_iter=3000, C=c, random_state=seed)
        clf.fit(xtr, y[tr])
        pred = clf.predict(xte)
        out.append({
            "C": c, "pred": pred, "train_acc": float(np.mean(clf.predict(xtr) == y[tr])),
            "w_norm": float(np.linalg.norm(clf.coef_)), "n_iter": int(np.max(clf.n_iter_)),
        })
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
        print(f"[{task}] {len(jobs)} folds x {len(C_GRID)} C values", flush=True)
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(run_fold)(covs, y, tr, te, seed, p, f) for (tr, te, seed, p, f) in jobs)
        store: dict[tuple, np.ndarray] = {}
        mask: dict[tuple, np.ndarray] = {}
        for r in results:
            for fit in r["fits"]:
                key = (r["protocol"], r["seed"], fit["C"])
                if key not in store:
                    store[key] = np.full(len(y), -1, dtype=np.int64)
                    mask[key] = np.zeros(len(y), dtype=bool)
                store[key][r["te"]] = fit["pred"]
                mask[key][r["te"]] = True
                pf_rows.append({
                    "task": task, "protocol": r["protocol"], "seed": r["seed"], "fold": r["fold"], "C": fit["C"],
                    "n_train": len(r["tr"]), "n_test": len(r["te"]), "train_acc": C.fmt(fit["train_acc"]),
                    "test_acc": C.fmt(np.mean(fit["pred"] == y[r["te"]])), "w_norm": C.fmt(fit["w_norm"]), "n_iter": fit["n_iter"],
                })
        for (protocol, seed, c), pred in sorted(store.items()):
            accs = C.per_subject_acc(y, pred, subject, mask[(protocol, seed, c)])
            for s, a in sorted(accs.items()):
                ps_rows.append({"task": task, "protocol": protocol, "seed": seed, "C": c, "subject": s, "accuracy": C.fmt(a)})
        for c in C_GRID:
            m = {p: np.mean([float(r["accuracy"]) for r in ps_rows if r["task"] == task and r["protocol"] == p and r["C"] == c]) for p in ("trial_random", "run_disjoint", "subject_disjoint")}
            print(f"    C={c:<8g} P0={m['trial_random']:.4f} P1={m['run_disjoint']:.4f} P2={m['subject_disjoint']:.4f}  d01={m['trial_random']-m['run_disjoint']:+.4f} d12={m['run_disjoint']-m['subject_disjoint']:+.4f}", flush=True)
    ps = C.write_csv(C.RESULTS / f"e2_{tag}_per_subject.csv", ps_rows)
    pf = C.write_csv(C.RESULTS / f"e2_{tag}_per_fold.csv", pf_rows)
    C.validate_csv(ps, ["task", "protocol", "seed", "C", "subject", "accuracy"], 1)
    C.validate_csv(pf, ["task", "protocol", "seed", "fold", "C", "train_acc", "test_acc", "w_norm"], 1)
    C.write_manifest(C.RESULTS / f"e2_{tag}_manifest.json", f"e2_{tag}", started, [ps, pf], {"args": vars(args), "C_grid": C_GRID, "smoke": bool(args.max_subjects)})
    print(f"DONE e2 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
