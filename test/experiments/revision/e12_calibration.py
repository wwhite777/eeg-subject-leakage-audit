#!/usr/bin/env python3
"""E12 labelled calibration-size curve for unseen subjects on EEGBCI (plan B14).

Base training set = the subject-disjoint 3-fold training subjects (seed 20260916) of the fold in
which the target subject is a test subject. The target's evaluation set = its third run (run index 2);
k in {0, 4, 8, 16, 30} labelled calibration trials are drawn (stratified, --n-draws draws) from its
first two runs and appended to the training set (weight 1). CSP+LDA and TS+LR are refitted; EEGNet
(the fold's P2 model, trained once per fold with subject-level validation) is fine-tuned for 20 epochs
on the k trials. k = 0 is the plain subject-disjoint model.

Outputs: e12_<tag>_per_subject.csv (task, model, k, draw, subject, n_test, accuracy), e12_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

K_GRID = [0, 4, 8, 16, 30]


def draw_calibration(y_pool: np.ndarray, k: int, rng) -> np.ndarray:
    """Stratified draw of k indices (into y_pool) — k/2 per class when possible."""
    if k == 0:
        return np.empty(0, dtype=np.int64)
    idx = []
    per = k // 2
    for c in np.unique(y_pool):
        cand = np.where(y_pool == c)[0]
        idx.extend(rng.choice(cand, size=min(per, len(cand)), replace=False).tolist())
    rest = k - len(idx)
    if rest > 0:
        remaining = np.setdiff1d(np.arange(len(y_pool)), idx)
        idx.extend(rng.choice(remaining, size=min(rest, len(remaining)), replace=False).tolist())
    return np.array(sorted(idx), dtype=np.int64)


def classical_subject(model_name, data, covs, tr_base, s, draws, k_grid, seed):
    """All (k, draw) cells for one target subject; returns rows."""
    import mne
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    mne.set_log_level("ERROR")
    y, subject, run_idx = data["y"], data["subject"], data["run_idx"]
    s_idx = np.where(subject == s)[0]
    eval_idx = s_idx[run_idx[s_idx] == 2]
    pool_idx = s_idx[run_idx[s_idx] != 2]
    rows = []
    for draw in range(draws):
        rng = np.random.default_rng(seed + 100 * draw + int(s))
        for k in k_grid:
            if k == 0 and draw > 0:
                continue  # k = 0 does not depend on the draw
            cal = pool_idx[draw_calibration(y[pool_idx], k, rng)] if k else np.empty(0, dtype=np.int64)
            tr = np.concatenate([tr_base, cal])
            if model_name == "csp_lda":
                m = Pipeline([("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)), ("lda", LinearDiscriminantAnalysis())])
                m.fit(data["x"][tr], y[tr])
                pred = m.predict(data["x"][eval_idx])
            else:
                xtr, xte, _ = C.fold_tangent_features(covs, tr, eval_idx)
                pred = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(xtr, y[tr]).predict(xte)
            rows.append({"model": model_name, "k": k, "draw": draw, "subject": int(s), "n_test": len(eval_idx), "accuracy": float(np.mean(pred == y[eval_idx]))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--models", default="csp_lda,riemann_ts_lr,eegnet_v4")
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    models = [m for m in args.models.split(",") if m]
    rows = []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        y, subject, run_idx = data["y"], data["subject"], data["run_idx"]
        covs = C.covariances(data["x"]) if "riemann_ts_lr" in models else None
        splits = C.split_subject_disjoint(subject, 3, C.SEED)
        print(f"[{task}] subjects={len(subs)}", flush=True)
        for model_name in [m for m in models if m != "eegnet_v4"]:
            jobs = [(tr, s) for tr, te, _ in splits for s in np.unique(subject[te])]
            res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(classical_subject)(model_name, data, covs, tr, s, args.n_draws, K_GRID, C.SEED) for tr, s in jobs)
            for rr in res:
                for r in rr:
                    rows.append({"task": task, **{k: (C.fmt(v) if isinstance(v, float) else v) for k, v in r.items()}})
            for k in K_GRID:
                vals = [float(r["accuracy"]) for r in rows if r["task"] == task and r["model"] == model_name and r["k"] == k]
                print(f"    {model_name:14s} k={k:<3d} mean_acc={np.mean(vals):.4f} n={len(vals)}", flush=True)
        if "eegnet_v4" in models:
            import torch
            import deep_common as D

            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            for tr, te, fold in splits:
                base = D.train_predict("eegnet_v4", data["x"][tr], y[tr], data["x"][te], device, C.SEED, val_mode="subject", groups_tr=subject[tr], return_model=True)
                for s in np.unique(subject[te]):
                    s_idx = np.where(subject == s)[0]
                    eval_idx = s_idx[run_idx[s_idx] == 2]
                    pool_idx = s_idx[run_idx[s_idx] != 2]
                    for draw in range(args.n_draws):
                        rng = np.random.default_rng(C.SEED + 100 * draw + int(s))
                        for k in K_GRID:
                            if k == 0 and draw > 0:
                                continue
                            cal = pool_idx[draw_calibration(y[pool_idx], k, rng)] if k else np.empty(0, dtype=np.int64)
                            pred = D.fine_tune_predict(base["model"], data["x"][cal], y[cal], data["x"][eval_idx], device, C.SEED, epochs=20, preproc="ztrial", scale=base["scale"])
                            rows.append({"task": task, "model": "eegnet_v4", "k": k, "draw": draw, "subject": int(s), "n_test": len(eval_idx), "accuracy": C.fmt(np.mean(pred == y[eval_idx]))})
                print(f"    eegnet fold {fold} done", flush=True)
            for k in K_GRID:
                vals = [float(r["accuracy"]) for r in rows if r["task"] == task and r["model"] == "eegnet_v4" and r["k"] == k]
                print(f"    {'eegnet_v4':14s} k={k:<3d} mean_acc={np.mean(vals):.4f} n={len(vals)}", flush=True)
        C.write_csv(C.RESULTS / f"e12_{tag}_per_subject.csv", rows)
    out = C.write_csv(C.RESULTS / f"e12_{tag}_per_subject.csv", rows)
    C.validate_csv(out, ["task", "model", "k", "draw", "subject", "accuracy"], 1)
    C.write_manifest(C.RESULTS / f"e12_{tag}_manifest.json", f"e12_{tag}", started, [out], {"args": vars(args), "k_grid": K_GRID, "smoke": bool(args.max_subjects)})
    print(f"DONE e12 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
