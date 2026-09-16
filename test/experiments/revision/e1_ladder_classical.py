#!/usr/bin/env python3
"""E1 hierarchical split ladder (plan B3) for the classical decoders, both datasets.

Also used for E13 (matched preprocessing, classical side) with --preproc ztrial.

EEGBCI protocols (training fraction 2/3 everywhere):
  trial_random (3-fold, 5 seeds) | run_disjoint (3 folds) | subject_disjoint (3-fold, 5 seeds)
  within_subject (leave-one-run-out inside each subject)
BCI-IV-2a protocols:
  trial_random_6 (6-fold, 5 seeds) | trial_random_2 (2-fold, 5 seeds) | run_disjoint (6 folds)
  session_disjoint (2 folds) | loso (9 folds) | subject_disjoint_5 (original 5-fold, 5 seeds)
  within_session (train one session, test the other, both directions)

Outputs (results/revision/):
  e1_<dataset>_<tag>_per_subject.csv : dataset, task, model, protocol, seed, subject, n_test, accuracy
  e1_<dataset>_<tag>_per_fold.csv    : dataset, task, model, protocol, seed, fold, n_train, n_test, train_acc, test_acc
  e1_<dataset>_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

N_SEEDS = 5


def build_model(name: str, seed: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import mne
    from mne.decoding import CSP

    mne.set_log_level("ERROR")
    if name == "csp_lda":
        return Pipeline([
            ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
    if name == "riemann_ts_lr":
        return LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    raise ValueError(name)


def fit_predict(model_name: str, data: dict, covs, tr: np.ndarray, te: np.ndarray, seed: int):
    """Returns (pred_te, train_acc)."""
    y = data["y"]
    if model_name == "csp_lda":
        m = build_model(model_name, seed)
        m.fit(data["x"][tr], y[tr])
        return m.predict(data["x"][te]), float(np.mean(m.predict(data["x"][tr]) == y[tr]))
    xtr, xte, _ = C.fold_tangent_features(covs, tr, te)
    m = build_model(model_name, seed)
    m.fit(xtr, y[tr])
    return m.predict(xte), float(np.mean(m.predict(xtr) == y[tr]))


def protocol_splits(data: dict, protocol: str, seed: int):
    if data["dataset"] == "eegbci":
        if protocol == "trial_random":
            return C.split_trial_random(data["y"], 3, seed)
        if protocol == "run_disjoint":
            return C.split_run_disjoint(data["run_idx"])
        if protocol == "subject_disjoint":
            return C.split_subject_disjoint(data["subject"], 3, seed)
    else:
        if protocol == "trial_random_6":
            return C.split_trial_random(data["y"], 6, seed)
        if protocol == "trial_random_2":
            return C.split_trial_random(data["y"], 2, seed)
        if protocol == "run_disjoint":
            return C.split_run_disjoint(data["run_idx"])
        if protocol == "session_disjoint":
            return C.split_session_disjoint(data["session"])
        if protocol == "loso":
            return C.split_loso(data["subject"])
        if protocol == "subject_disjoint_5":
            return C.split_subject_disjoint(data["subject"], 5, seed)
    raise ValueError(protocol)


def seeded_protocols(dataset: str):
    if dataset == "eegbci":
        return {"trial_random": True, "run_disjoint": False, "subject_disjoint": True}
    return {"trial_random_6": True, "trial_random_2": True, "run_disjoint": False, "session_disjoint": False, "loso": False, "subject_disjoint_5": True}


def within_splits(data: dict):
    """Per-subject splits: EEGBCI leave-one-run-out; BCI-IV-2a session-to-session (both directions)."""
    out = []
    idx = np.arange(len(data["y"]))
    for s in np.unique(data["subject"]):
        sm = data["subject"] == s
        key = data["run_idx"] if data["dataset"] == "eegbci" else data["session"]
        for k in np.unique(key[sm]):
            te = idx[sm & (key == k)]
            tr = idx[sm & (key != k)]
            if len(np.unique(data["y"][tr])) < 2 or len(te) == 0:
                continue
            out.append((tr, te, f"within_s{int(s)}_h{int(k)}"))
    return out


def run_fold(model_name, data, covs, tr, te, seed, protocol, fold):
    pred, train_acc = fit_predict(model_name, data, covs, tr, te, seed)
    return {"model": model_name, "protocol": protocol, "seed": seed, "fold": fold, "tr": tr, "te": te, "pred": pred, "train_acc": train_acc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["eegbci", "bci2a"], default="eegbci")
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--models", default="csp_lda,riemann_ts_lr")
    ap.add_argument("--preproc", choices=["none", "ztrial"], default="none")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0, help="smoke runs only; never reported")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-within", action="store_true")
    args = ap.parse_args()

    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or (("ztrial_" if args.preproc == "ztrial" else "") + C.stamp())
    out_dir = C.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    seeds = [C.SEED + r for r in range(args.n_seeds)]

    if args.dataset == "eegbci":
        tasks = [t for t in args.tasks.split(",") if t]
        datasets = []
        for t in tasks:
            subs = C.discover_subjects(t)
            if args.max_subjects:
                subs = subs[: args.max_subjects]
            datasets.append(C.load_eegbci_task(t, subs))
    else:
        d = C.load_bci2a()
        if args.max_subjects:
            keep = np.isin(d["subject"], np.unique(d["subject"])[: args.max_subjects])
            d = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in d.items()}
        datasets = [d]

    per_subject_rows, per_fold_rows = [], []
    for data in datasets:
        task = data["task"]
        if args.preproc == "ztrial":
            from deep_common import zscore

            data = dict(data)
            data["x"] = zscore(data["x"])
        y, subject = data["y"], data["subject"]
        covs = C.covariances(data["x"]) if "riemann_ts_lr" in models else None
        print(f"[{task}] trials={len(y)} subjects={len(np.unique(subject))} chans={data['x'].shape[1]} times={data['x'].shape[2]}", flush=True)

        jobs = []
        for protocol, seeded in seeded_protocols(args.dataset).items():
            for seed in (seeds if seeded else [C.SEED]):
                splits = protocol_splits(data, protocol, seed)
                for tr, te, fold in splits:
                    for m in models:
                        jobs.append((m, tr, te, seed, protocol, fold))
        if not args.skip_within:
            wp = "within_subject" if args.dataset == "eegbci" else "within_session"
            for tr, te, fold in within_splits(data):
                for m in models:
                    jobs.append((m, tr, te, C.SEED, wp, fold))
        print(f"[{task}] {len(jobs)} fold-jobs", flush=True)
        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(run_fold)(m, data, covs, tr, te, seed, protocol, fold) for (m, tr, te, seed, protocol, fold) in jobs
        )
        # assemble predictions per (model, protocol, seed)
        pred_store: dict[tuple, np.ndarray] = {}
        mask_store: dict[tuple, np.ndarray] = {}
        for r in results:
            key = (r["model"], r["protocol"], r["seed"])
            if key not in pred_store:
                pred_store[key] = np.full(len(y), -1, dtype=np.int64)
                mask_store[key] = np.zeros(len(y), dtype=bool)
            pred_store[key][r["te"]] = r["pred"]
            mask_store[key][r["te"]] = True
            per_fold_rows.append({
                "dataset": args.dataset, "task": task, "model": r["model"], "protocol": r["protocol"], "seed": r["seed"], "fold": r["fold"],
                "n_train": len(r["tr"]), "n_test": len(r["te"]), "train_acc": C.fmt(r["train_acc"]),
                "test_acc": C.fmt(np.mean(r["pred"] == y[r["te"]])),
            })
        for (m, protocol, seed), pred in sorted(pred_store.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            mask = mask_store[(m, protocol, seed)]
            accs = C.per_subject_acc(y, pred, subject, mask)
            for s, a in sorted(accs.items()):
                per_subject_rows.append({
                    "dataset": args.dataset, "task": task, "model": m, "protocol": protocol, "seed": seed, "subject": s,
                    "n_test": int(np.sum(mask & (subject == s))), "accuracy": C.fmt(a),
                })
            print(f"    {m:14s} {protocol:18s} seed={seed} mean_acc={np.mean(list(accs.values())):.4f} n_subj={len(accs)}", flush=True)

    ps_path = C.write_csv(out_dir / f"e1_{args.dataset}_{tag}_per_subject.csv", per_subject_rows)
    pf_path = C.write_csv(out_dir / f"e1_{args.dataset}_{tag}_per_fold.csv", per_fold_rows)
    C.validate_csv(ps_path, ["dataset", "task", "model", "protocol", "seed", "subject", "accuracy"], 1)
    C.validate_csv(pf_path, ["dataset", "task", "model", "protocol", "seed", "fold", "train_acc", "test_acc"], 1)
    C.write_manifest(out_dir / f"e1_{args.dataset}_{tag}_manifest.json", f"e1_{args.dataset}_{tag}", started, [ps_path, pf_path],
                     {"args": vars(args), "smoke": bool(args.max_subjects), "n_jobs": args.n_jobs})
    print(f"DONE e1 {args.dataset} tag={tag} rows={len(per_subject_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
