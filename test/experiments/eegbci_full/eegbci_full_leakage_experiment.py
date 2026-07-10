#!/usr/bin/env python3
"""Leakage-audited decoding on the full PhysioNet MMI corpus.

For each two-class motor paradigm we evaluate the same open-source classifiers
under three evaluation protocols that differ only in how subjects are allowed to
straddle the train/test boundary:

  pooled_random  - StratifiedKFold over pooled epochs, subject identity ignored.
                   The SAME subject appears in train and test => subject leakage.
                   This mimics naive multi-subject benchmarking.
  within_subject - leave-one-run-out inside each subject, averaged over subjects.
                   Legitimate calibrated (subject-specific) BCI evaluation.
  cross_subject  - GroupKFold with subject as the group. Test subjects are unseen.
                   Honest zero-calibration generalization.

Every protocol yields a per-subject accuracy vector, so the three are directly
comparable. The headline quantity is the leakage inflation:
    inflation = mean_acc(pooled_random) - mean_acc(cross_subject).

Reads only the npz cache written by eegbci_full_dataset_prepare.py; runs on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

TASKS = [
    "imagery_left_right_fist",
    "real_left_right_fist",
    "imagery_fists_feet",
    "real_fists_feet",
]
CACHE_RE = re.compile(r"sub(\d{3})_(.+)\.npz$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_models(seed: int) -> dict[str, Pipeline]:
    from mne.decoding import CSP
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    return {
        "csp_lda": Pipeline(
            [
                ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
                ("lda", LinearDiscriminantAnalysis()),
            ]
        ),
        "riemann_ts_lr": Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
            ]
        ),
    }


def discover_subjects(cache_dir: Path, task: str) -> list[int]:
    subjects = []
    for path in cache_dir.glob(f"sub*_{task}.npz"):
        m = CACHE_RE.search(path.name)
        if m and m.group(2) == task:
            subjects.append(int(m.group(1)))
    return sorted(subjects)


def load_task(cache_dir: Path, task: str, subjects: list[int]):
    xs, ys, groups, runs = [], [], [], []
    min_t = None
    blocks = []
    for subject in subjects:
        path = cache_dir / f"sub{subject:03d}_{task}.npz"
        with np.load(path, allow_pickle=True) as cached:
            x = cached["x"].astype(np.float32)
            y = cached["y"].astype(np.int64)
            r = cached["runs"].astype(np.int64)
        min_t = x.shape[2] if min_t is None else min(min_t, x.shape[2])
        blocks.append((subject, x, y, r))
    for subject, x, y, r in blocks:
        x = x[:, :, :min_t]
        xs.append(x)
        ys.append(y)
        groups.append(np.full(y.shape[0], subject, dtype=np.int64))
        runs.append(r)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(groups, axis=0),
        np.concatenate(runs, axis=0),
    )


def per_subject_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    out = {}
    for subject in np.unique(groups):
        mask = groups == subject
        out[int(subject)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return out


def bootstrap_ci(values: np.ndarray, seed: int, n: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eval_pooled_random(model_factory, x, y, groups, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for train_idx, test_idx in skf.split(x, y):
        model = model_factory()
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])
    return per_subject_accuracy(y, preds, groups)


def eval_cross_subject(model_factory, x, y, groups, seed):
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.empty_like(y)
    for train_idx, test_idx in gkf.split(x, y, groups):
        model = model_factory()
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])
    return per_subject_accuracy(y, preds, groups)


def eval_within_subject(model_factory, x, y, groups, runs, seed):
    out = {}
    for subject in np.unique(groups):
        smask = groups == subject
        sx, sy, sr = x[smask], y[smask], runs[smask]
        unique_runs = np.unique(sr)
        if len(unique_runs) < 2:
            continue
        preds = np.empty_like(sy)
        used = np.zeros(len(sy), dtype=bool)
        for held in unique_runs:
            test_mask = sr == held
            train_mask = ~test_mask
            if len(np.unique(sy[train_mask])) < 2 or test_mask.sum() == 0:
                continue
            model = model_factory()
            model.fit(sx[train_mask], sy[train_mask])
            preds[test_mask] = model.predict(sx[test_mask])
            used |= test_mask
        if used.sum() == 0:
            continue
        out[int(subject)] = float(accuracy_score(sy[used], preds[used]))
    return out


PROTOCOLS = {
    "pooled_random": "leaky_subject_ignored",
    "within_subject": "legit_leave_one_run_out",
    "cross_subject": "rigorous_group_kfold",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data_public/eegbci_full/cache")
    parser.add_argument("--out-dir", default="results/eegbci_full")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--min-subjects", type=int, default=10)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_subject_rows = []
    for task in TASKS:
        subjects = discover_subjects(cache_dir, task)
        if len(subjects) < args.min_subjects:
            print(f"skip {task}: only {len(subjects)} subjects cached", file=sys.stderr)
            continue
        x, y, groups, runs = load_task(cache_dir, task, subjects)
        chance = float(max(np.mean(y == 0), np.mean(y == 1)))
        print(f"[{task}] subjects={len(subjects)} epochs={len(y)} chance={chance:.3f}")

        models = build_models(args.seed)
        for model_name in models:
            def factory(mn=model_name):
                return build_models(args.seed)[mn]

            protocol_results = {}
            for protocol in PROTOCOLS:
                if protocol == "pooled_random":
                    acc_by_subject = eval_pooled_random(factory, x, y, groups, args.seed)
                elif protocol == "cross_subject":
                    acc_by_subject = eval_cross_subject(factory, x, y, groups, args.seed)
                else:
                    acc_by_subject = eval_within_subject(factory, x, y, groups, runs, args.seed)
                accs = np.array(sorted(acc_by_subject.values()) or [float("nan")], dtype=float)
                ci_low, ci_high = bootstrap_ci(accs, args.seed)
                protocol_results[protocol] = float(np.mean(accs))
                summary_rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "protocol": protocol,
                        "protocol_kind": PROTOCOLS[protocol],
                        "n_subjects": len(acc_by_subject),
                        "n_epochs": int(len(y)),
                        "chance": f"{chance:.6g}",
                        "mean_acc": f"{np.mean(accs):.6g}",
                        "std_acc": f"{np.std(accs):.6g}",
                        "ci_low": f"{ci_low:.6g}",
                        "ci_high": f"{ci_high:.6g}",
                    }
                )
                for subject, acc in sorted(acc_by_subject.items()):
                    per_subject_rows.append(
                        {"task": task, "model": model_name, "protocol": protocol, "subject": subject, "accuracy": f"{acc:.6g}"}
                    )
                print(f"    {model_name:14s} {protocol:14s} mean_acc={np.mean(accs):.4f} (n={len(acc_by_subject)})")
            inflation = protocol_results["pooled_random"] - protocol_results["cross_subject"]
            print(f"    -> LEAKAGE INFLATION {model_name}: {inflation:+.4f} (pooled_random - cross_subject)")

    tag = args.tag
    summary_path = out_dir / f"BrainFuse_eegbci_full_leakage_summary_{tag}.csv"
    per_subject_path = out_dir / f"BrainFuse_eegbci_full_leakage_per_subject_{tag}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["task", "model", "protocol", "protocol_kind", "n_subjects", "n_epochs", "chance", "mean_acc", "std_acc", "ci_low", "ci_high"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with per_subject_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["task", "model", "protocol", "subject", "accuracy"])
        writer.writeheader()
        writer.writerows(per_subject_rows)

    manifest = {
        "run_id": f"brainfuse_eegbci_full_leakage_{tag}",
        "python": sys.version.split()[0],
        "seed": args.seed,
        "hardware": "cpu",
        "cache_dir": str(cache_dir),
        "n_summary_rows": len(summary_rows),
        "start_time_utc": utc_now(),
        "notes": "CSP+LDA and Riemannian tangent-space LR under pooled_random/within_subject/cross_subject on full PhysioNet MMI.",
    }
    (out_dir / f"BrainFuse_eegbci_full_leakage_run_manifest_{tag}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary_rows": len(summary_rows), "summary_path": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
