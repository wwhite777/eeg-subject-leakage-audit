#!/usr/bin/env python3
"""E17 classical fixed-test, count-matched inclusion and adjacency contrasts (v2 B21).

The primary tangent reference is estimated once from the unchanged background.
The arm-own reference is a supplementary full-pipeline contrast. Cache row order
within each subject/run/class is recording order; no test labels select trials.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from e1_ladder_classical import build_model


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def run_arm(data, covs, features, tr, te, model_name, reference, seed, rotation, arm):
    y = data["y"]
    model = build_model(model_name, seed)
    if model_name == "csp_lda":
        xtr, xte = data["x"][tr], data["x"][te]
        ref_hash = "not_applicable"
    elif reference == "bg_fixed":
        xtr, xte = features[tr], features[te]
        ref_hash = ""  # filled with the one rotation-level reference hash below
    else:
        xtr, xte, ref = C.fold_tangent_features(covs, tr, te)
        ref_hash = hashlib.sha256(np.asarray(ref, dtype="<f8").tobytes()).hexdigest()
    model.fit(xtr, y[tr])
    pred = model.predict(xte)
    train_acc = float(np.mean(model.predict(xtr) == y[tr]))
    base = {"dataset": data["dataset"], "task": data["task"], "model": model_name,
            "reference": reference, "arm": arm, "seed": seed, "rotation": rotation}
    subjects = []
    for subject in np.unique(data["subject"][te]):
        mask = data["subject"][te] == subject
        subjects.append({**base, "subject": int(subject), "n_test": int(mask.sum()),
                         "accuracy": C.fmt(np.mean(pred[mask] == y[te][mask]), 12)})
    summary = {**base, "n_train": len(tr), "n_train_subjects": len(np.unique(data["subject"][tr])),
               "n_class0": int(np.sum(y[tr] == 0)), "n_class1": int(np.sum(y[tr] == 1)),
               "train_acc": C.fmt(train_acc, 12), "test_acc": C.fmt(np.mean(pred == y[te]), 12),
               "train_index_sha256": index_hash(tr), "test_index_sha256": index_hash(te),
               "reference_sha256": ref_hash}
    return subjects, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["eegbci", "bci2a"], default="eegbci")
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--models", default="csp_lda,riemann_ts_lr")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=36)
    ap.add_argument("--max-subjects", type=int, default=0, help="smoke only; reduced equal-sized blocks")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    if not models or len(set(models)) != len(models) or set(models) - {"csp_lda", "riemann_ts_lr"}:
        ap.error("invalid --models")
    if args.n_seeds < 1 or args.n_jobs < 1 or args.max_subjects < 0:
        ap.error("--n-seeds and --n-jobs must be positive; --max-subjects must be nonnegative")
    from joblib import Parallel, delayed

    started, tag = C.utc_now(), args.tag or C.stamp()
    if args.dataset == "eegbci":
        tasks = [t for t in args.tasks.split(",") if t]
        if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(C.TASKS):
            ap.error("invalid --tasks")
    else:
        tasks = ["bci2a_left_right"]
    per_subject, per_arm, matching, counts, rotations = [], [], [], [], []
    for task in tasks:
        if args.dataset == "eegbci":
            subjects = C.discover_subjects(task)
            data = C.load_eegbci_task(task, subjects[:args.max_subjects] if args.max_subjects else subjects)
        else:
            data = C.load_bci2a()
            if args.max_subjects:
                keep = np.isin(data["subject"], np.unique(data["subject"])[:args.max_subjects])
                data = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in data.items()}
        n_subjects = len(np.unique(data["subject"]))
        if not args.max_subjects and n_subjects != {"eegbci": 109, "bci2a": 9}[args.dataset]:
            raise AssertionError(f"full E17 requires the frozen cohort, got {n_subjects}")
        print(f"[{task}] E17 trials={len(data['y'])} subjects={n_subjects}", flush=True)
        covs = C.covariances(data["x"]) if "riemann_ts_lr" in models else None
        # All comparisons share the same loaded trial indices. Each rotation's
        # fixed reference and its feature map are computed once, before any fit.
        jobs, metadata = [], {}
        for seed in range(C.SEED, C.SEED + args.n_seeds):
            for rotation in range(3):
                design = C.fixed_test_arms(data, args.dataset, seed, rotation)
                te, bg = design["test"], design["bg"]
                ref = C.riemann_mean(covs[bg]) if covs is not None else None
                features = C.tangent(covs, ref) if ref is not None else None
                ref_hash = hashlib.sha256(np.asarray(ref, dtype="<f8").tobytes()).hexdigest() if ref is not None else "not_applicable"
                base = {"dataset": args.dataset, "task": task, "seed": seed, "rotation": rotation}
                matching.extend({**base, **row} for row in design["pairing"])
                counts.extend({**base, **row} for row in design["counts"])
                meta = {**base, "leftover_subject": design["leftover_subject"],
                        "groups": {k: np.asarray(v).tolist() for k, v in design["groups"].items()},
                        "test_indices": te.tolist(), "bg_indices": bg.tolist(),
                        "validation_indices": design["validation"].tolist(),
                        "arm_indices": {a: tr.tolist() for a, tr in design["arms"].items()},
                        "bg_index_sha256": index_hash(bg), "validation_index_sha256": index_hash(design["validation"]),
                        "reference_sha256": ref_hash}
                rotations.append(meta)
                metadata[(seed, rotation)] = meta
                for arm, tr in design["arms"].items():
                    for model in models:
                        references = ("bg_fixed", "arm_own") if model == "riemann_ts_lr" else ("bg_fixed",)
                        for reference in references:
                            jobs.append((tr, te, model, reference, seed, rotation, arm, features))
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_arm)(data, covs, features, tr, te, model, reference, seed, rotation, arm)
            for tr, te, model, reference, seed, rotation, arm, features in jobs)
        for subjects, summary in results:
            meta = metadata[(summary["seed"], summary["rotation"])]
            summary["bg_index_sha256"] = meta["bg_index_sha256"]
            summary["validation_index_sha256"] = meta["validation_index_sha256"]
            if summary["model"] == "riemann_ts_lr" and summary["reference"] == "bg_fixed":
                summary["reference_sha256"] = meta["reference_sha256"]
            per_subject.extend(subjects)
            per_arm.append(summary)
            print(f"[{task}] seed={summary['seed']} rotation={summary['rotation']} {summary['model']} {summary['reference']} {summary['arm']} accuracy={summary['test_acc']}", flush=True)
    stem = f"e17_{args.dataset}_{tag}"
    outputs = [C.write_csv(C.RESULTS / f"{stem}_{name}.csv", rows) for name, rows in (
        ("per_subject", per_subject), ("per_arm", per_arm), ("matching", matching), ("counts", counts))]
    C.validate_csv(outputs[0], ["dataset", "task", "model", "reference", "arm", "seed", "rotation", "subject", "n_test", "accuracy"], 1)
    C.validate_csv(outputs[1], ["n_train", "n_train_subjects", "n_class0", "n_class1", "train_acc", "test_acc"], 1)
    C.validate_csv(outputs[2], ["pair", "subject_T", "subject_D", "run", "class", "n_T", "n_D", "n_kept", "shortfall"], 1)
    C.write_manifest(C.RESULTS / f"{stem}_manifest.json", stem, started, outputs,
                     {"args": vars(args), "smoke": bool(args.max_subjects), "rotations": rotations,
                      "recording_time_order": "stable cache row order within subject/run/class",
                      "plan_v2_sha256": C.sha256_file(C.ROOT / "research" / "REVISION_PLAN_v2.md")})
    print(f"DONE {stem} per_subject={len(per_subject)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
