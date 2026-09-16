#!/usr/bin/env python3
"""E9 deep-network protocol ladder on EEGBCI (plan B4), also E7 (width sweeps, --widths) and
E13 (matched preprocessing, --preproc global).

Protocols: trial_random (3-fold), run_disjoint (3 folds), subject_disjoint (3-fold), all at
training fraction 2/3; per protocol the fold seed is the run seed. Validation for early stopping:
subject-level (GroupShuffleSplit over training subjects) for subject_disjoint; trial-level for
the other two. Seeds: --seeds (default 3).

Outputs: e9_<tag>_per_subject.csv, e9_<tag>_per_fold.csv (train_acc, val_acc, best_epoch,
n_params, width, preproc), e9_<tag>_manifest.json. One process per GPU; select with --gpu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--models", default="eegnet_v4,shallow_fbcsp")
    ap.add_argument("--protocols", default="trial_random,run_disjoint,subject_disjoint")
    ap.add_argument("--seeds", default="20260916,20260917,20260918")
    ap.add_argument("--widths", default="", help="comma list; empty = default width (E9); e.g. 2,4,8,16,32 (E7)")
    ap.add_argument("--preproc", choices=["ztrial", "global"], default="ztrial")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import torch
    import deep_common as D

    started = C.utc_now()
    tag = args.tag or C.stamp()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    models = [m for m in args.models.split(",") if m]
    protocols = [p for p in args.protocols.split(",") if p]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    widths = [int(w) for w in args.widths.split(",") if w] or [None]
    ps_rows, pf_rows = [], []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        x, y, subject = data["x"], data["y"], data["subject"]
        print(f"[{task}] trials={len(y)} subjects={len(np.unique(subject))} device={device}", flush=True)
        for model_name in models:
            for width in widths:
                for protocol in protocols:
                    for seed in seeds:
                        splits = C.ladder_splits(data, protocol, seed, 3)
                        pred = np.full(len(y), -1, dtype=np.int64)
                        for tr, te, fold in splits:
                            val_mode = "subject" if protocol == "subject_disjoint" else "trial"
                            r = D.train_predict(model_name, x[tr], y[tr], x[te], device, seed, val_mode=val_mode, groups_tr=subject[tr],
                                                epochs=args.epochs, preproc=args.preproc, width=width)
                            pred[te] = r["preds"]
                            pf_rows.append({"task": task, "model": model_name, "width": width if width is not None else "default", "preproc": args.preproc,
                                            "protocol": protocol, "seed": seed, "fold": fold, "n_train": len(tr), "n_test": len(te), "n_fit": r["n_fit"], "n_val": r["n_val"],
                                            "train_acc": C.fmt(r["train_acc"]), "val_acc": C.fmt(r["val_acc"]), "test_acc": C.fmt(np.mean(r["preds"] == y[te])),
                                            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "n_params": r["n_params"]})
                        accs = C.per_subject_acc(y, pred, subject)
                        for s, a in sorted(accs.items()):
                            ps_rows.append({"task": task, "model": model_name, "width": width if width is not None else "default", "preproc": args.preproc,
                                            "protocol": protocol, "seed": seed, "subject": s, "accuracy": C.fmt(a)})
                        print(f"    {model_name:14s} w={width!s:8s} {protocol:16s} seed={seed} mean_acc={np.mean(list(accs.values())):.4f} "
                              f"train_acc={np.mean([float(r['train_acc']) for r in pf_rows[-len(splits):]]):.3f} params={pf_rows[-1]['n_params']}", flush=True)
                        # checkpoint tables after every cell so a crash loses nothing
                        C.write_csv(C.RESULTS / f"e9_{tag}_per_subject.csv", ps_rows)
                        C.write_csv(C.RESULTS / f"e9_{tag}_per_fold.csv", pf_rows)
    ps = C.write_csv(C.RESULTS / f"e9_{tag}_per_subject.csv", ps_rows)
    pf = C.write_csv(C.RESULTS / f"e9_{tag}_per_fold.csv", pf_rows)
    C.validate_csv(ps, ["task", "model", "width", "preproc", "protocol", "seed", "subject", "accuracy"], 1)
    C.write_manifest(C.RESULTS / f"e9_{tag}_manifest.json", f"e9_{tag}", started, [ps, pf], {"args": vars(args), "device": str(device), "smoke": bool(args.max_subjects)})
    print(f"DONE e9 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
