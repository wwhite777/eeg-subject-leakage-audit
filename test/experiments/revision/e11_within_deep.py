#!/usr/bin/env python3
"""E11 within-subject deep-network results on EEGBCI (plan B13).

Leave-one-run-out inside each subject (3 runs per paradigm): the network trains on the
subject's other two runs (~30 trials) for a fixed 60 epochs without early stopping
(pre-specified: a validation slice of a 30-trial set is unreliable) and predicts the held-out run.
Per-subject accuracy = fraction of the subject's trials predicted correctly over its three folds.

Outputs: e11_<tag>_per_subject.csv, e11_<tag>_per_fold.csv, e11_<tag>_manifest.json
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
    ap.add_argument("--seeds", default="20260916")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--preproc", choices=["ztrial", "global"], default="ztrial")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    import torch
    import deep_common as D

    started = C.utc_now()
    tag = args.tag or C.stamp()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    ps_rows, pf_rows = [], []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        x, y, subject, run_idx = data["x"], data["y"], data["subject"], data["run_idx"]
        print(f"[{task}] subjects={len(subs)} device={device}", flush=True)
        for model_name in [m for m in args.models.split(",") if m]:
            for seed in seeds:
                for s in np.unique(subject):
                    sm = np.where(subject == s)[0]
                    pred = np.full(len(sm), -1, dtype=np.int64)
                    used = np.zeros(len(sm), dtype=bool)
                    for k in np.unique(run_idx[sm]):
                        te_local = np.where(run_idx[sm] == k)[0]
                        tr_local = np.where(run_idx[sm] != k)[0]
                        if len(np.unique(y[sm][tr_local])) < 2 or len(te_local) == 0:
                            continue
                        r = D.train_predict(model_name, x[sm][tr_local], y[sm][tr_local], x[sm][te_local], device, seed, val_mode="none", epochs=args.epochs, preproc=args.preproc)
                        pred[te_local] = r["preds"]
                        used[te_local] = True
                        pf_rows.append({"task": task, "model": model_name, "seed": seed, "subject": int(s), "held_out_run": int(k), "n_train": len(tr_local), "n_test": len(te_local),
                                        "train_acc": C.fmt(r["train_acc"]), "test_acc": C.fmt(np.mean(r["preds"] == y[sm][te_local])), "epochs_run": r["epochs_run"], "n_params": r["n_params"]})
                    if used.any():
                        ps_rows.append({"task": task, "model": model_name, "protocol": "within_subject", "seed": seed, "subject": int(s), "n_test": int(used.sum()),
                                        "accuracy": C.fmt(np.mean(pred[used] == y[sm][used]))})
                vals = [float(r["accuracy"]) for r in ps_rows if r["task"] == task and r["model"] == model_name and r["seed"] == seed]
                print(f"    {model_name:14s} within_subject seed={seed} mean_acc={np.mean(vals):.4f} n={len(vals)}", flush=True)
                C.write_csv(C.RESULTS / f"e11_{tag}_per_subject.csv", ps_rows)
                C.write_csv(C.RESULTS / f"e11_{tag}_per_fold.csv", pf_rows)
    ps = C.write_csv(C.RESULTS / f"e11_{tag}_per_subject.csv", ps_rows)
    pf = C.write_csv(C.RESULTS / f"e11_{tag}_per_fold.csv", pf_rows)
    C.validate_csv(ps, ["task", "model", "protocol", "seed", "subject", "accuracy"], 1)
    C.write_manifest(C.RESULTS / f"e11_{tag}_manifest.json", f"e11_{tag}", started, [ps, pf], {"args": vars(args), "device": str(device), "smoke": bool(args.max_subjects)})
    print(f"DONE e11 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
