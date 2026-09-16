#!/usr/bin/env python3
"""E10 deep networks on BCI-IV-2a (plan B12) and E11 within-session deep results (plan B13).

Protocols (same names as e1_ladder_classical.py for bci2a):
  trial_random_6 | trial_random_2 | run_disjoint | session_disjoint | loso | subject_disjoint_5 | within_session
Validation for early stopping: subject-level for loso/subject_disjoint_5, trial-level otherwise
(within_session: 15 % trial-level slice of the 144 training trials, plan B13). Seeds: --seeds.

Outputs: e10_<tag>_per_subject.csv, e10_<tag>_per_fold.csv, e10_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from e1_ladder_classical import protocol_splits, within_splits  # noqa: E402

PROTOCOLS = ["trial_random_6", "trial_random_2", "run_disjoint", "session_disjoint", "loso", "subject_disjoint_5", "within_session"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="eegnet_v4,shallow_fbcsp")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--seeds", default="20260916,20260917,20260918")
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
    data = C.load_bci2a()
    if args.max_subjects:
        keep = np.isin(data["subject"], np.unique(data["subject"])[: args.max_subjects])
        data = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in data.items()}
    x, y, subject = data["x"], data["y"], data["subject"]
    print(f"[bci2a] trials={len(y)} subjects={len(np.unique(subject))} chans={x.shape[1]} times={x.shape[2]} device={device}", flush=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    ps_rows, pf_rows = [], []
    for model_name in [m for m in args.models.split(",") if m]:
        for protocol in [p for p in args.protocols.split(",") if p]:
            for seed in seeds:
                if protocol == "within_session":
                    splits = within_splits(data)
                    val_mode = "trial"
                else:
                    splits = protocol_splits(data, protocol, seed)
                    val_mode = "subject" if protocol in ("loso", "subject_disjoint_5") else "trial"
                pred = np.full(len(y), -1, dtype=np.int64)
                mask = np.zeros(len(y), dtype=bool)
                for tr, te, fold in splits:
                    r = D.train_predict(model_name, x[tr], y[tr], x[te], device, seed, val_mode=val_mode, groups_tr=subject[tr], epochs=args.epochs, preproc=args.preproc)
                    pred[te] = r["preds"]
                    mask[te] = True
                    pf_rows.append({"dataset": "bci2a", "task": "bci2a_left_right", "model": model_name, "preproc": args.preproc, "protocol": protocol, "seed": seed, "fold": fold,
                                    "n_train": len(tr), "n_test": len(te), "n_fit": r["n_fit"], "n_val": r["n_val"], "train_acc": C.fmt(r["train_acc"]), "val_acc": C.fmt(r["val_acc"]),
                                    "test_acc": C.fmt(np.mean(r["preds"] == y[te])), "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "n_params": r["n_params"]})
                accs = C.per_subject_acc(y, pred, subject, mask)
                for s, a in sorted(accs.items()):
                    ps_rows.append({"dataset": "bci2a", "task": "bci2a_left_right", "model": model_name, "preproc": args.preproc, "protocol": protocol, "seed": seed, "subject": s,
                                    "n_test": int(np.sum(mask & (subject == s))), "accuracy": C.fmt(a)})
                print(f"    {model_name:14s} {protocol:18s} seed={seed} mean_acc={np.mean(list(accs.values())):.4f}", flush=True)
                C.write_csv(C.RESULTS / f"e10_{tag}_per_subject.csv", ps_rows)
                C.write_csv(C.RESULTS / f"e10_{tag}_per_fold.csv", pf_rows)
    ps = C.write_csv(C.RESULTS / f"e10_{tag}_per_subject.csv", ps_rows)
    pf = C.write_csv(C.RESULTS / f"e10_{tag}_per_fold.csv", pf_rows)
    C.validate_csv(ps, ["dataset", "task", "model", "protocol", "seed", "subject", "accuracy"], 1)
    C.write_manifest(C.RESULTS / f"e10_{tag}_manifest.json", f"e10_{tag}", started, [ps, pf], {"args": vars(args), "device": str(device), "smoke": bool(args.max_subjects)})
    print(f"DONE e10 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
