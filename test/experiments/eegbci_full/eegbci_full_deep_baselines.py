#!/usr/bin/env python3
"""Deep-net rung of the capacity ladder for the EEGBCI leakage audit.

Runs braindecode EEGNetv4 and ShallowFBCSPNet under the pooled_random (leaky)
and cross_subject (honest) protocols on all 109 PhysioNet MMI subjects, so the
leakage-inflation vs model-capacity relationship can be extended past the
classical CSP+LDA / Riemannian rungs. Reuses the cached npz and the split logic
from eegbci_full_leakage_experiment.py. Uses CUDA if usable, else CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eegbci_full_leakage_experiment import (  # noqa: E402
    TASKS,
    bootstrap_ci,
    discover_subjects,
    load_task,
    per_subject_accuracy,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def make_model(name: str, n_chans: int, n_outputs: int, n_times: int):
    from braindecode.models import EEGNetv4, ShallowFBCSPNet

    if name == "eegnet_v4":
        return EEGNetv4(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    if name == "shallow_fbcsp":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, final_conv_length="auto")
    raise ValueError(name)


def train_predict(model_name, x_tr, y_tr, x_te, device, seed, epochs, batch_size, lr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_out = int(len(np.unique(y_tr)))
    # carve a stratified validation slice for early stopping
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_tr)), test_size=0.15, random_state=seed, stratify=y_tr
    )
    xt = torch.from_numpy(zscore(x_tr[idx_tr]))
    yt = torch.from_numpy(y_tr[idx_tr])
    xv = torch.from_numpy(zscore(x_tr[idx_val])).to(device)
    yv = y_tr[idx_val]
    model = make_model(model_name, x_tr.shape[1], n_out, x_tr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    best_state, best_acc, bad = None, -1.0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        for s in range(0, len(order), batch_size):
            b = order[s : s + batch_size]
            xb = xt[b].to(device)
            yb = yt[b].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(xv).argmax(1).cpu().numpy()
        acc = accuracy_score(yv, vp)
        if acc > best_acc:
            best_acc, best_state, bad = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(zscore(x_te)).to(device)).argmax(1).cpu().numpy()
    return preds


def eval_pooled_random(model_name, x, y, groups, device, seed, epochs, bs, lr):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for tr, te in skf.split(x, y):
        preds[te] = train_predict(model_name, x[tr], y[tr], x[te], device, seed, epochs, bs, lr)
    return per_subject_accuracy(y, preds, groups)


def eval_cross_subject(model_name, x, y, groups, device, seed, epochs, bs, lr):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    preds = np.empty_like(y)
    for tr, te in gkf.split(x, y, groups):
        preds[te] = train_predict(model_name, x[tr], y[tr], x[te], device, seed, epochs, bs, lr)
    return per_subject_accuracy(y, preds, groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_public/eegbci_full/cache")
    ap.add_argument("--out-dir", default="results/eegbci_full")
    ap.add_argument("--tag", default="20260709")
    ap.add_argument("--models", default="eegnet_v4,shallow_fbcsp")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    tasks = [t for t in args.tasks.split(",") if t]

    rows = []
    for task in tasks:
        subjects = discover_subjects(cache_dir, task)
        if args.max_subjects:
            subjects = subjects[: args.max_subjects]
        if len(subjects) < 5:
            print(f"skip {task}: {len(subjects)} subjects", file=sys.stderr)
            continue
        x, y, groups, runs = load_task(cache_dir, task, subjects)
        chance = float(max(np.mean(y == 0), np.mean(y == 1)))
        print(f"[{task}] subjects={len(subjects)} epochs={len(y)} device={device}", flush=True)
        for model_name in models:
            for protocol, fn in (("pooled_random", eval_pooled_random), ("cross_subject", eval_cross_subject)):
                acc_by_subject = fn(model_name, x, y, groups, device, args.seed, args.epochs, args.batch_size, args.lr)
                accs = np.array(sorted(acc_by_subject.values()), dtype=float)
                lo, hi = bootstrap_ci(accs, args.seed)
                rows.append({
                    "task": task, "model": model_name, "protocol": protocol,
                    "protocol_kind": "leaky_subject_ignored" if protocol == "pooled_random" else "rigorous_group_kfold",
                    "n_subjects": len(acc_by_subject), "n_epochs": int(len(y)), "chance": f"{chance:.6g}",
                    "mean_acc": f"{np.mean(accs):.6g}", "std_acc": f"{np.std(accs):.6g}",
                    "ci_low": f"{lo:.6g}", "ci_high": f"{hi:.6g}",
                })
                print(f"    {model_name:14s} {protocol:14s} mean_acc={np.mean(accs):.4f}", flush=True)
            infl = float(rows[-2]["mean_acc"]) - float(rows[-1]["mean_acc"])
            print(f"    -> LEAKAGE INFLATION {model_name}: {infl:+.4f}", flush=True)

    out_path = out_dir / f"BrainFuse_eegbci_full_deep_summary_{args.tag}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "model", "protocol", "protocol_kind", "n_subjects", "n_epochs", "chance", "mean_acc", "std_acc", "ci_low", "ci_high"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"BrainFuse_eegbci_full_deep_run_manifest_{args.tag}.json").write_text(
        json.dumps({"run_id": f"eegbci_deep_{args.tag}", "device": str(device), "models": models,
                    "epochs": args.epochs, "seed": args.seed, "n_rows": len(rows), "time_utc": utc_now()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
