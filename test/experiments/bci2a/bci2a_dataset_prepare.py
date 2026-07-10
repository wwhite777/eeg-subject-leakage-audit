#!/usr/bin/env python3
"""Prepare BCI Competition IV-2a (BNCI2014_001) as the replication dataset.

Second public corpus for the tangent-space reference-leakage study: 9 subjects,
22 EEG channels, two-class left-hand vs right-hand motor imagery, via MOABB's
standard loader (which downloads, band-passes and epochs). Caches one npz with
per-trial signals, integer labels, subject and run indices -- the same schema the
EEGBCI experiments consume, so the leakage protocols apply unchanged.

Run un-sandboxed (MOABB download needs network); personal venv only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_public/bci2a")
    ap.add_argument("--fmin", type=float, default=8.0)
    ap.add_argument("--fmax", type=float, default=30.0)
    args = ap.parse_args()

    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=args.fmin, fmax=args.fmax)
    subjects = dataset.subject_list
    x, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    # x: (n_trials, n_channels, n_times); labels: array of class strings
    classes = sorted(np.unique(labels))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in labels], dtype=np.int64)
    subj = meta["subject"].to_numpy().astype(np.int64)
    # combine session+run into a run id so leave-one-run-out has >=2 groups/subject
    sess = meta["session"].astype(str).to_numpy()
    run = meta["run"].astype(str).to_numpy()
    run_key = np.array([hash(s + "|" + r) % 100000 for s, r in zip(sess, run)], dtype=np.int64)

    out_path = cache_dir / "bci2a_leftright.npz"
    np.savez_compressed(
        out_path,
        x=x.astype(np.float32),
        y=y,
        subject=subj,
        run=run_key,
        classes=np.asarray(classes),
    )
    summary = {
        "dataset": "BNCI2014_001_left_right_imagery",
        "n_trials": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_times": int(x.shape[2]),
        "n_subjects": int(len(np.unique(subj))),
        "classes": list(classes),
        "class_balance": {c: int(np.sum(y == label_map[c])) for c in classes},
        "band": [args.fmin, args.fmax],
        "cache": str(out_path),
    }
    (out_dir / "bci2a_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
