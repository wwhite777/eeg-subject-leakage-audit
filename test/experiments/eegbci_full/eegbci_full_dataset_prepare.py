#!/usr/bin/env python3
"""Download and epoch the full PhysioNet EEG Motor Movement/Imagery corpus.

Scales the BrainFuse EEGBCI track from the earlier 10-subject smoke test to all
109 subjects, for four standard two-class motor paradigms. Output is one cached
npz per (subject, task) holding leakage-audit-ready arrays: per-epoch signals,
integer labels, and the source run index (needed for leave-one-run-out
within-subject splits). No modelling happens here.

Run un-sandboxed (PhysioNet download needs DNS); uses the personal venv only.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# Four standard PhysioNet MMI two-class paradigms and the runs that carry them.
# T1/T2 annotations mean different movements depending on the run group.
TASK_RUNS: dict[str, list[int]] = {
    "imagery_left_right_fist": [4, 8, 12],
    "real_left_right_fist": [3, 7, 11],
    "imagery_fists_feet": [6, 10, 14],
    "real_fists_feet": [5, 9, 13],
}
TASK_LABELS: dict[str, dict[str, str]] = {
    "imagery_left_right_fist": {"T1": "left_fist", "T2": "right_fist"},
    "real_left_right_fist": {"T1": "left_fist", "T2": "right_fist"},
    "imagery_fists_feet": {"T1": "both_fists", "T2": "both_feet"},
    "real_fists_feet": {"T1": "both_fists", "T2": "both_feet"},
}
TARGET_SFREQ = 160.0
FILTER_LOW = 7.0
FILTER_HIGH = 30.0
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0
N_CHANS_EXPECTED = 64


def build_subject_task_epochs(subject: int, task: str, root: Path):
    """Return (X, y, runs, label_names) for one subject/task or raise on failure."""
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    runs = TASK_RUNS[task]
    files = eegbci.load_data(subject, runs, path=str(root), update_path=False, verbose="ERROR")
    raws = []
    run_of_file = []
    for run_number, fname in zip(runs, files):
        raw = read_raw_edf(fname, preload=True, verbose="ERROR")
        run_of_file.append((run_number, raw))
    # Standardise channel names and montage, then concatenate.
    prepared = []
    for run_number, raw in run_of_file:
        eegbci.standardize(raw)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore", verbose="ERROR")
        if abs(float(raw.info["sfreq"]) - TARGET_SFREQ) > 1e-3:
            raw.resample(TARGET_SFREQ, verbose="ERROR")
        raw.filter(FILTER_LOW, FILTER_HIGH, fir_design="firwin", verbose="ERROR")
        prepared.append((run_number, raw))

    xs, ys, run_ids = [], [], []
    label_names = TASK_LABELS[task]
    for run_number, raw in prepared:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        wanted = {name: event_id[name] for name in ("T1", "T2") if name in event_id}
        if len(wanted) < 2:
            continue
        epochs = mne.Epochs(
            raw,
            events,
            event_id=wanted,
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=None,
            picks="eeg",
            preload=True,
            reject=None,
            verbose="ERROR",
        )
        if len(epochs) == 0:
            continue
        data = epochs.get_data(copy=False)
        codes = epochs.events[:, 2]
        for name, code in wanted.items():
            mask = codes == code
            if not np.any(mask):
                continue
            block = data[mask]
            xs.append(block.astype(np.float32))
            ys.extend([0 if name == "T1" else 1] * block.shape[0])
            run_ids.extend([run_number] * block.shape[0])

    if not xs:
        raise RuntimeError(f"subject {subject} task {task}: no T1/T2 epochs")
    x = np.concatenate(xs, axis=0)
    if x.shape[1] != N_CHANS_EXPECTED:
        raise RuntimeError(f"subject {subject} task {task}: {x.shape[1]} channels != {N_CHANS_EXPECTED}")
    y = np.asarray(ys, dtype=np.int64)
    runs_arr = np.asarray(run_ids, dtype=np.int64)
    # Trim all subjects to a common epoch length (resampling can differ by 1 sample).
    return x, y, runs_arr, [label_names["T1"], label_names["T2"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_public/eegbci_full")
    parser.add_argument("--first-subject", type=int, default=1)
    parser.add_argument("--last-subject", type=int, default=109)
    parser.add_argument("--min-epochs-per-task", type=int, default=24)
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    inventory = []
    skipped = []
    for subject in range(args.first_subject, args.last_subject + 1):
        subject_ok = True
        subject_records = []
        for task in TASK_RUNS:
            out_path = cache_dir / f"sub{subject:03d}_{task}.npz"
            if out_path.exists():
                with np.load(out_path, allow_pickle=True) as cached:
                    n = int(cached["y"].shape[0])
                subject_records.append((task, n, str(out_path)))
                continue
            try:
                x, y, runs_arr, label_names = build_subject_task_epochs(subject, task, root)
            except Exception as exc:  # noqa: BLE001 - want to skip and log any bad subject
                skipped.append({"subject": subject, "task": task, "error": str(exc)})
                print(f"SKIP subject {subject} task {task}: {exc}", file=sys.stderr)
                subject_ok = False
                continue
            if y.shape[0] < args.min_epochs_per_task or len(np.unique(y)) < 2:
                skipped.append({"subject": subject, "task": task, "error": f"too_few_epochs_{y.shape[0]}"})
                print(f"SKIP subject {subject} task {task}: only {y.shape[0]} epochs", file=sys.stderr)
                subject_ok = False
                continue
            np.savez_compressed(
                out_path,
                x=x,
                y=y,
                runs=runs_arr,
                subject=subject,
                task=task,
                label_names=np.asarray(label_names),
                sfreq=TARGET_SFREQ,
            )
            subject_records.append((task, int(y.shape[0]), str(out_path)))
            print(f"OK   subject {subject:3d} task {task:24s} epochs={y.shape[0]:3d} shape={x.shape}")
        for task, n, path in subject_records:
            inventory.append({"subject": subject, "task": task, "epochs": n, "path": path, "subject_all_tasks_ok": subject_ok})

    summary = {
        "n_subjects_requested": args.last_subject - args.first_subject + 1,
        "n_cached_records": len(inventory),
        "n_skips": len(skipped),
        "subjects_fully_ok": sorted({r["subject"] for r in inventory if r["subject_all_tasks_ok"]}),
        "skips": skipped,
    }
    summary_path = root / "eegbci_full_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "skips"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
