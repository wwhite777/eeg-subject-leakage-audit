#!/usr/bin/env python3
# origin: bootstrap-seed sensitivity script, 2026-10-30
"""Reconstruct E8-v2 OLS and its draw-cluster bootstrap.

No project analysis modules are imported. Input traversal and random-number
consumption follow the released specification, including lexicographic cluster
ordering. Defaults read and write results/revision; --out-dir selects another destination.
"""

import os

for variable in (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import hashlib
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "revision"
DATA = RESULTS / "e8_rev2_per_draw.csv"
RELEASED = RESULTS / "stats_rev2_factorial.csv"
FIRST_SEED = 20260916
N_RESAMPLES = 2000
COEFFICIENTS = (
    "intercept_at_grid_centre", "log2N", "log2d", "log2N_x_log2d",
)
TASKS = {
    "imagery_fists_feet", "imagery_left_right_fist",
    "real_fists_feet", "real_left_right_fist",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_designs(data_path):
    # Keep the first occurrence of each cell, then draw order within the cell,
    # as in the released script. Never average delta values before fitting.
    paired = {}
    source_rows = read_csv(data_path)
    for row in source_rows:
        if row["model"] != "riemann_ts_lr":
            continue
        key = (row["task"], int(row["N"]), int(row["d"]), int(row["draw"]))
        values = paired.setdefault(key, {})
        assert row["protocol"] not in values, (key, row["protocol"])
        values[row["protocol"]] = float(row["mean_acc"])
    assert {key[0] for key in paired} == TASKS
    cells = defaultdict(list)
    for (task, n_subjects, dimension, draw), values in paired.items():
        assert set(values) == {"trial_random", "subject_disjoint"}
        cells[(task, n_subjects, dimension)].append(
            (draw, values["trial_random"] - values["subject_disjoint"])
        )
    n_levels = sorted({key[1] for key in cells})
    d_levels = sorted({key[2] for key in cells})
    assert n_levels == [9, 18, 36, 72, 109]
    assert d_levels == [8, 32, 128, 512, 2080]
    centre_n = np.mean(np.log2(np.array(n_levels)))
    centre_d = np.mean(np.log2(np.array(d_levels)))
    designs, design_info = {}, {}
    for task in sorted(TASKS):
        rows, response, labels = [], [], []
        for (cell_task, n_subjects, dimension), draws in cells.items():
            if cell_task != task:
                continue
            for draw, delta in draws:
                rows.append([1.0, np.log2(n_subjects), np.log2(dimension)])
                response.append(delta)
                labels.append(f"{n_subjects}_{draw}")
        raw = np.array(rows)
        xn, xd = raw[:, 1] - centre_n, raw[:, 2] - centre_d
        design = np.column_stack((raw[:, 0], xn, xd, xn * xd))
        response = np.array(response)
        assert np.linalg.matrix_rank(design) == 4
        expected_observations = 194 if task == "real_fists_feet" else 195
        assert len(response) == expected_observations
        clusters = np.array(labels)
        indices = {label: np.flatnonzero(clusters == label) for label in sorted(set(labels))}
        strata = defaultdict(list)
        for label in indices:
            strata[label.split("_")[0]].append(label)
        assert list(strata) == ["109", "18", "36", "72", "9"]
        assert [len(value) for value in strata.values()] == [1, 10, 10, 10, 10]
        beta = np.linalg.lstsq(design, response, rcond=None)[0]
        designs[task] = (design, response, beta, strata, indices)
        design_info[task] = {
            "n_observations": len(response), "rank": 4,
            "draw_clusters_by_N": {key: len(value) for key, value in strata.items()},
            "rows_per_cluster": {key: len(value) for key, value in indices.items()},
        }
    return designs, {
        "source_rows": len(source_rows), "paired_ts_lr_rows": len(paired),
        "N_levels": n_levels, "d_levels": d_levels,
        "log2N_grid_centre": float(centre_n), "log2d_grid_centre": float(centre_d),
        "tasks": design_info,
    }


def bootstrap(design, response, strata, indices, seed):
    rng = np.random.default_rng(seed)  # reset for each paradigm, as released
    coefficients = np.empty((N_RESAMPLES, len(COEFFICIENTS)))
    for repetition in range(N_RESAMPLES):
        sampled = []
        for labels in strata.values():
            selected = rng.choice(labels, size=len(labels), replace=True)
            sampled.extend(indices[label] for label in selected)
        index = np.concatenate(sampled)
        coefficients[repetition] = np.linalg.lstsq(design[index], response[index], rcond=None)[0]
    return np.percentile(coefficients, [2.5, 97.5], axis=0, method="linear")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, choices=(1, 50), default=50)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--released", type=Path, default=RELEASED)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    start = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.out_dir / ".e8_ols_seed_sensitivity"
    scratch.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "e8_rev2_ols_seed_sensitivity.csv"
    summary_path = args.out_dir / "e8_rev2_ols_seed_sensitivity_summary.csv"
    designs, design_info = prepare_designs(args.data)
    released = {
        (row["task"], row["d"]): row
        for row in read_csv(args.released) if row["N"] == "ols"
    }
    assert len(released) == 16
    detail, comparisons = [], []
    for seed in range(FIRST_SEED, FIRST_SEED + args.n_seeds):
        for task, (design, response, beta, strata, indices) in designs.items():
            limits = bootstrap(design, response, strata, indices, seed)
            for column, coefficient in enumerate(COEFFICIENTS):
                lo, hi = (float(value) for value in limits[:, column])
                detail.append({
                    "task": task, "coefficient": coefficient, "seed": seed,
                    "estimate": float(beta[column]), "ci_low": lo, "ci_high": hi,
                    "excludes_zero": int(lo > 0 or hi < 0),
                })
                if seed == FIRST_SEED:
                    ref = released[(task, coefficient)]
                    for field, ref_field, value in (
                        ("estimate", "inflation_mean", beta[column]),
                        ("ci_low", "ci_low", lo), ("ci_high", "ci_high", hi),
                    ):
                        comparisons.append({
                            "task": task, "coefficient": coefficient, "field": field,
                            "recomputed": float(value), "released": ref[ref_field],
                            "absolute_difference": abs(float(value) - float(ref[ref_field])),
                            "equal_at_released_precision": f"{value:.6g}" == ref[ref_field],
                        })
        print(f"seed={seed} complete ({len(detail)} coefficient rows); "
              f"elapsed={time.monotonic() - start:.1f}s", flush=True)
    assert len(detail) == 4 * 4 * args.n_seeds
    assert len({(row["task"], row["coefficient"], row["seed"]) for row in detail}) == len(detail)
    assert all(np.isfinite(row[key]) for row in detail for key in ("estimate", "ci_low", "ci_high"))
    assert all(row["ci_low"] <= row["ci_high"] for row in detail)
    summary = []
    for task in sorted(TASKS):
        for coefficient in COEFFICIENTS:
            subset = [row for row in detail if (row["task"], row["coefficient"]) == (task, coefficient)]
            # Match the six-significant-digit precision of the released table.
            # Detail rows retain round-trip floating-point precision.
            summary.append({
                "task": task, "coefficient": coefficient, "n_seeds": len(subset),
                "upper_min": f"{min(row['ci_high'] for row in subset):.6g}",
                "upper_max": f"{max(row['ci_high'] for row in subset):.6g}",
                "lower_min": f"{min(row['ci_low'] for row in subset):.6g}",
                "lower_max": f"{max(row['ci_low'] for row in subset):.6g}",
                "n_excluding_zero": sum(row["excludes_zero"] for row in subset),
            })
    max_difference = max(row["absolute_difference"] for row in comparisons)
    assert all(row["equal_at_released_precision"] for row in comparisons), comparisons
    assert max_difference <= 1e-6
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    report = {
        "origin": "bootstrap-seed sensitivity script, 2026-10-30",
        "generator": "e8_ols_seed_sensitivity.py",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "n_seeds": args.n_seeds,
        "first_seed": FIRST_SEED, "last_seed": FIRST_SEED + args.n_seeds - 1,
        "resamples_per_task_seed": N_RESAMPLES,
        "quantile_method": "linear (2.5th and 97.5th percentiles)",
        "detail_rows": len(detail), "summary_rows": len(summary),
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "os_thread_count": len(list(Path("/proc/self/task").iterdir())),
        "processes_used": 1,
        "thread_limits": {key: os.environ[key] for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
        )},
        "design": design_info,
        "max_absolute_difference": max_difference,
        "max_estimate_absolute_difference": max(row["absolute_difference"] for row in comparisons if row["field"] == "estimate"),
        "max_ci_absolute_difference": max(row["absolute_difference"] for row in comparisons if row["field"] != "estimate"),
        "all_48_values_equal_at_released_precision": all(row["equal_at_released_precision"] for row in comparisons),
        "comparisons": comparisons,
        "sha256": {str(path): sha256(path) for path in (args.data, args.released, Path(__file__).resolve(), detail_path, summary_path)},
    }
    (scratch / "analysis_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "detail_rows", "summary_rows", "max_absolute_difference",
        "max_estimate_absolute_difference", "max_ci_absolute_difference",
        "all_48_values_equal_at_released_precision", "os_thread_count", "elapsed_seconds",
    )}, indent=2), flush=True)
    for row in summary:
        if row["coefficient"] == "log2N":
            print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
