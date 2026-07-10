#!/usr/bin/env python3
"""Paired significance tests for the leakage inflation and calibration gap.

Subject-averaged (over the four EEGBCI paradigms) per-subject accuracy is compared
across protocols with a two-sided paired Wilcoxon signed-rank test (n=109 subjects).
Reads the per-subject CSV written by eegbci_full_leakage_experiment.py.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def main() -> int:
    src = Path("results/eegbci_full/BrainFuse_eegbci_full_leakage_per_subject_20260708.csv")
    rows = list(csv.DictReader(src.open()))
    tmp = defaultdict(list)
    for r in rows:
        tmp[(r["model"], r["protocol"], r["subject"])].append(float(r["accuracy"]))
    subjacc = defaultdict(dict)
    for (m, p, s), v in tmp.items():
        subjacc.setdefault(m, {}).setdefault(s, {})[p] = float(np.mean(v))

    out_rows = []
    for m in ("csp_lda", "riemann_ts_lr"):
        subs = sorted(subjacc[m])
        for a, b, name in (("pooled_random", "cross_subject", "leakage_inflation"),
                           ("within_subject", "cross_subject", "calibration_gap")):
            xa = np.array([subjacc[m][s][a] for s in subs])
            xb = np.array([subjacc[m][s][b] for s in subs])
            stat, p = wilcoxon(xa, xb)
            out_rows.append({"model": m, "contrast": name, "mean_a": f"{xa.mean():.4f}",
                             "mean_b": f"{xb.mean():.4f}", "mean_diff": f"{xa.mean()-xb.mean():+.4f}",
                             "wilcoxon_W": f"{stat:.1f}", "p_value": f"{p:.3e}", "n": len(subs)})
            print(f"{m:14s} {name:18s} Δ={xa.mean()-xb.mean():+.4f}  W={stat:.1f}  p={p:.3e}  (n={len(subs)})")

    out = Path("results/eegbci_full/BrainFuse_eegbci_paired_stats_20260710.csv")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
