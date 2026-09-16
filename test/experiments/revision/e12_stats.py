#!/usr/bin/env python3
"""Statistics for the calibration-size curve (E12, plan B14): per model and k, mean accuracy over subjects (draw-averaged,
paradigm mean over the paradigms present) with subject-bootstrap CI; paired difference k vs k = 0 with Wilcoxon,
sign-flip and Holm (family = one model); reference values: within-subject and subject-disjoint (P2) accuracy of the
same decoder on the same paradigms.
Usage: e12_stats.py --tag rev1 --e12 <csv,csv> --e1 <e1 eegbci per_subject.csv> --e11 <csv,csv> --e9 <csv,csv>
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import fig_common as F  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    ap.add_argument("--e12", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--e9", required=True)
    args = ap.parse_args()
    started = C.utc_now()
    rows = []
    for p in args.e12.split(","):
        rows += C.read_csv(Path(p))
    tasks = sorted({r["task"] for r in rows}, key=C.TASKS.index)
    subj = F.seed_avg_subject(rows, ("task", "model", "k"))
    pm = F.paradigm_mean(subj, 0, tasks=tasks)
    # references on the same paradigms
    ref_rows = [r for r in C.read_csv(Path(args.e1)) if r["protocol"] in ("within_subject", "subject_disjoint") and r["task"] in tasks]
    for p in args.e11.split(",") + args.e9.split(","):
        ref_rows += [r for r in C.read_csv(Path(p)) if r["task"] in tasks and r.get("protocol") in ("within_subject", "subject_disjoint") and r.get("width", "default") in ("default", "") and r.get("preproc", "ztrial") != "global"]
    ref = F.paradigm_mean(F.seed_avg_subject(ref_rows, ("task", "model", "protocol")), 0, tasks=tasks)
    out = []
    for model in sorted({k[1] for k in pm}):
        ks = sorted({int(k[2]) for k in pm if k[1] == model})
        base = pm[("paradigm_mean", model, "0")]
        fam = []
        for k in ks:
            d = pm[("paradigm_mean", model, str(k))]
            v = np.array(list(d.values()))
            lo, hi = C.bootstrap_ci(v)
            subs = sorted(set(d) & set(base))
            diff = np.array([d[s] - base[s] for s in subs])
            dlo, dhi = C.bootstrap_ci(diff)
            W, p = C.wilcoxon_paired(np.array([d[s] for s in subs]), np.array([base[s] for s in subs])) if k else (0.0, 1.0)
            fam.append(({"model": model, "k": k, "n": len(v), "mean_acc": C.fmt(v.mean()), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi),
                         "diff_vs_k0": C.fmt(diff.mean()), "diff_ci_low": C.fmt(dlo), "diff_ci_high": C.fmt(dhi), "p_wilcoxon": C.fmt(p),
                         "p_signflip": C.fmt(C.signflip_perm_p(diff)) if k else "1"}, p))
        for (row, _), ph in zip(fam, C.holm([p for _, p in fam[1:]]) and ([1.0] + C.holm([p for _, p in fam[1:]]))):
            row["p_holm"] = C.fmt(ph)
            out.append(row)
        for proto in ("within_subject", "subject_disjoint"):
            d = ref.get(("paradigm_mean", model, proto))
            if d:
                v = np.array(list(d.values())); lo, hi = C.bootstrap_ci(v)
                out.append({"model": model, "k": f"ref_{proto}", "n": len(v), "mean_acc": C.fmt(v.mean()), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi),
                            "diff_vs_k0": "", "diff_ci_low": "", "diff_ci_high": "", "p_wilcoxon": "", "p_signflip": "", "p_holm": ""})
        for row in out:
            if row["model"] == model:
                print(f"  {model:14s} k={row['k']!s:22s} acc={float(row['mean_acc']):.4f} [{float(row['ci_low']):.4f},{float(row['ci_high']):.4f}]" + (f"  Δ vs k0 {float(row['diff_vs_k0']):+.4f} [{float(row['diff_ci_low']):+.4f},{float(row['diff_ci_high']):+.4f}] holm={float(row['p_holm']):.2e}" if row["diff_vs_k0"] else ""))
    path = C.write_csv(C.RESULTS / f"stats_{args.tag}_e12_calibration.csv", out)
    C.write_manifest(C.RESULTS / f"stats_{args.tag}_e12_manifest.json", f"stats_{args.tag}_e12", started, [path], {"args": vars(args), "paradigms": tasks})
    print("DONE e12 stats", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
