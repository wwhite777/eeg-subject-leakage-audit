#!/usr/bin/env python3
"""Replicate the leakage findings on BCI-IV-2a (BNCI2014_001, left/right MI).

Reuses the EEGBCI protocol and reference-ablation code so the second dataset is
analysed identically:
  (1) method-specific leakage: pooled_random vs cross_subject for CSP+LDA,
      Riemannian TS+LR, and EEGNet (does only Riemannian inflate here too?);
  (2) reference channel: inductive vs transductive tangent-space reference under
      honest cross-subject CV (does the transductive reference inflate here too?).
CPU; reads the npz cache from bci2a_dataset_prepare.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eegbci_full"))
from eegbci_full_leakage_experiment import (  # noqa: E402
    build_models, bootstrap_ci, eval_cross_subject, eval_pooled_random, eval_within_subject,
)
from eegbci_reference_leakage_ablation import reference_ablation_from_arrays  # noqa: E402
# NB: braindecode/EEGNet deliberately omitted here -- moabb 1.5.0 breaks braindecode's
# import (old BNCI2014001 name). Deep rung already covered on EEGBCI; classical methods
# are sufficient to test whether the Riemannian-specific inflation replicates.

SEED = 20260709
CACHE = ROOT / "data_public/bci2a/cache/bci2a_leftright.npz"
OUT_DIR = ROOT / "results/bci2a"


def load():
    with np.load(CACHE, allow_pickle=True) as d:
        return d["x"].astype(np.float32), d["y"].astype(np.int64), d["subject"].astype(np.int64), d["run"].astype(np.int64)


def summarise(name, model, protocol, acc_by_subject, chance, n_epochs):
    accs = np.array(sorted(acc_by_subject.values()), dtype=float)
    lo, hi = bootstrap_ci(accs, SEED)
    return {
        "dataset": "bci2a", "model": model, "protocol": protocol, "n_subjects": len(acc_by_subject),
        "n_epochs": int(n_epochs), "chance": f"{chance:.6g}", "mean_acc": f"{np.mean(accs):.6g}",
        "std_acc": f"{np.std(accs):.6g}", "ci_low": f"{lo:.6g}", "ci_high": f"{hi:.6g}",
    }


def main() -> int:
    x, y, groups, runs = load()
    chance = float(max(np.mean(y == 0), np.mean(y == 1)))
    print(f"bci2a: trials={len(y)} chans={x.shape[1]} times={x.shape[2]} subjects={len(np.unique(groups))} chance={chance:.3f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # (1a) classical, three protocols
    for model_name in ("csp_lda", "riemann_ts_lr"):
        def factory(mn=model_name):
            return build_models(SEED)[mn]
        for protocol, fn in (("pooled_random", eval_pooled_random), ("within_subject", None), ("cross_subject", eval_cross_subject)):
            if protocol == "within_subject":
                acc = eval_within_subject(factory, x, y, groups, runs, SEED)
            else:
                acc = fn(factory, x, y, groups, SEED)
            rows.append(summarise("bci2a", model_name, protocol, acc, chance, len(y)))
            print(f"    {model_name:14s} {protocol:14s} {rows[-1]['mean_acc']}", flush=True)

    with (OUT_DIR / "BrainFuse_bci2a_protocols_20260709.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # (2) reference-leakage ablation
    acc_ind, acc_trans = reference_ablation_from_arrays(x, y, groups, SEED)
    subs = sorted(acc_ind)
    ind = np.array([acc_ind[s] for s in subs]); trans = np.array([acc_trans[s] for s in subs]); delta = trans - ind
    lo_d, hi_d = bootstrap_ci(delta, SEED)
    abl = [{
        "dataset": "bci2a", "n_subjects": len(subs),
        "acc_inductive": f"{ind.mean():.6g}", "acc_transductive": f"{trans.mean():.6g}",
        "reference_leakage": f"{delta.mean():.6g}", "leak_ci_low": f"{lo_d:.6g}", "leak_ci_high": f"{hi_d:.6g}",
    }]
    with (OUT_DIR / "BrainFuse_bci2a_reference_leakage_20260709.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(abl[0].keys())); w.writeheader(); w.writerows(abl)
    print(f"    reference ablation: inductive={ind.mean():.4f} transductive={trans.mean():.4f} "
          f"reference_leakage={delta.mean():+.4f} (95%CI {lo_d:+.4f},{hi_d:+.4f})", flush=True)
    print("DONE bci2a replication", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
