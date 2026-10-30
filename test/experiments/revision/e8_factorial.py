#!/usr/bin/env python3
"""E8 within-dataset factorial: sampled-subject count x tangent-feature dimension (plans B11/B17), EEGBCI.

For each paradigm, N in {9, 18, 36, 72, 109} subjects are drawn at random (10 draws for N < 109,
1 for N = 109; draw seeds 20260916 + i). Each draw is evaluated under P0 (trial-random 3-fold)
and P2 (subject-disjoint 3-fold) with TS+LR (C = 1) after PCA (fitted on the training fold) to
d in {8, 32, 128, 512, 2080 (no PCA)}, and with CSP+LDA (N only).
The per-draw quantity is the mean over the draw's subjects of per-subject accuracy.

Both protocols' folds are constructed before scoring. A PCA dimension is scored in BOTH protocols
only if d <= min(n_train over all six folds) - 1; full dimension is always scored. The released
fold-specific randomized PCA construction (min(512, n_train - 1), same seed) is preserved, then
sliced for each feasible d. Every prediction vector must cover every trial before accuracy is computed.

Outputs: e8_<tag>_per_draw.csv (task, model, N, d, draw, protocol, mean_acc, n_subjects, n_trials),
e8_<tag>_feasibility.csv (task, N, draw, d, min_train_fold, feasible, min_train_fold_P0,
min_train_fold_P2), and e8_<tag>_manifest.json (including prediction coverage counts).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

N_GRID = [9, 18, 36, 72, 109]
D_GRID = [8, 32, 128, 512, 2080]


def assert_complete_predictions(pred, n_trials, *, task, model, N, d, draw, protocol):
    """Fail before scoring if any trial lacks an out-of-fold prediction; return coverage count."""
    covered = int(np.count_nonzero(pred != -1))
    assert pred.shape == (n_trials,) and covered == n_trials, (
        f"incomplete predictions: task={task}, model={model}, N={N}, d={d}, draw={draw}, "
        f"protocol={protocol}: covered {covered}/{n_trials} trials"
    )
    return covered


def run_draw(data_sub, covs_sub, N, draw, d_grid, models):
    """One draw: return accuracy rows, paired feasibility rows and coverage records."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    y, subject = data_sub["y"], data_sub["subject"]
    seed = C.SEED + draw
    splits_by_protocol = {
        "trial_random": C.split_trial_random(y, 3, seed),
        "subject_disjoint": C.split_subject_disjoint(subject, 3, seed),
    }
    min_train = {p: min(len(tr) for tr, _, _ in splits) for p, splits in splits_by_protocol.items()}
    min_train_fold = min(min_train.values())
    n_channels = covs_sub.shape[1] if covs_sub is not None else data_sub["x"].shape[1]
    full_d = n_channels * (n_channels + 1) // 2
    feasibility = [{"N": N, "draw": draw, "d": d, "min_train_fold": min_train_fold,
                    "feasible": int(d == full_d or (d < full_d and d <= min_train_fold - 1)),
                    "min_train_fold_P0": min_train["trial_random"], "min_train_fold_P2": min_train["subject_disjoint"]}
                   for d in d_grid]
    feasible_dims = [r["d"] for r in feasibility if r["feasible"]]
    rows, coverage = [], []
    for protocol, splits in splits_by_protocol.items():
        preds = {}
        for tr, te, _ in splits:
            if "riemann_ts_lr" in models:
                xtr, xte, _ = C.fold_tangent_features(covs_sub, tr, te)
                # B17: keep the released PCA fit independent of the draw's scoring mask.
                max_pca = min(512, len(tr) - 1)
                pca = PCA(n_components=max_pca, svd_solver="randomized", random_state=seed).fit(xtr)
                ztr, zte = pca.transform(xtr), pca.transform(xte)
                for d in feasible_dims:
                    if d == full_d:
                        a, b, dd = xtr, xte, full_d
                    else:
                        a, b, dd = ztr[:, :d], zte[:, :d], d
                    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(a, y[tr])
                    preds.setdefault(("riemann_ts_lr", dd), np.full(len(y), -1))[te] = clf.predict(b)
            if "csp_lda" in models:
                import mne
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                from sklearn.pipeline import Pipeline
                from mne.decoding import CSP

                mne.set_log_level("ERROR")
                m = Pipeline([("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)), ("lda", LinearDiscriminantAnalysis())])
                m.fit(data_sub["x"][tr], y[tr])
                preds.setdefault(("csp_lda", 6), np.full(len(y), -1))[te] = m.predict(data_sub["x"][te])
        for (model, dd), pred in preds.items():
            covered = assert_complete_predictions(pred, len(y), task=data_sub.get("task", "unknown"),
                                                  model=model, N=N, d=dd, draw=draw, protocol=protocol)
            coverage.append({"model": model, "N": N, "d": dd, "draw": draw, "protocol": protocol,
                             "n_predicted": covered, "n_trials": int(len(y))})
            accs = C.per_subject_acc(y, pred, subject)
            rows.append({"model": model, "N": N, "d": dd, "draw": draw, "protocol": protocol, "mean_acc": float(np.mean(list(accs.values()))),
                         "n_subjects": len(accs), "n_trials": int(len(y))})
    return rows, feasibility, coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--models", default="riemann_ts_lr,csp_lda")
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    models = [m for m in args.models.split(",") if m]
    out_rows, feasibility_rows, coverage_rows = [], [], []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        y, subject = data["y"], data["subject"]
        covs = C.covariances(data["x"]) if "riemann_ts_lr" in models else None
        all_subs = np.unique(subject)
        jobs = []
        for N in N_GRID:
            if N > len(all_subs):
                continue
            n_draws = 1 if N == len(all_subs) else args.n_draws
            for draw in range(n_draws):
                rng = np.random.default_rng(C.SEED + draw + 1000 * N)
                chosen = np.sort(rng.choice(all_subs, size=N, replace=False)) if N < len(all_subs) else all_subs
                mask = np.isin(subject, chosen)
                data_sub = {"task": task, "y": y[mask], "subject": subject[mask], "x": data["x"][mask] if "csp_lda" in models else None}
                covs_sub = covs[mask] if covs is not None else None
                jobs.append((data_sub, covs_sub, N, draw))
        print(f"[{task}] {len(jobs)} draws", flush=True)
        res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(run_draw)(ds, cs, N, dr, D_GRID, models) for (ds, cs, N, dr) in jobs)
        for rows, feasibility, coverage in res:
            for r in rows:
                out_rows.append({"task": task, **{k: (C.fmt(v) if isinstance(v, float) else v) for k, v in r.items()}})
            feasibility_rows.extend({"task": task, **r} for r in feasibility)
            coverage_rows.extend({"task": task, **r} for r in coverage)
        for N in N_GRID:
            for d in D_GRID:
                p0 = [float(r["mean_acc"]) for r in out_rows if r["task"] == task and r["model"] == "riemann_ts_lr" and r["N"] == N and r["d"] == d and r["protocol"] == "trial_random"]
                p2 = [float(r["mean_acc"]) for r in out_rows if r["task"] == task and r["model"] == "riemann_ts_lr" and r["N"] == N and r["d"] == d and r["protocol"] == "subject_disjoint"]
                if p0 and p2:
                    print(f"    N={N:<4d} d={d:<5d} P0={np.mean(p0):.4f} P2={np.mean(p2):.4f} protocol contrast={np.mean(p0)-np.mean(p2):+.4f} (draws={len(p0)})", flush=True)
    out = C.write_csv(C.RESULTS / f"e8_{tag}_per_draw.csv", out_rows)
    C.validate_csv(out, ["task", "model", "N", "d", "draw", "protocol", "mean_acc", "n_subjects", "n_trials"], 1)
    feasibility_out = C.write_csv(C.RESULTS / f"e8_{tag}_feasibility.csv", feasibility_rows)
    C.validate_csv(feasibility_out, ["task", "N", "draw", "d", "min_train_fold", "feasible", "min_train_fold_P0", "min_train_fold_P2"], 1)
    C.write_manifest(C.RESULTS / f"e8_{tag}_manifest.json", f"e8_{tag}", started, [out, feasibility_out],
                     {"args": vars(args), "N_grid": N_GRID, "d_grid": D_GRID, "smoke": bool(args.max_subjects), "prediction_coverage": coverage_rows})
    print(f"DONE e8 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
