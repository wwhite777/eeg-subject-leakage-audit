#!/usr/bin/env python3
"""E8 within-dataset factorial: training-subject count x tangent-feature dimension (plan B11), EEGBCI.

For each paradigm, N in {9, 18, 36, 72, 109} subjects are drawn at random (10 draws for N < 109,
1 for N = 109; draw seeds 20260916 + i). Each draw is evaluated under P0 (trial-random 3-fold)
and P2 (subject-disjoint 3-fold) with TS+LR (C = 1) after PCA (fitted on the training fold) to
d in {8, 32, 128, 512, 2080 (no PCA)}, and with CSP+LDA (N only).
The per-draw quantity is the mean over the draw's subjects of per-subject accuracy.

Outputs: e8_<tag>_per_draw.csv (task, model, N, d, draw, protocol, mean_acc, n_subjects), e8_<tag>_manifest.json
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


def run_draw(data_sub, covs_sub, N, draw, d_grid, models):
    """One draw: both protocols, all d (TS+LR) and CSP+LDA. Returns rows."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    y, subject = data_sub["y"], data_sub["subject"]
    seed = C.SEED + draw
    rows = []
    for protocol in ("trial_random", "subject_disjoint"):
        splits = C.split_trial_random(y, 3, seed) if protocol == "trial_random" else C.split_subject_disjoint(subject, 3, seed)
        preds = {}
        for tr, te, _ in splits:
            if "riemann_ts_lr" in models:
                xtr, xte, _ = C.fold_tangent_features(covs_sub, tr, te)
                full_d = xtr.shape[1]
                max_pca = min(max(d for d in d_grid if d < full_d), len(tr) - 1)
                pca = PCA(n_components=max_pca, svd_solver="randomized", random_state=seed).fit(xtr)
                ztr, zte = pca.transform(xtr), pca.transform(xte)
                for d in d_grid:
                    if d >= full_d:
                        a, b, dd = xtr, xte, full_d
                    elif d > max_pca:
                        continue
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
            accs = C.per_subject_acc(y, pred, subject)
            rows.append({"model": model, "N": N, "d": dd, "draw": draw, "protocol": protocol, "mean_acc": float(np.mean(list(accs.values()))),
                         "n_subjects": len(accs), "n_trials": int(len(y))})
    return rows


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
    out_rows = []
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
                data_sub = {"y": y[mask], "subject": subject[mask], "x": data["x"][mask] if "csp_lda" in models else None}
                covs_sub = covs[mask] if covs is not None else None
                jobs.append((data_sub, covs_sub, N, draw))
        print(f"[{task}] {len(jobs)} draws", flush=True)
        res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(run_draw)(ds, cs, N, dr, D_GRID, models) for (ds, cs, N, dr) in jobs)
        for rows in res:
            for r in rows:
                out_rows.append({"task": task, **{k: (C.fmt(v) if isinstance(v, float) else v) for k, v in r.items()}})
        for N in N_GRID:
            for d in D_GRID:
                p0 = [float(r["mean_acc"]) for r in out_rows if r["task"] == task and r["model"] == "riemann_ts_lr" and r["N"] == N and r["d"] == d and r["protocol"] == "trial_random"]
                p2 = [float(r["mean_acc"]) for r in out_rows if r["task"] == task and r["model"] == "riemann_ts_lr" and r["N"] == N and r["d"] == d and r["protocol"] == "subject_disjoint"]
                if p0 and p2:
                    print(f"    N={N:<4d} d={d:<5d} P0={np.mean(p0):.4f} P2={np.mean(p2):.4f} infl={np.mean(p0)-np.mean(p2):+.4f} (draws={len(p0)})", flush=True)
    out = C.write_csv(C.RESULTS / f"e8_{tag}_per_draw.csv", out_rows)
    C.validate_csv(out, ["task", "model", "N", "d", "draw", "protocol", "mean_acc"], 1)
    C.write_manifest(C.RESULTS / f"e8_{tag}_manifest.json", f"e8_{tag}", started, [out], {"args": vars(args), "N_grid": N_GRID, "d_grid": D_GRID, "smoke": bool(args.max_subjects)})
    print(f"DONE e8 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
