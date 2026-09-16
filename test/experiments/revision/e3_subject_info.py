#!/usr/bin/env python3
"""E3 subject-information analyses on EEGBCI tangent features (plan B6).

E3a subject-ID decoding: multinomial LR predicting the subject (109 classes)
    from inductive tangent features under trial-random 3-fold and run-disjoint
    3-fold splits; also after per-subject mean removal (second-order identity
    information). Chance = 1/n_subjects.
E3b variance decomposition (common tangent space, reference = Frechet mean of
    all trials; used only for this descriptive analysis): sums of squares for
    subject, class, subject x class (cell means) and residual, summed over
    tangent dimensions.
E3c subject-mean removal and Riemannian recentering, then the ladder protocols
    (P0 3-fold x seeds, P1, P2 3-fold x seeds) with TS+LR (C = 1).
E3d alignment of LR weight vectors trained under P0 and P2 (common tangent
    space) with subject-specific class-separation directions d_s and with the
    global class direction d_global; fraction of ||w||^2 inside span{d_s}.

Outputs: e3a_<tag>.csv, e3b_<tag>.csv, e3c_<tag>_per_subject.csv, e3d_<tag>.csv, e3_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def center_by_subject(feats: np.ndarray, subject: np.ndarray) -> np.ndarray:
    out = feats.copy()
    for s in np.unique(subject):
        m = subject == s
        out[m] -= feats[m].mean(axis=0, keepdims=True)
    return out


# ---------------------------------------------------------------- E3a
def e3a_fold(covs, subject, tr, te, seed, split_name, fold, centered):
    from sklearn.linear_model import LogisticRegression

    xtr, xte, _ = C.fold_tangent_features(covs, tr, te)
    if centered:
        # label-free per-subject centering using each subject's mean over ALL its trials
        # (computed in this fold's tangent space); stated as transductive-unsupervised.
        allx = np.empty((len(subject), xtr.shape[1]))
        allx[tr] = xtr
        allx[te] = xte
        allx = center_by_subject(allx, subject)
        xtr, xte = allx[tr], allx[te]
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(xtr, subject[tr])
    pred = clf.predict(xte)
    return {"split": split_name, "fold": fold, "centered": centered, "te": te, "pred": pred, "train_acc": float(np.mean(clf.predict(xtr) == subject[tr]))}


# ---------------------------------------------------------------- E3b
def variance_decomposition(feats: np.ndarray, subject: np.ndarray, y: np.ndarray) -> dict:
    grand = feats.mean(axis=0)
    ss_total = float(np.sum((feats - grand) ** 2))
    ss_subject = 0.0
    for s in np.unique(subject):
        m = subject == s
        ss_subject += m.sum() * float(np.sum((feats[m].mean(axis=0) - grand) ** 2))
    ss_class = 0.0
    for c in np.unique(y):
        m = y == c
        ss_class += m.sum() * float(np.sum((feats[m].mean(axis=0) - grand) ** 2))
    ss_cells = 0.0
    for s in np.unique(subject):
        for c in np.unique(y):
            m = (subject == s) & (y == c)
            if m.sum() == 0:
                continue
            ss_cells += m.sum() * float(np.sum((feats[m].mean(axis=0) - grand) ** 2))
    ss_inter = ss_cells - ss_subject - ss_class
    ss_resid = ss_total - ss_cells
    return {
        "ss_total": ss_total, "frac_subject": ss_subject / ss_total, "frac_class": ss_class / ss_total,
        "frac_subject_x_class": ss_inter / ss_total, "frac_residual": ss_resid / ss_total,
        "n_subjects": int(len(np.unique(subject))), "n_trials": int(len(y)), "n_dims": int(feats.shape[1]),
    }


# ---------------------------------------------------------------- E3c
def e3c_fold(covs_variant, y, subject, tr, te, seed, protocol, fold, variant, center_feats):
    from sklearn.linear_model import LogisticRegression

    xtr, xte, _ = C.fold_tangent_features(covs_variant, tr, te)
    if center_feats:
        allx = np.empty((len(y), xtr.shape[1]))
        allx[tr] = xtr
        allx[te] = xte
        allx = center_by_subject(allx, subject)
        xtr, xte = allx[tr], allx[te]
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(xtr, y[tr])
    return {"variant": variant, "protocol": protocol, "seed": seed, "fold": fold, "te": te, "pred": clf.predict(xte), "train_acc": float(np.mean(clf.predict(xtr) == y[tr]))}


# ---------------------------------------------------------------- E3d
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def e3d_task(feats, y, subject, seeds):
    from sklearn.linear_model import LogisticRegression

    subs = np.unique(subject)
    D = {}
    for s in subs:
        m = subject == s
        D[int(s)] = unit(feats[m & (y == 1)].mean(axis=0) - feats[m & (y == 0)].mean(axis=0))
    Dm = np.stack([D[int(s)] for s in subs])  # (n_subj, dims)
    cos = np.abs(Dm @ Dm.T)
    off = cos[~np.eye(len(subs), dtype=bool)]
    d_global = unit(feats[y == 1].mean(axis=0) - feats[y == 0].mean(axis=0))
    Q, _ = np.linalg.qr(Dm.T)  # orthonormal basis of span{d_s}
    rows = []
    rows.append({"protocol": "descriptive", "seed": "", "fold": "", "metric": "mean_pairwise_abs_cos_d_s", "value": float(off.mean()),
                 "n": int(len(off)), "note": "mean |cos| between subject-specific class-separation directions (off-diagonal pairs)"})
    rows.append({"protocol": "descriptive", "seed": "", "fold": "", "metric": "mean_abs_cos_d_s_d_global", "value": float(np.mean(np.abs(Dm @ d_global))),
                 "n": int(len(subs)), "note": "mean |cos| between each subject's direction and the global class direction"})
    rows.append({"protocol": "descriptive", "seed": "", "fold": "", "metric": "random_direction_fraction_in_span",
                 "value": float(len(subs) / feats.shape[1]), "n": int(len(subs)), "note": "expected ||proj||^2/||w||^2 of a random direction (k/D)"})
    for protocol in ("trial_random", "subject_disjoint"):
        for seed in seeds:
            splits = C.split_trial_random(y, 3, seed) if protocol == "trial_random" else C.split_subject_disjoint(subject, 3, seed)
            for tr, te, fold in splits:
                clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(feats[tr], y[tr])
                w = clf.coef_.ravel()
                wu = unit(w)
                tr_subs = np.unique(subject[tr])
                te_subs = np.unique(subject[te])
                cos_tr = float(np.mean([abs(wu @ D[int(s)]) for s in tr_subs]))
                cos_te = float(np.mean([abs(wu @ D[int(s)]) for s in te_subs]))
                proj = Q.T @ w
                frac = float(np.sum(proj ** 2) / np.sum(w ** 2))
                # span of TRAINING subjects' directions only
                Qtr, _ = np.linalg.qr(np.stack([D[int(s)] for s in tr_subs]).T)
                frac_tr = float(np.sum((Qtr.T @ w) ** 2) / np.sum(w ** 2))
                base = {"protocol": protocol, "seed": seed, "fold": fold, "n": int(len(tr_subs))}
                rows.append({**base, "metric": "mean_abs_cos_w_d_s_train_subjects", "value": cos_tr, "note": ""})
                rows.append({**base, "metric": "mean_abs_cos_w_d_s_test_subjects", "value": cos_te, "note": ""})
                rows.append({**base, "metric": "abs_cos_w_d_global", "value": float(abs(wu @ d_global)), "note": ""})
                rows.append({**base, "metric": "fraction_w_in_span_all_d_s", "value": frac, "note": ""})
                rows.append({**base, "metric": "fraction_w_in_span_train_d_s", "value": frac_tr, "note": ""})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--parts", default="a,b,c,d")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    started = C.utc_now()
    tag = args.tag or C.stamp()
    seeds = [C.SEED + r for r in range(args.n_seeds)]
    parts = set(args.parts.split(","))
    a_rows, b_rows, c_rows, d_rows = [], [], [], []
    for task in [t for t in args.tasks.split(",") if t]:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[: args.max_subjects]
        data = C.load_eegbci_task(task, subs)
        y, subject = data["y"], data["subject"]
        covs = C.covariances(data["x"])
        n_subj = len(np.unique(subject))
        print(f"[{task}] trials={len(y)} subjects={n_subj}", flush=True)

        if "a" in parts:
            jobs = []
            for split_name, splits in (("trial_random", C.split_trial_random(y, 3, C.SEED)), ("run_disjoint", C.split_run_disjoint(data["run_idx"]))):
                for tr, te, fold in splits:
                    for centered in (False, True):
                        jobs.append((tr, te, split_name, fold, centered))
            res = Parallel(n_jobs=min(args.n_jobs, len(jobs)), backend="loky")(delayed(e3a_fold)(covs, subject, tr, te, C.SEED, sn, f, c) for (tr, te, sn, f, c) in jobs)
            store, mask = {}, {}
            for r in res:
                key = (r["split"], r["centered"])
                store.setdefault(key, np.full(len(y), -1))[r["te"]] = r["pred"]
                mask.setdefault(key, np.zeros(len(y), dtype=bool))[r["te"]] = True
            for (sn, cen), pred in sorted(store.items()):
                m = mask[(sn, cen)]
                acc = float(np.mean(pred[m] == subject[m]))
                per_s = C.per_subject_acc(subject, pred, subject, m)
                a_rows.append({"task": task, "split": sn, "centered": cen, "n_subjects": n_subj, "chance": C.fmt(1 / n_subj),
                               "subject_id_accuracy": C.fmt(acc), "per_subject_median": C.fmt(np.median(list(per_s.values()))),
                               "per_subject_min": C.fmt(min(per_s.values())), "per_subject_max": C.fmt(max(per_s.values()))})
                print(f"    E3a {sn:14s} centered={cen!s:5s} subject-ID acc={acc:.4f} (chance {1/n_subj:.4f})", flush=True)

        if "b" in parts or "d" in parts:
            ref_all = C.riemann_mean(covs)
            feats_all = C.tangent(covs, ref_all)
        if "b" in parts:
            vd = variance_decomposition(feats_all, subject, y)
            vd_c = variance_decomposition(center_by_subject(feats_all, subject), subject, y)
            b_rows.append({"task": task, "space": "common_tangent", **{k: C.fmt(v) for k, v in vd.items()}})
            b_rows.append({"task": task, "space": "common_tangent_subject_centered", **{k: C.fmt(v) for k, v in vd_c.items()}})
            print(f"    E3b subject={vd['frac_subject']:.3f} class={vd['frac_class']:.3f} sxc={vd['frac_subject_x_class']:.3f} resid={vd['frac_residual']:.3f}", flush=True)

        if "c" in parts:
            covs_rc = C.recenter_by_subject(covs, subject)
            jobs = []
            for protocol in ("trial_random", "run_disjoint", "subject_disjoint"):
                for seed in (seeds if protocol != "run_disjoint" else [C.SEED]):
                    for tr, te, fold in C.ladder_splits(data, protocol, seed, 3):
                        jobs.append(("feature_centered", covs, True, tr, te, seed, protocol, fold))
                        jobs.append(("riemann_recentered", covs_rc, False, tr, te, seed, protocol, fold))
            res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(e3c_fold)(cv, y, subject, tr, te, seed, p, f, v, cf) for (v, cv, cf, tr, te, seed, p, f) in jobs)
            store, mask = {}, {}
            for r in res:
                key = (r["variant"], r["protocol"], r["seed"])
                store.setdefault(key, np.full(len(y), -1))[r["te"]] = r["pred"]
                mask.setdefault(key, np.zeros(len(y), dtype=bool))[r["te"]] = True
            for (variant, protocol, seed), pred in sorted(store.items()):
                accs = C.per_subject_acc(y, pred, subject, mask[(variant, protocol, seed)])
                for s, a in sorted(accs.items()):
                    c_rows.append({"task": task, "variant": variant, "protocol": protocol, "seed": seed, "subject": s, "accuracy": C.fmt(a)})
            for variant in ("feature_centered", "riemann_recentered"):
                m = {p: np.mean([float(r["accuracy"]) for r in c_rows if r["task"] == task and r["variant"] == variant and r["protocol"] == p]) for p in ("trial_random", "run_disjoint", "subject_disjoint")}
                print(f"    E3c {variant:18s} P0={m['trial_random']:.4f} P1={m['run_disjoint']:.4f} P2={m['subject_disjoint']:.4f} d12={m['run_disjoint']-m['subject_disjoint']:+.4f}", flush=True)

        if "d" in parts:
            rows = e3d_task(feats_all, y, subject, seeds)
            for r in rows:
                d_rows.append({"task": task, **r, "value": C.fmt(r["value"])})
            for metric in ("mean_abs_cos_w_d_s_train_subjects", "fraction_w_in_span_train_d_s", "abs_cos_w_d_global"):
                for p in ("trial_random", "subject_disjoint"):
                    vals = [float(r["value"]) for r in rows if r["metric"] == metric and r["protocol"] == p]
                    print(f"    E3d {metric:36s} {p:16s} mean={np.mean(vals):.4f}", flush=True)

    outputs = []
    if a_rows:
        outputs.append(C.write_csv(C.RESULTS / f"e3a_{tag}.csv", a_rows))
    if b_rows:
        outputs.append(C.write_csv(C.RESULTS / f"e3b_{tag}.csv", b_rows))
    if c_rows:
        outputs.append(C.write_csv(C.RESULTS / f"e3c_{tag}_per_subject.csv", c_rows))
    if d_rows:
        outputs.append(C.write_csv(C.RESULTS / f"e3d_{tag}.csv", d_rows))
    C.write_manifest(C.RESULTS / f"e3_{tag}_manifest.json", f"e3_{tag}", started, outputs, {"args": vars(args), "smoke": bool(args.max_subjects)})
    print(f"DONE e3 tag={tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
