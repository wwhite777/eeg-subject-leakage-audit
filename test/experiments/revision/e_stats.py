#!/usr/bin/env python3
"""Statistics and aggregation for the revision analyses (plan B2, C1–C8).

Reads the per-subject CSVs written by e1..e12 (by tag) and writes:
  stats_<tag>_ladder_means.csv     mean accuracy per (dataset, task/paradigm-average, model, protocol) with
                                   subject-bootstrap 95% CI and across-seed SD (seed-averaged per subject first)
  stats_<tag>_contrasts.csv        paired contrasts Δ(Pa − Pb): mean, CI, Wilcoxon (exact for n ≤ 25), sign-flip
                                   permutation p, Holm-adjusted p within each family
  stats_<tag>_tost.csv             equivalence tests for the reference analyses (E6 and the LOSO transductive test)
  stats_<tag>_sweep.csv            E2/E4 curves: per C or d, protocol means and Δ with CIs; Spearman ρ over d
  stats_<tag>_factorial.csv        E8 cell means, draw-level CIs and the OLS on log2 N, log2 d
  stats_<tag>_width.csv            E7 per-width Δ(P0−P2), params, train–test gap, Spearman over width
Every function is written from the definitions in the plan, independent of the experiment scripts.
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


def seed_averaged(rows, keys=("dataset", "task", "model", "protocol"), extra_keys=()):
    """{(keys..., subject): mean accuracy over seeds}, plus per-seed store."""
    per_seed = defaultdict(list)
    for r in rows:
        k = tuple(r.get(x, "") for x in keys) + tuple(r.get(x, "") for x in extra_keys) + (int(r["subject"]),)
        per_seed[k].append(float(r["accuracy"]))
    return {k: float(np.mean(v)) for k, v in per_seed.items()}, per_seed


def paradigm_average(subj_acc: dict, keys_len: int, tasks: list[str]):
    """Average each subject's accuracy over the EEGBCI paradigms -> {(dataset,'paradigm_mean',model,protocol,subject): acc}."""
    agg = defaultdict(list)
    for k, v in subj_acc.items():
        dataset, task, model, protocol = k[0], k[1], k[2], k[3]
        rest = k[4:]
        if task in tasks:
            agg[(dataset, "paradigm_mean", model, protocol) + rest].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items() if len(v) == len(tasks)}


def contrast(subj_a: dict[int, float], subj_b: dict[int, float], seed=C.SEED) -> dict:
    subs = sorted(set(subj_a) & set(subj_b))
    a = np.array([subj_a[s] for s in subs])
    b = np.array([subj_b[s] for s in subs])
    d = a - b
    lo, hi = C.bootstrap_ci(d, seed=seed)
    W, p = C.wilcoxon_paired(a, b)
    return {"n": len(subs), "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
            "wilcoxon_W": W, "p_wilcoxon": p, "p_signflip": C.signflip_perm_p(d, seed=seed), "n_positive": int(np.sum(d > 0)), "n_negative": int(np.sum(d < 0))}


def by_group(subj_acc: dict, prefix: tuple) -> dict[int, float]:
    n = len(prefix)
    return {k[n]: v for k, v in subj_acc.items() if k[:n] == prefix}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="tag shared by the experiment outputs to aggregate")
    ap.add_argument("--e1-eegbci", default=None)
    ap.add_argument("--e1-bci2a", default=None)
    ap.add_argument("--e9", default=None)
    ap.add_argument("--e10", default=None)
    ap.add_argument("--e11", default=None)
    ap.add_argument("--e2", default=None)
    ap.add_argument("--e4", default=None)
    ap.add_argument("--e5", default=None)
    ap.add_argument("--e3c", default=None)
    ap.add_argument("--e6", default=None)
    ap.add_argument("--e14", default=None)
    ap.add_argument("--e7", default=None)
    ap.add_argument("--e8", default=None)
    ap.add_argument("--e13-classical", default=None)
    ap.add_argument("--e13-deep", default=None)
    args = ap.parse_args()
    started = C.utc_now()
    R = C.RESULTS
    outputs = []

    # ------------------------------------------------------------------ ladder means + contrasts
    rows = []
    for path in (args.e1_eegbci, args.e9, args.e11, args.e13_classical, args.e13_deep):
        if path:
            for r in C.read_csv(Path(path)):
                r.setdefault("dataset", "eegbci")
                if "preproc" in r and r["preproc"] == "global":
                    r["model"] = r["model"] + "_globalscale"
                if "width" in r and r["width"] not in ("", "default"):
                    continue  # width sweeps handled separately
                if path == args.e13_classical:
                    r["model"] = r["model"] + "_ztrial"
                rows.append(r)
    for path in (args.e1_bci2a, args.e10):
        if path:
            for r in C.read_csv(Path(path)):
                r.setdefault("dataset", "bci2a")
                rows.append(r)
    means_rows, contrast_rows = [], []
    if rows:
        subj_acc, per_seed = seed_averaged(rows)
        pm = paradigm_average(subj_acc, 4, C.TASKS)
        subj_acc_all = dict(subj_acc)
        subj_acc_all.update(pm)
        groups = sorted({k[:4] for k in subj_acc_all})
        for g in groups:
            accs = by_group(subj_acc_all, g)
            v = np.array(list(accs.values()))
            lo, hi = C.bootstrap_ci(v)
            # across-seed SD of the mean (per-seed means)
            seed_means = defaultdict(list)
            for k, vals in per_seed.items():
                if k[:4] == g:
                    for i, x in enumerate(vals):
                        seed_means[i].append(x)
            sm = [np.mean(x) for x in seed_means.values()] if g[1] != "paradigm_mean" else []
            means_rows.append({"dataset": g[0], "task": g[1], "model": g[2], "protocol": g[3], "n_subjects": len(v), "n_seeds": len(sm) if sm else "",
                               "mean_acc": C.fmt(v.mean()), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi), "sd_subjects": C.fmt(v.std(ddof=1)) if len(v) > 1 else "",
                               "sd_across_seeds": C.fmt(np.std(sm, ddof=1)) if len(sm) > 1 else ""})
        pairs_eegbci = [("trial_random", "run_disjoint", "run_temporal_component"), ("run_disjoint", "subject_disjoint", "subject_overlap_component"),
                        ("trial_random", "subject_disjoint", "delta_leak_original_definition"), ("within_subject", "subject_disjoint", "calibration_gap")]
        pairs_bci2a = [("trial_random_6", "run_disjoint", "run_temporal_component"), ("run_disjoint", "loso", "subject_overlap_component_run"),
                       ("trial_random_2", "session_disjoint", "run_session_component"), ("session_disjoint", "loso", "subject_overlap_component_session"),
                       ("trial_random_6", "loso", "delta_leak_original_definition"), ("within_session", "loso", "calibration_gap"),
                       ("subject_disjoint_5", "loso", "groupkfold5_vs_loso")]
        fam = defaultdict(list)
        for g in groups:
            dataset, task, model, protocol = g
            pairs = pairs_eegbci if dataset == "eegbci" else pairs_bci2a
            for pa, pb, name in pairs:
                if protocol != pa:
                    continue
                A = by_group(subj_acc_all, (dataset, task, model, pa))
                B = by_group(subj_acc_all, (dataset, task, model, pb))
                if not A or not B:
                    continue
                c = contrast(A, B)
                row = {"dataset": dataset, "task": task, "model": model, "contrast": name, "protocol_a": pa, "protocol_b": pb, **{k: (C.fmt(v) if isinstance(v, float) else v) for k, v in c.items()}}
                fam[(dataset, task)].append((row, c["p_wilcoxon"]))
        for (dataset, task), items in fam.items():
            adj = C.holm([p for _, p in items])
            for (row, _), pa in zip(items, adj):
                row["p_holm"] = C.fmt(pa)
                row["family"] = f"{dataset}/{task}"
                contrast_rows.append(row)
        outputs.append(C.write_csv(R / f"stats_{args.tag}_ladder_means.csv", means_rows))
        outputs.append(C.write_csv(R / f"stats_{args.tag}_contrasts.csv", contrast_rows))
        for r in contrast_rows:
            if r["task"] in ("paradigm_mean", "bci2a_left_right"):
                print(f"  {r['dataset']:6s} {r['model']:26s} {r['contrast']:34s} Δ={float(r['mean_diff']):+.4f} CI[{float(r['ci_low']):+.4f},{float(r['ci_high']):+.4f}] p={float(r['p_wilcoxon']):.2e} holm={float(r['p_holm']):.2e} n={r['n']}")

    # ------------------------------------------------------------------ E3c and E5 contrasts (same machinery, extra key)
    for path, key, label in ((args.e3c, "variant", "e3c"), (args.e5, "clf", "e5")):
        if not path:
            continue
        rr = C.read_csv(Path(path))
        for r in rr:
            r["dataset"] = "eegbci"
            r["model"] = r[key]
        subj_acc, _ = seed_averaged(rr)
        pm = paradigm_average(subj_acc, 4, C.TASKS)
        subj_acc.update(pm)
        out = []
        fam = defaultdict(list)
        for g in sorted({k[:4] for k in subj_acc}):
            dataset, task, model, protocol = g
            for pa, pb, name in (("run_disjoint", "subject_disjoint", "subject_overlap_component"), ("trial_random", "subject_disjoint", "delta_leak_original_definition"), ("trial_random", "run_disjoint", "run_temporal_component")):
                if protocol != pa:
                    continue
                A, B = by_group(subj_acc, (dataset, task, model, pa)), by_group(subj_acc, (dataset, task, model, pb))
                if A and B:
                    c = contrast(A, B)
                    fam[(task,)].append(({"task": task, key: model, "contrast": name, **{k: (C.fmt(v) if isinstance(v, float) else v) for k, v in c.items()}}, c["p_wilcoxon"]))
        for (task,), items in fam.items():
            for (row, _), pa in zip(items, C.holm([p for _, p in items])):
                row["p_holm"] = C.fmt(pa)
                out.append(row)
        outputs.append(C.write_csv(R / f"stats_{args.tag}_{label}_contrasts.csv", out))
        for r in out:
            if r["task"] == "paradigm_mean" and r["contrast"] == "subject_overlap_component":
                print(f"  {label} {r[key]:20s} Δ(P1−P2)={float(r['mean_diff']):+.4f} CI[{float(r['ci_low']):+.4f},{float(r['ci_high']):+.4f}] holm={float(r['p_holm']):.2e}")

    # ------------------------------------------------------------------ E6 / E14 + TOST
    out = []
    for path, label in ((args.e6, "e6_target_subject_in_reference"), (args.e14, "e14_transductive_reference")):
        if not path:
            continue
        rr = C.read_csv(Path(path))
        # unit = subject: when a subject appears in several folds (E6), average its delta over folds first
        per_cell = defaultdict(list)
        for r in rr:
            per_cell[(r.get("dataset", "eegbci"), r["task"], int(r["subject"]))].append(float(r["delta"]))
        per_task = defaultdict(list)
        for (dataset, task, subject), v in per_cell.items():
            per_task[(dataset, task)].append(float(np.mean(v)))
        allv = []
        for (dataset, task), d in per_task.items():
            if dataset == "eegbci":
                allv.extend(d)
            t = C.tost(np.array(d))
            lo, hi = C.bootstrap_ci(np.array(d))
            W, p = C.wilcoxon_paired(np.array(d), np.zeros(len(d)))
            out.append({"analysis": label, "dataset": dataset, "task": task, "n": len(d), "mean_delta": C.fmt(np.mean(d)), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi),
                        "p_wilcoxon": C.fmt(p), **{f"tost_{k}": C.fmt(v) for k, v in t.items()}})
        if allv:
            t = C.tost(np.array(allv))
            lo, hi = C.bootstrap_ci(np.array(allv))
            out.append({"analysis": label, "dataset": "eegbci", "task": "all_paradigms_pooled", "n": len(allv), "mean_delta": C.fmt(np.mean(allv)), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi),
                        "p_wilcoxon": C.fmt(C.wilcoxon_paired(np.array(allv), np.zeros(len(allv)))[1]), **{f"tost_{k}": C.fmt(v) for k, v in t.items()}})
    if out:
        outputs.append(C.write_csv(R / f"stats_{args.tag}_tost.csv", out))
        for r in out:
            print(f"  {r['analysis'][:3]} {r['dataset']:6s} {r['task']:26s} Δ={float(r['mean_delta']):+.5f} CI[{float(r['ci_low']):+.5f},{float(r['ci_high']):+.5f}] TOST equivalent={r['tost_equivalent']} (90%CI [{float(r['tost_ci90_low']):+.5f},{float(r['tost_ci90_high']):+.5f}])")

    # ------------------------------------------------------------------ E2 / E4 sweeps
    from scipy.stats import spearmanr

    for path, xkey, label in ((args.e2, "C", "e2"), (args.e4, "d", "e4")):
        if not path:
            continue
        rr = C.read_csv(Path(path))
        for r in rr:
            r["dataset"] = "eegbci"
            r["model"] = "riemann_ts_lr"
        subj_acc, _ = seed_averaged(rr, extra_keys=(xkey,))
        # key = (dataset, task, model, protocol, x, subject)
        pm = defaultdict(list)
        for k, v in subj_acc.items():
            pm[(k[0], "paradigm_mean", k[2], k[3], k[4], k[5])].append(v)
        subj_acc.update({k: float(np.mean(v)) for k, v in pm.items() if len(v) == 4})
        out = []
        xs = sorted({float(k[4]) for k in subj_acc})
        for task in sorted({k[1] for k in subj_acc}):
            d12_means = []
            for x in xs:
                P = {p: {k[5]: v for k, v in subj_acc.items() if k[1] == task and k[3] == p and float(k[4]) == x} for p in ("trial_random", "run_disjoint", "subject_disjoint")}
                if not all(P.values()):
                    continue
                c12 = contrast(P["run_disjoint"], P["subject_disjoint"])
                c02 = contrast(P["trial_random"], P["subject_disjoint"])
                d12_means.append(c12["mean_diff"])
                row = {"task": task, xkey: x}
                for p in P:
                    v = np.array(list(P[p].values()))
                    lo, hi = C.bootstrap_ci(v)
                    row.update({f"acc_{p}": C.fmt(v.mean()), f"acc_{p}_ci_low": C.fmt(lo), f"acc_{p}_ci_high": C.fmt(hi)})
                row.update({"d12_mean": C.fmt(c12["mean_diff"]), "d12_ci_low": C.fmt(c12["ci_low"]), "d12_ci_high": C.fmt(c12["ci_high"]), "d12_p": C.fmt(c12["p_wilcoxon"]),
                            "d02_mean": C.fmt(c02["mean_diff"]), "d02_ci_low": C.fmt(c02["ci_low"]), "d02_ci_high": C.fmt(c02["ci_high"]), "d02_p": C.fmt(c02["p_wilcoxon"]), "n": c12["n"]})
                out.append(row)
            if len(d12_means) >= 3:
                rho, p = spearmanr(xs[: len(d12_means)], d12_means)
                out.append({"task": task, xkey: "spearman_d12_vs_x", "d12_mean": C.fmt(rho), "d12_p": C.fmt(p), "n": len(d12_means)})
        outputs.append(C.write_csv(R / f"stats_{args.tag}_{label}_sweep.csv", out))
        for r in out:
            if r["task"] == "paradigm_mean":
                print(f"  {label} {xkey}={r[xkey]!s:18s} " + (f"P0={float(r['acc_trial_random']):.4f} P1={float(r['acc_run_disjoint']):.4f} P2={float(r['acc_subject_disjoint']):.4f} Δ12={float(r['d12_mean']):+.4f} [{float(r['d12_ci_low']):+.4f},{float(r['d12_ci_high']):+.4f}]" if "acc_trial_random" in r else f"rho={r['d12_mean']} p={r['d12_p']}"))

    # ------------------------------------------------------------------ E8 factorial
    if args.e8:
        rr = C.read_csv(Path(args.e8))
        out = []
        cells = defaultdict(list)
        for r in rr:
            cells[(r["task"], r["model"], int(r["N"]), int(r["d"]), int(r["draw"]))].append((r["protocol"], float(r["mean_acc"])))
        infl = defaultdict(list)
        for (task, model, N, d, draw), vals in cells.items():
            p = dict(vals)
            if "trial_random" in p and "subject_disjoint" in p:
                infl[(task, model, N, d)].append(p["trial_random"] - p["subject_disjoint"])
        for (task, model, N, d), v in sorted(infl.items()):
            v = np.array(v)
            lo, hi = C.bootstrap_ci(v) if len(v) > 1 else (float("nan"), float("nan"))
            out.append({"task": task, "model": model, "N": N, "d": d, "n_draws": len(v), "inflation_mean": C.fmt(v.mean()), "ci_low": C.fmt(lo), "ci_high": C.fmt(hi), "sd": C.fmt(v.std(ddof=1)) if len(v) > 1 else ""})
        # OLS on log2 N, log2 d (TS+LR), draw-bootstrap CI on coefficients
        for task in sorted({k[0] for k in infl}):
            X, Y = [], []
            for (t, model, N, d), v in infl.items():
                if t == task and model == "riemann_ts_lr":
                    for val in v:
                        X.append([1.0, np.log2(N), np.log2(d), np.log2(N) * np.log2(d)])
                        Y.append(val)
            if len(Y) > 8:
                X, Y = np.array(X), np.array(Y)
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                rng = np.random.default_rng(C.SEED)
                boots = []
                for _ in range(2000):
                    idx = rng.integers(0, len(Y), len(Y))
                    boots.append(np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0])
                boots = np.array(boots)
                for name, j in (("intercept", 0), ("log2N", 1), ("log2d", 2), ("log2N_x_log2d", 3)):
                    out.append({"task": task, "model": "riemann_ts_lr", "N": "ols", "d": name, "n_draws": len(Y), "inflation_mean": C.fmt(beta[j]),
                                "ci_low": C.fmt(np.percentile(boots[:, j], 2.5)), "ci_high": C.fmt(np.percentile(boots[:, j], 97.5)), "sd": ""})
        outputs.append(C.write_csv(R / f"stats_{args.tag}_factorial.csv", out))
        for r in out:
            if r["N"] == "ols":
                print(f"  E8 OLS {r['task']:24s} {r['d']:14s} β={float(r['inflation_mean']):+.4f} CI[{float(r['ci_low']):+.4f},{float(r['ci_high']):+.4f}]")

    # ------------------------------------------------------------------ E7 width sweep
    if args.e7:
        rr = C.read_csv(Path(args.e7))
        pf_path = Path(args.e7.replace("_per_subject.csv", "_per_fold.csv"))
        pf = C.read_csv(pf_path) if pf_path.exists() else []
        params = {}
        gaps = defaultdict(list)
        for r in pf:
            params[(r["model"], r["width"])] = int(r["n_params"])
            gaps[(r["task"], r["model"], r["width"], r["protocol"])].append(float(r["train_acc"]) - float(r["test_acc"]))
        for r in rr:
            r["dataset"] = "eegbci"
        subj_acc, _ = seed_averaged(rr, extra_keys=("width",))
        pm = defaultdict(list)
        for k, v in subj_acc.items():
            pm[(k[0], "paradigm_mean", k[2], k[3], k[4], k[5])].append(v)
        subj_acc.update({k: float(np.mean(v)) for k, v in pm.items() if len(v) == 4})
        out = []
        for model in sorted({k[2] for k in subj_acc}):
            widths = sorted({int(k[4]) for k in subj_acc if k[2] == model})
            for task in sorted({k[1] for k in subj_acc}):
                ds = []
                for w in widths:
                    A = {k[5]: v for k, v in subj_acc.items() if k[1] == task and k[2] == model and k[3] == "trial_random" and int(k[4]) == w}
                    B = {k[5]: v for k, v in subj_acc.items() if k[1] == task and k[2] == model and k[3] == "subject_disjoint" and int(k[4]) == w}
                    if not A or not B:
                        continue
                    c = contrast(A, B)
                    ds.append(c["mean_diff"])
                    if task == "paradigm_mean":
                        g0 = [x for (t_, m_, w_, p_), v in gaps.items() if m_ == model and w_ == str(w) and p_ == "trial_random" for x in v]
                        g2 = [x for (t_, m_, w_, p_), v in gaps.items() if m_ == model and w_ == str(w) and p_ == "subject_disjoint" for x in v]
                    else:
                        g0 = gaps.get((task, model, str(w), "trial_random"), [])
                        g2 = gaps.get((task, model, str(w), "subject_disjoint"), [])
                    out.append({"task": task, "model": model, "width": w, "n_params": params.get((model, str(w)), ""), "acc_P0": C.fmt(c["mean_a"]), "acc_P2": C.fmt(c["mean_b"]),
                                "d02_mean": C.fmt(c["mean_diff"]), "d02_ci_low": C.fmt(c["ci_low"]), "d02_ci_high": C.fmt(c["ci_high"]), "d02_p": C.fmt(c["p_wilcoxon"]),
                                "train_test_gap_P0": C.fmt(np.mean(g0)) if g0 else "", "train_test_gap_P2": C.fmt(np.mean(g2)) if g2 else "", "n": c["n"]})
                if len(ds) >= 3:
                    rho, p = spearmanr(widths[: len(ds)], ds)
                    out.append({"task": task, "model": model, "width": "spearman_d02_vs_width", "d02_mean": C.fmt(rho), "d02_p": C.fmt(p), "n": len(ds)})
        outputs.append(C.write_csv(R / f"stats_{args.tag}_width.csv", out))
        for r in out:
            if r["task"] == "paradigm_mean":
                print(f"  E7 {r['model']:14s} width={r['width']!s:22s} params={r.get('n_params','')!s:7s} Δ02={r['d02_mean']} p={r['d02_p']}")

    C.write_manifest(R / f"stats_{args.tag}_manifest.json", f"stats_{args.tag}", started, outputs, {"args": vars(args)})
    print(f"DONE stats tag={args.tag} ({len(outputs)} tables)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
