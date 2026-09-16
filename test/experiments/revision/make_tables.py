#!/usr/bin/env python3
"""Build the LaTeX tables of the revised manuscript from the stats/results CSVs (exact data only).

Outputs (manuscript/v2/tables/):
  table1_eegbci_ladder.tex   mean [95 % CI] per decoder x protocol (paradigm mean), Δ(P1−P2), Δ(P0−P1), Δ_cal with Holm p
  table2_bci2a_ladder.tex    same for BCI-IV-2a with LOSO; exact Wilcoxon p, Holm p, n positive/9
  table3_reference.tex       reference tests: transductive (LOSO/P2) and target-subject-in-reference (E6) with TOST
  table4_models.tex          decoder settings and trainable-parameter counts
  tableS_*.tex               per-paradigm EEGBCI ladder, classifiers (E5), mean removal (E3c), PCA (E4), width (E7), factorial (E8), calibration (E12)
Every number is read from a CSV named in the accompanying provenance file tables_provenance.txt.
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

OUT = C.ROOT / "manuscript" / "v2" / "tables"
MODEL_TEX = {"csp_lda": "CSP+LDA", "riemann_ts_lr": "TS+LR", "shallow_fbcsp": "ShallowFBCSPNet", "eegnet_v4": "EEGNet",
             "csp_lda_ztrial": "CSP+LDA (z-scored trials)", "riemann_ts_lr_ztrial": "TS+LR (z-scored trials)",
             "shallow_fbcsp_globalscale": "ShallowFBCSPNet (global scale)", "eegnet_v4_globalscale": "EEGNet (global scale)"}


def fp(p: float) -> str:
    if p != p:
        return "--"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def cell(mean, lo, hi, nd=3, sign=False) -> str:
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f"{f.format(float(mean))} [{f.format(float(lo))}, {f.format(float(hi))}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    R = C.RESULTS
    OUT.mkdir(parents=True, exist_ok=True)
    prov = []
    means = C.read_csv(R / f"stats_{args.tag}_ladder_means.csv")
    contr = C.read_csv(R / f"stats_{args.tag}_contrasts.csv")
    prov += [f"stats_{args.tag}_ladder_means.csv", f"stats_{args.tag}_contrasts.csv"]
    M = {(r["dataset"], r["task"], r["model"], r["protocol"]): r for r in means}
    K = {(r["dataset"], r["task"], r["model"], r["contrast"]): r for r in contr}

    # ------------------------------------------------------------- Table 1 EEGBCI
    lines = [r"\begin{table}", r"\caption{EEGBCI (109 subjects; four paradigms averaged per subject): mean accuracy over subjects with the 2000-resample subject-bootstrap 95\,\% CI, "
             r"and paired per-subject differences. P0 = trial-random 3-fold, P1 = run-disjoint subject-overlapping 3-fold, P2 = subject-disjoint 3-fold (training fraction 2/3 in all three; "
             r"P0 and P2 averaged over five partition seeds; networks additionally over three training seeds); within = leave-one-run-out inside each subject (networks: fixed 60 epochs). "
             r"$p$: two-sided paired Wilcoxon signed-rank test over the 109 subjects, Holm-adjusted within this table. Chance = 0.50.}", r"\label{tab:eegbci}", r"\centering", r"\footnotesize", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l c c c c}", r"\hline", r"Decoder & within & P0 trial-random & P1 run-disjoint & P2 subject-disjoint \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for p in ("within_subject", "trial_random", "run_disjoint", "subject_disjoint"):
            r = M.get(("eegbci", "paradigm_mean", m, p))
            row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\vspace{4pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c}", r"\hline",
              r"\multicolumn{4}{l}{\textit{Paired per-subject differences}: mean [95\,\% CI]; $p_{\rm Holm}$} \\", r"\hline",
              r"Decoder & $\Delta$(P1$-$P2) subject overlap & $\Delta$(P0$-$P1) run adjacency & $\Delta_{\rm cal}$ = within$-$P2 \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for c in ("subject_overlap_component", "run_temporal_component", "calibration_gap"):
            r = K.get(("eegbci", "paradigm_mean", m, c))
            row.append((cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) + "; " + fp(float(r["p_holm"]))) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table1_eegbci_ladder.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Table 2 BCI-IV-2a
    lines = [r"\begin{table}", r"\caption{BCI-IV-2a (9 subjects, left- versus right-hand imagery): mean accuracy with the subject-bootstrap 95\,\% CI and paired per-subject differences. "
             r"Protocols: within = session-to-session inside each subject (both directions averaged; training fraction 1/2); P0a = trial-random 6-fold (5/6); P1 = run-disjoint subject-overlapping 6-fold (5/6); "
             r"P0b = trial-random 2-fold (1/2); P1s = session-disjoint subject-overlapping 2-fold (1/2); P2 = leave-one-subject-out (8/9). "
             r"$p$: exact two-sided paired Wilcoxon test ($n=9$; the smallest attainable $p$ is 0.0039), Holm-adjusted within this table; $n_+$ = number of subjects with a positive difference.}",
             r"\label{tab:bci2a}", r"\centering", r"\footnotesize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c c c c}", r"\hline",
             r"Decoder & within & P0a & P1 & P0b & P1s & P2 (LOSO) \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for p in ("within_session", "trial_random_6", "run_disjoint", "trial_random_2", "session_disjoint", "loso"):
            r = M.get(("bci2a", "bci2a_left_right", m, p))
            row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\vspace{4pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c}", r"\hline",
              r"\multicolumn{4}{l}{\textit{Paired differences}: mean [95\,\% CI]; exact $p$; $p_{\rm Holm}$; $n_+$/9} \\", r"\hline",
              r"Decoder & $\Delta$(P1$-$P2) subject overlap (run-disjoint) & $\Delta$(P1s$-$P2) subject overlap (session-disjoint) & $\Delta$(P0a$-$P2) \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for c in ("subject_overlap_component_run", "subject_overlap_component_session", "delta_leak_original_definition"):
            r = K.get(("bci2a", "bci2a_left_right", m, c))
            row.append((cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) + f"; {fp(float(r['p_wilcoxon']))}; {fp(float(r['p_holm']))}; {r['n_positive']}/9") if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"Decoder & $\Delta$(P0a$-$P1) run adjacency & $\Delta$(P0b$-$P1s) run/session & $\Delta_{\rm cal}$ = within$-$P2 \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for c in ("run_temporal_component", "run_session_component", "calibration_gap"):
            r = K.get(("bci2a", "bci2a_left_right", m, c))
            row.append((cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) + f"; {fp(float(r['p_wilcoxon']))}; {fp(float(r['p_holm']))}; {r['n_positive']}/9") if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table2_bci2a_ladder.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Table 3 reference tests
    tost = C.read_csv(R / f"stats_{args.tag}_tost.csv") if (R / f"stats_{args.tag}_tost.csv").exists() else []
    prov.append(f"stats_{args.tag}_tost.csv")
    lines = [r"\begin{table}", r"\caption{Reference-mean tests for the tangent-space pipeline ($C$ = 1). Transductive reference: under the subject-disjoint protocol (EEGBCI 3-fold; BCI-IV-2a LOSO) the reference is estimated from training-subject covariances only or from training plus test covariances (label-free), classifier unchanged; $\Delta$ = transductive $-$ inductive per test subject. Target-subject-in-reference (EEGBCI, P0 trial-random 3-fold, one seed): for each test subject the reference is "
             r"estimated with (R$_{\rm full}$) or without (R$_{-s}$) that subject's training-fold covariances while the classifier is trained on the same training trials; $\Delta_{\rm ref}$ = acc(R$_{\rm full}$) $-$ acc(R$_{-s}$) per subject. "
             r"TOST: two one-sided tests against the pre-specified margin $\pm0.01$ (equivalence when the 90\,\% CI lies inside the margin). Per-paradigm rows use 109 subjects each; the pooled row uses all 436 subject-paradigm cells.}",
             r"\label{tab:reference}", r"\centering", r"\footnotesize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c c c}", r"\hline",
             r"Analysis & $n$ & mean $\Delta$ [95\,\% CI] & Wilcoxon $p$ & 90\,\% CI (TOST) & equivalent \\", r"\hline"]
    for r in tost:
        name = C.TASK_SHORT.get(r["task"], r["task"].replace("_", " "))
        label = "target subject in reference (P0), EEGBCI" if r["analysis"].startswith("e6") else ("transductive reference (LOSO), BCI-IV-2a" if r.get("dataset") == "bci2a" else "transductive reference (P2), EEGBCI")
        lines.append(f"{label}, {name} & {r['n']} & {cell(r['mean_delta'], r['ci_low'], r['ci_high'], nd=4, sign=True)} & {fp(float(r['p_wilcoxon']))} & "
                     f"[{float(r['tost_ci90_low']):+.4f}, {float(r['tost_ci90_high']):+.4f}] & {'yes' if r['tost_equivalent'] in ('1', 'True', 'true') else 'no'} \\\\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table3_reference.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Table 4 models (parameter counts from e9/e7 per-fold tables)
    params = {}
    for name in (f"e9_{args.tag}_t12_per_fold.csv", f"e9_{args.tag}_t34_per_fold.csv", f"e10_{args.tag}_per_fold.csv"):
        p = R / name
        if p.exists():
            prov.append(name)
            for r in C.read_csv(p):
                params[(r["model"], r.get("dataset", "eegbci"))] = int(r["n_params"])
    lines = [r"\begin{table}", r"\caption{Decoders and their settings. Feature dimension and trainable-parameter counts are given for EEGBCI (64 channels, 641 samples) and BCI-IV-2a (22 channels, 1001 samples).}",
             r"\label{tab:models}", r"\centering", r"\footnotesize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l p{5.2cm} p{3.4cm} p{4.2cm}}", r"\hline", r"Decoder & Features / architecture & Fitted parameters (EEGBCI / BCI-IV-2a) & Training \\", r"\hline",
             r"CSP+LDA & 6 CSP log-variance features (Ledoit--Wolf regularized covariances) & LDA on 6 features: 7 / 7 & closed form \\",
             r"TS+LR & OAS covariance, tangent space at the training Fr\'echet mean; $N_c(N_c+1)/2$ = 2080 / 253 features & $\ell_2$ logistic regression: 2081 / 254 & lbfgs, max 3000 iterations, $C$ = 1 unless swept \\",
             f"ShallowFBCSPNet & braindecode defaults (40 temporal, 40 spatial filters, square, mean-pool, log) & {params.get(('shallow_fbcsp', 'eegbci'), '--')} / {params.get(('shallow_fbcsp', 'bci2a'), '--')} & Adam $10^{{-3}}$, batch 64, $\\le$40 epochs, early stopping (patience 6) \\\\",
             f"EEGNet & braindecode EEGNet ($F_1$ = 8, $D$ = 2, $F_2$ = 16) & {params.get(('eegnet_v4', 'eegbci'), '--')} / {params.get(('eegnet_v4', 'bci2a'), '--')} & as above \\\\",
             r"\hline", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table4_models.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: per-paradigm EEGBCI
    lines = [r"\begin{table}", r"\caption{EEGBCI per paradigm (109 subjects): mean accuracy [95\,\% CI] under each protocol (top) and the paired per-subject differences with Holm-adjusted $p$ within each paradigm family (bottom). MI = motor imagery, ME = motor execution.}", r"\label{tab:s_paradigm}", r"\centering", r"\scriptsize", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l l c c c c}", r"\hline", r"Paradigm & Decoder & within & P0 trial-random & P1 run-disjoint & P2 subject-disjoint \\", r"\hline"]
    for t in C.TASKS:
        for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
            row = [C.TASK_SHORT[t], MODEL_TEX[m]]
            for p in ("within_subject", "trial_random", "run_disjoint", "subject_disjoint"):
                r = M.get(("eegbci", t, m, p)); row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\vspace{4pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l l c c c}", r"\hline",
              r"Paradigm & Decoder & $\Delta$(P1$-$P2) subject overlap; $p_{\rm Holm}$ & $\Delta$(P0$-$P1) run adjacency; $p_{\rm Holm}$ & $\Delta_{\rm cal}$ = within$-$P2; $p_{\rm Holm}$ \\", r"\hline"]
    for t in C.TASKS:
        for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
            row = [C.TASK_SHORT[t], MODEL_TEX[m]]
            for c in ("subject_overlap_component", "run_temporal_component", "calibration_gap"):
                r = K.get(("eegbci", t, m, c)); row.append((cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) + "; " + fp(float(r["p_holm"]))) if r else "--")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    (OUT / "tableS_paradigm.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: matched preprocessing (E13) rows if present
    rows13 = [r for r in contr if r["task"] == "paradigm_mean" and (r["model"].endswith("_ztrial") or r["model"].endswith("_globalscale"))]
    if rows13:
        lines = [r"\begin{table}", r"\caption{Matched-preprocessing control (EEGBCI, paradigm mean): classical decoders on per-trial z-scored trials and networks on globally scaled trials (one seed). Paired differences with Holm-adjusted $p$.}",
                 r"\label{tab:s_preproc}", r"\centering", r"\footnotesize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c}", r"\hline", r"Decoder (preprocessing) & P2 subject-disjoint & $\Delta$(P1$-$P2); $p_{\rm Holm}$ & $\Delta$(P0$-$P1); $p_{\rm Holm}$ \\", r"\hline"]
        for m in sorted({r["model"] for r in rows13}):
            r2 = M.get(("eegbci", "paradigm_mean", m, "subject_disjoint"))
            row = [MODEL_TEX.get(m, m), cell(r2["mean_acc"], r2["ci_low"], r2["ci_high"]) if r2 else "--"]
            for c in ("subject_overlap_component", "run_temporal_component"):
                r = K.get(("eegbci", "paradigm_mean", m, c)); row.append((cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) + "; " + fp(float(r["p_holm"]))) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
        (OUT / "tableS_preproc.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E5 classifiers
    p5 = R / f"stats_{args.tag}_e5_contrasts.csv"
    if p5.exists():
        prov.append(p5.name)
        rows = [r for r in C.read_csv(p5) if r["task"] == "paradigm_mean"]
        lines = [r"\begin{table}", r"\caption{Classifiers applied to identical inductive tangent-space features (EEGBCI, paradigm mean, 109 subjects; MDM operates on the covariances). Paired per-subject differences with Holm-adjusted $p$ (family = this table).}",
                 r"\label{tab:s_clf}", r"\centering", r"\small", r"\begin{tabular}{l c c c}", r"\hline", r"Classifier & $\Delta$(P1$-$P2) [CI]; $p_{\rm Holm}$ & $\Delta$(P0$-$P1) [CI]; $p_{\rm Holm}$ & $\Delta$(P0$-$P2) [CI]; $p_{\rm Holm}$ \\", r"\hline"]
        lab = {"lr": "$\\ell_2$ logistic regression ($C$ = 1)", "slda": "shrinkage LDA", "linsvm": "linear SVM ($C$ = 1)", "ridge": "ridge classifier ($\\alpha$ = 1)", "knn": "$k$-NN ($k$ = 5)", "mdm": "MDM (Riemannian class means)"}
        for c in ("lr", "slda", "linsvm", "ridge", "knn", "mdm"):
            row = [lab[c]]
            for con in ("subject_overlap_component", "run_temporal_component", "delta_leak_original_definition"):
                r = [x for x in rows if x["clf"] == c and x["contrast"] == con]
                row.append((cell(r[0]["mean_diff"], r[0]["ci_low"], r[0]["ci_high"], sign=True) + "; " + fp(float(r[0]["p_holm"]))) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_classifiers.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E3c mean removal
    p3 = R / f"stats_{args.tag}_e3c_contrasts.csv"
    if p3.exists():
        prov.append(p3.name)
        rows = [r for r in C.read_csv(p3) if r["task"] == "paradigm_mean"]
        lines = [r"\begin{table}", r"\caption{TS+LR ($C$ = 1) after removing subject-level structure (EEGBCI, paradigm mean): per-subject feature-mean removal (each subject centred on its own mean over all its trials) and Riemannian recentering "
                 r"($C_i \mapsto G_s^{-1/2} C_i G_s^{-1/2}$ with $G_s$ the subject's Fr\'echet mean). Paired differences with Holm-adjusted $p$.}", r"\label{tab:s_center}", r"\centering", r"\small", r"\begin{tabular}{l c c c}", r"\hline",
                 r"Variant & $\Delta$(P1$-$P2) [CI]; $p_{\rm Holm}$ & $\Delta$(P0$-$P1) [CI]; $p_{\rm Holm}$ & $\Delta$(P0$-$P2) [CI]; $p_{\rm Holm}$ \\", r"\hline"]
        for v, lab_ in (("feature_centered", "subject feature-mean removed"), ("riemann_recentered", "Riemannian recentering")):
            row = [lab_]
            for con in ("subject_overlap_component", "run_temporal_component", "delta_leak_original_definition"):
                r = [x for x in rows if x["variant"] == v and x["contrast"] == con]
                row.append((cell(r[0]["mean_diff"], r[0]["ci_low"], r[0]["ci_high"], sign=True) + "; " + fp(float(r[0]["p_holm"]))) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_center.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E4 PCA sweep and E2 sweep (paradigm mean)
    for label, xkey, capt in (("e4", "d", "PCA-dimension sweep of the tangent features (TS+LR, $C$ = 1; EEGBCI paradigm mean)."), ("e2", "C", "Regularization sweep of TS+LR (EEGBCI paradigm mean); $C$ = inverse $\\ell_2$ penalty (larger $C$ = weaker regularization).")):
        p = R / f"stats_{args.tag}_{label}_sweep.csv"
        if not p.exists():
            continue
        prov.append(p.name)
        rows = [r for r in C.read_csv(p) if r["task"] == "paradigm_mean"]
        lines = [r"\begin{table}", rf"\caption{{{capt} Mean accuracy [95\,\% CI] per protocol and the paired per-subject differences; the last row gives Spearman's $\rho$ of $\Delta$(P1$-$P2) against the swept variable.}}", rf"\label{{tab:s_{label}}}", r"\centering", r"\scriptsize",
                 r"\begin{tabular}{l c c c c c}", r"\hline", f"${xkey}$ & P0 & P1 & P2 & $\\Delta$(P1$-$P2) [CI]; $p$ & $\\Delta$(P0$-$P2) [CI]; $p$ \\\\", r"\hline"]
        for r in rows:
            if "acc_trial_random" in r and r["acc_trial_random"]:
                lines.append(f"{float(r[xkey]):g} & {cell(r['acc_trial_random'], r['acc_trial_random_ci_low'], r['acc_trial_random_ci_high'])} & {cell(r['acc_run_disjoint'], r['acc_run_disjoint_ci_low'], r['acc_run_disjoint_ci_high'])} & "
                             f"{cell(r['acc_subject_disjoint'], r['acc_subject_disjoint_ci_low'], r['acc_subject_disjoint_ci_high'])} & {cell(r['d12_mean'], r['d12_ci_low'], r['d12_ci_high'], sign=True)}; {fp(float(r['d12_p']))} & "
                             f"{cell(r['d02_mean'], r['d02_ci_low'], r['d02_ci_high'], sign=True)}; {fp(float(r['d02_p']))} \\\\")
            else:
                lines.append(rf"\multicolumn{{6}}{{l}}{{Spearman $\rho$($\Delta$(P1$-$P2), {xkey}) = {float(r['d12_mean']):.3f}, $p$ = {fp(float(r['d12_p']))} ($n$ = {r['n']} levels)}} \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / f"tableS_{label}.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E7 width, E8 factorial
    p7 = R / f"stats_{args.tag}_width.csv"
    if p7.exists():
        prov.append(p7.name)
        rows = [r for r in C.read_csv(p7) if r["task"] == "paradigm_mean"]
        lines = [r"\begin{table}", r"\caption{Network-width sweeps (EEGBCI, paradigm mean, one seed): trainable parameters, accuracy under P0 and P2, the paired difference and the mean train--test gap over folds.}", r"\label{tab:s_width}", r"\centering", r"\scriptsize", r"\resizebox{\linewidth}{!}{%",
                 r"\begin{tabular}{l l r c c c c c}", r"\hline", r"Network & width & parameters & P0 & P2 & $\Delta$(P0$-$P2) [CI]; $p$ & gap P0 & gap P2 \\", r"\hline"]
        for r in rows:
            if r["width"].startswith("spearman"):
                lines.append(rf"\multicolumn{{8}}{{l}}{{{MODEL_TEX.get(r['model'], r['model'])}: Spearman $\rho$($\Delta$(P0$-$P2), width) = {float(r['d02_mean']):.3f}, $p$ = {fp(float(r['d02_p']))}}} \\")
            else:
                lines.append(f"{MODEL_TEX.get(r['model'], r['model'])} & {r['width']} & {int(r['n_params']) if r['n_params'] else '--'} & {float(r['acc_P0']):.3f} & {float(r['acc_P2']):.3f} & {cell(r['d02_mean'], r['d02_ci_low'], r['d02_ci_high'], sign=True)}; {fp(float(r['d02_p']))} & "
                             f"{float(r['train_test_gap_P0']):+.3f} & {float(r['train_test_gap_P2']):+.3f} \\\\")
        lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
        (OUT / "tableS_width.tex").write_text("\n".join(lines) + "\n")
    p8 = R / f"stats_{args.tag}_factorial.csv"
    if p8.exists():
        prov.append(p8.name)
        rows = C.read_csv(p8)
        lines = [r"\begin{table}", r"\caption{Within-dataset factorial (EEGBCI): mean $\Delta$(P0$-$P2) of TS+LR ($C$ = 1) over 10 random draws of $N$ training subjects (1 draw for $N$ = 109) and $d$ principal components, per paradigm, with draw-bootstrap 95\,\% CI; "
                 r"and the descriptive OLS of $\Delta$ on $\log_2 N$, $\log_2 d$ and their interaction (draw-bootstrap CIs).}", r"\label{tab:s_factorial}", r"\centering", r"\scriptsize", r"\begin{tabular}{l l r r c}", r"\hline", r"Paradigm & model & $N$ & $d$ & $\Delta$(P0$-$P2) [CI] \\", r"\hline"]
        for r in rows:
            if r["N"] == "ols":
                nm = {"intercept": "intercept", "log2N": "$\\log_2 N$", "log2d": "$\\log_2 d$", "log2N_x_log2d": "$\\log_2 N \\times \\log_2 d$"}.get(r["d"], r["d"])
                lines.append(f"{C.TASK_SHORT.get(r['task'], r['task'])} & OLS slope & \\multicolumn{{2}}{{l}}{{{nm}}} & {cell(r['inflation_mean'], r['ci_low'], r['ci_high'], sign=True)} \\\\")
            else:
                val = cell(r['inflation_mean'], r['ci_low'], r['ci_high'], sign=True) if r['ci_low'] not in ('nan', '') else f"{float(r['inflation_mean']):+.3f} (single draw)"
                lines.append(f"{C.TASK_SHORT.get(r['task'], r['task'])} & {MODEL_TEX.get(r['model'], r['model'])} & {r['N']} & {r['d']} & {val} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_factorial.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E15 exploratory stratification
    p15 = R / f"stats_{args.tag}_e15.csv"
    e15 = R / f"e15_{args.tag}_summary.csv"
    if e15.exists():
        prov.append(e15.name)
        rows = C.read_csv(e15)
        agg = defaultdict(list)
        for r in rows:
            agg[(r["task"], r["protocol"])].append((float(r["subject_prior_accuracy_mean"]), float(r["corr_train_test_class_fraction"]) if r["corr_train_test_class_fraction"] not in ("nan", "") else float("nan")))
        lines = [r"\begin{table}", r"\caption{Exploratory (added after the ladder results were seen): a predictor that outputs the majority class of the subject's own training trials (global prior for subject-disjoint folds), and the correlation between a subject's training and test class-1 fractions across (subject, fold) cells. Means over partition seeds (five for P0 and P2, one for P1).}",
                 r"\label{tab:s_e15}", r"\centering", r"\small", r"\begin{tabular}{l l c c}", r"\hline", r"Paradigm & protocol & subject-prior accuracy & corr(train, test class fraction) \\", r"\hline"]
        for (task, protocol), v in sorted(agg.items(), key=lambda kv: (C.TASKS.index(kv[0][0]), kv[0][1])):
            cval = np.nanmean([c for _, c in v])
            lines.append(f"{C.TASK_SHORT[task]} & {protocol.replace('_', ' ')} & {np.mean([a for a, _ in v]):.3f} & {('%+.3f' % cval) if cval == cval else '--'} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_e15.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: network training details (per model x protocol x dataset)
    train_rows = []
    for name, dataset in ((f"e9_{args.tag}_t12_per_fold.csv", "EEGBCI"), (f"e9_{args.tag}_t34_per_fold.csv", "EEGBCI"), (f"e10_{args.tag}_per_fold.csv", "BCI-IV-2a"),
                          (f"e11_{args.tag}_t12_per_fold.csv", "EEGBCI within"), (f"e11_{args.tag}_t34_per_fold.csv", "EEGBCI within")):
        p = R / name
        if p.exists():
            prov.append(name)
            for r in C.read_csv(p):
                train_rows.append((dataset, r["model"], r.get("protocol", "within_subject"), float(r["train_acc"]), float(r.get("val_acc", "nan")) if r.get("val_acc", "nan") not in ("nan", "") else float("nan"),
                                   float(r["test_acc"]), int(r.get("best_epoch", r.get("epochs_run", 0))), int(r.get("n_val", 0)), int(r["n_params"])))
    if train_rows:
        agg = defaultdict(list)
        for row in train_rows:
            agg[row[:3]].append(row[3:])
        lines = [r"\begin{table}", r"\caption{Network training details: mean over folds and seeds of the training accuracy (on all training trials, best-epoch weights), validation accuracy at the stopping epoch, test accuracy, stopping epoch and validation-set size; trainable parameters. Within-subject runs use a fixed 60 epochs (EEGBCI) without early stopping.}",
                 r"\label{tab:s_training}", r"\centering", r"\scriptsize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l l l c c c c c r}", r"\hline", r"Dataset & network & protocol & train acc & val acc & test acc & stop epoch & $n_{\rm val}$ & params \\", r"\hline"]
        for (dataset, model, protocol), v in sorted(agg.items()):
            a = np.array(v, dtype=float)
            lines.append(f"{dataset} & {MODEL_TEX.get(model, model)} & {protocol.replace('_', ' ')} & {a[:, 0].mean():.3f} & {np.nanmean(a[:, 1]):.3f} & {a[:, 2].mean():.3f} & {a[:, 3].mean():.1f} & {a[:, 4].mean():.0f} & {int(a[0, 5])} \\\\")
        lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
        (OUT / "tableS_training.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E12 calibration curve
    p12 = R / f"stats_{args.tag}_e12_calibration.csv"
    if p12.exists():
        prov.append(p12.name)
        rows = C.read_csv(p12)
        lines = [r"\begin{table}", r"\caption{Calibration-size curve on the two imagery paradigms of EEGBCI (mean over both; 109 test subjects, evaluated on each subject's third run): accuracy of a subject-disjoint model after appending $k$ labelled trials of the test subject (CSP+LDA and TS+LR refitted; EEGNet fine-tuned for 20 epochs; three draws averaged), the paired difference to $k = 0$ with Holm-adjusted $p$ (family = one decoder), and the within-subject and subject-disjoint (P2) accuracies of the same decoders on the same paradigms.}",
                 r"\label{tab:s_calib}", r"\centering", r"\footnotesize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l l c c}", r"\hline", r"Decoder & $k$ & accuracy [95\,\% CI] & $\Delta$ vs $k = 0$ [CI]; $p_{\rm Holm}$ \\", r"\hline"]
        for r in rows:
            kk = {"ref_within_subject": "within-subject reference", "ref_subject_disjoint": "subject-disjoint (P2) reference"}.get(r["k"], r["k"])
            diff = (cell(r["diff_vs_k0"], r["diff_ci_low"], r["diff_ci_high"], sign=True) + "; " + fp(float(r["p_holm"]))) if r["diff_vs_k0"] not in ("", ) and r["k"] not in ("0",) and not r["k"].startswith("ref") else "--"
            lines.append(f"{MODEL_TEX.get(r['model'], r['model'])} & {kk} & {cell(r['mean_acc'], r['ci_low'], r['ci_high'])} & {diff} \\\\")
        lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
        (OUT / "tableS_calibration.tex").write_text("\n".join(lines) + "\n")

    (OUT / "tables_provenance.txt").write_text("\n".join(f"results/revision/{p}" for p in prov) + "\n")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
