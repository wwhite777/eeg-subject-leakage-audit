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


def num(x, nd=3, sign=False) -> str:
    """Format one number; a value that rounds to zero is printed unsigned (never "-0.000")."""
    t = (f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}").format(float(x))
    if set(t) <= set("+-0."):
        t = f"{{:.{nd}f}}".format(0.0)
    return t


def cell(mean, lo, hi, nd=3, sign=False) -> str:
    return f"{num(mean, nd, sign)} [{num(lo, nd, sign)}, {num(hi, nd, sign)}]"


# Supplementary tables are placed exactly where they are called ([H], float package),
# so that every supplementary section is followed by its own table; supplement.tex also
# loads placeins with the section option so that no float can escape past its heading.
TS = r"\begin{table}[H]"
TIGHT = r"\setlength{\tabcolsep}{3pt}"


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    ap.add_argument("--factorial-tag", default=None, help="tag of the factorial statistics export (default: --tag); revision 2 uses rev2 for Table S7 while the other tables stay rev1")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--p0s-tag", default="rev2", help="tag of the additional P0s exploratory rows")
    args = ap.parse_args()
    OUT = args.out_dir
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
             r"and paired per-subject protocol contrasts. P0 = trial-random 3-fold, P1 = run-disjoint subject-overlapping 3-fold, P2 = subject-disjoint 3-fold (outer training fraction about 2/3 in all three; the networks' nested validation removes a further 15\,\% of that outer training set from the optimization subset); "
             r"within = leave-one-run-out inside each subject (networks: fixed 60 epochs). Seeds: the classical decoders use five partition seeds for P0 and P2 (P1 is deterministic); the networks use three seeds, each of which sets both the partition and the training, so their values average three partition-and-training replicates and not a five-by-three design. "
             r"$p_{\rm Holm}$: two-sided paired Wilcoxon signed-rank test over the 109 subjects (differences rounded to 12 decimals before ranking), Holm-adjusted over the 12 contrasts displayed in this table. A value whose magnitude rounds below 0.0005 is printed as 0.000 without a sign; the exact limits are in the released statistics file. Chance = 0.50.}", r"\label{tab:eegbci}", r"\centering",
             TIGHT, r"\begin{tabular}{l c c c c}", r"\hline", r"Decoder & within & P0 trial-random & P1 run-disjoint & P2 subject-disjoint \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for p in ("within_subject", "trial_random", "run_disjoint", "subject_disjoint"):
            r = M.get(("eegbci", "paradigm_mean", m, p))
            row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
              r"\multicolumn{4}{l}{\textit{Paired per-subject protocol contrasts}: mean [95\,\% CI]} \\", r"\hline",
              r"Decoder & \shortstack{$\Delta$(P1$-$P2)\\run-disjoint $-$\\subject-disjoint\\(subject-overlap\\contrast)} & \shortstack{$\Delta$(P0$-$P1)\\trial-random $-$\\run-disjoint\\(run/temporal\\contrast)} & \shortstack{$\Delta_{\rm cal}$ = within$-$P2\\within-subject $-$\\subject-disjoint\\(calibration contrast)} \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for c in ("subject_overlap_component", "run_temporal_component", "calibration_gap"):
            r = K.get(("eegbci", "paradigm_mean", m, c))
            row.append(cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
              r"\multicolumn{4}{l}{$p_{\rm Holm}$ of the same protocol contrasts, Holm-adjusted over the 12 contrasts of this table} \\", r"\hline",
              r"Decoder & \shortstack{$\Delta$(P1$-$P2)\\run-disjoint $-$\\subject-disjoint\\(subject-overlap\\contrast)} & \shortstack{$\Delta$(P0$-$P1)\\trial-random $-$\\run-disjoint\\(run/temporal\\contrast)} & \shortstack{$\Delta_{\rm cal}$ = within$-$P2\\within-subject $-$\\subject-disjoint\\(calibration contrast)} \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for c in ("subject_overlap_component", "run_temporal_component", "calibration_gap"):
            r = K.get(("eegbci", "paradigm_mean", m, c))
            row.append(fp(float(r["p_holm"])) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "table1_eegbci_ladder.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Table 2 BCI-IV-2a
    lines = [r"\begin{table}", r"\caption{BCI-IV-2a (9 subjects, left- versus right-hand imagery): mean accuracy with the subject-bootstrap 95\,\% CI and paired per-subject protocol contrasts. All intervals in this table are descriptive and unadjusted. "
             r"Protocols: within = session-to-session inside each subject (both directions averaged; training fraction 1/2); P0a = trial-random 6-fold (5/6); P1 = run-disjoint subject-overlapping 6-fold (5/6); "
             r"P0b = trial-random 2-fold (1/2); P1s = session-disjoint subject-overlapping 2-fold (1/2); P2 = leave-one-subject-out (LOSO; 8/9). "
             r"$p$: exact conditional two-sided signed-rank test ($n=9$; zeros dropped, tied ranks averaged; the smallest attainable raw $p$ is 0.0039 when all nine differences are nonzero); $p_{\rm Holm}$: Holm-adjusted over the 24 contrasts displayed in this table\rev{ and its continuation}, so no adjusted value can fall below 0.09375 ($= 24 \times 2/2^{9}$); $n_+$ = number of subjects with a positive difference. The lower block reports the mean paired difference over the nine subjects with its subject-bootstrap CI; the individual subject differences are in the released per-subject files. Sign-flip permutation $p$-values are in the released statistics file. $p$-values are displayed rounded to three decimals.}",
             r"\label{tab:bci2a}", r"\centering", TIGHT, r"\begin{tabular}{l c c c}", r"\hline",
             r"Decoder & within & P0a trial-random & P1 run-disjoint \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for p in ("within_session", "trial_random_6", "run_disjoint"):
            r = M.get(("bci2a", "bci2a_left_right", m, p))
            row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
              r"Decoder & P0b trial-random & P1s session-disjoint & P2 (LOSO) \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        row = [MODEL_TEX[m]]
        for p in ("trial_random_2", "session_disjoint", "loso"):
            r = M.get(("bci2a", "bci2a_left_right", m, p))
            row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
        lines.append(" & ".join(row) + r" \\")
    CONTRASTS_2A = (("subject_overlap_component_run", r"\shortstack[l]{$\Delta$(P1$-$P2)\\run-disjoint $-$ LOSO\\(subject-overlap contrast)}"),
                    ("subject_overlap_component_session", r"\shortstack[l]{$\Delta$(P1s$-$P2)\\session-disjoint $-$ LOSO\\(subject-overlap contrast)}"),
                    ("run_temporal_component", r"\shortstack[l]{$\Delta$(P0a$-$P1)\\trial-random $-$ run-disjoint\\(run/temporal contrast)}"),
                    ("run_session_component", r"\shortstack[l]{$\Delta$(P0b$-$P1s)\\trial-random $-$ session-disjoint\\(run/session contrast)}"),
                    ("delta_leak_original_definition", r"\shortstack[l]{$\Delta$(P0a$-$P2)\\trial-random $-$ LOSO\\(protocol contrast)}"),
                    ("calibration_gap", r"\shortstack[l]{$\Delta_{\rm cal}$ = within$-$P2\\within-subject $-$ LOSO\\(calibration contrast)}"))
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l l c c c c}", r"\hline",
              r"\multicolumn{6}{l}{\textit{Paired protocol contrasts over the nine subjects}} \\", r"\hline",
              r"Decoder & Contrast & mean [95\,\% CI] & exact $p$ & $p_{\rm Holm}$ & $n_+$/9 \\", r"\hline"]
    for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
        if m == "shallow_fbcsp":
            lines += [r"\hline", r"\end{tabular}", r"\end{table}",
                      r"\begin{table}", r"\addtocounter{table}{-1}",
                      r"\caption{\rev{(Continued.) Paired protocol contrasts over the nine subjects for ShallowFBCSPNet and EEGNet; columns as above.}}",
                      r"\centering", TIGHT, r"\begin{tabular}{l l c c c c}", r"\hline",
                      r"Decoder & Contrast & mean [95\,\% CI] & exact $p$ & $p_{\rm Holm}$ & $n_+$/9 \\", r"\hline"]
        for c, lab in CONTRASTS_2A:
            r = K.get(("bci2a", "bci2a_left_right", m, c))
            if not r:
                continue
            lines.append(f"{MODEL_TEX[m]} & {lab} & {cell(r['mean_diff'], r['ci_low'], r['ci_high'], sign=True)} & "
                         f"{fp(float(r['p_wilcoxon']))} & {fp(float(r['p_holm']))} & {r['n_positive']}/9 \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "table2_bci2a_ladder.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Table 3 reference tests
    tost = C.read_csv(R / f"stats_{args.tag}_tost.csv") if (R / f"stats_{args.tag}_tost.csv").exists() else []
    prov.append(f"stats_{args.tag}_tost.csv")
    cap5 = (r"\caption{Reference-mean tests for the tangent-space pipeline ($C$ = 1). "
            r"\textbf{Row key.} \textbf{E6} = target subject in the reference (EEGBCI, P0 trial-random 3-fold, one seed): for each test subject the reference is estimated with (R$_{\rm full}$) or without (R$_{-s}$) that subject's training-fold covariances while the classifier is trained on the \emph{same} training trials; $\Delta_{\rm ref}$ = acc(R$_{\rm full}$) $-$ acc(R$_{-s}$), so a positive value means that including the subject's own covariances in the reference helped. "
            r"\textbf{E14} = transductive reference: under the subject-disjoint protocol (EEGBCI 3-fold; BCI-IV-2a LOSO) the reference is estimated from training-subject covariances only or from training plus test covariances (label-free), classifier unchanged; $\Delta$ = transductive $-$ inductive per test subject. "
            r"MI = motor imagery, ME = motor execution, L/R = left versus right fist; `mean' = mean of the four EEGBCI paradigms within each subject; `LOSO 2a' = leave-one-subject-out on BCI-IV-2a. Each subject's value averages its test folds weighted by the number of its test trials. "
            r"Upper block: paired contrast and the two-sided Wilcoxon signed-rank test, Holm-adjusted over the 11 Wilcoxon tests of this table. Lower block: equivalence. TOST = two one-sided $t$-tests against the pre-specified margin $\pm0.01$; equivalent when both one-sided $p < 0.05$, i.e. when the 90\,\% CI lies inside the margin. The TOST decisions are read unadjusted, as pre-specified.}")

    def ref_label(r):
        short = "mean" if r["task"] == "paradigm_mean" else C.TASK_SHORT.get(r["task"], r["task"].replace("_", " "))
        if r["analysis"].startswith("e6"):
            return f"E6 {short}"
        return "E14 LOSO 2a" if r.get("dataset") == "bci2a" else f"E14 {short}"

    lines = [r"\begin{table}", cap5, r"\label{tab:reference}", r"\centering", TIGHT,
             r"\begin{tabular}{l c c c c}", r"\hline",
             r"Row & $n$ & mean $\Delta$ [95\,\% CI] & Wilcoxon $p$ & $p_{\rm Holm}$ \\", r"\hline"]
    for r in tost:
        ph = r.get("p_wilcoxon_holm", "")
        lines.append(f"{ref_label(r)} & {r['n']} & {cell(r['mean_delta'], r['ci_low'], r['ci_high'], nd=4, sign=True)} & "
                     f"{fp(float(r['p_wilcoxon']))} & {fp(float(ph)) if ph not in ('', 'nan') else '--'} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
              r"\multicolumn{4}{l}{\textit{Equivalence against the pre-specified margin $\pm0.01$}} \\", r"\hline",
              r"Row & 90\,\% CI (TOST) & TOST $p_{\rm lower}$, $p_{\rm upper}$ & equivalent \\", r"\hline"]
    for r in tost:
        lines.append(f"{ref_label(r)} & [{num(r['tost_ci90_low'], 4, True)}, {num(r['tost_ci90_high'], 4, True)}] & "
                     f"{fp(float(r['tost_p_lower']))}, {fp(float(r['tost_p_upper']))} & {'yes' if r['tost_equivalent'] in ('1', 'True', 'true') else 'no'} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "table3_reference.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Table 4 models (parameter counts from e9/e7 per-fold tables)
    params = {}
    for name in (f"e9_{args.tag}_t12_per_fold.csv", f"e9_{args.tag}_t34_per_fold.csv", f"e10_{args.tag}_per_fold.csv"):
        p = R / name
        if p.exists():
            prov.append(name)
            for r in C.read_csv(p):
                params[(r["model"], r.get("dataset", "eegbci"))] = int(r["n_params"])
    lines = [r"\begin{table}", r"\caption{Decoders and their settings. Feature dimension and trainable-parameter counts are given for EEGBCI (64 channels, 641 samples) and BCI-IV-2a (22 channels, 1001 samples). \rev{lbfgs = limited-memory Broyden--Fletcher--Goldfarb--Shanno quasi-Newton solver.} \rev{For EEGNet, $F_1$ counts temporal filters, $D$ is the depth multiplier (spatial filters learned per temporal filter), and $F_2$ counts pointwise filters.}}",
             r"\label{tab:models}", r"\centering", TIGHT, r"\begin{tabular}{l p{4.7cm} p{3.0cm} p{4.0cm}}", r"\hline", r"Decoder & Features / architecture & Fitted parameters (EEGBCI / BCI-IV-2a) & Training \\", r"\hline",
             r"CSP+LDA & 6 CSP log-variance features (Ledoit--Wolf regularized covariances) & LDA on 6 features: 7 / 7 & closed form \\",
             r"TS+LR & OAS covariance, tangent space at the training Fr\'echet mean; $N_c(N_c+1)/2$ = 2080 / 253 features & $\ell_2$ logistic regression: 2081 / 254 & lbfgs, max 3000 iterations, $C$ = 1 unless swept \\",
             f"ShallowFBCSPNet & braindecode defaults (40 temporal, 40 spatial filters, square, mean-pool, log) & {params.get(('shallow_fbcsp', 'eegbci'), '--')} / {params.get(('shallow_fbcsp', 'bci2a'), '--')} & Adam $10^{{-3}}$, batch 64, $\\le$40 epochs, early stopping (patience 6) \\\\",
             f"EEGNet & braindecode EEGNet ($F_1$ = 8, $D$ = 2, $F_2$ = 16) & {params.get(('eegnet_v4', 'eegbci'), '--')} / {params.get(('eegnet_v4', 'bci2a'), '--')} & as above \\\\",
             r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "table4_models.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: per-paradigm EEGBCI
    lines = [TS, r"\caption{EEGBCI per paradigm (109 subjects): mean accuracy [95\,\% CI] under each protocol (top) and the paired per-subject protocol contrasts (bottom). "
             r"The 48 contrasts displayed in this table form one Holm family, as the frozen rule requires (a family = the contrasts reported in one table); the $p_{\rm Holm}$ values are therefore adjusted over 48 tests and are larger than a per-paradigm adjustment would give. "
             r"MI = motor imagery, ME = motor execution.}", r"\label{tab:s_paradigm}", r"\centering", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{l l c c c c}", r"\hline", r"Paradigm & Decoder & within & P0 trial-random & P1 run-disjoint & P2 subject-disjoint \\", r"\hline"]
    for t in C.TASKS:
        for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
            row = [C.TASK_SHORT[t], MODEL_TEX[m]]
            for p in ("within_subject", "trial_random", "run_disjoint", "subject_disjoint"):
                r = M.get(("eegbci", t, m, p)); row.append(cell(r["mean_acc"], r["ci_low"], r["ci_high"]) if r else "--")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l l c c c}", r"\hline",
              r"\multicolumn{5}{l}{\textit{Paired per-subject protocol contrasts}: mean [95\,\% CI]} \\", r"\hline",
              r"Paradigm & Decoder & \shortstack{$\Delta$(P1$-$P2)\\run-disjoint $-$\\subject-disjoint\\(subject-overlap\\contrast)} & \shortstack{$\Delta$(P0$-$P1)\\trial-random $-$\\run-disjoint\\(run/temporal\\contrast)} & \shortstack{$\Delta_{\rm cal}$ = within$-$P2\\within-subject $-$\\subject-disjoint\\(calibration contrast)} \\", r"\hline"]
    for t in C.TASKS:
        for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
            row = [C.TASK_SHORT[t], MODEL_TEX[m]]
            for c in ("subject_overlap_component", "run_temporal_component", "calibration_gap"):
                r = K.get(("eegbci", t, m, c)); row.append(cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) if r else "--")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l l c c c}", r"\hline",
              r"\multicolumn{5}{l}{$p_{\rm Holm}$, Holm-adjusted over the 48 contrasts of this table} \\", r"\hline",
              r"Paradigm & Decoder & \shortstack{$\Delta$(P1$-$P2)\\run-disjoint $-$\\subject-disjoint\\(subject-overlap\\contrast)} & \shortstack{$\Delta$(P0$-$P1)\\trial-random $-$\\run-disjoint\\(run/temporal\\contrast)} & \shortstack{$\Delta_{\rm cal}$ = within$-$P2\\within-subject $-$\\subject-disjoint\\(calibration contrast)} \\", r"\hline"]
    for t in C.TASKS:
        for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
            row = [C.TASK_SHORT[t], MODEL_TEX[m]]
            for c in ("subject_overlap_component", "run_temporal_component", "calibration_gap"):
                r = K.get(("eegbci", t, m, c)); row.append(fp(float(r["p_holm"])) if r else "--")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "tableS_paradigm.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: matched preprocessing (E13) rows if present
    rows13 = [r for r in contr if r["task"] == "paradigm_mean" and (r["model"].endswith("_ztrial") or r["model"].endswith("_globalscale"))]
    if rows13:
        lines = [TS, r"\caption{Matched-preprocessing control (EEGBCI, paradigm mean): the classical decoders on per-trial z-scored trials instead of band-passed trials, and the networks on band-passed trials with one global scale factor instead of per-trial z-scoring. "
                 r"Seeds: the z-scored classical control uses five partition seeds for P0 and P2 (P1 is deterministic); the global-scale network control uses one seed. "
                 r"Paired per-subject differences with $p_{\rm Holm}$ Holm-adjusted over the 8 contrasts displayed in this table.}",
                 r"\label{tab:s_preproc}", r"\centering", TIGHT, r"\begin{tabular}{l c c c}", r"\hline", r"Decoder (preprocessing) & P2 subject-disjoint & $\Delta$(P1$-$P2) & $\Delta$(P0$-$P1) \\", r"\hline"]
        models13 = sorted({r["model"] for r in rows13})
        for m in models13:
            r2 = M.get(("eegbci", "paradigm_mean", m, "subject_disjoint"))
            row = [MODEL_TEX.get(m, m), cell(r2["mean_acc"], r2["ci_low"], r2["ci_high"]) if r2 else "--"]
            for c in ("subject_overlap_component", "run_temporal_component"):
                r = K.get(("eegbci", "paradigm_mean", m, c)); row.append(cell(r["mean_diff"], r["ci_low"], r["ci_high"], sign=True) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c}", r"\hline",
                  r"\multicolumn{3}{l}{$p_{\rm Holm}$ of the same differences (eight contrasts in this table)} \\", r"\hline",
                  r"Decoder (preprocessing) & $\Delta$(P1$-$P2) & $\Delta$(P0$-$P1) \\", r"\hline"]
        for m in models13:
            row = [MODEL_TEX.get(m, m)]
            for c in ("subject_overlap_component", "run_temporal_component"):
                r = K.get(("eegbci", "paradigm_mean", m, c)); row.append(fp(float(r["p_holm"])) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_preproc.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E5 classifiers
    p5 = R / f"stats_{args.tag}_e5_contrasts.csv"
    if p5.exists():
        prov.append(p5.name)
        rows = [r for r in C.read_csv(p5) if r["task"] == "paradigm_mean"]
        lines = [TS, r"\caption{Classifiers applied to identical inductive tangent-space features (EEGBCI, paradigm mean, 109 subjects; MDM operates on the covariances). Upper block: paired per-subject differences, mean [95\,\% CI]. Lower block: $p_{\rm Holm}$, Holm-adjusted over the 18 contrasts of this table. "
                 r"LR = $\ell_2$ logistic regression ($C$ = 1); SVM = linear support vector machine ($C$ = 1); ridge = ridge classifier ($\alpha$ = 1); $k$-NN with $k$ = 5; MDM = minimum distance to the Riemannian class means.}",
                 r"\label{tab:s_clf}", r"\centering", TIGHT, r"\begin{tabular}{l c c c}", r"\hline", r"Classifier & $\Delta$(P1$-$P2) & $\Delta$(P0$-$P1) & $\Delta$(P0$-$P2) \\", r"\hline"]
        lab = {"lr": "LR", "slda": "shrinkage LDA", "linsvm": "linear SVM", "ridge": "ridge", "knn": "$k$-NN", "mdm": "MDM"}
        for c in ("lr", "slda", "linsvm", "ridge", "knn", "mdm"):
            row = [lab[c]]
            for con in ("subject_overlap_component", "run_temporal_component", "delta_leak_original_definition"):
                r = [x for x in rows if x["clf"] == c and x["contrast"] == con]
                row.append(cell(r[0]["mean_diff"], r[0]["ci_low"], r[0]["ci_high"], sign=True) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
                  r"Classifier & $p_{\rm Holm}$ $\Delta$(P1$-$P2) & $p_{\rm Holm}$ $\Delta$(P0$-$P1) & $p_{\rm Holm}$ $\Delta$(P0$-$P2) \\", r"\hline"]
        for c in ("lr", "slda", "linsvm", "ridge", "knn", "mdm"):
            row = [lab[c]]
            for con in ("subject_overlap_component", "run_temporal_component", "delta_leak_original_definition"):
                r = [x for x in rows if x["clf"] == c and x["contrast"] == con]
                row.append(fp(float(r[0]["p_holm"])) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_classifiers.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E3c mean removal
    p3 = R / f"stats_{args.tag}_e3c_contrasts.csv"
    if p3.exists():
        prov.append(p3.name)
        rows = [r for r in C.read_csv(p3) if r["task"] == "paradigm_mean"]
        lines = [TS, r"\caption{TS+LR ($C$ = 1) after removing subject-level structure (EEGBCI, paradigm mean): per-subject feature-mean removal (each subject centred on its own mean over all its trials) and Riemannian recentering "
                 r"($C_i \mapsto G_s^{-1/2} C_i G_s^{-1/2}$ with $G_s$ the subject's Fr\'echet mean); both use all of a subject's trials, including its test trials, and are therefore transductive diagnostics. Paired differences, mean [95\,\% CI]; $p_{\rm Holm}$ Holm-adjusted over the six contrasts of this table.}", r"\label{tab:s_center}", r"\centering", TIGHT, r"\begin{tabular}{l c c c}", r"\hline",
                 r"Variant & $\Delta$(P1$-$P2) & $\Delta$(P0$-$P1) & $\Delta$(P0$-$P2) \\", r"\hline"]
        for v, lab_ in (("feature_centered", "subject feature-mean removed"), ("riemann_recentered", "Riemannian recentering")):
            row = [lab_]
            for con in ("subject_overlap_component", "run_temporal_component", "delta_leak_original_definition"):
                r = [x for x in rows if x["variant"] == v and x["contrast"] == con]
                row.append(cell(r[0]["mean_diff"], r[0]["ci_low"], r[0]["ci_high"], sign=True) if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
                  r"\multicolumn{4}{l}{$p_{\rm Holm}$ of the same differences (six contrasts in this table)} \\", r"\hline",
                  r"Variant & $\Delta$(P1$-$P2) & $\Delta$(P0$-$P1) & $\Delta$(P0$-$P2) \\", r"\hline"]
        for v, lab_ in (("feature_centered", "subject feature-mean removed"), ("riemann_recentered", "Riemannian recentering")):
            row = [lab_]
            for con in ("subject_overlap_component", "run_temporal_component", "delta_leak_original_definition"):
                r = [x for x in rows if x["variant"] == v and x["contrast"] == con]
                row.append(fp(float(r[0]["p_holm"])) if r else "--")
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
        nfam = rows[0].get("family_size", "") if rows else ""
        lines = [TS, rf"\caption{{{capt} Upper block: mean accuracy [95\,\% CI] per protocol. Lower block: paired per-subject differences, mean [95\,\% CI], with $p_{{\rm Holm}}$ Holm-adjusted over the {nfam} tests of this table, and Spearman's $\rho$ of $\Delta$(P1$-$P2) against the swept variable.}}", rf"\label{{tab:s_{label}}}", r"\centering", TIGHT,
                 r"\begin{tabular}{l c c c}", r"\hline", f"${xkey}$ & P0 trial-random & P1 run-disjoint & P2 subject-disjoint \\\\", r"\hline"]
        for r in rows:
            if "acc_trial_random" in r and r["acc_trial_random"]:
                lines.append(f"{float(r[xkey]):g} & {cell(r['acc_trial_random'], r['acc_trial_random_ci_low'], r['acc_trial_random_ci_high'])} & {cell(r['acc_run_disjoint'], r['acc_run_disjoint_ci_low'], r['acc_run_disjoint_ci_high'])} & "
                             f"{cell(r['acc_subject_disjoint'], r['acc_subject_disjoint_ci_low'], r['acc_subject_disjoint_ci_high'])} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\vspace{4pt}", r"\begin{tabular}{l c c}", r"\hline",
                  f"${xkey}$ & $\\Delta$(P1$-$P2) [CI]; $p_{{\\rm Holm}}$ & $\\Delta$(P0$-$P2) [CI]; $p_{{\\rm Holm}}$ \\\\", r"\hline"]
        for r in rows:
            if "acc_trial_random" in r and r["acc_trial_random"]:
                lines.append(f"{float(r[xkey]):g} & {cell(r['d12_mean'], r['d12_ci_low'], r['d12_ci_high'], sign=True)}; {fp(float(r.get('d12_p_holm') or r['d12_p']))} & "
                             f"{cell(r['d02_mean'], r['d02_ci_low'], r['d02_ci_high'], sign=True)}; {fp(float(r.get('d02_p_holm') or r['d02_p']))} \\\\")
            else:
                lines.append(rf"\multicolumn{{3}}{{l}}{{Spearman $\rho$($\Delta$(P1$-$P2), {xkey}) = {float(r['d12_mean']):.3f}, $p$ = {fp(float(r['d12_p']))} ($n$ = {r['n']} levels)}} \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / f"tableS_{label}.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E7 width, E8 factorial
    p7 = R / f"stats_{args.tag}_width.csv"
    if p7.exists():
        prov.append(p7.name)
        rows = [r for r in C.read_csv(p7) if r["task"] == "paradigm_mean"]
        lines = [TS, r"\caption{Network-width sweeps (EEGBCI, paradigm mean, one seed): trainable parameters, accuracy under P0 and P2, the paired per-subject difference (Holm-adjusted over the ten width cells of this table) and the mean train--test gap over folds.}", r"\label{tab:s_width}", r"\centering", TIGHT,
                 r"\begin{tabular}{l l r c c c c c}", r"\hline", r"Network & width & parameters & P0 & P2 & $\Delta$(P0$-$P2) [CI]; $p_{\rm Holm}$ & gap P0 & gap P2 \\", r"\hline"]
        for r in rows:
            if r["width"].startswith("spearman"):
                lines.append(rf"\multicolumn{{8}}{{l}}{{{MODEL_TEX.get(r['model'], r['model'])}: Spearman $\rho$($\Delta$(P0$-$P2), width) = {float(r['d02_mean']):.3f}, $p$ = {fp(float(r['d02_p']))}}} \\")
            else:
                lines.append(f"{MODEL_TEX.get(r['model'], r['model'])} & {r['width']} & {int(r['n_params']) if r['n_params'] else '--'} & {float(r['acc_P0']):.3f} & {float(r['acc_P2']):.3f} & {cell(r['d02_mean'], r['d02_ci_low'], r['d02_ci_high'], sign=True)}; {fp(float(r.get('d02_p_holm') or r['d02_p']))} & "
                             f"{float(r['train_test_gap_P0']):+.3f} & {float(r['train_test_gap_P2']):+.3f} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_width.tex").write_text("\n".join(lines) + "\n")
    p8 = R / f"stats_{args.factorial_tag or args.tag}_factorial.csv"
    if p8.exists():
        prov.append(p8.name)
        rows = C.read_csv(p8)
        task_labels = {**C.TASK_SHORT, "paradigm_mean": "Four-paradigm mean"}
        ols_counts = sorted({(r["task"], int(r["n_obs"])) for r in rows if r["N"] == "ols"})
        observation_counts = "; ".join(f"{task_labels.get(task, task)}: {n}" for task, n in ols_counts)
        excluded = {condition for r in rows for condition in r.get("excluded_conditions", "").split(";") if condition}
        excluded_groups = defaultdict(list)
        for condition in excluded:
            task, N, draw, d = condition.split(":")
            excluded_groups[(task, int(N), int(d))].append(int(draw))
        excluded_text = "; ".join(f"{C.TASK_SHORT.get(task, task)} ($N$ = {N}, $d$ = {d}, draws {', '.join(map(str, sorted(draws)))})"
                                  for (task, N, d), draws in sorted(excluded_groups.items())) or "none listed"
        pooled_counts = defaultdict(list)
        for r in rows:
            if r["task"] == "paradigm_mean" and r["model"] == "riemann_ts_lr" and r["N"] != "ols":
                pooled_counts[(int(r["N"]), int(r["n_draws"]))].append(int(r["d"]))
        pooled_text = "; ".join(f"$N$ = {N}, $d$ = {', '.join(map(str, sorted(ds)))}: {n} complete draw{'s' if n != 1 else ''}"
                                for (N, n), ds in sorted(pooled_counts.items()))
        lines = [r"\begin{center}", r"\footnotesize", TIGHT, r"\begin{longtable}{l l r r c}",
                 r"\caption{Within-dataset factorial (EEGBCI): mean protocol contrast $\Delta$(P0$-$P2) of TS+LR ($C$ = 1) over feasible random draws of $N$ sampled subjects (P2 trains on about $2N/3$ of them) and $d$ principal components, per paradigm, with draw-bootstrap 95\,\% CI (single-draw cells have no CI). "
                 r"Full dimension ($d$ = 2080) uses the raw tangent features. Infeasible dimensions are excluded from both protocols of a draw before scoring. "
                 r"A PCA dimension requires at least $d+1$ training trials in every fold of both protocols; full dimension is always feasible. "
                 r"CSP+LDA is also reported across $N$, with six log-variance features. "
                 f"Excluded conditions recorded in the statistics file: {excluded_text}. "
                 r"OLS rows: descriptive least-squares fit of the draw-level protocol contrast $\Delta$ on $\log_2 N$ and $\log_2 d$ centred at the grid means, with their product; CIs from a bootstrap that resamples whole subject draws within each $N$ (stratified by $N$, all available $d$ cells of a draw kept together). "
                 f"Draw-by-dimension observation counts (TS+LR): {observation_counts}. "
                 r"Four-paradigm means require all four paired paradigm values within a draw; whole complete draws are resampled together for their cell CIs. "
                 f"Four-paradigm TS+LR cell counts: {pooled_text}." + r"\label{tab:s_factorial}}\\",
                 r"\hline", r"Paradigm & model & $N$ & $d$ & $\Delta$(P0$-$P2) [CI] \\", r"\hline", r"\endfirsthead", r"\hline", r"Paradigm & model & $N$ & $d$ & $\Delta$(P0$-$P2) [CI] \\", r"\hline", r"\endhead"]
        for r in rows:
            if r["N"] == "ols":
                nm = {"intercept_at_grid_centre": "intercept (grid centre)", "intercept": "intercept", "log2N": "$\\log_2 N$ (per doubling)", "log2d": "$\\log_2 d$ (per doubling)", "log2N_x_log2d": "$\\log_2 N \\times \\log_2 d$"}.get(r["d"], r["d"])
                lines.append(f"{task_labels.get(r['task'], r['task'])} & OLS slope & \\multicolumn{{2}}{{l}}{{{nm}}} & {cell(r['inflation_mean'], r['ci_low'], r['ci_high'], sign=True)} \\\\")
            else:
                val = cell(r['inflation_mean'], r['ci_low'], r['ci_high'], sign=True) if r['ci_low'] not in ('nan', '') else f"{float(r['inflation_mean']):+.3f} (single draw)"
                lines.append(f"{task_labels.get(r['task'], r['task'])} & {MODEL_TEX.get(r['model'], r['model'])} & {r['N']} & {r['d']} & {val} \\\\")
        lines += [r"\hline", r"\end{longtable}", r"\end{center}"]
        (OUT / "tableS_factorial.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E15 exploratory stratification
    p15 = R / f"stats_{args.tag}_e15.csv"
    e15 = R / f"e15_{args.tag}_summary.csv"
    if e15.exists():
        prov.append(e15.name)
        rows = C.read_csv(e15)
        p0s = R / f"e15_{args.p0s_tag}_summary.csv"
        if p0s != e15:
            prov.append(p0s.name)
            rows += [r for r in C.read_csv(p0s) if r["protocol"] == "trial_random_sc"]
        agg = defaultdict(list)
        for r in rows:
            agg[(r["task"], r["protocol"])].append((float(r["subject_prior_accuracy_mean"]), float(r["corr_train_test_class_fraction"]) if r["corr_train_test_class_fraction"] not in ("nan", "") else float("nan")))
        lines = [TS, r"\caption{Exploratory (added after the ladder results were seen): a predictor that outputs the majority class of the subject's own training trials (global prior for subject-disjoint folds), and the correlation between a subject's training and test class-1 fractions across (subject, fold) cells. Means over partition seeds (five for P0 and P2, one for P1). "
                 r"\rev{P0s rows add the subject-by-class-stratified trial-random control (five seeds); their accuracy is the seed mean with the minimum and maximum over seeds in brackets, and their correlation is the seed mean.} "
                 r"`--' = not defined: subject-disjoint folds contain no training trials of the test subject, so that subject has no training class fraction to correlate.}",
                 r"\label{tab:s_e15}", r"\centering", TIGHT, r"\begin{tabular}{l l c c}", r"\hline", r"Paradigm & protocol & subject-prior accuracy & corr(train, test class fraction) \\", r"\hline"]
        for (task, protocol), v in sorted(agg.items(), key=lambda kv: (C.TASKS.index(kv[0][0]), kv[0][1])):
            correlations = [c for _, c in v if np.isfinite(c)]
            cval = np.mean(correlations) if correlations else float("nan")
            accuracy = f"{np.mean([a for a, _ in v]):.3f}"
            label = protocol.replace('_', ' ')
            correlation = ('%+.3f' % cval) if cval == cval else '--'
            cells = [C.TASK_SHORT[task], label, accuracy, correlation]
            if protocol == "trial_random_sc":
                cells[1] = "P0s"
                cells[2] += f" [{min(a for a, _ in v):.3f}, {max(a for a, _ in v):.3f}]"
                cells = [r"\rev{" + c + "}" for c in cells]
            lines.append(" & ".join(cells) + r" \\")
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
        lines = [TS, r"\caption{Network training details: mean over folds and seeds of the training accuracy (on all training trials, best-epoch weights), validation accuracy at the stopping epoch, test accuracy, stopping epoch and validation-set size; trainable parameters. Within-subject runs use a fixed 60 epochs (EEGBCI) without early stopping. This table reports settings and per-fold diagnostics, not subject-level estimates, and therefore carries no confidence intervals.}",
                 r"\label{tab:s_training}", r"\centering", TIGHT, r"\begin{tabular}{l l l c c c c c r}", r"\hline", r"Dataset & network & protocol & train acc & val acc & test acc & stop epoch & $n_{\rm val}$ & params \\", r"\hline"]
        for (dataset, model, protocol), v in sorted(agg.items()):
            a = np.array(v, dtype=float)
            lines.append(f"{dataset} & {MODEL_TEX.get(model, model)} & {protocol.replace('_', ' ')} & {a[:, 0].mean():.3f} & {np.nanmean(a[:, 1]):.3f} & {a[:, 2].mean():.3f} & {a[:, 3].mean():.1f} & {a[:, 4].mean():.0f} & {int(a[0, 5])} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_training.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary: E12 calibration curve
    p12 = R / f"stats_{args.tag}_e12_calibration.csv"
    if p12.exists():
        prov.append(p12.name)
        rows = C.read_csv(p12)
        lines = [TS, r"\caption{Calibration-size curve on the two imagery paradigms of EEGBCI (mean over both; 109 test subjects, evaluated on each subject's third run): accuracy of a subject-disjoint model after appending $k$ labelled trials of the test subject (CSP+LDA and TS+LR refitted; EEGNet fine-tuned for 20 epochs; three draws averaged), and the paired difference to $k = 0$ with exact/asymptotic Wilcoxon $p$ Holm-adjusted over the 12 contrasts of this table. "
                 r"$k$ is a \emph{cap}, not an exact count: 20 of the 218 subject-paradigm cells have fewer than 30 labelled trials in their first two runs, so the nominal $k = 30$ row uses 24--30 trials (the per-cell counts are released). "
                 r"The within-subject reference row is the leave-one-run-out accuracy of the same decoder on the same paradigms; it uses all three runs under a different protocol, is therefore not matched to the calibration rows, and is shown for orientation only. The subject-disjoint (P2) reference row is the same decoder under the ladder's P2 protocol. `--' = not applicable: $k = 0$ is the baseline of the difference column, and the two reference rows are not calibration points.}",
                 r"\label{tab:s_calib}", r"\centering", TIGHT, r"\begin{tabular}{l l c c}", r"\hline", r"Decoder & $k$ & accuracy [95\,\% CI] & $\Delta$ vs $k = 0$ [CI]; $p_{\rm Holm}$ \\", r"\hline"]
        for r in rows:
            kk = {"ref_within_subject": "within-subject reference", "ref_subject_disjoint": "subject-disjoint (P2) reference"}.get(r["k"], r["k"])
            diff = (cell(r["diff_vs_k0"], r["diff_ci_low"], r["diff_ci_high"], sign=True) + "; " + fp(float(r["p_holm"]))) if r["diff_vs_k0"] not in ("", ) and r["k"] not in ("0",) and not r["k"].startswith("ref") else "--"
            lines.append(f"{MODEL_TEX.get(r['model'], r['model'])} & {kk} & {cell(r['mean_acc'], r['ci_low'], r['ci_high'])} & {diff} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_calibration.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------- Supplementary S12: original five-fold subject-grouped split vs LOSO
    g5 = [r for r in contr if r["dataset"] == "bci2a" and r["contrast"] == "groupkfold5_vs_loso"]
    if g5:
        lines = [TS, r"\caption{BCI-IV-2a: the five-fold subject-grouped split of the original submission versus leave-one-subject-out (LOSO), for the four decoders (9 subjects; networks averaged over three seeds, classical decoders over five partition seeds for the five-fold split). "
                 r"Both splits are subject-disjoint; they differ in the number of test subjects per fold (about two versus one) and hence in the training fraction (about 7/9 versus 8/9). "
                 r"$p$ is the exact conditional two-sided signed-rank test over the nine subjects and is \emph{descriptive}: this comparison appears in no Holm family and is reported unadjusted; the subject-bootstrap 95\,\% CIs are likewise descriptive and unadjusted, not a substitute for multiplicity-adjusted confirmation.}",
                 r"\label{tab:s_groupkfold}", r"\centering", TIGHT, r"\begin{tabular}{l c c c c}", r"\hline",
                 r"Decoder & five-fold subject-grouped [95\,\% CI] & LOSO [95\,\% CI] & difference [95\,\% CI] & raw $p$; $n_+$/9 \\", r"\hline"]
        for m in ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4"):
            k5 = [x for x in g5 if x["model"] == m]
            a = M.get(("bci2a", "bci2a_left_right", m, "subject_disjoint_5"))
            b = M.get(("bci2a", "bci2a_left_right", m, "loso"))
            if not (k5 and a and b):
                continue
            r = k5[0]
            lines.append(f"{MODEL_TEX[m]} & {cell(a['mean_acc'], a['ci_low'], a['ci_high'])} & {cell(b['mean_acc'], b['ci_low'], b['ci_high'])} & "
                         f"{cell(r['mean_diff'], r['ci_low'], r['ci_high'], sign=True)} & {fp(float(r['p_wilcoxon']))}; {r['n_positive']}/9 \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (OUT / "tableS_groupkfold.tex").write_text("\n".join(lines) + "\n")

    (OUT / "tables_provenance.txt").write_text("\n".join(f"results/revision/{p}" for p in prov) + "\n")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
