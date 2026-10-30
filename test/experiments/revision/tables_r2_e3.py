#!/usr/bin/env python3
"""Generate the E3-v2 supplementary tables from assignment/per-subject CSVs.

The default destination is manuscript/v2/tables; --out-dir permits a scratch
render without changing manuscript files. No manuscript source is edited.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import e3v2_subject_info as E  # noqa: E402

TS = r"\begin{table}[H]"
TIGHT = r"\setlength{\tabcolsep}{3pt}"
TERM_TEX = {
    "subject": "subject",
    "class_given_subject": r"class $\mid$ subject",
    "class": "class",
    "subject_given_class": r"subject $\mid$ class",
    "subject_x_class_given_both": r"subject $\times$ class $\mid$ both",
    "residual": "residual",
}


def num(value, places=3):
    text = f"{float(value):.{places}f}"
    return f"{0:.{places}f}" if float(text) == 0 else text


def component_num(value):
    return num(value, 4 if abs(float(value)) < 0.001 else 3)


def provenance(paths):
    return [f"% Input: {p}; sha256={C.sha256_file(p)}" for p in paths]


def identity_table(assignment_path):
    subject_path = assignment_path.with_name(assignment_path.stem + "_per_subject.csv")
    assignment_rows = C.read_csv(assignment_path)
    summaries = E.summarize_identity(C.read_csv(subject_path))
    tasks = sorted({r["task"] for r in assignment_rows}, key=C.TASKS.index)
    if {r["task"] for r in summaries} != set(tasks):
        raise ValueError("assignment and subject CSV paradigms differ")
    lines = provenance([assignment_path, subject_path]) + [TS,
        r"\caption{EEGBCI subject-identity probe: multinomial logistic regression ($C=1$) trained on one run and tested on another, "
        r"with the remaining run used as a centring reference. The tangent reference is the Fr\'echet mean of training-run covariances; "
        r"reference-run features therefore also depend on this training-fitted mean. Schemes: (i) no centring; (ii) the subject's reference-run "
        r"mean subtracted from both training and test; (iii) the subject's training-run mean subtracted from both; "
        r"(iv) the subject's all-three-run mean subtracted from both (the revision-1 construction); (v) alternating reference-run "
        r"trials form half A for training centring and half B for test centring. Accuracy is averaged over the six assignments within "
        r"subject and then over subjects. The range is the minimum and maximum per-assignment macro accuracy, not a confidence interval. "
        r"The separate 95\,\% CI resamples subjects (2000 resamples, seed 20260916); assignments are not independent units. "
        r"Pooled accuracy weights all test-trial appearances; each trial is tested twice across assignments. "
        r"The paired difference (ii)$-$(v) estimates the identity signal that a shared estimated centring vector introduces. \rev{It changes two things at once, whether the centring vector is shared between training and test and how many trials estimate each vector (a whole run versus half a run), under the assumption that the trials of a run are independent given the subject; it is a diagnostic contrast, not an isolated attribution.} "
        r"Scheme (iv) obeys a three-run weighted-zero relation; the original revision-1 two-subset probe forces centred training and test "
        r"centroids in opposite directions and can be below chance. These linear-probe scores do not establish removal of identity information.}",
        r"\label{tab:s_e3a2}", r"\centering", TIGHT, r"\begin{tabular}{l l c c c}", r"\hline",
        r"Paradigm / subjects; chance & scheme & mean [min, max] & 95\,\% CI & pooled \\", r"\hline"]
    lookup = {(r["task"], r["scheme"]): r for r in summaries}
    for task in tasks:
        for i, scheme in enumerate(E.SCHEMES):
            r = lookup[(task, scheme)]
            label = f"{C.TASK_SHORT[task]} / {r['n_subjects']}; {num(r['chance'])}" if i == 0 else ""
            mean_range = f"{num(r['mean'])} [{num(r['assignment_min'])}, {num(r['assignment_max'])}]"
            ci = f"[{num(r['ci_low'])}, {num(r['ci_high'])}]"
            lines.append(f"{label} & {E.SCHEME_LABELS[scheme].split(')')[0]}) & {mean_range} & {ci} & {num(r['pooled_trial_accuracy'])}" + r" \\")
        d = lookup[(task, "independent_reference_minus_split_reference")]
        lines.append(f" & (ii)$-$(v) & {num(d['mean'])} & [{num(d['ci_low'])}, {num(d['ci_high'])}] & --" + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def variance_table(path, legacy_path):
    rows = C.read_csv(path)
    tasks = sorted({r["task"] for r in rows}, key=C.TASKS.index)
    lookup = {(r["task"], r["order"], r["component"]): r for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(tasks) * sum(map(len, E.ORDER_COMPONENTS.values())):
        raise ValueError("duplicate or incomplete sequential SS rows")
    for task in tasks:
        for order, names in E.ORDER_COMPONENTS.items():
            values = [float(lookup[(task, order, name)]["fraction"]) for name in names]
            if min(values) < -E.REL_TOL or abs(sum(values) - 1) > E.REL_TOL:
                raise ValueError("invalid sequential fractions")
    lines = provenance([path] + ([legacy_path] if legacy_path else [])) + [TS,
        r"\caption{EEGBCI sequential tangent-feature sums of squares as fractions of total SS, summed over dimensions. "
        r"All trials share the tangent space whose reference is the all-trial Fr\'echet covariance mean (descriptive). "
        r"Every model is ordinary trial-weighted least squares with an intercept: RSS0 is grand-mean residual SS; RSS$_S$ subject-only; "
        r"RSS$_C$ class-only; RSS$_{SC}$ additive subject and class; RSS$_{\rm cell}$ subject-by-class cell means. "
        r"Order A (subject $\to$ class $\to$ interaction) uses RSS0$-$RSS$_S$, RSS$_S-$RSS$_{SC}$, "
        r"RSS$_{SC}-$RSS$_{\rm cell}$ and RSS$_{\rm cell}$. Order B (class $\to$ subject $\to$ interaction) uses "
        r"RSS0$-$RSS$_C$, RSS$_C-$RSS$_{SC}$ and the same interaction and residual terms. "
        r"Each is divided by RSS0. The order-A subject term is the marginal between-subject sum of squares. "
        r"These are sequential projection increments, not uniquely separated variance sources or subject-level accuracy quantities. "
        + (r"The revision-1 marginal-subtraction values are retained below for comparison; they are not an orthogonal decomposition. " if legacy_path else "")
        + r"All displayed values are read from the released CSVs.}",
        r"\label{tab:s_e3b2}", r"\centering", TIGHT,
        r"\begin{tabular}{l l " + "c " * len(tasks) + "}", r"\hline",
        "Order & SS term & " + " & ".join(C.TASK_SHORT[t] for t in tasks) + r" \\", r"\hline"]
    for order, names in E.ORDER_COMPONENTS.items():
        for name in names:
            lines.append(f"{order} & {TERM_TEX[name]} & " + " & ".join(component_num(lookup[(t, order, name)]["fraction"]) for t in tasks) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}"]
    if legacy_path:
        legacy = {r["task"]: r for r in C.read_csv(legacy_path) if r["space"] == "common_tangent"}
        lines += [r"\vspace{4pt}", r"\begin{tabular}{l " + "c " * len(tasks) + "}", r"\hline",
                  "Revision-1 marginal subtraction & " + " & ".join(C.TASK_SHORT[t] for t in tasks) + r" \\", r"\hline"]
        for name, label in (("frac_subject", "subject"), ("frac_class", "class"),
                            ("frac_subject_x_class", r"subject $\times$ class"), ("frac_residual", "residual")):
            lines.append(label + " & " + " & ".join(component_num(legacy[t][name]) for t in tasks) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
    lines += [r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="rev2")
    ap.add_argument("--e3a2", type=Path)
    ap.add_argument("--e3b2", type=Path)
    ap.add_argument("--e3b", type=Path, default=C.RESULTS / "e3b_rev1.csv")
    ap.add_argument("--out-dir", type=Path, default=C.ROOT / "manuscript" / "v2" / "tables")
    args = ap.parse_args()
    a = args.e3a2 or C.RESULTS / f"e3a2_{args.tag}.csv"
    b = args.e3b2 or C.RESULTS / f"e3b2_{args.tag}.csv"
    texts = {"tableS_e3a2.tex": identity_table(a), "tableS_e3b2.tex": variance_table(b, args.e3b)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        path = args.out_dir / name
        path.write_text(text, encoding="utf-8")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
