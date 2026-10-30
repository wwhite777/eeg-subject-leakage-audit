#!/usr/bin/env python3
"""Generate the E16/E17 tables and rev2 training details from released CSV values.

Use --out-dir to render fixture/smoke tables in private workspace scratch.
The default destination is manuscript/v3/tables; this script changes no prose.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import e_stats_r2 as S  # noqa: E402

MODEL_TEX = {"csp_lda": "CSP+LDA", "riemann_ts_lr": "TS+LR",
             "shallow_fbcsp": "ShallowFBCSPNet", "eegnet_v4": "EEGNet"}
CONTRAST_TEX = {
    "p0s_minus_p1": r"$\Delta$(P0s$-$P1) protocol contrast",
    "p0_minus_p0s": r"$\Delta$(P0$-$P0s) protocol contrast",
    "p0s_minus_p2": r"$\Delta$(P0s$-$P2) protocol contrast",
    "p0as_minus_p1": r"$\Delta$(P0a-s$-$P1) protocol contrast",
    "p0bs_minus_p1s": r"$\Delta$(P0b-s$-$P1s) protocol contrast",
    "p0a_minus_p0as": r"$\Delta$(P0a$-$P0a-s) protocol contrast",
    "p0b_minus_p0bs": r"$\Delta$(P0b$-$P0b-s) protocol contrast",
    "p0as_minus_p2": r"$\Delta$(P0a-s$-$P2) protocol contrast",
    "p0bs_minus_p2": r"$\Delta$(P0b-s$-$P2) protocol contrast",
    "target_inclusion": r"$\Delta_{\rm incl}$ target-inclusion contrast",
    "same_run_adjacency": r"$\Delta_{\rm adj}$ same-run adjacency contrast",
    "inclusion_minus_adjacency": r"$\Delta_{\rm incl} - \Delta_{\rm adj}$ paired difference (descriptive)",
}
PROTOCOL_TEX = {"trial_random_sc": "P0s", "run_disjoint": "P1", "trial_random": "P0",
                "subject_disjoint": "P2", "trial_random_6_sc": "P0a-s", "trial_random_2_sc": "P0b-s",
                "session_disjoint": "P1s", "trial_random_6": "P0a", "trial_random_2": "P0b", "loso": "P2 (LOSO)"}


def num(value, sign=False):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite table value: {value}")
    if abs(value) < 0.0005:
        value = 0.0
        sign = False
    return f"{value:+.3f}" if sign else f"{value:.3f}"


def interval(row, sign=False):
    return f"{num(row['mean'], sign)} [{num(row['ci_low'], sign)}, {num(row['ci_high'], sign)}]"


def fp(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid table p value: {value}")
    if value < 1e-3 and value:
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{value:.3f}"


def task_label(task):
    return "Paradigm mean" if task == "paradigm_mean" else C.TASK_SHORT.get(task, "L/R imagery")


def contrast_label(name, dataset):
    if name == "same_run_adjacency" and dataset == "bci2a":
        return r"$\Delta_{\rm adj}$ same-session adjacency contrast"
    return CONTRAST_TEX[name]


def begin(caption, label, columns, header, supplement=False):
    if supplement:
        return [r"\begingroup", r"\setlength{\tabcolsep}{3pt}", r"\small",
                r"\begin{longtable}{" + columns + "}",
                r"\caption{" + caption + r"}\label{" + label + r"} \\",
                r"\hline", header + r" \\", r"\hline", r"\endfirsthead",
                r"\hline", header + r" \\", r"\hline", r"\endhead"]
    return [r"\begin{table}",
            r"\caption{" + caption + "}", r"\label{" + label + "}", r"\centering",
            r"\setlength{\tabcolsep}{3pt}", r"\small", r"\begin{tabular}{" + columns + "}",
            r"\hline", header + r" \\", r"\hline"]


def end(lines):
    if lines[0] == r"\begingroup":
        return lines + [r"\hline", r"\end{longtable}", r"\endgroup"]
    return lines + [r"\hline", r"\end{tabular}", r"\end{table}"]


def index_unique(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in out:
            raise ValueError(f"duplicate statistics group: {key}")
        out[key] = row
    return out


def main_table(M, K):
    family_rows = [r for r in K.values() if r["family"] == S.FAMILY]
    S._check(len(family_rows) == len(S.DECODERS) * len(S.PRIMARY), "incomplete main-table Holm family")
    S._check(all(int(r["family_size"]) == len(family_rows) and r["p_holm"] != "" for r in family_rows),
             "invalid main-table Holm adjustment")
    n_values = {int(r["n"]) for r in family_rows}
    S._check(len(n_values) == 1, "different subject counts across main-table contrasts")
    n = next(iter(n_values))
    caption = (f"EEGBCI matched controls ({n} subjects; paradigms averaged within subject after averaging seeds \\rev{{(five partition seeds for the classical decoders, 20260916--20260920; three for the networks, 20260916--20260918)}}). "
               r"Accuracy and paired contrasts are means over subjects with subject-bootstrap 95\,\% CIs. "
               r"P0s is the subject-by-class-stratified trial-random protocol; P1 is run-disjoint. "
               r"The fixed test is half of the last run, alternating within subject and class. "
               r"All arms share a background of other subjects' first two runs; target and donor subjects are paired "
               r"by trial-count rank and matched within run and class. A0 adds matched donors, A1 adds targets' other-run trials, "
               r"and A2 replaces the matched latter-run trials with the held-in half of the targets' last run. "
               r"Any cell shortfalls are recorded in the released matching and count tables \rev{(none occurred in either dataset)}. "
               r"TS+LR uses one common tangent reference estimated from the background, and networks use one common "
               r"class-stratified validation slice from that background, removed from every optimization set. "
               r"The target-inclusion contrast is A1$-$A0; the same-run adjacency contrast is A2$-$A1. "
               r"The P0s$-$P1 protocol contrast uses each protocol's own training reference and validation slice. "
               f"Two-sided paired Wilcoxon $p_{{\\rm Holm}}$ values use one family of {len(family_rows)} contrasts.")
    lines = begin(caption, "tab:matched", "l c c", "Decoder & P0s & P1")
    for decoder in S.DECODERS:
        row = [MODEL_TEX[decoder]]
        for protocol in ("trial_random_sc", "run_disjoint"):
            row.append(interval(M["eegbci", "paradigm_mean", decoder, "protocol_own", protocol, ""]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\vspace{4pt}", r"\begin{tabular}{l c c c}", r"\hline",
              r"Decoder & A0 & A1 & A2 \\", r"\hline"]
    for decoder in S.DECODERS:
        row = [MODEL_TEX[decoder]]
        for arm in S.ARMS:
            row.append(interval(M["eegbci", "paradigm_mean", decoder, "bg_fixed", "fixed_test", arm]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\vspace{4pt}", r"\begin{tabular}{l l c c}", r"\hline",
              r"Decoder & Contrast & Mean [95\,\% CI] & $p_{\rm Holm}$ \\", r"\hline"]
    for decoder in S.DECODERS:
        for name in S.PRIMARY:
            reference = "protocol_own" if name == "p0s_minus_p1" else "bg_fixed"
            r = K["eegbci", "paradigm_mean", decoder, reference, name]
            lines.append(" & ".join((MODEL_TEX[decoder], CONTRAST_TEX[name], interval(r, True), fp(r["p_holm"]))) + r" \\")
        r = K["eegbci", "paradigm_mean", decoder, "bg_fixed", "inclusion_minus_adjacency"]
        lines.append(" & ".join((MODEL_TEX[decoder], CONTRAST_TEX["inclusion_minus_adjacency"], interval(r, True), "--")) + r" \\")
    return end(lines)


def bci_table(M, K):
    caption = (r"BCI-IV-2a matched controls: descriptive, unadjusted subject-bootstrap 95\,\% CIs and exact "
               r"conditional two-sided signed-rank $p$ values; these results are not confirmatory. "
               r"P0a-s and P0b-s stratify trial-random folds by subject and class; their comparators are P1 (run-disjoint) "
               r"and P1s (session-disjoint). The fixed test is alternating half h1 of the last session within subject and class. "
               r"A common background uses the background subjects' first session; target and donor subjects are paired by "
               r"trial-count rank and class counts are matched. A0 adds donors, A1 adds targets' first-session trials, "
               r"and A2 substitutes half h2 of the targets' last session. TS+LR uses a common background reference; "
               r"networks share one background validation slice. The adjacency contrast is same-session adjacency. "
               r"$n_+/n$ reports subjects with a positive contrast divided by the paired subject count.")
    lines = begin(caption, "tab:s_matched_bci2a", "l l c c c",
                  r"Decoder & Protocol, arm or contrast & Mean [95\,\% CI] & Exact $p$ & $n_+/n$", True)
    protocols = ("trial_random_6_sc", "run_disjoint", "trial_random_2_sc", "session_disjoint")
    for decoder in S.DECODERS:
        for protocol in protocols:
            r = M["bci2a", "bci2a_left_right", decoder, "protocol_own", protocol, ""]
            lines.append(" & ".join((MODEL_TEX[decoder], PROTOCOL_TEX[protocol], interval(r), "", "")) + r" \\")
        for arm in S.ARMS:
            r = M["bci2a", "bci2a_left_right", decoder, "bg_fixed", "fixed_test", arm]
            lines.append(" & ".join((MODEL_TEX[decoder], arm, interval(r), "", "")) + r" \\")
    lines += [r"\hline",
              r"Decoder & Contrast & Mean [95\,\% CI] & Exact $p$ & $n_+/n$ \\", r"\hline"]
    for decoder in S.DECODERS:
        for _, _, name in S.LADDER["bci2a"]:
            r = K["bci2a", "bci2a_left_right", decoder, "protocol_own", name]
            lines.append(" & ".join((MODEL_TEX[decoder], contrast_label(name, "bci2a"), interval(r, True), fp(r["p_wilcoxon"]), f"{r['n_positive']}/{r['n']}")) + r" \\")
        for name in ("target_inclusion", "same_run_adjacency", "inclusion_minus_adjacency"):
            r = K["bci2a", "bci2a_left_right", decoder, "bg_fixed", name]
            lines.append(" & ".join((MODEL_TEX[decoder], contrast_label(name, "bci2a"), interval(r, True), fp(r["p_wilcoxon"]), f"{r['n_positive']}/{r['n']}")) + r" \\")
    return end(lines)


def paradigm_table(M, K):
    caption = (r"EEGBCI per-paradigm matched controls and supplementary protocol contrasts. "
               r"Each subject's seed-mean accuracy precedes subject aggregation; CIs are descriptive, unadjusted "
               r"subject-bootstrap 95\,\% intervals. A0, A1 and A2 use the same fixed half-last-run test, background and "
               r"trial-count-rank donor pairing; TS+LR uses the common background reference and networks the common "
               r"background validation slice. P0s$-$P1, P0$-$P0s and P0s$-$P2 are protocol contrasts. "
               r"The lower block also reports the paradigm-mean supplementary protocol contrasts; only the "
               r"paradigm-mean primary contrasts \rev{in table 6 of the main text} belong to its Holm family.")
    lines = begin(caption, "tab:s_matched_per_paradigm", "l l l c c",
                  r"Paradigm & Decoder & Protocol, arm or contrast & Mean [95\,\% CI] & Raw $p$", True)
    for task in C.TASKS:
        for decoder in S.DECODERS:
            for protocol in ("trial_random_sc", "run_disjoint"):
                r = M["eegbci", task, decoder, "protocol_own", protocol, ""]
                lines.append(" & ".join((task_label(task), MODEL_TEX[decoder], PROTOCOL_TEX[protocol], interval(r), "")) + r" \\")
            for arm in S.ARMS:
                r = M["eegbci", task, decoder, "bg_fixed", "fixed_test", arm]
                lines.append(" & ".join((task_label(task), MODEL_TEX[decoder], arm, interval(r), "")) + r" \\")
    lines += [r"\hline",
              r"Paradigm & Decoder & Contrast & Mean [95\,\% CI] & Raw $p$ \\", r"\hline"]
    for task in C.TASKS + ["paradigm_mean"]:
        for decoder in S.DECODERS:
            names = ("p0s_minus_p1", "p0_minus_p0s", "p0s_minus_p2", "target_inclusion", "same_run_adjacency", "inclusion_minus_adjacency")
            if task == "paradigm_mean":
                names = ("p0_minus_p0s", "p0s_minus_p2")
            for name in names:
                reference = "bg_fixed" if name in ("target_inclusion", "same_run_adjacency", "inclusion_minus_adjacency") else "protocol_own"
                r = K["eegbci", task, decoder, reference, name]
                cells = (task_label(task), MODEL_TEX[decoder], CONTRAST_TEX[name], interval(r, True), fp(r["p_wilcoxon"]))
                if name == "inclusion_minus_adjacency":
                    cells = tuple(r"\rev{" + c + "}" for c in cells)
                lines.append(" & ".join(cells) + r" \\")
    return end(lines)


def fullpipeline_table(M, K):
    caption = (r"TS+LR full-pipeline variant: each arm's own training covariances determine its tangent reference, "
               r"rather than the common background reference of the primary analysis. Fixed tests, background, donor "
               r"pairing and count-matched arms are unchanged. Accuracy and paired contrasts have descriptive, unadjusted "
               r"subject-bootstrap 95\,\% CIs and raw paired signed-rank $p$ values (exact conditional for BCI-IV-2a). "
               r"BCI-IV-2a values are descriptive and unadjusted; they are not confirmatory. "
               r"Paradigm means average paradigms within subject after averaging the subject's tested seeds.")
    lines = begin(caption, "tab:s_e17_fullpipeline", "l l l c c c",
                  r"Dataset & Paradigm & Arm or contrast & Mean [95\,\% CI] & Raw $p$ & $n_+/n$", True)
    for dataset in ("eegbci", "bci2a"):
        for task in S.tasks_for(dataset) + (["paradigm_mean"] if dataset == "eegbci" else []):
            for arm in S.ARMS:
                r = M[dataset, task, "riemann_ts_lr", "arm_own", "fixed_test", arm]
                lines.append(" & ".join(("EEGBCI" if dataset == "eegbci" else "BCI-IV-2a", task_label(task), arm, interval(r), "", "")) + r" \\")
    lines += [r"\hline",
              r"Dataset & Paradigm & Contrast & Mean [95\,\% CI] & Raw $p$ & $n_+/n$ \\", r"\hline"]
    for dataset in ("eegbci", "bci2a"):
        for task in S.tasks_for(dataset) + (["paradigm_mean"] if dataset == "eegbci" else []):
            for name in ("target_inclusion", "same_run_adjacency"):
                r = K[dataset, task, "riemann_ts_lr", "arm_own", name]
                lines.append(" & ".join(("EEGBCI" if dataset == "eegbci" else "BCI-IV-2a", task_label(task), contrast_label(name, dataset), interval(r, True), fp(r["p_wilcoxon"]), f"{r['n_positive']}/{r['n']}")) + r" \\")
    return end(lines)


def int_range(rows, name):
    values = [int(r[name]) for r in rows]
    return str(min(values)) if min(values) == max(values) else f"{min(values)}--{max(values)}"


def training_table(rows):
    """Report logged fits, keeping protocols and fixed-test arms separate."""
    groups, unique = defaultdict(list), set()
    for r in rows:
        run_type = r["arm"] if r["protocol"] == "fixed_test" else PROTOCOL_TEX[r["protocol"]]
        key = tuple(r.get(k, "") for k in ("dataset", "task", "model", "protocol", "arm", "seed", "fold", "rotation"))
        S._check(key not in unique, f"duplicate training log: {key}")
        unique.add(key)
        S._check(int(r["n_fit"]) + int(r["n_val"]) == int(r["n_train"]), f"fit/validation count mismatch: {key}")
        S._check(1 <= int(r["selected_epoch"]) <= int(r["epochs_run"]) <= 40, f"invalid epoch range: {key}")
        groups[r["dataset"], r["model"], run_type].append(r)
    caption = (r"\rev{Second-revision network training details from the released per-fold logs. "
               r"Ranges cover all folds and seeds, and all four paradigms for EEGBCI; each paradigm--fold "
               r"(or paradigm--rotation for fixed tests) counts once per seed in the fold count. "
               r"P0s, P0a-s and P0b-s are the subject-by-class-stratified protocols; A0, A1 and A2 are "
               r"the fixed-test arms. $n_{\rm fit}$ excludes the validation trials and $n_{\rm val}$ is "
               r"the validation size. All runs use at most 40 epochs and stop after six consecutive "
               r"epochs without a strict increase in validation accuracy (ties count as no improvement); "
               r"the selected epoch is the first epoch attaining the best validation accuracy, whose "
               r"weights are restored. Fixed-test arms share the background validation slice and paired "
               r"training seeds. These diagnostics are descriptive and have no confidence intervals.}")
    header = (r"\rev{Run type} & \rev{Folds $\times$ seeds} & \rev{$n_{\rm fit}$} & "
              r"\rev{$n_{\rm val}$} & \rev{Selected epoch} & \rev{Epochs run}")
    lines = begin(caption, "tab:s_rev2_training", "l c c c c c", header, True)
    for dataset in ("eegbci", "bci2a"):
        for model in ("shallow_fbcsp", "eegnet_v4"):
            label = ("EEGBCI" if dataset == "eegbci" else "BCI-IV-2a") + " / " + MODEL_TEX[model]
            lines.append(r"\multicolumn{6}{l}{\rev{" + label + r"}} \\")
            run_types = ("P0s", "A0", "A1", "A2") if dataset == "eegbci" else ("P0a-s", "P0b-s", "A0", "A1", "A2")
            for run_type in run_types:
                values = groups[dataset, model, run_type]
                S._check(bool(values), f"missing training group: {dataset}, {model}, {run_type}")
                by_seed = defaultdict(set)
                for r in values:
                    by_seed[r["seed"]].add((r["task"], r["fold"], r.get("rotation", "")))
                fold_sets = list(by_seed.values())
                S._check(all(s == fold_sets[0] for s in fold_sets), f"unbalanced fold/seed grid: {dataset}, {model}, {run_type}")
                cells = [run_type, f"{len(fold_sets[0])} $\\times$ {len(by_seed)}"]
                cells += [int_range(values, name) for name in ("n_fit", "n_val", "selected_epoch", "epochs_run")]
                lines.append(" & ".join(r"\rev{" + c + "}" for c in cells) + r" \\")
    return end(lines)


def training_inputs(results_dir, tag):
    paths = [results_dir / f"e9_{tag}_{part}_per_fold.csv" for part in ("t12", "t34")]
    paths += [results_dir / f"e10_{tag}_per_fold.csv"]
    paths += [results_dir / f"e9_{tag}_fixed_{part}_per_fold.csv" for part in ("t12", "t34")]
    paths += [results_dir / f"e10_{tag}_fixed_per_fold.csv"]
    rows = []
    for path in paths:
        for r in C.read_csv(path):
            if r["protocol"] in ("trial_random_sc", "trial_random_6_sc", "trial_random_2_sc", "fixed_test"):
                rows.append(dict(r, dataset=r.get("dataset") or ("eegbci" if path.name.startswith("e9_") else "bci2a")))
    return rows, paths


def construction_table(rows):
    groups, unique = defaultdict(list), set()
    for r in rows:
        key = tuple(r[k] for k in ("dataset", "task", "protocol", "seed", "fold", "subject", "run", "class"))
        S._check(key not in unique, f"duplicate P0s construction count: {key}")
        unique.add(key)
        for column in ("train", "test"):
            S._check(int(r[f"n_{column}"]) - int(r[f"p1_n_{column}"]) == int(r[f"delta_{column}"]),
                     f"incorrect released P1 count difference: {key}, {column}")
        S._check(int(r["n_train"]) + int(r["n_test"]) == int(r["n_total"]), f"construction counts do not complement: {key}")
        groups[r["dataset"], r["task"], r["protocol"], r["run"], r["class"]].append(r)
    caption = (r"P0s construction audit: ranges of per-subject, per-class training and test counts across partition seeds "
               r"and folds, shown separately for each physical run and for all runs. P0s count differences are relative to "
               r"P1's actual held-out run in the corresponding fold (P1s's actual held-out session for P0b-s), "
               r"not an idealized equal-count comparator. $\ne$ counts the subject--seed--fold cells with a nonzero "
               r"test-count difference. The released construction CSV preserves every subject, seed, fold and comparator "
               r"count. All-run rows are summaries, not additional trials. These count differences describe protocol contrasts; "
               r"no train/test correlation threshold is an acceptance criterion.")
    lines = begin(caption, "tab:s_p0s_construction", "l l l l c c c c",
                  r"Dataset & Paradigm & Protocol & Run/class & P0s train/test & P1 train/test & $\Delta$ train/test & $\ne$/cells", True)
    for (dataset, task, protocol, run, label), values in sorted(groups.items()):
        different = sum(int(r["delta_test"]) != 0 for r in values)
        delta = int_range(values, "delta_train") + "/" + int_range(values, "delta_test")
        row = ["EEGBCI" if dataset == "eegbci" else "BCI-IV-2a", task_label(task), PROTOCOL_TEX[protocol],
               f"{run}/{label}", int_range(values, "n_train") + "/" + int_range(values, "n_test"),
               int_range(values, "p1_n_train") + "/" + int_range(values, "p1_n_test"), delta, f"{different}/{len(values)}"]
        lines.append(" & ".join(row) + r" \\")
    return end(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="rev2")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--out-dir", type=Path, default=C.ROOT / "manuscript" / "v3" / "tables")
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--means", type=Path)
    ap.add_argument("--construction", nargs="+", type=Path)
    args = ap.parse_args()
    stats_path = args.stats or args.results_dir / f"stats_{args.tag}_matched.csv"
    means_path = args.means or args.results_dir / f"stats_{args.tag}_matched_means.csv"
    construction_paths = args.construction or [args.results_dir / f"e16_{dataset}_{args.tag}_construction.csv"
                                               for dataset in ("eegbci", "bci2a")]
    stats, means = C.read_csv(stats_path), C.read_csv(means_path)
    S._check(bool(stats) and bool(means), "empty matched statistics")
    K = index_unique(stats, ("dataset", "task", "decoder", "reference", "contrast"))
    M = index_unique(means, ("dataset", "task", "decoder", "reference", "protocol", "arm"))
    construction = [row for path in construction_paths for row in C.read_csv(path)]
    S._check(bool(construction), "empty construction table")
    training, training_paths = training_inputs(args.results_dir, args.tag)
    tables = {
        "table_matched_eegbci.tex": main_table(M, K),
        "tableS_matched_bci2a.tex": bci_table(M, K),
        "tableS_matched_per_paradigm.tex": paradigm_table(M, K),
        "tableS_p0s_construction.tex": construction_table(construction),
        "tableS_e17_fullpipeline.tex": fullpipeline_table(M, K),
        "tableS_rev2_training.tex": training_table(training),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = [f"tag: {args.tag}", "Every scientific value is read or calculated from the following CSV files:"]
    for path in [stats_path, means_path] + construction_paths + training_paths:
        provenance.append(f"{path.resolve()} sha256={C.sha256_file(path)}")
    for name, lines in tables.items():
        text = "\n".join(lines) + "\n"
        S._check("component" not in text.lower(), f"retired ladder wording in {name}")
        path = args.out_dir / name
        path.write_text(text)
        provenance.append(f"output: {path.resolve()} sha256={C.sha256_file(path)}")
        print(f"wrote {path}")
    (args.out_dir / "tables_matched_provenance.txt").write_text("\n".join(provenance) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
