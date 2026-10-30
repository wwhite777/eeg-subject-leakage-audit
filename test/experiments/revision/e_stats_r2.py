#!/usr/bin/env python3
"""Subject-paired E16/E17 statistics (frozen revision plan v2, B20--B22).

Seeds are averaged within subject before paradigms are averaged within subject.
The only Holm family is the twelve primary EEGBCI matched-controls contrasts.
Missing subjects, paradigms, seeds within a protocol, or fixed-test arms fail closed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

DECODERS = ("csp_lda", "riemann_ts_lr", "shallow_fbcsp", "eegnet_v4")
CLASSICAL = set(DECODERS[:2])
ARMS = ("A0", "A1", "A2")
FAMILY = "eegbci_matched_controls"
PRIMARY = ("p0s_minus_p1", "target_inclusion", "same_run_adjacency")
LADDER = {
    "eegbci": (("trial_random_sc", "run_disjoint", "p0s_minus_p1"),
                ("trial_random", "trial_random_sc", "p0_minus_p0s"),
                ("trial_random_sc", "subject_disjoint", "p0s_minus_p2")),
    "bci2a": (("trial_random_6_sc", "run_disjoint", "p0as_minus_p1"),
               ("trial_random_2_sc", "session_disjoint", "p0bs_minus_p1s"),
               ("trial_random_6", "trial_random_6_sc", "p0a_minus_p0as"),
               ("trial_random_2", "trial_random_2_sc", "p0b_minus_p0bs"),
               ("trial_random_6_sc", "loso", "p0as_minus_p2"),
               ("trial_random_2_sc", "loso", "p0bs_minus_p2")),
}


def tasks_for(dataset):
    return C.TASKS if dataset == "eegbci" else ["bci2a_left_right"]


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def load_rows(paths, dataset):
    rows = []
    for path in paths:
        _check(path.exists(), f"missing input: {path}")
        incoming = C.read_csv(path)
        _check(bool(incoming), f"empty input: {path}")
        for raw in incoming:
            r = dict(raw)
            r.setdefault("dataset", dataset)
            _check(r["dataset"] == dataset, f"wrong dataset in {path}: {r['dataset']}")
            if r.get("width", "default") not in ("", "default"):
                continue
            if r["model"] not in CLASSICAL and r.get("preproc", "ztrial") != "ztrial":
                continue
            if r["model"] not in DECODERS:
                continue
            r["_source"] = str(path)
            rows.append(r)
    return rows


def _accuracy(row):
    value = float(row["accuracy"])
    _check(np.isfinite(value) and 0 <= value <= 1,
           f"invalid accuracy in {row['_source']}: {row}")
    return value


def ladder_subjects(rows, dataset, required_protocols, *, incomplete=False, new=False):
    """Return (task, decoder, protocol) -> subject -> seed-mean accuracy."""
    cells = defaultdict(dict)
    for r in rows:
        protocol = r.get("protocol", "")
        if protocol not in required_protocols:
            continue
        _check(r["task"] in tasks_for(dataset), f"unexpected paradigm: {r['task']}")
        key = (r["task"], r["model"], protocol)
        seed, subject = int(r["seed"]), int(r["subject"])
        _check((seed, subject) not in cells[key], f"duplicate ladder row: {key}, {seed}, {subject}")
        cells[key][seed, subject] = _accuracy(r)
    expected_subjects = set(range(1, (109 if dataset == "eegbci" else 9) + 1))
    out = {}
    for task in tasks_for(dataset):
        for decoder in DECODERS:
            for protocol in sorted(required_protocols):
                key = task, decoder, protocol
                values = cells.get(key, {})
                _check(bool(values), f"missing ladder group: {dataset}, {key}")
                subjects = {s for _, s in values}
                if not incomplete:
                    _check(subjects == expected_subjects,
                           f"subject coverage mismatch: {dataset}, {key}; missing={sorted(expected_subjects-subjects)}, extra={sorted(subjects-expected_subjects)}")
                seeds = {seed for seed, _ in values}
                if new and not incomplete:
                    expected_seeds = set(range(C.SEED, C.SEED + (5 if decoder in CLASSICAL else 3)))
                    _check(seeds == expected_seeds, f"partition/training seed mismatch: {dataset}, {key}: {sorted(seeds)}")
                _check(set(values) == {(seed, subject) for seed in seeds for subject in subjects},
                       f"missing subject-seed ladder cell: {dataset}, {key}")
                out[key] = {s: float(np.mean([values[seed, s] for seed in sorted(seeds)]))
                            for s in sorted(subjects)}
    return out


def fixed_subjects(rows, dataset, *, incomplete=False):
    """Validate paired arm/rotation records before averaging each subject's tested seeds."""
    cells = defaultdict(dict)
    for r in rows:
        _check(r.get("arm", "") in ("",) + ARMS, f"unknown fixed-test arm: {r.get('arm')}")
        _check(r.get("protocol", "") != "fixed_test" or r.get("arm", "") in ARMS,
               f"missing fixed-test arm in {r['_source']}")
        if r.get("arm", "") not in ARMS:
            continue
        _check(r["task"] in tasks_for(dataset), f"unexpected fixed-test paradigm: {r['task']}")
        reference = r.get("reference", "bg_fixed")
        _check(reference in ("bg_fixed", "arm_own"), f"unexpected reference: {reference}")
        _check(reference != "arm_own" or r["model"] == "riemann_ts_lr",
               "the full-pipeline reference variant is defined for TS+LR only")
        key = r["task"], r["model"], reference
        seed, rotation, subject = int(r["seed"]), int(r["rotation"]), int(r["subject"])
        sample = seed, rotation, subject
        arm = r["arm"]
        _check((sample, arm) not in cells[key], f"duplicate fixed-test row: {dataset}, {key}, {sample}, {arm}")
        n_test = int(r["n_test"])
        _check(n_test > 0, f"empty fixed test: {key}, {sample}")
        cells[key][sample, arm] = (_accuracy(r), n_test)
    out = {}
    all_subjects = set(range(1, (109 if dataset == "eegbci" else 9) + 1))
    for task in tasks_for(dataset):
        for decoder in DECODERS:
            references = ("bg_fixed", "arm_own") if decoder == "riemann_ts_lr" else ("bg_fixed",)
            reference_samples = None
            reference_counts = None
            for reference in references:
                key = task, decoder, reference
                values = cells.get(key, {})
                _check(bool(values), f"missing fixed-test group: {dataset}, {key}")
                samples = {sample for sample, _ in values}
                if reference_samples is not None:
                    _check(samples == reference_samples, f"different test sample sets across references: {dataset}, {key}")
                reference_samples = samples
                _check(set(values) == {(sample, arm) for sample in samples for arm in ARMS},
                       f"missing paired fixed-test arm: {dataset}, {key}")
                for sample in samples:
                    _check(len({values[sample, arm][1] for arm in ARMS}) == 1,
                           f"different test counts across arms: {dataset}, {key}, {sample}")
                counts = {sample: values[sample, "A0"][1] for sample in samples}
                if reference_counts is not None:
                    _check(counts == reference_counts, f"different test counts across references: {dataset}, {key}")
                reference_counts = counts
                seed_subjects = [(seed, subject) for seed, _, subject in samples]
                _check(len(seed_subjects) == len(set(seed_subjects)),
                       f"subject tested in multiple rotations in one seed: {dataset}, {key}")
                subjects = {s for _, _, s in samples}
                seeds = {seed for seed, _, _ in samples}
                if not incomplete:
                    _check(subjects == all_subjects, f"fixed-test subject coverage mismatch: {dataset}, {key}")
                    expected_seeds = set(range(C.SEED, C.SEED + (5 if decoder in CLASSICAL else 3)))
                    _check(seeds == expected_seeds, f"fixed-test seed mismatch: {dataset}, {key}: {sorted(seeds)}")
                    for seed in seeds:
                        rotations = {rot for ss, rot, _ in samples if ss == seed}
                        _check(len(rotations) == 3, f"missing fixed-test rotation: {dataset}, {key}, {seed}")
                        target_n = 36 if dataset == "eegbci" else 3
                        permutation = np.random.default_rng(seed).permutation(sorted(all_subjects))
                        groups = permutation[:3 * target_n].reshape(3, target_n)
                        expected = {(seed, rotation, int(subject)) for rotation, group in enumerate(groups) for subject in group}
                        actual = {sample for sample in samples if sample[0] == seed}
                        _check(actual == expected,
                               f"seeded target/rotation assignment mismatch: {dataset}, {key}, {seed}; "
                               f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
                        for rotation in rotations:
                            _check(sum(ss == seed and rot == rotation for ss, rot, _ in samples) == target_n,
                                   f"incorrect target block size: {dataset}, {key}, {seed}, {rotation}")
                for arm in ARMS:
                    out[task, decoder, reference, arm] = {
                        s: float(np.mean([values[sample, arm][0] for sample in sorted(samples) if sample[2] == s]))
                        for s in sorted(subjects)}
    return out


def add_paradigm_means(groups, dataset):
    """Require all four paradigms and identical subject sets; never average survivors."""
    out = dict(groups)
    if dataset != "eegbci":
        return out
    suffixes = {key[1:] for key in groups}
    for suffix in sorted(suffixes):
        values = [groups.get((task,) + suffix) for task in C.TASKS]
        _check(all(v is not None for v in values), f"missing paradigm for {suffix}")
        subjects = set(values[0])
        _check(all(set(v) == subjects for v in values), f"different subject sets across paradigms for {suffix}")
        out[("paradigm_mean",) + suffix] = {s: float(np.mean([v[s] for v in values])) for s in sorted(subjects)}
    return out


def paired(a, b):
    _check(set(a) == set(b), f"paired subject sets differ: missing from A={sorted(set(b)-set(a))}, missing from B={sorted(set(a)-set(b))}")
    subjects = sorted(a)
    av, bv = np.array([a[s] for s in subjects]), np.array([b[s] for s in subjects])
    d = av - bv
    summary = C.summarize_subject_vector(d)
    _, p = C.wilcoxon_paired(av, bv)
    rounded = np.round(d, C.ROUND_DECIMALS)
    return {"n": summary["n"], "mean": summary["mean"], "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"], "p_wilcoxon": p, "p_signflip": C.signflip_perm_p(d),
            "n_positive": int(np.sum(rounded > 0)), "n_negative": int(np.sum(rounded < 0)),
            "n_zero": int(np.sum(rounded == 0))}


def analyze(ladder, fixed, dataset):
    ladder = add_paradigm_means(ladder, dataset)
    fixed = add_paradigm_means(fixed, dataset)
    contrasts, means = [], []
    for (task, decoder, protocol), values in sorted(ladder.items()):
        summary = C.summarize_subject_vector(list(values.values()))
        means.append({"dataset": dataset, "task": task, "decoder": decoder, "reference": "protocol_own",
                      "protocol": protocol, "arm": "", **{k: summary[k] for k in ("n", "mean", "ci_low", "ci_high")}})
    for (task, decoder, reference, arm), values in sorted(fixed.items()):
        summary = C.summarize_subject_vector(list(values.values()))
        means.append({"dataset": dataset, "task": task, "decoder": decoder, "reference": reference,
                      "protocol": "fixed_test", "arm": arm, **{k: summary[k] for k in ("n", "mean", "ci_low", "ci_high")}})
    tasks = tasks_for(dataset) + (["paradigm_mean"] if dataset == "eegbci" else [])
    for task in tasks:
        for decoder in DECODERS:
            for pa, pb, name in LADDER[dataset]:
                c = paired(ladder[task, decoder, pa], ladder[task, decoder, pb])
                contrasts.append({"dataset": dataset, "task": task, "decoder": decoder, "contrast": name,
                                  "reference": "protocol_own", "protocol_a": pa, "protocol_b": pb, **c})
            for reference in (("bg_fixed", "arm_own") if decoder == "riemann_ts_lr" else ("bg_fixed",)):
                for pa, pb, name in (("A1", "A0", "target_inclusion"), ("A2", "A1", "same_run_adjacency")):
                    c = paired(fixed[task, decoder, reference, pa], fixed[task, decoder, reference, pb])
                    contrasts.append({"dataset": dataset, "task": task, "decoder": decoder, "contrast": name,
                                      "reference": reference, "protocol_a": pa, "protocol_b": pb, **c})
                # Descriptive paired difference of the two matched contrasts per subject (DECISION_LOG 2026-10-30T04:40Z):
                # (A1 - A0) - (A2 - A1); never part of the Holm family and never a rule-C12 criterion.
                a0, a1, a2 = (fixed[task, decoder, reference, arm] for arm in ("A0", "A1", "A2"))
                incl = {sub: a1[sub] - a0[sub] for sub in a1}
                adj = {sub: a2[sub] - a1[sub] for sub in a1}
                c = paired(incl, adj)
                contrasts.append({"dataset": dataset, "task": task, "decoder": decoder, "contrast": "inclusion_minus_adjacency",
                                  "reference": reference, "protocol_a": "A1-A0", "protocol_b": "A2-A1", **c})
    return contrasts, means


def apply_family(rows):
    primary = [r for r in rows if r["dataset"] == "eegbci" and r["task"] == "paradigm_mean"
               and r["contrast"] in PRIMARY and r["reference"] != "arm_own"]
    _check(len(primary) == 12 and {(r["decoder"], r["contrast"]) for r in primary}
           == {(model, contrast) for model in DECODERS for contrast in PRIMARY},
           f"primary matched-controls family is incomplete or duplicated: {len(primary)} rows")
    for r in rows:
        r.update({"p_holm": "", "family": "descriptive", "family_size": ""})
    for row, p in zip(primary, C.holm([r["p_wilcoxon"] for r in primary])):
        row.update({"p_holm": p, "family": FAMILY, "family_size": len(primary)})


def discover_deep(root, prefix, tag):
    """Use explicit task shards when present; never silently ingest aggregate duplicates."""
    paths = []
    for stem in (tag, f"{tag}_fixed"):
        whole = root / f"{prefix}_{stem}_per_subject.csv"
        shards = [root / f"{prefix}_{stem}_{part}_per_subject.csv" for part in ("t12", "t34")]
        present_shards = [p for p in shards if p.exists()]
        _check(not (whole.exists() and present_shards),
               f"ambiguous aggregate and task shards: {whole}; pass explicit --{prefix} paths")
        if whole.exists():
            paths.append(whole)
        paths.extend(present_shards)
    _check(bool(paths), f"no {prefix} inputs for tag {tag}; pass explicit --{prefix} paths")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="rev2")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--datasets", default="eegbci,bci2a")
    ap.add_argument("--allow-incomplete", action="store_true", help="fixture/smoke subject and seed counts only; all decoders, paradigms, paired arms remain required")
    for name in ("e1-eegbci", "e1-bci2a", "e17-eegbci", "e17-bci2a", "e9", "e10",
                 "rev1-e1-eegbci", "rev1-e1-bci2a", "rev1-e9", "rev1-e10"):
        ap.add_argument(f"--{name}", nargs="+", type=Path)
    args = ap.parse_args()
    datasets = args.datasets.split(",")
    _check(set(datasets) <= set(LADDER) and len(set(datasets)) == len(datasets), "invalid datasets")
    R, outputs = args.results_dir, []
    O = args.out_dir or R
    sources, contrasts, means = [], [], []
    for dataset in datasets:
        e1_paths = getattr(args, f"e1_{dataset}") or [R / f"e1_{dataset}_{args.tag}_per_subject.csv"]
        e17_paths = getattr(args, f"e17_{dataset}") or [R / f"e17_{dataset}_{args.tag}_per_subject.csv"]
        deep = "e9" if dataset == "eegbci" else "e10"
        deep_paths = getattr(args, deep) or discover_deep(R, deep, args.tag)
        old_e1 = getattr(args, f"rev1_e1_{dataset}") or [R / f"e1_{dataset}_rev1_per_subject.csv"]
        old_deep = getattr(args, f"rev1_{deep}") or [R / ("e9_rev1_all_per_subject.csv" if deep == "e9" else "e10_rev1_per_subject.csv")]
        sources.extend(e1_paths + e17_paths + deep_paths + old_e1 + old_deep)
        new_rows = load_rows(e1_paths + deep_paths, dataset)
        old_rows = load_rows(old_e1 + old_deep, dataset)
        e17_rows = load_rows(e17_paths + deep_paths, dataset)
        new_protocols = {p for pair in LADDER[dataset] for p in pair[:2] if p.endswith("_sc")}
        old_protocols = {p for pair in LADDER[dataset] for p in pair[:2]} - new_protocols
        ladder = ladder_subjects(old_rows, dataset, old_protocols, incomplete=args.allow_incomplete)
        ladder.update(ladder_subjects(new_rows, dataset, new_protocols, incomplete=args.allow_incomplete, new=True))
        fixed = fixed_subjects(e17_rows, dataset, incomplete=args.allow_incomplete)
        cc, mm = analyze(ladder, fixed, dataset)
        contrasts.extend(cc)
        means.extend(mm)
    if "eegbci" in datasets:
        apply_family(contrasts)
    else:
        for row in contrasts:
            row.update({"p_holm": "", "family": "descriptive", "family_size": ""})
    fields = ["dataset", "decoder", "contrast", "reference", "n", "mean", "ci_low", "ci_high",
              "p_wilcoxon", "p_signflip", "p_holm", "family", "family_size", "n_positive", "n_negative",
              "n_zero", "task", "protocol_a", "protocol_b"]
    outputs.append(C.write_csv(O / f"stats_{args.tag}_matched.csv", contrasts, fields))
    outputs.append(C.write_csv(O / f"stats_{args.tag}_matched_means.csv", means))
    manifest = {"analysis": "E16/E17 subject-paired contrasts", "plan": "REVISION_PLAN_v2 B20--B22",
                "tag": args.tag, "allow_incomplete": args.allow_incomplete,
                "inputs": {str(p): C.sha256_file(p) for p in sorted(set(sources))},
                "outputs": {str(p): C.sha256_file(p) for p in outputs}}
    path = O / f"stats_{args.tag}_matched_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for r in contrasts:
        if r["task"] in ("paradigm_mean", "bci2a_left_right"):
            print(f"{r['dataset']} {r['decoder']} contrast {r['contrast']} {r['reference']}: "
                  f"n={r['n']} mean={r['mean']:+.6f} CI[{r['ci_low']:+.6f},{r['ci_high']:+.6f}] "
                  f"p={r['p_wilcoxon']:.6g} Holm={r['p_holm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
