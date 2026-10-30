#!/usr/bin/env python3
"""E3a-v2 identity controls and E3b-v2 sequential SS (frozen plan B18/B19).

Assignment CSVs retain the task-card schemas. Subject-level aggregation, the
paired (ii)-(v) estimate and bootstrap CIs are released in the JSON manifest.
Run labels in the CSVs are physical EEGBCI recording numbers, not run_idx.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SCHEMES = ("none", "independent_reference", "training_run", "all_trial", "split_reference")
SCHEME_LABELS = {
    "none": "(i) no centring",
    "independent_reference": "(ii) independent reference",
    "training_run": "(iii) training run",
    "all_trial": "(iv) all trials (revision-1 construction)",
    "split_reference": "(v) split reference",
}
A_FIELDS = ["task", "ref_run", "train_run", "test_run", "scheme", "n_subjects", "chance",
            "identity_accuracy", "per_subject_median", "per_subject_min", "per_subject_max"]
SUBJECT_FIELDS = ["task", "ref_run", "train_run", "test_run", "scheme", "subject", "accuracy", "n_test"]
B_FIELDS = ["task", "order", "component", "ss", "fraction", "n_subjects", "n_trials", "n_dims"]
ORDER_COMPONENTS = {
    "A": ("subject", "class_given_subject", "subject_x_class_given_both", "residual"),
    "B": ("class", "subject_given_class", "subject_x_class_given_both", "residual"),
}
REL_TOL = 1e-9


def interleave_class_rows(cached_labels, recording_labels, source):
    """Match each retained cue to the next cached row of its class."""
    cached_labels, recording_labels = np.asarray(cached_labels), np.asarray(recording_labels)
    rows = {c: np.flatnonzero(cached_labels == c) for c in (0, 1)}
    counts = {c: int(np.sum(recording_labels == c)) for c in (0, 1)}
    if (sum(map(len, rows.values())) != len(cached_labels)
            or sum(counts.values()) != len(recording_labels)
            or any(len(rows[c]) != counts[c] for c in (0, 1))):
        raise ValueError(f"{source}: retained EDF/cache class counts differ: "
                         f"EDF={counts}, cache={ {c: len(rows[c]) for c in (0, 1)} }")
    used = {0: 0, 1: 0}
    order = []
    for label in recording_labels:
        order.append(rows[label][used[label]])
        used[label] += 1
    return np.asarray(order, dtype=np.int64)


def recording_order(task, subject, run_number, *, provenance=None):
    """Return run-local cached-row indices in retained EDF cue-onset order.

    The cache writer groups T1 then T2, preserving order within each class.
    Recover the order using annotations only, with the writer's 160 Hz
    resampling length/rounding and inclusive 0--4 s epoch bounds.
    """
    import mne

    if run_number not in C.TASK_RUNS[task]:
        raise ValueError(f"{task}: invalid physical run {run_number}")
    path = (C.ROOT / "data_public/eegbci_full/MNE-eegbci-data/files/eegmmidb/1.0.0"
            / f"S{subject:03d}" / f"S{subject:03d}R{run_number:02d}.edf")
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing EDF; fetch it with test/env/fetch_eegbci_s3.sh")
    cache_path = C.EEGBCI_CACHE / f"sub{subject:03d}_{task}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        cached_labels = cached["y"][cached["runs"] == run_number]
    with mne.io.read_raw_edf(path, preload=False, verbose="ERROR") as raw:
        sfreq = float(raw.info["sfreq"])
        target_sfreq = 160.0 if abs(sfreq - 160.0) > 1e-3 else sfreq
        n_times = int(round(raw.n_times * target_sfreq / sfreq))
        # EEGBCI EDFs start at sample zero; refuse an unsupported time origin.
        if raw.first_samp != 0:
            raise ValueError(f"{path}: expected EDF first_samp=0, got {raw.first_samp}")
        annotations = raw.annotations
        wanted = np.isin(annotations.description, ["T1", "T2"])
        samples = np.round(annotations.onset[wanted] * target_sfreq).astype(np.int64)
        labels = np.array([0 if name == "T1" else 1 for name in annotations.description[wanted]])
        # MNE resampling leaves annotation times unchanged. Epochs includes tmax,
        # so an onset at sample k needs k + round(4 * sfreq) < n_times.
        keep = (samples >= 0) & (samples + int(round(4.0 * target_sfreq)) < n_times)
        labels = labels[keep] if len(set(annotations.description[wanted])) == 2 else labels[:0]
    order = interleave_class_rows(cached_labels, labels, path)
    if provenance is not None:
        provenance.update(task=task, subject=int(subject), run_number=int(run_number),
                          edf_path=str(path.relative_to(C.ROOT)), edf_sha256=C.sha256_file(path),
                          class_sequence=labels.tolist())
    return order


def centred_splits(feats, subject, run_idx, ref_run, train_run, test_run, scheme,
                   recording_orders=None):
    """Apply one B18 scheme; cached rows within a run are grouped by class.

    Only (v) uses the EDF-derived run-local permutation to alternate within each
    subject's reference run. Neither half is reused for the other split.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown centring scheme: {scheme}")
    if len({ref_run, train_run, test_run}) != 3:
        raise ValueError("reference, training and test runs must be distinct")
    tr, te = np.flatnonzero(run_idx == train_run), np.flatnonzero(run_idx == test_run)
    xtr, xte = feats[tr].copy(), feats[te].copy()
    for s in np.unique(subject):
        ref = np.flatnonzero((subject == s) & (run_idx == ref_run))
        train = np.flatnonzero((subject == s) & (run_idx == train_run))
        test = np.flatnonzero((subject == s) & (run_idx == test_run))
        if not len(ref) or not len(train) or not len(test):
            raise ValueError(f"subject {s} lacks one of the three runs")
        if scheme == "none":
            continue
        if scheme == "independent_reference":
            mtr = mte = feats[ref].mean(axis=0)
        elif scheme == "training_run":
            mtr = mte = feats[train].mean(axis=0)
        elif scheme == "all_trial":
            mtr = mte = feats[subject == s].mean(axis=0)
        else:
            if len(ref) < 2:
                raise ValueError(f"subject {s} needs two reference trials for alternating halves")
            if recording_orders is None:
                raise ValueError("split_reference requires EDF-derived recording orders")
            order = recording_orders[(int(s), ref_run)]
            if not np.array_equal(np.sort(order), np.arange(len(ref))):
                raise ValueError(f"subject {s} run_idx {ref_run}: invalid recording-order permutation")
            ref = ref[order]
            mtr = feats[ref[::2]].mean(axis=0)
            mte = feats[ref[1::2]].mean(axis=0)
        xtr[subject[tr] == s] -= mtr
        xte[subject[te] == s] -= mte
    return xtr, xte, tr, te


def identity_probe(feats, subject, run_idx, ref_run, train_run, test_run, scheme,
                   recording_orders=None):
    from sklearn.linear_model import LogisticRegression

    xtr, xte, tr, te = centred_splits(
        feats, subject, run_idx, ref_run, train_run, test_run, scheme, recording_orders)
    clf = LogisticRegression(solver="lbfgs", max_iter=3000, C=1.0, random_state=C.SEED)
    clf.fit(xtr, subject[tr])
    if np.any(clf.n_iter_ >= clf.max_iter):
        raise RuntimeError("identity probe reached max_iter; refusing an unconverged result")
    pred = clf.predict(xte)
    per_subject = C.per_subject_acc(subject[te], pred, subject[te])
    return float(np.mean(pred == subject[te])), per_subject, {
        int(s): int(np.sum(subject[te] == s)) for s in np.unique(subject[te])}


def training_space(covs, run_idx, train_run):
    ref = C.riemann_mean(covs[run_idx == train_run])
    return train_run, C.tangent(covs, ref)


def identity_assignment(task, feats, subject, run_idx, assignment, scheme, recording_orders=None):
    ref, train, test = assignment
    acc, per_s, counts = identity_probe(feats, subject, run_idx, ref, train, test, scheme,
                                      recording_orders)
    base = dict(task=task, ref_run=C.TASK_RUNS[task][ref], train_run=C.TASK_RUNS[task][train],
                test_run=C.TASK_RUNS[task][test], scheme=scheme)
    values = list(per_s.values())
    row = dict(base, n_subjects=len(per_s), chance=1 / len(per_s), identity_accuracy=acc,
               per_subject_median=float(np.median(values)), per_subject_min=min(values),
               per_subject_max=max(values))
    subjects = [dict(base, subject=s, accuracy=per_s[s], n_test=counts[s]) for s in sorted(per_s)]
    return row, subjects


def summarize_identity(rows):
    """Average six assignments within subject, then bootstrap subjects (B18).

    Ranges use per-assignment macro accuracy. Pooled trial estimates count all
    test appearances (each trial occurs as test in two assignments). No CI uses
    the six assignments as independent observations.
    """
    summaries = []
    for task in sorted({r["task"] for r in rows}, key=C.TASKS.index):
        vectors = {}
        for scheme in SCHEMES:
            rr = [r for r in rows if r["task"] == task and r["scheme"] == scheme]
            subjects = sorted({int(r["subject"]) for r in rr})
            assignments = sorted({tuple(int(r[k]) for k in ("ref_run", "train_run", "test_run")) for r in rr})
            expected = set(permutations(C.TASK_RUNS[task]))
            if set(assignments) != expected or len(subjects) < 2:
                raise ValueError(f"{task}/{scheme}: need all six assignments and at least two subjects")
            lookup = {(tuple(int(r[k]) for k in ("ref_run", "train_run", "test_run")), int(r["subject"])): r for r in rr}
            if len(lookup) != len(rr) or len(rr) != len(subjects) * len(assignments):
                raise ValueError(f"{task}/{scheme}: duplicate or missing subject/assignment")
            scores = np.array([[float(lookup[(a, s)]["accuracy"]) for s in subjects] for a in assignments])
            counts = np.array([[int(lookup[(a, s)]["n_test"]) for s in subjects] for a in assignments])
            if np.any(counts <= 0) or np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
                raise ValueError("invalid per-subject identity scores or counts")
            means = scores.mean(axis=0)
            vectors[scheme] = (subjects, means)
            lo, hi = C.bootstrap_ci(means, n=2000, seed=20260916)
            assignment_macro = scores.mean(axis=1)
            assignment_pooled = (scores * counts).sum(axis=1) / counts.sum(axis=1)
            summaries.append(dict(task=task, scheme=scheme, n_subjects=len(subjects), chance=1 / len(subjects),
                                  mean=float(means.mean()), ci_low=lo, ci_high=hi,
                                  assignment_min=float(assignment_macro.min()), assignment_max=float(assignment_macro.max()),
                                  pooled_trial_accuracy=float((scores * counts).sum() / counts.sum()),
                                  assignments=[dict(zip(("ref_run", "train_run", "test_run"), a),
                                                    macro_accuracy=float(m), pooled_trial_accuracy=float(p))
                                               for a, m, p in zip(assignments, assignment_macro, assignment_pooled)],
                                  per_subject=[dict(subject=s, mean_accuracy=float(v)) for s, v in zip(subjects, means)]))
        subjects, independent = vectors["independent_reference"]
        split_subjects, split = vectors["split_reference"]
        if subjects != split_subjects or any(v[0] != subjects for v in vectors.values()):
            raise ValueError("schemes must use identical subjects")
        diff = independent - split
        lo, hi = C.bootstrap_ci(diff, n=2000, seed=20260916)
        summaries.append(dict(task=task, scheme="independent_reference_minus_split_reference", n_subjects=len(subjects),
                              mean=float(diff.mean()), ci_low=lo, ci_high=hi,
                              per_subject=[dict(subject=s, mean_difference=float(v)) for s, v in zip(subjects, diff)]))
    return summaries


def grouped_rss(feats, groups):
    """OLS residual SS per dimension: one fitted mean per group (includes intercept)."""
    residual = np.empty_like(feats)
    for group in np.unique(groups):
        mask = groups == group
        residual[mask] = feats[mask] - feats[mask].mean(axis=0)
    return np.sum(residual ** 2, axis=0)


def variance_decomposition(feats, subject, y):
    """B19 ordinary trial-weighted nested OLS, one additive solve for all columns."""
    feats = np.asarray(feats, dtype=np.float64)
    _, si = np.unique(subject, return_inverse=True)
    _, ci = np.unique(y, return_inverse=True)
    if feats.ndim != 2 or len(feats) != len(si) or len(ci) != len(si) or not np.all(np.isfinite(feats)):
        raise ValueError("features and grouping arrays must be finite and trial-aligned")
    ns, nc = int(si.max()) + 1, int(ci.max()) + 1
    # Treatment indicators plus an explicit intercept: no cell balancing/reweighting.
    design = np.column_stack((np.ones(len(feats)), np.eye(ns)[si, 1:], np.eye(nc)[ci, 1:]))
    beta = np.linalg.lstsq(design, feats, rcond=None)[0]
    rss = {
        "RSS0": np.sum((feats - feats.mean(axis=0)) ** 2, axis=0),
        "RSS_S": grouped_rss(feats, si),
        "RSS_C": grouped_rss(feats, ci),
        "RSS_SC": np.sum((feats - design @ beta) ** 2, axis=0),
        "RSS_cell": grouped_rss(feats, si * nc + ci),
    }
    total = float(rss["RSS0"].sum())
    if total <= 0:
        raise ValueError("variance fractions need positive total SS")
    shared = [rss["RSS_SC"] - rss["RSS_cell"], rss["RSS_cell"]]
    orders = {}
    for order, values in (("A", [rss["RSS0"] - rss["RSS_S"], rss["RSS_S"] - rss["RSS_SC"], *shared]),
                          ("B", [rss["RSS0"] - rss["RSS_C"], rss["RSS_C"] - rss["RSS_SC"], *shared])):
        values = np.array(values)
        # Check per dimension as well as the total, before clipping roundoff only.
        if np.any(values < -REL_TOL * np.maximum(rss["RSS0"], total * np.finfo(float).eps)):
            raise RuntimeError(f"order {order}: negative sequential SS beyond floating-point tolerance")
        ss = np.maximum(values, 0).sum(axis=1)
        if not np.isclose(ss.sum(), total, rtol=REL_TOL, atol=0):
            raise RuntimeError(f"order {order}: sequential SS do not sum to RSS0")
        orders[order] = {name: dict(ss=float(value), fraction=float(value / total))
                         for name, value in zip(ORDER_COMPONENTS[order], ss)}
    return dict(rss={name: float(value.sum()) for name, value in rss.items()}, orders=orders,
                n_subjects=ns, n_trials=len(feats), n_dims=feats.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--parts", default="a,b")
    args = ap.parse_args()
    tasks, parts = args.tasks.split(","), set(args.parts.split(","))
    tag = args.tag or C.stamp()
    if (not tasks or any(t not in C.TASKS for t in tasks) or len(set(tasks)) != len(tasks)
            or not parts <= {"a", "b"} or not parts or args.n_jobs < 1 or args.max_subjects < 0
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag) or "rev1" in tag):
        ap.error("invalid tasks, parts, worker count, subject limit or tag (rev1 is frozen)")
    from joblib import Parallel, delayed, parallel_config

    started = C.utc_now()
    a_rows, subject_rows, b_rows, rss_checks, cache_hashes = [], [], [], {}, {}
    recording_provenance = []
    for task in tasks:
        subs = C.discover_subjects(task)
        if args.max_subjects:
            subs = subs[:args.max_subjects]
        if len(subs) < 2:
            raise ValueError(f"{task}: at least two subjects required")
        for s in subs:
            path = C.EEGBCI_CACHE / f"sub{s:03d}_{task}.npz"
            cache_hashes[str(path.relative_to(C.ROOT))] = C.sha256_file(path)
        data = C.load_eegbci_task(task, subs)
        subject, run_idx = data["subject"], data["run_idx"]
        if set(np.unique(run_idx)) != {0, 1, 2}:
            raise ValueError("B18 requires exactly three recording-order runs")
        recording_orders = {}
        if "a" in parts:
            for s in subs:
                for r, run_number in enumerate(C.TASK_RUNS[task]):
                    entry = {}
                    recording_orders[(s, r)] = recording_order(task, s, run_number, provenance=entry)
                    recording_provenance.append(entry)
        covs = C.covariances(data["x"])
        del data["x"]
        print(f"[{task}] trials={len(subject)} subjects={len(subs)}", flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            if "a" in parts:
                # Reuse each identical training-run tangent space across its two assignments.
                spaces = dict(Parallel(n_jobs=min(args.n_jobs, 3))(
                    delayed(training_space)(covs, run_idx, train) for train in range(3)))
                results = Parallel(n_jobs=min(args.n_jobs, 6 * len(SCHEMES)))(
                    delayed(identity_assignment)(task, spaces[a[1]], subject, run_idx, a, scheme,
                                                 recording_orders if scheme == "split_reference" else None)
                    for a in permutations(range(3)) for scheme in SCHEMES)
                for row, subjects in results:
                    a_rows.append(row)
                    subject_rows.extend(subjects)
                del spaces
            if "b" in parts:
                feats = C.tangent(covs, C.riemann_mean(covs))
                vd = variance_decomposition(feats, subject, data["y"])
                rss_checks[task] = vd["rss"]
                for order, terms in vd["orders"].items():
                    for name, values in terms.items():
                        b_rows.append(dict(task=task, order=order, component=name, **values,
                                           **{k: vd[k] for k in ("n_subjects", "n_trials", "n_dims")}))
                del feats
        del covs

    outputs = []
    for name, rows, fields in ((f"e3a2_{tag}.csv", a_rows, A_FIELDS),
                               (f"e3a2_{tag}_per_subject.csv", subject_rows, SUBJECT_FIELDS),
                               (f"e3b2_{tag}.csv", b_rows, B_FIELDS)):
        if rows:
            path = C.write_csv(C.RESULTS / name, rows, fields)
            C.validate_csv(path, fields, len(rows))
            outputs.append(path)
    summaries = summarize_identity(subject_rows) if subject_rows else []
    for row in summaries:
        print(f"{row['task']} {row['scheme']}: macro mean {row['mean']:.6g} "
              f"95% subject-bootstrap CI [{row['ci_low']:.6g}, {row['ci_high']:.6g}]", flush=True)
    plan = C.ROOT / "research" / "REVISION_PLAN_v2.md"
    extra = dict(plan_sha256=C.sha256_file(plan), plan_path=str(plan.relative_to(C.ROOT)),
                 script_sha256=C.sha256_file(Path(__file__)), common_sha256=C.sha256_file(HERE / "common.py"),
                 cache_sha256=cache_hashes, identity_summary=summaries, nested_rss=rss_checks,
                 split_reference_recording_order=recording_provenance,
                 bootstrap=dict(unit="subject after six-assignment within-subject mean", n=2000, seed=20260916),
                 run_labels="physical recording numbers; run_idx 0,1,2 follows recording order",
                 centring_schemes=SCHEME_LABELS, ss_relative_tolerance=REL_TOL,
                 tangent_reference="E3a: training-run covariance mean; E3b: all-trial covariance mean",
                 git_head_note="not queried: task prohibits git")
    # C.write_manifest is retained unchanged; suppress its git subprocess for this task.
    with patch.object(C, "git_head", return_value="not_queried_task_prohibits_git"):
        C.write_manifest(C.RESULTS / f"e3v2_{tag}_manifest.json", f"e3v2_{tag}", started, outputs, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
