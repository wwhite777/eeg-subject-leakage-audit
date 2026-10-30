#!/usr/bin/env python3
"""Shared code for the revision analyses (REVISION_PLAN_v1.md).

Everything here is deterministic given the seeds in the plan. No modelling
decision is made in this module; it provides loaders, split constructors that
implement the protocol ladder (plan B3), tangent-space feature helpers, the
subject-level statistics of plan B2, and manifest/validation utilities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EEGBCI_CACHE = ROOT / "data_public" / "eegbci_full" / "cache"
BCI2A_CACHE = ROOT / "data_public" / "bci2a" / "cache" / "bci2a_leftright.npz"
RESULTS = ROOT / "results" / "revision"
PLAN = ROOT / "research" / "REVISION_PLAN_v1.md"

SEED = 20260916
BOOT_N = 2000
N_PERM = 10000
TOST_MARGIN = 0.01

TASKS = [
    "imagery_left_right_fist",
    "real_left_right_fist",
    "imagery_fists_feet",
    "real_fists_feet",
]
TASK_SHORT = {
    "imagery_left_right_fist": "MI L/R fist",
    "real_left_right_fist": "ME L/R fist",
    "imagery_fists_feet": "MI fists/feet",
    "real_fists_feet": "ME fists/feet",
}
# PhysioNet run numbers per paradigm, in recording order (run index 0, 1, 2).
TASK_RUNS = {
    "imagery_left_right_fist": [4, 8, 12],
    "real_left_right_fist": [3, 7, 11],
    "imagery_fists_feet": [6, 10, 14],
    "real_fists_feet": [5, 9, 13],
}
CACHE_RE = re.compile(r"sub(\d{3})_(.+)\.npz$")

PROTOCOL_LABEL = {
    "trial_random": "P0 trial-random (subject- and run-overlapping)",
    "run_disjoint": "P1 run-disjoint, subject-overlapping",
    "session_disjoint": "P1s session-disjoint, subject-overlapping",
    "subject_disjoint": "P2 subject-disjoint (GroupKFold)",
    "loso": "P2 subject-disjoint (leave-one-subject-out)",
    "within_subject": "within-subject (leave-one-run-out)",
    "within_session": "within-subject (session-to-session)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")


# ----------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------
def discover_subjects(task: str, cache_dir: Path = EEGBCI_CACHE) -> list[int]:
    subjects = []
    for path in cache_dir.glob(f"sub*_{task}.npz"):
        m = CACHE_RE.search(path.name)
        if m and m.group(2) == task:
            subjects.append(int(m.group(1)))
    return sorted(subjects)


def load_eegbci_task(task: str, subjects: list[int] | None = None, cache_dir: Path = EEGBCI_CACHE) -> dict:
    """Return dict(x, y, subject, run_idx, run_number, task). run_idx = 0,1,2 in recording order."""
    if subjects is None:
        subjects = discover_subjects(task, cache_dir)
    blocks = []
    min_t = None
    for subject in subjects:
        path = cache_dir / f"sub{subject:03d}_{task}.npz"
        with np.load(path, allow_pickle=True) as cached:
            x = cached["x"].astype(np.float32)
            y = cached["y"].astype(np.int64)
            r = cached["runs"].astype(np.int64)
        min_t = x.shape[2] if min_t is None else min(min_t, x.shape[2])
        blocks.append((subject, x, y, r))
    run_map = {rn: i for i, rn in enumerate(TASK_RUNS[task])}
    xs, ys, ss, ri, rn = [], [], [], [], []
    for subject, x, y, r in blocks:
        xs.append(x[:, :, :min_t])
        ys.append(y)
        ss.append(np.full(len(y), subject, dtype=np.int64))
        ri.append(np.array([run_map[int(v)] for v in r], dtype=np.int64))
        rn.append(r)
    return {
        "task": task,
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "subject": np.concatenate(ss, axis=0),
        "run_idx": np.concatenate(ri, axis=0),
        "run_number": np.concatenate(rn, axis=0),
        "dataset": "eegbci",
    }


def load_bci2a(path: Path = BCI2A_CACHE) -> dict:
    with np.load(path, allow_pickle=True) as d:
        out = {
            "task": "bci2a_left_right",
            "x": d["x"].astype(np.float32),
            "y": d["y"].astype(np.int64),
            "subject": d["subject"].astype(np.int64),
            "session": d["session"].astype(np.int64),
            "run": d["run"].astype(np.int64),
            "dataset": "bci2a",
        }
    out["run_idx"] = (out["run"] % 100).astype(np.int64)  # run within session, 0..5
    return out


# ----------------------------------------------------------------------------
# splits (plan B3) — each returns a list of (train_idx, test_idx, fold_name)
# ----------------------------------------------------------------------------
def split_trial_random(y: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te, f"trial{k}") for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))]


def split_run_disjoint(run_idx: np.ndarray):
    """Fold k holds out run index k of every subject (subject-overlapping, run-disjoint)."""
    out = []
    idx = np.arange(len(run_idx))
    for k in np.unique(run_idx):
        te = idx[run_idx == k]
        tr = idx[run_idx != k]
        out.append((tr, te, f"run{int(k)}"))
    return out


def split_session_disjoint(session: np.ndarray):
    out = []
    idx = np.arange(len(session))
    for k in np.unique(session):
        te = idx[session == k]
        tr = idx[session != k]
        out.append((tr, te, f"sess{int(k)}"))
    return out


def split_subject_disjoint(subject: np.ndarray, n_splits: int, seed: int):
    """Subject-disjoint folds: subjects permuted by `seed`, then cut into n_splits balanced chunks."""
    rng = np.random.default_rng(seed)
    subs = np.unique(subject)
    perm = rng.permutation(subs)
    chunks = np.array_split(perm, n_splits)
    idx = np.arange(len(subject))
    out = []
    for k, chunk in enumerate(chunks):
        if len(chunk) == 0:  # fewer subjects than folds (smoke runs only)
            continue
        mask = np.isin(subject, chunk)
        out.append((idx[~mask], idx[mask], f"subj{k}"))
    return out


def split_loso(subject: np.ndarray):
    idx = np.arange(len(subject))
    out = []
    for s in np.unique(subject):
        mask = subject == s
        out.append((idx[~mask], idx[mask], f"loso{int(s)}"))
    return out


def ladder_splits(data: dict, protocol: str, seed: int, n_splits: int | None = None):
    """Protocol ladder of plan B3 for one dataset dict."""
    if protocol == "trial_random_sc":
        return split_subject_class_stratified(data["y"], data["subject"], n_splits or 3, seed)
    if protocol == "trial_random":
        return split_trial_random(data["y"], n_splits or 3, seed)
    if protocol == "run_disjoint":
        return split_run_disjoint(data["run_idx"])
    if protocol == "session_disjoint":
        return split_session_disjoint(data["session"])
    if protocol == "subject_disjoint":
        return split_subject_disjoint(data["subject"], n_splits or 3, seed)
    if protocol == "loso":
        return split_loso(data["subject"])
    raise ValueError(protocol)


def train_fraction(splits, n: int) -> float:
    return float(np.mean([len(tr) / n for tr, _, _ in splits]))


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def covariances(x: np.ndarray) -> np.ndarray:
    from pyriemann.estimation import Covariances

    return Covariances(estimator="oas").transform(x.astype(np.float64))


def riemann_mean(covs: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    from pyriemann.utils.mean import mean_riemann

    return mean_riemann(covs, init=init)


def tangent(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    from pyriemann.utils.tangentspace import tangent_space

    return tangent_space(covs, ref, metric="riemann")


def fold_tangent_features(covs: np.ndarray, tr: np.ndarray, te: np.ndarray, ref: np.ndarray | None = None):
    """Inductive tangent features: reference = Riemannian mean of the TRAINING covariances."""
    if ref is None:
        ref = riemann_mean(covs[tr])
    return tangent(covs[tr], ref), tangent(covs[te], ref), ref


def recenter_by_subject(covs: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Riemannian recentering: C_i -> G_s^{-1/2} C_i G_s^{-1/2}, G_s = Frechet mean of subject s (all trials)."""
    from pyriemann.utils.base import invsqrtm

    out = np.empty_like(covs)
    for s in np.unique(subject):
        m = subject == s
        g = riemann_mean(covs[m])
        w = invsqrtm(g)
        out[m] = w @ covs[m] @ w
    return out


# ----------------------------------------------------------------------------
# accuracy bookkeeping and statistics (plan B2)
# ----------------------------------------------------------------------------
def per_subject_acc(y: np.ndarray, pred: np.ndarray, subject: np.ndarray, mask: np.ndarray | None = None) -> dict[int, float]:
    out = {}
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    for s in np.unique(subject[mask]):
        m = mask & (subject == s)
        out[int(s)] = float(np.mean(y[m] == pred[m]))
    return out


def bootstrap_ci(values, n: int = BOOT_N, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


ROUND_DECIMALS = 12  # differences are rounded before ranking so that mathematically tied CSV values stay tied


def _exact_signed_rank_p(d: np.ndarray) -> tuple[float, float]:
    """Exact conditional signed-rank test (two-sided): zeros dropped, ties get average ranks, and the null
    distribution is the enumeration of all 2^n sign assignments on the OBSERVED rank vector (n <= 25)."""
    from scipy.stats import rankdata

    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    r = rankdata(np.abs(d))
    w_plus = float(np.sum(r[d > 0]))
    total = float(np.sum(r))
    # distribution of W+ over all sign assignments via dynamic programming on (scaled) integer ranks
    scale = 2  # average ranks are multiples of 0.5
    ri = np.round(r * scale).astype(int)
    dist = np.zeros(int(np.sum(ri)) + 1)
    dist[0] = 1.0
    for x in ri:
        new_dist = dist.copy()
        new_dist[x:] += dist[: len(dist) - x]
        dist = new_dist
    dist /= dist.sum()
    w_int = int(round(w_plus * scale))
    mean_int = np.sum(ri) / 2.0
    dev = abs(w_int - mean_int)
    lo = int(np.floor(mean_int - dev + 1e-9)); hi = int(np.ceil(mean_int + dev - 1e-9))
    p = float(dist[: max(lo, 0) + 1].sum() + dist[hi:].sum()) if lo <= hi else 1.0
    p = min(1.0, p)
    return min(w_plus, total - w_plus), p


def wilcoxon_paired(a, b) -> tuple[float, float]:
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.round(a - b, ROUND_DECIMALS)
    if np.all(d == 0):
        return 0.0, 1.0
    if len(d) <= 25:
        return _exact_signed_rank_p(d)
    res = wilcoxon(d, method="auto")
    return float(res.statistic), float(res.pvalue)


def exact_floor_p(n_nonzero: int) -> float:
    """Smallest attainable two-sided exact p for n nonzero, untied differences."""
    return 2.0 / (2 ** n_nonzero) if n_nonzero > 0 else 1.0


def signflip_perm_p(d, n_perm: int = N_PERM, seed: int = SEED) -> float:
    d = np.round(np.asarray(d, dtype=float), ROUND_DECIMALS)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def tost(d, margin: float = TOST_MARGIN) -> dict:
    """Two one-sided tests for the mean of paired differences within (-margin, +margin)."""
    from scipy.stats import ttest_1samp, t as tdist

    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return {"n": n, "p_lower": float("nan"), "p_upper": float("nan"), "ci90_low": float("nan"), "ci90_high": float("nan"), "equivalent": False}
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {"n": n, "p_lower": 0.0, "p_upper": 0.0, "ci90_low": float(mean), "ci90_high": float(mean), "equivalent": bool(abs(mean) < margin)}
    p_lower = float(ttest_1samp(d, -margin, alternative="greater").pvalue)
    p_upper = float(ttest_1samp(d, margin, alternative="less").pvalue)
    h = tdist.ppf(0.95, n - 1) * se
    return {
        "n": n,
        "mean": float(mean),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "ci90_low": float(mean - h),
        "ci90_high": float(mean + h),
        "equivalent": bool(max(p_lower, p_upper) < 0.05),
        "margin": margin,
    }


def summarize_subject_vector(values, seed: int = SEED) -> dict:
    v = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(v, seed=seed)
    return {"n": int(len(v)), "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), "ci_low": lo, "ci_high": hi}


# ----------------------------------------------------------------------------
# io, manifests, validation
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "sklearn", "mne", "pyriemann", "moabb", "braindecode", "torch"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[m] = "not_imported"
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    if fieldnames is None:
        fieldnames = []
        for r in rows:  # union of keys in order of first appearance (rows may gain columns, e.g. Holm-adjusted p)
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate_csv(path: Path, required: list[str], min_rows: int) -> int:
    rows = read_csv(path)
    if len(rows) < min_rows:
        raise RuntimeError(f"{path}: {len(rows)} rows < {min_rows}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    return len(rows)


def write_manifest(path: Path, run_id: str, started: str, outputs: list[Path], extra: dict | None = None) -> Path:
    plan_hash = sha256_file(PLAN) if PLAN.exists() else "missing"
    man = {
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "git_head": git_head(),
        "plan_sha256": plan_hash,
        "versions": versions(),
        "hostname": platform.node(),
        "started_utc": started,
        "finished_utc": utc_now(),
        "outputs": {str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p): sha256_file(p) for p in outputs},
    }
    if extra:
        man.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return path


def fmt(v, nd: int = 6) -> str:
    try:
        return f"{float(v):.{nd}g}"
    except Exception:  # noqa: BLE001
        return str(v)


# ----------------------------------------------------------------------------
# revision 2 matched controls (frozen plan v2 B20--B21)
# ----------------------------------------------------------------------------
def assert_subject_class_stratified_splits(y, subject, splits, n_splits=None):
    """Fail closed on cell imbalance, duplicate coverage or incomplete complements."""
    y, subject = np.asarray(y), np.asarray(subject)
    assert y.ndim == subject.ndim == 1 and len(y) == len(subject) > 0
    k = len(splits) if n_splits is None else n_splits
    assert k >= 2 and len(splits) == k
    idx = np.arange(len(y))
    coverage = np.zeros(len(y), dtype=int)
    for tr, te, _ in splits:
        tr, te = np.asarray(tr), np.asarray(te)
        for part in (tr, te):
            assert part.ndim == 1 and np.issubdtype(part.dtype, np.integer)
            assert np.all((part >= 0) & (part < len(y)))
            assert len(np.unique(part)) == len(part), "duplicate trial within fold"
        assert not np.intersect1d(tr, te).size, "overlapping train/test"
        assert np.array_equal(np.sort(np.r_[tr, te]), idx), "incomplete train/test complement"
        coverage[te] += 1
        for s in np.unique(subject):
            for c in np.unique(y):
                n = int(np.sum((subject == s) & (y == c)))
                n_test = int(np.sum((subject[te] == s) & (y[te] == c)))
                assert n_test in {n // k, (n + k - 1) // k}, "unbalanced subject/class cell"
    assert np.all(coverage == 1), "every trial must occur in exactly one test fold"


def split_subject_class_stratified(y, subject, n_splits, seed):
    """P0s: shuffle each subject/class cell and deal with a cycling cell offset."""
    y, subject = np.asarray(y), np.asarray(subject)
    assert y.ndim == subject.ndim == 1 and len(y) == len(subject) > 0
    assert isinstance(n_splits, (int, np.integer)) and n_splits >= 2
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1, dtype=int)
    cell = 0
    for s in np.unique(subject):
        for c in np.unique(y):
            idx = np.flatnonzero((subject == s) & (y == c))
            shuffled = rng.permutation(idx)
            fold[shuffled] = (np.arange(len(idx)) + cell % n_splits) % n_splits
            cell += 1
    splits = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k), f"trial_sc{k}")
              for k in range(n_splits)]
    assert_subject_class_stratified_splits(y, subject, splits, n_splits)
    return splits


def p0s_construction_table(data, splits, protocol="trial_random_sc", seed=None):
    """Actual cell/run counts and discrepancies from the corresponding P1 holdout.

    The two-fold IV-2a protocol is paired with P1s (actual sessions); the
    six-fold protocol is paired with P1 (actual runs across both sessions).
    Rows with run='all' give complete subject/class counts. Other rows give
    each physical recording's contribution, including zero contributions.
    """
    y, subject = data["y"], data["subject"]
    assert_subject_class_stratified_splits(y, subject, splits)
    session_comparator = data.get("dataset") == "bci2a" and len(splits) == 2
    p1 = split_session_disjoint(data["session"]) if session_comparator else split_run_disjoint(data["run_idx"])
    assert len(p1) == len(splits), "P0s and its actual P1 comparator need the same number of folds"
    actual_run = data.get("run_number", data.get("run", data["run_idx"]))
    rows = []
    for (tr, te, fold), (p1_tr, p1_te, p1_fold) in zip(splits, p1):
        for s in np.unique(subject):
            for c in np.unique(y):
                cell = (subject == s) & (y == c)
                for run in ["all", *np.unique(actual_run[subject == s]).tolist()]:
                    mask = cell if run == "all" else cell & (actual_run == run)
                    n_tr, n_te = int(mask[tr].sum()), int(mask[te].sum())
                    n_p1_tr, n_p1_te = int(mask[p1_tr].sum()), int(mask[p1_te].sum())
                    rows.append({"dataset": data.get("dataset", "eegbci"), "task": data.get("task", ""),
                                 "protocol": protocol, "seed": seed, "fold": fold,
                                 "p1_protocol": "session_disjoint" if session_comparator else "run_disjoint",
                                 "p1_fold": p1_fold, "subject": int(s), "run": run, "class": int(c),
                                 "n_total": int(mask.sum()), "n_train": n_tr, "n_test": n_te,
                                 "p1_n_train": n_p1_tr, "p1_n_test": n_p1_te,
                                 "delta_train": n_tr - n_p1_tr, "delta_test": n_te - n_p1_te})
    return rows


def fixed_test_arms(data, dataset, seed, rotation):
    """Construct E17's matched arms, fixed test and common BG reference/validation.

    Rotation is zero-based. Cache rows preserve recording order; sorting global
    indices is therefore the recording-time ordering within a cell. Full EEGBCI
    uses three groups of 36 and one leftover. Capped smoke data use floor(N/3)
    subjects per group and explicitly report *every* leftover.

    A2 deliberately substitutes the final run/session into the preceding
    run/session's training *slot*. Physical run counts cannot be equal after
    that substitution: ``counts`` releases both actual runs and matched slots.
    Equality assertions apply to the slots, with explicitly recorded excess
    when h2 is larger than the retained source cell (the plan's shortfall).
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    assert dataset in {"eegbci", "bci2a"} and rotation in (0, 1, 2)
    y, subject = np.asarray(data["y"]), np.asarray(data["subject"])
    assert y.ndim == subject.ndim == 1 and len(y) == len(subject) > 0
    subjects = np.unique(subject)
    group_size = len(subjects) // 3
    assert group_size > 0, "E17 requires three nonempty subject blocks"
    perm = np.random.default_rng(seed).permutation(subjects)
    groups = [perm[k * group_size:(k + 1) * group_size] for k in range(3)]
    leftover = perm[3 * group_size:]
    target, donor, background = [groups[(rotation + k) % 3] for k in range(3)]
    if dataset == "eegbci":
        source = np.asarray(data["run_idx"])
        assert np.array_equal(np.unique(source), [0, 1, 2]), "EEGBCI requires three ordered runs"
        train_units, last_unit = [0, 1], 2
    else:
        source = np.asarray(data["session"])
        sessions = np.unique(source)
        assert len(sessions) == 2, "IV-2a requires two ordered sessions"
        train_units, last_unit = [sessions[0]], sessions[1]
    classes = np.unique(y)
    assert len(classes) == 2, "E17 is a two-class design"

    def cell(s, run, c):
        return np.flatnonzero((subject == s) & (source == run) & (y == c))

    def ranked(block):
        return sorted(block.tolist(), key=lambda s: (int(np.sum(subject == s)), int(s)))

    bg = np.flatnonzero(np.isin(subject, background) & np.isin(source, train_units))
    test, h2, kept_t, kept_d, replacement = [], [], [], [], []
    pairing = []
    shortfall_by_class = {int(c): 0 for c in classes}
    for pair, (s_t, s_d) in enumerate(zip(ranked(target), ranked(donor))):
        for c in classes:
            last = cell(s_t, last_unit, c)
            test.extend(last[::2])
            adjacent = last[1::2]
            h2.extend(adjacent)
            for run in train_units:
                ct, cd = cell(s_t, run, c), cell(s_d, run, c)
                n = min(len(ct), len(cd))
                kt, kd = ct[:n], cd[:n]
                kept_t.extend(kt)
                kept_d.extend(kd)
                replacing = run == train_units[-1]
                deficit = max(0, len(adjacent) - n) if replacing else 0
                if replacing:
                    replacement.extend(kt[:max(0, n - len(adjacent))])
                    replacement.extend(adjacent)
                    shortfall_by_class[int(c)] += deficit
                else:
                    replacement.extend(kt)
                pairing.append({"dataset": dataset, "task": data.get("task", ""), "seed": int(seed),
                                "rotation": int(rotation), "pair": pair, "subject_T": int(s_t), "subject_D": int(s_d),
                                "run": int(run), "class": int(c), "n_T": len(ct), "n_D": len(cd),
                                "n_kept": n, "shortfall": deficit})
    arms = {"A0": np.sort(np.r_[bg, np.asarray(kept_d, dtype=int)]),
            "A1": np.sort(np.r_[bg, np.asarray(kept_t, dtype=int)]),
            "A2": np.sort(np.r_[bg, np.asarray(replacement, dtype=int)])}
    test, h2 = np.sort(np.asarray(test, dtype=int)), np.sort(np.asarray(h2, dtype=int))
    assert len(test) > 0 and len(bg) > 0
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    _, val_local = next(val_split.split(np.zeros(len(bg)), y[bg]))
    validation = np.sort(bg[val_local])
    counts = []
    actual_run = np.asarray(data.get("run_number", data.get("run", data["run_idx"])))
    for arm, idx in arms.items():
        slots = source[idx].copy()
        if arm == "A2":
            slots[np.isin(idx, h2)] = train_units[-1]
        for c in classes:
            for run in np.unique(actual_run):
                mask = (y[idx] == c) & (actual_run[idx] == run)
                counts.append({"dataset": dataset, "task": data.get("task", ""), "seed": int(seed), "rotation": int(rotation),
                               "arm": arm, "count_type": "actual_run", "run": int(run), "class": int(c),
                               "n_trials": int(mask.sum()), "n_subjects": len(np.unique(subject[idx][mask])),
                               "shortfall": 0})
            for run in train_units:
                mask = (y[idx] == c) & (slots == run)
                deficit = shortfall_by_class[int(c)] if arm == "A2" and run == train_units[-1] else 0
                counts.append({"dataset": dataset, "task": data.get("task", ""), "seed": int(seed), "rotation": int(rotation),
                               "arm": arm, "count_type": "matched_slot", "run": int(run), "class": int(c),
                               "n_trials": int(mask.sum()), "n_subjects": len(np.unique(subject[idx][mask])),
                               "shortfall": deficit})
    out = {"test": test, "arms": arms, "bg": bg, "reference_indices": bg,
           "validation": validation, "reference_indices_by_arm": {a: bg for a in arms},
           "validation_indices_by_arm": {a: validation for a in arms},
           "leftover_subject": int(leftover[0]) if len(leftover) == 1 else (leftover.tolist() if len(leftover) else None),
           "leftover_subjects": leftover.tolist(), "pairing": pairing, "counts": counts,
           "groups": {"T": target, "D": donor, "B": background, **{f"G{k + 1}": g for k, g in enumerate(groups)}},
           "h2": h2, "dataset": dataset, "seed": int(seed), "rotation": int(rotation),
           "group_size": group_size, "shortfall_by_class": shortfall_by_class}
    assert_fixed_test_arms(data, out)
    return out


def assert_fixed_test_arms(data, design):
    """Check released E17 construction, including shared preprocessing sources."""
    from sklearn.model_selection import StratifiedShuffleSplit

    y, subject = np.asarray(data["y"]), np.asarray(data["subject"])
    arms, bg, test, h2 = design["arms"], design["bg"], design["test"], design["h2"]
    target, donor, background = [design["groups"][k] for k in ("T", "D", "B")]
    groups = [target, donor, background]
    perm = np.random.default_rng(design["seed"]).permutation(np.unique(subject))
    expected_groups = [perm[k * design["group_size"]:(k + 1) * design["group_size"]] for k in range(3)]
    for offset, group in enumerate(groups):
        assert np.array_equal(group, expected_groups[(design["rotation"] + offset) % 3]), "seeded subject blocks differ"
    assert np.array_equal(design["leftover_subjects"], perm[3 * design["group_size"]:]), "incorrect leftover reporting"
    assert all(len(g) == design["group_size"] for g in groups)
    assert len(np.unique(np.concatenate(groups))) == 3 * design["group_size"]
    assert not np.intersect1d(np.concatenate(groups), design["leftover_subjects"]).size
    assert np.array_equal(np.sort(np.r_[np.concatenate(groups), design["leftover_subjects"]]), np.unique(subject))
    source = data["run_idx"] if design["dataset"] == "eegbci" else data["session"]
    units = np.unique(source)
    train_units, last_unit = units[:-1], units[-1]
    expected_bg = np.flatnonzero(np.isin(subject, background) & np.isin(source, train_units))
    assert np.array_equal(bg, expected_bg), "background must contain all prior-run/session trials"
    expected_test, expected_h2 = [], []
    for s in target:
        for c in np.unique(y):
            idx = np.flatnonzero((subject == s) & (source == last_unit) & (y == c))
            expected_test.extend(idx[::2])
            expected_h2.extend(idx[1::2])
    assert np.array_equal(test, np.sort(expected_test)), "h1 must use alternating even positions"
    assert np.array_equal(h2, np.sort(expected_h2)), "h2 must use alternating odd positions"
    assert not np.intersect1d(test, h2).size
    assert np.array_equal(design["reference_indices"], bg), "reference must use exactly BG"
    assert np.isin(design["validation"], bg).all(), "validation must use only BG"
    _, expected_val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=design["seed"]).split(np.zeros(len(bg)), y[bg]))
    assert np.array_equal(design["validation"], np.sort(bg[expected_val])), "validation must be the seeded 15% BG slice"
    for arm, idx in arms.items():
        assert np.issubdtype(idx.dtype, np.integer) and np.all((idx >= 0) & (idx < len(y)))
        assert len(idx) == len(np.unique(idx)), "duplicate arm trial"
        assert not np.intersect1d(test, idx).size, "test trial in training arm"
        assert not np.isin(subject[idx], design["leftover_subjects"]).any(), "leftover subject in arm"
        expected_subjects = np.r_[background, donor if arm == "A0" else target]
        assert np.array_equal(np.unique(subject[idx]), np.sort(expected_subjects)), "incorrect training subjects"
        assert len(np.unique(subject[idx])) == 2 * design["group_size"]
        assert np.isin(bg, idx).all(), "each arm must retain the unchanged BG"
        assert np.array_equal(design["reference_indices_by_arm"][arm], bg), "moving reference source"
        assert np.array_equal(design["validation_indices_by_arm"][arm], design["validation"]), "moving validation slice"
    ranked = lambda block: sorted(block.tolist(), key=lambda s: (int(np.sum(subject == s)), int(s)))
    pair_ids = sorted({(r["pair"], r["subject_T"], r["subject_D"]) for r in design["pairing"]})
    assert pair_ids == [(p, int(t), int(d)) for p, (t, d) in enumerate(zip(ranked(target), ranked(donor)))], "rank pairing mismatch"
    expected_arms = {a: list(bg) for a in arms}
    for row in design["pairing"]:
        c, run, st, sd = [row[k] for k in ("class", "run", "subject_T", "subject_D")]
        ct = np.flatnonzero((subject == st) & (source == run) & (y == c))
        cd = np.flatnonzero((subject == sd) & (source == run) & (y == c))
        n = min(len(ct), len(cd))
        assert (row["n_T"], row["n_D"], row["n_kept"]) == (len(ct), len(cd), n)
        adjacent = h2[(subject[h2] == st) & (y[h2] == c)]
        replacing = run == train_units[-1]
        deficit = max(0, len(adjacent) - n) if replacing else 0
        assert row["shortfall"] == deficit, "unreported h2 shortfall"
        expected_arms["A0"].extend(cd[:n])
        expected_arms["A1"].extend(ct[:n])
        expected_arms["A2"].extend(ct[:max(0, n - len(adjacent))] if replacing else ct[:n])
        if replacing:
            expected_arms["A2"].extend(adjacent)
    for arm in arms:
        assert np.array_equal(arms[arm], np.sort(expected_arms[arm])), "arm does not implement matched cell substitution"
    for c in np.unique(y):
        a0, a1, a2 = [arms[a] for a in ("A0", "A1", "A2")]
        assert np.sum(y[a0] == c) == np.sum(y[a1] == c), "A0/A1 class count mismatch"
        assert np.sum(y[a2] == c) - np.sum(y[a1] == c) == design["shortfall_by_class"][int(c)]
        for run in train_units:
            n0 = int(np.sum((y[a0] == c) & (source[a0] == run)))
            n1 = int(np.sum((y[a1] == c) & (source[a1] == run)))
            assert n0 == n1, "A0/A1 run/class count mismatch"
            n2 = int(np.sum((y[a2] == c) & (source[a2] == run)))
            if run == train_units[-1]:
                n2 += int(np.sum(y[h2] == c))
            expected_excess = design["shortfall_by_class"][int(c)] if run == train_units[-1] else 0
            assert n2 - n1 == expected_excess, "A1/A2 matched slot/class count mismatch"
