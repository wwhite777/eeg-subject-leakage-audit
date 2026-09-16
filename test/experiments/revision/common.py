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


def wilcoxon_paired(a, b) -> tuple[float, float]:
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if np.allclose(d, 0):
        return 0.0, 1.0
    method = "exact" if len(d) <= 25 else "auto"
    try:
        res = wilcoxon(a, b, method=method)
    except ValueError:
        res = wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)


def signflip_perm_p(d, n_perm: int = N_PERM, seed: int = SEED) -> float:
    d = np.asarray(d, dtype=float)
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
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
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
