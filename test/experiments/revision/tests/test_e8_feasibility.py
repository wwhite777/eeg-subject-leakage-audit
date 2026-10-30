#!/usr/bin/env python3
"""Synthetic E8 regression tests; run directly with Python or collect with pytest."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import e8_factorial as E  # noqa: E402


def old_per_fold_predictions(data, covs, d_grid):
    """Minimal released TS+LR logic: skip a dimension separately in each fold."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    y, subject = data["y"], data["subject"]
    result = {}
    for protocol, splits in (
        ("trial_random", E.C.split_trial_random(y, 3, E.C.SEED)),
        ("subject_disjoint", E.C.split_subject_disjoint(subject, 3, E.C.SEED)),
    ):
        preds = {}
        for tr, te, _ in splits:
            xtr, xte, _ = E.C.fold_tangent_features(covs, tr, te)
            full_d = xtr.shape[1]
            max_pca = min(max(d for d in d_grid if d < full_d), len(tr) - 1)
            pca = PCA(n_components=max_pca, svd_solver="randomized", random_state=E.C.SEED).fit(xtr)
            ztr, zte = pca.transform(xtr), pca.transform(xte)
            for d in d_grid:
                if d >= full_d:
                    a, b = xtr, xte
                elif d > max_pca:
                    continue
                else:
                    a, b = ztr[:, :d], zte[:, :d]
                clf = LogisticRegression(max_iter=3000, C=1.0, random_state=E.C.SEED).fit(a, y[tr])
                preds.setdefault(d, np.full(len(y), -1))[te] = clf.predict(b)
        result[protocol] = preds
    return result


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # MNE reads its configuration with a write lock even during import.
        mne_home = os.environ.get("_MNE_FAKE_HOME_DIR")
        if mne_home is None:
            scratch = Path(__file__).resolve().parent / ".scratch"
            scratch.mkdir(exist_ok=True)
            mne_home = tempfile.mkdtemp(prefix="mne-", dir=scratch)
        env = patch.dict(os.environ, {"_MNE_FAKE_HOME_DIR": mne_home})
        env.start()
        cls.addClassCleanup(env.stop)
        rng = np.random.default_rng(812)
        counts = [4, 6, 8]
        subject = np.repeat([1, 2, 3], counts)
        y = np.concatenate([np.arange(n) % 2 for n in counts])
        a = rng.normal(size=(len(y), 6, 6))
        cls.covs = a @ a.transpose(0, 2, 1) / 6 + np.eye(6)
        cls.data = {"task": "synthetic", "y": y, "subject": subject,
                    "x": rng.normal(size=(len(y), 6, 64))}
        # P0 train sizes = 12,12,12; P2 = a permutation of 10,12,14; full_d = 21.
        cls.d_grid = [2, 9, 10, 11, 14, 21]

    def test_old_skip_counts_missing_predictions_as_misses(self):
        old = old_per_fold_predictions(self.data, self.covs, self.d_grid)
        y, subject = self.data["y"], self.data["subject"]
        for d in (10, 11):
            self.assertTrue(np.all(old["trial_random"][d] != -1))
            pred = old["subject_disjoint"][d]
            missing = pred == -1
            self.assertTrue(missing.any())
            acc = E.C.per_subject_acc(y, pred, subject)
            for s in np.unique(subject[missing]):
                self.assertEqual(acc[int(s)], 0.0)
            repaired = pred.copy()
            repaired[missing] = y[missing]
            repaired_acc = E.C.per_subject_acc(y, repaired, subject)
            self.assertGreater(np.mean(list(repaired_acc.values())), np.mean(list(acc.values())))
        self.assertNotIn(14, old["trial_random"])
        self.assertNotIn(14, old["subject_disjoint"])

    def test_new_mask_boundaries_both_protocols_and_coverage(self):
        scored = []
        per_subject_acc = E.C.per_subject_acc

        def checked_acc(y, pred, subject):
            self.assertTrue(np.all(pred != -1))
            scored.append(len(pred))
            return per_subject_acc(y, pred, subject)

        with patch.object(E.C, "per_subject_acc", side_effect=checked_acc):
            rows, feasibility, coverage = E.run_draw(
                self.data, self.covs, 3, 0, self.d_grid, ["riemann_ts_lr", "csp_lda"])
        expected = {2: 1, 9: 1, 10: 0, 11: 0, 14: 0, 21: 1}
        self.assertEqual({r["d"]: r["feasible"] for r in feasibility}, expected)
        self.assertTrue(all(r["min_train_fold"] == 10 for r in feasibility))
        self.assertTrue(all(r["min_train_fold_P0"] == 12 and r["min_train_fold_P2"] == 10 for r in feasibility))
        for protocol in ("trial_random", "subject_disjoint"):
            self.assertEqual({r["d"] for r in rows if r["protocol"] == protocol and r["model"] == "riemann_ts_lr"}, {2, 9, 21})
            self.assertEqual(len([r for r in rows if r["protocol"] == protocol and r["model"] == "csp_lda"]), 1)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(scored), len(rows))
        self.assertEqual(len(coverage), len(rows))
        self.assertTrue(all(r["n_predicted"] == r["n_trials"] == 18 for r in coverage))
        self.assertTrue(all(set(r) == {"model", "N", "d", "draw", "protocol", "mean_acc", "n_subjects", "n_trials"} for r in rows))

    def test_pca_fit_is_independent_of_feasible_grid_and_retained_values_match(self):
        from sklearn.decomposition import PCA

        with patch("sklearn.decomposition.PCA", wraps=PCA) as pca:
            rows, _, _ = E.run_draw(self.data, self.covs, 3, 0, self.d_grid, ["riemann_ts_lr"])
        splits = E.C.split_trial_random(self.data["y"], 3, E.C.SEED) + E.C.split_subject_disjoint(self.data["subject"], 3, E.C.SEED)
        self.assertEqual([c.kwargs["n_components"] for c in pca.call_args_list], [min(512, len(tr) - 1) for tr, _, _ in splits])
        self.assertTrue(all(c.kwargs["random_state"] == E.C.SEED for c in pca.call_args_list))
        old = old_per_fold_predictions(self.data, self.covs, self.d_grid)
        for r in rows:
            acc = E.C.per_subject_acc(self.data["y"], old[r["protocol"]][r["d"]], self.data["subject"])
            self.assertAlmostEqual(r["mean_acc"], np.mean(list(acc.values())), places=12)

    def test_coverage_assertion_rejects_injected_minus_one_before_accuracy(self):
        from sklearn.linear_model import LogisticRegression

        predict = LogisticRegression.predict

        def incomplete_predict(clf, x):
            pred = predict(clf, x)
            pred[0] = -1
            return pred

        with patch.object(LogisticRegression, "predict", incomplete_predict), patch.object(E.C, "per_subject_acc") as acc:
            with self.assertRaisesRegex(AssertionError, r"task=synthetic, model=riemann_ts_lr, N=3, d=2, draw=0, protocol=trial_random"):
                E.run_draw(self.data, self.covs, 3, 0, self.d_grid, ["riemann_ts_lr"])
            acc.assert_not_called()


class AggregationTests(unittest.TestCase):
    def test_stats_require_four_paired_paradigms_and_report_counts_and_exclusions(self):
        import e_stats as S

        rows, feasibility = [], []
        for task_index, task in enumerate(E.C.TASKS):
            for N in (18, 36):
                for d in (8, 512):
                    for draw in range(10):
                        missing = task == "real_fists_feet" and N == 18 and d == 512 and draw == 6
                        feasibility.append({"task": task, "N": N, "d": d, "draw": draw, "feasible": int(not missing)})
                        if missing:
                            continue
                        value = 0.01 * (task_index + 1) + 0.001 * draw
                        # A surviving three-paradigm draw must not contaminate the aggregate.
                        if N == 18 and d == 512 and draw == 6:
                            value += 0.3
                        for protocol, accuracy in (("trial_random", 0.5 + value), ("subject_disjoint", 0.5)):
                            rows.append({"task": task, "model": "riemann_ts_lr", "N": N, "d": d,
                                         "draw": draw, "protocol": protocol, "mean_acc": accuracy})
        source = Path("synthetic_per_draw.csv")
        captured = []

        def read_csv(path):
            return rows if path == source else feasibility

        def write_csv(path, result):
            captured.extend(result)
            return path

        with patch.object(S.C, "read_csv", side_effect=read_csv), patch.object(S.C, "write_csv", side_effect=write_csv), \
                patch.object(S.C, "write_manifest"), patch.object(Path, "exists", return_value=True), \
                patch.object(sys, "argv", ["e_stats.py", "--tag", "synthetic", "--e8", str(source)]), redirect_stdout(io.StringIO()):
            self.assertEqual(S.main(), 0)
        cell = next(r for r in captured if r["task"] == "paradigm_mean" and r["N"] == 18 and r["d"] == 512)
        complete_values = np.array([0.025 + 0.001 * draw for draw in range(10) if draw != 6])
        self.assertEqual(cell["n_draws"], 9)
        self.assertEqual(cell["n_obs"], 9)
        self.assertEqual(cell["inflation_mean"], E.C.fmt(complete_values.mean()))
        lo, hi = E.C.bootstrap_ci(complete_values)
        self.assertEqual((cell["ci_low"], cell["ci_high"]), (E.C.fmt(lo), E.C.fmt(hi)))
        self.assertEqual(cell["excluded_conditions"], "real_fists_feet:18:6:512")
        self.assertEqual({r["task"] for r in captured if r["N"] == "ols"}, set(E.C.TASKS))
        for r in captured:
            if r["N"] == "ols":
                self.assertEqual(r["n_obs"], 39 if r["task"] in ("paradigm_mean", "real_fists_feet") else 40)


if __name__ == "__main__":
    unittest.main()
