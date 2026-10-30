"""Regression checks for frozen B18/B19; run with venv python -m unittest."""

import csv
import os
from itertools import permutations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e3v2_subject_info as E


class SequentialSSTests(unittest.TestCase):
    def fixture(self):
        # Counts (3,1), (1,3); the only signal is the subject offset 0 or 1.
        subject = np.repeat([0, 1], 4)
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        return subject[:, None].astype(float), subject, y

    def assert_partition(self, result):
        total = result["rss"]["RSS0"]
        for terms in result["orders"].values():
            self.assertTrue(all(v["ss"] >= 0 for v in terms.values()))
            self.assertAlmostEqual(sum(v["ss"] for v in terms.values()) / total, 1, delta=1e-9)
            self.assertAlmostEqual(sum(v["fraction"] for v in terms.values()), 1, delta=1e-9)

    def test_exact_unbalanced_offsets_fixture(self):
        feats, subject, y = self.fixture()
        result = E.variance_decomposition(feats, subject, y)
        for key, expected in dict(RSS0=2, RSS_S=0, RSS_C=1.5, RSS_SC=0, RSS_cell=0).items():
            self.assertAlmostEqual(result["rss"][key], expected, delta=1e-9 * result["rss"]["RSS0"])
        for order, expected in (("A", [2, 0, 0, 0]), ("B", [0.5, 1.5, 0, 0])):
            actual = [result["orders"][order][name]["ss"] for name in E.ORDER_COMPONENTS[order]]
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9 * result["rss"]["RSS0"])
            self.assertLess(abs(result["orders"][order]["subject_x_class_given_both"]["fraction"]), 1e-10)
        # The released marginal subtraction is deliberately falsified by this fixture.
        rss = result["rss"]
        legacy_interaction = (rss["RSS0"] - rss["RSS_cell"]) - (rss["RSS0"] - rss["RSS_S"]) - (rss["RSS0"] - rss["RSS_C"])
        self.assertAlmostEqual(legacy_interaction, -0.5, delta=1e-9)
        self.assert_partition(result)

    def test_true_positive_interaction(self):
        feats, subject, y = self.fixture()
        # This contrast cannot be represented by additive subject and class means.
        feats = feats + ((2 * subject - 1) * (2 * y - 1))[:, None]
        result = E.variance_decomposition(feats, subject, y)
        for order in ("A", "B"):
            self.assertGreater(result["orders"][order]["subject_x_class_given_both"]["fraction"], 0)
        self.assert_partition(result)

    def test_constant_dimension_does_not_turn_roundoff_into_a_failure(self):
        feats, subject, y = self.fixture()
        result = E.variance_decomposition(np.c_[feats, np.ones(len(feats))], subject, y)
        self.assert_partition(result)

    def test_multivariate_fit_against_explicit_indicator_ols(self):
        rng = np.random.default_rng(20260916)
        _, subject, y = self.fixture()
        feats = rng.normal(size=(len(y), 7))
        result = E.variance_decomposition(feats, subject, y)
        # A second, overparameterized encoding includes an explicit intercept in
        # every model. Compare residuals rather than non-identifiable coefficients.
        S, Y = np.eye(2)[subject], np.eye(2)[y]
        cell = np.eye(4)[2 * subject + y]
        ones = np.ones((len(y), 1))
        for key, design in (("RSS0", ones), ("RSS_S", np.c_[ones, S]),
                            ("RSS_C", np.c_[ones, Y]), ("RSS_SC", np.c_[ones, S, Y]),
                            ("RSS_cell", np.c_[ones, cell])):
            fit = design @ np.linalg.lstsq(design, feats, rcond=None)[0]
            self.assertAlmostEqual(result["rss"][key], float(np.sum((feats - fit) ** 2)), delta=1e-9 * result["rss"]["RSS0"])
        self.assert_partition(result)


class IdentityTests(unittest.TestCase):
    def test_each_centring_vector_and_alternating_reference(self):
        # Unequal reference/training/test counts detect accidentally unweighted means.
        subject = np.repeat([1, 2], 9)
        runs = np.tile([0, 0, 0, 0, 1, 1, 2, 2, 2], 2)
        feats = np.arange(len(subject) * 2, dtype=float).reshape(-1, 2) ** 2
        orders = {(s, 0): np.arange(4) for s in (1, 2)}
        for scheme in E.SCHEMES:
            xtr, xte, tr, te = E.centred_splits(feats, subject, runs, 0, 1, 2, scheme, orders)
            for s in (1, 2):
                reference = feats[(subject == s) & (runs == 0)]
                training = feats[(subject == s) & (runs == 1)]
                if scheme == "none":
                    a = b = np.zeros(feats.shape[1])
                elif scheme == "independent_reference":
                    a = b = reference.mean(axis=0)
                elif scheme == "training_run":
                    a = b = training.mean(axis=0)
                elif scheme == "all_trial":
                    a = b = feats[subject == s].mean(axis=0)
                else:
                    a, b = reference[[0, 2]].mean(axis=0), reference[[1, 3]].mean(axis=0)
                np.testing.assert_allclose(xtr[subject[tr] == s], feats[tr][subject[tr] == s] - a)
                np.testing.assert_allclose(xte[subject[te] == s], feats[te][subject[te] == s] - b)

    def test_class_blocked_reference_is_interleaved_before_alternation(self):
        cached_labels = [0, 0, 0, 1, 1, 1]
        recording_labels = [1, 0, 0, 1, 1, 0]
        order = E.interleave_class_rows(cached_labels, recording_labels, "synthetic.edf")
        np.testing.assert_array_equal(order, [3, 0, 1, 4, 5, 2])
        np.testing.assert_array_equal(np.asarray(cached_labels)[order], recording_labels)
        feats = np.array([1, 2, 4, 8, 16, 32, 100, 200], dtype=float)[:, None]
        subject, runs = np.ones(8, dtype=int), np.array([0] * 6 + [1, 2])
        xtr, xte, _, _ = E.centred_splits(
            feats, subject, runs, 0, 1, 2, "split_reference", {(1, 0): order})
        np.testing.assert_allclose(xtr, [[100 - (8 + 2 + 32) / 3]])
        np.testing.assert_allclose(xte, [[200 - (1 + 16 + 4) / 3]])
        with self.assertRaisesRegex(ValueError, "EDF-derived recording orders"):
            E.centred_splits(feats, subject, runs, 0, 1, 2, "split_reference")

    def test_recording_order_matches_verifier_half_labels_for_all_subjects(self):
        task = "imagery_left_right_fist"
        # The chronology table is a private verifier artefact.
        chronology_csv = os.environ.get("E3V2_CHRONOLOGY_CSV")
        path = Path(chronology_csv) if chronology_csv else None
        if path is None or not path.is_file():
            self.skipTest("verifier chronology table not distributed; set E3V2_CHRONOLOGY_CSV to its path to run this check")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        subjects = sorted({int(r["subject"]) for r in rows})
        self.assertEqual(len(subjects), 109)
        offset, agreed = 0, 0
        for s in subjects:
            with self.subTest(subject=s):
                expected = [r for r in rows if int(r["subject"]) == s]
                self.assertEqual({int(r["physical_run"]) for r in expected}, {4})
                with np.load(E.C.EEGBCI_CACHE / f"sub{s:03d}_{task}.npz", allow_pickle=False) as cached:
                    positions = offset + np.flatnonzero(cached["runs"] == 4)
                    offset += len(cached["y"])
                order = E.recording_order(task, s, 4)
                halves = {int(index): "A" if pos % 2 == 0 else "B"
                          for pos, index in enumerate(positions[order])}
                actual = [halves[int(r["cached_global_index"])] for r in expected]
                self.assertEqual(actual, [r["required_half"] for r in expected])
                agreed += len(expected)
        print(f"verifier half-label agreement: {len(subjects)}/{len(subjects)} subjects; "
              f"{agreed}/{len(rows)} labels")

    def test_recording_order_fails_closed_on_missing_edf_or_class_count_mismatch(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "S001R04.edf.*test/env/fetch_eegbci_s3.sh"):
                E.recording_order("imagery_left_right_fist", 1, 4)
        with patch.object(E.np, "load") as load:
            load.return_value.__enter__.return_value = {
                "y": np.array([0, 1]), "runs": np.array([4, 4])}
            with self.assertRaisesRegex(ValueError, "S001R04.edf.*class counts differ"):
                E.recording_order("imagery_left_right_fist", 1, 4)

    def test_all_trial_anticorrelation_below_chance(self):
        # Subject offsets plus independent trial noise; no subject-by-class signal.
        # A high-dimensional finite sample makes the forced centroid dependence visible.
        rng = np.random.default_rng(20260916)
        ns, n_run, dims = 12, 12, 128
        subject = np.repeat(np.arange(ns), 3 * n_run)
        runs = np.tile(np.repeat(np.arange(3), n_run), ns)
        offsets = rng.normal(size=(ns, dims))
        feats = offsets[subject] + rng.normal(size=(len(subject), dims))
        chance = 1 / ns
        raw = E.identity_probe(feats, subject, runs, 0, 1, 2, "none")[0]
        independent = E.identity_probe(feats, subject, runs, 0, 1, 2, "independent_reference")[0]
        all_trial = E.identity_probe(feats, subject, runs, 0, 1, 2, "all_trial")[0]
        self.assertGreater(raw, chance)
        self.assertGreaterEqual(independent, chance)
        self.assertLess(all_trial, chance)
        print(f"synthetic identity: chance={chance:.6g}, raw={raw:.6g}, independent={independent:.6g}, all-trial={all_trial:.6g}")
        xtr, xte, tr, te = E.centred_splits(feats, subject, runs, 0, 1, 2, "all_trial")
        for s in np.unique(subject):
            # The three-run weighted-zero identity includes the reference run.
            total_mean = feats[subject == s].mean(axis=0)
            ref_centred = feats[(subject == s) & (runs == 0)] - total_mean
            weighted_sum = ref_centred.sum(axis=0) + xtr[subject[tr] == s].sum(axis=0) + xte[subject[te] == s].sum(axis=0)
            np.testing.assert_allclose(weighted_sum, 0, atol=1e-12)

    def test_subject_aggregation_and_paired_bootstrap(self):
        task = E.C.TASKS[0]
        rows = []
        for a in permutations(E.C.TASK_RUNS[task]):
            for scheme in E.SCHEMES:
                for subject, accuracy, count in ((1, 0.2, 10), (2, 0.8, 30)):
                    if scheme == "split_reference":
                        accuracy -= 0.1
                    rows.append(dict(task=task, ref_run=a[0], train_run=a[1], test_run=a[2],
                                     scheme=scheme, subject=subject, accuracy=accuracy, n_test=count))
        summary = {r["scheme"]: r for r in E.summarize_identity(rows)}
        r = summary["independent_reference"]
        self.assertAlmostEqual(r["mean"], 0.5)
        self.assertAlmostEqual(r["pooled_trial_accuracy"], 0.65)
        lo, hi = E.C.bootstrap_ci([0.2, 0.8], n=2000, seed=20260916)
        np.testing.assert_allclose([r["ci_low"], r["ci_high"]], [lo, hi])
        d = summary["independent_reference_minus_split_reference"]
        np.testing.assert_allclose([d["mean"], d["ci_low"], d["ci_high"]], 0.1)
        with self.assertRaises(ValueError):
            E.summarize_identity(rows[:-1])


if __name__ == "__main__":
    unittest.main()
