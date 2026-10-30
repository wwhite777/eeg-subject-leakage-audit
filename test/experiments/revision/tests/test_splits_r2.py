"""Independent synthetic checks of frozen revision-2 split/control definitions.

Run with the project's venv Python -m unittest discover -s tests -p
test_splits_r2.py. All training is replaced by capturing fakes in glue tests.
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REVISION = Path(__file__).resolve().parents[1]  # test/experiments/revision (the tests live in its tests/ folder)
sys.path.insert(0, str(REVISION))
import common as C


def synthetic_eeg(n_subjects=10, shortfall=False):
    rows = []
    for s in range(1, n_subjects + 1):
        for r in range(3):
            counts = [8 + s % 3 + (r + c) % 2 for c in range(2)]
            if shortfall and r == 1:
                counts = [2, 3]
            for pos in range(max(counts)):
                for c in range(2):
                    if pos < counts[c]:
                        rows.append((s, r, c))
    subject, run, y = np.asarray(rows).T
    return {"dataset": "eegbci", "task": C.TASKS[0], "subject": subject,
            "run_idx": run, "run_number": np.asarray([4, 8, 12])[run], "y": y,
            "x": np.arange(len(y), dtype=np.float32)[:, None, None]}


def synthetic_bci(n_subjects=9, shortfall=False):
    rows = []
    for s in range(1, n_subjects + 1):
        for session in (0, 1):
            counts = [12 + s % 3 + c for c in range(2)]
            if shortfall and session == 0:
                counts = [3, 4]
            for pos in range(max(counts)):
                for c in range(2):
                    if pos < counts[c]:
                        rows.append((s, session, session * 100 + pos % 6, c))
    subject, session, run, y = np.asarray(rows).T
    return {"dataset": "bci2a", "task": "bci2a_left_right", "subject": subject,
            "session": session, "run": run, "run_idx": run % 100, "y": y,
            "x": np.arange(len(y), dtype=np.float32)[:, None, None]}


def independently_expected_design(data, seed, rotation):
    """Directly implement the plan with Python row lists, without builder helpers."""
    subject, y = data["subject"], data["y"]
    ids = sorted(set(subject.tolist()))
    order = np.random.default_rng(seed).permutation(ids).tolist()
    size = len(ids) // 3
    blocks = [order[i * size:(i + 1) * size] for i in range(3)]
    t, d, b = [blocks[(rotation + i) % 3] for i in range(3)]
    source = data["run_idx"] if data["dataset"] == "eegbci" else data["session"]
    previous = [0, 1] if data["dataset"] == "eegbci" else [0]
    final = 2 if data["dataset"] == "eegbci" else 1

    def indices(s, r, c):
        return [i for i in range(len(y)) if subject[i] == s and source[i] == r and y[i] == c]

    rank = lambda block: sorted(block, key=lambda s: (subject.tolist().count(s), s))
    pairs = list(zip(rank(t), rank(d)))
    bg = [i for i in range(len(y)) if subject[i] in b and source[i] in previous]
    arms = {a: bg.copy() for a in ("A0", "A1", "A2")}
    test, h2, matching = [], [], []
    for p, (st, sd) in enumerate(pairs):
        for c in sorted(set(y.tolist())):
            final_trials = indices(st, final, c)
            test += final_trials[::2]
            adjacent = final_trials[1::2]
            h2 += adjacent
            for run in previous:
                ct, cd = indices(st, run, c), indices(sd, run, c)
                retained = min(len(ct), len(cd))
                arms["A0"] += cd[:retained]
                arms["A1"] += ct[:retained]
                replacing = run == previous[-1]
                arms["A2"] += ct[:max(0, retained - len(adjacent))] + adjacent if replacing else ct[:retained]
                matching.append((p, st, sd, run, c, len(ct), len(cd), retained,
                                 max(0, len(adjacent) - retained) if replacing else 0))
    return {"groups": (t, d, b), "leftovers": order[3 * size:], "test": sorted(test),
            "h2": sorted(h2), "bg": bg, "arms": {a: sorted(v) for a, v in arms.items()},
            "matching": matching}


class SubjectClassSplitsTest(unittest.TestCase):
    def setUp(self):
        self.subject = np.repeat([2, 7, 11], [13, 17, 23])
        self.y = np.r_[np.repeat([0, 1], [5, 8]), np.repeat([0, 1], [11, 6]), np.repeat([0, 1], [10, 13])]

    def test_rotation_offsets_and_seed_exactly_match_independent_dealing(self):
        for k in (2, 3, 6):
            splits = C.split_subject_class_stratified(self.y, self.subject, k, C.SEED)
            rng = np.random.default_rng(C.SEED)
            expected = [set() for _ in range(k)]
            offset = 0
            for s in (2, 7, 11):
                for c in (0, 1):
                    cell = [i for i in range(len(self.y)) if self.subject[i] == s and self.y[i] == c]
                    shuffled = rng.permutation(cell).tolist()
                    for position, trial in enumerate(shuffled):
                        expected[(position + offset) % k].add(trial)
                    offset = (offset + 1) % k
            for (tr, te, _), want in zip(splits, expected):
                self.assertEqual(set(te.tolist()), want)
                self.assertEqual(set(tr.tolist()), set(range(len(self.y))) - want)
            C.assert_subject_class_stratified_splits(self.y, self.subject, splits, k)

    def test_deliberately_broken_dealing_and_coverage_fail_closed(self):
        splits = C.split_subject_class_stratified(self.y, self.subject, 3, C.SEED)
        # Reassign one trial while retaining complete test coverage and complements.
        cell = np.flatnonzero((self.subject == 2) & (self.y == 0))
        original = next(k for k, (_, te, _) in enumerate(splits) if cell[0] in te)
        destination = next(k for k, (_, te, _) in enumerate(splits)
                           if k != original and np.isin(te, cell).sum() == 2)
        broken = []
        for k, (_, te, name) in enumerate(splits):
            changed = set(te.tolist())
            if k == original:
                changed.remove(int(cell[0]))
            if k == destination:
                changed.add(int(cell[0]))
            test = np.asarray(sorted(changed), dtype=int)
            broken.append((np.setdiff1d(np.arange(len(self.y)), test), test, name))
        with self.assertRaisesRegex(AssertionError, "unbalanced"):
            C.assert_subject_class_stratified_splits(self.y, self.subject, broken)
        duplicate = [splits[0], splits[0], splits[2]]
        with self.assertRaisesRegex(AssertionError, "exactly one"):
            C.assert_subject_class_stratified_splits(self.y, self.subject, duplicate)
        overlap = copy.deepcopy(splits)
        overlap[0] = (np.r_[overlap[0][0], overlap[0][1][0]], overlap[0][1], overlap[0][2])
        with self.assertRaisesRegex(AssertionError, "overlapping"):
            C.assert_subject_class_stratified_splits(self.y, self.subject, overlap)

    def test_construction_counts_compare_actual_p1_not_ideal_quotas(self):
        data = synthetic_eeg()
        splits = C.ladder_splits(data, "trial_random_sc", C.SEED, 3)
        rows = C.p0s_construction_table(data, splits, seed=C.SEED)
        for row in rows:
            k = int(row["p1_fold"].removeprefix("run"))
            mask = (data["subject"] == row["subject"]) & (data["y"] == row["class"])
            if row["run"] != "all":
                mask &= data["run_number"] == row["run"]
            p1_test = int((mask & (data["run_idx"] == k)).sum())
            self.assertEqual(row["p1_n_test"], p1_test)
            self.assertEqual(row["p1_n_train"], int(mask.sum()) - p1_test)
            self.assertEqual(row["delta_test"], row["n_test"] - p1_test)
            self.assertEqual(row["delta_train"], -row["delta_test"])
        self.assertTrue(any(r["delta_test"] != 0 for r in rows))
        bci = synthetic_bci()
        for k, comparator in ((2, "session_disjoint"), (6, "run_disjoint")):
            rows = C.p0s_construction_table(bci, C.ladder_splits(bci, "trial_random_sc", C.SEED, k))
            self.assertEqual({r["p1_protocol"] for r in rows}, {comparator})


class FixedTestArmsTest(unittest.TestCase):
    def test_independent_reconstruction_all_rotations_and_shortfalls(self):
        from sklearn.model_selection import train_test_split

        for factory in (synthetic_eeg, synthetic_bci):
            for shortfall in (False, True):
                data = factory(shortfall=shortfall)
                for rotation in range(3):
                    with self.subTest(dataset=data["dataset"], shortfall=shortfall, rotation=rotation):
                        got = C.fixed_test_arms(data, data["dataset"], C.SEED, rotation)
                        expected = independently_expected_design(data, C.SEED, rotation)
                        self.assertEqual(got["leftover_subjects"], expected["leftovers"])
                        for key in ("test", "h2", "bg"):
                            self.assertEqual(got[key].tolist(), expected[key])
                        for arm in ("A0", "A1", "A2"):
                            self.assertEqual(got["arms"][arm].tolist(), expected["arms"][arm])
                            self.assertEqual(len(np.unique(data["subject"][got["arms"][arm]])), 6)
                            self.assertFalse(set(got["test"]) & set(got["arms"][arm]))
                        cols = ("pair", "subject_T", "subject_D", "run", "class", "n_T", "n_D", "n_kept", "shortfall")
                        self.assertEqual([tuple(row[c] for c in cols) for row in got["pairing"]], expected["matching"])
                        _, val = train_test_split(np.asarray(expected["bg"]), test_size=0.15,
                                                  random_state=C.SEED, stratify=data["y"][expected["bg"]])
                        np.testing.assert_array_equal(got["validation"], np.sort(val))
                        self.assertEqual(sum(got["shortfall_by_class"].values()) > 0, shortfall)
                        slots = {(r["arm"], r["run"], r["class"]): r for r in got["counts"]
                                 if r["count_type"] == "matched_slot"}
                        for (_, run, c), row in list(slots.items()):
                            if row["arm"] == "A0":
                                self.assertEqual(row["n_trials"], slots[("A1", run, c)]["n_trials"])
                                a2 = slots[("A2", run, c)]
                                self.assertEqual(a2["n_trials"] - row["n_trials"], a2["shortfall"])

    def test_all_smoke_leftovers_reported_and_full_group_sizes(self):
        for n, size, leftovers in ((12, 4, 0), (20, 6, 2), (109, 36, 1)):
            data = synthetic_eeg(n)
            got = C.fixed_test_arms(data, "eegbci", C.SEED, 0)
            self.assertEqual(got["group_size"], size)
            self.assertEqual(len(got["leftover_subjects"]), leftovers)
            for idx in got["arms"].values():
                self.assertEqual(len(np.unique(data["subject"][idx])), 2 * size)

    def test_planted_test_leak_unmatched_arm_and_moving_preprocessing_fail(self):
        data = synthetic_eeg()
        original = C.fixed_test_arms(data, "eegbci", C.SEED, 0)
        broken = copy.deepcopy(original)
        broken["arms"]["A0"] = np.r_[broken["arms"]["A0"], broken["test"][0]]
        with self.assertRaisesRegex(AssertionError, "test trial"):
            C.assert_fixed_test_arms(data, broken)
        broken = copy.deepcopy(original)
        non_bg = next(i for i in broken["arms"]["A1"] if i not in broken["bg"])
        broken["arms"]["A1"] = broken["arms"]["A1"][broken["arms"]["A1"] != non_bg]
        with self.assertRaisesRegex(AssertionError, "matched cell"):
            C.assert_fixed_test_arms(data, broken)
        for key, message in (("reference_indices_by_arm", "moving reference"),
                             ("validation_indices_by_arm", "moving validation")):
            broken = copy.deepcopy(original)
            broken[key]["A1"] = broken[key]["A1"][:-1]
            with self.assertRaisesRegex(AssertionError, message):
                C.assert_fixed_test_arms(data, broken)

    def test_deep_glue_passes_actual_same_global_validation_and_paired_seed(self):
        import deep_common as D

        for data in (synthetic_eeg(), synthetic_bci()):
            calls = []

            def fake_train(model, x_tr, y_tr, x_te, device, seed, **kwargs):
                global_ids = x_tr[:, 0, 0].astype(int)
                validation = global_ids[kwargs["validation_indices"]]
                fit, _ = D.explicit_validation_split(len(y_tr), kwargs["validation_indices"])
                calls.append((seed, validation.copy(), global_ids[fit].copy(), x_te[:, 0, 0].astype(int)))
                return {"preds": np.zeros(len(x_te), dtype=int), "n_fit": len(fit), "n_val": len(validation),
                        "train_acc": 0.5, "val_acc": 0.5, "best_epoch": 1, "epochs_run": 2, "n_params": 1}

            with patch.object(D, "train_predict", side_effect=fake_train), contextlib.redirect_stdout(io.StringIO()):
                rows = list(D.fixed_test_fits(data, "eegnetv4", "cpu", C.SEED, epochs=2))
            self.assertEqual(len(rows), 9)
            for rotation in range(3):
                design = C.fixed_test_arms(data, data["dataset"], C.SEED, rotation)
                for call in calls[rotation * 3:(rotation + 1) * 3]:
                    self.assertEqual(call[0], C.SEED)
                    np.testing.assert_array_equal(call[1], design["validation"])
                    self.assertFalse(np.intersect1d(call[1], call[2]).size)
                    np.testing.assert_array_equal(call[3], design["test"])
                self.assertEqual(len({row[1]["validation_sha256"] for row in rows[rotation * 3:(rotation + 1) * 3]}), 1)
            for bad in (np.array([], dtype=int), np.array([1, 1]), np.array([-1]), np.array([4]), np.array([0.0])):
                with self.assertRaises(ValueError):
                    D.explicit_validation_split(4, bad)

    def test_classical_glue_estimates_bg_reference_once_and_reuses_actual_features(self):
        import e17_fixed_test as E

        data = synthetic_eeg()
        means, fitted, written = [], [], {}
        covs = np.arange(len(data["y"]), dtype=float)[:, None, None]

        def fake_mean(values):
            means.append(values[:, 0, 0].astype(int).tolist())
            return np.array([[len(means) + 100.0]])

        def fake_tangent(values, reference):
            return np.c_[values[:, 0, 0], np.full(len(values), reference[0, 0])]

        class CapturingModel:
            def fit(self, x, y):
                fitted.append(np.asarray(x).copy())

            def predict(self, x):
                return np.zeros(len(x), dtype=int)

        def fake_write(path, rows):
            written[path.name] = rows
            return path

        argv = [str(REVISION / "e17_fixed_test.py"), "--tasks", C.TASKS[0], "--models", "riemann_ts_lr",
                "--n-seeds", "1", "--n-jobs", "1", "--max-subjects", "10", "--tag", "synthetic"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(C, "discover_subjects", return_value=list(range(1, 11))))
            stack.enter_context(patch.object(C, "load_eegbci_task", return_value=data))
            stack.enter_context(patch.object(C, "covariances", return_value=covs))
            stack.enter_context(patch.object(C, "riemann_mean", side_effect=fake_mean))
            stack.enter_context(patch.object(C, "tangent", side_effect=fake_tangent))
            stack.enter_context(patch.object(E, "build_model", side_effect=lambda *_: CapturingModel()))
            stack.enter_context(patch.object(C, "write_csv", side_effect=fake_write))
            stack.enter_context(patch.object(C, "validate_csv", return_value=1))
            stack.enter_context(patch.object(C, "write_manifest"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(E.main(), 0)
        # First three means precede fitting; each is exactly a rotation's BG.
        self.assertEqual(len(means), 12)
        for rotation in range(3):
            design = C.fixed_test_arms(data, "eegbci", C.SEED, rotation)
            self.assertEqual(means[rotation], design["bg"].tolist())
            fixed_fits = fitted[rotation * 6:(rotation + 1) * 6:2]
            for arm, fit in zip(("A0", "A1", "A2"), fixed_fits):
                np.testing.assert_array_equal(fit[:, 0].astype(int), design["arms"][arm])
                np.testing.assert_array_equal(fit[:, 1], np.full(len(fit), rotation + 101.0))
        arm_rows = written["e17_eegbci_synthetic_per_arm.csv"]
        for rotation in range(3):
            self.assertEqual(len({r["reference_sha256"] for r in arm_rows
                                  if r["rotation"] == rotation and r["reference"] == "bg_fixed"}), 1)


if __name__ == "__main__":
    unittest.main()
