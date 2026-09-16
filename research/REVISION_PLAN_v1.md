# REVISION_PLAN_v1 — major revision of the article of the subject-leakage audit (bf/)

Frozen: 2026-09-16 (before any new result exists; sha256 recorded in DECISION_LOG.md).
Scope: the analyses, definitions and decision rules that answer Referee 1 (R1.1–R1.10 + minor),
Referee 2 (R2.1–R2.7) and the Editor-in-Chief (captions, axes/units, per-subject points).
Status of the original analyses: the July 2026 results (5-fold pooled/cross, within-subject) are
kept as the "original configuration" in the supplement; nothing already reported is deleted.
All analyses below are reviewer-requested, post-submission analyses. They were not preregistered
before the original submission; the decision rules in §C are fixed here before the new runs.

## A. Referee comment → analysis map

| Ref | Comment (short) | Analysis |
|---|---|---|
| R1.1, R2.3, R2.4 | protocols differ in more than subject overlap (run/session adjacency, train composition) | E1 hierarchical split ladder, matched training fraction |
| R1.2, R2.6, minor | regularization sweep shows only the difference; underfitting not excluded | E2 sweep v2: both accuracy curves, training accuracy, ‖w‖, per-subject CIs, repeated partitions |
| R1.2, R1.3 | show that tangent features contain and the classifier exploits subject information; subject identity alone cannot predict the label | E3a subject-ID decoding; E3b variance decomposition (subject / class / subject×class); E3c subject-mean removal and Riemannian recentering; E3d alignment of pooled vs subject-disjoint weight vectors with subject-specific class-separation vectors |
| R1.2, R1.5 | dimensionality-controlled experiments; several classifiers on the same features | E4 PCA-dimension sweep; E5 classifiers on identical tangent features (incl. MDM as a zero-capacity Riemannian control) |
| R1.4 | test whether the target subject's presence in the reference matters with classifier data fixed | E6 target-subject-in-reference test under the pooled protocol; E14 TOST equivalence for both reference tests |
| R1.5, R2.1, minor | capacity not quantified; "capacity ladder" | E7 width sweeps with parameter counts and train–test gaps; the ladder wording is dropped |
| R1.6, R2.6 | severity rule inferred from two confounded datasets | E8 within-dataset factorial: training-subject count × tangent-feature dimension, repeated draws; CSP+LDA subject-count sweep |
| R1.7, R2.3 | fold construction, seeds, nested subject-level validation; LOSO and inferential statistics for BCI-IV-2a | E9 deep-net protocol v2 (nested subject-level validation, 3 seeds); E10 BCI-IV-2a full grid with LOSO, all four decoders, per-subject differences, exact tests |
| R1.8, R2.1 | calibration gap not controlled; deep within-subject results missing | E11 within-subject deep results on both datasets; E12 labelled calibration-size curve for unseen subjects |
| R2.3 | preprocessing not matched across decoder families | E13 matched-preprocessing control (both directions) |
| R1.9 | literature | Del Pup 2025 and the directly related partition/identity-confound literature (G1-verified) in Introduction and Discussion |
| R1.10, R2.5 | reproducibility, proof errors, notation, CIs in tables, repository link | Reproducibility section + supplement; renumbered citations; N_c for channels, C reserved for the penalty; "bootstrap (2000 resamples)"; repository URL + tag |
| R2.2 | dataset scope (executed movement), incomplete grid | title/abstract/methods say "motor imagery and motor execution"; the full 2 datasets × 4 decoders × protocols grid is run |
| R2.4 | Figure 1 wrong/overclaiming | Figure 1 redrawn as a neutral workflow (regularization direction corrected: larger C = weaker penalty) |
| R2.7, R2.1, R1 general | abstract/title/discussion overclaim | rewritten: observations vs interpretation separated; mechanistic wording conditional on §C outcomes |
| EIC | captions, axes with units, per-subject points | every figure: per-subject points, labelled axes with units, caption defines abbreviations and the statistic shown |

## B. Frozen definitions

B1. Data. EEGBCI: 109 subjects, 64 channels, 160 Hz, 7–30 Hz FIR, epochs 0–4 s after the T1/T2 cue,
four two-class paradigms (imagined and executed left/right fist; imagined and executed fists/feet),
3 runs per paradigm per subject (PhysioNet runs 4/8/12, 3/7/11, 6/10/14, 5/9/13); no trial rejection.
BCI-IV-2a: 9 subjects, 22 channels, 250 Hz, 8–30 Hz, MOABB LeftRightImagery epochs (paradigm
default window), 2 sessions × 6 runs, 288 left/right trials per subject; no trial rejection.

B2. Unit and statistics. The unit of analysis is the subject: per-subject accuracy = fraction of
that subject's test trials predicted correctly (test trials of a subject are aggregated across folds
where a subject appears in several test folds). Aggregate = mean over subjects. Uncertainty =
subject-level percentile bootstrap, 2000 resamples, seed 20260916. Paired contrasts = two-sided
Wilcoxon signed-rank test on per-subject differences (exact distribution when n ≤ 25) and a
sign-flip permutation test (10 000 permutations) as a check. Multiplicity: Holm correction within
each pre-specified family (a family = the contrasts reported in one table). Equivalence: TOST with
margin ±0.01 accuracy (90 % CI inside the margin). Effects are reported with the adjusted p and the CI.

B3. Protocols (E1 ladder). EEGBCI, all three folds matched at training fraction 2/3:
- P0 trial-random: StratifiedKFold(3, shuffle) over pooled trials (subject, run ignored).
- P1 run-disjoint, subject-overlapping: fold k holds out the k-th run (k = 1..3 in recording order)
  of every subject; training = the other two runs of all subjects.
- P2 subject-disjoint: 3-fold GroupKFold by subject (subject order permuted by the repetition seed).
Repetitions: P0 and P2 use 5 partition seeds (20260916 + r); P1 is deterministic.
BCI-IV-2a:
- P0a trial-random StratifiedKFold(6, shuffle) [training fraction 5/6]; P0b trial-random
  StratifiedKFold(2, shuffle) [1/2, the size-matched partner of P1s].
- P1 run-disjoint, subject- and session-overlapping: 6 folds, fold k holds out run k of both
  sessions of every subject [5/6].
- P1s session-disjoint, subject-overlapping: 2 folds, train on one session of all subjects, test on
  the other [1/2].
- P2 subject-disjoint: leave-one-subject-out (LOSO, 9 folds) [8/9]; the original 5-fold GroupKFold
  is kept in the supplement for continuity.
- Within-subject: EEGBCI leave-one-run-out inside each subject (as submitted); BCI-IV-2a
  session-to-session inside each subject (train one session, test the other, both directions averaged).
Derived quantities: Δ(Pa − Pb) = mean over subjects of the per-subject accuracy difference;
the reported "subject-overlap component" is Δ(P1 − P2) [EEGBCI] and Δ(P1s − P2), Δ(P1 − P2) [BCI-IV-2a];
the "run/temporal component" is Δ(P0 − P1). The original quantity Δ_leak = Δ(P0 − P2) is kept.

B4. Decoders and settings (unchanged unless stated). CSP+LDA: MNE CSP, 6 components, Ledoit–Wolf
regularization, log-variance, LDA (SVD solver). TS+LR: OAS covariance, Riemannian tangent space at
the training-fold Fréchet mean (inductive), ℓ2 logistic regression (lbfgs, max_iter 3000, C = 1
unless swept), no feature standardization. EEGNet-v4 and ShallowFBCSPNet: braindecode defaults,
Adam lr 1e-3, batch 64, cross-entropy, ≤40 epochs, early stopping on validation accuracy with
patience 6; per-trial per-channel z-scoring. Validation split for early stopping: P2 → 15 % of the
training subjects (GroupShuffleSplit, subject-disjoint from both training and test subjects);
P0/P1 → 15 % of training trials (stratified), which is consistent with those protocols'
definitions. Seeds: 3 (20260916, 20260917, 20260918) for every deep-net cell of E9/E10/E13;
1 seed for E7 width sweeps and E11/E12 unless time allows 3. Parameter counts are reported for
every network (sum of trainable parameters).

B5. E2 sweep v2. C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}; protocols P0, P1, P2 (EEGBCI ladder);
per fold: inductive tangent features, then LR at each C; record test accuracy (per subject), training
accuracy (on the training fold), ‖w‖2, and the per-subject bootstrap CI; 5 partition seeds for P0/P2.
Reported: both accuracy curves, the training curves, the ‖w‖2 curve, and Δ(P0 − P2)(C), Δ(P1 − P2)(C).

B6. E3 subject-information analyses (EEGBCI; C = 1; tangent features).
- E3a subject-ID decoding: multinomial LR (C = 1) predicting the subject (109 classes) from tangent
  features under (i) trial-random 3-fold and (ii) run-disjoint 3-fold (identity must generalize across
  runs); chance = 1/109; also after per-subject mean removal (second-order identity information).
- E3b variance decomposition: for each tangent dimension, two-way sums of squares for subject, class,
  subject×class (cell-mean model) and residual; the reported fractions are sums over dimensions
  divided by the total sum of squares.
- E3c mean removal / recentering: (i) tangent-feature centering by the subject's own mean over all of
  its trials (label-free, transductive-unsupervised); (ii) Riemannian recentering
  C_i → G_s^{-1/2} C_i G_s^{-1/2} with G_s the subject's Fréchet mean over all its trials; then
  P0/P1/P2 with TS+LR. Interpretation rule in §C3.
- E3d weight alignment: in a common tangent space (reference = Fréchet mean of all trials, used
  only for this diagnostic and stated as such), d_s = normalized class-mean difference of subject s;
  reported: mean pairwise |cos(d_s, d_s')| over subject pairs; for LR weight vectors trained under
  P0 and under P2 (same fold structure, per fold): mean over subjects of |cos(w, d_s)| and the fraction
  of ‖w‖² inside span{d_s}; paired comparison P0 vs P2 over folds × seeds.

B7. E4 PCA sweep. Tangent features (inductive per fold) → PCA fitted on the training fold →
d ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2080 (no PCA)} → LR (C = 1); protocols P0/P1/P2;
reported acc per protocol and Δ(P1 − P2), Δ(P0 − P2) versus d; Spearman ρ of Δ(P1 − P2) over d.

B8. E5 classifiers on identical tangent features: LR (C = 1), shrinkage LDA (Ledoit–Wolf), linear
SVM (C = 1), ridge classifier (α = 1), k-NN (k = 5), and MDM (minimum distance to Riemannian mean,
operating on covariances, no tangent space) under P0/P1/P2.

B9. E6 target-subject-in-reference (EEGBCI, P0 3-fold, one seed). For each fold and each test
subject s: reference R_full from all training-fold trials vs R_−s from the training-fold trials of
all subjects except s (Fréchet mean warm-started from R_full); the classifier is trained on the same
training trials (features recomputed under each reference); Δ_ref(s) = acc_s(R_full) − acc_s(R_−s).
Reported with CI, Wilcoxon and TOST (±0.01).

B10. E7 width sweeps (EEGBCI, P0 and P2 of the ladder, 1 seed). EEGNet F1 ∈ {2, 4, 8, 16, 32}
(D = 2, F2 = 2·F1); ShallowFBCSPNet n_filters_time = n_filters_spat ∈ {5, 10, 20, 40, 80}.
Reported: parameter count, training accuracy, test accuracy, train–test gap, Δ(P0 − P2) per width;
Spearman ρ of Δ(P0 − P2) over width.

B11. E8 within-dataset factorial (EEGBCI). Training-subject count N ∈ {9, 18, 36, 72, 109} ×
tangent dimension d ∈ {8, 32, 128, 512, 2080}; 10 random subject draws for N < 109 (1 for 109),
draw seeds 20260916 + i; each draw evaluated under P0 (trial-random 3-fold) and P2
(subject-disjoint 3-fold) with TS+LR (C = 1, PCA on the training fold); CSP+LDA across N only.
Analysis: cell means with draw-level CIs and OLS of Δ(P0 − P2) on log2 N, log2 d and their
interaction with draw-bootstrap CIs (exploratory descriptive model, stated as such).

B12. E10 BCI-IV-2a. All four decoders under P0a, P0b, P1, P1s, P2 (LOSO), within-subject;
deep nets with 3 seeds; per-subject differences with exact Wilcoxon, sign-flip permutation and
bootstrap CIs; the transductive-reference ablation repeated under LOSO with TOST.

B13. E11 within-subject deep results. EEGBCI: leave-one-run-out per subject; because a validation
slice of a ~30-trial training set is unreliable, within-subject networks train for a fixed 60 epochs
without early stopping (pre-specified). BCI-IV-2a: session-to-session per subject, same early
stopping as B4 with a 15 % trial-level validation slice (144 training trials), 3 seeds.

B14. E12 calibration-size curve (EEGBCI). Base = P2 training subjects (3-fold, seed 20260916);
target subject's evaluation set = its third run; k ∈ {0, 4, 8, 16, 30} labelled calibration trials
drawn (stratified, 3 draws) from its first two runs and appended to the training set (weight 1) for
CSP+LDA and TS+LR (refit); for EEGNet the P2 model is fine-tuned on the k trials (20 epochs, lr 1e-3,
1 seed). Reported: accuracy vs k with per-subject points; the within-subject (leave-one-run-out) and
P2 values are shown as reference lines.

B15. E13 matched preprocessing. (i) Networks on band-passed trials with one global scale factor
(1/median absolute amplitude of the training fold) instead of per-trial z-scoring; (ii) CSP+LDA and
TS+LR on per-trial per-channel z-scored trials; both under P0/P1/P2 (EEGBCI), 1 seed.

B16. E14 equivalence. Reference tests (transductive vs inductive under P2; R_full vs R_−s under P0):
TOST with margin ±0.01; report the 90 % CI and both one-sided p-values.

## C. Decision rules (fixed before any new run)

C1 Subject-overlap inflation (TS+LR). Supported if Δ(P1 − P2) > 0 with Holm-adjusted p < 0.05 and
a bootstrap CI excluding 0 on EEGBCI (paradigm-averaged) and the same sign on BCI-IV-2a (LOSO).
If the CI of Δ(P1 − P2) includes 0 while Δ(P0 − P1) does not, the manuscript attributes the
pooled inflation to run/temporal adjacency, not to subject overlap, and the title changes accordingly.

C2 Classifier mediation. The underfitting explanation of the low-C null is rejected if, at the
largest C whose Δ(P1 − P2) CI includes 0, the P2 accuracy is within 0.02 of its maximum over C.
Supported additionally if at least two other classifiers on identical features (E5) show Δ(P1 − P2)
with a CI excluding 0 in the same direction, and MDM shows a smaller Δ(P1 − P2) than LR.

C3 Mechanism. "Subject×class interaction" is the stated mechanism only if Δ(P1 − P2) persists
(CI excluding 0) after both subject-mean removal and Riemannian recentering (E3c) AND the P0-trained
weight vectors align more with subject-specific separation vectors than the P2-trained ones (E3d,
paired, p < 0.05). If Δ(P1 − P2) vanishes after mean removal, the mechanism is stated as a
subject-mean offset exploited by the classifier. Otherwise the mechanism is reported as unresolved.

C4 Reference. "The reference is not a leakage channel in the evaluated settings" requires TOST
equivalence (±0.01) for the transductive test (P2, both datasets) and for the target-subject-in-
reference test (E6). Otherwise the statement is narrowed to whichever test passed.

C5 Dimensionality. Supported if Δ(P1 − P2) increases with d (E4: Spearman ρ > 0, p < 0.05) and
the E8 d main effect is positive with a draw-bootstrap CI excluding 0.

C6 Subject count. Supported if the E8 N main effect is negative with a CI excluding 0; the
"even CSP+LDA inflates at small N" statement requires the CSP+LDA sweep to show Δ(P0 − P2) > 0
at N = 9 within EEGBCI (CI excluding 0). Otherwise the BCI-IV-2a CSP+LDA observation is reported as
a single-dataset observation without a rule.

C7 Capacity. The manuscript states only what E7 shows: if Spearman ρ of Δ(P0 − P2) over width is
not significantly positive for either network, "no monotone increase of inflation with width within
the evaluated ranges"; any positive trend is reported as such.

C8 Calibration. The sentence "deep networks close the calibration gap" is removed. Δ_cal is
reported for all four decoders from E11 with its confound stated (it is a descriptive contrast, not
a controlled estimate); E12 provides the controlled calibration-size curve.

C9 Words retired regardless of outcome: "immune", "robust to the leak", "removes leakage",
"capacity ladder", "not a ... model-capacity effect" (as a title-level claim), "definitively".

## D. Compute plan and stop rules
CPU: ≤40 of 48 cores (joblib) for classical experiments; GPU: the 4 L40S for E7/E9/E10/E11/E12/E13,
one process per GPU. Estimates: E1+E2+E4+E5 ≈ 1 h wall; E3 ≈ 30 min; E6 ≈ 20 min; E8 ≈ 30 min;
E9 ≈ 2 h; E7 ≈ 1 h; E11 ≈ 1 h; E12 ≈ 1.5 h; E10/E13 ≈ 30 min. If an experiment exceeds twice its
estimate: seeds 3 → 2, draws 10 → 5, E12 restricted to the two imagery paradigms — disclosed.
Every run: one log, a JSON manifest (command, seed, versions, hashes, start/end), exit code 0 AND
schema-validated outputs before use. Smoke runs (≤10 subjects) are never reported.

## E. Reporting rules
Every number in the revised manuscript traces to a CSV under results/ named in the provenance
appendix. Tables carry the CI next to the mean. Figures show per-subject points, labelled axes with
units, and captions that define abbreviations and name the statistic and n. The response quotes the
changed text and gives line numbers. Nothing from the original submission is deleted without a
statement in the response.
