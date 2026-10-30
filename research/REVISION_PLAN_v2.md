# REVISION_PLAN_v2 — second major revision of the subject-leakage audit (bf/; manuscript R1 → R2)

Status: FROZEN 2026-10-30 after one design attack (T001-falsify-plan, findings F1–F11, all amended below); sha256 in
research/REVISION_PLAN_v2.sha256 and DECISION_LOG.md. Frozen before any new run exists.
Scope: the analyses, definitions and decision rules that answer Referee 1's second-round report (points R1-2.1 … R1-2.5 and
the additional clarifications; Referee 2 is satisfied). REVISION_PLAN_v1.md (frozen 2026-09-16, sha256 8d9b1f5b…) stays in
force for everything it covers, except that this document supersedes: the factorial feasibility rule (v1 B11), the two
subject-information diagnostics it corrects (v1 B6 E3a, E3b), and the title/mechanism attribution branch of v1 rule C1 (its
"attributes the pooled inflation to run/temporal adjacency ... and the title changes accordingly" clause), which is replaced by
rules C12 and C15 below; every inherited interpretation obeys the protocol-contrast limitation of §C. Nothing reported in
revision 1 is deleted; corrected values replace wrong ones with a disclosure. All analyses below are reviewer-requested,
post-submission analyses; the decision rules in §C are fixed here before any new run. The revision-1 results (tag rev1) are
frozen; every new run carries the tag rev2.

## A. Referee comment → analysis map

| Ref | Comment (short) | Analysis / change |
|---|---|---|
| R1-2.1 | factorial: a PCA dimension skipped in one fold left −1 predictions that were counted as misses (real_fists_feet, N = 18, d = 512, draw 6: P2 = 0.419) | E8-v2: feasibility decided over all folds of both protocols before a paired condition is scored; the fold-specific PCA construction of the released code is preserved so that retained cells reproduce; coverage assertion; regression tests incl. the boundary cases; full rerun; Figure 6(b,c), Table S7, OLS estimates, CIs and text regenerated; the error disclosed |
| R1-2.2 | ladder differences are protocol contrasts, not isolated mechanisms (P0−P1 mixes run overlap with the stratification effect; P1 and P2 differ in training composition; IV-2a contrasts differ in size) | Wording: every ladder difference is defined and named a protocol contrast (title, abstract, §2.4, §3, §4, §5). Controls: E16 = the subject-by-class-stratified trial-random split P0s the referee named (reported as a protocol contrast, not an isolated bound); E17 = a fixed-test framework with count-matched training arms on an unchanged background, a common tangent reference and a common validation slice, giving a target-inclusion contrast and a same-run-adjacency contrast on the same test set; rule C12 decides the title |
| R1-2.3 | identity probe after all-trial centring is below chance by construction | E3a-v2: centring from an independent reference run, probe trained on a second run and tested on the third, plus a split-reference control that separates shared centring-estimation error from cross-run identity; the construction explained in §3.4; Figure 5(b) interpretation restricted (C13) |
| R1-2.4 | marginal subject and class sums of squares are not orthogonal in the unbalanced design | E3b-v2: sequential (Type I) sums of squares from nested trial-weighted least-squares models in two stated orders; definitions and weighting stated; hand-checkable fixtures; Figure 5(a) and §3.4 updated |
| R1-2.5 | Jamalabadi et al. concern subclasses nested within classes | Discussion wording: nested versus crossed; shared subject offsets versus subject-by-class structure; the connection heuristic (C16) |
| R1-2.add-1 | IV-2a CIs descriptive and unadjusted | §2.6, §3.2, Table 4 caption, abstract, every new IV-2a table |
| R1-2.add-2 | reference-mean equivalence = equivalence of the mean change | §3.5, Discussion, abstract; v1 E6 is never used as a per-subject or E17 bound |
| R1-2.add-3 | frozen-plan reference and checksum consistent; link the immutable version; provide the decision log | data statement with commit permalinks and the sha256 of the distributed files; transformation record for the public copy of plan v1 (original 8d9b1f5b…, distributed 334cf5c0…, exact substitutions, verified 2026-10-30); plan v2 written without the journal's name so its frozen bytes are released unchanged; a public decision log of the scientific decisions released with the code |
| R1-2.add-4 | qualify the subject-disjoint recommendation by the deployment target | Discussion recommendation (i), Conclusion |

## B. Frozen definitions (additions to plan v1; v1 definitions B1–B16 apply unless restated)

B17. E8-v2 factorial feasibility (replaces the per-fold skip of v1 B11). Grid, draws, seeds, decoders and protocols as in B11
(N ∈ {9, 18, 36, 72, 109}; d ∈ {8, 32, 128, 512, 2080}; 10 draws for N < 109, 1 for N = 109; draw seeds 20260916 + i; subject
sampling with default_rng(20260916 + draw + 1000·N); P0 = trial-random 3-fold and P2 = subject-disjoint 3-fold with seed
20260916 + draw; TS+LR, C = 1; CSP+LDA across N). The PCA construction of the released code is preserved exactly: one randomized
PCA per fold with n_components = min(512, n_train − 1), random_state = the fold seed, components sliced to each d < 2080;
d = 2080 uses the raw features. What changes is the scoring mask: for each (task, N, draw) the folds of BOTH protocols are
constructed first; a dimension d < 2080 is FEASIBLE for the draw iff d ≤ (min over all six folds of n_train) − 1; an infeasible
d is excluded from scoring for BOTH protocols of that draw and recorded in e8_<tag>_feasibility.csv (task, N, draw, d,
min_train_fold_P0, min_train_fold_P2, feasible). Before any accuracy is computed the code asserts that every scored prediction
vector covers every trial (no −1 left) and records the coverage count. Regression tests on synthetic data cover n_train = d
(infeasible), n_train = d + 1 (feasible), a partially feasible pair (feasible in P0, infeasible in one P2 fold → excluded from
both) and a completely infeasible pair, and show that the old per-fold skip scores −1 entries as misses while the new mask
excludes the condition. Analysis: per paradigm as in B11 on that paradigm's feasible rows (cell means with draw-level CIs; OLS of
Δ(P0 − P2) on log2 N, log2 d and their product with centred regressors and a bootstrap over whole draws within each N), with the
observation count reported. Four-paradigm aggregation (Figure 6(b), the pooled cells of Table S7): the "four-paradigm mean
within a draw" requires all four paired paradigm values; a draw with a missing paradigm is excluded from that aggregate cell
(the N = 18, d = 512 aggregate therefore has nine complete draws) and whole complete draws are resampled together in the
aggregate bootstrap; nothing averages whichever paradigms survive. Reproduction check: for every condition feasible under both
versions the rev2 value must equal the rev1 value within 1e-6 (the CSV carries six significant digits); a larger difference is
investigated (environment, solver) and reported, and the tolerance is never relaxed to pass.

B18. E3a-v2 identity probe with an independent centring reference (EEGBCI, four paradigms, tangent features, C = 1). Each paradigm
has three runs per subject in recording order (r1, r2, r3). For each of the six ordered assignments (reference run, probe-training
run, probe-test run): the tangent-space reference is the Fréchet mean of the probe-training run's covariances (all subjects;
inductive with respect to the test run; the reference-run features therefore depend on the probe-training covariances through
the fitted mean, which is stated); all trials are mapped with it. Centring schemes on identical features: (i) none; (ii)
independent-reference centring: each subject's centring vector is the mean of its reference-run features, subtracted from its
training- and test-run features; (iii) training-run centring (inductive): the subject's mean over the probe-training run,
subtracted from training and test; (iv) all-trial centring: the subject's mean over all three runs (for exposition; its
three-run weighted-zero relation differs from the two-subset relation of the revision-1 probe, which is said); (v)
split-reference control: the reference run's trials of each subject are split into two halves (alternating trials in recording
order); the training-run features are centred with the mean of half A and the test-run features with the mean of half B, so that
the two centring-estimation errors are independent. A multinomial logistic regression (lbfgs, max_iter 3000, C = 1) predicts the
subject from the training run and is scored on the test run. Unit and aggregation: per-subject accuracy (fraction of the subject's
test-run trials identified) averaged over the six assignments within subject, then the macro mean over subjects with the
subject-bootstrap 95 % CI (2000 resamples, seed 20260916); the pooled-trial accuracy and the per-assignment values are released
as well; chance = 1/109 = 0.0092. The difference (ii) − (v) estimates the identity signal that a shared estimated centring
vector introduces. The revision-1 E3a values (trial-random and run-disjoint 3-fold; all-trial centring) remain reported with
the explanation of the forced anticorrelation of each subject's centred fold means.

B19. E3b-v2 variance decomposition for the unbalanced subject × class design (EEGBCI, common tangent space as in v1 B6; descriptive).
Ordinary trial-weighted least squares (one weight per trial, intercept included in every model) for each tangent dimension j:
RSS0 (grand mean), RSS_S (subject indicators), RSS_C (class indicator), RSS_SC (subject and class indicators, additive), RSS_cell
(one mean per subject × class cell). Order A (subject → class → interaction): SS_subject = RSS0 − RSS_S; SS_class|subject =
RSS_S − RSS_SC; SS_subject×class|subject,class = RSS_SC − RSS_cell; SS_residual = RSS_cell. Order B (class → subject →
interaction): SS_class = RSS0 − RSS_C; SS_subject|class = RSS_C − RSS_SC; interaction and residual as in A. Fractions = component
summed over dimensions divided by RSS0 summed over dimensions; every component is non-negative up to a relative tolerance of
1e-9 and the components sum to RSS0 in each order; both orders are reported, the order named; SS_subject of order A (the marginal
between-subject sum of squares) is the descriptive quantity carried into the text. These are trial-weighted sums of squares of
the features, not subject-level accuracy quantities. The revision-1 subtraction-defined values are kept in the supplement with
the note that they are not an orthogonal decomposition. Fixtures (noiseless, hand-checkable, non-orthogonal): two subjects with
offsets 0 and 1 and class-cell counts (3, 1) and (1, 3) must give RSS0 = 2, RSS_S = 0, RSS_C = 1.5, RSS_SC = RSS_cell = 0, order-A
components (2, 0, 0, 0), order-B components (0.5, 1.5, 0, 0), and the revision-1 subtraction must give −0.5 for the interaction;
a second fixture with a true interaction must give a positive interaction term. Tolerance 1e-9 relative.

B20. E16 subject-by-class-stratified trial-random split (P0s) — a protocol contrast. EEGBCI: 3 folds; within every (subject,
class) cell the trials are shuffled with the partition seed and dealt to the folds in rotation (trial i of cell c → fold
(i + o_c) mod 3, the offset o_c cycling 0, 1, 2 over cells in subject-then-class order), so that every test fold of a cell with n
trials holds floor(n/3) or ceil(n/3) of them; training fraction 2/3; subject and run still straddle the boundary. Construction
assertions (fail-closed, replacing any correlation threshold): per-cell fold counts in {floor(n/3), ceil(n/3)}, every trial in
exactly one test fold, train/test disjointness and complementarity in every fold, and a released table of per-subject, per-run
and per-class train/test counts with their differences from P1's actual run counts. The E15 train/test class-fraction correlation
and the subject-prior predictor are recomputed for P0s and reported descriptively (they are not acceptance criteria; the
correlation implied by the construction is positive in some paradigms). Five partition seeds (20260916 + r) for the classical
decoders; three seeds (20260916–20260918) for the networks with the trial-level nested validation of v1 B4. BCI-IV-2a: P0a-s
(6 folds) and P0b-s (2 folds) built the same way. Derived contrasts: Δ(P0s − P1) and Δ(P0 − P0s) and Δ(P0s − P2), all of them
protocol contrasts: P0s trains on parts of all three runs with a training-derived tangent reference and its own validation
slice, P1 on two whole runs, so Δ(P0s − P1) is "trial-random with matched per-subject class proportions minus run-disjoint",
not an isolated adjacency contribution or bound. The isolated contrasts come from B21.

B21. E17 fixed-test framework: count-matched training arms on an unchanged background. EEGBCI, per partition seed (20260916 + r,
r = 0..4 for the classical decoders; 20260916–20260918 for the networks): subjects are permuted; the first 108 form three groups
G1, G2, G3 of 36 (the 109th subject of the permutation is left out for that seed and reported). For each rotation k ∈ {1, 2, 3}:
target block T = Gk, donor block D = G(k+1), background B = G(k+2) (indices mod 3). Within each (subject, run, class) cell the
trials are ordered by recording time. Test set = half h1 of run r3 of every T subject (alternating trials in each T (subject,
class) cell of r3: positions 0, 2, 4, … → h1; 1, 3, 5, … → h2). Count matching: T and D subjects are paired by the rank of their
total trial count (ties by subject id); within every pair and every (run ∈ {r1, r2}, class) cell both subjects retain the first
n = min(n_T, n_D) trials in recording order, so that the paired subjects contribute identical per-run, per-class counts. Training
arms, all containing the background block BG = runs r1 and r2 of every B subject (all trials):
- A0 (without target) = BG ∪ runs r1, r2 of the D subjects (count-matched trials);
- A1 (with target, other runs) = BG ∪ runs r1, r2 of the T subjects (count-matched trials);
- A2 (with target, same-run) = BG ∪ T subjects' trials with, per (subject, class) cell, all retained r1 trials, the first
  n2 − m trials of the retained r2 trials and the m = |h2| trials of r3 half h2 (so that the number of T trials per cell equals
  A1's; if m > n2 the cell uses all m h2 trials and no r2 trials and the shortfall is recorded).
Assertions released per rotation: identical subject counts (72), run counts and per-class trial counts between A0 and A1 and
between A1 and A2 up to the recorded shortfalls; no test trial in any arm; the leftover subject in no block. Common
preprocessing per rotation: for TS+LR one tangent reference, the Fréchet mean of the BG covariances, is estimated once and used
for every arm's training features and for the test features (fixed-reference primary analysis); in a secondary analysis the
reference is the Fréchet mean of each arm's own training covariances (full-pipeline variant, reported in the supplement).
CSP filters and all classifiers are fitted on each arm's training set (the decoder sees the arm; that is the estimand). Networks:
one validation slice per rotation, 15 % of the BG trials stratified by class, selected once and removed from the optimization
set of every arm (so the fit sets are the arms minus the same BG trials), with paired training seeds; selected epochs are logged;
preprocessing as in v1 B4 (per-trial z-scoring). Outcomes per subject s ∈ T (accuracy on its h1 test trials), averaged over the
seeds in which s was tested: Δ_incl(s) = acc(A1) − acc(A0) (target inclusion at matched composition: a target-versus-donor
substitution, conditional on the background and the donor sampling), Δ_adj(s) = acc(A2) − acc(A1) (same-run adjacency at matched
composition: replacing other-run target trials by same-run target trials of equal count). Aggregate = mean over subjects with the
subject-bootstrap CI, paired Wilcoxon, sign counts (EEGBCI: paradigm mean within subject, n = 109 subjects with at least one
tested seed). BCI-IV-2a analogue: nine subjects form T, D, B of three (no leftover); test = half h1 of session 2 of the T subjects
(alternating trials within (subject, class)); BG = session 1 of the B subjects; A0 = BG ∪ session 1 of D (count-matched by
pairing); A1 = BG ∪ session 1 of T; A2 = BG ∪ T's session-1 trials with the m = |h2| trials of session-2 half h2 replacing the
last m session-1 trials per (subject, class) cell — a same-session adjacency contrast; five seeds (classical), three (networks);
exact conditional tests; descriptive, unadjusted CIs.

B22. Statistics (as v1 B2): unit = subject; subject-level percentile bootstrap (2000, seed 20260916); two-sided Wilcoxon
signed-rank (exact conditional for n ≤ 25) on differences rounded to 12 decimals; sign-flip permutation check; Holm within
each displayed table. The EEGBCI matched-controls table (main text) = 12 contrasts, one family: Δ(P0s − P1), Δ_incl and Δ_adj
for the four decoders; Δ(P0 − P0s), Δ(P0s − P2), the TS+LR full-pipeline variant and the per-paradigm values are reported
descriptively in the supplement; the IV-2a tables (old and new) report exact p and CIs that are descriptive and unadjusted and
are never read as confirmation. TOST is not used for the new contrasts.

## C. Decision rules (fixed before any new run)

C11 Factorial correction. The dimension and subject-count statements (v1 rules C5, C6) are re-read from the rev2 factorial with
the same rules; whatever they show replaces the revision-1 values everywhere (abstract, §3.6, Discussion, Figure 6, Table S7,
ledger). The implementation error is disclosed in §3.6 and in the provenance material: one condition (real_fists_feet, N = 18,
d = 512, draw 6) was scored with an unevaluated fold; the rev2 feasibility table lists every excluded condition.

C12 Attribution in the title and abstract. Every ladder difference is a protocol contrast; "isolated contribution", "contributed
at most" and "bounded by" are used only for Δ_incl and Δ_adj of B21, and only with their design stated (fixed test = half of the
last run; background and donor sampling; common reference and validation slice). The manuscript may state that subject overlap
inflates more than same-run adjacency ONLY IF, within the B21 framework on EEGBCI (paradigm mean, n = 109), for TS+LR: (a)
Δ_incl > 0 with a 95 % CI excluding 0 and Holm p < 0.05 in the matched-controls table; (b) the lower 95 % confidence limit of
Δ_incl lies above the upper 95 % confidence limit of Δ_adj; and (c) the point estimate of Δ_incl exceeds that of Δ_adj for at
least three of the four decoders. Then the title is Title A and the abstract states both matched contrasts with their CIs and
the design. Otherwise the title is Title B, the abstract reports the protocol contrasts and the matched contrasts without
ranking them, and the Discussion says which alternative the data could not separate. On BCI-IV-2a the analogue is reported
descriptively and never ranks. Title candidates fixed now (final wording is the corresponding author's decision):
- Title A: "Subject overlap inflates subject-mixed cross-validation of EEG sensorimotor decoders more than same-run adjacency
  at matched training composition: a hierarchical split audit with matched controls on two public datasets"
- Title B: "A hierarchical split audit of subject-mixed cross-validation in EEG sensorimotor decoding: protocol contrasts and
  matched controls for four decoders on two public datasets"
Rule C12 supersedes the attribution branch of v1 rule C1 (see Scope).

C13 Identity probe. "Subject information eliminated/removed" is never claimed, and above-chance identity accuracy after centring
is never described as residual biological identity, a subject × class mechanism, motor-label exploitation or a mediation
fraction. The text states, for scheme (ii) and for the split-reference control (v), the macro identity accuracy with its CI and
the range over assignments, and says: the specified centred representation is (or is not detectably) decodable by this linear
probe under the reported assignments; a shared estimated centring vector can itself introduce identity information, and the
difference (ii) − (v) estimates that contribution. Chance-level accuracy is not evidence of information removal for other probes.
The revision-1 near-zero value is explained as a consequence of the all-trial construction.

C14 Variance decomposition. The text reports the order-A fractions with the order named, gives the order-B class and subject terms
beside them, uses "between-subject sum of squares" for the descriptive quantity, and says that these are sequential projection
increments of trial-weighted sums of squares, not uniquely separated variance sources. "Subject/class/interaction decomposition"
appears only with the sequential definition attached.

C15 Words retired for revision 2 (in addition to v1 C9): "isolated contribution of run adjacency" (unless from B21 with its design
stated), "eliminated subject information", "removed the subject information", "identical composition" and "differ only" for
any arms that are not the count-matched arms of B21, and any statement that a ladder contrast bounds a mechanism.

C16 Nested-subclass theory. The Jamalabadi et al. connection is stated as heuristic and conditional (their result concerns
subclasses nested within classes; here every subject contributes both classes), and shared subject offsets are distinguished
from subject-by-class structure in the sentences that cite it.

C17 Supersession record. v1 C1's clause "If the CI of Δ(P1 − P2) includes 0 while Δ(P0 − P1) does not, the manuscript attributes
the pooled inflation to run/temporal adjacency ... and the title changes accordingly" is withdrawn as an inference rule from this
freeze on; the frozen v1 bytes are unchanged and the supersession is recorded in DECISION_LOG.md.

## D. Compute plan and stop rules
CPU: ≤ 36 joblib workers (OMP/MKL/OPENBLAS threads = 1) for E8-v2, E3a-v2, E3b-v2, E16 and E17 classical; GPUs 1–3 only (GPU 0 is
another user's job) for the E16/E17 networks, one process per GPU, three seeds. Estimates: E8-v2 ≈ 2 h; E3a-v2 ≈ 40 min;
E3b-v2 ≈ 10 min; E16 + E17 classical (both datasets) ≈ 30 min; E16 networks ≈ 1 h per GPU; E17 networks (3 arms × 3 rotations ×
3 seeds × 4 paradigms × 2 networks = 216 fits) ≈ 1.5 h per GPU on three GPUs. If a network job exceeds twice its estimate the
seeds drop 3 → 2 (disclosed). Every run: one log, a JSON manifest (command, seeds, versions, code revision, output hashes), exit
code 0 AND schema-validated outputs before use. Smoke runs (≤ 20 subjects) are never reported.

## E. Reporting rules
As plan v1 §E, plus: the response quotes each second-round comment in full and gives the four-place resolution (code, result
files, tables/figures, text); the implementation error is disclosed in the manuscript; the public repository update carries the
corrected code, the rev2 results, this plan at its frozen bytes, the transformation record for the public copy of plan v1 and
the public decision log; the data statement cites the release tag and a commit permalink with the sha256 of the distributed plan
files; every ladder quantity is called a protocol contrast; the supplement adds the feasibility table, the E3a-v2 table (schemes
i–v), the sequential decomposition table (both orders and the revision-1 values), the P0s construction table and the
matched-controls tables (per decoder, per paradigm, both datasets, the full-pipeline TS+LR variant, logged epochs for the
networks).
