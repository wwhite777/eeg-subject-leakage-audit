# Decision log (public) — scientific decisions of the subject-leakage audit, revisions 1 and 2

This log lists every decision taken after results had been seen, with what had been seen at the time. It is the record referred to
in the manuscript's statistical-analysis section and data-availability statement. Dates are UTC. Internal tooling, review
logistics and drafts are not part of this record; the frozen analysis plans are `research/REVISION_PLAN_v1.md` (first revision)
and `research/REVISION_PLAN_v2.md` (second revision), with checksums in the companion `.sha256` files and in
`research/PLAN_V1_TRANSFORMATION_RECORD.md`.

## 2026-09-16 04:35 — First-revision plan frozen before any new run
- Plan v1 (SHA-256 8d9b1f5b…) fixed the analyses E1–E14 and the decision rules C1–C9 that answer the first-round referee reports.
  None of the new analyses had run; the July 2026 results of the original submission were known and are kept as the "original
  configuration" in the supplement. All new analyses are reviewer-requested, post-submission analyses, not pre-registered before
  the original submission.

## 2026-09-16 05:09 — Exploratory analysis E15 added after seeing E1/E2
- After the ladder results showed the trial-random split to be LESS accurate than the run-disjoint split for CSP+LDA and for
  strongly regularized TS+LR, an analysis of the class-proportion structure of stratified trial-random folds (E15: no model fitted)
  was added. It is labelled exploratory; no confirmatory claim depends on it; rules C1–C9 unchanged.

## 2026-09-16 06:33 — Calibration-size curve (E12) restricted to the two imagery paradigms
- The first launch on all four paradigms was stopped after 66 minutes without completing a paradigm (the machine was
  oversubscribed); the relaunch was restricted to the imagery paradigms. This is a deviation from plan v1 (its "twice the
  estimate" trigger was not literally met) and is disclosed in the manuscript.

## 2026-09-16 10:05 and 2026-09-21 — Corrections after an independent recomputation of the first-revision package
- The frozen rule C3 has no "partially resolved" branch; the mechanism is stated as UNRESOLVED and the centring effect is reported as
  a transformation effect, not a mediation fraction.
- Multiplicity families redefined as the displayed tables (every table, including per-paradigm supplements and the table of raw
  reference tests); exact conditional signed-rank test for n ≤ 25 with differences rounded to 12 decimals; per-subject pooling
  within subject (n = 109), fold values weighted by test-trial counts; the factorial bootstrap resamples whole subject draws within
  each N with centred regressors (the subject-count slope became negative with a CI excluding zero in all four paradigms); sign
  counts from the rounded differences; superseded partial exports archived and a source index released.
- Title changed from a categorical to a bounded form ("dominates" rather than "not run adjacency"); the run-adjacency statement was
  written as bounded by its upper confidence limits; seeds and training fractions described per experiment; several wordings
  narrowed to "consistent with" or "interpretation".

## 2026-10-30 03:30 — Second revision opened on the second-round report
- Referee 1 identified an implementation error in `test/experiments/revision/e8_factorial.py`: a PCA dimension that exceeded one
  fold's training size was skipped for that fold only, and the trials of the skipped fold were counted as misclassified. Confirmed
  from the released code and table before any change: one condition (real_fists_feet, N = 18, d = 512, draw 6; subject-disjoint
  fold sizes 511, 516, 517) was affected; a count audit over all 820 requested conditions found no other partially scored condition
  (the 40 N = 9, d = 512 conditions were already absent as completely infeasible).
- The public copy of plan v1 differs from the frozen bytes because the release staging replaced the journal's name; the transformation
  record states both checksums and the exact substitutions.

## 2026-10-30 04:02 — Second-revision plan frozen after a pre-freeze design review (SHA-256 888c7635…)
- Plan v2 (B17–B22, rules C11–C17) was drafted, attacked in a design review, amended, and then frozen; no second-revision result
  existed. Amendments made before the freeze: the fixed-test matched-composition control was redesigned so that its arms have
  identical subject, run and per-class trial counts (target/donor pairing on an unchanged background block), share one tangent
  reference estimated from the background and one validation slice for the networks, and include a same-run-adjacency arm of equal
  count on the same test set; the subject-by-class-stratified trial-random split is reported as a protocol contrast with fail-closed
  construction assertions (a correlation acceptance threshold in the draft was removed because the construction itself implies a
  positive correlation in some paradigms); the title rule compares the inclusion and adjacency contrasts only within that common
  framework and otherwise prescribes a neutral title; the identity probe gained a split-reference control that separates shared
  centring-estimation error from cross-run identity, and its inference is limited to the decodability of the specified centred
  representation; the variance decomposition uses sequential (Type I) sums of squares in two stated orders with hand-checkable
  fixtures; the factorial rerun preserves the released PCA construction and masks only the scoring, with the four-paradigm
  aggregate requiring complete draws; the attribution branch of plan-v1 rule C1 is superseded.
- Every ladder difference is henceforth named a protocol contrast; the two title candidates are fixed in rule C12 and the final
  wording is the corresponding author's decision.

## 2026-10-30 04:40 — One reported quantity added before the matched-control results existed
- A reader challenge of the author brief asked for the paired per-subject difference between the inclusion and the adjacency
  contrasts (Δ_incl − Δ_adj) to be reported beside the two contrasts. It was added to the statistics script and the matched-controls
  table as a descriptive quantity with its subject-bootstrap CI. Rule C12 is unchanged and still decides the title by the
  non-overlap of the two contrasts' intervals. Data exposure at the time: the identity-probe and variance-decomposition results of
  the second revision had been seen; no factorial or matched-control result existed yet (those runs finished after 05:00).

## 2026-10-30 04:50–05:15 — Corrected factorial (E8-v2): implementation accepted, rerun reproduces every retained cell
- The second-round report identified an error in the factorial script: feasibility of a PCA dimension was decided per fold, so one
  condition (executed fists/feet, N = 18, d = 512, draw 6) was scored with one unevaluated fold whose trials counted as errors.
- Fix (frozen plan v2 §B17): feasibility decided over all folds of both protocols before scoring; the PCA request min(512, n_train − 1)
  unchanged; a coverage assertion before any accuracy; complete-draw aggregation for the four-paradigm cell. An independent
  re-derivation (fresh context) reproduced the feasibility table (164 draws, 820 conditions, 41 infeasible) with zero differences and
  confirmed each item by code line; a planted per-fold skip fails the regression suite.
- Rerun: 1886 retained conditions reproduce the first-revision values exactly (maximum absolute difference 0); the one flagged
  condition is excluded for both protocols; one per-paradigm cell changes (+0.191 → +0.168); the executed-fists/feet slopes change in
  the third decimal; the rule-C11 re-read finds every conclusion unchanged.

## 2026-10-30 04:55–05:05 — Identity probe (E3a-v2): frozen definition enforced over the cached order
- The independent re-derivation found that scheme (v) of the identity probe (split-reference halves) alternated within the epoch
  caches' class blocks, whereas §B18 defines alternating trials in recording order. Decision: comply with the frozen text — recover the
  recording order from the raw annotations (fail-closed on count mismatch), rerun, re-verify against the verifier's chronology
  (109/109 subjects agree). Values of the superseded run were seen before the decision; the fix changes only scheme (v) and the
  (ii) − (v) difference, and the decision does not depend on their direction. Superseded files are archived, not deleted.
- Results (macro accuracy, 109 subjects, four paradigms): uncentred 0.994–0.998; independent-reference centring 0.676–0.699;
  training-run centring ≈ chance (0.009–0.010); all-trial centring on one-run assignments 0.017–0.023; split-reference control
  0.503–0.555; (ii) − (v) 0.129–0.186 with CIs excluding zero. Per rule C13 the text states decodability of the specified centred
  representation and the shared-estimation-error contribution, not removal or exploitation of identity.
- Variance decomposition (E3b-v2): sequential (Type I) sums of squares in two orders (subject 0.859–0.869; class given subject
  0.0002–0.0004; interaction 0.003–0.007; residual 0.127–0.138; order B identical to three decimals; fixtures reproduced to 1e-15).

## 2026-10-30 05:12 — Matched controls (E16/E17) and the title rule C12
- Results (109 subjects, paradigm mean within subject; Holm over the 12-contrast table): target inclusion at matched composition
  Δ_incl: TS+LR +0.073 [+0.061, +0.085], EEGNet +0.037 [+0.027, +0.046], ShallowFBCSPNet +0.021 [+0.009, +0.032], CSP+LDA +0.004
  [−0.004, +0.012]; same-run adjacency Δ_adj: TS+LR +0.016 [+0.009, +0.023], CSP+LDA +0.004, ShallowFBCSPNet −0.001, EEGNet −0.007;
  paired Δ_incl − Δ_adj (descriptive): TS+LR +0.057 [+0.043, +0.072], EEGNet +0.044, ShallowFBCSPNet +0.021, CSP+LDA −0.001.
  Δ(P0s − P1) between −0.008 and +0.006; Δ(P0 − P0s) negative for the classical decoders (−0.017, −0.008). No matched-cell shortfall
  occurred (matching tables released).
- Rule C12 (frozen before the runs): (a) met; (b) lower CL of Δ_incl (+0.061) > upper CL of Δ_adj (+0.023), met; (c) point ordering
  in three of four decoders, met → the Title A wording is permitted. The nine-subject between-session analogue (descriptive) does
  not rank the two for TS+LR (adjacency +0.059 vs inclusion +0.038); the abstract and discussion state this so that the title is read
  as a within-session statement.

## 2026-10-30 05:55 — Readability review of the revised text: one inference softened, no value changed
- A locked cold read of the revised PDF by a fresh context found one over-inference: that the agreement between the matched
  inclusion contrast (+0.073) and the ladder's subject-overlap contrast (+0.068) shows that the change of training composition
  "did not drive" the ladder contrast. The sentence is corrected to state only that an advantage of that size persists at matched
  composition; the agreement does not identify the composition contribution, because the designs differ in test set and training
  composition. The ordering statement is qualified as descriptive, holding for TS+LR and the two networks and unresolved for CSP+LDA.
- The same read found a layout defect (one table ran off its page) and undefined terms; these are fixed by insertions. The
  AI-assistance disclosure is corrected to name both tools used for the second-revision code.

## 2026-10-30 07:00–08:00 — Fix round after two independent reviews and two cold reads: claim narrowings and digit corrections, no value changed
- Two independent reviewers (one per model family, blind Phase A) reproduced every second-revision value and the title rule. Their
  findings, and those of two cold reads of the PDF, led to wording corrections only: (a) the statement that the agreement between
  the matched inclusion contrast (+0.073) and the ladder's subject-overlap contrast (+0.068) showed that the change of training
  composition "did not drive" the ladder contrast was withdrawn — the agreement does not identify the composition contribution;
  (b) the nine-subject ladder sentences name the protocols and training fractions instead of attributing the differences to
  "subject overlap" or "session separation"; (c) the downward bias of globally stratified folds is stated as "can bias" with its
  decoder-dependent sign (positive for EEGNet); (d) the factorial subject-count slope is reported as negative in all four paradigms
  with the confidence interval excluding zero in three — for imagery left/right the upper limit (−0.0001) depends on the bootstrap
  resampling seed, which corrects the earlier entry's "CI excluding zero in all four paradigms"; (e) three displayed digits were
  corrected to their sources (−0.020, +0.059, 0.032) and the P0s subject-prior accuracy is 0.50 (not "0.50–0.51"); (f) the
  split-reference identity contrast is described as changing two things at once (shared vector and trials per estimate), under a
  conditional-independence assumption that a run-level component would violate; (g) the paired ordering is stated as descriptive,
  holding for TS+LR and the two networks (one EEGNet paradigm reversed it) and unresolved for CSP+LDA.
- Reporting additions: per-paradigm paired differences, P0s correlations and subject-prior accuracies per paradigm, second-revision
  network training epochs, the matching and shortfall rule (no shortfall occurred), the omitted final cues (103 runs, 11 subjects).
- Outcome exposure: all second-revision results had been seen; no rule, threshold or value changed.

## 2026-10-30 08:20–09:10 — After two further independent re-checks: one measured qualification, two wordings, one caption, disclosure accuracy; no value changed
- A cold read of the revised PDF found that the factorial's qualification "the upper limit lies at zero within the Monte Carlo
  error of the bootstrap" stated no method or number. The sensitivity was then measured (post-results, descriptive; frozen rules
  unchanged): over 50 bootstrap seeds the imagery left/right subject-count upper limit ranged from −0.0002 to +0.0004 and 16 of
  the 50 intervals excluded zero, all below zero; the per-seed table and its generator are released
  (results/revision/e8_rev2_ols_seed_sensitivity*.csv) and an independent recomputation matched every value.
- Wording: the conclusion says the subject's presence "carried most of the gain" (same-run adjacency added +0.016 for TS+LR);
  "independent halves" became "disjoint halves"; the supplementary identity-probe caption carries the same two-things-at-once
  qualification as the main text.
- Disclosure: the generative-AI statement now says the tools assisted in drafting and editing the manuscript text and in writing
  and cross-checking the analysis and figure code under the authors' specification (the earlier sentence said "editing" only).
- Record note: the template line "(Entries for the second-revision results follow as they are recorded.)" that closed this log
  was replaced by the dated entries above; no earlier entry was changed.

## 2026-10-30 10:40 — Record correction (append-only from here on)
- The note of 08:20–09:10 said "no earlier entry was changed". That was inaccurate in one respect: the heading of the plan-freeze
  entry had been edited in place twice (a placeholder hour "04:1x" → "04:15" → "04:02", the time recorded in the private log).
  No entry body was altered. From this note on, corrections are appended, never edited in place.
- The released second-revision plan (REVISION_PLAN_v2.md) is distributed byte-for-byte (its checksum is cited in the manuscript);
  its third line names, by its internal task identifier, the review task that attacked the draft before the freeze. The
  identifier is kept as part of the frozen record; it is not a result, a secret or a venue name. (This note was reworded
  before publication so that the public tree carries the identifier only inside the frozen plan file.)
