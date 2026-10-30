# Source index — current statistics exports (tags rev1 + rev2, regenerated 2026-10-30 for the second revision)

This index maps manuscript tables, figure panels and numerical claims to their released source exports. The generated tables_provenance.txt, tables_matched_provenance.txt and per-table input hash comments record the actual generator inputs; manuscript prose also contains stated design constants. Earlier partial exports are kept privately for provenance and cited nowhere; the public tree ships only the current exports.

| Table / figure / claim | Source file | Generator |
|---|---|---|
| Table 1 (decoder settings), Table 2 (protocols) | text only (no statistics) | make_tables.py |
| Table 3 (EEGBCI ladder, 12-contrast Holm family), Table 4 (BCI-IV-2a, 24-contrast family), Table S1 (per-paradigm ladder, 48-contrast family), Table S8 (preprocessing controls, 8-contrast family) | stats_rev1_contrasts.csv (columns family, family_size, p_holm, n_positive/n_negative/n_zero from the 12-decimal-rounded differences), stats_rev1_ladder_means.csv | e_stats.py → make_tables.py |
| Table 5 (reference tests: E6 target-in-reference, E14 transductive; TOST decisions unadjusted, Wilcoxon p Holm-adjusted over the 11 rows) | stats_rev1_tost.csv | e_stats.py → make_tables.py |
| Table S2 (classifiers, 18-contrast family), Figure S1 | stats_rev1_e5_contrasts.csv | e_stats.py → make_tables.py, figS2_classifiers.py |
| Table S3 (centring / recentering, 6-contrast family), Figure 5(c) | stats_rev1_e3c_contrasts.csv; e3a_rev1.csv (identity decoding), e3b_rev1.csv (variance decomposition), e3d_rev1.csv (weight alignment) | e_stats.py, e3_subject_info.py |
| Table S4 (regularization sweep, 16 tests), Figure 4 | stats_rev1_e2_sweep.csv | e_stats.py |
| Table S5 (PCA sweep, 22 tests), Figure 6(a) | stats_rev1_e4_sweep.csv | e_stats.py |
| Table S6 (width sweeps, 10 tests), Figure 7 | stats_rev1_width.csv | e_stats.py |
| Table S7 (factorial; OLS with centred regressors, bootstrap over whole draws within each N), Figure 6(b,c) | stats_rev2_factorial.csv; e8_rev2_per_draw.csv; e8_rev2_feasibility.csv | e_stats.py --tag rev2; make_tables.py --tag rev1 --factorial-tag rev2; fig6_dimension.py |
| §3.6 and Discussion (v): bootstrap-seed sensitivity of factorial OLS confidence limits (descriptive) | e8_rev2_ols_seed_sensitivity.csv; e8_rev2_ols_seed_sensitivity_summary.csv | e8_ols_seed_sensitivity.py (TS+LR; 2000 whole-draw resamples within each N; 50 seeds 20260916–20260965) |
| Table S9 (exploratory stratification; all five seeds) | e15_rev1_summary.csv, e15_rev1_per_cell.csv | make_tables.py |
| Table S10 (network training details) | e9_rev1_all_per_fold.csv, e11_rev1_t12_per_fold.csv, e11_rev1_t34_per_fold.csv, e10_rev1_per_fold.csv | make_tables.py |
| Table S11 and Figure 8 (calibration curve; 12-contrast family; k is a cap) | stats_rev1_e12_calibration.csv; e12_rev1_availability.csv | e12_stats.py |
| Table S12 (original five-fold subject-grouped split vs LOSO, descriptive) | stats_rev1_contrasts.csv (family "none", contrast groupkfold5_vs_loso), stats_rev1_ladder_means.csv | make_tables.py |
| Training accuracy and train–test gaps (Table 3 / §3.1 text), weight norms and training accuracy of the regularization sweep (§3.3, Figure 4) | e1_eegbci_rev1_per_fold.csv (train_acc per fold), e2_rev1_per_fold.csv (train_acc, w_norm per fold) | make_tables.py, fig4_sweep.py, text |
| Per-subject inputs | e1_eegbci_rev1_per_subject.csv, e1_bci2a_rev1_per_subject.csv, e1_eegbci_rev1_ztrial_per_subject.csv, e9_rev1_all_per_subject.csv, e9_rev1_global_all_per_subject.csv, e10_rev1_per_subject.csv, e11_rev1_all_per_subject.csv, e2_rev1_per_subject.csv, e4_rev1_per_subject.csv, e5_rev1_per_subject.csv, e3c_rev1_per_subject.csv, e6_rev1_per_subject.csv, e14_rev1_per_subject.csv, e7_rev1_width_all_per_subject.csv, e8_rev1_per_draw.csv, e12_rev1_mi_lr_per_subject.csv, e12_rev1_mi_ff_per_subject.csv | experiment scripts |


| Table 6; Tables S15–S18 (matched-control means and contrasts) | stats_rev2_matched.csv; stats_rev2_matched_means.csv | e_stats_r2.py → tables_r2_matched.py --tag rev2 |
| Table S13; Figure 5(b) (rev2 identity probe) | e3a2_rev2_per_subject.csv; e3a2_rev2.csv; e3v2_rev2_manifest.json (recording-order and run-assignment provenance) | e3v2_subject_info.py → tables_r2_e3.py; fig5_mechanism.py |
| Table S14; Figure 5(a) (sequential trial-weighted sums of squares) | e3b2_rev2.csv; e3b_rev1.csv (legacy comparison only) | e3v2_subject_info.py → tables_r2_e3.py; fig5_mechanism.py |
| Table S9 P0s rows and §3.10 | e15_rev2_summary.csv; e15_rev2_per_cell.csv (released cell-level detail); existing P0/P1/P2 rows remain from e15_rev1_summary.csv | e15_stratification_artifact.py → make_tables.py --tag rev1 --p0s-tag rev2 |
| Table S15 construction audit and companion matched-design count checks (§2.5, Table 6 caption) | e16_eegbci_rev2_construction.csv; e16_bci2a_rev2_construction.csv (generated S15 rows); e17_eegbci_rev2_matching.csv; e17_bci2a_rev2_matching.csv; e17_eegbci_rev2_counts.csv; e17_bci2a_rev2_counts.csv (companion donor/target and arm-count evidence) | e1_ladder_classical.py (P0s construction); e17_fixed_test.py; tables_r2_matched.py (S15 uses E16 construction rows; E17 files document matching and shortfalls) |
| Table S19 (rev2 network training details) | e9_rev2_t12_per_fold.csv; e9_rev2_t34_per_fold.csv; e10_rev2_per_fold.csv; e9_rev2_fixed_t12_per_fold.csv; e9_rev2_fixed_t34_per_fold.csv; e10_rev2_fixed_per_fold.csv | e9_deep_ladder.py; e10_bci2a_deep.py → tables_r2_matched.py |
| Rev2 matched-control per-subject inputs | e1_eegbci_rev2_per_subject.csv; e1_bci2a_rev2_per_subject.csv; e17_eegbci_rev2_per_subject.csv; e17_bci2a_rev2_per_subject.csv; e9_rev2_t12_per_subject.csv; e9_rev2_t34_per_subject.csv; e10_rev2_per_subject.csv; e9_rev2_fixed_t12_per_subject.csv; e9_rev2_fixed_t34_per_subject.csv; e10_rev2_fixed_per_subject.csv | experiment scripts → e_stats_r2.py |

## Superseded (revision 2)
- The superseded directory is a private provenance archive and is not distributed; the public tree ships only the current exports. It retains the first E3a-v2/E3b-v2 run (2026-10-30, 04:22–04:26Z), whose scheme (v) alternated within the cache's class blocks instead of in recording order; the current exports use the corrected recording-order rerun and the earlier run is cited nowhere. The E3b-v2 values in the private archive are identical to the rerun's.


Generator invocations (regenerated 2026-10-30 for the second revision): `make_tables.py --tag rev1 --factorial-tag rev2 --p0s-tag rev2 --out-dir <tables>`;
`tables_r2_matched.py --tag rev2 --out-dir <tables>`; `tables_r2_e3.py --tag rev2 --out-dir <tables>`.
The first preserves the first-revision tables while selecting rev2 for S7 and the additional S9 P0s rows.
The second generates Table 6 and S15–S19; S13/S14 carry their own source hashes from the third.
The E17 matching/count files are companion construction evidence; they do not supply displayed S15 cells.
