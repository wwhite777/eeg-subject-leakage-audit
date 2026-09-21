# Source index — current statistics exports (tag rev1, regenerated 2026-09-21 for snapshot r003)

Every table, figure caption and numerical claim of the manuscript is generated from exactly these files. Files under
`superseded/` are outputs of earlier partial aggregation passes, kept for provenance only, and are cited nowhere.

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
| Table S7 (factorial; OLS with centred regressors, bootstrap over whole draws within each N), Figure 6(b,c) | stats_rev1_factorial.csv; e8_rev1_per_draw.csv | e_stats.py, fig6_dimension.py |
| Table S9 (exploratory stratification; all five seeds) | e15_rev1_summary.csv, e15_rev1_per_cell.csv | make_tables.py |
| Table S10 (network training details) | e9_rev1_all_per_fold.csv, e11_rev1_t12_per_fold.csv, e11_rev1_t34_per_fold.csv, e10_rev1_per_fold.csv | make_tables.py |
| Table S11 and Figure 8 (calibration curve; 12-contrast family; k is a cap) | stats_rev1_e12_calibration.csv; e12_rev1_availability.csv | e12_stats.py |
| Table S12 (original five-fold subject-grouped split vs LOSO, descriptive) | stats_rev1_contrasts.csv (family "none", contrast groupkfold5_vs_loso), stats_rev1_ladder_means.csv | make_tables.py |
| Per-subject inputs | e1_eegbci_rev1_per_subject.csv, e1_bci2a_rev1_per_subject.csv, e1_eegbci_rev1_ztrial_per_subject.csv, e9_rev1_all_per_subject.csv, e9_rev1_global_all_per_subject.csv, e10_rev1_per_subject.csv, e11_rev1_all_per_subject.csv, e2_rev1_per_subject.csv, e4_rev1_per_subject.csv, e5_rev1_per_subject.csv, e3c_rev1_per_subject.csv, e6_rev1_per_subject.csv, e14_rev1_per_subject.csv, e7_rev1_width_all_per_subject.csv, e8_rev1_per_draw.csv, e12_rev1_mi_lr_per_subject.csv, e12_rev1_mi_ff_per_subject.csv | experiment scripts |
