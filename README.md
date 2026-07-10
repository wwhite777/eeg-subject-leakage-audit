# Subject Leakage in EEG Motor-Imagery Evaluation

A leakage-audited re-evaluation of EEG motor-imagery decoding across **three
evaluation protocols × four decoder families × two public datasets (118 subjects)**.
The study isolates *why* subject-agnostic ("pooled") cross-validation inflates
reported accuracy, and prescribes how to evaluate honestly.

## TL;DR

Pooled/record-wise cross-validation — where trials from one subject fall in both
train and test — inflates EEG decoding accuracy. We show this inflation:

- **is not a model-capacity effect** — the two highest-capacity models (EEGNet,
  ShallowFBCSPNet) are essentially immune, while the mid-capacity Riemannian
  tangent-space pipeline inflates most;
- **is not caused by the Riemannian reference mean** — estimating the tangent-space
  reference transductively (on train **and** test) vs inductively changes accuracy
  by `< 0.001` on both datasets;
- **is classifier overfitting** to subject-correlated, high-dimensional tangent
  features — it vanishes under strong regularization and peaks at moderate penalty;
- **grows with feature dimensionality *and* small subject count** — on 9-subject
  BCI-IV-2a even CSP+LDA inflates, while on 109-subject EEGBCI it does not.

**Recommendation:** use subject-disjoint (grouped) cross-validation, regularize the
classifier, and report the within-vs-cross *calibration gap* next to headline accuracy.

## Key numbers

Leakage inflation = `acc(pooled_random) − acc(cross_subject)`, mean over paradigms:

| decoder | capacity | EEGBCI-109 | BCI-IV-2a (9) |
|---|---|---|---|
| CSP + LDA | low | −0.014 | +0.048 |
| Riemannian TS + LR | mid | +0.068 | +0.101 |
| ShallowFBCSPNet | high | +0.012 | — |
| EEGNet | high | +0.001 | — |

Paired Wilcoxon signed-rank (n=109 subjects, subject-averaged): Riemannian leakage
inflation `+0.068` is highly significant (`p = 1.4e-17`); CSP+LDA does **not** inflate
(`Δ = −0.014`, significantly non-positive); calibration gaps `+0.100` / `+0.088`
(`p < 1e-17`). Deep-net inflation CIs include zero.

Reference-source ablation (transductive − inductive, honest cross-subject CV):
EEGBCI `−0.0005`, BCI-IV-2a `+0.0004` (both CIs straddle 0 → reference is not the leak).

Classifier regularization sweep (EEGBCI, mean leakage inflation vs LR penalty `C`):
`C=1e-3: −0.009 · 1e-2: +0.042 · 0.1: +0.072 · 1: +0.065 · 10: +0.053 · 100: +0.053`.

Within-vs-cross calibration gap is large for classical methods (up to +0.245) but
deep nets largely close it (EEGNet cross-subject 0.755 vs 0.576–0.619 classical).

## Datasets (public)

- **PhysioNet Motor Movement/Imagery (EEGBCI)** — 109 subjects, 64ch @ 160 Hz, four
  two-class paradigms (imagined/executed left-vs-right fist, fists-vs-feet).
- **BCI Competition IV-2a (BNCI2014-001)** — 9 subjects, 22ch @ 250 Hz, left-vs-right
  hand imagery.

Both fetched through open loaders; band-passed to the sensorimotor band.

## Repository layout

```
test/experiments/eegbci_full/     # PhysioNet MMI: prepare, protocols, ablations, figures
  eegbci_full_dataset_prepare.py       # download + epoch all 109 subjects -> npz cache
  eegbci_full_leakage_experiment.py    # 3 protocols x CSP+LDA / Riemannian
  eegbci_full_deep_baselines.py        # EEGNet / ShallowFBCSPNet rungs
  eegbci_reference_leakage_ablation.py # inductive vs transductive reference
  eegbci_classifier_mechanism.py       # LR regularization sweep
  eegbci_full_make_figures.py, make_mechanism_figures.py
test/experiments/bci2a/           # BCI-IV-2a replication
results/eegbci_full/, results/bci2a/   # result tables (CSV)
result/figure/exp1|exp2|exp3/          # figures (exact data)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-learn mne braindecode pyriemann moabb matplotlib

# EEGBCI: prepare (downloads ~109 subjects), then run
python test/experiments/eegbci_full/eegbci_full_dataset_prepare.py
python test/experiments/eegbci_full/eegbci_full_leakage_experiment.py --tag run1
python test/experiments/eegbci_full/eegbci_full_deep_baselines.py --models eegnet_v4,shallow_fbcsp
python test/experiments/eegbci_full/eegbci_reference_leakage_ablation.py
python test/experiments/eegbci_full/eegbci_classifier_mechanism.py

# paired significance tests (Wilcoxon signed-rank)
python test/experiments/eegbci_full/eegbci_paired_stats.py

# BCI-IV-2a
python test/experiments/bci2a/bci2a_dataset_prepare.py
python test/experiments/bci2a/bci2a_replication.py
```

## Status

Complete two-dataset audit with per-subject bootstrap confidence intervals and
paired significance tests. Manuscript prepared for the *Journal of Neural
Engineering* (measurement/methodology article). Full write-up (manuscript +
figures) is maintained separately.
