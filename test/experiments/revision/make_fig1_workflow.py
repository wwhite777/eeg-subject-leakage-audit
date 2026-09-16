#!/usr/bin/env python3
"""Figure 1 (revision) — a NEUTRAL workflow diagram of the audit (Referee 2, point 4): datasets, preprocessing,
decoders, the protocol ladder, the controlled analyses and the outputs. No finding, no conclusion, no
"capacity ladder"; the regularization direction is stated correctly (larger C = weaker penalty).
Built with the lab's figkit toolkit; check_overlaps() must be empty.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, "<path-to-figkit>")
from figkit import Fig1  # noqa: E402

OUT_DIR = Path("result/figure/revision")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGES = [
    ("Datasets", "green",
     "EEGBCI (PhysioNet MMI): 109 subjects,\n64 ch, 160 Hz, 4 paradigms\n(MI / ME, L/R fist, fists/feet),\n3 runs per paradigm\n\nBCI-IV-2a: 9 subjects, 22 ch,\n250 Hz, L/R motor imagery,\n2 sessions × 6 runs"),
    ("Preprocessing", "blue",
     "Zero-phase FIR band-pass\n7–30 Hz (EEGBCI), 8–30 Hz (2a)\n0–4 s epochs after the cue\nNo trial rejection\n\nNetworks: per-trial z-score\n(matched-preprocessing control:\nglobal scale only)"),
    ("Decoders", "purple",
     "CSP+LDA — 6 log-variance features\nTS+LR — OAS covariance, tangent space\nat the training Fréchet mean,\n$\\ell_2$ logistic regression\n($N_c(N_c+1)/2$ = 2080 / 253 features)\nShallowFBCSPNet, EEGNet\n(trainable parameters reported)"),
    ("Protocols", "red",
     "within-subject (LORO / session)\nP0 trial-random\nP1 run-disjoint, subject-overlapping\nP1s session-disjoint (2a)\nP2 subject-disjoint\n(GroupKFold; LOSO on 2a)\nmatched training fraction 2/3\n(EEGBCI); repeated partitions"),
    ("Analyses", "teal",
     "penalty sweep (larger C = weaker)\nPCA-dimension sweep\nsubject information: ID decoding,\nvariance decomposition, mean removal,\nrecentering, weight alignment\nreference tests (transductive;\ntarget subject in / out)\nwidth sweeps; N × d factorial;\ncalibration-size curve"),
    ("Outputs", "gold",
     "Per-subject accuracy (unit = subject)\nΔ(P1 − P2): subject-overlap part\nΔ(P0 − P1): run / temporal part\nΔ(within − P2): calibration gap\nsubject-bootstrap 95 % CI,\npaired Wilcoxon + sign-flip,\nHolm correction, TOST (±0.01)"),
]


def main() -> int:
    f = Fig1(16, 9)
    f.title("Subject-mixed cross-validation in EEG sensorimotor decoding: audit workflow",
            "two public datasets → matched preprocessing → four decoders → protocol ladder → controlled analyses → per-subject statistics")
    n = len(STAGES)
    x0, gap = 2.5, 1.4
    colw = (95.0 - (n - 1) * gap) / n
    y, h = 30.0, 46.0
    xs = [x0 + i * (colw + gap) for i in range(n)]
    for i, (header, th, body) in enumerate(STAGES):
        f.panel(xs[i], y, colw, h, header, theme_name=th, number=i + 1)
        f.text(xs[i] + colw / 2, y + h * 0.46, body, fontsize=8.4, ha="center", va="center")
        if i < n - 1:
            f.arrow(xs[i] + colw + 0.15, y + h / 2, xs[i + 1] - 0.15, y + h / 2, style="main")
    f.callout(x0, 15.0, 95.0, 11.0, "Reading guide",
              textwrap.fill("Protocols differ in which recording units may straddle the train/test boundary; the ladder separates the run/temporal and the subject-overlap "
                            "components. Every quantity is computed per subject and summarized with confidence intervals; the analyses in stage 5 test whether "
                            "the tangent-space features carry subject information and whether the classifier, the reference mean, the feature dimension, the "
                            "network width and the training-subject count change the protocol differences.", 190), kind="recommendation")
    short = ["EEGBCI 109 + BCI-IV-2a 9 subjects", "band-pass, 0–4 s epochs", "CSP+LDA, TS+LR, two CNNs", "within / P0 / P1 / P1s / P2", "sweeps, subject information, references", "per-subject Δ with CIs and tests"]
    f.banner([(s[0], sh) for s, sh in zip(STAGES, short)], themes=[s[1] for s in STAGES])
    bad = f.check_overlaps()
    if bad:
        print("OVERLAPS:", bad)
        return 1
    out = OUT_DIR / "fig1_workflow_rev1.png"
    f.save(str(out), dpi=300)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
