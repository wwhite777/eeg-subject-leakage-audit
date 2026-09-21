#!/usr/bin/env python3
"""Figure 1 (revision) — a NEUTRAL workflow diagram of the audit (Referee 2, point 4): datasets, preprocessing,
decoders, the protocol ladder, the controlled analyses and the outputs. No finding, no conclusion, no
"capacity ladder"; the regularization direction is stated correctly (larger C = weaker penalty).

D-08/D-11: the figure is drawn at the manuscript line width (6.024 in) instead of 16 in, so the placement
scale is 1.000 and the in-box text prints at its generated size (>= 8 pt) instead of about 4 pt. The six
stages are laid out 3 x 2 so that each panel is wide enough for 8.5 pt text. Three checks run before the
file is written: figkit check_overlaps() (boxes), text inside its own panel, and nothing off the canvas.

F-r003-5: the decoder panel used mathtext ($\\ell_2$), whose subscript printed at 5.95 pt although the text
artist reported 8.5 pt. It now reads "L2-penalized", and the PDF companion is written first and re-measured
glyph by glyph with pymupdf (fig_common.assert_pdf_min_glyph) before the PNG is written.

F-r002-4: the matched training fraction is stated as EEGBCI-only (the IV-2a fractions differ).
Built with the lab's figkit toolkit.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, "<path-to-figkit>")
from figkit import Fig1  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as F  # noqa: E402

OUT_DIR = Path("result/figure/revision")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W_IN = 6.024   # \\linewidth of manuscript/v2, measured on main_clean.pdf (433.70 pt)
FIG_H_IN = 5.6     # keeps figure + caption inside 0.7 * textheight so the [t] float can be placed
BODY_FS = 8.5
MIN_PT = 8.0
MAX_BODY_LINES = 10

# Paragraphs, not pre-wrapped lines: each panel body is wrapped to the measured panel width so that
# a wording change can never push text through a box border again (D-08).
STAGES = [
    ("Datasets", "green", [
        "EEGBCI (PhysioNet MMI): 109 subjects, 64 ch, 160 Hz, 4 paradigms (MI / ME, L/R fist, fists / feet), 3 runs per paradigm",
        "",
        "BCI-IV-2a: 9 subjects, 22 ch, 250 Hz, L/R imagery, 2 sessions × 6 runs",
    ]),
    ("Preprocessing", "blue", [
        "Zero-phase FIR band-pass 7–30 Hz (EEGBCI), 8–30 Hz (BCI-IV-2a)",
        "0–4 s epochs after the cue; no trial rejection",
        "",
        "Networks: per-trial z-score (matched-preprocessing control: global scale only)",
    ]),
    ("Decoders", "purple", [
        "CSP+LDA — 6 log-variance features",
        "TS+LR — OAS covariance, tangent space at the training Fréchet mean, L2-penalized logistic regression (2080 / 253 features)",
        "ShallowFBCSPNet, EEGNet (parameter counts reported)",
    ]),
    ("Protocols", "red", [
        "within-subject (leave-one-run-out / session)",
        "P0 trial-random; P1 run-disjoint, subject-overlapping",
        "P1s session-disjoint (IV-2a); P2 subject-disjoint (GroupKFold; LOSO on IV-2a)",
        "Matched 2/3 training fraction on EEGBCI only",
    ]),
    ("Analyses", "teal", [
        "Penalty sweep (larger C = weaker penalty)",
        "PCA-dimension sweep",
        "Subject information: identity decoding, variance decomposition, mean removal, weight alignment",
        "Target-subject in/out tests",
        "Width sweeps; N × d factorial; calibration curve",
    ]),
    ("Outputs", "gold", [
        "Per-subject accuracy (unit = subject)",
        "Δ(P1 − P2): subject-overlap",
        "Δ(P0 − P1): run / temporal",
        "Δ(within − P2): calibration gap",
        "Subject-bootstrap 95 % CI, paired Wilcoxon + sign-flip, Holm correction, TOST (±0.01)",
    ]),
]

SHORT = [
    ("Inputs", "EEGBCI 109 + BCI-IV-2a 9\nsubjects; band-pass epochs"),
    ("Analyses", "Sweeps, subject\ninformation, references"),
    ("Outputs", "Per-subject Δ with CIs\nand paired tests"),
]


def wrap_paragraphs(f, paragraphs, fontsize, width_px):
    """Wrap each paragraph to `width_px` using the real renderer; blank paragraphs stay blank lines."""
    f.fig.canvas.draw()
    rend = f.fig.canvas.get_renderer()

    def w(txt):
        t = f.ax.text(0, 0, txt, fontsize=fontsize)
        out = t.get_window_extent(renderer=rend).width
        t.remove()
        return out

    lines = []
    for para in paragraphs:
        if not para.strip():
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            cand = (cur + " " + word).strip()
            if cur and w(cand) > width_px:
                lines.append(cur)
                cur = word
            else:
                cur = cand
        lines.append(cur)
    return lines


def rect_px(f, x, y, w, h):
    """Panel rectangle in display pixels."""
    t = f.ax.transData
    (x0, y0) = t.transform((x, y))
    (x1, y1) = t.transform((x + w, y + h))
    return x0, y0, x1, y1


def box_containment(f, inset=0.4):
    """Every text whose anchor sits inside a registered figkit box must fit inside that box."""
    from matplotlib.text import Text

    fig = f.fig
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = []
    for t in fig.findobj(match=lambda o: isinstance(o, Text)):
        if not str(t.get_text()).strip():
            continue
        ax_, ay = t.get_position()
        for (bx, by, bw, bh, _lab) in f._boxes:
            if bx <= ax_ <= bx + bw and by <= ay <= by + bh:
                bb = t.get_window_extent(renderer=rend)
                x0, y0, x1, y1 = rect_px(f, bx + inset, by + inset, bw - 2 * inset, bh - 2 * inset)
                over = max(x0 - bb.x0, bb.x1 - x1, y0 - bb.y0, bb.y1 - y1)
                if over > 0.5:
                    bad.append((str(t.get_text())[:40].replace("\n", " / "), round(float(over), 1)))
                break
    return bad


def check(f, texts, boxes):
    """(1) text inside its own panel, (2) nothing off the canvas, (3) no text below MIN_PT."""
    fig = f.fig
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = []
    for t, (bx, by, bw, bh) in zip(texts, boxes):
        bb = t.get_window_extent(renderer=rend)
        x0, y0, x1, y1 = rect_px(f, bx, by, bw, bh)
        over = max(x0 - bb.x0, bb.x1 - x1, y0 - bb.y0, bb.y1 - y1)
        if over > 0.5:
            bad.append((str(t.get_text())[:40].replace("\n", " / "), round(float(over), 1)))
    cw, ch = fig.get_size_inches() * fig.dpi
    from matplotlib.text import Text
    off, small = [], []
    for t in fig.findobj(match=lambda o: isinstance(o, Text)):
        if not str(t.get_text()).strip():
            continue
        if float(t.get_fontsize()) < MIN_PT - 1e-6:
            small.append((str(t.get_text())[:30], t.get_fontsize()))
        bb = t.get_window_extent(renderer=rend)
        o = max(-bb.x0, bb.x1 - cw, -bb.y0, bb.y1 - ch)
        if o > 0.5:
            off.append((str(t.get_text())[:40].replace("\n", " / "), round(float(o), 1)))
    return bad, off, small


def main() -> int:
    f = Fig1(FIG_W_IN, FIG_H_IN)
    f.text(50, 97.2, "Audit workflow: subject-mixed cross-validation in EEG decoding",
           fontsize=10.5, weight="bold", color="#111111")
    f.text(50, 94.2, "datasets → preprocessing → decoders → protocols → analyses → per-subject statistics",
           fontsize=8.5, color="#333333")
    x0, gap = 1.2, 1.6
    n_col = 3
    colw = (97.6 - (n_col - 1) * gap) / n_col
    rows = [60.0, 25.5]
    h = 32.0
    inset = 0.45
    width_px = (colw - 2 * inset) * FIG_W_IN * f.fig.dpi / 100.0
    texts, boxes = [], []
    for i, (header, th, paras) in enumerate(STAGES):
        r, c = divmod(i, n_col)
        x = x0 + c * (colw + gap)
        y = rows[r]
        body = "\n".join(wrap_paragraphs(f, paras, BODY_FS, width_px))
        n_lines = body.count("\n") + 1
        if n_lines > MAX_BODY_LINES:
            print(f"PANEL TOO LONG: {header} wraps to {n_lines} lines (budget {MAX_BODY_LINES})")
        f.panel(x, y, colw, h, header, theme_name=th, number=i + 1, header_fs=10.5)
        t = f.ax.text(x + colw / 2, y + h - 4.4, body, ha="center", va="top", fontsize=BODY_FS,
                      color="#222222", zorder=3, linespacing=1.28)
        texts.append(t)
        boxes.append((x + inset, y + 0.5, colw - 2 * inset, h - 3.9))
        if c < n_col - 1:
            f.arrow(x + colw + 0.15, y + h / 2, x + colw + gap - 0.15, y + h / 2, style="main")
    # wrap arrow: stage 3 (end of the top row) continues at stage 4 (start of the bottom row)
    f.arrow(x0 + 2 * (colw + gap) + colw / 2, rows[0] - 0.4, x0 + colw / 2, rows[1] + h + 0.4, style="thin")
    f.banner(SHORT, y=13.0, h=10.0, x0=1.2, x1=98.8, themes=["green", "teal", "gold"])
    f.callout(1.2, 0.5, 97.6, 11.0, "Reading guide",
              textwrap.fill("The protocols differ in which recording unit may straddle the train/test boundary; "
                            "the ladder separates run/temporal from subject-overlap effects.", 74), kind="recommendation")
    overlaps = f.check_overlaps()
    bad, off, small = check(f, texts, boxes)
    bad += box_containment(f)
    clashes = F.text_collisions(f.fig)
    for c in clashes:
        print("TEXT COLLISION:", c)
    for b in bad:
        print("TEXT OUTSIDE ITS PANEL:", b)
    for o in off:
        print("TEXT OFF CANVAS:", o)
    for s in small:
        print("TEXT BELOW 8 PT:", s)
    if overlaps:
        print("OVERLAPS:", overlaps)
    if overlaps or bad or off or small or clashes:
        return 1
    out = OUT_DIR / "fig1_workflow_rev1.png"
    pdf = out.with_suffix(".pdf")
    f.fig.savefig(str(pdf), facecolor="white")                   # vector companion first: the glyph guard reads it
    min_glyph = F.assert_pdf_min_glyph(pdf, "fig1_workflow_rev1")  # F-r003-5: real rendered sizes, mathtext included
    f.fig.savefig(str(out), dpi=300, facecolor="white")          # exact canvas: scale 1.000 at \linewidth
    print(f"fig1_workflow_rev1: canvas {FIG_W_IN:.3f} x {FIG_H_IN:.3f} in, smallest text artist {BODY_FS:.1f} pt, "
          f"smallest rendered glyph {min_glyph:.4f} pt (scale 1.000 at placement)")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
