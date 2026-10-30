#!/usr/bin/env python3
"""Figure 1 (revision) — a NEUTRAL workflow diagram of the audit (Referee 2, point 4): datasets, preprocessing,
decoders, the protocol ladder, the controlled analyses and the outputs. No finding, no conclusion, no
"capacity ladder"; the regularization direction is stated correctly (larger C = weaker penalty).

D-08/D-11: the figure is drawn at the manuscript line width (6.024 in) instead of 16 in, so the placement
scale is 1.000 and the in-box text prints at its generated size (>= 8 pt) instead of about 4 pt. Seven
stages include the P0s protocol and the fixed-test controls. Three checks run before the
file is written: figkit check_overlaps() (boxes), text inside its own panel, and nothing off the canvas.

F-r003-5: the decoder panel used mathtext ($\\ell_2$), whose subscript printed at 5.95 pt although the text
artist reported 8.5 pt. It now reads "L2-penalized", and the PDF companion is written first and re-measured
glyph by glyph with pymupdf (fig_common.assert_pdf_min_glyph) before the PNG is written.

F-r002-4: the matched training fraction is stated as EEGBCI-only (the IV-2a fractions differ).
Built with the lab's figkit toolkit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from figkit import Fig1  # noqa: E402
except ImportError as exc:
    toolkit = os.environ.get("FIGKIT_PATH")
    if not toolkit or not (Path(toolkit) / "figkit.py").is_file():
        raise ImportError(
            "Figure 1 requires test/experiments/revision/figkit.py; "
            "restore the vendored module or set FIGKIT_PATH to its directory."
        ) from exc
    sys.path.insert(0, str(toolkit))
    from figkit import Fig1  # noqa: E402
import fig_common as F  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, default=HERE.parents[2] / "result" / "figure" / "revision")
OUT_DIR = parser.parse_args().out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W_IN = 6.024   # \\linewidth of manuscript/v2, measured on main_clean.pdf (433.70 pt)
FIG_H_IN = 6.5     # added controls retain 8.5 pt text at the manuscript line width
BODY_FS = 8.5
MIN_PT = 8.0

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
        "0–4 s epochs within recording; no artefact rejection",
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
        "P0 trial-random; P0s trial-random with subject-by-class stratification",
        "P1 run-disjoint, subject-overlapping",
        "P1s session-disjoint (IV-2a); P2 subjects permuted and cut into three groups (LOSO on IV-2a)",
        "Matched 2/3 training fraction on EEGBCI only",
    ]),
    ("Fixed-test controls", "orange", [
        "T / D / B: target / donor / background subject blocks",
        "Fixed test = half of the last run of T",
        "Count-matched training arms:",
        "A0 = B + D (other runs)",
        "A1 = B + T (other runs)",
        "A2 = B + T (same-run trials replace other-run trials)",
        "Shared B: other-run training, one tangent reference and one validation slice",
        "IV-2a analogue: last session",
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
        "Protocol contrasts: run-disjoint − subject-disjoint; trial-random − run-disjoint; within-subject − subject-disjoint (calibration)",
        "Matched arm contrasts: A1 − A0; A2 − A1",
        "Subject-bootstrap 95 % CI, paired Wilcoxon + sign-flip, Holm correction, TOST (±0.01)",
    ]),
]

CAPTION = (
    "Workflow of the audit. (1) Two public datasets: PhysioNet motor movement/imagery "
    "(MMI), accessed through the EEG motor movement/imagery brain-computer interface "
    "(EEGBCI) dataset, and Brain-Computer Interface (BCI) Competition IV, dataset 2a "
    "(BCI-IV-2a, abbreviated IV-2a in the diagram). EEGBCI includes 109 subjects, 64 "
    "channels, four two-class paradigms and three runs per paradigm; IV-2a includes "
    "nine subjects, 22 channels and two sessions of six runs. EEG means electroencephalography; "
    "ch means recording channels; Hz means hertz. MI = motor imagery; ME = motor execution; "
    "L/R = left versus right. (2) Preprocessing uses zero-phase finite impulse response "
    "(FIR) band-pass filtering and cue-locked 0–4 s epochs; s denotes seconds. There is no "
    "rejection for artefacts; only cues whose full 4-s epoch lies within the recording are epoched. Networks receive per-trial z-scored trials, with a global-scale "
    "matched-preprocessing control. (3) Decoders are common spatial patterns with linear "
    "discriminant analysis (CSP+LDA), tangent space with logistic regression (TS+LR), "
    "ShallowFBCSPNet (shallow filter-bank common spatial pattern network) and EEGNet "
    "(electroencephalography network). Trainable-parameter counts are reported in the "
    "decoder-settings table. OAS = oracle-approximating shrinkage. The tangent "
    "reference is the training Fréchet mean; L2 denotes the squared Euclidean coefficient "
    "penalty. (4) The protocol ladder includes within-subject leave-one-run-out or "
    "session-to-session evaluation; P0, trial-random; P0s, subject-by-class-stratified "
    "trial-random; P1, run-disjoint but subject-overlapping; P1s, session-disjoint but "
    "subject-overlapping on IV-2a; and P2, subject-disjoint: subjects permuted and cut into three groups "
    "on EEGBCI; LOSO = leave-one-subject-out, the P2 protocol on IV-2a. On EEGBCI the outer "
    "training fractions are matched at two thirds. On IV-2a the trial-random protocols "
    "are P0a (six folds) and P0b (two folds), with subject-by-class-stratified counterparts "
    "P0a-s and P0b-s; its ladder training fractions differ. Partitions and network training "
    "are repeated over prespecified seeds. The protocols specify which recording unit "
    "may occur on both sides of the train/test boundary. Ladder differences are protocol "
    "contrasts. (5) The fixed-test framework partitions subjects into target (T), donor (D) "
    "and background (B) blocks, rotated across evaluations. For EEGBCI the test set is "
    "half of the last run of T, selected by alternating trials within subject and class. "
    "B in the arm labels denotes the shared background training trials from its other "
    "runs. Arm A0 combines B with count-matched D trials from other runs; A1 combines B "
    "with count-matched T trials from other runs; A2 combines B with T trials, replacing "
    "other-run trials with the unused half of the test run. Target and donor subjects "
    "are paired by trial count, with matching within run and class. Any count shortfall "
    "in the replacement is recorded. No test trial enters any training arm. For TS+LR, "
    "one tangent reference is fitted to B and shared across arms; the arm-specific "
    "reference variant is secondary. Networks share one class-stratified validation "
    "slice selected from B and removed from every optimization set, with paired training "
    "seeds. CSP filters and classifiers are fitted separately for each arm. A1 minus A0 "
    "is the target-inclusion contrast, conditional on the background and donor sampling; "
    "A2 minus A1 is the same-run-adjacency contrast. The IV-2a analogue uses half of the "
    "last session as test, first-session background and donor trials, and unused "
    "last-session trials in A2, giving a same-session contrast. (6) Controlled analyses "
    "include an inverse-penalty sweep (C is the inverse regularization strength; larger "
    "C means weaker penalty), principal component analysis (PCA) dimension sweeps, subject "
    "identity decoding, variance decomposition, subject-mean removal and Riemannian "
    "recentering, weight alignment, target-subject in/out reference tests, network-width "
    "sweeps, a subject-count (N) by feature-dimension (d) factorial, and a calibration-size "
    "curve. (7) Outputs are per-subject accuracies and paired protocol and matched-arm "
    "contrasts. CI = confidence interval; subject-bootstrap 95 % intervals accompany paired "
    "Wilcoxon and sign-flip tests with table-wise Holm correction. The IV-2a intervals "
    "are descriptive and unadjusted. TOST = two one-sided tests for equivalence against "
    "the prespecified margin of ±0.01 in the reference-mean analyses; TOST is not used "
    "for the new matched contrasts. The diagram shows the design, with no numerical results."
)


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


def save_pptx(f, path):
    """Editable panel shapes and text, at the same physical size as the PDF."""
    from matplotlib.colors import to_hex
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR_TYPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(FIG_W_IN), Inches(FIG_H_IN)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    def rgb(color):
        return RGBColor.from_string(to_hex(color)[1:].upper())

    def xy(x, y):
        return Inches(x * FIG_W_IN / 100), Inches((100 - y) * FIG_H_IN / 100)

    # Rounded panels and circular badges precede their editable text.
    for patch in f.ax.patches:
        if isinstance(patch, FancyBboxPatch):
            x, y, w, h = patch.get_bbox().bounds
            shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                                           *xy(x, y + h), Inches(w * FIG_W_IN / 100),
                                           Inches(h * FIG_H_IN / 100))
            shape.adjustments[0] = 0.08
        elif isinstance(patch, Ellipse):
            x, y = patch.center
            w, h = patch.width, patch.height
            shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL,
                                           *xy(x - w / 2, y + h / 2),
                                           Inches(w * FIG_W_IN / 100), Inches(h * FIG_H_IN / 100))
        elif isinstance(patch, FancyArrowPatch):
            start, end = patch._posA_posB
            shape = slide.shapes.add_connector(MSO_CONNECTOR_TYPE.STRAIGHT, *xy(*start), *xy(*end))
            # Native DrawingML arrowhead keeps the workflow arrows editable.
            from pptx.oxml.xmlchemy import OxmlElement
            head = OxmlElement("a:tailEnd")
            head.set("type", "triangle")
            shape.line._get_or_add_ln().append(head)
            shape.line.color.rgb = rgb(patch.get_edgecolor())
            shape.line.width = Pt(patch.get_linewidth())
            continue
        else:
            continue
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(patch.get_facecolor())
        shape.line.color.rgb = rgb(patch.get_edgecolor())
        shape.line.width = Pt(patch.get_linewidth())

    f.fig.canvas.draw()
    renderer = f.fig.canvas.get_renderer()
    for text in f.ax.texts:
        if not text.get_text().strip():
            continue
        bb = text.get_window_extent(renderer=renderer)
        # The measured line breaks and font sizes are preserved; PowerPoint can edit each text box.
        left, top = bb.x0 / f.fig.dpi, FIG_H_IN - bb.y1 / f.fig.dpi
        width, height = bb.width / f.fig.dpi, bb.height / f.fig.dpi
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width + 0.015), Inches(height + 0.015))
        tf = box.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.TOP
        for i, line in enumerate(text.get_text().split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.alignment = PP_ALIGN.CENTER if text.get_ha() == "center" else PP_ALIGN.LEFT
            p.space_before = p.space_after = Pt(0)
            p.line_spacing = 1.28
            p.font.name = "DejaVu Sans"
            p.font.size = Pt(text.get_fontsize())
            p.font.bold = text.get_fontweight() == "bold"
            p.font.color.rgb = rgb(text.get_color())
    prs.save(str(path))


def main() -> int:
    f = Fig1(FIG_W_IN, FIG_H_IN)
    f.text(50, 98.2, "Audit workflow: subject-mixed cross-validation in EEG decoding",
           fontsize=10.5, weight="bold", color="#111111")
    f.text(50, 95.8, "datasets → preprocessing → decoders → protocols and controls → analyses → statistics",
           fontsize=8.5, color="#333333")
    x0, gap = 1.2, 1.6
    colw = (97.6 - 2 * gap) / 3
    halfw = (97.6 - gap) / 2
    # Three input panels, then the ladder beside its matched controls, then analyses and outputs.
    layout = [(x0 + c * (colw + gap), 68.0, colw, 26.5) for c in range(3)]
    layout += [(x0, 34.0, 43.0, 32.0), (45.8, 34.0, 53.0, 32.0)]
    layout += [(x0 + c * (halfw + gap), 6.3, halfw, 25.7) for c in range(2)]
    inset = 0.45
    texts, boxes = [], []
    for i, ((header, th, paras), (x, y, w, h)) in enumerate(zip(STAGES, layout)):
        width_px = (w - 2 * inset) * FIG_W_IN * f.fig.dpi / 100.0
        body = "\n".join(wrap_paragraphs(f, paras, BODY_FS, width_px))
        print(f"PANEL {i + 1}: {header}, {body.count(chr(10)) + 1} lines at {BODY_FS} pt")
        f.panel(x, y, w, h, header, theme_name=th, number=i + 1, header_fs=10.5)
        t = f.ax.text(x + w / 2, y + h - 4.4, body, ha="center", va="top", fontsize=BODY_FS,
                      color="#222222", zorder=3, linespacing=1.28)
        texts.append(t)
        boxes.append((x + inset, y + 0.5, w - 2 * inset, h - 3.9))
    # Arrows stay in the gutters, clear of both panel borders and text.
    for c in range(2):
        x = x0 + c * (colw + gap) + colw
        f.arrow(x + 0.15, 81.0, x + gap - 0.15, 81.0, style="thin")
    f.arrow(82.5, 67.6, 22.7, 66.4, style="thin")
    f.arrow(44.35, 50.0, 45.65, 50.0, style="thin")
    f.arrow(72.3, 33.6, 25.2, 32.4, style="thin")
    f.arrow(49.35, 19.0, 50.65, 19.0, style="thin")
    guide = "Ladder differences are protocol contrasts. Matched arms share the test set, background, reference and validation slice."
    guide = "\n".join(wrap_paragraphs(f, [guide], BODY_FS, 96 * FIG_W_IN * f.fig.dpi / 100))
    f.box(1.2, 0.5, 97.6, 5.3, body=guide, theme_name="blue", body_fs=BODY_FS, lw=1.2)
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
    out = OUT_DIR / "fig1_workflow_rev2.png"
    pdf = out.with_suffix(".pdf")
    f.fig.savefig(str(pdf), facecolor="white")                   # vector companion first: the glyph guard reads it
    min_glyph = F.assert_pdf_min_glyph(pdf, "fig1_workflow_rev2", delete_on_fail=False)
    f.fig.savefig(str(out), dpi=300, facecolor="white")          # exact canvas: scale 1.000 at \linewidth
    (OUT_DIR / "fig1_workflow_rev2_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    save_pptx(f, out.with_suffix(".pptx"))
    print(f"fig1_workflow_rev2: canvas {FIG_W_IN:.3f} x {FIG_H_IN:.3f} in, smallest text artist {BODY_FS:.1f} pt, "
          f"smallest rendered glyph {min_glyph:.4f} pt (scale 1.000 at placement)")
    print("PASS: box overlap, text containment, canvas bounds, text collisions and placed-width glyph checks")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
