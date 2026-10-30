#!/usr/bin/env python3
"""Shared plotting helpers for the revision figures (exact data from results/revision CSVs only).

Design rules (dataviz + research-figures skills, EIC request): every panel shows the individual
subjects, labelled axes with units, a legend for >= 2 series, colourblind-safe fixed hue order
(validated with the dataviz palette validator on 2026-09-16), fonts designed for the printed width
(IOP text width 153 mm), and a caption written from the same CSVs. Markers never encode significance.

Print-size rule (IOP: "aim for text sizes of 8 to 12 pt at the final figure size", R1 minor 7, D-11):
every figure is drawn at FIG_W_IN = 6.024 in, the measured text width of the compiled manuscript
(433.70 pt on main_clean.pdf), and saved at exactly that canvas size (no bbox_inches="tight"), so the
placement scale factor is 1.000 and the generated point sizes are the printed point sizes. No text is
below 8 pt; MIN_PT below is asserted over every text artist before the file is written.

Review correction r003 F-r003-5: Text.get_fontsize() reports the PARENT size of a mathtext expression, so a
$\\ell_2$ subscript or a 10^{-4} log-tick exponent passed that check while rendering at 5.6-6.0 pt. The
vector companion is therefore written FIRST and re-measured with pymupdf (assert_pdf_min_glyph); a figure
whose PDF carries any text span below MIN_PT is deleted, not published. plain_log_ticks() replaces the
mathtext power-of-ten tick labels with plain strings drawn at the full tick font size.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

FIG_DIR = C.ROOT / "result" / "figure" / "revision"

# \linewidth of manuscript/v2 measured with fitz on main_clean.pdf: 433.70 pt = 6.024 in.
FIG_W_IN = 6.024
MIN_PT = 8.0  # IOP minimum text size at the final figure size
# Width the figures actually occupy in main_clean.pdf (fitz, r003 audit): 433.718994 pt, i.e. the PDF
# companion (6.024 in = 433.728 pt) is placed at scale 0.99998, so native pt == printed pt to 2e-4 pt.
PLACED_W_PT = 433.718994

# fixed hue order (Okabe–Ito), validated: protocols and models never share a slot within one figure
PROTO_COLOR = {"within_subject": "#009E73", "within_session": "#009E73", "trial_random": "#D55E00", "trial_random_6": "#D55E00",
               "trial_random_2": "#CC79A7", "run_disjoint": "#E69F00", "session_disjoint": "#56B4E9", "subject_disjoint": "#0072B2",
               "loso": "#0072B2", "subject_disjoint_5": "#999999"}
PROTO_LABEL = {"within_subject": "within-subject", "within_session": "within-subject (session→session)", "trial_random": "trial-random (P0)",
               "trial_random_6": "trial-random 6-fold (P0a)", "trial_random_2": "trial-random 2-fold (P0b)", "run_disjoint": "run-disjoint (P1)",
               "session_disjoint": "session-disjoint (P1s)", "subject_disjoint": "subject-disjoint (P2)", "loso": "subject-disjoint LOSO (P2)",
               "subject_disjoint_5": "subject-disjoint 5-fold (original)"}
MODEL_COLOR = {"csp_lda": "#CC79A7", "riemann_ts_lr": "#0072B2", "shallow_fbcsp": "#E69F00", "eegnet_v4": "#009E73"}
MODEL_LABEL = {"csp_lda": "CSP+LDA", "riemann_ts_lr": "TS+LR", "shallow_fbcsp": "ShallowFBCSPNet", "eegnet_v4": "EEGNet"}
MODEL_MARK = {"csp_lda": "s", "riemann_ts_lr": "o", "shallow_fbcsp": "^", "eegnet_v4": "D"}


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "savefig.dpi": 300, "figure.dpi": 100,
    })
    return plt


def min_font_pt(fig) -> float:
    """Smallest font size (pt) of any non-empty text artist on the figure."""
    sizes = []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize") and hasattr(o, "get_text")):
        if str(t.get_text()).strip() and t.get_visible():
            sizes.append(float(t.get_fontsize()))
    return min(sizes) if sizes else float("nan")


def pdf_glyph_spans(pdf_path):
    """(page width in pt, [(size_pt, text, font)]) for every non-blank text span of a saved figure PDF.

    Measured with pymupdf on the vector companion, so these are the sizes the glyphs are ACTUALLY drawn
    at — including mathtext sub/superscripts, which Text.get_fontsize() reports at the parent size.
    """
    import pymupdf  # noqa: PLC0415  (fitz; only needed when a figure is written)

    doc = pymupdf.open(str(pdf_path))
    page_w = float(doc[0].rect.width)
    spans = []
    for page in doc:
        for block in page.get_text("rawdict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = "".join(ch["c"] for ch in span.get("chars", []))
                    if txt.strip():
                        spans.append((float(span["size"]), txt, str(span.get("font", ""))))
    doc.close()
    return page_w, spans


def assert_pdf_min_glyph(pdf_path, name, min_pt=MIN_PT, delete_on_fail=True):
    """Refuse a figure whose PDF companion carries a rendered glyph below min_pt at the placed width.

    F-r003-5: this is the check that catches mathtext scaling. The figure is saved at FIG_W_IN, so the
    page width must equal the placed width; the printed size of a span is size * PLACED_W_PT / page_w.
    Returns the smallest printed size. Raises SystemExit (and removes the file) otherwise; also raises if
    the PDF holds no text at all, so "verified nothing" cannot look like "nothing to verify".
    """
    pdf_path = Path(pdf_path)
    page_w, spans = pdf_glyph_spans(pdf_path)

    def fail(msg):
        if delete_on_fail:
            pdf_path.unlink(missing_ok=True)
        raise SystemExit(f"{name}: {msg}")

    if abs(page_w - FIG_W_IN * 72.0) > 0.5:
        fail(f"pdf page width {page_w:.2f} pt != {FIG_W_IN * 72.0:.2f} pt (printed sizes would not be 1:1)")
    if not spans:
        fail(f"no text span found in {pdf_path.name} — the glyph guard would verify nothing")
    scale = PLACED_W_PT / page_w
    small = sorted({(round(s * scale, 4), t) for s, t, _ in spans if s * scale < min_pt - 1e-3})
    if small:
        for pt, t in small[:10]:
            print(f"  GLYPH BELOW {min_pt} PT: {pt} pt {t!r}")
        fail(f"{len(small)} rendered glyph size(s) below {min_pt} pt in the PDF companion (mathtext?)")
    return min(s * scale for s, _, _ in spans)


def plain_log_ticks(ax, which, values, fmt=None):
    """Label a log axis with plain strings instead of 10^{x} mathtext (whose exponent renders at 5.6 pt).

    `values` are the data positions to label; minor ticks keep their marks but lose their labels, so the
    smallest glyph on the axis is the tick font size itself (F-r003-5).
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    fmt = fmt or (lambda v: f"{v:g}")
    axis = ax.xaxis if which == "x" else ax.yaxis
    vals = list(values)
    axis.set_major_locator(FixedLocator(vals))
    axis.set_major_formatter(FixedFormatter([fmt(v) for v in vals]))
    axis.set_minor_formatter(NullFormatter())


def wrap_to_width(fig, text, fontsize, frac=0.97):
    """Re-wrap a figure-level explanatory string so every rendered line fits inside the canvas.

    Measured with the real renderer (not a character-count guess), so it is exact for the font in use.
    Input newlines are kept as hard paragraph breaks.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    target = frac * fig.get_size_inches()[0] * fig.dpi

    def width_px(s: str) -> float:
        t = fig.text(0, 0, s, fontsize=fontsize)
        w = t.get_window_extent(renderer=rend).width
        t.remove()
        return w

    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = (line + " " + word).strip()
            if line and width_px(cand) > target:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return "\n".join(out)


def _visible_texts(fig):
    """[(label, bbox_px)] for every text the reader actually sees.

    Tick labels belonging to hidden ticks (matplotlib keeps a pool of Tick objects and hides the
    unused ones) are excluded, otherwise they produce phantom collisions off the axes.
    """
    from matplotlib.text import Text

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tick.get_loc()
                off_view = loc is None or not (lo - eps <= loc <= hi + eps)
                for lab, on in ((tick.label1, tick.label1.get_visible()), (tick.label2, tick.label2.get_visible())):
                    if off_view or not tick.get_visible() or not on:
                        hidden.add(id(lab))
    items, seen = [], set()
    for t in fig.findobj(match=lambda o: isinstance(o, Text)):  # offsetbox TextArea wrappers excluded
        if id(t) in seen or id(t) in hidden:
            continue
        seen.add(id(t))
        if not str(t.get_text()).strip() or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(renderer=rend)
        except Exception:  # noqa: BLE001
            continue
        items.append((str(t.get_text()).replace("\n", " / ")[:45], bb))
    return items


def text_outside_canvas(fig, tol_px=1.0):
    """Text that runs off the printed canvas (the figure is saved at its exact size, so this is clipped)."""
    w, h = fig.get_size_inches() * fig.dpi
    out = []
    for label, bb in _visible_texts(fig):
        over = max(-bb.x0, bb.x1 - w, -bb.y0, bb.y1 - h)
        if over > tol_px:
            out.append((label, round(over, 1)))
    return out


def text_collisions(fig, tol_px=1.0):
    """Pairs of visible text artists whose rendered boxes overlap (D-08/D-09 style collisions).

    Uses the real renderer, so it catches tick labels, titles, in-axes annotations and legend text
    running into each other. Returns [(text_a, text_b, overlap_x_px, overlap_y_px), ...].
    """
    items = _visible_texts(fig)
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > tol_px and oy > tol_px:
                bad.append((items[i][0], items[j][0], round(ox, 1), round(oy, 1)))
    return bad


def seed_avg_subject(rows, keys, subject_key="subject", acc_key="accuracy"):
    """{key_tuple: {subject: mean over seeds/draws}}"""
    tmp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = tuple(r[x] for x in keys)
        tmp[k][int(r[subject_key])].append(float(r[acc_key]))
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in tmp.items()}


def paradigm_mean(subj_by_key, key_index_task, tasks=C.TASKS):
    """Average per subject over the four EEGBCI paradigms (key element key_index_task is the task)."""
    agg = defaultdict(lambda: defaultdict(list))
    for k, d in subj_by_key.items():
        if k[key_index_task] not in tasks:
            continue
        nk = tuple("paradigm_mean" if i == key_index_task else x for i, x in enumerate(k))
        for s, v in d.items():
            agg[nk][s].append(v)
    return {k: {s: float(np.mean(v)) for s, v in d.items() if len(v) == len(tasks)} for k, d in agg.items()}


def strip_points(ax, x, values, color, rng, width=0.16, size=7, alpha=0.45, zorder=2, marker="o"):
    values = np.asarray(values, dtype=float)
    jitter = rng.uniform(-width, width, size=len(values))
    ax.scatter(x + jitter, values, s=size, color=color, alpha=alpha, linewidths=0, zorder=zorder, marker=marker, rasterized=True)


def mean_ci(ax, x, values, color, seed=C.SEED, width=0.28, lw=1.6, zorder=4, marker="_"):
    values = np.asarray(values, dtype=float)
    m = values.mean()
    lo, hi = C.bootstrap_ci(values, seed=seed)
    ax.plot([x - width, x + width], [m, m], color="black", lw=lw, zorder=zorder + 1, solid_capstyle="butt")
    ax.plot([x, x], [lo, hi], color="black", lw=1.0, zorder=zorder)
    return m, lo, hi


def save(fig, name: str, caption: str, data_files: list[Path]):
    """PNG (300 dpi) + PDF + caption .txt naming the data files; also a large-font pptx with the PNG.

    Saved at the exact canvas size (no tight bbox) so that the placed scale is 1.000; refuses to write
    a figure whose canvas is not the manuscript line width, that carries a text artist below MIN_PT, or
    whose PDF companion contains a glyph rendered below MIN_PT (mathtext sub/superscripts, F-r003-5).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    if abs(w_in - FIG_W_IN) > 0.01:
        raise SystemExit(f"{name}: canvas width {w_in:.3f} in != placed width {FIG_W_IN} in (scale would not be 1.0)")
    smallest = min_font_pt(fig)
    if smallest < MIN_PT - 1e-6:
        raise SystemExit(f"{name}: smallest text {smallest} pt < {MIN_PT} pt at print size")
    clashes = text_collisions(fig)
    for c in clashes:
        print("  TEXT COLLISION:", c)
    clipped = text_outside_canvas(fig)
    for c in clipped:
        print("  TEXT OFF CANVAS:", c)
    if clashes or clipped:
        raise SystemExit(f"{name}: {len(clashes)} overlapping text pairs, {len(clipped)} texts off the canvas")
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(pdf)  # vector companion first: it is what the rendered-glyph guard measures
    min_glyph = assert_pdf_min_glyph(pdf, name)  # deletes the pdf and stops if any glyph prints < MIN_PT
    fig.savefig(png)
    print(f"{name}: canvas {w_in:.3f} x {h_in:.3f} in, smallest text artist {smallest:.1f} pt, "
          f"smallest rendered glyph {min_glyph:.4f} pt (scale 1.000 at placement)")
    def rel(p: Path) -> str:
        p = Path(p).resolve()
        return str(p.relative_to(C.ROOT)) if str(p).startswith(str(C.ROOT)) else str(p)

    (FIG_DIR / f"{name}_caption.txt").write_text(caption + "\n\nData: " + ", ".join(rel(p) for p in data_files) + "\n", encoding="utf-8")
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt

        prs = Presentation()
        prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.7))
        tb.text_frame.text = name.replace("_", " ")
        tb.text_frame.paragraphs[0].runs[0].font.size = Pt(28)
        slide.shapes.add_picture(str(png), Inches(0.6), Inches(0.9), height=Inches(6.3))
        prs.save(str(FIG_DIR / f"{name}.pptx"))
    except Exception as exc:  # noqa: BLE001
        print("pptx skipped:", exc)
    print("saved", png)
    return png
