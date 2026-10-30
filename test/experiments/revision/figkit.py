#!/usr/bin/env python3
# vendored from the authors' figure toolkit; MIT
# Verbatim module follows.
"""figkit — reusable toolkit for publication-grade Figure 1 (system/architecture) diagrams.

Encodes the lab's Figure-1 design language (numbered/lettered color-coded panels, rounded
boxes with thin colored borders, bold flow arrows, inline LaTeX, exact-data mini-plots, a
bottom summary banner, and finding/recommendation callouts) as composable matplotlib
components. Coordinate system is 0..100 in x and y; place components by explicit (x,y,w,h)
so layouts match hand-designed references. Enforces the lab's figure conventions: exact-data plots,
no overlapping components (use check_overlaps), large readable fonts.

Only stdlib + matplotlib/numpy. No file exceeds 1500 lines.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Ellipse, Rectangle
import numpy as np

# ----------------------------------------------------------------------------- palette
# Each theme: edge (saturated border/header), fill (pale panel/box background), head (header text).
PALETTE = {
    "green":  {"edge": "#2e8b57", "fill": "#eaf6ef", "head": "#1f6b41"},
    "blue":   {"edge": "#2c6fb5", "fill": "#e9f1fb", "head": "#1d4f86"},
    "purple": {"edge": "#7a52c7", "fill": "#f1ecfb", "head": "#5a37a0"},
    "red":    {"edge": "#d1495b", "fill": "#fdecef", "head": "#a4293a"},
    "teal":   {"edge": "#1b9e94", "fill": "#e6f6f4", "head": "#127a72"},
    "gold":   {"edge": "#d9a300", "fill": "#fdf6e3", "head": "#9a7300"},
    "orange": {"edge": "#e07b1a", "fill": "#fdefe1", "head": "#b45c0c"},
    "pink":   {"edge": "#c74f8a", "fill": "#fbecf4", "head": "#97316a"},
    "gray":   {"edge": "#6b7280", "fill": "#f3f4f6", "head": "#374151"},
    "navy":   {"edge": "#1b2a4a", "fill": "#e8ecf4", "head": "#12203b"},
}
STAGE_ORDER = ["green", "blue", "purple", "red", "teal", "gold"]  # default stage colors


def theme(name):
    return PALETTE.get(name, PALETTE["gray"])


class Fig1:
    """A Figure-1 canvas on a 0..100 x 0..100 coordinate grid."""

    def __init__(self, width_in=16.0, height_in=9.0, dpi=200):
        self.fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.axis("off")
        self.aspect = width_in / height_in  # to keep circles round in data coords
        self._boxes = []  # (x,y,w,h,label) for overlap checking

    # ---------- primitives ----------
    def title(self, text, subtitle=None, y=97.0):
        self.ax.text(50, y, text, ha="center", va="top", fontsize=20, fontweight="bold",
                     color="#111111")
        if subtitle:
            self.ax.text(50, y - 3.2, subtitle, ha="center", va="top", fontsize=11.5,
                         color="#333333")

    def _register(self, x, y, w, h, label=""):
        self._boxes.append((x, y, w, h, label))

    def box(self, x, y, w, h, title=None, body=None, theme_name="gray", fill=None,
            edge=None, lw=1.6, radius=1.4, title_fs=11, body_fs=9.5, align="center",
            title_color=None, dashed=False, register=True):
        """Rounded rectangle with optional bold title and body text (x,y = lower-left)."""
        th = theme(theme_name)
        fc = fill if fill is not None else th["fill"]
        ec = edge if edge is not None else th["edge"]
        ls = (0, (4, 3)) if dashed else "solid"
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                           linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
                           mutation_aspect=self.aspect, zorder=2)
        self.ax.add_patch(p)
        cx = x + w / 2
        if title and body:
            self.ax.text(cx, y + h - h * 0.28, title, ha="center", va="center",
                         fontsize=title_fs, fontweight="bold",
                         color=title_color or th["head"], zorder=3)
            self.ax.text(cx, y + h * 0.36, body, ha="center", va="center",
                         fontsize=body_fs, color="#222222", zorder=3, wrap=True)
        elif title:
            self.ax.text(cx, y + h / 2, title, ha="center", va="center",
                         fontsize=title_fs, fontweight="bold",
                         color=title_color or th["head"], zorder=3)
        elif body:
            ha = {"center": "center", "left": "left"}[align]
            tx = cx if align == "center" else x + w * 0.06
            self.ax.text(tx, y + h / 2, body, ha=ha, va="center", fontsize=body_fs,
                         color="#222222", zorder=3)
        if register:
            self._register(x, y, w, h, title or "")
        return (x, y, w, h)

    def panel(self, x, y, w, h, header, theme_name="blue", number=None, letter=None,
              header_fs=13, fill=None):
        """Color-coded panel: pale rounded rect + bold colored header with optional badge."""
        th = theme(theme_name)
        self.box(x, y, w, h, theme_name=theme_name, fill=fill if fill is not None else "#ffffff",
                 edge=th["edge"], lw=1.8, radius=1.6, register=True)
        hx = x + 2.0
        hy = y + h - 2.4
        if number is not None:
            self.badge(hx + 1.4, hy, number, theme_name, r=1.7)
            hx += 4.0
        elif letter is not None:
            self.badge(hx + 1.4, hy, letter, theme_name, r=1.7, text=letter)
            hx += 4.0
        self.ax.text(hx, hy, header, ha="left", va="center", fontsize=header_fs,
                     fontweight="bold", color=th["head"], zorder=4)
        return (x, y, w, h)

    def badge(self, cx, cy, n, theme_name="blue", r=1.7, text=None):
        """Filled circle badge with a white number/letter."""
        th = theme(theme_name)
        c = Circle((cx, cy), r, facecolor=th["edge"], edgecolor="white", linewidth=1.2,
                   zorder=5)
        c.set_transform(self.ax.transData)
        # keep visually round despite non-square aspect: draw as ellipse in data coords
        e = Ellipse((cx, cy), width=2 * r, height=2 * r * self.aspect, facecolor=th["edge"],
                    edgecolor="white", linewidth=1.2, zorder=5)
        self.ax.add_patch(e)
        self.ax.text(cx, cy, str(text if text is not None else n), ha="center", va="center",
                     color="white", fontsize=11, fontweight="bold", zorder=6)

    def arrow(self, x0, y0, x1, y1, style="main", color=None):
        """Flow arrow. style: main (thick black), thin (gray), dashed."""
        styles = {
            "main": dict(lw=3.2, color=color or "#111111", ls="solid",
                         arrowstyle="-|>", mutation_scale=26),
            "thin": dict(lw=1.4, color=color or "#444444", ls="solid",
                         arrowstyle="-|>", mutation_scale=14),
            "dashed": dict(lw=1.4, color=color or "#666666", ls=(0, (4, 3)),
                           arrowstyle="-|>", mutation_scale=14),
        }
        s = styles[style]
        a = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0",
                            linewidth=s["lw"], color=s["color"], linestyle=s["ls"],
                            arrowstyle=s["arrowstyle"], mutation_scale=s["mutation_scale"],
                            zorder=4)
        self.ax.add_patch(a)

    def mathlabel(self, x, y, latex, fontsize=12, ha="center", va="center", color="#111111"):
        """Render a LaTeX expression via matplotlib mathtext, e.g. r'$x_\\theta=\\cos\\theta x$'."""
        self.ax.text(x, y, latex, ha=ha, va=va, fontsize=fontsize, color=color, zorder=3)

    def text(self, x, y, s, fontsize=9.5, ha="center", va="center", color="#222222",
             weight="normal", style="normal"):
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, color=color,
                     fontweight=weight, fontstyle=style, zorder=3)

    def callout(self, x, y, w, h, title, body, kind="finding", icon=None):
        """Finding (gold) / recommendation (blue) callout box with icon + bold title."""
        tmap = {"finding": "gold", "recommendation": "blue", "warning": "red"}
        th = theme(tmap.get(kind, "gold"))
        default_icon = {"finding": "★", "recommendation": "✓", "warning": "!"}[kind]
        ic = icon or default_icon
        self.box(x, y, w, h, theme_name=tmap.get(kind, "gold"), lw=2.0, radius=1.6)
        self.ax.text(x + 1.8, y + h - 2.4, ic, ha="center", va="center", fontsize=14,
                     color=th["edge"], zorder=4, fontweight="bold")
        self.ax.text(x + 3.4, y + h - 2.4, title, ha="left", va="center", fontsize=11.5,
                     fontweight="bold", color=th["head"], zorder=4)
        self.ax.text(x + w / 2, y + (h - 4) / 2, body, ha="center", va="center",
                     fontsize=9.3, color="#222222", zorder=4, wrap=True)

    def banner(self, items, y=3.5, h=6.0, x0=4.0, x1=96.0, themes=None):
        """Bottom summary strip: items = [(label, sublabel), ...] with arrows between."""
        n = len(items)
        gap = 3.0
        total = x1 - x0
        bw = (total - (n - 1) * gap) / n
        themes = themes or STAGE_ORDER
        for i, (label, sub) in enumerate(items):
            bx = x0 + i * (bw + gap)
            th = theme(themes[i % len(themes)])
            self.box(bx, y, bw, h, theme_name=themes[i % len(themes)], lw=1.6, radius=1.4)
            self.ax.text(bx + bw / 2, y + h - 1.9, label, ha="center", va="center",
                         fontsize=11, fontweight="bold", color=th["head"], zorder=3)
            self.ax.text(bx + bw / 2, y + h * 0.35, sub, ha="center", va="center",
                         fontsize=8.6, color="#333333", zorder=3, wrap=True)
            if i < n - 1:
                self.arrow(bx + bw + 0.3, y + h / 2, bx + bw + gap - 0.3, y + h / 2,
                           style="main", color=th["edge"])

    def miniplot(self, x, y, w, h, plotfn, title=None):
        """Inset axes for an EXACT-DATA plot. plotfn(ax) draws it. (x,y = lower-left, 0..100.)"""
        ax = self.fig.add_axes([x / 100.0, y / 100.0, w / 100.0, h / 100.0])
        plotfn(ax)
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold")
        self._register(x, y, w, h, "miniplot")
        return ax

    # ---------- simple vector icons (best-effort; extend as needed) ----------
    def icon(self, cx, cy, name, size=2.2, color="#333333"):
        s = size
        a = self.aspect
        if name == "database":
            for k, yy in enumerate([cy + s * a, cy, cy - s * a]):
                self.ax.add_patch(Ellipse((cx, yy), 2 * s, s * a, facecolor="none",
                                          edgecolor=color, lw=1.4, zorder=3))
            self.ax.add_patch(Rectangle((cx - s, cy - s * a), 2 * s, 2 * s * a, facecolor="none",
                                        edgecolor="none", zorder=3))
        elif name == "waves":
            xs = np.linspace(cx - s, cx + s, 60)
            ys = cy + (s * 0.5 * a) * np.sin((xs - cx) / s * 3.14159 * 2)
            self.ax.plot(xs, ys, color=color, lw=1.4, zorder=3)
        elif name == "network":
            pts = [(cx, cy + s * a), (cx - s, cy - s * a), (cx + s, cy - s * a)]
            for p in pts:
                self.ax.plot([cx, p[0]], [cy, p[1]], color=color, lw=1.1, zorder=3)
            for p in pts + [(cx, cy)]:
                self.ax.add_patch(Ellipse(p, 0.7, 0.7 * a, facecolor=color, edgecolor="none",
                                          zorder=4))
        elif name == "chip":
            self.ax.add_patch(Rectangle((cx - s, cy - s * a), 2 * s, 2 * s * a, facecolor="none",
                                        edgecolor=color, lw=1.4, zorder=3))
            for t in np.linspace(-s * 0.6, s * 0.6, 3):
                self.ax.plot([cx + t, cx + t], [cy + s * a, cy + s * a + 0.6 * a], color=color, lw=1.1)
                self.ax.plot([cx + t, cx + t], [cy - s * a, cy - s * a - 0.6 * a], color=color, lw=1.1)
        elif name == "check":
            self.ax.text(cx, cy, "✓", ha="center", va="center", color=color,
                         fontsize=size * 6, fontweight="bold", zorder=4)
        else:  # fallback: labeled dot
            self.ax.add_patch(Ellipse((cx, cy), s, s * a, facecolor=color, edgecolor="none",
                                      zorder=3))

    # ---------- QA ----------
    def check_overlaps(self, tol=0.2):
        """Return list of overlapping registered box pairs (the lab's figure conventions: no overlaps)."""
        bad = []
        b = self._boxes
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                x1, y1, w1, h1, l1 = b[i]
                x2, y2, w2, h2, l2 = b[j]
                ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                # ignore nesting (one fully inside another = intended, e.g. box in panel)
                inside = (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2) or \
                         (x2 >= x1 and y2 >= y1 and x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1)
                if ox > tol and oy > tol and not inside:
                    bad.append((l1, l2, round(ox, 1), round(oy, 1)))
        return bad

    def save(self, path, dpi=None):
        self.fig.savefig(path, dpi=dpi or self.fig.dpi, bbox_inches="tight",
                         facecolor="white")
        plt.close(self.fig)
        return path
