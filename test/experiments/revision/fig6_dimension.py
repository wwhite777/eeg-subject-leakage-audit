#!/usr/bin/env python3
"""Figure 6 — dimensionality and subject count: (a) PCA sweep (E4) accuracies and Δ(P1 − P2) with per-subject
points; (b) factorial heat map of Δ(P0 − P2) over sampled-subject count N × tangent dimension d (E8, TS+LR);
(c) CSP+LDA and TS+LR (full dimension) inflation versus N with draw-level points.

Codex r002 finding F-r002-2: for N < 109 the interval in (c) is a bootstrap over the TEN SUBJECT DRAWS after
averaging the four paradigms within each matching draw (same draw id across paradigms), not over the 40
paradigm x draw cells; no interval at N = 109, where a single draw exists. N is the number of SAMPLED
subjects (P2 trains on about 2N/3 of them).
Codex r003 finding F-r003-5: the logarithmic x axes of (a) and (c) are labelled with the plain d and N
values (no 10^{x} mathtext, whose exponent printed at 5.6 pt), and fig_common.save() re-measures every
glyph of the PDF companion before the figure is written.
Usage: fig6_dimension.py --e4 <per_subject.csv> --e8 <per_draw.csv> --tag rev1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import fig_common as F  # noqa: E402

SHORT = {"trial_random": "P0 trial-random", "run_disjoint": "P1 run-disjoint", "subject_disjoint": "P2 subject-disjoint"}
# F-r003-5: the default log formatter labelled the (a) x axis 10^{1}, 10^{2}, 10^{3} as mathtext, whose
# exponent printed at 5.6 pt. Both log axes are now labelled with the plain d and N values themselves.
D_TICKS = [2, 8, 32, 128, 512, 2080]
INT = lambda v: f"{int(round(v))}"  # noqa: E731  plain integer tick label, no mathtext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", required=True)
    ap.add_argument("--e8", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    rng = np.random.default_rng(C.SEED)
    fig = plt.figure(figsize=(F.FIG_W_IN, 5.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.0])
    cap = []
    # (a) PCA sweep
    ax = fig.add_subplot(gs[0])
    ps = C.read_csv(Path(args.e4))
    for r in ps:
        r["model"] = "riemann_ts_lr"
    subj = F.seed_avg_subject(ps, ("task", "model", "protocol", "d"))
    pm = F.paradigm_mean(subj, 0)
    ds = sorted({int(k[3]) for k in pm})
    for p in ("trial_random", "run_disjoint", "subject_disjoint"):
        means, los, his = [], [], []
        for d in ds:
            v = np.array(list(pm[("paradigm_mean", "riemann_ts_lr", p, str(d))].values()))
            lo, hi = C.bootstrap_ci(v); means.append(v.mean()); los.append(lo); his.append(hi)
        ax.plot(ds, means, marker="o", ms=3, color=F.PROTO_COLOR[p], label=SHORT[p])
        ax.fill_between(ds, los, his, color=F.PROTO_COLOR[p], alpha=0.18, linewidth=0)
        cap.append(f"PCA {p}: " + ", ".join(f"d={d}: {m:.3f}" for d, m in zip(ds, means)))
    ax2 = ax.twinx()
    dm, dl, dh = [], [], []
    for d in ds:
        A, B = pm[("paradigm_mean", "riemann_ts_lr", "run_disjoint", str(d))], pm[("paradigm_mean", "riemann_ts_lr", "subject_disjoint", str(d))]
        subs = sorted(set(A) & set(B)); diff = np.array([A[s] - B[s] for s in subs])
        lo, hi = C.bootstrap_ci(diff); dm.append(diff.mean()); dl.append(lo); dh.append(hi)
        ax2.scatter(d * np.exp(rng.uniform(-0.1, 0.1, len(diff))), diff, s=3, color="#E69F00", alpha=0.12, linewidths=0, rasterized=True)
    ax2.plot(ds, dm, marker="s", ms=3, color="#E69F00", ls="-", lw=1.2, label="Δ(P1 − P2), right axis")
    ax2.fill_between(ds, dl, dh, color="#E69F00", alpha=0.25, linewidth=0)
    ax2.axhline(0, color="#E69F00", lw=0.6, ls=":"); ax2.set_ylabel("Δ(P1 − P2)\n(fraction correct)", color="#B07800"); ax2.tick_params(axis="y", colors="#B07800")
    ax2.grid(False)
    cap.append("PCA Δ(P1−P2): " + ", ".join(f"d={d}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]" for d, m, lo, hi in zip(ds, dm, dl, dh)))
    ax.set_xscale("log"); F.plain_log_ticks(ax, "x", [d for d in D_TICKS if d in ds], INT)
    ax.set_xlabel("Tangent-feature dimension after PCA (count)"); ax.set_ylabel("Accuracy\n(fraction correct)")
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--"); ax.set_title("(a) PCA-dimension sweep (TS+LR, C = 1)")
    ax.set_ylim(top=0.80)  # headroom for the legend so it never covers a curve
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, frameon=True, handlelength=1.2, handletextpad=0.5, labelspacing=0.25, borderpad=0.3, columnspacing=1.0)
    # (b) factorial heat map over sampled-subject count x dimension
    ax = fig.add_subplot(gs[1])
    e8 = C.read_csv(Path(args.e8))
    cells = defaultdict(dict)
    for r in e8:
        cells[(r["task"], r["model"], int(r["N"]), int(r["d"]), int(r["draw"]))][r["protocol"]] = float(r["mean_acc"])
    by_draw = defaultdict(lambda: defaultdict(list))  # (model, N, d) -> draw -> [per-paradigm inflation]
    for (task, model, N, d, draw), p in cells.items():
        if "trial_random" in p and "subject_disjoint" in p:
            by_draw[(model, N, d)][draw].append(p["trial_random"] - p["subject_disjoint"])

    def draw_means(key):
        """One value per subject draw: the mean over the four paradigms of that same draw (F-r002-2)."""
        d = by_draw.get(key, {})
        return np.array([float(np.mean(v)) for _, v in sorted(d.items())])

    Ns = sorted({k[1] for k in by_draw if k[0] == "riemann_ts_lr"}); Ds = sorted({k[2] for k in by_draw if k[0] == "riemann_ts_lr"})
    M = np.array([[draw_means(("riemann_ts_lr", N, d)).mean() if by_draw.get(("riemann_ts_lr", N, d)) else np.nan for d in Ds] for N in Ns])
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    for i, N in enumerate(Ns):
        for j, d in enumerate(Ds):
            if np.isnan(M[i, j]):
                ax.text(j, i, "not feasible", ha="center", va="center", fontsize=8, color="#666666")
                cap.append(f"N={N} d={d}: not feasible (fewer training trials than components)")
                continue
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="black")
            nd = len(draw_means(("riemann_ts_lr", N, d)))
            cap.append(f"N={N} d={d}: {M[i, j]:+.3f} (mean over {nd} subject draw{'s' if nd > 1 else ''} of the 4-paradigm mean)")
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels(Ds); ax.set_yticks(range(len(Ns))); ax.set_yticklabels(Ns)
    ax.set_xlabel("Tangent-feature dimension d (count)"); ax.set_ylabel("Sampled subjects N (count)"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Δ(P0 − P2) (fraction correct)", fontsize=8); cb.ax.tick_params(labelsize=8)
    ax.set_title("(b) N × d factorial (TS+LR)")
    # (c) N sweep for CSP+LDA and full-d TS+LR, with the ten subject draws as points
    ax = fig.add_subplot(gs[2])
    for m, d, lab in (("csp_lda", 6, "CSP+LDA (6 log-variance features)"), ("riemann_ts_lr", max(Ds), f"TS+LR (d = {max(Ds)})")):
        xs = sorted({k[1] for k in by_draw if k[0] == m and k[2] == d})
        means, los, his = [], [], []
        for N in xs:
            v = draw_means((m, N, d))
            means.append(float(v.mean()))
            if len(v) < 2:  # a single subject draw at N = 109 -> no draw-level interval (F-r002-2)
                los.append(np.nan); his.append(np.nan)
            else:
                lo, hi = C.bootstrap_ci(v); los.append(lo); his.append(hi)
            ax.scatter(N * np.exp(rng.uniform(-0.05, 0.05, len(v))), v, s=10, color=F.MODEL_COLOR[m], alpha=0.5, linewidths=0, rasterized=True)
        ax.plot(xs, means, marker=F.MODEL_MARK[m], ms=4, color=F.MODEL_COLOR[m], label=lab, zorder=5)
        ok = [i for i, l in enumerate(los) if l == l]
        ax.fill_between([xs[i] for i in ok], [los[i] for i in ok], [his[i] for i in ok], color=F.MODEL_COLOR[m], alpha=0.2, linewidth=0)
        cap.append(f"{lab} vs N: " + ", ".join((f"N={N}: {mm:+.3f} [{lo:+.3f},{hi:+.3f}] (10 draws)" if lo == lo else f"N={N}: {mm:+.3f} (single draw, no interval)")
                                               for N, mm, lo, hi in zip(xs, means, los, his)))
    ax.axhline(0, color="#888888", lw=0.8, ls="--"); ax.set_xscale("log"); F.plain_log_ticks(ax, "x", Ns, INT)
    ax.set_xlabel("Sampled subjects N (count; P2 trains on about 2N/3 of them)")
    ax.set_ylabel("Δ(P0 − P2)\n(fraction correct)")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.06)  # headroom so the legend never sits on the curves
    ax.legend(loc="upper right", frameon=True, handlelength=1.4, handletextpad=0.5)
    ax.set_title("(c) Δ(P0 − P2) versus sampled-subject count\n(points = subject draws, paradigm mean within each draw)")
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    caption = ("Figure 6. Feature dimension and subject count within EEGBCI. (a) TS+LR accuracy after projecting the tangent features on the first d principal "
               "components fitted on each training fold (paradigm mean; lines = mean, bands = subject-bootstrap 95 % CI; the right axis shows the per-subject Δ(P1 − P2), points = subjects). "
               "(b) Mean Δ(P0 − P2) of TS+LR for random subsets of N sampled subjects (ten draws for fewer than 109 subjects, one draw at 109) and d principal components; each cell is the mean over "
               "the draws of the four-paradigm mean within a draw. "
               "(c) Δ(P0 − P2) versus N for CSP+LDA (six log-variance features) and full-dimension TS+LR. Each point is one subject draw, averaged over the four paradigms of that same draw; "
               "line = mean over draws; band = 95 % bootstrap over the ten subject draws (paradigm mean within each draw); no band at N = 109, where a single draw exists. "
               "N counts the sampled subjects; the subject-disjoint protocol trains on about 2N/3 of them. Values: " + " | ".join(cap))
    F.save(fig, f"fig6_dimension_{args.tag}", caption, [Path(args.e4), Path(args.e8)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
