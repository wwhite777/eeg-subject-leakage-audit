#!/usr/bin/env python3
"""Figure 6 — dimensionality and subject count: (a) PCA sweep (E4) accuracies and Δ(P1 − P2) with per-subject
points; (b) factorial heat map of Δ(P0 − P2) over training-subject count N × tangent dimension d (E8, TS+LR);
(c) CSP+LDA and TS+LR (full dimension) inflation versus N with draw-level points.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", required=True)
    ap.add_argument("--e8", required=True)
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    plt = F.setup_mpl()
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    cap = []
    # (a) PCA sweep
    ax = axes[0]
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
        ax.plot(ds, means, marker="o", ms=3, color=F.PROTO_COLOR[p], label=F.PROTO_LABEL[p])
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
    ax2.axhline(0, color="#E69F00", lw=0.6, ls=":"); ax2.set_ylabel("Δ(P1 − P2) (fraction correct)", color="#B07800"); ax2.tick_params(axis="y", colors="#B07800")
    ax2.grid(False)
    cap.append("PCA Δ(P1−P2): " + ", ".join(f"d={d}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]" for d, m, lo, hi in zip(ds, dm, dl, dh)))
    ax.set_xscale("log"); ax.set_xlabel("Tangent-feature dimension after PCA (count)"); ax.set_ylabel("Accuracy (fraction correct)")
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--"); ax.set_title("(a) PCA-dimension sweep (TS+LR, C = 1)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fontsize=6)
    # (b) factorial heat map
    ax = axes[1]
    e8 = C.read_csv(Path(args.e8))
    cells = defaultdict(dict)
    for r in e8:
        cells[(r["task"], r["model"], int(r["N"]), int(r["d"]), int(r["draw"]))][r["protocol"]] = float(r["mean_acc"])
    infl = defaultdict(list)
    for (task, model, N, d, draw), p in cells.items():
        if "trial_random" in p and "subject_disjoint" in p:
            infl[(model, N, d)].append(p["trial_random"] - p["subject_disjoint"])  # pooled over tasks and draws
    Ns = sorted({k[1] for k in infl if k[0] == "riemann_ts_lr"}); Ds = sorted({k[2] for k in infl if k[0] == "riemann_ts_lr"})
    M = np.array([[np.mean(infl[("riemann_ts_lr", N, d)]) if infl.get(("riemann_ts_lr", N, d)) else np.nan for d in Ds] for N in Ns])
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    for i, N in enumerate(Ns):
        for j, d in enumerate(Ds):
            if np.isnan(M[i, j]):
                ax.text(j, i, "not\nfeasible", ha="center", va="center", fontsize=5, color="#666666")
                cap.append(f"N={N} d={d}: not feasible (fewer training trials than components)")
                continue
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=5.5, color="black")
            cap.append(f"N={N} d={d}: {M[i, j]:+.3f} (n={len(infl[('riemann_ts_lr', N, d)])} task×draw cells)")
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels(Ds); ax.set_yticks(range(len(Ns))); ax.set_yticklabels(Ns)
    ax.set_xlabel("Tangent-feature dimension d (count)"); ax.set_ylabel("Training-subject count N (count)"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Δ(P0 − P2) (fraction correct)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title("(b) N × d factorial (TS+LR)")
    # (c) N sweep for CSP+LDA and full-d TS+LR with draw-level points
    ax = axes[2]
    for m, d, lab in (("csp_lda", 6, "CSP+LDA"), ("riemann_ts_lr", max(Ds), f"TS+LR (d = {max(Ds)})")):
        xs = sorted({k[1] for k in infl if k[0] == m and k[2] == d})
        means, los, his = [], [], []
        for N in xs:
            v = np.array(infl[(m, N, d)]); lo, hi = C.bootstrap_ci(v); means.append(v.mean()); los.append(lo); his.append(hi)
            ax.scatter(N * np.exp(rng.uniform(-0.06, 0.06, len(v))), v, s=5, color=F.MODEL_COLOR[m], alpha=0.3, linewidths=0, rasterized=True)
        ax.plot(xs, means, marker=F.MODEL_MARK[m], ms=4, color=F.MODEL_COLOR[m], label=lab, zorder=5)
        ax.fill_between(xs, los, his, color=F.MODEL_COLOR[m], alpha=0.2, linewidth=0)
        cap.append(f"{lab} vs N: " + ", ".join(f"N={N}: {mm:+.3f} [{lo:+.3f},{hi:+.3f}]" for N, mm, lo, hi in zip(xs, means, los, his)))
    ax.axhline(0, color="#888888", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("Training-subject count N (count)"); ax.set_ylabel("Δ(P0 − P2) (fraction correct)"); ax.legend(loc="upper right", frameon=True, fontsize=6.5)
    ax.set_title("(c) Δ(P0 − P2) vs N (points = paradigm × draw)")
    fig.tight_layout(w_pad=1.0)
    caption = ("Figure 6. Feature dimension and subject count within EEGBCI. (a) TS+LR accuracy after projecting the tangent features on the first d principal "
               "components fitted on each training fold (paradigm mean; lines = mean, bands = subject-bootstrap 95 % CI; the right axis shows the per-subject Δ(P1 − P2), points = subjects). "
               "(b) Mean Δ(P0 − P2) of TS+LR for random subsets of N training subjects (10 draws for N < 109) and d principal components, pooled over the four paradigms. "
               "(c) Δ(P0 − P2) versus N for CSP+LDA (six log-variance features) and full-dimension TS+LR (points = paradigm × draw cells; line = mean; band = 95 % CI). Values: " + " | ".join(cap))
    F.save(fig, f"fig6_dimension_{args.tag}", caption, [Path(args.e4), Path(args.e8)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
