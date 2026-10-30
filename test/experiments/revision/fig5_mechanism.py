#!/usr/bin/env python3
"""Figure 5 — subject information in tangent features (E3): (a) variance decomposition, (b) subject-ID
decoding, (c) Δ(P1 − P2) before/after subject-mean removal and Riemannian recentering (per-subject points),
(d) alignment of weight vectors with subject-specific class directions (per fold).
Usage: fig5_mechanism.py --e3a <csv> --e3b <csv> --e3c <per_subject.csv> --e3d <csv> --e1 <e1 eegbci per_subject.csv> --tag rev1
Revision 2: replace --e3b with --e3b2 <csv>, add --e3a2 <csv>, retain --e3a
for the original identity values, and use --tag rev2.
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
from matplotlib.lines import Line2D  # noqa: E402


def panels_v2(axes, args, files, cap):
    """B18/B19 replacement for panels (a,b); panels (c,d) retain their data and code."""
    import e3v2_subject_info as E

    b = C.read_csv(Path(args.e3b2))
    a = C.read_csv(Path(args.e3a2))
    per_subject_path = Path(args.e3a2).with_name(Path(args.e3a2).stem + "_per_subject.csv")
    summaries = E.summarize_identity(C.read_csv(per_subject_path))
    files.extend([Path(args.e3a2), per_subject_path, Path(args.e3b2)])
    tasks = sorted({r["task"] for r in a}, key=C.TASKS.index)
    if {r["task"] for r in b} != set(tasks):
        raise ValueError("E3a-v2 and E3b-v2 must contain the same paradigms")
    ax = axes[0, 0]
    terms = [("subject", "subject", "#0072B2"),
             ("class_given_subject", "class | subject", "#D55E00"),
             ("subject_x_class_given_both", "subject × class | both", "#E69F00"),
             ("residual", "residual", "#BBBBBB")]
    for i, task in enumerate(tasks):
        bottom = 0.0
        for name, label, color in terms:
            matches = [r for r in b if r["task"] == task and r["order"] == "A" and r["component"] == name]
            if len(matches) != 1:
                raise ValueError(f"missing/duplicate order-A term: {task}/{name}")
            value = float(matches[0]["fraction"])
            ax.bar(i, value, bottom=bottom, color=color, width=0.7, linewidth=0, label=label if i == 0 else None)
            if value > 0.10:
                ax.text(i, bottom + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if name == "residual" else "white")
            bottom += value
            cap.append(f"{C.TASK_SHORT[task]} order A {name}: {value:.6g}")
        if not np.isclose(bottom, 1, rtol=E.REL_TOL):
            raise ValueError("order-A fractions do not sum to one")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([C.TASK_SHORT[t].replace(" ", "\n").replace("fists/feet", "fists/\nfeet") for t in tasks], fontsize=8)
    ax.set_ylabel("Fraction of total sum\nof squares (dimensionless)")
    ax.set_ylim(0, 1.65)
    ax.legend(loc="upper center", ncol=1, frameon=False, handlelength=1.2, handletextpad=0.5)
    ax.set_title("(a) sequential sums of squares\norder A: subject → class → interaction")

    ax = axes[0, 1]
    schemes = ["none", "independent_reference", "training_run", "split_reference", "all_trial"]
    labels = ["(i)", "(ii)", "(iii)", "(v)", "(iv)"]
    old = C.read_csv(Path(args.e3a)) if args.e3a else []
    if old:
        labels += ["rev1\nP0", "rev1\nP1"]
        files.append(Path(args.e3a))
    lookup = {(r["task"], r["scheme"]): r for r in summaries}
    width = 0.8 / len(tasks)
    colors = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7"]
    chances = []
    for ti, task in enumerate(tasks):
        values, lows, highs = [], [], []
        cap.append(f"subject-ID {C.TASK_SHORT[task]}: {lookup[(task, 'none')]['n_subjects']} subjects, "
                   f"chance {lookup[(task, 'none')]['chance']:.6g}")
        for scheme in schemes:
            r = lookup[(task, scheme)]
            values.append(r["mean"])
            lows.append(r["assignment_min"])
            highs.append(r["assignment_max"])
            cap.append(f"subject-ID {C.TASK_SHORT[task]} {E.SCHEME_LABELS[scheme]}: macro {r['mean']:.6g} "
                       f"95% subject-bootstrap CI [{r['ci_low']:.6g}, {r['ci_high']:.6g}], "
                       f"assignment range [{r['assignment_min']:.6g}, {r['assignment_max']:.6g}], "
                       f"pooled {r['pooled_trial_accuracy']:.6g}")
        chances.append(lookup[(task, "none")]["chance"])
        diff = lookup[(task, "independent_reference_minus_split_reference")]
        cap.append(f"{C.TASK_SHORT[task]} (ii)−(v): {diff['mean']:.6g} "
                   f"95% paired subject-bootstrap CI [{diff['ci_low']:.6g}, {diff['ci_high']:.6g}]")
        if old:
            for split in ("trial_random", "run_disjoint"):
                rr = [r for r in old if r["task"] == task and r["split"] == split and r["centered"] == "True"]
                if len(rr) != 1 or int(rr[0]["n_subjects"]) != lookup[(task, "none")]["n_subjects"]:
                    raise ValueError("revision-1 identity values require matching paradigms and subjects")
                v = float(rr[0]["subject_id_accuracy"])
                values.append(v); lows.append(v); highs.append(v)
            for r in old:
                if r["task"] == task:
                    cap.append(f"revision-1 {C.TASK_SHORT[task]} {r['split']} centred={r['centered']}: {r['subject_id_accuracy']}")
        positions = np.arange(len(labels)) + (ti - (len(tasks) - 1) / 2) * width
        ax.bar(positions, values, width=width * 0.95, color=colors[ti], linewidth=0, label=C.TASK_SHORT[task])
        ax.errorbar(positions, values, yerr=[np.maximum(np.array(values) - lows, 0),
                                           np.maximum(np.array(highs) - values, 0)],
                    fmt="none", color="black", lw=0.7, capsize=1.2)
        # Zero-valued bars retain a visible plotted mark; the exact values are in the caption.
        ax.scatter(positions, values, marker="_", s=8, color=colors[ti], zorder=4)
    if not np.allclose(chances, chances[0]):
        raise ValueError("paradigms must have the same identity chance")
    ax.axhline(chances[0], color="#888888", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Centring scheme / revision-1 probe")
    ax.set_ylabel("Subject-identity accuracy\n(fraction correct)")
    ax.set_ylim(0, 1.45)
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#888888", lw=0.8, ls="--"))
    names.append(f"chance ({chances[0]:.3f})")
    ax.legend(handles, names, loc="upper center", ncol=2, frameon=False,
              columnspacing=0.7, handlelength=1.0, handletextpad=0.4)
    ax.set_title("(b) decoding subject identity\ncentring controls")
    return ("(a) Order A (subject → class → interaction) sequential fractions of total tangent-feature sum of squares: "
            "subject = RSS0−RSS_S; class given subject = RSS_S−RSS_SC; subject × class given both = RSS_SC−RSS_cell; "
            "residual = RSS_cell; each divided by RSS0, summed over dimensions. Fits are ordinary trial-weighted "
            "least squares with intercepts, in a common tangent space (reference = Fréchet mean of all trials; descriptive). "
            "These sequential projection increments are not uniquely separated variance sources. "
            "(b) Multinomial logistic identity-probe accuracy (C = 1): (i) no centring, (ii) independent-reference centring, "
            "(iii) training-run centring, (v) split-reference centring with alternating reference-run halves, and "
            "(iv) all-trial centring, the revision-1 construction applied here across all three runs. "
            "Each of the six ordered (reference, training, test) run assignments uses a tangent reference fitted only on "
            "training-run covariances; reference-run features therefore depend on the training covariances through this mean. "
            "Bars average accuracy over assignments within subject, then over subjects; whiskers show the min–max of "
            "per-assignment macro accuracies, not confidence intervals. Subject-bootstrap 95% CIs (2000 resamples; seed 20260916), "
            "pooled-trial accuracy and paired (ii)−(v) are given below. The difference (ii)−(v) estimates the identity signal "
            "that a shared estimated centring vector introduces. Dashed = 1/number of subjects. "
            + ("The rev1 P0/P1 bars retain the original all-trial-centred trial-random/run-disjoint 3-fold pooled accuracies; "
               "their raw accuracies are retained below. " if old else "")
            + "In the revision-1 two-subset probe, n_train × centred training mean + n_test × centred test mean = 0, "
            "forcing opposite residual centroids and potentially below-chance test accuracy. Scheme (iv) here instead "
            "obeys a three-run weighted-zero relation including the reference run. Linear-probe performance under these "
            "assignments does not establish removal of identity information, biological identity, motor-label exploitation "
            "or a mediation fraction. ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3a", help="revision-1 identity CSV (optional beside --e3a2)")
    ap.add_argument("--e3b", help="revision-1 variance CSV")
    ap.add_argument("--e3a2", help="E3a-v2 assignment CSV; matching _per_subject.csv is required")
    ap.add_argument("--e3b2", help="E3b-v2 sequential SS CSV")
    ap.add_argument("--e3c", required=True)
    ap.add_argument("--e3d", required=True)
    ap.add_argument("--e1", required=True, help="e1 eegbci per-subject (original TS+LR ladder)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if bool(args.e3a2) != bool(args.e3b2) or (not args.e3a2 and not (args.e3a and args.e3b)):
        ap.error("supply both --e3a2 and --e3b2, or both revision-1 --e3a and --e3b")
    args.tag = args.tag or ("rev2" if args.e3a2 else "rev1")
    plt = F.setup_mpl()
    files = [Path(args.e3c), Path(args.e3d), Path(args.e1)]
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(2, 2, figsize=(F.FIG_W_IN, 6.0 if args.e3a2 else 5.2))
    cap = []
    if args.e3a2:
        ab_caption = panels_v2(axes, args, files, cap)
    else:
        files.extend([Path(args.e3a), Path(args.e3b)])
        # (a) variance decomposition
        ax = axes[0, 0]
        b = [r for r in C.read_csv(Path(args.e3b)) if r["space"] == "common_tangent"]
        comps = [("frac_subject", "subject", "#0072B2"), ("frac_class", "class", "#D55E00"), ("frac_subject_x_class", "subject × class", "#E69F00"), ("frac_residual", "residual", "#BBBBBB")]
        for i, r in enumerate(b):
            bottom = 0.0
            for key, lab, col in comps:
                v = float(r[key])
                ax.bar(i, v, bottom=bottom, color=col, width=0.7, label=lab if i == 0 else None, linewidth=0)
                if v > 0.10:  # only label a segment tall enough to hold 8 pt text
                    ax.text(i, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if key != "frac_residual" else "black")
                bottom += v
            cap.append(f"{C.TASK_SHORT[r['task']]}: subject {float(r['frac_subject']):.3f}, class {float(r['frac_class']):.3f}, subject×class {float(r['frac_subject_x_class']):.3f}, residual {float(r['frac_residual']):.3f}")
        ax.set_xticks(range(len(b))); ax.set_xticklabels([C.TASK_SHORT[r["task"]].replace(" ", "\n").replace("fists/feet", "fists/\nfeet") for r in b], fontsize=8)
        ax.set_ylabel("Fraction of total sum\nof squares (dimensionless)"); ax.set_ylim(0, 1.42)
        ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03), columnspacing=1.0, handlelength=1.2, handletextpad=0.5)
        ax.set_title("(a) variance decomposition\nof tangent features")
        # (b) subject-ID decoding
        ax = axes[0, 1]
        a = C.read_csv(Path(args.e3a))
        conds = [("trial_random", "False"), ("trial_random", "True"), ("run_disjoint", "False"), ("run_disjoint", "True")]
        labels = ["P0\nraw", "P0\nsubject-\ncentred", "P1\nraw", "P1\nsubject-\ncentred"]
        tasks = sorted({r["task"] for r in a}, key=C.TASKS.index)
        w = 0.8 / max(1, len(tasks))
        group_vals = {}
        for ti, t in enumerate(tasks):
            vals = []
            for split, cen in conds:
                rr = [r for r in a if r["task"] == t and r["split"] == split and r["centered"] == cen]
                vals.append(float(rr[0]["subject_id_accuracy"]) if rr else np.nan)
            ax.bar(np.arange(4) + (ti - (len(tasks) - 1) / 2) * w, vals, width=w * 0.95, label=C.TASK_SHORT[t], color=["#0072B2", "#56B4E9", "#009E73", "#CC79A7"][ti % 4], linewidth=0)
            group_vals[t] = vals
            cap.append(f"subject-ID {C.TASK_SHORT[t]}: " + ", ".join(f"{l.replace(chr(10), ' ')} {v:.3f}" for l, v in zip(labels, vals)))
        # D-30: a bar of height zero must not read as "not evaluated" -> print the value over the whole group.
        # F-r003-3: the printed number is the GROUP MEAN over the four paradigms, exactly as the caption says
        # (the underlying scores are 0 or 2.04e-4, so the mean rounds to 0.000 but is not identically zero).
        for ci in range(4):
            col = [group_vals[t][ci] for t in tasks if not np.isnan(group_vals[t][ci])]
            if col and max(col) < 0.0005:  # every bar of the group is invisible at this scale
                ax.text(ci, 0.03, f"{float(np.mean(col)):.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
        chance = float(a[0]["chance"])
        ax.axhline(chance, color="#888888", lw=0.8, ls="--")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("Subject-identity accuracy\n(fraction correct)"); ax.set_ylim(0, 1.45)
        hb, lb = ax.get_legend_handles_labels()
        hb.append(Line2D([], [], color="#888888", lw=0.8, ls="--")); lb.append(f"chance ({chance:.3f})")
        ax.legend(hb, lb, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03), columnspacing=1.0, handlelength=1.2, handletextpad=0.5)
        ax.set_title("(b) decoding subject identity\nfrom tangent features")
    # (c) mean removal / recentering
    ax = axes[1, 0]
    e1 = [r for r in C.read_csv(Path(args.e1)) if r["model"] == "riemann_ts_lr"]
    for r in e1:
        r["variant"] = "original"
    c3 = C.read_csv(Path(args.e3c))
    subj = F.seed_avg_subject(e1 + c3, ("task", "variant", "protocol"))
    pm = F.paradigm_mean(subj, 0)
    variants = [("original", "original"), ("feature_centered", "subject\nmean\nremoved"), ("riemann_recentered", "Riemannian\nrecentred")]
    for xi, (v, lab) in enumerate(variants):
        for k, (pa, pb, col, off) in enumerate((("run_disjoint", "subject_disjoint", "#E69F00", -0.18), ("trial_random", "subject_disjoint", "#D55E00", 0.18))):
            A, B = pm.get(("paradigm_mean", v, pa), {}), pm.get(("paradigm_mean", v, pb), {})
            subs = sorted(set(A) & set(B))
            if not subs:
                continue
            d = np.array([A[s] - B[s] for s in subs])
            F.strip_points(ax, xi + off, d, col, rng, width=0.12)
            m, lo, hi = F.mean_ci(ax, xi + off, d, col, width=0.15)
            cap.append(f"{v} Δ({'P1' if pa == 'run_disjoint' else 'P0'}−P2) {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(d)}")
    ax.scatter([], [], color="#E69F00", label="Δ(P1 − P2)"); ax.scatter([], [], color="#D55E00", label="Δ(P0 − P2)")
    ax.axhline(0, color="#888888", lw=0.8, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in variants], fontsize=8)
    ax.set_ylabel("Accuracy contrast\n(fraction correct)"); ax.legend(loc="upper right", frameon=True, handlelength=1.2, handletextpad=0.5)
    ax.set_title("(c) TS+LR after removing\nsubject means")
    # (d) weight alignment
    ax = axes[1, 1]
    d3 = C.read_csv(Path(args.e3d))
    metrics = [("mean_abs_cos_w_d_s_train_subjects", "|cos(w, d_s)|"), ("fraction_w_in_span_train_d_s", "‖w‖² in\nspan{d_s}"), ("abs_cos_w_d_global", "|cos(w,\nd_global)|")]
    for xi, (met, lab) in enumerate(metrics):
        for pa, col, off in (("trial_random", "#D55E00", -0.18), ("subject_disjoint", "#0072B2", 0.18)):
            v = np.array([float(r["value"]) for r in d3 if r["metric"] == met and r["protocol"] == pa])
            if len(v) == 0:
                continue
            F.strip_points(ax, xi + off, v, col, rng, width=0.1, size=9, alpha=0.6)
            m, lo, hi = F.mean_ci(ax, xi + off, v, col, width=0.15)
            cap.append(f"{met} {pa}: {m:.3f} [{lo:.3f},{hi:.3f}] n={len(v)} folds")
    base = [float(r["value"]) for r in d3 if r["metric"] == "random_direction_fraction_in_span"]
    if base:
        ax.hlines(np.mean(base), 0.55, 1.45, color="#888888", ls="--", lw=0.8)
    pw = [float(r["value"]) for r in d3 if r["metric"] == "mean_pairwise_abs_cos_d_s"]
    if pw:
        cap.append(f"mean pairwise |cos(d_s, d_s')| over paradigms = {np.mean(pw):.3f}")
    ax.scatter([], [], color="#D55E00", label="w from P0 (trial-random)"); ax.scatter([], [], color="#0072B2", label="w from P2 (subject-disjoint)")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in metrics], fontsize=8); ax.set_ylabel("Alignment (dimensionless)")
    ax.set_ylim(top=1.30)
    hd, ld = ax.get_legend_handles_labels()
    if base:
        hd.append(Line2D([], [], color="#888888", lw=0.8, ls="--")); ld.append(f"random direction in span ({np.mean(base):.3f})")
    ax.legend(hd, ld, loc="upper left", frameon=True, handlelength=1.0, handletextpad=0.5)
    ax.set_title("(d) weight alignment\n(points = folds × seeds)")
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    caption = ("Figure 5. Subject information in the tangent-space features (EEGBCI, 109 subjects). (a) Fraction of the total sum of squares of the tangent features "
               "attributable to subject, class, subject × class (cell means) and residual, in a common tangent space (reference = Fréchet mean of all trials; descriptive only). "
               "(b) Accuracy of a multinomial logistic regression predicting the subject (109 classes) from tangent features under the trial-random (P0) and run-disjoint (P1) splits, "
               "raw and after removing each subject's feature mean (subject-centred); dashed = chance; the subject-centred groups were evaluated, not omitted: their identity accuracy after per-subject centring is printed above the bars as the four-paradigm group mean, which rounds to 0.000. (c) Per-subject protocol contrasts Δ(P1 − P2) and Δ(P0 − P2) for TS+LR (C = 1) with the original features, "
               "after subtracting each subject's feature mean, and after Riemannian recentering (points = subjects; bar = mean; whisker = 95 % CI). "
               "(d) Alignment of the fitted weight vector w with the subject-specific class-separation directions d_s and with the global direction (points = folds × seeds). Values: " + " | ".join(cap))
    if args.e3a2:
        caption = ("Figure 5. Subject information in tangent-space features (EEGBCI). "
                   + ab_caption + caption[caption.index("(c) Per-subject"):])
    F.save(fig, f"fig5_mechanism_{args.tag}", caption, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
