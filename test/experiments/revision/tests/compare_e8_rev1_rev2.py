#!/usr/bin/env python3
"""Compare two E8 per-draw CSVs without changing either file (always exit 0)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEYS = ("task", "model", "N", "d", "draw", "protocol")


def read_conditions(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {tuple(r[k] for k in KEYS): float(r["mean_acc"]) for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("first", help="released per_draw CSV")
    ap.add_argument("second", help="corrected per_draw CSV")
    args = ap.parse_args()
    first, second = read_conditions(args.first), read_conditions(args.second)
    common = first.keys() & second.keys()
    deltas = [abs(first[k] - second[k]) for k in common]
    print("| Comparison | Value |")
    print("|---|---:|")
    print(f"| Conditions in both | {len(common)} |")
    print(f"| Max absolute change in mean_acc | {max(deltas, default=0.0):.12g} |")
    print(f"| Absolute changes > 1e-6 | {sum(d > 1e-6 for d in deltas)} |")
    print(f"| Only in first (expected exclusions) | {len(first.keys() - second.keys())} |")
    print(f"| Only in second | {len(second.keys() - first.keys())} |")
    for label, keys, source in (("Only in first", first.keys() - second.keys(), first),
                                ("Only in second", second.keys() - first.keys(), second)):
        print(f"\n{label}:")
        print("| " + " | ".join(KEYS + ("mean_acc",)) + " |")
        print("| " + " | ".join(["---"] * (len(KEYS) + 1)) + " |")
        for k in sorted(keys, key=lambda k: (k[0], k[1], int(k[2]), int(k[3]), int(k[4]), k[5])):
            print("| " + " | ".join(k + (f"{source[k]:.12g}",)) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
