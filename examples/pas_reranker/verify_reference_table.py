#!/usr/bin/env python3
"""Compare the pasted PAS table with available trace and fresh-run metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from sweep_saved_rerank_fusion import evaluate, load_trace


ROOT = Path(__file__).resolve().parents[2]
MODES = ("Easy", "Medium", "Hard", "Overall")
METRICS = ("mAP", "Rank-1", "Rank-5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path,
        default=ROOT / "reports/pas_cr3_nano_reference_table.csv",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "reports/pas_cr3_baseline_reference.csv",
    )
    parser.add_argument(
        "--base-cr3-trace-summary", type=Path,
        default=ROOT / "reports/pas_cr3_base_cr3_trace_replay.json",
        help="Saved-trace replay for the public and zero-shot CR3 rows.",
    )
    parser.add_argument(
        "--checkpoint-summary", type=Path,
        default=ROOT / "reports/pas_cr3_step8000_fusion_replay.json",
    )
    parser.add_argument(
        "--zeroshot-fusion-summary", type=Path,
        default=ROOT / "reports/pas_cr3_zeroshot_fusion_replay.json",
    )
    parser.add_argument("--binary-trace", type=Path)
    parser.add_argument("--public-cr3-trace", type=Path, help="Fresh public CR3 query rankings")
    parser.add_argument("--zeroshot-cr3-trace", type=Path, help="Fresh zero-shot CR3 query rankings")
    parser.add_argument(
        "--max-delta-pp", type=float, default=0.0,
        help="Maximum allowed absolute cell difference in percentage points (default: exact rounded cells).",
    )
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def from_summary(row: dict, mode: str) -> tuple[str, str, str]:
    name = mode.lower()
    return tuple(percent(row[f"{name}_{metric}"]) for metric in METRICS)


def from_base_trace_summary(summary: dict, model: str, mode: str) -> tuple[str, str, str]:
    return tuple(
        summary[model]["metrics"][mode.lower()][metric]["replayed"]
        for metric in METRICS
    )


def main() -> None:
    args = parse_args()
    if args.max_delta_pp < 0:
        raise SystemExit("--max-delta-pp must be nonnegative")
    with args.reference.open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    with args.baseline.open(newline="") as stream:
        baseline = list(csv.DictReader(stream))
    checkpoint = json.loads(args.checkpoint_summary.read_text())
    zeroshot = json.loads(args.zeroshot_fusion_summary.read_text())
    base_cr3 = json.loads(args.base_cr3_trace_summary.read_text())
    if len(reference) != 32 or len(baseline) != 16:
        raise SystemExit("Expected 32 reference rows and 16 baseline rows")

    public_fresh = evaluate(load_trace(args.public_cr3_trace), 0.0) if args.public_cr3_trace else None
    zeroshot_trace = load_trace(args.zeroshot_cr3_trace) if args.zeroshot_cr3_trace else None
    zeroshot_fresh = evaluate(zeroshot_trace, 0.0) if zeroshot_trace is not None else None
    binary = None
    if args.binary_trace:
        trace = zeroshot_trace if args.binary_trace == args.zeroshot_cr3_trace else load_trace(args.binary_trace)
        trace["cr3"] = (trace["cr3"] > 0).astype(np.float64)
        binary = evaluate(trace, 0.0)

    matched_rows = 0
    missing_rows = 0
    different_rows = 0
    max_absolute_delta_pp = 0.0
    for mode_index, mode in enumerate(MODES):
        rows = reference[mode_index * 8 : mode_index * 8 + 8]
        if any(row["Mode"] != mode for row in rows):
            raise SystemExit(f"Unexpected reference row order for {mode}")
        expected = [tuple(row[metric] for metric in METRICS) for row in rows]
        actual = [
            tuple(baseline[mode_index * 4][metric] for metric in METRICS),
            from_summary(public_fresh, mode) if public_fresh else from_base_trace_summary(base_cr3, "public", mode),
            tuple(baseline[mode_index * 4 + 2][metric] for metric in METRICS),
            from_summary(zeroshot_fresh, mode) if zeroshot_fresh else from_base_trace_summary(base_cr3, "finetuned", mode),
        ]
        actual += [
            from_summary(checkpoint["test_cr3_only"], mode),
            from_summary(checkpoint["test_at_frozen_validation_alpha"], mode),
            from_summary(zeroshot["best"], mode),
            from_summary(binary, mode) if binary else None,
        ]
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            label = f"{mode:<7} row {index + 1}"
            if got is None:
                missing_rows += 1
                print(f"MISSING   {label}: binary trace not supplied")
                continue
            normalized_want = tuple(f"{float(value.removesuffix('%')):.2f}%" for value in want)
            max_absolute_delta_pp = max(
                max_absolute_delta_pp,
                *(abs(float(old.removesuffix('%')) - float(new.removesuffix('%')))
                  for old, new in zip(normalized_want, got, strict=True)),
            )
            if got == normalized_want:
                matched_rows += 1
                print(f"MATCH     {label}: {', '.join(got)}")
            else:
                different_rows += 1
                differences = "; ".join(
                    f"{name} {old} vs {new}"
                    for name, old, new in zip(METRICS, normalized_want, got, strict=True)
                    if old != new
                )
                print(f"DIFFERENT {label}: {differences}")
    print(
        f"Matched {matched_rows}/32 rows; {different_rows} differ; "
        f"{missing_rows} lack a binary trace"
    )
    print(
        f"Maximum absolute cell difference: {max_absolute_delta_pp:.2f} percentage "
        f"points; allowed: {args.max_delta_pp:.2f}"
    )
    if missing_rows or max_absolute_delta_pp > args.max_delta_pp + 1e-9:
        sys.exit(1)


if __name__ == "__main__":
    main()
