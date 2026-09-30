#!/usr/bin/env python3
"""Compare a step-8000 replay's reranking and validation-fused rows to PAS V3.1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("mAP", "Rank-1", "Rank-5")
MODES = ("Easy", "Medium", "Hard", "Overall")
ROWS = ((4, "reranking", "test_cr3_only"), (5, "validation_fusion", "test_at_frozen_validation_alpha"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Val999-fit/test-applied fusion summary.json")
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "reports/pas_cr3_nano_reference_table.csv"
    )
    parser.add_argument("--output", type=Path, help="Optional machine-readable comparison JSON")
    args = parser.parse_args()

    with args.reference.open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    if len(reference) != 32:
        raise ValueError(f"Expected 32 reference rows, found {len(reference)}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("selection_metric") != "validation overall mAP":
        raise ValueError("Summary does not select fusion on validation overall mAP")

    comparisons = []
    for mode_index, mode in enumerate(MODES):
        for row_index, label, summary_key in ROWS:
            table_row = reference[mode_index * 8 + row_index]
            if table_row["Mode"] != mode:
                raise ValueError(f"Unexpected reference row order for {mode} {label}")
            actual_row = summary[summary_key]
            metrics = {}
            for metric in METRICS:
                expected = float(table_row[metric].removesuffix("%"))
                actual = round(100 * actual_row[f"{mode.lower()}_{metric}"], 2)
                metrics[metric] = {
                    "reference_percent": expected,
                    "replayed_percent": actual,
                    "delta_percentage_points": round(actual - expected, 2),
                }
            comparisons.append({"mode": mode, "row": label, "metrics": metrics})

    result = {
        "reference": str(args.reference.resolve()),
        "summary": str(args.summary.resolve()),
        "test_rankings": summary.get("test_rankings"),
        "validation_rankings": summary.get("validation_rankings"),
        "validation_selected_alpha": summary["frozen_alpha"],
        "comparisons": comparisons,
        "max_absolute_delta_percentage_points": max(
            abs(value["delta_percentage_points"])
            for row in comparisons for value in row["metrics"].values()
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
