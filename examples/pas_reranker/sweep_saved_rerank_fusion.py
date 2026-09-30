#!/usr/bin/env python3
"""Sweep normalized SigLIP2/CR3 score fusion from a saved PAS ranking trace."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


QUERY_TYPES = ("easy", "medium", "hard")
METRICS = ("mAP", "Rank-1", "Rank-5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        choices=("none", "global_zscore", "query_zscore", "query_minmax"),
        default="query_zscore",
    )
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    stage1, cr3, labels, num_gt, tail_ap, query_types = [], [], [], [], [], []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            candidates = sorted(row["candidates"], key=lambda item: item["stage1_rank"])
            if len(candidates) != 20:
                raise ValueError(f"line {line_number}: expected depth 20, got {len(candidates)}")
            y = np.asarray([bool(item["is_gt"]) for item in candidates], dtype=np.bool_)
            ranks = np.arange(1, 21, dtype=np.float64)
            head_precision_sum = float(np.sum((np.cumsum(y) / ranks)[y]))
            n_gt = int(row["num_gt"])
            stage1.append([float(item["stage1_score"]) for item in candidates])
            cr3.append([float(item["rerank_score"]) for item in candidates])
            labels.append(y)
            num_gt.append(n_gt)
            tail_ap.append(float(row["retriever_ap"]) * n_gt - head_precision_sum)
            query_types.append(row["query_type"])
    return {
        "stage1": np.asarray(stage1, dtype=np.float64),
        "cr3": np.asarray(cr3, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.bool_),
        "num_gt": np.asarray(num_gt, dtype=np.float64),
        "tail_ap": np.asarray(tail_ap, dtype=np.float64),
        "query_type": np.asarray(query_types),
    }


def normalize_scores(data: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    """Return a shallow copy with both score arrays put on comparable scales."""
    normalized = dict(data)
    if mode == "none":
        return normalized

    for key in ("stage1", "cr3"):
        values = data[key]
        if mode == "global_zscore":
            center = values.mean()
            scale = values.std()
        elif mode == "query_zscore":
            center = values.mean(axis=1, keepdims=True)
            scale = values.std(axis=1, keepdims=True)
        elif mode == "query_minmax":
            center = values.min(axis=1, keepdims=True)
            scale = values.max(axis=1, keepdims=True) - center
        else:
            raise ValueError(f"unsupported normalization: {mode}")
        normalized[key] = (values - center) / np.maximum(scale, 1e-12)
    return normalized


def evaluate(data: dict[str, np.ndarray], alpha: float) -> dict[str, float]:
    # Stable sorting preserves SigLIP2 order on exact fused-score ties.
    scores = alpha * data["stage1"] + (1.0 - alpha) * data["cr3"]
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_labels = np.take_along_axis(data["labels"], order, axis=1)
    ranks = np.arange(1, 21, dtype=np.float64)
    precision = np.cumsum(ranked_labels, axis=1) / ranks[None, :]
    head_ap_sum = np.sum(precision * ranked_labels, axis=1)
    ap = (head_ap_sum + data["tail_ap"]) / data["num_gt"]
    rank1 = ranked_labels[:, 0].astype(np.float64)
    rank5 = np.any(ranked_labels[:, :5], axis=1).astype(np.float64)
    result: dict[str, float] = {"alpha": float(alpha)}
    for query_type in QUERY_TYPES:
        mask = data["query_type"] == query_type
        result[f"{query_type}_num_queries"] = int(mask.sum())
        result[f"{query_type}_mAP"] = float(ap[mask].mean())
        result[f"{query_type}_Rank-1"] = float(rank1[mask].mean())
        result[f"{query_type}_Rank-5"] = float(rank5[mask].mean())
    result["overall_num_queries"] = int(len(ap))
    result["overall_mAP"] = float(ap.mean())
    result["overall_Rank-1"] = float(rank1.mean())
    result["overall_Rank-5"] = float(rank5.mean())
    return result


def unique_grid(values: np.ndarray) -> list[float]:
    return sorted({float(np.clip(value, 0.0, 1.0)) for value in values})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_data = load_trace(args.rankings)
    data = normalize_scores(raw_data, args.normalization)

    coarse = unique_grid(np.linspace(0.0, 1.0, 201))
    coarse_rows = [evaluate(data, alpha) for alpha in coarse]
    coarse_best = max(coarse_rows, key=lambda row: row["overall_mAP"])
    fine = unique_grid(np.arange(coarse_best["alpha"] - 0.01, coarse_best["alpha"] + 0.01001, 0.0001))
    fine_rows = [evaluate(data, alpha) for alpha in fine]
    fine_best = max(fine_rows, key=lambda row: row["overall_mAP"])
    ultra = unique_grid(np.arange(fine_best["alpha"] - 0.0002, fine_best["alpha"] + 0.000201, 0.00001))
    ultra_rows = [evaluate(data, alpha) for alpha in ultra]

    rows_by_alpha = {row["alpha"]: row for row in coarse_rows + fine_rows + ultra_rows}
    rows = [rows_by_alpha[alpha] for alpha in sorted(rows_by_alpha)]
    best = max(rows, key=lambda row: row["overall_mAP"])
    cr3_only = evaluate(data, 0.0)
    siglip_only = evaluate(data, 1.0)

    columns = list(rows[0])
    with (args.output_dir / "alpha_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "warning": "alpha selected on test; diagnostic/test-tuned, not a held-out test claim",
        "normalization": args.normalization,
        "formula": "alpha * normalize(SigLIP2_score) + (1 - alpha) * normalize(CR3_score)",
        "selection_metric": "overall_mAP",
        "rankings": str(args.rankings.resolve()),
        "raw_score_statistics": {
            key: {
                "mean": float(raw_data[key].mean()),
                "std": float(raw_data[key].std()),
                "min": float(raw_data[key].min()),
                "max": float(raw_data[key].max()),
            }
            for key in ("stage1", "cr3")
        },
        "evaluated_alpha_count": len(rows),
        "best": best,
        "cr3_only_alpha_0": cr3_only,
        "siglip2_only_alpha_1": siglip_only,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
