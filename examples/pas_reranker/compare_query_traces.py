#!/usr/bin/env python3
"""Compare two full PAS reranker traces query by query and candidate by candidate."""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--order-independent", action="store_true", help="Match queries by dataset, type, and caption when shard counts differ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "queries": 0,
        "candidates": 0,
        "different_rerank_order": 0,
        "different_rank1": 0,
        "different_rank5": 0,
        "exact_candidate_scores": 0,
        "different_retriever_head": 0,
        "different_candidate_set": 0,
    }
    score_abs_sum = score_abs_max = ap_delta_sum = 0.0
    key = lambda row: (row["dataset"], row["query_type"], row["caption"])
    old_by_key = None
    if args.order_independent:
        old_by_key = {}
        with args.old.open() as old_stream:
            for old_line in old_stream:
                old = json.loads(old_line)
                old_key = key(old)
                if old_key in old_by_key:
                    raise SystemExit(f"Duplicate old query: {old_key}")
                old_by_key[old_key] = old
    with args.old.open() as old_stream, args.new.open() as new_stream:
        pairs = ((None, line) for line in new_stream) if args.order_independent else zip_longest(old_stream, new_stream)
        for line_number, (old_line, new_line) in enumerate(pairs, start=1):
            if args.limit and line_number > args.limit:
                break
            if new_line is None or (old_line is None and not args.order_independent):
                raise SystemExit(f"Trace lengths differ at line {line_number}")
            new = json.loads(new_line)
            old = old_by_key.pop(key(new), None) if args.order_independent else json.loads(old_line)
            if old is None:
                raise SystemExit(f"Query missing from old trace at line {line_number}")
            if key(old) != key(new):
                raise SystemExit(f"Query differs at line {line_number}")
            counts["different_retriever_head"] += old["stage1_head"] != new["stage1_head"]
            if len(old["candidates"]) != len(new["candidates"]):
                raise SystemExit(f"Candidate count differs at line {line_number}")
            counts["queries"] += 1
            counts["different_rerank_order"] += old["reranked_head"] != new["reranked_head"]
            old_rank = old["reranker_first_gt_rank"]
            new_rank = new["reranker_first_gt_rank"]
            counts["different_rank1"] += (old_rank == 1) != (new_rank == 1)
            counts["different_rank5"] += (old_rank <= 5) != (new_rank <= 5)
            ap_delta_sum += new["reranker_ap"] - old["reranker_ap"]
            old_candidates = {candidate["local_index"]: candidate for candidate in old["candidates"]}
            new_candidates = {candidate["local_index"]: candidate for candidate in new["candidates"]}
            if old_candidates.keys() != new_candidates.keys():
                counts["different_candidate_set"] += 1
            for candidate_id in old_candidates.keys() & new_candidates.keys():
                old_candidate, new_candidate = old_candidates[candidate_id], new_candidates[candidate_id]
                difference = abs(
                    new_candidate["rerank_score"] - old_candidate["rerank_score"]
                )
                counts["candidates"] += 1
                counts["exact_candidate_scores"] += difference == 0.0
                score_abs_sum += difference
                score_abs_max = max(score_abs_max, difference)
    if not counts["queries"]:
        raise SystemExit("No queries compared")
    if old_by_key and not args.limit:
        raise SystemExit(f"{len(old_by_key)} old queries missing from new trace")
    result = {
        **counts,
        "mean_abs_candidate_score_delta": score_abs_sum / counts["candidates"],
        "max_abs_candidate_score_delta": score_abs_max,
        "mean_new_minus_old_ap": ap_delta_sum / counts["queries"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
