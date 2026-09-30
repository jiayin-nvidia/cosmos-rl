#!/usr/bin/env python3
"""Render portable replay TOMLs, optionally selecting freshly mined data."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pas-data-root", type=Path)
    parser.add_argument("--cr3-base-model", type=Path)
    parser.add_argument(
        "--data-parallel-ranks", type=int, default=8,
        help="GPU count; keeps the original global batch of 640 K20 candidates",
    )
    parser.add_argument(
        "--run-root", type=Path,
        help="Separate output root for a changed data-parallel GPU count",
    )
    args = parser.parse_args()
    if (args.data_root is None) != (args.cache_root is None):
        parser.error("--data-root and --cache-root must be provided together")
    if args.data_parallel_ranks < 1 or 640 % args.data_parallel_ranks:
        parser.error("--data-parallel-ranks must divide the original global batch of 640")
    batch_per_rank = 640 // args.data_parallel_ranks
    if batch_per_rank % 20:
        parser.error("The per-rank batch must preserve complete K20 microbatches")
    if args.data_parallel_ranks != 8 and args.run_root is None:
        parser.error("--run-root is required when changing the data-parallel GPU count")
    repo = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replacements = {
        "/workspace/jiayin/cosmos-rl-pas-cr3-repro/": str(repo.resolve()) + "/",
    }
    if args.data_root is not None:
        replacements.update({
            "/workspace/jiayin/cosmos-rl/data/pas_cr3_step03906/":
                str(args.data_root.resolve()) + "/",
            "/workspace/jiayin/cosmos-rl/outputs/pas_cr3_attribute_replay_records_alltrain/":
                str((args.data_root / "attribute_replay").resolve()) + "/",
            "/workspace/jiayin/cache/pas_cr3_block24_prefix/":
                str(args.cache_root.resolve()) + "/",
        })
    if args.pas_data_root is not None:
        replacements[
            "/workspace/data/PAS_Datasets/NVIDIA_PAS_Filtered_06152026_V3.1_tao_ft_metadata_base_plus_val_test_medium_attr_matched_aug_val20_accessory_v2.1/"
        ] = str(args.pas_data_root.resolve()) + "/"
    if args.cr3_base_model is not None:
        replacements["/workspace/jiayin/models/Cosmos3-Nano-VLM"] = str(
            args.cr3_base_model.resolve()
        )
    names = (
        "train_stage0_to_step2000_replay.toml",
        "train_stage2000_to_step8000_replay.toml",
    )
    for name in names:
        source = repo / "examples/pas_reranker" / name
        content = source.read_text()
        for old, new in replacements.items():
            if old not in content:
                raise ValueError(f"Missing expected path prefix in {source}: {old}")
            content = content.replace(old, new)
        if args.run_root is not None:
            for stage in ("stage0", "stage1"):
                old = str(repo.resolve() / "reproduction/finetune" / stage)
                new = str(args.run_root.resolve() / stage)
                if old in content:
                    content = content.replace(old, new)
        if args.data_parallel_ranks != 8:
            stage = "stage0" if "stage0" in name else "stage1"
            for old, new in (
                ("train_batch_per_replica = 80", f"train_batch_per_replica = {batch_per_rank}"),
                ("dataloader_batch_size = 80", f"dataloader_batch_size = {batch_per_rank}"),
                ("dp_replicate_size = 8", f"dp_replicate_size = {args.data_parallel_ranks}"),
                (
                    f'timestamp = "pascr3repro{stage}"',
                    f'timestamp = "pascr3reprofast{args.data_parallel_ranks}{stage}"',
                ),
            ):
                if old not in content:
                    raise ValueError(f"Missing expected training setting in {source}: {old}")
                content = content.replace(old, new)
        tomllib.loads(content)
        destination = args.output_dir / name
        destination.write_text(content)
        print(destination)


if __name__ == "__main__":
    main()
