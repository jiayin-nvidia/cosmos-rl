#!/usr/bin/env python3
"""Build exact train-only full-gallery AP denominators for PAS top-K groups.

Existing grouped annotations contain the labels and SigLIP2 ranks of the
reranked head, but not the total number of relevant gallery images.  That
missing denominator makes within-head AP a different objective from deployed
full-gallery AP.  This builder reconstructs the deterministic train-only
pseudo-gallery used by the miner and emits one ``total_relevant`` value per
group.  It accepts no validation or test input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import ijson
import numpy as np

try:
    from examples.pas_reranker.build_hcr_constraint_assets import (
        DenseHCRAssets,
        load_dense_hcr_assets,
        raw_index_from_alias,
    )
    from examples.pas_reranker.full_gallery_ap import ap_numerator
    from examples.pas_reranker.prepare_annotations import iter_json_records
except ModuleNotFoundError:
    from build_hcr_constraint_assets import (  # type: ignore[no-redef]
        DenseHCRAssets,
        load_dense_hcr_assets,
        raw_index_from_alias,
    )
    from full_gallery_ap import ap_numerator  # type: ignore[no-redef]
    from prepare_annotations import iter_json_records  # type: ignore[no-redef]


ASSET_VERSION = 1
_QUERY_INDEX = re.compile(r"^query_(\d+)(?:_view_\d+)?$")


@dataclass(frozen=True)
class GroupDescriptor:
    group_id: str
    dataset: str
    query_index: int
    labels: tuple[int, ...]
    pseudo_gallery_fraction: float
    pseudo_gallery_size: int
    partition_seed: int


@dataclass(frozen=True)
class GalleryCandidate:
    raw_index: int
    image_key: str


@dataclass(frozen=True)
class GalleryPartition:
    buckets: tuple[np.ndarray, ...]
    bucket_by_image_key: Mapping[str, int]


@dataclass(frozen=True)
class DenseFullGalleryAPMetadata:
    group_ids: np.ndarray
    total_relevant: np.ndarray
    topk_positive_count: np.ndarray
    retriever_head_ap_contribution: np.ndarray
    manifest_path: Path

    def as_lookup(self) -> dict[str, int]:
        return {
            str(group_id): int(total)
            for group_id, total in zip(
                self.group_ids, self.total_relevant, strict=True
            )
        }


def load_full_gallery_ap_metadata(
    manifest_path: str | Path,
) -> DenseFullGalleryAPMetadata:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train-only":
        raise ValueError("AP metadata must declare source_split=train-only")
    if manifest.get("validation_rows_read") or manifest.get("test_rows_read"):
        raise ValueError("AP metadata manifest declares held-out row access")
    dense = manifest.get("dense_assets")
    if not isinstance(dense, dict):
        raise ValueError("AP metadata manifest has no dense_assets")

    group_ids = np.load(Path(dense["group_ids"]), mmap_mode="r")
    totals = np.load(Path(dense["total_relevant_uint32"]), mmap_mode="r")
    positive_count = np.load(Path(dense["topk_positive_count_uint8"]), mmap_mode="r")
    contribution = np.load(
        Path(dense["retriever_head_ap_contribution_float32"]), mmap_mode="r"
    )
    expected = int(dense["group_count"])
    if any(value.shape != (expected,) for value in (group_ids, totals, positive_count, contribution)):
        raise ValueError("AP metadata dense arrays are not group aligned")
    if len({str(value) for value in group_ids}) != expected:
        raise ValueError("AP metadata group IDs are not unique")
    if np.any(totals < positive_count) or np.any(totals == 0):
        raise ValueError("AP metadata contains invalid total-relevant counts")
    return DenseFullGalleryAPMetadata(
        group_ids=group_ids,
        total_relevant=totals,
        topk_positive_count=positive_count,
        retriever_head_ap_contribution=contribution,
        manifest_path=path,
    )


def stable_u64(*values: object) -> int:
    """Exact partition hash used by the train-only pseudo-gallery miner."""

    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def build_partition(
    candidates: Sequence[GalleryCandidate],
    *,
    fraction: float,
    seed: int,
    minimum_bucket_size: int,
) -> GalleryPartition:
    if not candidates or not 0 < fraction <= 1:
        raise ValueError("nonempty candidates and fraction in (0, 1] are required")
    requested_buckets = max(1, int(round(1.0 / fraction)))
    bucket_count = min(requested_buckets, len(candidates) // minimum_bucket_size)
    if bucket_count <= 0:
        raise ValueError("gallery is smaller than minimum_bucket_size")
    order = sorted(
        range(len(candidates)),
        key=lambda index: stable_u64(seed, candidates[index].image_key),
    )
    bucket_lists: list[list[int]] = [[] for _ in range(bucket_count)]
    bucket_by_image_key: dict[str, int] = {}
    for offset, candidate_index in enumerate(order):
        bucket_index = offset % bucket_count
        candidate = candidates[candidate_index]
        bucket_lists[bucket_index].append(candidate.raw_index)
        bucket_by_image_key[candidate.image_key] = bucket_index
    return GalleryPartition(
        buckets=tuple(np.asarray(bucket, dtype=np.int64) for bucket in bucket_lists),
        bucket_by_image_key=bucket_by_image_key,
    )


def count_relevant_in_pool(
    assets: DenseHCRAssets,
    *,
    query_index: int,
    candidate_raw_indices: np.ndarray,
) -> int:
    """Count exact PAS conjunction matches in a reconstructed gallery pool."""

    wanted = np.asarray(assets.query_attr[query_index])
    observed = np.asarray(assets.image_attr[candidate_raw_indices])
    active = wanted >= 0
    matches = np.ones(candidate_raw_indices.shape[0], dtype=bool)
    if np.any(active):
        matches &= np.all(observed[:, active] == wanted[active], axis=1)

    query_begin = int(assets.query_offsets[query_index])
    query_end = int(assets.query_offsets[query_index + 1])
    required_accessories = set(
        int(value) for value in assets.query_values[query_begin:query_end]
    )
    if required_accessories and np.any(matches):
        for local_index in np.flatnonzero(matches):
            raw_index = int(candidate_raw_indices[local_index])
            begin = int(assets.image_offsets[raw_index])
            end = int(assets.image_offsets[raw_index + 1])
            present = set(int(value) for value in assets.image_values[begin:end])
            if not required_accessories.issubset(present):
                matches[local_index] = False
    return int(matches.sum())


def iter_group_descriptors(
    path: Path, *, group_size: int, allow_single_class_groups: bool = False
) -> Iterable[GroupDescriptor]:
    """Stream exact top-K groups while validating rank and pool metadata."""

    with path.open("rb") as handle:
        rows = ijson.items(handle, "item")
        while True:
            try:
                first = next(rows)
            except StopIteration:
                return
            group = [first]
            for _ in range(group_size - 1):
                try:
                    group.append(next(rows))
                except StopIteration as error:
                    raise ValueError("annotations end inside a candidate group") from error
            group_ids = {str(row.get("group_id") or "") for row in group}
            if len(group_ids) != 1 or not next(iter(group_ids)):
                raise ValueError("annotation rows are not contiguous complete groups")
            ranks = [int(row.get("retriever_rank", -1)) for row in group]
            if ranks != list(range(1, group_size + 1)):
                raise ValueError("full-gallery AP metadata requires exact retriever ranks 1..K")
            labels = tuple(int(row.get("label", -1)) for row in group)
            if any(value not in {0, 1} for value in labels):
                raise ValueError("annotation labels must be binary")
            if not allow_single_class_groups and not 0 < sum(labels) < group_size:
                raise ValueError("LambdaAP groups must contain both classes")
            if allow_single_class_groups and sum(labels) == 0:
                raise ValueError("full-gallery AP evaluation requires at least one positive")
            group_id = next(iter(group_ids))
            query_indices = {row.get("query_index") for row in group}
            if len(query_indices) == 1 and next(iter(query_indices)) is not None:
                query_index = int(next(iter(query_indices)))
            else:
                match = _QUERY_INDEX.fullmatch(group_id)
                if match is None:
                    raise ValueError(f"cannot resolve query row from {group_id!r}")
                query_index = int(match.group(1))
            invariant_fields = (
                "dataset",
                "pseudo_gallery_fraction",
                "pseudo_gallery_size",
                "pseudo_gallery_partition_seed",
            )
            for field in invariant_fields:
                if len({str(row.get(field)) for row in group}) != 1:
                    raise ValueError(f"group disagrees on {field}")
            yield GroupDescriptor(
                group_id=group_id,
                dataset=str(first["dataset"]),
                query_index=query_index,
                labels=labels,
                pseudo_gallery_fraction=float(first["pseudo_gallery_fraction"]),
                pseudo_gallery_size=int(first["pseudo_gallery_size"]),
                partition_seed=int(first["pseudo_gallery_partition_seed"]),
            )


def load_galleries_and_query_sources(
    train_pairs: Path,
    *,
    gallery_unique_names: set[str],
    query_indices: set[int],
) -> tuple[dict[str, list[GalleryCandidate]], dict[int, str]]:
    """Recreate the miner's first-seen image order and selected gallery filter."""

    galleries: dict[str, list[GalleryCandidate]] = {}
    seen_image_keys: set[str] = set()
    query_sources: dict[int, str] = {}
    matched_names: set[str] = set()
    for raw_index, row in enumerate(iter_json_records(train_pairs)):
        dataset = str(row.get("dataset") or "")
        image_path = str(row.get("image_path") or "")
        image_key = f"{dataset}\t{image_path}"
        if raw_index in query_indices:
            query_sources[raw_index] = image_key
        # The miner deduplicates first and applies the unique-name allowlist to
        # the first occurrence, so filtering before deduplication is not exact.
        if image_key in seen_image_keys:
            continue
        seen_image_keys.add(image_key)
        unique_name = str(row.get("unique_name") or "")
        if unique_name in gallery_unique_names:
            alias_index = raw_index_from_alias(unique_name, kind="candidate")
            if alias_index != raw_index:
                raise ValueError(
                    f"candidate alias {unique_name!r} does not identify raw row {raw_index}"
                )
            galleries.setdefault(dataset, []).append(
                GalleryCandidate(raw_index=raw_index, image_key=image_key)
            )
            matched_names.add(unique_name)
    missing_queries = query_indices.difference(query_sources)
    if missing_queries:
        raise ValueError(f"train pairs omitted {len(missing_queries)} requested query rows")
    missing_names = gallery_unique_names.difference(matched_names)
    if missing_names:
        raise ValueError(f"gallery allowlist omitted {len(missing_names)} first-seen images")
    return galleries, query_sources


def write_metadata(
    *,
    groups: Sequence[GroupDescriptor],
    galleries: Mapping[str, Sequence[GalleryCandidate]],
    query_sources: Mapping[int, str],
    assets: DenseHCRAssets,
    output_dir: Path,
    annotation_path: Path,
    train_pairs: Path,
    gallery_allowlist_path: Path,
    group_size: int,
    allow_source_outside_gallery: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    partition_cache: dict[tuple[str, float, int], GalleryPartition] = {}
    count_cache: dict[tuple[str, float, int, int, tuple[int, ...], tuple[int, ...]], int] = {}
    totals = np.empty(len(groups), dtype=np.uint32)
    positives = np.empty(len(groups), dtype=np.uint8)
    contributions = np.empty(len(groups), dtype=np.float32)

    for offset, group in enumerate(groups):
        key = (group.dataset, group.pseudo_gallery_fraction, group.partition_seed)
        partition = partition_cache.get(key)
        if partition is None:
            partition = build_partition(
                galleries[group.dataset],
                fraction=group.pseudo_gallery_fraction,
                seed=group.partition_seed,
                minimum_bucket_size=group_size,
            )
            partition_cache[key] = partition
        source_key = query_sources[group.query_index]
        source_outside = False
        try:
            bucket_index = partition.bucket_by_image_key[source_key]
        except KeyError as error:
            if not allow_source_outside_gallery:
                raise ValueError(
                    f"query source for {group.group_id} is outside its gallery"
                ) from error
            bucket_index = stable_u64(
                "outside-source-bucket",
                group.partition_seed,
                group.query_index,
                group.dataset,
            ) % len(partition.buckets)
            source_outside = True
        pool = partition.buckets[bucket_index]
        expected_pool_size = pool.size + int(source_outside)
        if expected_pool_size != group.pseudo_gallery_size:
            raise ValueError(
                f"{group.group_id}: reconstructed pool size {expected_pool_size} != "
                f"annotation {group.pseudo_gallery_size}"
            )
        query_attr = tuple(int(value) for value in assets.query_attr[group.query_index])
        begin = int(assets.query_offsets[group.query_index])
        end = int(assets.query_offsets[group.query_index + 1])
        query_accessories = tuple(int(value) for value in assets.query_values[begin:end])
        cache_key = (*key, bucket_index, source_outside, query_attr, query_accessories)
        total = count_cache.get(cache_key)
        if total is None:
            total = count_relevant_in_pool(
                assets, query_index=group.query_index, candidate_raw_indices=pool
            )
            # An injected query source is, by construction, an exact match for
            # its own scalar/accessory query metadata.
            total += int(source_outside)
            count_cache[cache_key] = total
        head_positive_count = sum(group.labels)
        if total < head_positive_count:
            raise ValueError(
                f"{group.group_id}: total GT {total} < top-K positives {head_positive_count}"
            )
        totals[offset] = total
        positives[offset] = head_positive_count
        contributions[offset] = float(
            ap_numerator(np_to_tensor(group.labels)).item() / total
        )

    max_width = max(len(group.group_id) for group in groups)
    group_ids = np.asarray([group.group_id for group in groups], dtype=f"<U{max_width}")
    group_ids_path = output_dir / "group_ids.npy"
    totals_path = output_dir / "total_relevant.npy"
    positives_path = output_dir / "topk_positive_count.npy"
    contributions_path = output_dir / "retriever_head_ap_contribution.npy"
    np.save(group_ids_path, group_ids)
    np.save(totals_path, totals)
    np.save(positives_path, positives)
    np.save(contributions_path, contributions)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "asset_version": ASSET_VERSION,
        "method": "exact_trainonly_pseudogallery_full_ap_denominator",
        "source_split": "train-only",
        "validation_rows_read": False,
        "test_rows_read": False,
        "annotation_path": str(annotation_path.resolve()),
        "train_pairs": str(train_pairs.resolve()),
        "gallery_unique_names": str(gallery_allowlist_path.resolve()),
        "group_size": group_size,
        "group_count": len(groups),
        "unique_constraint_pool_counts": len(count_cache),
        "full_gallery_ap_identity": (
            "delta_AP=(head_AP_numerator_after-head_AP_numerator_before)/total_relevant"
        ),
        "allow_source_outside_gallery": allow_source_outside_gallery,
        "dense_assets": {
            "group_count": len(groups),
            "group_ids": str(group_ids_path.resolve()),
            "total_relevant_uint32": str(totals_path.resolve()),
            "topk_positive_count_uint8": str(positives_path.resolve()),
            "retriever_head_ap_contribution_float32": str(contributions_path.resolve()),
        },
        "total_relevant_summary": {
            "minimum": int(totals.min()),
            "maximum": int(totals.max()),
            "mean": float(totals.mean()),
            "median": float(np.median(totals)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def np_to_tensor(values: Sequence[int]):
    # Local import keeps the large builder's module import side effects small.
    import torch

    return torch.tensor(values, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--gallery-unique-names", type=Path, required=True)
    parser.add_argument("--hcr-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--allow-source-outside-gallery", action="store_true")
    parser.add_argument("--allow-single-class-groups", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_size <= 1:
        raise SystemExit("group-size must be greater than one")
    for path in (
        args.annotations,
        args.train_pairs,
        args.gallery_unique_names,
        args.hcr_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    groups = list(
        iter_group_descriptors(
            args.annotations,
            group_size=args.group_size,
            allow_single_class_groups=args.allow_single_class_groups,
        )
    )
    if not groups:
        raise ValueError("annotations contain no groups")
    allowlist = {
        value.strip()
        for value in args.gallery_unique_names.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    galleries, query_sources = load_galleries_and_query_sources(
        args.train_pairs,
        gallery_unique_names=allowlist,
        query_indices={group.query_index for group in groups},
    )
    manifest = write_metadata(
        groups=groups,
        galleries=galleries,
        query_sources=query_sources,
        assets=load_dense_hcr_assets(args.hcr_manifest),
        output_dir=args.output_dir,
        annotation_path=args.annotations,
        train_pairs=args.train_pairs,
        gallery_allowlist_path=args.gallery_unique_names,
        group_size=args.group_size,
        allow_source_outside_gallery=args.allow_source_outside_gallery,
    )
    print(manifest)


if __name__ == "__main__":
    main()
