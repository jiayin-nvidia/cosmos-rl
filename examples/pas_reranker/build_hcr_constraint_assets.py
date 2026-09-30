#!/usr/bin/env python3
"""Build a streaming, train-only PAS metadata asset for HCR.

HCR (Hierarchical Constraint-Risk reranking) needs the *query-side* values for
the seven PAS scalar constraints and accessory requirements, as well as the
corresponding *image-side* values.  This tool reads only ``--train-pairs`` and
writes one deduplicated JSONL record per query and image.  It deliberately
does not take validation or test inputs.

The emitted values are enough to reconstruct the eight exact binary targets:
seven wildcard-aware scalar equalities plus one accessory-subset constraint.
``constraint_label_audit.json`` checks, on a bounded streaming sample, that
the target conjunction agrees with :func:`metadata_match`, the source of truth
used by PAS evaluation.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import ijson
import numpy as np

try:
    from examples.pas_reranker.prepare_annotations import as_pair, metadata_match
except ModuleNotFoundError:
    from prepare_annotations import as_pair, metadata_match


SCALAR_FIELDS = (
    "top_outer_color",
    "top_outer_type",
    "bottom_color",
    "bottom_type",
    "shoe_color",
    "shoe_type",
    "viewpoint",
)
ACCESSORY_FIELD = "accessory_subset"
CONSTRAINT_FIELDS = (*SCALAR_FIELDS, ACCESSORY_FIELD)
ASSET_VERSION = 1
_ALIAS_PATTERNS = {
    "query": re.compile(r"^query_(\d+)(?:_.*)?$"),
    "candidate": re.compile(r"^train_(\d+)(?:\.[^.]+)?$"),
}


@dataclass(frozen=True)
class DenseHCRAssets:
    """Memory-mapped, raw-index-addressable training metadata."""

    query_attr: np.ndarray
    image_attr: np.ndarray
    query_offsets: np.ndarray
    query_values: np.ndarray
    image_offsets: np.ndarray
    image_values: np.ndarray
    manifest_path: Path

    def constraint_values(
        self, *, query_alias: str, candidate_alias: str
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        query_index = raw_index_from_alias(query_alias, kind="query")
        candidate_index = raw_index_from_alias(candidate_alias, kind="candidate")
        row_count = int(self.query_attr.shape[0])
        if not 0 <= query_index < row_count:
            raise IndexError(f"HCR query index {query_index} outside [0, {row_count})")
        if not 0 <= candidate_index < row_count:
            raise IndexError(
                f"HCR candidate index {candidate_index} outside [0, {row_count})"
            )

        def accessories(offsets: np.ndarray, values: np.ndarray, index: int):
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            return tuple(int(value) for value in values[begin:end])

        return (
            tuple(int(value) for value in self.query_attr[query_index]),
            tuple(int(value) for value in self.image_attr[candidate_index]),
            accessories(self.query_offsets, self.query_values, query_index),
            accessories(self.image_offsets, self.image_values, candidate_index),
        )


def load_dense_hcr_assets(manifest_path: str | Path) -> DenseHCRAssets:
    """Open a validated HCR asset without materializing it in process memory."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train-only":
        raise ValueError("HCR supervision assets must declare source_split=train-only")
    if manifest.get("validation_rows_read") or manifest.get("test_rows_read"):
        raise ValueError("HCR supervision asset read held-out rows")
    if not manifest.get("audit_proof_passed"):
        raise ValueError("HCR constraint-label audit did not pass")
    dense = manifest.get("dense_assets")
    if not isinstance(dense, dict):
        raise ValueError("HCR manifest has no dense raw-index asset")

    def load(name: str, dtype: np.dtype) -> np.ndarray:
        array_path = Path(str(dense[name]))
        value = np.load(array_path, mmap_mode="r")
        if value.dtype != dtype:
            raise ValueError(
                f"HCR {name} dtype mismatch: expected {dtype}, found {value.dtype}"
            )
        return value

    assets = DenseHCRAssets(
        query_attr=load("query_text_attr_values_int16", np.dtype(np.int16)),
        image_attr=load("image_attr_values_int16", np.dtype(np.int16)),
        query_offsets=load("query_accessory_offsets_uint64", np.dtype(np.uint64)),
        query_values=load("query_accessory_values_int32", np.dtype(np.int32)),
        image_offsets=load("image_accessory_offsets_uint64", np.dtype(np.uint64)),
        image_values=load("image_accessory_values_int32", np.dtype(np.int32)),
        manifest_path=path,
    )
    row_count = int(dense["index_count"])
    if assets.query_attr.shape != (row_count, len(SCALAR_FIELDS)):
        raise ValueError("HCR query scalar array shape mismatch")
    if assets.image_attr.shape != (row_count, len(SCALAR_FIELDS)):
        raise ValueError("HCR image scalar array shape mismatch")
    if assets.query_offsets.shape != (row_count + 1,):
        raise ValueError("HCR query accessory offsets shape mismatch")
    if assets.image_offsets.shape != (row_count + 1,):
        raise ValueError("HCR image accessory offsets shape mismatch")
    if int(assets.query_offsets[-1]) != int(assets.query_values.shape[0]):
        raise ValueError("HCR query accessory offsets do not span values")
    if int(assets.image_offsets[-1]) != int(assets.image_values.shape[0]):
        raise ValueError("HCR image accessory offsets do not span values")
    return assets


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def query_key(row: Mapping) -> str:
    """Stable key for a query's text *and exact logical requirements*.

    PAS can reuse the same natural-language caption for different annotation
    vectors. The vector and accessory set are therefore part of the query
    identity; merging by caption alone would silently corrupt supervision.
    """

    pair = as_pair(row)
    text_values = _require_width(pair.text_attr_values, side="text_attr_values")
    accessories = normalized_ids(pair.text_accessory_ids)

    return "\t".join(
        (
            str(row.get("dataset") or "").strip(),
            str(row.get("query_type") or "").strip(),
            normalized_text(row.get("caption") or row.get("query")),
            ",".join(str(value) for value in text_values),
            ",".join(str(value) for value in accessories),
        )
    )


def image_key(row: Mapping) -> str:
    """Stable key aligned with retrieval candidates' dataset/unique_name IDs."""

    dataset = str(row.get("dataset") or "").strip()
    unique_name = str(row.get("unique_name") or "").strip()
    if not unique_name:
        raise ValueError("PAS row has no unique_name")
    return f"{dataset}\t{unique_name}"


def asset_id(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}"


def raw_index_from_alias(alias: str, *, kind: str) -> int:
    """Resolve deployment annotation aliases to a train-pairs row index.

    ``query_00496293_view_21`` and ``train_00762498.png`` resolve without a
    caption join.  This is intentionally limited to known PAS aliases so a
    malformed identifier cannot select an arbitrary dense-array row.
    """

    if kind not in _ALIAS_PATTERNS:
        raise ValueError(f"unknown alias kind {kind!r}")
    name = Path(str(alias)).name
    match = _ALIAS_PATTERNS[kind].fullmatch(name)
    if match is None:
        raise ValueError(f"invalid {kind} alias {alias!r}")
    return int(match.group(1))


def normalized_ids(values: Iterable[object]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _require_width(values: tuple[int, ...], *, side: str) -> tuple[int, ...]:
    if len(values) != len(SCALAR_FIELDS):
        raise ValueError(
            f"{side} must contain exactly {len(SCALAR_FIELDS)} PAS scalar values; "
            f"got {len(values)}"
        )
    return values


def exact_constraint_targets(
    text_attr_values: Iterable[object],
    image_attr_values: Iterable[object],
    text_accessory_ids: Iterable[object],
    image_accessory_ids: Iterable[object],
) -> tuple[bool, ...]:
    """Return the eight PAS constraint targets in ``CONSTRAINT_FIELDS`` order.

    A negative text scalar is an unspecified field, hence is satisfied by any
    image value.  Accessories use the exact PAS subset rule.
    """

    text = _require_width(tuple(int(value) for value in text_attr_values), side="text")
    image = _require_width(tuple(int(value) for value in image_attr_values), side="image")
    scalar = tuple(wanted < 0 or wanted == observed for wanted, observed in zip(text, image))
    accessories = set(normalized_ids(text_accessory_ids)).issubset(
        normalized_ids(image_accessory_ids)
    )
    return (*scalar, accessories)


def exact_constraint_label(targets: Iterable[bool]) -> bool:
    values = tuple(bool(value) for value in targets)
    if len(values) != len(CONSTRAINT_FIELDS):
        raise ValueError(f"expected {len(CONSTRAINT_FIELDS)} constraint targets")
    return all(values)


def query_record(row: Mapping) -> dict:
    pair = as_pair(row)
    values = _require_width(pair.text_attr_values, side="text_attr_values")
    key = query_key(row)
    return {
        "asset_version": ASSET_VERSION,
        "query_id": asset_id("query", key),
        "dataset": pair.dataset,
        "query_type": pair.query_type,
        "caption": pair.caption,
        "normalized_caption": normalized_text(pair.caption),
        "constraint_signature": {
            "text_attr_values": list(values),
            "text_accessory_ids": list(normalized_ids(pair.text_accessory_ids)),
        },
        "text_attr_values": list(values),
        "text_accessory_ids": list(normalized_ids(pair.text_accessory_ids)),
    }


def image_record(row: Mapping) -> dict:
    pair = as_pair(row)
    values = _require_width(pair.image_attr_values, side="image_attr_values")
    key = image_key(row)
    return {
        "asset_version": ASSET_VERSION,
        "image_id": asset_id("image", key),
        "dataset": pair.dataset,
        "unique_name": pair.unique_name,
        "image_path": pair.image_path,
        "image_attr_values": list(values),
        "image_accessory_ids": list(normalized_ids(pair.image_accessory_ids)),
    }


def canonical_payload(record: Mapping, *, kind: str) -> str:
    ignored = {"caption", "image_path", "normalized_caption"}
    payload = {key: value for key, value in record.items() if key not in ignored}
    payload.pop(f"{kind}_id", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _upsert(conn: sqlite3.Connection, *, kind: str, key: str, record: dict) -> None:
    payload = canonical_payload(record, kind=kind)
    existing = conn.execute(
        "SELECT payload FROM metadata WHERE kind = ? AND source_key = ?", (kind, key)
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO metadata(kind, source_key, payload, record) VALUES (?, ?, ?, ?)",
            (kind, key, payload, json.dumps(record, ensure_ascii=False, separators=(",", ":"))),
        )
    elif existing[0] != payload:
        raise ValueError(f"inconsistent {kind} metadata for key {key!r}")


def write_jsonl_from_db(conn: sqlite3.Connection, *, kind: str, output: Path) -> int:
    partial = output.with_suffix(output.suffix + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for (record,) in conn.execute(
            "SELECT record FROM metadata WHERE kind = ? ORDER BY source_key", (kind,)
        ):
            handle.write(record)
            handle.write("\n")
            count += 1
    partial.replace(output)
    return count


def _write_npy_from_raw(
    raw_path: Path,
    output: Path,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> None:
    """Convert an append-only raw stream to a standard memory-mappable NPY."""

    if int(np.prod(shape, dtype=np.int64)) == 0:
        np.save(output, np.empty(shape, dtype=dtype))
        raw_path.unlink()
        return
    destination = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=shape)
    source = np.memmap(raw_path, mode="r", dtype=dtype, shape=shape)
    if shape[0]:
        chunk_rows = max(1, min(shape[0], 65_536))
        for start in range(0, shape[0], chunk_rows):
            destination[start : start + chunk_rows] = source[start : start + chunk_rows]
    destination.flush()
    del destination
    del source
    raw_path.unlink()


class DenseMetadataWriter:
    """Append raw-index-addressable metadata while the train JSON array streams."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._files = {
            "query_attr": (output_dir / ".query_attr.int16.raw").open("wb"),
            "image_attr": (output_dir / ".image_attr.int16.raw").open("wb"),
            "query_offsets": (output_dir / ".query_offsets.uint64.raw").open("wb"),
            "image_offsets": (output_dir / ".image_offsets.uint64.raw").open("wb"),
            "query_values": (output_dir / ".query_values.int32.raw").open("wb"),
            "image_values": (output_dir / ".image_values.int32.raw").open("wb"),
        }
        self._files["query_offsets"].write(np.asarray([0], dtype=np.uint64).tobytes())
        self._files["image_offsets"].write(np.asarray([0], dtype=np.uint64).tobytes())
        self.rows = 0
        self.query_accessory_count = 0
        self.image_accessory_count = 0
        self.max_accessory_id = -1

    def append(self, raw: Mapping, *, query: Mapping, image: Mapping) -> None:
        index = raw.get("idx")
        if not isinstance(index, int) or index != self.rows:
            raise ValueError(
                "dense raw-index assets require contiguous integer train-pairs idx values; "
                f"expected {self.rows}, got {index!r}"
            )
        np.asarray(query["text_attr_values"], dtype=np.int16).tofile(self._files["query_attr"])
        np.asarray(image["image_attr_values"], dtype=np.int16).tofile(self._files["image_attr"])
        for source, count_name, offset_name, values_name in (
            (query["text_accessory_ids"], "query_accessory_count", "query_offsets", "query_values"),
            (image["image_accessory_ids"], "image_accessory_count", "image_offsets", "image_values"),
        ):
            values = tuple(int(value) for value in source)
            if values and (min(values) < 0 or max(values) > np.iinfo(np.int32).max):
                raise ValueError("accessory IDs must fit non-negative int32")
            if values:
                np.asarray(values, dtype=np.int32).tofile(self._files[values_name])
                self.max_accessory_id = max(self.max_accessory_id, max(values))
            new_count = getattr(self, count_name) + len(values)
            setattr(self, count_name, new_count)
            np.asarray([new_count], dtype=np.uint64).tofile(self._files[offset_name])
        self.rows += 1

    def finish(self) -> dict:
        for handle in self._files.values():
            handle.close()
        specs = (
            ("query_attr", np.dtype(np.int16), (self.rows, len(SCALAR_FIELDS))),
            ("image_attr", np.dtype(np.int16), (self.rows, len(SCALAR_FIELDS))),
            ("query_offsets", np.dtype(np.uint64), (self.rows + 1,)),
            ("image_offsets", np.dtype(np.uint64), (self.rows + 1,)),
            ("query_values", np.dtype(np.int32), (self.query_accessory_count,)),
            ("image_values", np.dtype(np.int32), (self.image_accessory_count,)),
        )
        outputs: dict[str, str] = {}
        for name, dtype, shape in specs:
            raw_path = self.output_dir / f".{name}.{dtype.name}.raw"
            output = self.output_dir / f"{name}.npy"
            _write_npy_from_raw(raw_path, output, dtype=dtype, shape=shape)
            outputs[name] = str(output.resolve())
        width = max(self.max_accessory_id + 1, 0)
        return {
            "index_count": self.rows,
            "query_alias_pattern": "query_<raw_idx>[_view_<n>]",
            "candidate_alias_pattern": "train_<raw_idx>.<extension>",
            "query_text_attr_values_int16": outputs["query_attr"],
            "image_attr_values_int16": outputs["image_attr"],
            "query_accessory_offsets_uint64": outputs["query_offsets"],
            "query_accessory_values_int32": outputs["query_values"],
            "image_accessory_offsets_uint64": outputs["image_offsets"],
            "image_accessory_values_int32": outputs["image_values"],
            "max_accessory_id": self.max_accessory_id,
            "fixed_bitset_width_bits": width,
            "fixed_bitset_practical_under_256_bits": width <= 256,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pairs",
        type=Path,
        required=True,
        help="The sole data input. It must be the PAS train split.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Skip deduplicated JSONL records and emit only raw-index dense NPY assets.",
    )
    parser.add_argument(
        "--audit-sample-limit",
        type=int,
        default=4096,
        help="Maximum number of streamed rows used for the exact-label audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_sample_limit <= 0:
        raise ValueError("--audit-sample-limit must be positive")
    if not args.train_pairs.is_file():
        raise FileNotFoundError(args.train_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries_output = args.output_dir / "hcr_queries.jsonl"
    images_output = args.output_dir / "hcr_images.jsonl"
    audit_output = args.output_dir / "constraint_label_audit.json"
    manifest_output = args.output_dir / "manifest.json"
    dense_outputs = tuple(
        args.output_dir / name
        for name in (
            "query_attr.npy",
            "image_attr.npy",
            "query_offsets.npy",
            "image_offsets.npy",
            "query_values.npy",
            "image_values.npy",
        )
    )
    outputs = (queries_output, images_output, audit_output, manifest_output, *dense_outputs)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing HCR asset: " + ", ".join(existing))

    database = args.output_dir / ".hcr_metadata.sqlite3"
    if not args.dense_only and database.exists():
        raise FileExistsError(f"temporary metadata database already exists: {database}")
    conn = None if args.dense_only else sqlite3.connect(database)
    if conn is not None:
        conn.execute(
            "CREATE TABLE metadata (kind TEXT NOT NULL, source_key TEXT NOT NULL, "
            "payload TEXT NOT NULL, record TEXT NOT NULL, PRIMARY KEY(kind, source_key))"
        )
    dense = DenseMetadataWriter(args.output_dir)

    rows = 0
    positives = 0
    per_constraint = Counter()
    audit_checked = 0
    conjunction_mismatches = 0
    metadata_match_mismatches = 0
    source_label_checked = 0
    source_label_mismatches = 0
    try:
        with args.train_pairs.open("rb") as handle:
            for raw in ijson.items(handle, "item"):
                rows += 1
                query = query_record(raw)
                image = image_record(raw)
                if conn is not None:
                    _upsert(conn, kind="query", key=query_key(raw), record=query)
                    _upsert(conn, kind="image", key=image_key(raw), record=image)
                dense.append(raw, query=query, image=image)

                targets = exact_constraint_targets(
                    query["text_attr_values"],
                    image["image_attr_values"],
                    query["text_accessory_ids"],
                    image["image_accessory_ids"],
                )
                label = exact_constraint_label(targets)
                positives += int(label)
                per_constraint.update(
                    name for name, satisfied in zip(CONSTRAINT_FIELDS, targets) if satisfied
                )
                if "label" in raw and raw["label"] is not None:
                    source_label_checked += 1
                    source_label_mismatches += int(bool(int(raw["label"])) != label)
                if audit_checked < args.audit_sample_limit:
                    audit_checked += 1
                    conjunction_mismatches += int(label != all(targets))
                    pair = as_pair(raw)
                    metadata_match_mismatches += int(
                        label != metadata_match(pair, pair, accessories=True)
                    )
                if conn is not None and rows % 10_000 == 0:
                    conn.commit()
        if conn is not None:
            conn.commit()
            queries = write_jsonl_from_db(conn, kind="query", output=queries_output)
            images = write_jsonl_from_db(conn, kind="image", output=images_output)
        else:
            queries = 0
            images = 0
        dense_manifest = dense.finish()
    finally:
        if conn is not None:
            conn.close()
            database.unlink(missing_ok=True)

    # ``metadata_match(pair, pair)`` intentionally compares the row's query
    # metadata to its image metadata, so it is a valid independent statement
    # of the same exact PAS rule.  No model, SigLIP score, or held-out row is
    # consulted here.
    audit = {
        "asset_version": ASSET_VERSION,
        "source_split": "train-only",
        "validation_rows_read": False,
        "test_rows_read": False,
        "pairs_streamed": rows,
        "audit_sample_limit": args.audit_sample_limit,
        "audit_pairs_checked": audit_checked,
        "constraint_fields": list(CONSTRAINT_FIELDS),
        "label_definition": "all(eight constraint_targets)",
        "conjunction_mismatches": conjunction_mismatches,
        "metadata_match_mismatches": metadata_match_mismatches,
        "proof_passed": conjunction_mismatches == 0 and metadata_match_mismatches == 0,
        "exact_positive_pairs": positives,
        "exact_negative_pairs": rows - positives,
        "satisfied_counts_by_constraint": {
            name: per_constraint[name] for name in CONSTRAINT_FIELDS
        },
        "source_label_checked": source_label_checked,
        "source_label_mismatches": source_label_mismatches,
        "dense_index_count": dense.rows,
    }
    audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "asset_version": ASSET_VERSION,
        "method": "streaming_train_only_hcr_query_image_metadata",
        "source_split": "train-only",
        "validation_rows_read": False,
        "test_rows_read": False,
        "train_pairs": str(args.train_pairs.resolve()),
        "constraint_fields": list(CONSTRAINT_FIELDS),
        "dense_only": args.dense_only,
        "queries": str(queries_output.resolve()) if not args.dense_only else None,
        "images": str(images_output.resolve()) if not args.dense_only else None,
        "query_records": queries,
        "image_records": images,
        "constraint_label_audit": str(audit_output.resolve()),
        "audit_proof_passed": audit["proof_passed"],
        "audit_pairs_checked": audit_checked,
        "dense_assets": dense_manifest,
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
