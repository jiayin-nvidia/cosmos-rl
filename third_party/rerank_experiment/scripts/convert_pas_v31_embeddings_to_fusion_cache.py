#!/usr/bin/env python3
"""Convert PAS V3.1 row-aligned embedding arrays to fusion-search-utils cache pickles.

``evaluate_pas_three_modes_from_cache.py`` in fusion-search-utils expects:

* image embeddings as a dict keyed by ``"<dataset>\t<image_path>"``
* text embeddings as a dict keyed by exact caption text

The evaluator I originally used wrote row-aligned NumPy arrays instead:

* ``image_embeddings.npy`` rows correspond to first-seen unique images in
  ``test_pairs.json`` order
* ``text_embeddings.npy`` rows correspond to every pair/query row in
  ``test_pairs.json`` order

This script preserves the conversion step so the fusion-search-utils evaluator
can be used directly without keeping the local standalone evaluator code.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS_FILE = REPO_ROOT / "artifacts" / "pas_v31_test_tao" / "test_pairs.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "pas_v31_step01953" / "cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "pas_v31_step01953" / "fusion_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS_FILE)
    parser.add_argument(
        "--image-embeddings",
        type=Path,
        default=DEFAULT_CACHE_DIR / "image_embeddings.npy",
        help="Row-aligned image embedding array.",
    )
    parser.add_argument(
        "--text-embeddings",
        type=Path,
        default=DEFAULT_CACHE_DIR / "text_embeddings.npy",
        help="Row-aligned text embedding array.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--image-output-name",
        default="test_pairs_source_image_embeddings.pkl",
        help="Output filename expected by the fusion-search-utils PAS evaluator.",
    )
    parser.add_argument(
        "--text-output-name",
        default="test_pairs_text_embeddings_lower.pkl",
        help="Output filename expected by the fusion-search-utils PAS evaluator.",
    )
    parser.add_argument(
        "--duplicate-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance when validating duplicate caption embeddings.",
    )
    parser.add_argument(
        "--duplicate-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance when validating duplicate caption embeddings.",
    )
    return parser.parse_args()


def iter_json_records(path: Path) -> Iterator[Mapping[str, Any]]:
    """Iterate ordinary JSON arrays and compact one-record-per-line arrays."""

    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        second = handle.readline()

    compact_line_mode = second.lstrip().startswith("{") and second.rstrip().rstrip(",").endswith(
        "}"
    )
    if compact_line_mode:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped in {"[", "]"}:
                    continue
                if stripped.endswith(","):
                    stripped = stripped[:-1]
                if stripped:
                    yield json.loads(stripped)
        return

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected JSON array in {path}, got {type(records).__name__}")
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"Expected pair record dict in {path}, got {type(record).__name__}")
        yield record


def infer_dataset(image_path: str, explicit_dataset: str = "") -> str:
    if explicit_dataset:
        return explicit_dataset.strip()
    normalized = image_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    for marker in ("images", "data"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1].strip()
    if len(parts) > 1:
        return parts[0].strip()
    return ""


def image_key(record: Mapping[str, Any]) -> str:
    image_path = str(record.get("image_path") or "").replace("\\", "/").strip()
    dataset = infer_dataset(image_path, str(record.get("dataset") or ""))
    if not dataset or not image_path:
        raise ValueError(f"Cannot build image key from record: {record!r}")
    return f"{dataset}\t{image_path}"


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    pairs_file = args.pairs_file.expanduser().resolve()
    image_embeddings_path = args.image_embeddings.expanduser().resolve()
    text_embeddings_path = args.text_embeddings.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not pairs_file.is_file():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    if not image_embeddings_path.is_file():
        raise FileNotFoundError(f"Image embeddings not found: {image_embeddings_path}")
    if not text_embeddings_path.is_file():
        raise FileNotFoundError(f"Text embeddings not found: {text_embeddings_path}")

    records = list(iter_json_records(pairs_file))
    image_embeddings = np.load(image_embeddings_path, mmap_mode="r")
    text_embeddings = np.load(text_embeddings_path, mmap_mode="r")

    if text_embeddings.shape[0] != len(records):
        raise ValueError(
            "Text embedding row count must match test_pairs.json row count: "
            f"{text_embeddings.shape[0]} != {len(records)}"
        )

    image_cache: dict[str, np.ndarray] = {}
    text_cache: dict[str, np.ndarray] = {}
    next_image_row = 0

    for query_row, record in enumerate(records):
        key = image_key(record)
        if key not in image_cache:
            if next_image_row >= image_embeddings.shape[0]:
                raise ValueError("Image embedding array has fewer rows than unique pair images")
            image_cache[key] = np.asarray(image_embeddings[next_image_row], dtype=np.float32)
            next_image_row += 1

        caption = str(record.get("caption") or "").strip()
        if not caption:
            continue
        text_value = np.asarray(text_embeddings[query_row], dtype=np.float32)
        existing = text_cache.get(caption)
        if existing is not None:
            if not np.allclose(
                existing,
                text_value,
                atol=args.duplicate_atol,
                rtol=args.duplicate_rtol,
            ):
                raise ValueError(f"Duplicate caption has non-matching embedding: {caption!r}")
            continue
        text_cache[caption] = text_value

    if next_image_row != image_embeddings.shape[0]:
        raise ValueError(
            "Image embedding row count must match unique image count in test_pairs.json: "
            f"{image_embeddings.shape[0]} != {next_image_row}"
        )

    image_output = output_dir / args.image_output_name
    text_output = output_dir / args.text_output_name
    atomic_pickle(image_output, image_cache)
    atomic_pickle(text_output, text_cache)

    metadata = {
        "pairs_file": str(pairs_file),
        "image_embeddings": str(image_embeddings_path),
        "text_embeddings": str(text_embeddings_path),
        "image_output": str(image_output),
        "text_output": str(text_output),
        "num_pair_rows": len(records),
        "num_image_embeddings": len(image_cache),
        "num_text_embeddings": len(text_cache),
        "image_embedding_shape": list(image_embeddings.shape),
        "text_embedding_shape": list(text_embeddings.shape),
        "image_key_format": "<dataset>\\t<image_path>",
        "text_key_format": "exact caption text",
    }
    metadata_path = output_dir / "fusion_cache_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {image_output} entries={len(image_cache):,}")
    print(f"Wrote {text_output} entries={len(text_cache):,}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
