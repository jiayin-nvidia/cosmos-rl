#!/usr/bin/env python3
"""Generate full PAS V3.1 caches with public zero-shot Google SigLIP2.

The output arrays use the row ordering expected by
``convert_pas_v31_embeddings_to_fusion_cache.py``:

* image_embeddings.npy: first-seen unique image order in test_pairs.json
* text_embeddings.npy: one row per test_pairs.json query row
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_pas_v31_siglip2_embeddings import load_pas_rows  # noqa: E402


DEFAULT_MODEL = REPO_ROOT / "models" / "siglip2-so400m-patch16-256"
DEFAULT_PAIRS = REPO_ROOT / "artifacts" / "pas_v31_test_tao" / "test_pairs.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "pas_v31_public_siglip2" / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()
    if args.image_batch_size <= 0 or args.text_batch_size <= 0:
        parser.error("Batch sizes must be positive")
    if args.skip_images and args.skip_text:
        parser.error("At least one of image or text embeddings must be generated")
    return args


def pooled(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if getattr(value, "pooler_output", None) is not None:
        return value.pooler_output
    if isinstance(value, (tuple, list)) and len(value) > 1:
        return value[1]
    raise TypeError(f"Cannot extract pooled features from {type(value).__name__}")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class PublicSiglip2:
    def __init__(self, model_path: Path, device: str, dtype: str) -> None:
        self.device = torch.device(device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        text_config = self.model.config.text_config
        self.embedding_dim = int(
            getattr(text_config, "projection_size", None)
            or getattr(text_config, "hidden_size")
        )

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        opened: list[Image.Image] = []
        try:
            for path in paths:
                opened.append(Image.open(path).convert("RGB"))
            inputs = self.processor(images=opened, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            with torch.inference_mode():
                values = pooled(self.model.get_image_features(pixel_values=pixel_values))
            return F.normalize(values.float(), dim=-1).cpu().numpy()
        finally:
            for image in opened:
                image.close()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=[text.lower() for text in texts],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            values = pooled(self.model.get_text_features(**inputs))
        return F.normalize(values.float(), dim=-1).cpu().numpy()


def write_manifest(
    output_dir: Path,
    name: str,
    *,
    shape: tuple[int, int],
    row_order: str,
    args: argparse.Namespace,
) -> None:
    atomic_json(
        output_dir / f"{name}.json",
        {
            "format_version": 1,
            "array": str(output_dir / f"{name}.npy"),
            "shape": list(shape),
            "dtype": "float32",
            "model": str(args.model),
            "model_id": "google/siglip2-so400m-patch16-256",
            "pairs_file": str(args.pairs_file),
            "row_order": row_order,
            "encoder": {
                "class": "transformers.AutoModel",
                "device": args.device,
                "dtype": args.dtype,
                "image_batch_size": args.image_batch_size,
                "text_batch_size": args.text_batch_size,
                "text_case": "lower",
                "text_max_length": 64,
                "normalization": "l2",
            },
        },
    )


def generate_images(
    encoder: PublicSiglip2,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    path = args.output_dir / "image_embeddings.npy"
    prepare_output(path, args.overwrite)
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), encoder.embedding_dim),
    )
    for start in tqdm(
        range(0, len(rows), args.image_batch_size),
        desc="Public SigLIP2 PAS images",
    ):
        batch = rows[start : start + args.image_batch_size]
        paths = [Path(row["path"]) for row in batch]
        missing = [str(item) for item in paths if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing exported image: {missing[0]}")
        values = encoder.embed_images(paths)
        array[start : start + len(batch)] = values
        array.flush()
    write_manifest(
        args.output_dir,
        "image_embeddings",
        shape=array.shape,
        row_order="first-seen unique <dataset>\\t<image_path> in test_pairs.json order",
        args=args,
    )


def generate_texts(
    encoder: PublicSiglip2,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    path = args.output_dir / "text_embeddings.npy"
    prepare_output(path, args.overwrite)
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), encoder.embedding_dim),
    )
    for start in tqdm(
        range(0, len(rows), args.text_batch_size),
        desc="Public SigLIP2 PAS captions",
    ):
        batch = rows[start : start + args.text_batch_size]
        values = encoder.embed_texts([str(row["caption"]) for row in batch])
        array[start : start + len(batch)] = values
        array.flush()
    write_manifest(
        args.output_dir,
        "text_embeddings",
        shape=array.shape,
        row_order="one row per test_pairs.json query row",
        args=args,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.model = args.model.expanduser().resolve()
    args.pairs_file = args.pairs_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root is not None
        else args.pairs_file.parent / "images"
    )
    if not args.model.is_dir():
        raise FileNotFoundError(f"Public model directory not found: {args.model}")
    if not args.pairs_file.is_file():
        raise FileNotFoundError(f"Pairs file not found: {args.pairs_file}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    images, queries = load_pas_rows(args.pairs_file, args.image_root)
    logging.info("Loaded %s unique images and %s queries", f"{len(images):,}", f"{len(queries):,}")
    encoder = PublicSiglip2(args.model, args.device, args.dtype)
    logging.info("Loaded public SigLIP2 with embedding dimension %s", encoder.embedding_dim)

    if not args.skip_images:
        generate_images(encoder, images, args)
    if not args.skip_text:
        generate_texts(encoder, queries, args)
    logging.info("Wrote public PAS arrays under %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
