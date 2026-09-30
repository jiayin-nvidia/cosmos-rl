#!/usr/bin/env python3
"""Reassemble and verify the Git-friendly step-8000 language adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARTS = sorted((ROOT / "language").glob("adapter_model.safetensors.part-*"))
OUTPUT = ROOT / "language" / "adapter_model.safetensors"
EXPECTED_SHA256 = "c6c994316eb1a727f349aa6ad0b6a84c52a88c9858fb952f95b8a9d6d341e750"
VISUAL_SHA256 = "a5e4724d5212d5a1a52df0f957d650e7988b7b23bfe4d3540fde1db402a0caaa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(PARTS) != 2:
        raise SystemExit(f"Expected two language adapter parts; found {len(PARTS)}")
    visual = ROOT / "visual" / "adapter_model.safetensors"
    if sha256(visual) != VISUAL_SHA256:
        raise SystemExit(f"Visual adapter checksum mismatch: {visual}")
    if not OUTPUT.exists() or sha256(OUTPUT) != EXPECTED_SHA256:
        temporary = OUTPUT.with_suffix(".safetensors.tmp")
        with temporary.open("wb") as destination:
            for part in PARTS:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(block)
        if sha256(temporary) != EXPECTED_SHA256:
            temporary.unlink()
            raise SystemExit("Language adapter checksum mismatch after reconstruction")
        temporary.replace(OUTPUT)
    print(f"Verified language adapter: {OUTPUT}")
    print(f"Verified visual adapter: {visual}")


if __name__ == "__main__":
    main()
