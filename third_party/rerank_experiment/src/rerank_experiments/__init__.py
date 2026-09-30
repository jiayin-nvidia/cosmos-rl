"""rerank_experiments: text-to-video retrieval experiments on the VSS KPI search datasets.

The package evaluates a two-stage retrieval pipeline on temporally-annotated video
search datasets:

    stage 1  embedder  -- bi-encoder that maps queries and video segments into a
                          shared space and retrieves the top-K candidates.
    stage 2  reranker  -- (optional) re-scores the top-K candidates.

The current release implements the **embedder-only baseline**. The pipeline,
configuration and metric stack are built so that adding -- and later
fine-tuning -- a reranker is purely additive (see
:mod:`rerank_experiments.rerankers`).
"""

import os as _os

# We only use the (torch-free) tokenizer from transformers; silence its noisy
# "PyTorch was not found" advisory and downgrade logging before it is imported.
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

__version__ = "0.1.0"
