# PAS baseline evaluator package

This is the PAS evaluation code copied from
`/workspace/jiayin/rerank_experiment` on 2026-09-29. Its Git HEAD was
`c06e589` with local PAS evaluation modifications; the copied files in this
branch are the complete evaluation source used by the reproduction wrappers.
The package includes the CR3 Nano vLLM reranker, the SigLIP2 checkpoint
loader, the deduplicated PAS three-mode evaluator, and sharded run/merge
scripts. Embedding caches, exported PAS images, and large result traces are
external inputs documented in `examples/pas_reranker/README.md`.

Run the wrapper scripts at the Cosmos-RL repository root. They supply the
paths and protocol options used for the reference table.
