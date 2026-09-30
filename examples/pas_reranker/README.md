# PAS V3.1 Cosmos/CR3 Nano reranker reproduction

This branch starts at Cosmos-RL `89b1fdd` (the common base of
`jn/cr3_reranker_ft` and `main`). It contains the PAS trainer extensions,
the baseline reranker evaluator under `third_party/rerank_experiment`, and
the selected CR3 Nano step-8000 language and visual LoRA checkpoint. The
reference table is in `reports/pas_cr3_nano_reference_table.csv`. The original
Cosmos-RL configurations, model support, tests, and documentation are preserved.
The vendored reranker package includes the CR3 components used here.

## Evaluation contract and inputs

The table measures `scalar_plus_accessories` relevance on the **deduplicated
full PAS test**: 22,190 easy, 23,559 medium, and 23,884 hard queries across six
datasets. Queries are deduplicated by `(dataset, query_type, caption)` and their
ground-truth sets are unioned. The gallery and query pairs are in
`/workspace/jiayin/rerank_experiment/artifacts/pas_v31_test_tao/test_pairs.json`;
images are in its `images/` directory. Evaluation reranks the SigLIP2 top 20
and reports full-gallery AP, Rank-1, and Rank-5. `K=5` is the Rank-5 metric,
not the rerank depth. The evaluator and merge script are included in
`third_party/rerank_experiment/scripts/`.

For each query and candidate image, CR3 is prompted to continue after
`<answer>` with one decision token. The deployed scalar score is
`log P(yes) - log P(no)`, equivalently `logit(yes) - logit(no)` because the
shared normalization cancels. The evaluator requests both token log
probabilities and sorts the SigLIP2 top 20 by this margin; a larger value
means a stronger match. The remaining gallery order is unchanged. This is
the evaluator's `logit_delta` mode, implemented in the vendored
`rerank_experiments/rerankers/vlm/common.py`.

| Input | Exact source used for the table |
| --- | --- |
| Public SigLIP2 embeddings | `/workspace/jiayin/rerank_experiment/artifacts/pas_v31_public_siglip2/` |
| Fine-tuned SigLIP2 embeddings | `/workspace/jiayin/rerank_experiment/artifacts/pas_v31_step03906_cache/` |
| Fine-tuned SigLIP2 retriever weights | `/workspace/jiayin/models/model_epoch_001_step_03906.pth` |
| Public CR3 Nano | `nvidia/Cosmos3-Nano` |
| CR3 Nano base for the fine-tuned adapter | `/workspace/jiayin/models/Cosmos3-Nano-VLM` |
| PAS training images and pairs | `/workspace/data/PAS_Datasets/NVIDIA_PAS_Filtered_06152026_V3.1_tao_ft_metadata_base_plus_val_test_medium_attr_matched_aug_val20_accessory_v2.1/` |
| Training K=20 annotations | `/workspace/jiayin/cosmos-rl/data/pas_cr3_step03906/train_deployed_top20_k20_{trainonly750k_seed880821,remaining_after750k_seed880822}.json` |
| Structured attribute replay records | `/workspace/jiayin/cosmos-rl/outputs/pas_cr3_attribute_replay_records_alltrain/` |
| Block-24 visual cache | `/workspace/jiayin/cache/pas_cr3_block24_prefix/` (specific manifests in both training TOMLs) |

The two annotation manifests are included in `recipe_manifests/`. They record
669,871 and 1,017,346 K=20 groups, respectively. The retriever determines
candidate order; PAS metadata determines relevance targets. The paired
public/fine-tuned embedding generation scripts are included in the vendored
package. Its `pyproject.toml` records the evaluation dependencies. On this
host, the working evaluator is
`/workspace/jiayin/rerank_experiment/.venv/bin/python` with vLLM 0.23.0,
Transformers 5.14.1, and Torch 2.11.0+cu130; the training environment is
`/opt/venv/cosmos_rl`. The full reranker run uses eight A100 80 GB GPUs.

On a new host, install the vendored evaluator and its VLM dependencies in a
Python environment, then set `RERANK_PYTHON` to that environment's Python:

```bash
python3 -m venv third_party/rerank_experiment/.venv
third_party/rerank_experiment/.venv/bin/pip install -e 'third_party/rerank_experiment[vlm]'
export RERANK_PYTHON="$PWD/third_party/rerank_experiment/.venv/bin/python"
```

For training or rebuilding the mined data and cache on another host, start
with CUDA-compatible PyTorch and FlashAttention wheels for that host, then
install this repository's training extra. The working environment here has
Torch `2.11.0a0+nv26.3`, Transformers `4.57.5`, FlashAttention `2.7.4`,
PEFT `0.17.1`, Safetensors `0.8.0`, and W&B `0.28.1`.

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install -e '.[train]'
export TRAIN_PYTHON_BIN="$PWD/.venv-train/bin/python"
export COSMOS_RL_BIN="$PWD/.venv-train/bin/cosmos-rl"
export PYTHON_BIN="$TRAIN_PYTHON_BIN"
export TORCHRUN_BIN="$PWD/.venv-train/bin/torchrun"
```

The training and evaluation wrappers honor these interpreter overrides.
The batch submitters forward the evaluator interpreter and input-path
overrides to their remote workers. Install the evaluator environment in a
location visible on the shared mount before submitting a Lepton job.
Set `CUDA_HOME` to the installed CUDA toolkit if it differs from this host's
`/usr/local/cuda-13.2`. The large PAS data, CR3 base weights, and SigLIP2
retriever checkpoint remain external inputs at the paths below.

The inference wrapper accepts `SHARDS` and a comma-separated `GPUS` list to
spread independent query shards across more GPUs. The recorded runs use
`SHARDS=8` and `GPUS=0,1,2,3,4,5,6,7`. Check
`recipe_manifests/source_checksums.json` when transferring the external PAS
data and model files.

## Mine training pairs and build the visual cache

`prepare_training_data.sh` runs the upstream recipe. It accepts `indices`,
`embeddings`, `mine`, `attributes`, `cache`, or `all`, and writes new artifacts
under ignored `reproduction/data/` and `reproduction/cache/` by default.
The checked-in `recipe_manifests/` directory contains the original query
selection, mining, and attribute manifests plus the exact two cache image
lists. Override `PAS_DATA_ROOT`, `RETRIEVER_CHECKPOINT`, `SIGLIP_TOKENIZER`,
`VAL_CAPTION_DENYLIST`, `CR3_BASE_MODEL`, and output roots in the script's
environment when working on another host. Eight GPUs reproduce the recorded
distributed shard layout.

```bash
bash examples/pas_reranker/prepare_training_data.sh all
```

The stages are:

1. Read the 1,999,953 train-pair rows. Deny captions present in the fixed
   Val999 query subset without reading its labels or metrics. Reservoir
   sample 250,000 queries from each of easy, medium, and hard with seed
   `880821`. Select all remaining eligible, query-disjoint rows with seed
   `880822`. The original index manifests record 750,000 and 1,131,764
   selected rows.
2. Encode every first-seen unique train image with the step-03906 SigLIP2
   retriever (222,217 image embeddings, 1,152 dimensions); encode selected
   query text into eight global-row-aligned shards for each selection. The
   retriever checkpoint is
   `/workspace/jiayin/models/model_epoch_001_step_03906.pth`; the tokenizer
   is `/workspace/jiayin/models/siglip2-so400m-patch16-256-tokenizer`.
   `TRAIN_IMAGE_EMBEDDINGS` may point to the existing verified gallery array
   to skip its extraction.
3. Search each query against its own dataset gallery by dot product and take
   SigLIP2 top 20. Match PAS scalar attributes and accessory IDs for binary
   labels. The first selection keeps the deployed top 20 in rank order when
   both positive and negative candidates exist; it yielded 669,871 K20 groups
   and 13,397,420 examples. The second selection uses the earliest positive and
   negative anchors and rank-spanning candidates. With top 20 and one view,
   it preserves the entire rank-ordered top 20 when both classes exist; it
   yielded 1,017,346 groups and 20,346,920 examples. Retriever scores are
   candidate ordering metadata, not labels. The miner's
   `--deployed-topk-group` mode reproduces the first stage's saved record
   schema; its default mode reproduces the second.
4. Enumerate unique train and validation images to make the structured
   attribute replay records. The source files contain 222,217 training and
   8,138 validation records, with zero overlapping image identities. The
   included `build_attribute_replay_records.py` reproduced both original
   record files byte for byte (SHA-256 `e2ab5b98...` train and
   `af048644...` validation).
5. Run `build_visual_cache.py` on eight GPUs for the 217,743 primary and
   4,474 supplementary images in `recipe_manifests/cache_*_images.txt`.
   The original caches use BF16, visual prefix block 24 plus deep-stack
   features, max 81,920 pixels, batch size 32, and the linear patch
   projection. All 222,217 train images are covered by the two lists, with
   no overlap. The supplement was originally built from a temporary clean
   copy of the same CR3 base; the script uses the local clean base for both.

Each stage can also be run separately with `prepare_training_data.sh
indices`, `embeddings`, `mine`, `attributes`, or `cache`. The outputs and
their consumers are:

| Stage | Output under the default roots | Used by |
| --- | --- | --- |
| `indices` | `reproduction/data/step03906/query_indices_trainonly_{balanced250k_seed880821,remaining_after750k_seed880822}.json` | Selects the two disjoint training query sets for text extraction and mining. |
| `embeddings` | `reproduction/data/step03906_embeddings/trainonly_balanced750k_seed880821/image_embeddings.npy` and eight row-aligned text shards in each selection directory | The miner searches the full per-dataset image gallery for each selected query. |
| `mine` | `reproduction/data/step03906/train_deployed_top20_k20_{trainonly750k_seed880821,remaining_after750k_seed880822}.json` plus `.manifest.json` files | The two K20 train datasets in both replay TOMLs. |
| `attributes` | `reproduction/data/step03906/attribute_replay/{train,val}_records.json` | The structured attribute auxiliary loss. |
| `cache` | `reproduction/cache/pas_cr3_block24_prefix_step03906_{trainonly750k_star,top50_supplement4474_clean}/manifest_rank00.json` through `manifest_rank07.json`, with BF16 `.safetensors` shards | The frozen vision prefix inputs loaded by the trainer. |

The cache key is the SHA-1 of each image's canonical absolute path. Each
rank takes every eighth image from its ordered input list and writes one
manifest entry per image, pointing to its shard, token offsets, lengths, and
`grid_thw`. The shard stores `prefix_features` immediately before visual
block 24 and the earlier `deepstack_features`; the trainer resumes the last
three visual blocks and merger from these tensors. Keep the same image-root
symlink targets or regenerate the cache: changing the canonical absolute
image targets changes its keys. The PAS `images/` export contains symlinks to
original assets, and the cache keys resolve those links. The two cache sets
use equivalent feature-layout names, which `visual_cache.py` canonicalizes.

Check the original input and output SHA-256 values in
`recipe_manifests/source_checksums.json`, the expected mined group counts in
the two checked-in mining manifests, and the two image-list hashes and per-rank
cache counts in `recipe_manifests/cache_recipe_summary.json`. The JSON mining
records include the dataset, query, candidate image, yes/no label, SigLIP2
rank and score, and exact CR3 prompt/answer; the score is metadata and is not
used as a training target. CUDA top-k tie behavior can change a few candidate
orders across hardware, so source checksums are a provenance reference rather
than a promise of byte-identical regenerated mining output. The first 1,000
saved K20 groups passed an independent metadata-label audit; full mining and
cache regeneration were also run on eight GPUs, with the results below.

The checked-in training TOMLs point at the original annotations, attribute
records, and cache manifests listed above. To train from freshly generated
data, render parallel TOMLs after `prepare_training_data.sh all`:

```bash
python examples/pas_reranker/render_fresh_training_configs.py \
  --data-root reproduction/data/step03906 \
  --cache-root reproduction/cache \
  --output-dir reproduction/finetune/generated_configs
```

Then pass the generated TOMLs to the two `cosmos-rl` commands below. The
renderer also accepts `--pas-data-root` and `--cr3-base-model` for moved
source data or weights. The source PAS image export and 13.6 GB retriever
checkpoint are external inputs; the large generated annotations and cache
tensors are not committed to Git.
Both rendered TOMLs were parsed and all 20 referenced annotation, attribute,
and cache manifest paths per TOML were verified to exist after regeneration.

If the two embedding caches are unavailable, regenerate them from the
exported test pairs with the vendored scripts. Run SigLIP2 extraction in a
Transformers 4.57.1 environment, then return to the vLLM environment for CR3
scoring. For example, from this repository root:

```bash
repo="$PWD"
eval_repo="$repo/third_party/rerank_experiment"
pairs=/workspace/jiayin/rerank_experiment/artifacts/pas_v31_test_tao/test_pairs.json
images=/workspace/jiayin/rerank_experiment/artifacts/pas_v31_test_tao/images
hf download google/siglip2-so400m-patch16-256 \
  --local-dir "$repo/reproduction/public_siglip2_model"
cd "$eval_repo"
python scripts/generate_pas_v31_public_siglip2_embeddings.py \
  --model "$repo/reproduction/public_siglip2_model" \
  --pairs-file "$pairs" --image-root "$images" \
  --output-dir "$repo/reproduction/public_arrays" --dtype float16
python scripts/generate_pas_v31_siglip2_embeddings.py \
  --pairs-file "$pairs" --image-root "$images" \
  --checkpoint /workspace/jiayin/models/model_epoch_001_step_03906.pth \
  --tokenizer-dir /workspace/jiayin/models/siglip2-so400m-patch16-256-tokenizer \
  --output-dir "$repo/reproduction/finetuned_arrays" --dtype float16
for name in public finetuned; do
  python scripts/convert_pas_v31_embeddings_to_fusion_cache.py \
    --pairs-file "$pairs" \
    --image-embeddings "$repo/reproduction/${name}_arrays/image_embeddings.npy" \
    --text-embeddings "$repo/reproduction/${name}_arrays/text_embeddings.npy" \
    --output-dir "$repo/reproduction/${name}_cache"
done
```

Set `PUBLIC_SIGLIP_CACHE` and `FINETUNED_SIGLIP_CACHE` to those cache
directories when running `reproduce_full_test.sh`.

## Reproduce with the supplied checkpoint

From this repository root:

```bash
python3 checkpoints/cr3_nano_step8000/materialize.py
bash examples/pas_reranker/reproduce_full_test.sh finetuned_step8000
```

`materialize.py` reconstructs the 167 MB language adapter from two Git-sized
parts and checks SHA-256; the 4.4 MB visual adapter is stored directly. The
runner merges the visual adapter into a copy of the frozen CR3 base in FP32,
saves that base in FP16, and passes the language adapter dynamically to vLLM.
It writes a full query trace and aggregate metrics under
`reproduction/full_test/finetuned_step8000/`. Set `PAS_MERGED_BASE` and
`PAS_OUTPUT_ROOT` to change those locations. The adapter hashes are embedded
in `materialize.py`. The checkpoint test wrapper uses the recorded vLLM GPU
memory utilization of 0.42 and maximum length of 2,048 tokens; the recorded
base CR3 test rows used 0.85 and 32,768 tokens. Their default score chunk
is 256, while the checkpoint wrapper uses 64. `MAX_MODEL_LEN` and
`SCORE_CHUNK` can override these defaults. The Val999 wrapper uses 0.85.

Run the other retrieval/reranking rows with the same evaluation contract:

```bash
bash examples/pas_reranker/reproduce_full_test.sh public_identity
bash examples/pas_reranker/reproduce_full_test.sh public_cr3
bash examples/pas_reranker/reproduce_full_test.sh finetuned_identity
bash examples/pas_reranker/reproduce_full_test.sh finetuned_zeroshot
```

For a quick score diagnostic, set `QUERY_SUBSET_FILE` to a PAS pairs-format
JSON containing selected test query rows. The runner still uses the full
`test_pairs.json` gallery and embeddings. Do not use `LIMIT_PAIRS` for this:
it also truncates the gallery and changes the retrieval candidates.

On the Lepton batch cluster, these commands use two eight-A100 workers and
16 independent query shards per mode. The jobs read the mounted checkout and
write shard traces under `reproduction/full_test/`. After both workers of a
job finish, run its CPU merge command:

```bash
bash examples/pas_reranker/submit_full_test_batch.sh public_cr3
bash examples/pas_reranker/submit_full_test_batch.sh finetuned_zeroshot
# After each corresponding job completes:
bash examples/pas_reranker/run_full_test_batch_worker.sh public_cr3 merge
bash examples/pas_reranker/run_full_test_batch_worker.sh finetuned_zeroshot merge
```

The public CR3 rows use the base model with no LoRA. The fine-tuned SigLIP2
plus zero-shot CR3 rows use fine-tuned SigLIP2 candidates and that same base
CR3 model. Set `PUBLIC_CR3_MODEL` to a verified local copy of
`nvidia/Cosmos3-Nano` if it is already downloaded. The fine-tuned CR3 row
uses the supplied step-8000 checkpoint.

To refit the **validation-selected** fusion weight, generate the 999-query
validation trace with the checkpoint, then run:

```bash
bash examples/pas_reranker/reproduce_val999.sh
python examples/pas_reranker/fit_val_apply_test_fusion.py \
  --validation-rankings reproduction/val999/finetuned_step8000/scalar_plus_accessories_val999_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl \
  --test-rankings reproduction/full_test/finetuned_step8000/scalar_plus_accessories_full_dedup_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl \
  --output-dir reproduction/fusion
```

This uses global z-score means and standard deviations fitted on Val999, then
selects `alpha` by Val999 overall mAP. The original selected alpha was
`0.31788`, applied unchanged to test. The fused score is
`alpha * normalized_SigLIP2 + (1-alpha) * normalized_CR3`. The recorded
full-test result is 82.99% mAP / 85.09% Rank-1 / 98.12% Rank-5, reproduced
from the saved traces by the branch code in
`reports/pas_cr3_step8000_fusion_replay.json`.

The table's zero-shot fusion *best weight* was selected on test, so it is a
diagnostic upper bound. `sweep_saved_rerank_fusion.py --normalization none`
replayed that trace at 79.94% overall mAP; several other rounded cells differ
slightly from the pasted table. The `binary CR3` row appears to binarize the
zero-shot CR3 yes-minus-no score and preserve retriever order for ties. A
`score > 0` replay yields 78.31% overall mAP / 79.59% Rank-1 / 96.31% Rank-5,
but the pasted per-difficulty figures do not all match; its original decision
and tie convention still needs confirmation. These two rows must not be used
as held-out improvements.

## Re-fine-tune the CR3 Nano checkpoint

The effective source configurations saved inside the original checkpoint are
`stage0_effective_config.json` (step 500) and
`stage2000_effective_config.json`. The latter belongs to the original stage 0
job after it resumed at step 1906; it is not the separate smallbatch stage 1
configuration. The stage 1 replay TOML preserves the training settings of the
original `train_fresh_rank32_k20_smallbatch_resume_step2000_lepton.toml`, with
only the resume/output paths and run timestamp changed. The runnable TOMLs are
`train_stage0_to_step2000_replay.toml` and
`train_stage2000_to_step8000_replay.toml`. They preserve rank-32 LoRA with
alpha 64, the seven language projections, visual blocks 24-26 and merger,
eight data-parallel GPUs, 80 candidates per GPU, intact K=20 microbatches,
seed 160901, and the rank-probability-mass plus class-balanced point loss
with weight 1 each. Structured attribute choice loss has weight 0.25.

Start stage 0 from the clean CR3 base, wait for the complete step-2000
checkpoint, then stop it. Stage 1 resumes that exact optimizer/scheduler/RNG
checkpoint in a separate run directory. Select the
complete step-8000 safetensors export, split it into language and visual
adapters with `split_joint_lora_adapter.py`, and use the evaluation commands
above with those regenerated adapters.

```bash
python examples/pas_reranker/render_fresh_training_configs.py \
  --output-dir reproduction/finetune/generated_configs
bash examples/pas_reranker/run_replay_stage.sh stage0
bash examples/pas_reranker/run_replay_stage.sh stage1
```

The stage runner launches Cosmos-RL with the rendered TOML and sends its
graceful stop signal after the target checkpoint completes. At step 8000 it
also waits for the LoRA export to finish before stopping. Stage 1 requires the
complete step-2000 checkpoint and splits its final LoRA export into language
and visual adapters. Re-running stage 1 after its target checkpoint exists
completes a missing adapter split without retraining.

For a faster two-node Lepton run, use 16 A100 GPUs with 40 candidates per
rank. This keeps the 640-candidate global batch and complete K20 microbatches;
the different rank partition can produce small numerical differences. The
eight-GPU TOMLs above remain the source checkpoint's configuration. On a
Lepton login host with access to the PAS mount, run:

```bash
export PAS_DATA_PARALLEL_RANKS=16
python examples/pas_reranker/render_fresh_training_configs.py \
  --data-parallel-ranks 16 \
  --run-root reproduction/finetune/fast16 \
  --output-dir reproduction/finetune/fast16_configs
bash examples/pas_reranker/submit_fast_lepton.sh stage0
# Once fast16/stage0/checkpoints/step_2000/policy/.rank_0_complete exists:
lep job stop -i "<stage0 job ID printed by submission>"
bash examples/pas_reranker/submit_fast_lepton.sh stage1
# Once fast16/stage1/checkpoints/step_8000/policy/.rank_0_complete exists:
/opt/venv/cosmos_rl/bin/python examples/pas_reranker/wait_adapter_export.py \
  --adapter-dir reproduction/finetune/fast16/stage1/safetensors/step_8000
lep job stop -i "<stage1 job ID printed by submission>"
python examples/pas_reranker/split_joint_lora_adapter.py \
  --adapter reproduction/finetune/fast16/stage1/safetensors/step_8000 \
  --language-output reproduction/finetune/fast16/deploy/language \
  --visual-output reproduction/finetune/fast16/deploy/visual
```

The submitter uses the NGC TAO 7.1 Cosmos-RL image, two eight-A100 workers,
the shared `/workspace` mount, and offline W&B. Override `LEPTON_IMAGE`,
`LEPTON_NODE_GROUP`, `LEPTON_MOUNT`, or `LEPTON_IMAGE_PULL_SECRET` for another
workspace. Set `PAS_ADAPTER_DIR` to the fast16 deploy directory in the
evaluation commands below.

For four workers and 32 A100 GPUs, keep the same global batch of 640 by using
20 candidates per rank. The rank partition differs further from the source
run. Render the separate configs and set the rank count when submitting each
stage; use the `fast32` paths in the checkpoint, adapter-export, and split
commands above:

```bash
export PAS_DATA_PARALLEL_RANKS=32
python examples/pas_reranker/render_fresh_training_configs.py \
  --data-parallel-ranks 32 \
  --run-root reproduction/finetune/fast32 \
  --output-dir reproduction/finetune/fast32_configs
bash examples/pas_reranker/submit_fast_lepton.sh stage0
# After fast32/stage0/checkpoints/step_2000/policy/.rank_0_complete exists:
lep job stop -i "<32-GPU stage0 job ID>"
bash examples/pas_reranker/submit_fast_lepton.sh stage1
# After fast32/stage1/checkpoints/step_8000/policy/.rank_0_complete exists:
/opt/venv/cosmos_rl/bin/python examples/pas_reranker/wait_adapter_export.py \
  --adapter-dir reproduction/finetune/fast32/stage1/safetensors/step_8000
lep job stop -i "<32-GPU stage1 job ID>"
python examples/pas_reranker/split_joint_lora_adapter.py \
  --adapter reproduction/finetune/fast32/stage1/safetensors/step_8000 \
  --language-output reproduction/finetune/fast32/deploy/language \
  --visual-output reproduction/finetune/fast32/deploy/visual
```

Both TOMLs retain the original `max_num_steps=200000` because changing the
total steps changes the cosine learning-rate schedule. Select the step-8000
checkpoint after it is complete. The current TOMLs point at the paths listed
above; the renderer updates run directories for the current checkout and
writes into its ignored `reproduction/finetune/` tree.
Keep eight data-parallel ranks for the closest checkpoint parity. Changing
that count alone changes the effective batch; the optional 16- and 32-GPU
variants lower the per-rank batch to preserve the global batch of 640, but
can still change the numerical trajectory. Thirty-two is the largest rank
count that keeps each per-rank batch a whole 20-candidate group. The 16-GPU
inference jobs shard independent test
queries without changing model scores through a training update.
Set `PAS_ADAPTER_DIR="$PWD/reproduction/finetune/deploy"` and a unique
`PAS_MERGED_BASE` when evaluating a regenerated adapter. Adjust the paths
when using another host. The visual cache must match the
training image pixel ceiling of 81,920 and the block-24 prefix; the final
three vision blocks and merger remain trainable.

After stage 1 exports both adapters, rerun the full test and Val999 with that
checkpoint, fit fusion on Val999 only, and compare the two step-8000 rows with
the pasted table. The comparison prints all eight rows and their metric deltas
in percentage points; it does not select a new weight on test.

```bash
export PAS_ADAPTER_DIR="$PWD/reproduction/finetune/deploy"
export PAS_MERGED_BASE=/tmp/pas-cr3-refinetuned-step8000-merged-base
export PAS_OUTPUT_ROOT="$PWD/reproduction/refinetuned/full_test"
bash examples/pas_reranker/reproduce_full_test.sh finetuned_step8000
export PAS_OUTPUT_ROOT="$PWD/reproduction/refinetuned/val999"
bash examples/pas_reranker/reproduce_val999.sh
python examples/pas_reranker/fit_val_apply_test_fusion.py \
  --validation-rankings reproduction/refinetuned/val999/finetuned_step8000/scalar_plus_accessories_val999_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl \
  --test-rankings reproduction/refinetuned/full_test/finetuned_step8000/scalar_plus_accessories_full_dedup_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl \
  --output-dir reproduction/refinetuned/fusion
python examples/pas_reranker/compare_finetuned_checkpoint.py \
  --summary reproduction/refinetuned/fusion/summary.json \
  --output reproduction/refinetuned/reference_comparison.json
```

To apply the table's original validation calibration to the regenerated
adapter, use the same fusion command with
`--calibration-summary reports/pas_cr3_step8000_fusion_replay.json` and a
separate `--output-dir reproduction/refinetuned/original_calibration`.
This freezes both `alpha=0.31788` and the original Val999 score statistics.
The default command above independently refits calibration; compare both
protocols when verifying a newly trained checkpoint.

To score the regenerated adapter faster on Lepton, shard the full test over
two eight-GPU workers. Use a merged-base path under shared `/workspace` so
both workers read the same visual adapter merge. The first worker creates that
merge; the second waits for its completion marker. Run Val999 on one eight-GPU
worker after the merge, then fit fusion on the resulting traces.

```bash
export PAS_ADAPTER_DIR="$PWD/reproduction/finetune/fast16/deploy"
export PAS_MERGED_BASE="$PWD/reproduction/refinetuned_fast16/merged_base"
export PAS_OUTPUT_ROOT="$PWD/reproduction/refinetuned_fast16/full_test"
bash examples/pas_reranker/submit_full_test_batch.sh finetuned_step8000
# Once all 16 test shards finish:
bash examples/pas_reranker/run_full_test_batch_worker.sh finetuned_step8000 merge
export PAS_OUTPUT_ROOT="$PWD/reproduction/refinetuned_fast16/val999"
bash examples/pas_reranker/submit_val999_batch.sh
# Once the Val999 job finishes:
python examples/pas_reranker/fit_val_apply_test_fusion.py \
  --validation-rankings reproduction/refinetuned_fast16/val999/finetuned_step8000/scalar_plus_accessories_val999_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl \
  --test-rankings reproduction/refinetuned_fast16/full_test/finetuned_step8000/scalar_plus_accessories_full_dedup_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl \
  --output-dir reproduction/refinetuned_fast16/fusion
python examples/pas_reranker/compare_finetuned_checkpoint.py \
  --summary reproduction/refinetuned_fast16/fusion/summary.json \
  --output reproduction/refinetuned_fast16/reference_comparison.json
```

## Verification state

Running `prepare_training_data.sh indices` and `attributes` from this clean
checkout regenerated both query-index files and both structured-attribute
record files byte for byte, matching all four hashes in
`recipe_manifests/source_checksums.json`.

The full top-20 miner reproduced both saved group counts using the original
SigLIP2 embedding shards and regenerated selections. Its second annotation
file was byte identical to the 20,346,920-record original. In the first
13,397,420-record file, 13,396,931 records were identical, 485 differed only
in retriever-score metadata (maximum absolute difference 0.0000465), and four
candidate positions swapped within two groups. See
`reports/pas_cr3_mining_replay.json`.

Eight-GPU batch jobs also regenerated the step-3906 SigLIP2 embeddings and
the block-24 visual cache under ignored `reproduction/`. The merged gallery
embedding and all 16 text embedding shards were byte identical to the
saved source files. All 16 cache manifests had identical mappings for their
222,217 images; their build-time metadata differs. All 6,808 primary and 144
supplementary `.safetensors` shards were byte identical to the saved cache,
covering 163.5 GB. See `reports/pas_cr3_artifact_regeneration.json`.

`reports/pas_cr3_baseline_reference.csv` records the four baseline rows from
the earlier full-test run. Fresh CPU-only runs of `public_identity` and
`finetuned_identity` matched all 12 table cells each; see
`reports/pas_cr3_fresh_identity_eval.json`. Replaying the saved step-8000
test and Val999 query
traces with the branch fusion code reproduced all 12 cells of the fine-tuned
CR3 rerank row and all 12 cells of the validation-selected fusion row to the
pasted precision. A fresh eight-GPU inference run with the byte-identical
step-8000 adapters processed all 69,633 test queries. Its merged result is
recorded in `reports/pas_cr3_fresh_checkpoint_eval.json`:

| Overall metric | Pasted table | Fresh checkpoint run |
| --- | --- | --- |
| CR3 rerank mAP / Rank-1 / Rank-5 | 80.63% / 81.35% / 97.29% | 80.62% / 81.33% / 97.28% |
| Validation-selected fusion mAP / Rank-1 / Rank-5 | 82.99% / 85.09% / 98.12% | 82.99% / 85.08% / 98.12% |

A separate two-worker, 16-A100 evaluation of the supplied adapter scored all
69,633 test queries, merged the 16 shards, then fitted fusion on 999 Val999
queries using one eight-A100 worker. Its validation-selected alpha was
`0.31522`; the maximum difference across the eight fine-tuned table rows was
0.07 percentage points. See
`reports/pas_cr3_supplied_checkpoint_16shard_eval.json` for every metric:

| Overall metric | Pasted table | Fresh 16-GPU checkpoint run |
| --- | --- | --- |
| CR3 rerank mAP / Rank-1 / Rank-5 | 80.63% / 81.35% / 97.29% | 80.63% / 81.35% / 97.28% |
| Validation-selected fusion mAP / Rank-1 / Rank-5 | 82.99% / 85.09% / 98.12% | 82.98% / 85.07% / 98.12% |

Two 16-GPU batch evaluations independently ran public CR3 and fine-tuned
SigLIP2 with zero-shot CR3 across all 69,633 test queries using the recorded
32,768-token base-model limit. The fresh overall results are below; details
are in `reports/pas_cr3_source_settings_16shard_eval.json`. The earlier
2,048-token diagnostic is retained in `reports/pas_cr3_fresh_baseline_eval.json`.

| Overall metric | Pasted table | Fresh 16-GPU run |
| --- | --- | --- |
| Public CR3 rerank mAP / Rank-1 / Rank-5 | 52.22% / 54.21% / 70.46% | 52.24% / 54.25% / 70.47% |
| Zero-shot CR3 rerank mAP / Rank-1 / Rank-5 | 68.29% / 63.62% / 92.57% | 68.29% / 63.66% / 92.54% |
| Zero-shot CR3 test-selected fusion mAP / Rank-1 / Rank-5 | 79.94% / 81.65% / 97.31% | 79.94% / 81.64% / 97.33% |

The fresh zero-shot best fusion selected `alpha=0.99304` by test-set mAP;
the saved source selected `0.99331`. This sweep is a diagnostic test-set
upper bound, not a validation-selected result. Both fresh rerank traces are
close to the pasted table but have changed candidate scores and rank orders.
`compare_query_traces.py --order-independent` matches queries across the
different eight-shard and sixteen-shard trace orderings.

In the earlier eight-GPU run, the frozen Val999 fusion weight remained
`0.31788`. Compared candidate by
candidate with the saved reference trace, 108 queries changed Rank-1 and 13
changed Rank-5; the mean absolute CR3 score difference was 0.00785. The
cause of those score differences has not been isolated, so exact rounded-cell
parity should be checked against the saved trace as well as a fresh runtime.
Independent re-fine-tuning completed step 8000 on thirty-two GPUs, with a
global batch of 640. Eight- and sixteen-GPU runs provided earlier parity checks. Saved adapters were compared with
the corresponding source checkpoint, and each loss difference below is the
mean absolute difference over the preceding 500 training steps:

| Replay checkpoint | Language cosine | Visual cosine | Loss difference |
| --- | ---: | ---: | ---: |
| Eight GPUs, stage 0 step 2000 | 0.999391 | 0.999979 | 0.00922 |
| Sixteen GPUs, stage 0 step 2000 | 0.999365 | 0.999979 | 0.00893 |
| Thirty-two GPUs, stage 0 step 2000 | 0.999374 | 0.999982 | 0.00855 |
| Sixteen GPUs, resumed stage 1 step 2500 | 0.999162 | 0.999974 | 0.00970 |
| Thirty-two GPUs, resumed stage 1 step 3000 | 0.998996 | 0.999972 | 0.01037 |
| Sixteen GPUs, resumed stage 1 step 4000 | 0.998450 | 0.999953 | 0.01168 |
| Thirty-two GPUs, resumed stage 1 step 4000 | 0.998542 | 0.999958 | 0.01148 |
| Thirty-two GPUs, resumed stage 1 step 5000 | 0.998045 | 0.999942 | 0.01213 |
| Thirty-two GPUs, resumed stage 1 step 5500 | 0.997533 | 0.999932 | 0.01359 |
| Thirty-two GPUs, resumed stage 1 step 6000 | 0.997079 | 0.999920 | 0.01375 |
| Thirty-two GPUs, resumed stage 1 step 6500 | 0.996437 | 0.999907 | 0.01605 |
| Thirty-two GPUs, resumed stage 1 step 7000 | 0.995744 | 0.999891 | 0.01639 |
| Thirty-two GPUs, resumed stage 1 step 7500 | 0.994966 | 0.999874 | 0.01931 |
| Thirty-two GPUs, resumed stage 1 step 8000 | 0.993832 | 0.999854 | 0.02010 |

The independent 32-GPU **step-4000** adapter was also evaluated on all 69,633
test queries and Val999. Overall reranking reached **80.81% mAP / 81.75%
Rank-1 / 97.42% Rank-5**. Val999 selected fusion weight `0.30526`, which
gave **83.07% / 85.18% / 98.14%** on the test set, versus the reference
step-8000 fusion row's **82.99% / 85.09% / 98.12%**. The largest fusion
difference is 0.09 percentage points. The detailed intermediate comparison
is in `reports/pas_cr3_independent_step4000_eval.json`.

`reports/pas_cr3_retraining_parity.json` records the earlier checkpoint and
loss comparisons as well. The final independent step-8000 adapter was evaluated
on all 69,633 test queries using 16 GPUs and on Val999 using eight GPUs:

| Independent step-8000 result | Reference mAP / Rank-1 / Rank-5 | Reproduction |
| --- | --- | --- |
| Reranking | 80.63% / 81.35% / 97.29% | 80.62% / 81.37% / 97.30% |
| Fusion with original validation calibration (`alpha=0.31788`) | 82.99% / 85.09% / 98.12% | 82.96% / 85.01% / 98.10% |
| Fusion with independently refitted Val999 calibration (`alpha=0.20035`) | 82.99% / 85.09% / 98.12% | 82.65% / 84.41% / 98.02% |

Across difficulties, reranking differs by at most 0.07 percentage points.
Reusing the original validation alpha and normalization gives a maximum
fusion difference of 0.24 points. Independently refitting on the same 999
validation queries selects a different alpha and gives a maximum fusion
difference of 1.01 points. The two fusion protocols are reported separately;
no weight was selected on test for these fine-tuned results. Full metrics,
checkpoint hashes, and job provenance are in
`reports/pas_cr3_independent_step8000_eval.json`. Large traces and generated
training artifacts belong under ignored `reproduction/`.

`verify_reference_table.py` compares all 32 rows with fresh identity results
and the available saved-trace replays. It matches 20 rows exactly. The four
zero-shot reranking rows, four zero-shot best-fusion rows, and four
binary-score rows differ, as recorded in
`reports/pas_cr3_reference_replay_check.txt` for follow-up. The historical
comparison CSV supplies the zero-shot reranking row, but recalculating that
row from its saved candidate trace matches none of its 12 rounded cells: the
trace's own overall mAP is 68.28% versus 68.29% in the comparison CSV. The
comparison CSV was written at 21:46 during the source run, before the merged
trace and aggregate were written at 22:10; no candidate trace matching the
21:46 CSV has been found. This timing does not establish why the values
differ. The
public-CR3 trace does match all 12 cells. See
`reports/pas_cr3_base_cr3_trace_replay.json`; the saved zero-shot trace cannot
be treated as exact evidence for the pasted reranking row. To check all fresh
baseline rows, run the test-selected zero-shot fusion sweep and pass both
fresh traces to the table verifier. The same zero-shot trace tests the binary
convention. With the fresh 16-GPU checkpoint and Val999 results included,
all 32 rows are within 0.12 percentage points of the pasted table; nine rows
match every rounded cell exactly. `reports/pas_cr3_all32_fresh_check.txt`
records every difference. The explicit tolerance accepts these small changes
without hiding them; omitting it requires exact rounded-cell equality:

```bash
public_trace=reproduction/full_test_source_settings/public_cr3/scalar_plus_accessories_full_dedup_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl
zeroshot_trace=reproduction/full_test_source_settings/finetuned_zeroshot/scalar_plus_accessories_full_dedup_cosmos_reason_depth20_k5/scalar_plus_accessories/query_rankings.jsonl
python examples/pas_reranker/sweep_saved_rerank_fusion.py \
  --rankings "$zeroshot_trace" \
  --output-dir reproduction/full_test_source_settings/finetuned_zeroshot_fusion \
  --normalization none
python examples/pas_reranker/verify_reference_table.py \
  --max-delta-pp 0.12 \
  --checkpoint-summary reproduction/supplied16/fusion/summary.json \
  --public-cr3-trace "$public_trace" \
  --zeroshot-cr3-trace "$zeroshot_trace" \
  --zeroshot-fusion-summary reproduction/full_test_source_settings/finetuned_zeroshot_fusion/summary.json \
  --binary-trace "$zeroshot_trace"
```

For the independent step-8000 adapter, applying the original validation
calibration places all 32 rows within 0.24 percentage points. The complete
comparison is in `reports/pas_cr3_all32_independent_check.txt`. Repeat the
verifier command above with `--max-delta-pp 0.25` and
`--checkpoint-summary reproduction/refinetuned_fast32/original_calibration/summary.json`
(or the equivalent output directory from your run). This tolerance applies
to the frozen original calibration; the independently refitted calibration
has the larger differences reported above.

A CPU audit of the first 1,000 saved K20 training groups checked all 20,000
candidate labels against the exported PAS scalar/accessory metadata. Every
label and retriever-rank position matched the documented rule, and each group
contained a negative.
