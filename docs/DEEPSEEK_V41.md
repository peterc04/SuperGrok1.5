# DeepSeek-V4.1-Flash, the race's model

`deepseek_v41/` is a from-scratch, trainable, pure-PyTorch port of the
DeepSeek-V4.1-Flash text backbone (inference code: `deepseek-ai/DeepSeek-V4.1-Flash`,
`inference/model.py`, MIT). Vision (and the DSpark/MTP heads) are left out; Engram is on.
Parameter names follow the official code, so `Config.from_json` reads the released
`config.json` and the tests can load the same weights into both.

## What is in it

| Part | What it does |
|---|---|
| **Attention** | Multi-query attention with a low-rank query (`wq_a → q_norm → wq_b`, one query per head) and a single shared key/value head (`wkv`). Output goes through a grouped low-rank projection (`wo_a` is block-diagonal: one matrix per output group, then `wo_b`). RoPE on the last `rope_head_dim` dims, applied inversely to the output; a learned per-head attention sink. |
| **Sparse attention (CSA)** | Every layer attends to a causal 128-token sliding window. Layers with `compress_ratio > 0` also attend to compressed KV: groups of `ratio` tokens pooled by a learned per-channel softmax (`Compressor`). Which compressed entries a query sees is chosen by the **Indexer** (ReLU scores, top-k 512). Layers are KV sources, index sources, re-index layers or reuse layers exactly as in the release (`kv_source_layers`, `index_source_layers`); the decoder half re-uses the encoder's KV and a hierarchical candidate pool (`candidate_source_layer`). |
| **MoE** | 384 routed experts + 1 shared, top-6, `sqrtsoftplus` scores, route scale 1.5, SwiGLU clamped at 10. Load balancing is DeepSeek's auxiliary-loss-free bias (`model.update_router_bias()`, ±0.001 per step) plus the sequence-wise balance loss (α = 1e-4). |
| **mHC** | Four residual copies mixed by a Sinkhorn-normalized (20 iterations) doubly stochastic matrix before and after every sublayer. |
| **Engram** | Hashed n-gram (2..4) memory: XOR-multiplicative hashes into prime-sized tables per head, gated into the residual stream, at the layers listed in `engram_layer_ids`. |
| **Norms** | RMSNorm, eps 1e-20 (never run in fp16). |

Not ported (absent from V4.1 pre-training or out of scope): MTP, z-loss, the FP8/FP4
quantization-aware-training emulation, the indexer's KL training loss, vision and DSpark.

## How it is checked (`tests/test_deepseek*.py`)

* **Against the official code.** `tests/reference_deepseek/` holds the official
  `model.py` and `engram.py` unmodified, with pure-PyTorch stand-ins for the CUDA/TileLang
  kernels. The same weights are loaded into both, and full-model logits agree to about
  2e-7 (tolerance 1e-5) for sequence lengths 1, 2, 3, 4, 5, 8, 11 and 17. During
  development, deliberately breaking individual components was confirmed to make this
  comparison fail.
* **Parameter counts** of the `flash` preset match the model card: 551.88B backbone + 196.61B
  Engram tables.
* Unit tests cover the sink attention against a gather reference, Sinkhorn, the router,
  candidate blocks, the Engram hash against a naive transcription, causality, gradient flow,
  deterministic init, router-load tracking and the per-parameter roles given to optimizers.

## Presets (`deepseek_v41.PRESETS`)

| Preset | Size at the race vocabulary (99) | Use |
|---|---|---|
| `flash` | 552B (+197B Engram) | the released shapes; for counting only |
| `tiny` | 4.0M total, 0.46M active (3.1M are Engram tables) | CPU tests and smoke races; every mechanism present |
| `h100` | 8.69B total, 1.34B active | the single-GPU target (40 layers, d 2048, 16 heads × 256, 64 experts) |

`tiny` and `h100` use plain RoPE (`original_seq_len=0`); YaRN is a context-extension
setting of the released checkpoint, not part of training from scratch. Any field can be
overridden: `--model-overrides '{"n_routed_experts": 16}'`.

## Things to know when training it on grokking tasks

* **The sparse attention is inert at these lengths.** Grokking sequences are 4-8 tokens,
  well inside the 128-token window, and top-k 512 keeps every compressed entry. The
  indexer therefore changes nothing (randomizing its weights changes the output by exactly
  0), and it gets no gradient from the loss (top-k is discrete). It is kept for fidelity
  and for the long-sequence phase.
* **Engram at the faithful size memorizes.** Engram tables have `160 × (vocab + 1)`
  buckets per head. On `a / b = c` with vocab 99 that is 16K rows per head, so most
  n-grams get their own row: a per-example lookup of training answers, while test
  examples read rows that were never trained. A smaller table makes rows shared, which is
  a very different regime: `--model-overrides '{"engram_vocab_size": 1009}'`. Which one
  the race should use is a design decision; record it with the results.
* **Router statistics.** Only the step's own training forward updates the router's
  load counters. Every extra forward an optimizer runs (SAM probes, meta steps) happens
  under `model.frozen_router_stats()`, so all optimizers see the same load balancing.
* **Idle experts get no gradient** (`grad=None`), so they get no optimizer step at all,
  like in `torch.optim`.
* **Precision.** Master weights and the mHC residual stream are fp32. Under
  `--dtype bf16` (CUDA autocast) norms, Sinkhorn, the router, attention softmax, Engram
  gates and the loss stay in fp32. fp16 is not supported (eps 1e-20).
* **Two hardenings over the inference code.** The indexer never selects a position with a
  non-finite score (outside the visible range or the candidate pool), and the config
  rejects candidate pools that could be smaller than `index_topk`.

## Memory on one H100 (80 GB), `h100` preset

At 8.69B parameters the plain fp32 recipe does not fit: weights and gradients take
70 GB, and fp32 AdamW moments add 70 GB more. Options until the CUDA phase: fewer routed
experts (per-token compute is unchanged; 16 experts gives 2.65B, which fits fp32 AdamW at
42 GB), `--micro-batches` to bound activations, or pure-bf16 weights with at most
4 bytes/param of optimizer state. See `docs/ALGORITHMS.md` for the per-optimizer table.

## Notes for the CUDA phase

* `sink_attention` is a dense masked softmax with a sink column. The official kernel's
  identity `o = o_nosink · sigmoid(LSE − sink)` lets FlashAttention/flex-attention with
  `return_lse` implement it.
* The MoE loops over experts that received tokens; a grouped GEMM replaces that loop.
* mHC mixing and Sinkhorn are small per-token ops that are worth fusing.
* Engram hashing depends only on the input ids and can be precomputed per batch.
