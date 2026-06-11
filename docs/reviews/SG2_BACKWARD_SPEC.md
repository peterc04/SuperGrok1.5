# SG2 CSA/HCA Bilevel Backward — Implementation Spec

The hand-written saved-activation adjoint through the SuperGrok2 meta-net
(input-proj+sort → CSA → HCA → GRU → PEER → smart_grad). Oracle = the
differentiable PyTorch `forward_for_bilevel` in
`grokking_optimizers/optimizers/supergrok2.py:734`.

## Forward pipeline (what the adjoint reverses)

1. **input_proj + sort** (`csrc/algorithms/supergrok2.h:54`):
   `x_out[i,d] = grad[i]*W[d,0] + sharp[i]*W[d,1] + b[d]`; sort rows by `|grad|`
   desc → `x_sorted`, `sort_indices`.
2. **CSA** (compressed-sparse attention): learned-pool KV compression
   (`softmax(csa_compress_w)` over a window of `csa_compress`), q/k/v projections,
   lightning-indexer top-k select (`idx_DQ/idx_UQ/idx_K`, `indexer_rank`,
   `csa_topk`) + sliding `csa_window`, online softmax → `csa_ctx` via `csa_out_W`.
   Saves `csa_saved_denom[N]`, `csa_saved_sel_idx[N,topk]`.
3. **HCA**: mean-pool KV compression (`hca_compress`), dense attention, `hca_out_W`.
   Saves `hca_saved_denom[N]`.
4. **GRU** (`gru_Wz/Wr/Wh` + biases): standard GRU on `[grad,sharp,csa_ctx,hca_ctx]`.
5. **PEER** product-key routing (`peer_query_Ws`, `prod_keys_A/B`, k=4 per side →
   16 experts) + expert MLP (`expert_W1/W2`, `expert_hidden`) → `expert_out`.
6. **smart_grad** = `grad + rescale * expert_out`.

## Backward contract (`launch_csa_hca_backward`)
- **In:** `d_smart_grad[N]` + all saved activations + all weights + dims.
- **Out (accumulate, zero-init):** 24 weight-grad buffers —
  `d_input_proj_W/b`, `d_csa_{q,k,v,out}_W` + `d_csa_compress_w` +
  `d_csa_idx_{DQ,UQ,K}`, `d_hca_{q,k,v,out}_W`, `d_gru_{Wz,bz,Wr,br,Wh,bh}`,
  `d_peer_query_Ws`, `d_prod_keys_{A,B}`, `d_expert_{W1,b1,W2,b2}`.
- No `d_grad`/`d_x_sorted` outputs (adjoint flows only into meta-params).

## Reusable primitive
`sg2_bilevel_precompute_timestep` (`csrc/algorithms/supergrok2.h:365`) recomputes
per-row Q/K/V/QI projections so they need not be saved.

## Recompute / checkpoint
`checkpoint_interval` (≤ `MAX_CKPT_INTERVAL=32`): save activations every K steps,
recompute the rest in backward. For the first cut, support `checkpoint_interval`
by recomputing per-row projections via the precompute primitive; the heavy
contexts (`csa_ctx`, `hca_ctx`) are saved by `bilevel_fwd_save`.

## Reverse order
smart_grad → expert MLP+routing (atomic accum on expert/key/query grads) →
GRU gates → HCA dense-attn (recover softmax from `hca_saved_denom`) →
CSA sparse-attn (recover from `csa_saved_denom`, `csa_saved_sel_idx`) →
compression + projections → input_proj (scatter via `sort_indices`).

## Numerics oracle (hardware-deferred)
Compare `d_*_W` against `torch.autograd.grad` through `forward_for_bilevel`
for N=20 rows, rtol=1e-3/atol=1e-5 (fp32). Command in HARDWARE_VALIDATION.md.
