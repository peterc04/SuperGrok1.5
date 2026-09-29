"""Pure-PyTorch stand-ins for the tilelang kernels imported by the reference model.py.

Quantization kernels are identities (the port trains in full precision); sparse_attn and
hc_split_sinkhorn reproduce the kernels' exact semantics.
"""

import torch


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert inplace
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    assert inplace
    return x


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("the test builds the reference with plain fp32 weights")


fp4_gemm = fp8_gemm


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather-based, exactly the kernel: running max starts at -1e30, -1 slots are skipped."""
    b = q.size(0)
    idx = topk_idxs.long()
    valid = idx >= 0
    bi = torch.arange(b, device=q.device)[:, None, None]
    kv_sel = kv[bi, idx.clamp_min(0)].float() * valid[..., None]
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), kv_sel) * softmax_scale
    logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
    mx = logits.amax(-1).clamp_min(-1e30)
    e = torch.exp(logits - mx[..., None])
    denom = e.sum(-1) + torch.exp(attn_sink.float()[None, None, :] - mx)
    return (torch.einsum("bmhk,bmkd->bmhd", e, kv_sel) / denom[..., None]).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
