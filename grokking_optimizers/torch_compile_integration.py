"""
torch.library Custom Op Registration for SuperGrok v2 Kernels.

Registers SuperGrok v2 CUDA/CPU kernels as torch.library custom ops so they
are visible to torch.compile (Dynamo + Inductor). Each op has:

  1. A schema definition via supergrok_lib.define()
  2. A concrete implementation that delegates to the C++ extension
  3. A FakeTensor (abstract) implementation for tracing/shape inference

This lets torch.compile trace through SuperGrok2 optimizer steps without
graph breaks, enabling fusion with surrounding user code.

Usage::

    import grokking_optimizers.torch_compile_integration  # registers ops
    compiled_step = torch.compile(model_step_fn)
"""

import torch
import torch.library

from grokking_optimizers._ops_loader import get_ops

# ---------------------------------------------------------------------------
# Library definition
# ---------------------------------------------------------------------------
supergrok_lib = torch.library.Library("supergrok", "DEF")

# ===================================================================
# 1. supergrok2_scan_step  --  single-parameter scan step
# ===================================================================

supergrok_lib.define(
    "supergrok2_scan_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    Tensor exp_avg_sq,"
    "    Tensor gru_hidden,"
    "    Tensor scan_state,"
    "    float lr,"
    "    float beta1,"
    "    float beta2,"
    "    float eps,"
    "    float weight_decay,"
    "    int step"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


def _supergrok2_scan_step_impl(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    gru_hidden: torch.Tensor,
    scan_state: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple:
    ops = get_ops()
    return ops.supergrok2_scan_step(
        param, grad, exp_avg, exp_avg_sq, gru_hidden, scan_state,
        lr, beta1, beta2, eps, weight_decay, step,
    )


supergrok_lib.impl("supergrok2_scan_step", _supergrok2_scan_step_impl, "CUDA")


@torch.library.register_fake("supergrok::supergrok2_scan_step")
def _supergrok2_scan_step_abstract(
    param, grad, exp_avg, exp_avg_sq, gru_hidden, scan_state,
    lr, beta1, beta2, eps, weight_decay, step,
):
    return (
        torch.empty_like(param),       # updated param
        torch.empty_like(exp_avg),     # updated exp_avg
        torch.empty_like(exp_avg_sq),  # updated exp_avg_sq
        torch.empty_like(gru_hidden),  # updated gru_hidden
        torch.empty_like(scan_state),  # updated scan_state
    )


# ===================================================================
# 2. supergrok2_batched_step  --  batched multi-parameter step
# ===================================================================

supergrok_lib.define(
    "supergrok2_batched_step("
    "    Tensor[] params,"
    "    Tensor[] grads,"
    "    Tensor[] exp_avgs,"
    "    Tensor[] exp_avg_sqs,"
    "    Tensor[] gru_hiddens,"
    "    Tensor[] scan_states,"
    "    float lr,"
    "    float beta1,"
    "    float beta2,"
    "    float eps,"
    "    float weight_decay,"
    "    int step"
    ") -> (Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])"
)


def _supergrok2_batched_step_impl(
    params, grads, exp_avgs, exp_avg_sqs, gru_hiddens, scan_states,
    lr, beta1, beta2, eps, weight_decay, step,
):
    ops = get_ops()
    return ops.supergrok2_batched_step(
        params, grads, exp_avgs, exp_avg_sqs, gru_hiddens, scan_states,
        lr, beta1, beta2, eps, weight_decay, step,
    )


supergrok_lib.impl("supergrok2_batched_step", _supergrok2_batched_step_impl, "CUDA")


@torch.library.register_fake("supergrok::supergrok2_batched_step")
def _supergrok2_batched_step_abstract(
    params, grads, exp_avgs, exp_avg_sqs, gru_hiddens, scan_states,
    lr, beta1, beta2, eps, weight_decay, step,
):
    return (
        [torch.empty_like(p) for p in params],
        [torch.empty_like(m) for m in exp_avgs],
        [torch.empty_like(v) for v in exp_avg_sqs],
        [torch.empty_like(h) for h in gru_hiddens],
        [torch.empty_like(s) for s in scan_states],
    )


# ===================================================================
# 3. supergrok2_distributed_scan  --  multi-GPU distributed scan
# ===================================================================

supergrok_lib.define(
    "supergrok2_distributed_scan("
    "    Tensor local_input,"
    "    Tensor scan_state,"
    "    Tensor A_log,"
    "    Tensor D_param,"
    "    int d_inner,"
    "    int d_state,"
    "    int world_size,"
    "    int rank"
    ") -> (Tensor, Tensor)"
)


def _supergrok2_distributed_scan_impl(
    local_input, scan_state, A_log, D_param,
    d_inner, d_state, world_size, rank,
):
    ops = get_ops()
    return ops.supergrok2_distributed_scan(
        local_input, scan_state, A_log, D_param,
        d_inner, d_state, world_size, rank,
    )


supergrok_lib.impl(
    "supergrok2_distributed_scan", _supergrok2_distributed_scan_impl, "CUDA",
)


@torch.library.register_fake("supergrok::supergrok2_distributed_scan")
def _supergrok2_distributed_scan_abstract(
    local_input, scan_state, A_log, D_param,
    d_inner, d_state, world_size, rank,
):
    return (
        torch.empty_like(local_input),  # scan output (same shape as input)
        torch.empty_like(scan_state),   # updated scan state
    )


# ===================================================================
# 4. supergrok2_fused_elem_step  --  fused element-wise Adam+GRU step
# ===================================================================

supergrok_lib.define(
    "supergrok2_fused_elem_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    Tensor exp_avg_sq,"
    "    Tensor gru_h,"
    "    Tensor meta_weight,"
    "    float lr,"
    "    float beta1,"
    "    float beta2,"
    "    float eps,"
    "    float weight_decay,"
    "    float meta_rescale,"
    "    int step"
    ") -> (Tensor, Tensor, Tensor, Tensor)"
)


def _supergrok2_fused_elem_step_impl(
    param, grad, exp_avg, exp_avg_sq, gru_h, meta_weight,
    lr, beta1, beta2, eps, weight_decay, meta_rescale, step,
):
    ops = get_ops()
    return ops.supergrok2_fused_elem_step(
        param, grad, exp_avg, exp_avg_sq, gru_h, meta_weight,
        lr, beta1, beta2, eps, weight_decay, meta_rescale, step,
    )


supergrok_lib.impl(
    "supergrok2_fused_elem_step", _supergrok2_fused_elem_step_impl, "CUDA",
)


@torch.library.register_fake("supergrok::supergrok2_fused_elem_step")
def _supergrok2_fused_elem_step_abstract(
    param, grad, exp_avg, exp_avg_sq, gru_h, meta_weight,
    lr, beta1, beta2, eps, weight_decay, meta_rescale, step,
):
    return (
        torch.empty_like(param),       # updated param
        torch.empty_like(exp_avg),     # updated exp_avg
        torch.empty_like(exp_avg_sq),  # updated exp_avg_sq
        torch.empty_like(gru_h),       # updated GRU hidden state
    )


# ===================================================================
# 5. supergrok2_scan_backward  --  backward pass for scan
# ===================================================================

supergrok_lib.define(
    "supergrok2_scan_backward("
    "    Tensor grad_output,"
    "    Tensor scan_state,"
    "    Tensor A_log,"
    "    Tensor D_param,"
    "    Tensor input_cache,"
    "    int d_inner,"
    "    int d_state"
    ") -> (Tensor, Tensor, Tensor)"
)


def _supergrok2_scan_backward_impl(
    grad_output, scan_state, A_log, D_param, input_cache,
    d_inner, d_state,
):
    ops = get_ops()
    return ops.supergrok2_scan_backward(
        grad_output, scan_state, A_log, D_param, input_cache,
        d_inner, d_state,
    )


supergrok_lib.impl(
    "supergrok2_scan_backward", _supergrok2_scan_backward_impl, "CUDA",
)


@torch.library.register_fake("supergrok::supergrok2_scan_backward")
def _supergrok2_scan_backward_abstract(
    grad_output, scan_state, A_log, D_param, input_cache,
    d_inner, d_state,
):
    return (
        torch.empty_like(input_cache),  # grad w.r.t. input
        torch.empty_like(A_log),        # grad w.r.t. A_log
        torch.empty_like(D_param),      # grad w.r.t. D_param
    )


# ===================================================================
# 6. supergrok2_cpu_step  --  CPU fallback step
# ===================================================================

supergrok_lib.define(
    "supergrok2_cpu_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    Tensor exp_avg_sq,"
    "    Tensor gru_hidden,"
    "    float lr,"
    "    float beta1,"
    "    float beta2,"
    "    float eps,"
    "    float weight_decay,"
    "    int step"
    ") -> (Tensor, Tensor, Tensor, Tensor)"
)


def _supergrok2_cpu_step_impl(
    param, grad, exp_avg, exp_avg_sq, gru_hidden,
    lr, beta1, beta2, eps, weight_decay, step,
):
    ops = get_ops()
    return ops.supergrok2_cpu_step(
        param, grad, exp_avg, exp_avg_sq, gru_hidden,
        lr, beta1, beta2, eps, weight_decay, step,
    )


supergrok_lib.impl("supergrok2_cpu_step", _supergrok2_cpu_step_impl, "CPU")


@torch.library.register_fake("supergrok::supergrok2_cpu_step")
def _supergrok2_cpu_step_abstract(
    param, grad, exp_avg, exp_avg_sq, gru_hidden,
    lr, beta1, beta2, eps, weight_decay, step,
):
    return (
        torch.empty_like(param),       # updated param
        torch.empty_like(exp_avg),     # updated exp_avg
        torch.empty_like(exp_avg_sq),  # updated exp_avg_sq
        torch.empty_like(gru_hidden),  # updated gru_hidden
    )


# ===================================================================
# 7. grokadamw_fused_step  --  GrokAdamW single-parameter step
# ===================================================================

supergrok_lib.define(
    "grokadamw_fused_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    Tensor exp_avg_sq,"
    "    Tensor ema,"
    "    float alpha,"
    "    float lamb,"
    "    float beta1,"
    "    float beta2,"
    "    float lr,"
    "    float weight_decay,"
    "    float eps,"
    "    float bc1,"
    "    float bc2"
    ") -> (Tensor, Tensor, Tensor, Tensor)"
)


def _grokadamw_fused_step_impl(
    param, grad, exp_avg, exp_avg_sq, ema,
    alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
):
    ops = get_ops()
    ops.grokadamw_fused_step(
        [param], [grad], [exp_avg], [exp_avg_sq], [ema],
        alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
    )
    return param, exp_avg, exp_avg_sq, ema


supergrok_lib.impl("grokadamw_fused_step", _grokadamw_fused_step_impl, "CUDA")


@torch.library.register_fake("supergrok::grokadamw_fused_step")
def _grokadamw_fused_step_abstract(
    param, grad, exp_avg, exp_avg_sq, ema,
    alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
):
    return (
        torch.empty_like(param),
        torch.empty_like(exp_avg),
        torch.empty_like(exp_avg_sq),
        torch.empty_like(ema),
    )


# ===================================================================
# 8. lion_fused_step  --  Lion single-parameter step
# ===================================================================

supergrok_lib.define(
    "lion_fused_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    float lr,"
    "    float beta1,"
    "    float beta2,"
    "    float weight_decay"
    ") -> (Tensor, Tensor)"
)


def _lion_fused_step_impl(param, grad, exp_avg, lr, beta1, beta2, weight_decay):
    ops = get_ops()
    ops.lion_fused_step([param], [grad], [exp_avg], lr, beta1, beta2, weight_decay)
    return param, exp_avg


supergrok_lib.impl("lion_fused_step", _lion_fused_step_impl, "CUDA")


@torch.library.register_fake("supergrok::lion_fused_step")
def _lion_fused_step_abstract(param, grad, exp_avg, lr, beta1, beta2, weight_decay):
    return (torch.empty_like(param), torch.empty_like(exp_avg))


# ===================================================================
# 9. grokfast_fused_step  --  Grokfast EMA amplification
# ===================================================================

supergrok_lib.define(
    "grokfast_fused_step("
    "    Tensor grad,"
    "    Tensor ema,"
    "    float alpha,"
    "    float lamb"
    ") -> (Tensor, Tensor)"
)


def _grokfast_fused_step_impl(grad, ema, alpha, lamb):
    ops = get_ops()
    ops.grokfast_fused_step([grad], [ema], alpha, lamb)
    return grad, ema


supergrok_lib.impl("grokfast_fused_step", _grokfast_fused_step_impl, "CUDA")


@torch.library.register_fake("supergrok::grokfast_fused_step")
def _grokfast_fused_step_abstract(grad, ema, alpha, lamb):
    return (torch.empty_like(grad), torch.empty_like(ema))


# ===================================================================
# 10. prodigy_fused_step  --  Prodigy per-element step
# ===================================================================

supergrok_lib.define(
    "prodigy_fused_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    Tensor exp_avg_sq,"
    "    Tensor s,"
    "    float d_lr,"
    "    float beta1,"
    "    float beta2,"
    "    float lr,"
    "    float weight_decay,"
    "    float eps,"
    "    float bc1,"
    "    float bc2"
    ") -> (Tensor, Tensor, Tensor, Tensor)"
)


def _prodigy_fused_step_impl(
    param, grad, exp_avg, exp_avg_sq, s,
    d_lr, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
):
    ops = get_ops()
    ops.prodigy_fused_step(
        [param], [grad], [exp_avg], [exp_avg_sq], [s],
        d_lr, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
    )
    return param, exp_avg, exp_avg_sq, s


supergrok_lib.impl("prodigy_fused_step", _prodigy_fused_step_impl, "CUDA")


@torch.library.register_fake("supergrok::prodigy_fused_step")
def _prodigy_fused_step_abstract(
    param, grad, exp_avg, exp_avg_sq, s,
    d_lr, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
):
    return (
        torch.empty_like(param),
        torch.empty_like(exp_avg),
        torch.empty_like(exp_avg_sq),
        torch.empty_like(s),
    )


# ===================================================================
# 11. neuralgrok_fused_full_step  --  NeuralGrok amplifier + Adam
# ===================================================================

supergrok_lib.define(
    "neuralgrok_fused_full_step("
    "    Tensor param,"
    "    Tensor grad,"
    "    Tensor exp_avg,"
    "    Tensor exp_avg_sq,"
    "    Tensor W1,"
    "    Tensor b1,"
    "    Tensor W2,"
    "    Tensor b2,"
    "    float alpha_amp,"
    "    float beta_amp,"
    "    int hidden_dim,"
    "    float beta1,"
    "    float beta2,"
    "    float lr,"
    "    float weight_decay,"
    "    float eps,"
    "    float bc1,"
    "    float bc2"
    ") -> (Tensor, Tensor, Tensor)"
)


def _neuralgrok_fused_full_step_impl(
    param, grad, exp_avg, exp_avg_sq,
    W1, b1, W2, b2,
    alpha_amp, beta_amp, hidden_dim,
    beta1, beta2, lr, weight_decay, eps, bc1, bc2,
):
    ops = get_ops()
    ops.neuralgrok_fused_full_step(
        [param], [grad], [exp_avg], [exp_avg_sq],
        W1, b1, W2, b2,
        alpha_amp, beta_amp, hidden_dim,
        beta1, beta2, lr, weight_decay, eps, bc1, bc2,
    )
    return param, exp_avg, exp_avg_sq


supergrok_lib.impl("neuralgrok_fused_full_step", _neuralgrok_fused_full_step_impl, "CUDA")


@torch.library.register_fake("supergrok::neuralgrok_fused_full_step")
def _neuralgrok_fused_full_step_abstract(
    param, grad, exp_avg, exp_avg_sq,
    W1, b1, W2, b2,
    alpha_amp, beta_amp, hidden_dim,
    beta1, beta2, lr, weight_decay, eps, bc1, bc2,
):
    return (
        torch.empty_like(param),
        torch.empty_like(exp_avg),
        torch.empty_like(exp_avg_sq),
    )


# ===================================================================
# 12. muon_fused_step  --  Muon momentum + NS + update
# ===================================================================

supergrok_lib.define(
    "muon_fused_step("
    "    Tensor param,"
    "    Tensor momentum_buffer,"
    "    Tensor grad,"
    "    float lr,"
    "    float momentum,"
    "    float weight_decay,"
    "    int ns_steps,"
    "    float a,"
    "    float b,"
    "    float c"
    ") -> (Tensor, Tensor)"
)


def _muon_fused_step_impl(
    param, momentum_buffer, grad,
    lr, momentum, weight_decay, ns_steps, a, b, c,
):
    ops = get_ops()
    ops.muon_fused_step(
        param, momentum_buffer, grad,
        lr, momentum, weight_decay, ns_steps, a, b, c,
    )
    return param, momentum_buffer


supergrok_lib.impl("muon_fused_step", _muon_fused_step_impl, "CUDA")


@torch.library.register_fake("supergrok::muon_fused_step")
def _muon_fused_step_abstract(
    param, momentum_buffer, grad,
    lr, momentum, weight_decay, ns_steps, a, b, c,
):
    return (torch.empty_like(param), torch.empty_like(momentum_buffer))
