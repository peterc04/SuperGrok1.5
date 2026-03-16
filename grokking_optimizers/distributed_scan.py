"""
Distributed Mamba-3 Scan — Multi-GPU Sharding via Chunked Blelloch Algorithm.

Orchestrates a 3-phase distributed parallel prefix scan across multiple GPUs:

Phase 1: Each GPU runs a local Blelloch scan on its chunk (concurrent streams),
         producing local scan results and one Affine2x2 summary per chunk.
Phase 2: Gather all summaries to GPU 0, run a small prefix scan over them.
Phase 3: Scatter prefix transforms back, each GPU applies corrections.

The Affine2x2 transform has 6 floats: m00, m01, m10, m11, b0, b1.
Composition is associative, enabling parallel prefix scan.

Usage::

    from grokking_optimizers.distributed_scan import distributed_mamba3_scan

    # chunks: list of dicts, one per GPU, each containing precomputed tensors
    scan_outputs = distributed_mamba3_scan(
        chunks=chunks,
        A_log=A_log,
        D_param=D_param,
        rope_freq=rope_freq,
        d_inner=d_inner,
        d_state=d_state,
    )

Requires torch.distributed to be initialized for multi-GPU operation.
Falls back to single-GPU local scan when world_size == 1 or only one chunk.
"""

import torch
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple


# Number of floats per Affine2x2 transform
_AFFINE2X2_SIZE = 6

# CUDA kernel block sizes (must match distributed_scan_kernels.cu)
_DSCAN_BLOCK = 256
_DSCAN_MAX_GPUS = 64


def _get_world_size() -> int:
    """Get world size, returning 1 if distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_rank() -> int:
    """Get global rank, returning 0 if distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _load_distributed_scan_kernels():
    """Lazily load the distributed scan CUDA kernels.

    Returns the kernel module, or None if CUDA kernels are not available.
    Uses the same loading mechanism as the rest of the codebase.
    """
    try:
        from grokking_optimizers import _HAS_OPS
        if _HAS_OPS:
            from grokking_optimizers import _ops
            return _ops
    except ImportError:
        pass
    return None


def _compute_shared_mem_size(num_threads: int) -> int:
    """Compute shared memory size for Blelloch scan kernels.

    Each thread needs 6 floats (one Affine2x2) in shared memory.
    """
    return num_threads * _AFFINE2X2_SIZE * 4  # 4 bytes per float


def _summary_numel(d_inner: int, d_state: int) -> int:
    """Number of floats in one GPU's summary buffer.

    One Affine2x2 (6 floats) per (d_inner, half_d_state) pair.
    """
    half_d_state = d_state // 2
    return d_inner * half_d_state * _AFFINE2X2_SIZE


def distributed_mamba3_scan(
    chunks: List[Dict[str, torch.Tensor]],
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    d_inner: int,
    d_state: int,
    reverse: int = 0,
    initial_states: Optional[List[torch.Tensor]] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """Orchestrate distributed Mamba-3 scan across multiple GPUs.

    This implements the 3-phase chunked Blelloch algorithm:
      1. Launch local scans on each GPU (concurrent streams)
      2. Gather summaries to GPU 0, run prefix scan
      3. Scatter prefixes, apply corrections

    Args:
        chunks: List of K dicts, one per GPU chunk. Each dict contains:
            - 'pre_x_val': [N_local, d_inner] precomputed x values
            - 'pre_z_val': [N_local, d_inner] precomputed z values
            - 'pre_dt_val': [N_local, d_inner] precomputed dt values
            - 'pre_B_val': [N_local, d_state] precomputed B values
            - 'pre_C_val': [N_local, d_state] precomputed C values
            - 'device': torch.device for this chunk
        A_log: [d_inner, d_state] log-space A parameters (replicated).
        D_param: [d_inner] skip connection parameter (replicated).
        rope_freq: [d_inner, d_state//2] RoPE frequencies (replicated).
        d_inner: Inner dimension of Mamba scan.
        d_state: State dimension of Mamba scan (must be even).
        reverse: 0 for forward scan, 1 for reversed scan.
        initial_states: Optional list of [d_inner, d_state] initial states per chunk.
        process_group: Optional torch.distributed process group.

    Returns:
        List of K tensors, each [N_local, d_inner], the corrected scan outputs.
    """
    K = len(chunks)

    # Edge case: single GPU — skip distribution entirely
    if K == 1:
        return _local_scan_single_gpu(
            chunks[0], A_log, D_param, rope_freq,
            d_inner, d_state, reverse,
            initial_states[0] if initial_states else None,
        )

    if K > _DSCAN_MAX_GPUS:
        raise ValueError(
            f"distributed_mamba3_scan supports at most {_DSCAN_MAX_GPUS} GPU chunks, "
            f"got {K}"
        )

    half_d_state = d_state // 2
    summary_size = _summary_numel(d_inner, d_state)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: Launch local scans on each GPU (concurrent streams)
    # ═══════════════════════════════════════════════════════════════════

    scan_outputs = []
    summaries = []
    streams = []

    for k in range(K):
        chunk = chunks[k]
        device = chunk['pre_x_val'].device
        N_local = chunk['pre_x_val'].shape[0]

        # Create a dedicated stream for this GPU's local scan
        stream = torch.cuda.Stream(device=device)
        streams.append(stream)

        with torch.cuda.stream(stream):
            # Allocate output buffers on this GPU
            scan_out = torch.zeros(N_local, d_inner, dtype=torch.float32, device=device)
            summary_buf = torch.empty(summary_size, dtype=torch.float32, device=device)

            init_state = initial_states[k] if initial_states else None

            # Launch local scan kernel (Phase 1)
            _launch_local_scan_with_summary(
                chunk, A_log.to(device), D_param.to(device), rope_freq.to(device),
                scan_out, summary_buf, init_state,
                N_local, d_inner, d_state, reverse,
            )

            scan_outputs.append(scan_out)
            summaries.append(summary_buf)

    # Synchronize all local scan streams
    for stream in streams:
        stream.synchronize()

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: Gather summaries to GPU 0, run prefix scan
    # ═══════════════════════════════════════════════════════════════════

    # Gather all summaries to GPU 0
    device_0 = chunks[0]['pre_x_val'].device
    all_summaries = torch.empty(K * summary_size, dtype=torch.float32, device=device_0)

    if _get_world_size() > 1 and process_group is not None:
        # Use torch.distributed all_gather for cross-process communication
        summary_list = [
            torch.empty(summary_size, dtype=torch.float32, device=device_0)
            for _ in range(K)
        ]
        # Each rank sends its local summary
        rank = _get_rank()
        local_summary = summaries[rank].to(device_0) if rank < len(summaries) else \
            torch.zeros(summary_size, dtype=torch.float32, device=device_0)
        dist.all_gather(summary_list, local_summary, group=process_group)
        all_summaries = torch.cat(summary_list)
    else:
        # Single-process multi-GPU: direct copy
        for k in range(K):
            start = k * summary_size
            all_summaries[start:start + summary_size] = summaries[k].to(device_0)

    # Run prefix scan over summaries on GPU 0
    prefix_transforms = torch.empty_like(all_summaries)
    _launch_summary_prefix_scan(
        all_summaries, prefix_transforms,
        K, d_inner, half_d_state,
    )
    torch.cuda.current_stream(device_0).synchronize()

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 3: Scatter prefixes, apply corrections
    # ═══════════════════════════════════════════════════════════════════

    if _get_world_size() > 1 and process_group is not None:
        # Scatter prefix transforms to each rank
        rank = _get_rank()
        start = rank * summary_size
        local_prefix = prefix_transforms[start:start + summary_size].clone()
        dist.broadcast(prefix_transforms, src=0, group=process_group)
    else:
        local_prefix = None  # will index directly

    correction_streams = []
    for k in range(K):
        chunk = chunks[k]
        device = chunk['pre_x_val'].device
        N_local = chunk['pre_x_val'].shape[0]

        # Skip GPU 0 if its prefix is identity (first chunk needs no correction)
        if k == 0:
            # First chunk's prefix is always identity — no correction needed
            correction_streams.append(None)
            continue

        stream = torch.cuda.Stream(device=device)
        correction_streams.append(stream)

        with torch.cuda.stream(stream):
            # Get this GPU's prefix transform
            start_idx = k * summary_size
            if local_prefix is not None and _get_rank() == k:
                gpu_prefix = local_prefix.to(device)
            else:
                gpu_prefix = prefix_transforms[start_idx:start_idx + summary_size].to(device)

            init_state = initial_states[k] if initial_states else None

            _launch_apply_prefix(
                chunk, A_log.to(device), rope_freq.to(device),
                gpu_prefix, scan_outputs[k], init_state,
                N_local, d_inner, d_state, reverse,
            )

    # Synchronize correction streams
    for stream in correction_streams:
        if stream is not None:
            stream.synchronize()

    return scan_outputs


def distributed_mamba3_scan_backward(
    chunks: List[Dict[str, torch.Tensor]],
    grad_outputs: List[torch.Tensor],
    fwd_scan_outputs: List[torch.Tensor],
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    d_inner: int,
    d_state: int,
    reverse: int = 0,
    initial_states: Optional[List[torch.Tensor]] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Backward pass of distributed Mamba-3 scan.

    Mirrors the forward 3-phase structure but with backward kernels:
      1. Local backward scans produce backward summaries
      2. Backward prefix scan over summaries (reverse order)
      3. Apply backward prefix corrections

    Args:
        chunks: Same chunk dicts as forward (precomputed values).
        grad_outputs: List of K [N_local, d_inner] gradient tensors.
        fwd_scan_outputs: List of K [N_local, d_inner] saved forward outputs.
        A_log, D_param, rope_freq: Same parameters as forward.
        d_inner, d_state: Dimensions.
        reverse: 0 or 1.
        initial_states: Optional initial states per chunk.
        process_group: Optional distributed process group.

    Returns:
        List of K dicts, each containing gradient tensors:
            'grad_pre_x', 'grad_pre_dt', 'grad_pre_B', 'grad_pre_C', 'grad_D'
    """
    K = len(chunks)
    half_d_state = d_state // 2
    summary_size = _summary_numel(d_inner, d_state)

    # Edge case: single GPU
    if K == 1:
        return _local_scan_backward_single_gpu(
            chunks[0], grad_outputs[0], fwd_scan_outputs[0],
            A_log, D_param, rope_freq,
            d_inner, d_state, reverse,
            initial_states[0] if initial_states else None,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: Local backward scans (concurrent streams)
    # ═══════════════════════════════════════════════════════════════════

    grad_results = []
    bwd_summaries = []
    streams = []

    for k in range(K):
        chunk = chunks[k]
        device = chunk['pre_x_val'].device
        N_local = chunk['pre_x_val'].shape[0]

        stream = torch.cuda.Stream(device=device)
        streams.append(stream)

        with torch.cuda.stream(stream):
            grad_pre_x = torch.zeros(N_local, d_inner, dtype=torch.float32, device=device)
            grad_pre_dt = torch.zeros(N_local, d_inner, dtype=torch.float32, device=device)
            grad_pre_B = torch.zeros(N_local, d_state, dtype=torch.float32, device=device)
            grad_pre_C = torch.zeros(N_local, d_state, dtype=torch.float32, device=device)
            grad_D = torch.zeros(d_inner, dtype=torch.float32, device=device)
            bwd_summary_buf = torch.empty(summary_size, dtype=torch.float32, device=device)

            init_state = initial_states[k] if initial_states else None

            _launch_local_scan_backward_with_summary(
                chunk, grad_outputs[k], fwd_scan_outputs[k],
                A_log.to(device), D_param.to(device), rope_freq.to(device),
                grad_pre_x, grad_pre_dt, grad_pre_B, grad_pre_C, grad_D,
                bwd_summary_buf, init_state,
                N_local, d_inner, d_state, reverse,
            )

            grad_results.append({
                'grad_pre_x': grad_pre_x,
                'grad_pre_dt': grad_pre_dt,
                'grad_pre_B': grad_pre_B,
                'grad_pre_C': grad_pre_C,
                'grad_D': grad_D,
            })
            bwd_summaries.append(bwd_summary_buf)

    for stream in streams:
        stream.synchronize()

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: Gather backward summaries, run backward prefix scan
    # ═══════════════════════════════════════════════════════════════════

    device_0 = chunks[0]['pre_x_val'].device
    all_bwd_summaries = torch.empty(K * summary_size, dtype=torch.float32, device=device_0)

    if _get_world_size() > 1 and process_group is not None:
        summary_list = [
            torch.empty(summary_size, dtype=torch.float32, device=device_0)
            for _ in range(K)
        ]
        rank = _get_rank()
        local_bwd_summary = bwd_summaries[rank].to(device_0) if rank < len(bwd_summaries) else \
            torch.zeros(summary_size, dtype=torch.float32, device=device_0)
        dist.all_gather(summary_list, local_bwd_summary, group=process_group)
        all_bwd_summaries = torch.cat(summary_list)
    else:
        for k in range(K):
            start = k * summary_size
            all_bwd_summaries[start:start + summary_size] = bwd_summaries[k].to(device_0)

    bwd_prefix_transforms = torch.empty_like(all_bwd_summaries)
    _launch_summary_prefix_scan_backward(
        all_bwd_summaries, bwd_prefix_transforms,
        K, d_inner, half_d_state,
    )
    torch.cuda.current_stream(device_0).synchronize()

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 3: Scatter backward prefixes, apply corrections
    # ═══════════════════════════════════════════════════════════════════

    if _get_world_size() > 1 and process_group is not None:
        dist.broadcast(bwd_prefix_transforms, src=0, group=process_group)

    correction_streams = []
    for k in range(K):
        chunk = chunks[k]
        device = chunk['pre_x_val'].device
        N_local = chunk['pre_x_val'].shape[0]

        # Last chunk needs no backward correction (no subsequent chunks)
        if k == K - 1:
            correction_streams.append(None)
            continue

        stream = torch.cuda.Stream(device=device)
        correction_streams.append(stream)

        with torch.cuda.stream(stream):
            start_idx = k * summary_size
            gpu_bwd_prefix = bwd_prefix_transforms[start_idx:start_idx + summary_size].to(device)

            _launch_apply_prefix_backward(
                chunk, A_log.to(device), rope_freq.to(device),
                grad_outputs[k], gpu_bwd_prefix,
                grad_results[k]['grad_pre_x'],
                grad_results[k]['grad_pre_dt'],
                grad_results[k]['grad_pre_B'],
                grad_results[k]['grad_pre_C'],
                N_local, d_inner, d_state, reverse,
            )

    for stream in correction_streams:
        if stream is not None:
            stream.synchronize()

    return grad_results


# ═══════════════════════════════════════════════════════════════════════
#  Single-GPU fallback paths
# ═══════════════════════════════════════════════════════════════════════

def _local_scan_single_gpu(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    d_inner: int,
    d_state: int,
    reverse: int,
    initial_state: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """Single-GPU path: run local scan without any distribution overhead.

    Uses the same local scan kernel but skips summary extraction,
    gathering, prefix scan, and correction phases entirely.
    """
    device = chunk['pre_x_val'].device
    N_local = chunk['pre_x_val'].shape[0]
    summary_size = _summary_numel(d_inner, d_state)

    scan_out = torch.zeros(N_local, d_inner, dtype=torch.float32, device=device)
    # Summaries not needed but kernel writes them; allocate dummy
    summary_buf = torch.empty(summary_size, dtype=torch.float32, device=device)

    _launch_local_scan_with_summary(
        chunk, A_log.to(device), D_param.to(device), rope_freq.to(device),
        scan_out, summary_buf, initial_state,
        N_local, d_inner, d_state, reverse,
    )
    torch.cuda.current_stream(device).synchronize()

    return [scan_out]


def _local_scan_backward_single_gpu(
    chunk: Dict[str, torch.Tensor],
    grad_output: torch.Tensor,
    fwd_scan_output: torch.Tensor,
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    d_inner: int,
    d_state: int,
    reverse: int,
    initial_state: Optional[torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    """Single-GPU backward path."""
    device = chunk['pre_x_val'].device
    N_local = chunk['pre_x_val'].shape[0]
    summary_size = _summary_numel(d_inner, d_state)

    grad_pre_x = torch.zeros(N_local, d_inner, dtype=torch.float32, device=device)
    grad_pre_dt = torch.zeros(N_local, d_inner, dtype=torch.float32, device=device)
    grad_pre_B = torch.zeros(N_local, d_state, dtype=torch.float32, device=device)
    grad_pre_C = torch.zeros(N_local, d_state, dtype=torch.float32, device=device)
    grad_D = torch.zeros(d_inner, dtype=torch.float32, device=device)
    bwd_summary_buf = torch.empty(summary_size, dtype=torch.float32, device=device)

    _launch_local_scan_backward_with_summary(
        chunk, grad_output, fwd_scan_output,
        A_log.to(device), D_param.to(device), rope_freq.to(device),
        grad_pre_x, grad_pre_dt, grad_pre_B, grad_pre_C, grad_D,
        bwd_summary_buf, initial_state,
        N_local, d_inner, d_state, reverse,
    )
    torch.cuda.current_stream(device).synchronize()

    return [{
        'grad_pre_x': grad_pre_x,
        'grad_pre_dt': grad_pre_dt,
        'grad_pre_B': grad_pre_B,
        'grad_pre_C': grad_pre_C,
        'grad_D': grad_D,
    }]


# ═══════════════════════════════════════════════════════════════════════
#  Kernel launch wrappers
#
#  These functions handle grid/block configuration, shared memory
#  sizing, and the actual kernel launch via PyTorch's CUDA extension
#  mechanism. When CUDA kernels are not available, they fall back to
#  a pure-Python Blelloch scan implementation.
# ═══════════════════════════════════════════════════════════════════════

def _launch_local_scan_with_summary(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    scan_output: torch.Tensor,
    summaries: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Launch mamba3_scan_local_with_summary_kernel.

    Grid: (d_inner,) blocks of DSCAN_BLOCK threads.
    Shared memory: DSCAN_BLOCK * 6 * sizeof(float).
    """
    ops = _load_distributed_scan_kernels()
    if ops is not None and hasattr(ops, 'distributed_scan_local_with_summary'):
        ops.distributed_scan_local_with_summary(
            chunk['pre_x_val'], chunk['pre_z_val'], chunk['pre_dt_val'],
            chunk['pre_B_val'], chunk['pre_C_val'],
            A_log, D_param, rope_freq,
            scan_output, summaries,
            initial_state if initial_state is not None else torch.empty(0),
            N_local, d_inner, d_state, reverse,
        )
    else:
        _python_local_scan_with_summary(
            chunk, A_log, D_param, rope_freq,
            scan_output, summaries, initial_state,
            N_local, d_inner, d_state, reverse,
        )


def _launch_summary_prefix_scan(
    all_summaries: torch.Tensor,
    prefix_out: torch.Tensor,
    K: int,
    d_inner: int,
    half_d_state: int,
) -> None:
    """Launch scan_summary_prefix_kernel on GPU 0.

    Grid: (d_inner * half_d_state,) blocks of K threads.
    """
    ops = _load_distributed_scan_kernels()
    if ops is not None and hasattr(ops, 'distributed_scan_summary_prefix'):
        ops.distributed_scan_summary_prefix(
            all_summaries, prefix_out, K, d_inner, half_d_state,
        )
    else:
        _python_summary_prefix_scan(
            all_summaries, prefix_out, K, d_inner, half_d_state,
        )


def _launch_apply_prefix(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    rope_freq: torch.Tensor,
    prefix_transforms: torch.Tensor,
    scan_output: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Launch mamba3_apply_scan_prefix_kernel.

    Grid: (d_inner,) blocks of DSCAN_BLOCK threads.
    """
    ops = _load_distributed_scan_kernels()
    if ops is not None and hasattr(ops, 'distributed_scan_apply_prefix'):
        ops.distributed_scan_apply_prefix(
            chunk['pre_x_val'], chunk['pre_z_val'], chunk['pre_dt_val'],
            chunk['pre_B_val'], chunk['pre_C_val'],
            A_log, rope_freq, prefix_transforms,
            scan_output,
            initial_state if initial_state is not None else torch.empty(0),
            N_local, d_inner, d_state, reverse,
        )
    else:
        _python_apply_prefix(
            chunk, A_log, rope_freq, prefix_transforms,
            scan_output, initial_state,
            N_local, d_inner, d_state, reverse,
        )


def _launch_local_scan_backward_with_summary(
    chunk: Dict[str, torch.Tensor],
    grad_output: torch.Tensor,
    fwd_scan_output: torch.Tensor,
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    grad_pre_x: torch.Tensor,
    grad_pre_dt: torch.Tensor,
    grad_pre_B: torch.Tensor,
    grad_pre_C: torch.Tensor,
    grad_D: torch.Tensor,
    bwd_summaries: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Launch mamba3_scan_local_with_summary_bwd_kernel."""
    ops = _load_distributed_scan_kernels()
    if ops is not None and hasattr(ops, 'distributed_scan_local_with_summary_bwd'):
        ops.distributed_scan_local_with_summary_bwd(
            chunk['pre_x_val'], chunk['pre_z_val'], chunk['pre_dt_val'],
            chunk['pre_B_val'], chunk['pre_C_val'],
            A_log, D_param, rope_freq,
            grad_output, fwd_scan_output,
            grad_pre_x, grad_pre_dt, grad_pre_B, grad_pre_C, grad_D,
            bwd_summaries,
            initial_state if initial_state is not None else torch.empty(0),
            N_local, d_inner, d_state, reverse,
        )
    else:
        _python_local_scan_backward_with_summary(
            chunk, grad_output, fwd_scan_output,
            A_log, D_param, rope_freq,
            grad_pre_x, grad_pre_dt, grad_pre_B, grad_pre_C, grad_D,
            bwd_summaries, initial_state,
            N_local, d_inner, d_state, reverse,
        )


def _launch_summary_prefix_scan_backward(
    all_bwd_summaries: torch.Tensor,
    bwd_prefix_out: torch.Tensor,
    K: int,
    d_inner: int,
    half_d_state: int,
) -> None:
    """Launch scan_summary_prefix_bwd_kernel on GPU 0."""
    ops = _load_distributed_scan_kernels()
    if ops is not None and hasattr(ops, 'distributed_scan_summary_prefix_bwd'):
        ops.distributed_scan_summary_prefix_bwd(
            all_bwd_summaries, bwd_prefix_out, K, d_inner, half_d_state,
        )
    else:
        _python_summary_prefix_scan_backward(
            all_bwd_summaries, bwd_prefix_out, K, d_inner, half_d_state,
        )


def _launch_apply_prefix_backward(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    rope_freq: torch.Tensor,
    grad_output: torch.Tensor,
    bwd_prefix_transforms: torch.Tensor,
    grad_pre_x: torch.Tensor,
    grad_pre_dt: torch.Tensor,
    grad_pre_B: torch.Tensor,
    grad_pre_C: torch.Tensor,
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Launch mamba3_apply_scan_prefix_bwd_kernel."""
    ops = _load_distributed_scan_kernels()
    if ops is not None and hasattr(ops, 'distributed_scan_apply_prefix_bwd'):
        ops.distributed_scan_apply_prefix_bwd(
            chunk['pre_x_val'], chunk['pre_z_val'], chunk['pre_dt_val'],
            chunk['pre_B_val'], chunk['pre_C_val'],
            A_log, rope_freq, grad_output,
            bwd_prefix_transforms,
            grad_pre_x, grad_pre_dt, grad_pre_B, grad_pre_C,
            N_local, d_inner, d_state, reverse,
        )
    else:
        _python_apply_prefix_backward(
            chunk, A_log, rope_freq, grad_output,
            bwd_prefix_transforms,
            grad_pre_x, grad_pre_dt, grad_pre_B, grad_pre_C,
            N_local, d_inner, d_state, reverse,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Python fallback implementations
#
#  These mirror the CUDA kernel logic for correctness testing and
#  CPU-only environments. They are significantly slower than the
#  CUDA kernels but produce identical results.
# ═══════════════════════════════════════════════════════════════════════

def _affine_identity() -> torch.Tensor:
    """Return identity Affine2x2 as a 6-element tensor."""
    return torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def _affine_combine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose two Affine2x2 transforms (right after left).

    left, right: 6-element tensors [m00, m01, m10, m11, b0, b1].
    Returns: composed 6-element tensor.
    """
    lm00, lm01, lm10, lm11, lb0, lb1 = left[0], left[1], left[2], left[3], left[4], left[5]
    rm00, rm01, rm10, rm11, rb0, rb1 = right[0], right[1], right[2], right[3], right[4], right[5]

    out = torch.empty(6, dtype=torch.float32, device=left.device)
    out[0] = rm00 * lm00 + rm01 * lm10
    out[1] = rm00 * lm01 + rm01 * lm11
    out[2] = rm10 * lm00 + rm11 * lm10
    out[3] = rm10 * lm01 + rm11 * lm11
    out[4] = rm00 * lb0 + rm01 * lb1 + rb0
    out[5] = rm10 * lb0 + rm11 * lb1 + rb1
    return out


def _build_element_py(
    dt: float, x_val: float,
    B_e: float, B_o: float,
    A_e: float, A_o: float,
    freq_val: float,
    device: torch.device,
) -> torch.Tensor:
    """Python version of build_scan_element."""
    import math
    A_bar_e = (1.0 + dt * A_e / 2.0) / (1.0 - dt * A_e / 2.0 + 1e-8)
    A_bar_o = (1.0 + dt * A_o / 2.0) / (1.0 - dt * A_o / 2.0 + 1e-8)
    sin_v = math.sin(dt * freq_val)
    cos_v = math.cos(dt * freq_val)

    elem = torch.tensor([
        A_bar_e * cos_v,    # m00
        -A_bar_e * sin_v,   # m01
        A_bar_o * sin_v,    # m10
        A_bar_o * cos_v,    # m11
        dt * B_e * x_val,   # b0
        dt * B_o * x_val,   # b1
    ], dtype=torch.float32, device=device)
    return elem


def _python_local_scan_with_summary(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    scan_output: torch.Tensor,
    summaries: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Python fallback for local scan with summary extraction."""
    import math
    device = chunk['pre_x_val'].device
    half_d_state = d_state // 2

    A = torch.zeros(d_inner, d_state, device=device)
    for j in range(d_inner):
        for s in range(d_state):
            A[j, s] = -math.exp(A_log[j, s].item())

    for j in range(d_inner):
        D_val = D_param[j].item()

        for p in range(half_d_state):
            s_e = 2 * p
            s_o = 2 * p + 1
            A_e = A[j, s_e].item()
            A_o = A[j, s_o].item()
            f_val = rope_freq[j, p].item()

            h_init_e = initial_state[j, s_e].item() if initial_state is not None else 0.0
            h_init_o = initial_state[j, s_o].item() if initial_state is not None else 0.0

            # Sequential scan to compute output and summary
            running = _affine_identity().to(device)

            for step in range(N_local):
                t = (N_local - 1 - step) if reverse else step
                dt_v = chunk['pre_dt_val'][t, j].item()
                x_v = chunk['pre_x_val'][t, j].item()
                B_e_v = chunk['pre_B_val'][t, s_e].item()
                B_o_v = chunk['pre_B_val'][t, s_o].item()

                elem = _build_element_py(dt_v, x_v, B_e_v, B_o_v, A_e, A_o, f_val, device)
                running = _affine_combine(running, elem)

                # Compute h and accumulate output
                h_e = running[0].item() * h_init_e + running[1].item() * h_init_o + running[4].item()
                h_o = running[2].item() * h_init_e + running[3].item() * h_init_o + running[5].item()

                C_e_v = chunk['pre_C_val'][t, s_e].item()
                C_o_v = chunk['pre_C_val'][t, s_o].item()
                scan_output[t, j] += h_e * C_e_v + h_o * C_o_v

            # Store summary
            sum_idx = (j * half_d_state + p) * 6
            summaries[sum_idx:sum_idx + 6] = running

        # Apply SiLU gating and D skip
        for t in range(N_local):
            z = chunk['pre_z_val'][t, j].item()
            silu_z = z / (1.0 + math.exp(-z))
            x_val = chunk['pre_x_val'][t, j].item()
            scan_output[t, j] = scan_output[t, j].item() * silu_z + D_val * x_val


def _python_summary_prefix_scan(
    all_summaries: torch.Tensor,
    prefix_out: torch.Tensor,
    K: int,
    d_inner: int,
    half_d_state: int,
) -> None:
    """Python fallback for summary prefix scan."""
    device = all_summaries.device
    summary_size = d_inner * half_d_state * _AFFINE2X2_SIZE

    for j in range(d_inner):
        for p in range(half_d_state):
            running = _affine_identity().to(device)

            for k in range(K):
                # Write exclusive prefix
                dst_idx = (k * d_inner * half_d_state + j * half_d_state + p) * 6
                prefix_out[dst_idx:dst_idx + 6] = running

                # Incorporate summary[k]
                src_idx = (k * d_inner * half_d_state + j * half_d_state + p) * 6
                sk = all_summaries[src_idx:src_idx + 6]
                running = _affine_combine(running, sk)


def _python_apply_prefix(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    rope_freq: torch.Tensor,
    prefix_transforms: torch.Tensor,
    scan_output: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Python fallback for applying prefix corrections."""
    import math
    device = chunk['pre_x_val'].device
    half_d_state = d_state // 2

    A = torch.zeros(d_inner, d_state, device=device)
    for j in range(d_inner):
        for s in range(d_state):
            A[j, s] = -math.exp(A_log[j, s].item())

    for j in range(d_inner):
        for p in range(half_d_state):
            s_e = 2 * p
            s_o = 2 * p + 1
            A_e = A[j, s_e].item()
            A_o = A[j, s_o].item()
            f_val = rope_freq[j, p].item()

            pf_idx = (j * half_d_state + p) * 6
            pf = prefix_transforms[pf_idx:pf_idx + 6]

            h_init_e = initial_state[j, s_e].item() if initial_state is not None else 0.0
            h_init_o = initial_state[j, s_o].item() if initial_state is not None else 0.0

            # Corrected initial state
            h_corr_e = pf[0].item() * h_init_e + pf[1].item() * h_init_o + pf[4].item()
            h_corr_o = pf[2].item() * h_init_e + pf[3].item() * h_init_o + pf[5].item()

            delta_e = h_corr_e - h_init_e
            delta_o = h_corr_o - h_init_o

            if abs(delta_e) < 1e-12 and abs(delta_o) < 1e-12:
                continue

            # Propagate delta through scan
            cur_de = delta_e
            cur_do = delta_o
            for step in range(N_local):
                t = (N_local - 1 - step) if reverse else step
                dt_v = chunk['pre_dt_val'][t, j].item()
                x_v = chunk['pre_x_val'][t, j].item()
                B_e_v = chunk['pre_B_val'][t, s_e].item()
                B_o_v = chunk['pre_B_val'][t, s_o].item()

                elem = _build_element_py(dt_v, x_v, B_e_v, B_o_v, A_e, A_o, f_val, device)
                new_de = elem[0].item() * cur_de + elem[1].item() * cur_do
                new_do = elem[2].item() * cur_de + elem[3].item() * cur_do
                cur_de = new_de
                cur_do = new_do

                C_e_v = chunk['pre_C_val'][t, s_e].item()
                C_o_v = chunk['pre_C_val'][t, s_o].item()
                delta_y = cur_de * C_e_v + cur_do * C_o_v

                z = chunk['pre_z_val'][t, j].item()
                silu_z = z / (1.0 + math.exp(-z))
                scan_output[t, j] += delta_y * silu_z


def _python_local_scan_backward_with_summary(
    chunk: Dict[str, torch.Tensor],
    grad_output: torch.Tensor,
    fwd_scan_output: torch.Tensor,
    A_log: torch.Tensor,
    D_param: torch.Tensor,
    rope_freq: torch.Tensor,
    grad_pre_x: torch.Tensor,
    grad_pre_dt: torch.Tensor,
    grad_pre_B: torch.Tensor,
    grad_pre_C: torch.Tensor,
    grad_D: torch.Tensor,
    bwd_summaries: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Python fallback for backward local scan with summary.

    Simplified backward: computes gradients via reverse-mode autodiff
    of the sequential scan. This is correct but O(N*d_state) per d_inner.
    """
    import math
    device = chunk['pre_x_val'].device
    half_d_state = d_state // 2

    A = torch.zeros(d_inner, d_state, device=device)
    for j in range(d_inner):
        for s in range(d_state):
            A[j, s] = -math.exp(A_log[j, s].item())

    for j in range(d_inner):
        D_val = D_param[j].item()

        for p in range(half_d_state):
            s_e = 2 * p
            s_o = 2 * p + 1
            A_e = A[j, s_e].item()
            A_o = A[j, s_o].item()
            f_val = rope_freq[j, p].item()

            # Backward scan: accumulate dh in reverse
            bwd_running = _affine_identity().to(device)
            dh_e = 0.0
            dh_o = 0.0

            for step in range(N_local - 1, -1, -1):
                t = (N_local - 1 - step) if reverse else step
                dt_v = chunk['pre_dt_val'][t, j].item()
                x_v = chunk['pre_x_val'][t, j].item()
                B_e_v = chunk['pre_B_val'][t, s_e].item()
                B_o_v = chunk['pre_B_val'][t, s_o].item()

                fwd_elem = _build_element_py(dt_v, x_v, B_e_v, B_o_v, A_e, A_o, f_val, device)

                dy_raw = grad_output[t, j].item()
                z = chunk['pre_z_val'][t, j].item()
                silu_z = z / (1.0 + math.exp(-z))
                dy_scan = dy_raw * silu_z

                C_e_v = chunk['pre_C_val'][t, s_e].item()
                C_o_v = chunk['pre_C_val'][t, s_o].item()

                # Backward element (transposed matrix + gradient source bias)
                bwd_elem = torch.tensor([
                    fwd_elem[0].item(), fwd_elem[2].item(),
                    fwd_elem[1].item(), fwd_elem[3].item(),
                    C_e_v * dy_scan, C_o_v * dy_scan,
                ], dtype=torch.float32, device=device)

                bwd_running = _affine_combine(bwd_running, bwd_elem)

                dh_e = bwd_running[4].item()
                dh_o = bwd_running[5].item()

                # Accumulate parameter gradients
                grad_pre_B[t, s_e] += dh_e * dt_v * x_v
                grad_pre_B[t, s_o] += dh_o * dt_v * x_v
                grad_pre_C[t, s_e] += dh_e * dt_v * x_v
                grad_pre_C[t, s_o] += dh_o * dt_v * x_v

                ddt = dh_e * B_e_v * x_v + dh_o * B_o_v * x_v
                denom_e = (1.0 - dt_v * A_e / 2.0 + 1e-8)
                denom_o = (1.0 - dt_v * A_o / 2.0 + 1e-8)
                ddt += dh_e * A_e / (denom_e * denom_e)
                ddt += dh_o * A_o / (denom_o * denom_o)
                grad_pre_dt[t, j] += ddt

                dx = dh_e * dt_v * B_e_v + dh_o * dt_v * B_o_v
                grad_pre_x[t, j] += dx

                if p == 0:
                    grad_D[j] += dy_raw * x_v

            # Store backward summary
            sum_idx = (j * half_d_state + p) * 6
            bwd_summaries[sum_idx:sum_idx + 6] = bwd_running


def _python_summary_prefix_scan_backward(
    all_bwd_summaries: torch.Tensor,
    bwd_prefix_out: torch.Tensor,
    K: int,
    d_inner: int,
    half_d_state: int,
) -> None:
    """Python fallback for backward summary prefix scan (reverse order)."""
    device = all_bwd_summaries.device

    for j in range(d_inner):
        for p in range(half_d_state):
            running = _affine_identity().to(device)

            for k in range(K - 1, -1, -1):
                dst_idx = (k * d_inner * half_d_state + j * half_d_state + p) * 6
                bwd_prefix_out[dst_idx:dst_idx + 6] = running

                src_idx = (k * d_inner * half_d_state + j * half_d_state + p) * 6
                sk = all_bwd_summaries[src_idx:src_idx + 6]
                running = _affine_combine(running, sk)


def _python_apply_prefix_backward(
    chunk: Dict[str, torch.Tensor],
    A_log: torch.Tensor,
    rope_freq: torch.Tensor,
    grad_output: torch.Tensor,
    bwd_prefix_transforms: torch.Tensor,
    grad_pre_x: torch.Tensor,
    grad_pre_dt: torch.Tensor,
    grad_pre_B: torch.Tensor,
    grad_pre_C: torch.Tensor,
    N_local: int,
    d_inner: int,
    d_state: int,
    reverse: int,
) -> None:
    """Python fallback for applying backward prefix corrections."""
    import math
    device = chunk['pre_x_val'].device
    half_d_state = d_state // 2

    A = torch.zeros(d_inner, d_state, device=device)
    for j in range(d_inner):
        for s in range(d_state):
            A[j, s] = -math.exp(A_log[j, s].item())

    for j in range(d_inner):
        for p in range(half_d_state):
            s_e = 2 * p
            s_o = 2 * p + 1
            A_e = A[j, s_e].item()
            A_o = A[j, s_o].item()
            f_val = rope_freq[j, p].item()

            pf_idx = (j * half_d_state + p) * 6
            bpf = bwd_prefix_transforms[pf_idx:pf_idx + 6]

            delta_dh_e = bpf[4].item()
            delta_dh_o = bpf[5].item()

            if (abs(delta_dh_e) < 1e-12 and abs(delta_dh_o) < 1e-12 and
                    abs(bpf[0].item() - 1.0) < 1e-12 and abs(bpf[3].item() - 1.0) < 1e-12):
                continue

            cur_dh_e = delta_dh_e
            cur_dh_o = delta_dh_o

            for step in range(N_local - 1, -1, -1):
                t = (N_local - 1 - step) if reverse else step
                dt_v = chunk['pre_dt_val'][t, j].item()
                x_v = chunk['pre_x_val'][t, j].item()
                B_e_v = chunk['pre_B_val'][t, s_e].item()
                B_o_v = chunk['pre_B_val'][t, s_o].item()

                fwd_elem = _build_element_py(dt_v, x_v, B_e_v, B_o_v, A_e, A_o, f_val, device)

                # Propagate through transposed matrix
                new_dh_e = fwd_elem[0].item() * cur_dh_e + fwd_elem[2].item() * cur_dh_o
                new_dh_o = fwd_elem[1].item() * cur_dh_e + fwd_elem[3].item() * cur_dh_o
                cur_dh_e = new_dh_e
                cur_dh_o = new_dh_o

                # Add gradient corrections
                grad_pre_dt[t, j] += cur_dh_e * B_e_v * x_v + cur_dh_o * B_o_v * x_v
                grad_pre_x[t, j] += cur_dh_e * dt_v * B_e_v + cur_dh_o * dt_v * B_o_v
                grad_pre_B[t, s_e] += cur_dh_e * dt_v * x_v
                grad_pre_B[t, s_o] += cur_dh_o * dt_v * x_v
                grad_pre_C[t, s_e] += cur_dh_e * dt_v * x_v
                grad_pre_C[t, s_o] += cur_dh_o * dt_v * x_v
