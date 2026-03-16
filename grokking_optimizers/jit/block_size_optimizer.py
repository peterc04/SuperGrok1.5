"""Block Size Optimizer — compute optimal CUDA block sizes per parameter group.

Maps parameter sizes to block configurations that maximize GPU occupancy:
  - Small params (<1024 elements): block=64, many blocks per SM
  - Medium params (1K-100K): block=256 (default, good occupancy)
  - Large params (>100K): block=256 or 512 depending on register pressure
"""

from typing import Dict, List


def compute_optimal_block_sizes(
    param_sizes: List[int],
    sm_count: int,
    max_threads: int,
) -> Dict[str, int]:
    """Compute optimal CUDA block size for each kernel type.

    Args:
        param_sizes: List of parameter tensor element counts.
        sm_count: Number of streaming multiprocessors on the GPU.
        max_threads: Maximum threads per SM.

    Returns:
        Dict mapping kernel type ('scan', 'elem', 'persistent', 'distributed')
        to block size (number of threads per block).
    """
    if not param_sizes:
        return {'scan': 16, 'elem': 256, 'persistent': 256, 'distributed': 256}

    # Determine the dominant parameter size range
    total_params = sum(param_sizes)
    avg_param_size = total_params // max(1, len(param_sizes))

    # Scan kernel: processes d_inner elements per block
    # Block size = d_inner (typically 16), but we use 16 as minimum
    scan_block = 16

    # Fused elem kernel: processes one element per thread
    # Target: enough blocks to fill all SMs with 2+ blocks each
    if avg_param_size < 1024:
        elem_block = 64   # Small params: more blocks, less threads
    elif avg_param_size < 100_000:
        elem_block = 256  # Medium: standard
    else:
        # Large params: check if 512 threads gives enough occupancy
        # 512 threads * 32 regs = 16384 regs, fits 4 blocks on most SMs
        if max_threads >= 2048:
            elem_block = 256  # Prefer more blocks for better latency hiding
        else:
            elem_block = 256

    # Persistent kernel: one block per SM, max threads
    persistent_block = min(256, max_threads // 4)

    # Distributed kernel: same as elem
    distributed_block = elem_block

    return {
        'scan': scan_block,
        'elem': elem_block,
        'persistent': persistent_block,
        'distributed': distributed_block,
    }


def compute_grid_size(
    num_elements: int,
    block_size: int,
    sm_count: int,
    max_blocks_per_sm: int = 8,
) -> int:
    """Compute grid size (number of blocks) for a kernel launch.

    Caps grid size to avoid oversaturation beyond what the GPU can schedule.

    Args:
        num_elements: Total elements to process.
        block_size: Threads per block.
        sm_count: Number of SMs.
        max_blocks_per_sm: Maximum concurrent blocks per SM.

    Returns:
        Grid size (number of blocks).
    """
    min_grid = (num_elements + block_size - 1) // block_size
    max_grid = sm_count * max_blocks_per_sm
    return min(min_grid, max_grid) if max_grid > 0 else min_grid
