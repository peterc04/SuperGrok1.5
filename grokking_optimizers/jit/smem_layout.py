"""Shared Memory Layout — compute padding to avoid bank conflicts.

NVIDIA GPUs have 32 banks of shared memory. If consecutive threads
access addresses that map to the same bank, a bank conflict occurs,
serializing the access. Adding padding (1 float per row) when
expert_hidden is a multiple of 32 staggerers the access pattern.

AMD GPUs have similar LDS bank behavior with 32 banks.
"""


def compute_smem_padding(
    expert_hidden: int,
    num_experts: int,
    bank_count: int = 32,
    bank_width: int = 4,
) -> int:
    """Compute shared memory padding to avoid bank conflicts.

    If expert_hidden is a multiple of bank_count, consecutive threads
    reading a row of expert weights will all hit the same bank.
    Adding 1 float of padding per row staggerers the access pattern.

    Args:
        expert_hidden: Width of expert MLP hidden layer.
        num_experts: Number of experts (affects total smem usage).
        bank_count: Number of shared memory banks (32 on NVIDIA, 32 on AMD).
        bank_width: Width of each bank in bytes (4 for float32).

    Returns:
        Number of float32 padding elements to add per row (0 or 1).
    """
    # Elements per bank cycle
    elements_per_cycle = bank_count * bank_width // 4  # 32 for float32

    if expert_hidden % elements_per_cycle == 0:
        return 1  # Add 1 float padding per expert row
    return 0


def compute_smem_size(
    expert_hidden: int,
    num_experts: int,
    gru_hidden: int,
    num_heads: int,
    pk_dim: int,
    padding: int = 0,
) -> int:
    """Compute total shared memory required for expert + meta-net weights.

    Layout:
      - Expert W1: num_experts * (expert_hidden + padding) * 3 * 4 bytes
      - Expert b1: num_experts * (expert_hidden + padding) * 4 bytes
      - Expert W2: num_experts * (expert_hidden + padding) * 4 bytes
      - Expert b2: num_experts * 4 bytes
      - GRU: gru_hidden * 6 * 4 bytes (Wz, bz, Wr, br, Wh, bh)
      - PEER keys: num_heads * pk_dim * 4 bytes

    Args:
        expert_hidden: Width of expert MLP hidden layer.
        num_experts: Number of experts.
        gru_hidden: GRU hidden size.
        num_heads: Number of PEER heads.
        pk_dim: PEER product key dimension.
        padding: Padding per row (from compute_smem_padding).

    Returns:
        Total shared memory in bytes.
    """
    eh_padded = expert_hidden + padding

    # Expert weights
    expert_w1 = num_experts * eh_padded * 3 * 4   # 3 inputs (grad, sharpness, scan_out)
    expert_b1 = num_experts * eh_padded * 4
    expert_w2 = num_experts * eh_padded * 4
    expert_b2 = num_experts * 4

    # GRU weights (Wz, bz, Wr, br, Wh, bh)
    gru_weights = gru_hidden * 6 * 4

    # PEER product keys
    peer_keys = num_heads * pk_dim * 4

    total = expert_w1 + expert_b1 + expert_w2 + expert_b2 + gru_weights + peer_keys
    return total
