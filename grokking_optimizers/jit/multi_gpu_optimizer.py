"""Multi-GPU Interconnect Probing and Strategy Selection.

Measures actual interconnect bandwidth and latency at initialization,
then uses these measurements to decide:
  - Whether to distribute the scan across GPUs (or replicate)
  - Optimal chunk size for distributed scan
  - Whether to use NCCL all-gather or peer-to-peer for summaries

NVLink (>600 Gbps) vs PCIe (<64 Gbps) dramatically affects the
break-even point for distributed scan: with NVLink, even small
models benefit; with PCIe, only N_total > ~10K is worth distributing.
"""

import torch
from dataclasses import dataclass
from typing import Optional

from .specializer import ModelConfig


@dataclass
class TopologyConfig:
    """Measured interconnect properties."""
    bandwidth_gbps: float = 0.0
    latency_us: float = float('inf')
    is_nvlink: bool = False
    is_nvswitch: bool = False
    peer_to_peer: bool = False


class MultiGPUSpecializer:
    """Probes interconnect topology and computes optimal multi-GPU strategy."""

    # All-gather message size for scan summaries: 6 floats * 4 bytes = 24 bytes per GPU
    SUMMARY_BYTES = 24

    def __init__(self, world_size: int, model_config: ModelConfig):
        self.world_size = world_size
        self.config = model_config
        self.topology = self._probe_topology()

    def _probe_topology(self) -> TopologyConfig:
        """Measure actual interconnect bandwidth and latency.

        Uses two all-reduce operations:
          1. Small message (64 bytes): measures latency
          2. Large message (1 MB): measures bandwidth

        Returns TopologyConfig with measured values.
        """
        if not torch.distributed.is_initialized():
            return TopologyConfig()

        if not torch.cuda.is_available():
            return TopologyConfig()

        device = torch.device('cuda')

        # Warmup
        warmup = torch.zeros(16, device=device)
        torch.distributed.all_reduce(warmup)
        torch.cuda.synchronize()

        # Small message: measure latency
        small_tensor = torch.zeros(16, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Average over multiple iterations for stability
        n_iters = 10
        start.record()
        for _ in range(n_iters):
            torch.distributed.all_reduce(small_tensor)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / n_iters

        # Large message: measure bandwidth
        large_tensor = torch.zeros(256 * 1024, device=device)  # 1 MB
        start.record()
        for _ in range(n_iters):
            torch.distributed.all_reduce(large_tensor)
        end.record()
        torch.cuda.synchronize()
        bandwidth_ms = start.elapsed_time(end) / n_iters

        # Convert to Gbps: 1 MB / time_in_seconds * 8 bits/byte / 1e9
        bandwidth_gbps = (1.0 / max(bandwidth_ms, 1e-6)) * 1000.0 * 8.0

        # NVLink detection: NVLink is >600 Gbps, PCIe gen4 is ~64 Gbps
        is_nvlink = bandwidth_gbps > 100.0

        # NVSwitch detection: NVSwitch provides full-bisection bandwidth
        # across all GPUs. With NVSwitch, bandwidth stays constant regardless
        # of world_size. Without it, bandwidth degrades with more GPUs.
        is_nvswitch = bandwidth_gbps > 400.0 and self.world_size >= 4

        # Peer-to-peer check
        peer_to_peer = False
        if self.world_size >= 2:
            try:
                peer_to_peer = torch.cuda.can_device_access_peer(0, 1)
            except Exception:
                pass

        return TopologyConfig(
            bandwidth_gbps=bandwidth_gbps,
            latency_us=latency_ms * 1000.0,
            is_nvlink=is_nvlink,
            is_nvswitch=is_nvswitch,
            peer_to_peer=peer_to_peer,
        )

    def should_distribute_scan(self, N_total: int) -> bool:
        """Decide if multi-GPU scan is faster than single-GPU.

        The distributed scan has overhead:
          - All-gather latency: topology.latency_us
          - All-gather bandwidth: SUMMARY_BYTES * world_size / bandwidth

        The scan speedup is:
          - Each GPU processes N_total / world_size elements
          - Approximate: 50ns per element (measured on H100)

        Distribute if scan_time_saved > 3 * allgather_overhead.
        The 3x factor accounts for prefix scan + apply overhead.
        """
        if self.world_size <= 1:
            return False

        scan_ns_per_element = 50  # Approximate, measured on H100
        chunk = N_total // self.world_size
        full_scan_us = N_total * scan_ns_per_element / 1000.0
        chunk_scan_us = chunk * scan_ns_per_element / 1000.0

        # All-gather overhead
        summary_bytes = self.SUMMARY_BYTES * self.world_size
        bandwidth_bytes_per_us = self.topology.bandwidth_gbps * 1e9 / 8.0 / 1e6
        allgather_us = self.topology.latency_us + summary_bytes / max(bandwidth_bytes_per_us, 1e-6)

        # Distribute if the time saved exceeds 3x the all-gather cost
        time_saved = full_scan_us - chunk_scan_us
        return time_saved > allgather_us * 3.0

    def compute_optimal_chunk_size(self, N_total: int) -> int:
        """Compute optimal per-GPU chunk size.

        If distribution is not beneficial, returns N_total (replicate).
        Otherwise returns N_total // world_size, rounded to cache line boundary.
        """
        if not self.should_distribute_scan(N_total):
            return N_total

        chunk = N_total // self.world_size
        # Round down to 64-element boundary (cache line alignment)
        chunk = (chunk // 64) * 64
        return max(64, chunk)

    def get_strategy_label(self, N_total: int) -> str:
        """Human-readable label for the selected strategy."""
        if self.world_size <= 1:
            return "single-GPU"
        if not self.should_distribute_scan(N_total):
            return f"replicated ({self.world_size} GPUs, scan too small to distribute)"
        link = "NVLink" if self.topology.is_nvlink else "PCIe"
        chunk = self.compute_optimal_chunk_size(N_total)
        return f"distributed ({self.world_size} GPUs, {link}, chunk={chunk})"
