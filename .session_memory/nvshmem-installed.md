---
name: nvshmem-installed
description: NVSHMEM 3.7.0 installed on the H100 box with sm_90 device bitcode; in-kernel device all-reduce is buildable
metadata: 
  node_type: memory
  type: project
  originSessionId: 6354dc07-b50f-40a0-8748-5189102539d3
---

NVSHMEM **3.7.0** is installed on the 8×H100 box (2026-06-25) via `pip install nvidia-nvshmem-cu12`.
It ships everything the **in-kernel device-NVSHMEM TP/SP all-reduce** needs (the user's explicit design
over a CUDA graph):
- header: `/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem/include/nvshmem.h`
- host lib: `.../nvshmem/lib/libnvshmem_host.so.3`
- **device bitcode for sm_90**: `.../nvshmem/lib/libnvshmem_device_sm_90.bc` + `libnvshmem_device.a`
- device headers under `.../nvshmem/include/device/` + `non_abi/device/`

Verified: `nvcc -std=c++17 -arch=sm_90a -rdc=true -I.../nvshmem/include -c <tu>.cu` compiles clean (rc=0).
So `-DSG_HAS_NVSHMEM=1` (the gate in tp_transport.cuh `NvshmemTransport`) is now buildable. NVSHMEM_HOME =
`/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem`. This RESOLVES the prior env gate (it was absent).

Caveats for the real cross-GPU run: device-side calls need `-rdc=true` + link `libnvshmem_device.a`;
operands must live in the `nvshmem_malloc` SYMMETRIC heap (not plain cudaMalloc — forces a TP-comm-slot
allocator split); multi-GPU needs the NVSHMEM bootstrap. See [[flagship-distributed-config]].
Contrast with [[ncu-blocked-runpod]] (that gate is NOT resolvable in-container).
