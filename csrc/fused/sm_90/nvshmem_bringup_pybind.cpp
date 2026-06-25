// csrc/fused/sm_90/nvshmem_bringup_pybind.cpp — the HOST NVSHMEM TP-team bring-up
// pybind (the capability-track missing piece, GOAL: nvshmem_pybind area).
//
// WHY THIS TU EXISTS
//   The fused L3-TC megakernel's in-kernel TP all-reduce (tp_transport.cuh ::
//   NvshmemTransport) needs, BEFORE the megakernel launch, a populated
//   par::CommCtx whose TP fields point at a SYMMETRIC nvshmem_malloc'd heap and
//   carry the TP-team id. The host bring-up for that is a three-step dance
//   (host_bringup.py header §1-3): nvshmem_init (UID bootstrap) +
//   nvshmem_team_split_strided (the TP team) + the COLLECTIVE nvshmem_malloc of
//   the symmetric heap. NVSHMEM 3.7.0's host lib (libnvshmem_host.so.3) is on
//   this box WITH the UID bootstrap plugin (nvshmem_bootstrap_uid.so.3) but there
//   is NO `nvshmem` Python binding, and MPI/PMI launchers are absent. So Python
//   cannot drive the bring-up. THIS pybind is the seam: it exposes the exact host
//   C symbols the launcher already links (-DSG_HAS_NVSHMEM=1) so the harness drives
//   nvshmem_init over the UID broadcast carried by torch.distributed (NCCL).
//
// HONESTY / SCOPE
//   * This is a SEPARATE pybind module (sg_nvshmem_bringup), JIT-built by the
//     harness with torch.utils.cpp_extension.load against the REAL NVSHMEM headers
//     + libnvshmem_host.so.3. It does NOT touch _ops, the committed launcher TUs,
//     or the SingleGPU/default path. It is gated entirely behind being built at all
//     (the harness only builds it on the --nvshmem path).
//   * Because we compile against the REAL headers, there is NO ABI guessing
//     (nvshmemx_init_attr_t / nvshmemx_uniqueid_t come from the toolkit headers,
//     static_assert'd to 144 / 128 bytes there) — the host_bringup.py blocker note
//     about "must not silently dlopen-and-guess the ABI" is RESOLVED by linking the
//     headers instead of ctypes-poking the struct.
//   * The UID exchange is the operator's job via torch.distributed: rank 0 calls
//     get_uniqueid() -> bytes, broadcasts the 128-byte blob over the PG, every rank
//     calls init_with_uniqueid(rank, nranks, uid_bytes). This is exactly the
//     NVSHMEMX_INIT_WITH_UNIQUEID path the UID plugin implements.
//
// BUILD (the harness does this; documented here for reproducibility):
//   nvcc/c++ -O3 -std=c++17 -I$NVSHMEM_HOME/include
//            -L$NVSHMEM_HOME/lib -lnvshmem_host  (host-only; no -rdc, no device link)
//   NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem
//
// This TU is HOST-ONLY: it calls the host-callable NVSHMEM symbols (init / team
// split / malloc / free / the host-callable nvshmem_float_sum_reduce collective +
// nvshmemx_barrier_all_on_stream). It needs NO device link (libnvshmem_device.a /
// .bc) — those are only required for IN-KERNEL nvshmem_* calls, which live in the
// megakernel launcher TU, not here.

#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

namespace {

// ── module-local bring-up state (process-lived; one NVSHMEM world per process) ──
struct BringupState {
    bool   initialized   = false;   // nvshmem_init has run
    int    world_pe      = -1;      // nvshmem_my_pe() after init
    int    world_npes    = -1;      // nvshmem_n_pes()
    nvshmem_team_t tp_team = NVSHMEM_TEAM_INVALID;  // the carved TP team
    int    tp_local_pe   = -1;      // nvshmem_team_my_pe(tp_team)
    int    tp_n_pes      = -1;      // nvshmem_team_n_pes(tp_team)
    void*  sym_heap      = nullptr; // the symmetric heap (nvshmem_malloc)
    int64_t sym_floats   = 0;       // heap size in floats
};

BringupState g_state;

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────
//  get_uniqueid — rank 0 generates the NVSHMEM unique id (the bootstrap blob the
//  UID plugin uses to rendezvous the PEs). Returned as the raw bytes of
//  nvshmemx_uniqueid_t (128 bytes: int version + 124 char internal) so the caller
//  can broadcast it over torch.distributed. Every rank passes the SAME blob back
//  into init_with_uniqueid().
// ─────────────────────────────────────────────────────────────────────────
static py::bytes nvshmem_get_uniqueid() {
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    int rc = nvshmemx_get_uniqueid(&uid);
    if (rc != 0) {
        throw std::runtime_error(
            "nvshmemx_get_uniqueid failed (rc=" + std::to_string(rc) + ")");
    }
    return py::bytes(reinterpret_cast<const char*>(&uid), sizeof(uid));
}

// uniqueid_size — the exact byte length of nvshmemx_uniqueid_t (128), so the Python
// side can size its broadcast buffer EXACTLY (no magic constant on the Python side).
static int64_t nvshmem_uniqueid_size() {
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

// ─────────────────────────────────────────────────────────────────────────
//  init_with_uniqueid — attach THIS process's PE to the NVSHMEM world via the UID
//  bootstrap. `uid_bytes` is the 128-byte blob rank 0 produced and broadcast;
//  `my_rank`/`n_ranks` are the global job coordinates (== torch.distributed
//  rank/world_size). This is the NVSHMEMX_INIT_WITH_UNIQUEID path the bootstrap
//  plugin implements — no MPI/PMI needed.
//
//  Sets the device FIRST (nvshmem binds its symmetric heap to the current CUDA
//  device; the caller must have torch.cuda.set_device(local_rank) already, but we
//  also accept an explicit device_ordinal for robustness).
// ─────────────────────────────────────────────────────────────────────────
static void nvshmem_init_with_uniqueid(int my_rank, int n_ranks,
                                       py::bytes uid_bytes, int device_ordinal) {
    if (g_state.initialized) {
        throw std::runtime_error("nvshmem already initialized in this process");
    }
    std::string blob = uid_bytes;  // copy out of py::bytes
    if (blob.size() != sizeof(nvshmemx_uniqueid_t)) {
        throw std::runtime_error(
            "uid_bytes wrong size: got " + std::to_string(blob.size()) +
            ", expected " + std::to_string(sizeof(nvshmemx_uniqueid_t)));
    }
    if (device_ordinal >= 0) {
        check_cuda(cudaSetDevice(device_ordinal), "cudaSetDevice(pre-init)");
    }

    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, blob.data(), sizeof(uid));

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int rc = nvshmemx_set_attr_uniqueid_args(my_rank, n_ranks, &uid, &attr);
    if (rc != 0) {
        throw std::runtime_error(
            "nvshmemx_set_attr_uniqueid_args failed (rc=" + std::to_string(rc) + ")");
    }
    // NB: use the EXPORTED nvshmemx_hostlib_init_attr, NOT the header-inline
    // nvshmemx_init_attr (which calls the UNEXPORTED nvshmemi_init_thread and would
    // be an undefined symbol when linking only libnvshmem_host.so.3). The inline
    // wrapper's only extra work is the version stamp, which NVSHMEMX_INIT_ATTR_
    // INITIALIZER already set.
    rc = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (rc != 0) {
        throw std::runtime_error(
            "nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID) failed (rc=" +
            std::to_string(rc) + ")");
    }
    g_state.initialized = true;
    g_state.world_pe    = nvshmem_my_pe();
    g_state.world_npes  = nvshmem_n_pes();
    // The world PE assigned by NVSHMEM must equal the rank we passed (UID bootstrap
    // preserves rank order); assert LOUDLY rather than silently mis-map.
    if (g_state.world_pe != my_rank || g_state.world_npes != n_ranks) {
        throw std::runtime_error(
            "nvshmem world (pe=" + std::to_string(g_state.world_pe) + "/npes=" +
            std::to_string(g_state.world_npes) + ") != requested (rank=" +
            std::to_string(my_rank) + "/nranks=" + std::to_string(n_ranks) + ")");
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  team_split_strided — carve THIS rank's TP team out of NVSHMEM_TEAM_WORLD with
//  (PE_start, PE_stride, PE_size) from the DP×TP mesh (host_bringup.TPBootstrap.
//  pe_range()). Stores the team id + the team-local pe/size. Returns the int-encoded
//  team handle (the value the launcher stores in CommCtx.tp_comm_handle via
//  reinterpret_cast<void*>((intptr_t)team)). For single-node pure TP the split is
//  (0,1,world) ⇒ the carved team is equivalent to NVSHMEM_TEAM_WORLD.
//
//  NB: nvshmem_team_split_strided is a COLLECTIVE over the PARENT team (WORLD) —
//  every PE in the world must call it with the SAME args, even PEs not in the new
//  team. The harness drives it on every rank with that rank's own pe_range (which
//  for single-node pure TP is identical on all ranks).
// ─────────────────────────────────────────────────────────────────────────
static int64_t nvshmem_team_split_tp(int pe_start, int pe_stride, int pe_size) {
    if (!g_state.initialized) {
        throw std::runtime_error("team_split_strided before nvshmem_init");
    }
    nvshmem_team_t new_team = NVSHMEM_TEAM_INVALID;
    int rc = nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, pe_start, pe_stride,
                                        pe_size, /*config=*/nullptr,
                                        /*config_mask=*/0, &new_team);
    if (rc != 0) {
        throw std::runtime_error(
            "nvshmem_team_split_strided failed (rc=" + std::to_string(rc) +
            ", start=" + std::to_string(pe_start) + ", stride=" +
            std::to_string(pe_stride) + ", size=" + std::to_string(pe_size) + ")");
    }
    g_state.tp_team = new_team;
    // PEs inside the team get a valid local pe; PEs outside get NVSHMEM_TEAM_INVALID
    // for the team and my_pe == -1. For our pure-TP mesh every PE is in its team.
    if (new_team != NVSHMEM_TEAM_INVALID) {
        g_state.tp_local_pe = nvshmem_team_my_pe(new_team);
        g_state.tp_n_pes    = nvshmem_team_n_pes(new_team);
    }
    return static_cast<int64_t>(static_cast<intptr_t>(new_team));
}

// ─────────────────────────────────────────────────────────────────────────
//  malloc_symmetric_heap — the COLLECTIVE nvshmem_malloc of the symmetric TP-slot
//  heap, sized to need_floats (== host_bringup.tp_heap_stride_floats(ctas_per_pe);
//  matches dec_tc_ensure_tp_sym_heap exactly). Returns the device pointer as an
//  integer (the value CommCtx.tp_sym_heap holds). COLLECTIVE: every PE in the
//  WORLD must call with the SAME size in the SAME order (nvshmem_malloc is a
//  collective barrier internally).
// ─────────────────────────────────────────────────────────────────────────
static int64_t nvshmem_malloc_symmetric(int64_t need_floats) {
    if (!g_state.initialized) {
        throw std::runtime_error("malloc_symmetric_heap before nvshmem_init");
    }
    if (need_floats < 1) {
        throw std::runtime_error("need_floats must be >= 1");
    }
    if (g_state.sym_heap != nullptr && g_state.sym_floats >= need_floats) {
        return reinterpret_cast<int64_t>(g_state.sym_heap);
    }
    if (g_state.sym_heap != nullptr) {
        nvshmem_free(g_state.sym_heap);
        g_state.sym_heap = nullptr;
        g_state.sym_floats = 0;
    }
    void* p = nvshmem_malloc(static_cast<size_t>(need_floats) * sizeof(float));
    if (p == nullptr) {
        throw std::runtime_error(
            "nvshmem_malloc returned nullptr for " + std::to_string(need_floats) +
            " floats (collective alloc failed — out of symmetric heap?)");
    }
    g_state.sym_heap   = p;
    g_state.sym_floats = need_floats;
    return reinterpret_cast<int64_t>(p);
}

// ─────────────────────────────────────────────────────────────────────────
//  tp_allreduce_smoke — a tiny CROSS-RANK correctness probe (the SMOKE gate).
//  Each rank fills the head of the symmetric heap with `n` floats == (my_world_pe
//  + 1), runs the HOST-callable nvshmem_float_sum_reduce over the TP team
//  in-place-safe (separate src/dst regions on the heap), and returns the reduced
//  vector to Python. Expected: every element == Σ_{pe in team}(global_pe+1).
//
//  This exercises the FULL bring-up: symmetric heap addressing + a team-scoped
//  collective over the carved TP team. It is the genuine cross-GPU assert the
//  harness checks (not a single-GPU loopback).
//
//  Layout on the symmetric heap: [ src(n) | dst(n) ] at the head. Requires
//  sym_floats >= 2*n (the real flagship heap is far larger; the smoke uses small n).
// ─────────────────────────────────────────────────────────────────────────
static torch::Tensor nvshmem_tp_allreduce_smoke(int64_t n) {
    if (!g_state.initialized) {
        throw std::runtime_error("tp_allreduce_smoke before nvshmem_init");
    }
    if (g_state.tp_team == NVSHMEM_TEAM_INVALID) {
        throw std::runtime_error("tp_allreduce_smoke before team_split (no TP team)");
    }
    if (g_state.sym_heap == nullptr || g_state.sym_floats < 2 * n) {
        throw std::runtime_error(
            "tp_allreduce_smoke needs a symmetric heap with >= 2*n floats; have " +
            std::to_string(g_state.sym_floats) + " for n=" + std::to_string(n));
    }
    float* heap = reinterpret_cast<float*>(g_state.sym_heap);
    float* src  = heap;
    float* dst  = heap + n;

    // Fill src on-device with (world_pe + 1) via a host-staged copy.
    std::vector<float> hsrc(static_cast<size_t>(n),
                            static_cast<float>(g_state.world_pe + 1));
    check_cuda(cudaMemcpy(src, hsrc.data(), static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyHostToDevice), "H2D src");
    check_cuda(cudaDeviceSynchronize(), "sync before reduce");

    // TEAM-scoped sum-reduce (the cross-rank collective). Host-callable variant
    // (blocking). Both src and dst are on the symmetric heap.
    int rc = nvshmem_float_sum_reduce(g_state.tp_team, dst, src,
                                      static_cast<size_t>(n));
    if (rc != 0) {
        throw std::runtime_error(
            "nvshmem_float_sum_reduce failed (rc=" + std::to_string(rc) + ")");
    }
    // Barrier so every PE's reduce has completed before we read dst back.
    nvshmem_barrier_all();
    check_cuda(cudaDeviceSynchronize(), "sync after reduce");

    auto out = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    check_cuda(cudaMemcpy(out.data_ptr<float>(), dst,
                          static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H dst");
    return out;
}

// ── introspection accessors (so the harness can assert/print the live state) ──
static bool   nvshmem_is_initialized() { return g_state.initialized; }
static int    nvshmem_world_pe()       { return g_state.world_pe; }
static int    nvshmem_world_npes()     { return g_state.world_npes; }
static int    nvshmem_tp_local_pe()    { return g_state.tp_local_pe; }
static int    nvshmem_tp_n_pes()       { return g_state.tp_n_pes; }
static int64_t nvshmem_sym_heap_ptr()  {
    return reinterpret_cast<int64_t>(g_state.sym_heap);
}
static int64_t nvshmem_sym_heap_floats() { return g_state.sym_floats; }

// barrier_all — expose the world barrier so the harness can fence between phases.
static void nvshmem_barrier_world() {
    if (!g_state.initialized) return;
    nvshmem_barrier_all();
}

// finalize — tear down (frees the symmetric heap + the NVSHMEM world). Idempotent.
static void nvshmem_finalize_world() {
    if (!g_state.initialized) return;
    if (g_state.sym_heap != nullptr) {
        nvshmem_free(g_state.sym_heap);
        g_state.sym_heap = nullptr;
        g_state.sym_floats = 0;
    }
    if (g_state.tp_team != NVSHMEM_TEAM_INVALID &&
        g_state.tp_team != NVSHMEM_TEAM_WORLD) {
        nvshmem_team_destroy(g_state.tp_team);
    }
    g_state.tp_team = NVSHMEM_TEAM_INVALID;
    // EXPORTED teardown (the header-inline nvshmem_finalize calls the UNEXPORTED
    // nvshmemi_finalize; nvshmemx_hostlib_finalize is the linkable entry point).
    nvshmemx_hostlib_finalize();
    g_state.initialized = false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NVSHMEM host TP-team bring-up pybind (UID bootstrap + team split + "
              "symmetric malloc + a tiny team all-reduce smoke). Real NVSHMEM 3.7 "
              "headers; host-only link against libnvshmem_host.so.3.";
    m.def("get_uniqueid", &nvshmem_get_uniqueid,
          "rank-0 NVSHMEM unique id bytes (broadcast over torch.distributed)");
    m.def("uniqueid_size", &nvshmem_uniqueid_size,
          "sizeof(nvshmemx_uniqueid_t) in bytes");
    m.def("init_with_uniqueid", &nvshmem_init_with_uniqueid,
          py::arg("my_rank"), py::arg("n_ranks"), py::arg("uid_bytes"),
          py::arg("device_ordinal") = -1,
          "nvshmem_init via NVSHMEMX_INIT_WITH_UNIQUEID (the UID bootstrap)");
    m.def("team_split_strided", &nvshmem_team_split_tp,
          py::arg("pe_start"), py::arg("pe_stride"), py::arg("pe_size"),
          "carve the TP team out of NVSHMEM_TEAM_WORLD; returns int team handle");
    m.def("malloc_symmetric_heap", &nvshmem_malloc_symmetric,
          py::arg("need_floats"),
          "collective nvshmem_malloc of the symmetric TP heap; returns device ptr int");
    m.def("tp_allreduce_smoke", &nvshmem_tp_allreduce_smoke, py::arg("n"),
          "tiny team-scoped sum-reduce of (world_pe+1); returns the reduced vector");
    m.def("barrier_world", &nvshmem_barrier_world, "nvshmem_barrier_all()");
    m.def("finalize", &nvshmem_finalize_world, "tear down the NVSHMEM world");
    m.def("is_initialized", &nvshmem_is_initialized);
    m.def("world_pe", &nvshmem_world_pe);
    m.def("world_npes", &nvshmem_world_npes);
    m.def("tp_local_pe", &nvshmem_tp_local_pe);
    m.def("tp_n_pes", &nvshmem_tp_n_pes);
    m.def("sym_heap_ptr", &nvshmem_sym_heap_ptr);
    m.def("sym_heap_floats", &nvshmem_sym_heap_floats);
}
