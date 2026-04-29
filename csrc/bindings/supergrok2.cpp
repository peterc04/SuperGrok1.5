// bindings/supergrok2.cpp — runtime dispatch to per-arch SG v2 launchers.
//
// SG v2 has the largest dispatcher surface in the codebase. The Python
// optimizer (grokking_optimizers/supergrok2.py) invokes six entry points:
//
//   supergrok2_mamba_peer_batched_step       (vector signature)
//   supergrok2_prepare_and_batched_step      (vector + scalars)
//   supergrok2_bilevel_fwd_save              (single-tensor, ~50 args)
//   supergrok2_bilevel_fwd_save_batched      (vector + ~30 weights)
//   supergrok2_bilevel_backward              (single-tensor, ~60 args)
//   supergrok2_bilevel_backward_batched      (vector + ~30 weights)
//
// The per-arch launchers already exist as
//   sg::<arch>::launch_mamba3_peer_step
//   sg::<arch>::launch_mamba3_peer_batched_step
//   sg::<arch>::launch_mamba3_peer_bilevel_fwd_save
//   sg::<arch>::launch_mamba3_peer_bilevel_fwd_save_batched
//   sg::<arch>::launch_mamba3_peer_backward
//   sg::<arch>::launch_mamba3_peer_backward_batched
//
// in csrc/kernels/cuda/<arch>/supergrok2_{fwd,bwd}_<arch>.cu and
// csrc/kernels/hip/<arch>/supergrok2_{fwd,bwd}_<arch>.hip.cpp. Each per-
// arch launcher's signature is verbatim from the canonical sm_80 file
// (the wrap-baseline pass preserved bit-identical signatures across all
// eight arches).
//
// To bind these for Python: each m.def("supergrok2_<entry>", ...) needs a
// thin wrapper here that takes the same args as the per-arch launcher,
// switches on detect_arch(), and forwards the call. Because the launcher
// signatures are 30-60 args and span 5-7 different files (SG2 is split
// into supergrok2_fwd_<arch>.cu / supergrok2_bwd_<arch>.cu), this should
// be done by mechanically copying the launcher signature from
// csrc/kernels/cuda/sm_80/supergrok2_fwd_sm80.cu into a DECLARE_SG2(NS)
// macro here, then writing the SG_DISPATCH call.
//
// TODO(post-refactor): when a session has hardware to validate against
// (so the build can verify the signatures match the linker), expand this
// stub. Reference: the deleted csrc/common/ops.cpp@682eab4^ supplies the
// exact pre-refactor wrappers (supergrok2_mamba_peer_step etc.) — the
// new bindings just need to do `case 80: ::sg::sm80::launch_X(...)`
// instead of `tier == AMPERE ? launch_X_ampere : launch_X`.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <vector>

namespace sg {

// The DECLARE_SG2 macro and the public Python entry points remain to
// be filled in. Until they are, importing _ops will succeed (pybind11
// only registers what's defined) but Python calls into
// _ops.supergrok2_* will raise AttributeError.

} // namespace sg
