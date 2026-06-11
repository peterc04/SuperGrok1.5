"""Quiet baseline: measure the vit L3-TC megakernel (tc_train_step) roofline
fraction at the SM-saturation batch, using the SAME FLOP/AI/ceiling logic as
tuning/roofline.py. Run MPS-isolated (CUDA_MPS_PIPE_DIRECTORY=/nonexistent).

Reports achieved TF/s, ceiling, fraction, wall/step, and the scalar/TC ratio.
This is the optimization metric for the vit cells (the converted-3 share this
fwd+bwd+reduce megakernel; the opt tail is a negligible fraction of the wall).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import tuning.roofline as R  # noqa: E402


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    model = "vit"
    # FLOPs/step + n_params from the eager-profiled adamw pipeline (path-independent
    # GEMM count). We run the cheap profile pass of measure_pipeline to grab it.
    data, init, n_params = R._data_init(model)
    eager_B_full = data[0].shape[0]
    # Tile the data to B for the FLOP basis (FLOPs linear in B; measure_tc_cell
    # rescales by B/eager_B_full, so pass the eager profile's native B + flops).
    # Use measure_pipeline's profiler to get eager flops at native B.
    import torch.profiler as tp
    g = R._g()
    dev = torch.device("cuda")
    c = R._cfg("adamw", model, 10, 10 ** 9)
    c["matmul_precision"] = "bf16"; c["use_amp"] = False
    c["use_fused"] = False
    c["max_steps"] = 20; c["early_stop_max_steps"] = 20
    torch.manual_seed(42)
    with tp.profile(activities=[tp.ProfilerActivity.CUDA, tp.ProfilerActivity.CPU],
                    with_flops=True) as prof:
        g.train_adamw(c, init, *data, dev, bp=0)
    flops_total = sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages())
    eager_flops_full = flops_total / 20.0
    print(f"[baseline] eager FLOPs/step @ B={eager_B_full}: {eager_flops_full:.4e}", flush=True)

    dc = R._cfg("adamw", model, 10, 10 ** 9)
    seq = dc.get("num_patches", 16)
    dim = dc.get("dim_model", 128)
    layers = dc.get("num_layers", 2)

    # Tile FLOP basis to the requested measurement batch B (measure_tc_cell uses
    # eager_B_full to rescale, so we feed it the native eager flops + B and let it
    # scale to its own B%16). To measure at B, we pass eager_B_full=B by rescaling:
    flops_at_B = eager_flops_full * (B / float(eager_B_full))

    rows = R.measure_tc_cell(model, flops_at_B, B, n_params, seq, dim, layers)
    print("\n[baseline] vit TC + scalar rows:", flush=True)
    for r in rows:
        print(f"  path={r['path']:38s} prec={r['precision']:7s} "
              f"B={r['tc_batch']} wall/step={r['wall_per_step_ms']:.4f}ms "
              f"achieved={r['achieved_flops_per_s']/1e12:.2f}TF/s "
              f"ceiling={r['roofline_ceiling_flops_per_s']/1e12:.2f}TF/s "
              f"frac={r['fraction_of_roofline']:.4f} "
              f"AI={r['arithmetic_intensity']:.1f} bound={r['bound_regime_at_AI']}",
              flush=True)
    tc = next((r for r in rows if r["path_family"] == "tc"), None)
    if tc:
        print(f"\n[VIT-TC-BASELINE] B={tc['tc_batch']} achieved={tc['achieved_flops_per_s']/1e12:.3f} TF/s "
              f"frac_of_roofline={tc['fraction_of_roofline']:.4f} "
              f"wall/step={tc['wall_per_step_ms']:.4f}ms", flush=True)


if __name__ == "__main__":
    main()
