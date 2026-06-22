# SuperGrok2

A grokking-optimizer + model training stack built as **44 reusable components
that compose into 99 fused training pipelines** across three accelerator
families — **NVIDIA sm_90 (Hopper)**, **AMD gfx942 (CDNA3 / MI300X)**, and
**Google TPU v6e** — with **one canonical source of truth per component**
(no parallel math trees, no dead duplicates; enforced by a self-test drift
guard).

## Lines of code

Project source counted per language and per file (excludes the vendored **CUTLASS** dependency under `third_party/`, which is tracked as a git submodule). Generated 2026-06-21.

| Language | Files | Lines |
|----------|------:|------:|
| Python | 162 | 87,693 |
| CUDA | 44 | 23,803 |
| C/C++ | 63 | 19,640 |
| HIP (AMD ROCm) | 48 | 2,309 |
| Shell | 15 | 934 |
| **Source subtotal** | **332** | **134,379** |
| Logs/diffs (artifacts) | 124 | 39,907 |
| JSON (data/results) | 47 | 28,958 |
| Markdown (docs) | 81 | 14,534 |
| Config | 15 | 2,354 |
| **Total tracked text** | **599** | **220,132** |

<details>
<summary>Per-file breakdown — 332 source files</summary>


**Python** (162 files, 87,693 lines)

| File | Lines |
|------|------:|
| `grokking_optimizers/compile.py` | 32,900 |
| `grokking_race_v2.py` | 2,505 |
| `grokking_optimizers/optimizers/supergrok2.py` | 2,397 |
| `grokking_optimizers/dispatch.py` | 2,013 |
| `grokking_optimizers/megakernel_codegen.py` | 1,561 |
| `tests/hw/test_l3tc_tail_gate.py` | 1,512 |
| `csrc/backends/pallas/_pallas_kernels.py` | 1,319 |
| `scripts/fast_triage.py` | 1,197 |
| `csrc/backends/pallas/_pallas_fused.py` | 1,131 |
| `csrc/backends/pallas/launch_supergrok2.py` | 1,067 |
| `tuning/roofline.py` | 1,037 |
| `setup.py` | 1,029 |
| `tests/hw/sg2_kernel_mirror.py` | 1,002 |
| `grokking_optimizers/distributed.py` | 976 |
| `tests/hw/mamba3_oracle.py` | 958 |
| `tests/hw/test_reference_parity.py` | 938 |
| `grokking_optimizers/_tuned_inject.py` | 849 |
| `tests/hw/test_decoder_tc.py` | 808 |
| `grokking_optimizers/profile_maximal.py` | 762 |
| `grokking_optimizers/profile.py` | 740 |
| `tests/hw/test_multistep_parity.py` | 737 |
| `bench_backends.py` | 728 |
| `tests/hw/test_vit_tc.py` | 723 |
| `grokking_optimizers/utilization.py` | 722 |
| `tests/tpu/test_pallas_parity_interpret.py` | 707 |
| `grokking_optimizers/verify_all.py` | 699 |
| `tuning/tune_optimizers.py` | 693 |
| `csrc/backends/pallas/_pallas_models.py` | 681 |
| `tests/hw/test_mamba_tc.py` | 681 |
| `grokking_optimizers/mamba3_block.py` | 638 |
| `grokking_optimizers/optimizers/neuralgrok.py` | 622 |
| `tests/hw/mamba_oracle.py` | 619 |
| `grokking_optimizers/optimizers/supergrok15.py` | 604 |
| `tuning/test_build_injection.py` | 576 |
| `grokking_optimizers/optimizers/supergrok11.py` | 575 |
| `tests/hw/test_sg2_megakernel.py` | 565 |
| `scripts/check_math_single_source.py` | 537 |
| `grokking_optimizers/megakernel_engine.py` | 521 |
| `tests/hw/test_opt_stages.py` | 502 |
| `tests/hw/test_dp2_loopback_determinism.py` | 485 |
| `tests/hw/test_wgmma_substrate.py` | 483 |
| `tests/hw/vit_kernel_mirror.py` | 482 |
| `tests/hw/mamba_kernel_mirror.py` | 478 |
| `scripts/nvcc_baseline.py` | 470 |
| `tests/hw/vit_oracle.py` | 463 |
| `tests/hw/test_vit_megakernel.py` | 449 |
| `tests/hw/decoder_oracle.py` | 444 |
| `tests/hw/decoder_kernel_mirror.py` | 431 |
| `grokking_optimizers/optimizers/grokadamw.py` | 379 |
| `grokking_optimizers/parallel/pipeline.py` | 367 |
| `wiring_check.py` | 367 |
| `tuning/_grokadamw_multistep_parity.py` | 362 |
| `tests/hw/_sg2_l3tc_gate.py` | 356 |
| `tests/hw/test_3d_parallel.py` | 355 |
| `grokking_optimizers/parallel/zero3.py` | 331 |
| `grokking_optimizers/tune_hook.py` | 327 |
| `tests/hw/test_mamba_megakernel.py` | 325 |
| `tests/hw/test_pp2_loopback_determinism.py` | 323 |
| `tuning/decoder_bench.py` | 323 |
| `grokking_optimizers/megakernel.py` | 310 |
| `tuning/precision_analysis.py` | 309 |
| `tests/hw/test_distributed_step.py` | 300 |
| `tests/hw/test_sharded_optimizer.py` | 290 |
| `tests/hw/test_step_graph_capture.py` | 285 |
| `tests/test_shard_map.py` | 275 |
| `grokking_optimizers/parallel/shard_map.py` | 272 |
| `tests/hw/test_zero3_roundtrip.py` | 270 |
| `grokking_optimizers/optimizers/muon.py` | 263 |
| `tuning/_prodigy_multistep_parity.py` | 256 |
| `tests/hw/test_tp_loopback.py` | 255 |
| `grokking_optimizers/parallel/distributed_step.py` | 250 |
| `tests/hw/test_parallel_instantiation.py` | 235 |
| `grokking_optimizers/lowprec.py` | 233 |
| `tuning/vit_bench.py` | 233 |
| `grokking_optimizers/optimizers/looksam.py` | 228 |
| `grokking_optimizers/optimizers/prodigy.py` | 228 |
| `examples/autotune_demo/run_autotune.py` | 204 |
| `tests/test_pipeline_schedule.py` | 204 |
| `grokking_optimizers/optimizers/grokfast.py` | 201 |
| `grokking_optimizers/optimizers/adamw.py` | 192 |
| `scripts/diag_neuralgrok_seed123.py` | 189 |
| `tests/test_zero3_plan.py` | 185 |
| `scripts/roofline_bench.py` | 177 |
| `tuning/mamba_bench.py` | 165 |
| `grokking_optimizers/optimizers/lion.py` | 161 |
| `csrc/backends/pallas/launch_supergrok11.py` | 159 |
| `csrc/backends/pallas/launch_looksam.py` | 149 |
| `csrc/backends/pallas/launch_supergrok15.py` | 149 |
| `grokking_optimizers/__init__.py` | 147 |
| `examples/toy_tune_project/tune_hook.py` | 146 |
| `examples/autotune_demo/tune_hook.py` | 145 |
| `.regpressure/gpu/prodtime.py` | 144 |
| `csrc/backends/pallas/launch_muon.py` | 143 |
| `tests/hw/_mamba_race_probe.py` | 134 |
| `tuning/_prodigy_owner_block_unit.py` | 134 |
| `csrc/backends/pallas/launch_prodigy.py` | 129 |
| `tuning/_h3_dw_aaa.py` | 125 |
| `csrc/backends/pallas/launch_neuralgrok.py` | 121 |
| `_sg_realsg_probe.py` | 111 |
| `tests/hw/_mamba_fill_test.py` | 111 |
| `csrc/backends/pallas/launch_grokadamw.py` | 107 |
| `tuning/_mbtc_bypass_profile.py` | 101 |
| `csrc/backends/pallas/v6e/__init__.py` | 98 |
| `tests/hw/test_mb3_scalar.py` | 98 |
| `tests/hw/_mamba_prodigy_production_probe.py` | 95 |
| `csrc/backends/pallas/v5p/__init__.py` | 90 |
| `scripts/time_cell.py` | 87 |
| `scripts/_vit_phase_profile.py` | 85 |
| `.smoke12_driver.py` | 84 |
| `scripts/diag_sg_sharpness.py` | 76 |
| `csrc/backends/pallas/launch_grokfast.py` | 75 |
| `csrc/backends/pallas/launch_lion.py` | 74 |
| `tuning/_decoder_validate.py` | 73 |
| `scripts/_vit_baseline.py` | 72 |
| `tuning/_grokadamw_final_revalidate.py` | 69 |
| `tuning/_h3_splitk_sweep.py` | 67 |
| `scripts/diag_looksam_samdir.py` | 64 |
| `.regpressure/parse2.py` | 63 |
| `.regpressure/parse_ptxas.py` | 62 |
| `examples/autotune_demo/build_variant.py` | 53 |
| `tuning/_embed_aaa.py` | 49 |
| `grokking_optimizers/parallel/__init__.py` | 43 |
| `grokking_optimizers/optimizers/__init__.py` | 41 |
| `csrc/backends/pallas/launch_adamw.py` | 40 |
| `scripts/_vit_ncu_driver.py` | 39 |
| `csrc/fused/tpu_v6e/mega_mamba3_adamw.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_grokadamw.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_grokfast.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_lion.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_looksam.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_muon.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_neuralgrok.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_prodigy.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_supergrok11.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_supergrok15.py` | 24 |
| `csrc/fused/tpu_v6e/mega_mamba3_supergrok2.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_adamw.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_grokadamw.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_grokfast.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_lion.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_looksam.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_muon.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_neuralgrok.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_prodigy.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_supergrok11.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_supergrok15.py` | 24 |
| `csrc/fused/tpu_v6e/mega_transformer_decoder_supergrok2.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_adamw.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_grokadamw.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_grokfast.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_lion.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_looksam.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_muon.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_neuralgrok.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_prodigy.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_supergrok11.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_supergrok15.py` | 24 |
| `csrc/fused/tpu_v6e/mega_vit_supergrok2.py` | 24 |
| `csrc/backends/pallas/__init__.py` | 17 |
| `tests/conftest.py` | 17 |
| `tests/tpu/__init__.py` | 8 |
| `csrc/backends/__init__.py` | 6 |

**CUDA** (44 files, 23,803 lines)

| File | Lines |
|------|------:|
| `csrc/fused/sm_90/model_stage_decoder_tc.cuh` | 2,595 |
| `csrc/fused/sm_90/model_stage_vit_tc.cuh` | 1,704 |
| `csrc/fused/sm_90/fused_decoder_megakernel.cuh` | 1,575 |
| `csrc/fused/sm_90/opt_stage_supergrok2.cuh` | 1,343 |
| `csrc/fused/sm_90/model_stage_mamba3.cuh` | 1,305 |
| `csrc/fused/sm_90/fused_mamba_megakernel.cuh` | 1,282 |
| `csrc/fused/sm_90/fused_vit_megakernel.cuh` | 1,187 |
| `csrc/fused/sm_90/dec_weights.cuh` | 1,023 |
| `csrc/fused/sm_90/model_stage_vit.cuh` | 947 |
| `csrc/backends/cuda/sm_90/mma.cuh` | 877 |
| `csrc/backends/cuda/sm_90/primitives.cuh` | 679 |
| `csrc/fused/sm_90/opt_stages_precompute.cuh` | 673 |
| `csrc/backends/cuda/sm_90/wgmma.cuh` | 646 |
| `csrc/fused/sm_90/opt_components.cuh` | 524 |
| `tests/hw/tp_loopback_binding.cu` | 487 |
| `csrc/fused/sm_90/mega_vit_real_adamw_tc.cu` | 452 |
| `csrc/fused/sm_90/pp_stage_decoder_tc.cuh` | 400 |
| `csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu` | 373 |
| `csrc/fused/sm_90/mega_mamba_real_adamw_tc_launcher.cu` | 370 |
| `csrc/backends/cuda/sm_90/tile_pipeline.cuh` | 366 |
| `csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu` | 351 |
| `tests/hw/_mamba_prodigy_probe.cu` | 297 |
| `csrc/fused/sm_90/vit_layout.cuh` | 296 |
| `csrc/backends/cuda/sm_90/wgmma_selftest.cu` | 288 |
| `csrc/fused/sm_90/tp_layer.cuh` | 288 |
| `csrc/fused/sm_90/mega_vit_real_adamw_tc_launcher.cu` | 285 |
| `csrc/fused/sm_90/mamba3_layout.cuh` | 283 |
| `csrc/fused/megakernel_common.cuh` | 278 |
| `csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu` | 267 |
| `csrc/fused/sm_90/tp_transport.cuh` | 233 |
| `csrc/fused/sm_90/decoder_tc_selftest.cu` | 199 |
| `csrc/fused/sm_90/decoder_layout.cuh` | 198 |
| `examples/autotune_demo/gemm_kernel.cu` | 176 |
| `csrc/backends/cuda/sm_90/warp_specialize.cuh` | 173 |
| `csrc/fused/sm_90/sharded_optimizer_kernel.cuh` | 167 |
| `tests/hw/sharded_optimizer_binding.cu` | 167 |
| `tests/hw/pp_stage_binding.cu` | 166 |
| `csrc/fused/sm_90/sg2_meta_tail.cu` | 164 |
| `.perf/M0_mamba_ws_agreement_proof.cu` | 152 |
| `csrc/fused/sm_90/parallel_config.cuh` | 124 |
| `csrc/common/utils.cuh` | 120 |
| `examples/toy_tune_project/toy_kernel.cu` | 119 |
| `csrc/fused/sm_90/model_stage_mamba_tc.cuh` | 118 |
| `tests/hw/_mb3_scalar_probe.cu` | 86 |

**C/C++** (63 files, 19,640 lines)

| File | Lines |
|------|------:|
| `grokking_optimizers/kernels/gfx942/supergrok2_gfx942.hip.hpp` | 2,692 |
| `grokking_optimizers/kernels/gfx942/mamba3_gfx942.hip.hpp` | 2,040 |
| `csrc/bindings/dispatch.cpp` | 1,378 |
| `csrc/algorithms/supergrok2_bilevel_adjoint.h` | 869 |
| `csrc/backends/hip/gfx942/supergrok2_bilevel_adjoint_gfx942.hip.hpp` | 861 |
| `grokking_optimizers/kernels/gfx942/attention_gfx942.hip.hpp` | 798 |
| `grokking_optimizers/kernels/gfx942/muon_gfx942.hip.hpp` | 648 |
| `csrc/algorithms/supergrok2.h` | 578 |
| `grokking_optimizers/kernels/gfx942/prodigy_gfx942.hip.hpp` | 566 |
| `grokking_optimizers/kernels/gfx942/vit_gfx942.hip.hpp` | 557 |
| `grokking_optimizers/kernels/gfx942/supergrok11_gfx942.hip.hpp` | 544 |
| `grokking_optimizers/kernels/gfx942/transformer_decoder_gfx942.hip.hpp` | 493 |
| `grokking_optimizers/kernels/gfx942/supergrok15_gfx942.hip.hpp` | 472 |
| `grokking_optimizers/kernels/gfx942/neuralgrok_gfx942.hip.hpp` | 437 |
| `grokking_optimizers/kernels/gfx942/looksam_gfx942.hip.hpp` | 433 |
| `grokking_optimizers/kernels/gfx942/grokadamw_gfx942.hip.hpp` | 427 |
| `csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp` | 348 |
| `csrc/backends/hip/gfx942/moe_compaction_gfx942.hip.hpp` | 346 |
| `csrc/bindings/helpers.h` | 345 |
| `grokking_optimizers/kernels/gfx942/grokfast_gfx942.hip.hpp` | 336 |
| `grokking_optimizers/kernels/gfx942/lion_gfx942.hip.hpp` | 312 |
| `csrc/common/platform.h` | 302 |
| `csrc/fused/megakernel_common_hip.hip.hpp` | 300 |
| `csrc/bindings/bindings.cpp` | 296 |
| `grokking_optimizers/kernels/gfx942/adamw_gfx942.hip.hpp` | 295 |
| `csrc/fused/gfx942/fused_megakernel.hip.hpp` | 288 |
| `csrc/fused/gfx942/opt_components.hip.hpp` | 233 |
| `csrc/fused/fused_wired_cells.inc` | 206 |
| `csrc/algorithms/supergrok11.h` | 178 |
| `csrc/backends/hip/gfx942/primitives.hpp` | 145 |
| `csrc/fused/gfx942/model_stages.hip.hpp` | 141 |
| `csrc/backends/hip/gfx942/models/mamba.hip.cpp` | 139 |
| `csrc/backends/hip/gfx942/models/decoder.hip.cpp` | 125 |
| `csrc/algorithms/neuralgrok.h` | 119 |
| `csrc/fused/sm_90/fused_dispatch_table.inc` | 119 |
| `csrc/fused/gfx942/fused_dispatch_table.inc` | 116 |
| `csrc/algorithms/supergrok15.h` | 114 |
| `csrc/algorithms/adamw.h` | 107 |
| `csrc/backends/hip/gfx942/models/vit.hip.cpp` | 107 |
| `csrc/algorithms/grokadamw.h` | 103 |
| `csrc/algorithms/prodigy.h` | 99 |
| `csrc/algorithms/looksam.h` | 93 |
| `csrc/common/types.h` | 80 |
| `csrc/algorithms/muon.h` | 78 |
| `grokking_optimizers/kernels/gfx942/common_gfx942.hip.hpp` | 78 |
| `csrc/algorithms/grokfast.h` | 77 |
| `csrc/algorithms/lion.h` | 77 |
| `csrc/scan/affine2x2.h` | 74 |
| `csrc/backends/hip/gfx942/launch_adamw.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_grokadamw.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_grokfast.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_lion.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_looksam.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_muon.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_neuralgrok.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_prodigy.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_supergrok11.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_supergrok15.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/launch_supergrok2.hip.cpp` | 5 |
| `csrc/backends/hip/gfx942/models/attention.hip.h` | 4 |
| `csrc/backends/hip/gfx942/models/decoder.hip.h` | 4 |
| `csrc/backends/hip/gfx942/models/mamba.hip.h` | 4 |
| `csrc/backends/hip/gfx942/models/vit.hip.h` | 4 |

**HIP (AMD ROCm)** (48 files, 2,309 lines)

| File | Lines |
|------|------:|
| `csrc/fused/gfx942/mega_mamba3_neuralgrok.hip` | 69 |
| `csrc/fused/gfx942/mega_transformer_decoder_neuralgrok.hip` | 69 |
| `csrc/fused/gfx942/mega_vit_neuralgrok.hip` | 69 |
| `csrc/fused/gfx942/mega_mamba3_adamw.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_grokadamw.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_grokfast.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_lion.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_looksam.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_muon.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_prodigy.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_supergrok11.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_supergrok15.hip` | 62 |
| `csrc/fused/gfx942/mega_mamba3_supergrok2.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_adamw.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_grokadamw.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_grokfast.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_lion.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_looksam.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_muon.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_prodigy.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_supergrok11.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_supergrok15.hip` | 62 |
| `csrc/fused/gfx942/mega_transformer_decoder_supergrok2.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_adamw.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_grokadamw.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_grokfast.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_lion.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_looksam.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_muon.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_prodigy.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_supergrok11.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_supergrok15.hip` | 62 |
| `csrc/fused/gfx942/mega_vit_supergrok2.hip` | 62 |
| `csrc/backends/hip/gfx942/device_adamw.hip` | 23 |
| `csrc/backends/hip/gfx942/device_grokadamw.hip` | 16 |
| `csrc/backends/hip/gfx942/device_grokfast.hip` | 16 |
| `csrc/backends/hip/gfx942/device_lion.hip` | 16 |
| `csrc/backends/hip/gfx942/device_looksam.hip` | 16 |
| `csrc/backends/hip/gfx942/device_neuralgrok.hip` | 16 |
| `csrc/backends/hip/gfx942/device_prodigy.hip` | 16 |
| `csrc/backends/hip/gfx942/device_supergrok11.hip` | 16 |
| `csrc/backends/hip/gfx942/device_supergrok15.hip` | 16 |
| `csrc/backends/hip/gfx942/device_supergrok2.hip` | 16 |
| `csrc/backends/hip/gfx942/device_attention.hip` | 15 |
| `csrc/backends/hip/gfx942/device_mamba3.hip` | 15 |
| `csrc/backends/hip/gfx942/device_muon.hip` | 15 |
| `csrc/backends/hip/gfx942/device_transformer_decoder.hip` | 15 |
| `csrc/backends/hip/gfx942/device_vit.hip` | 15 |

**Shell** (15 files, 934 lines)

| File | Lines |
|------|------:|
| `build.sh` | 306 |
| `scripts/amdgcn_check.sh` | 113 |
| `.perf/batch/run_12h_frontload.sh` | 96 |
| `scripts/install_deps.sh` | 77 |
| `scripts/bootstrap_env.sh` | 72 |
| `.regpressure/gpu/patchgate.sh` | 47 |
| `.fast_build_env.sh` | 45 |
| `.regpressure/compile_one.sh` | 38 |
| `.build_tools/nvcc-cached` | 30 |
| `.regpressure/gpu/baseline_chain.sh` | 27 |
| `scripts/verify_stage0.sh` | 26 |
| `LICENSE` | 21 |
| `scripts/compile_to_object.sh` | 15 |
| `.regpressure/env.sh` | 14 |
| `.build_tools/g++-cached` | 7 |

</details>


---

> **Status honesty.** The **NVIDIA sm_90 (H100)** path is **verified on real
> silicon**: the `_ops` extension builds, links, imports, and runs on an H100
> 80GB, the fused kernels are **numerically parity-exact** (11/0 on
> `tests/hw/parity_gate_h100.py`) and **maximal** (11/0 on `profile_maximal.py`:
> sm_90a WGMMA/TMA, no wgmma-serialization, 0 live spills), and the grokking race
> trains and **groks** there — **8/11 optimizers** reach the grok threshold on
> `a÷b mod 97`. **Muon (400 steps) and Prodigy (1,000 steps) were both
> flat-at-random for all 15k steps before this audit's on-silicon fixes**; Muon is
> now the fastest of the field and holds its solution. 7 of the 8 grok cleanly and
> sustain ~100%; Prodigy hits the threshold at step 1,000 but does **not yet
> sustain** it (its adaptive `d` later destabilizes — final 0.007). See
> [`results/h100_grokking_race/`](results/h100_grokking_race/). Running on-device
> surfaced (and this branch fixes) a cluster of silent kernel bugs the CPU
> `nvcc -c` gate could not catch — inverted weight-decay in fused Muon, a
> degree-bug + unbounded d-ratchet in Prodigy, a pybind-copied frozen
> step-counter and meta-net OOB in SuperGrok, a silently-throwing SAM step, and
> more. Remaining 🟡: the **gfx942 (MI300X)** and **TPU v6e** runtime paths, and
> the **3 SuperGrok variants** — whose sm_90 kernels are now parity-exact but
> whose *trained meta-net* destabilizes training (a research-owned dynamics issue,
> **not** a kernel bug: freezing the meta-net (rescale=0) reduces SuperGrok1.1 to
> a layerwise-β1 AdamW and it groks at step 2,700 — see the race results README).
> Per-arch detail in
> [`HARDWARE_VALIDATION.md`](HARDWARE_VALIDATION.md). Historical phase/build
> reports moved to [`archived_reports/`](archived_reports/).

---

## 1. The 44 → 99 architecture

**44 components**, each with exactly one canonical home:

| component group | count | canonical home |
|-----------------|------:|----------------|
| optimizer × arch | 11 × 3 = 33 | per-arch (below) |
| model × arch | 3 × 3 = 9 | per-arch (below) |
| dispatch + compile | 2 | `csrc/fused/` + `grokking_optimizers/megakernel*.py` |

**11 optimizers:** AdamW, Lion, Grokfast, GrokAdamW, LookSAM, Prodigy,
NeuralGrok, Muon, SuperGrok1.1, SuperGrok1.5, SuperGrok2.
**3 models:** Transformer-Decoder, ViT, Mamba-3.
**3 archs:** sm_90, gfx942, tpu_v6e.

The **dispatch/compile layer composes any optimizer component with any model
component** into one fused L3/L1 persistent megakernel per (model, optimizer,
arch) → **99 pipelines**. Each cell is a *real composition* of the canonical
component device-functions — there are **no template wrappers, no demo
includes** (anti-false-positive sweep = 0).

### Canonical directory layout (one source per component)

```
csrc/algorithms/<opt>.h            ← CANONICAL per-element optimizer math (CUDA),
                                      ONE definition per optimizer. The SG2
                                      bilevel adjoint lives in
                                      supergrok2_bilevel_adjoint.h.
                                      SOURCE_OF_TRUTH.md documents the contract.

grokking_optimizers/kernels/
  sm_90/<opt>_sm90.cuh             ← per-op LAUNCH wrapper (#includes the
                                      canonical header; zero math duplication)
  sm_90/<model>_sm90.cuh           ← CANONICAL CUDA model (CUTLASS Sm90 TMA/WGMMA)
  gfx942/<opt|model>_gfx942.hip.hpp← CANONICAL AMDGCN device kernels (MFMA/DPP +
                                      f32x4-vectorized apply-steps)
  (there is NO kernels/tpu/ tree — the canonical TPU path is pallas, below)

csrc/backends/
  cuda/sm_90/*.cu                  ← pure entry-point shims (~5 LOC, #include only)
  hip/gfx942/*.hip.cpp             ← entry points + amdgcn_primitives + SG2
                                      device adjoint + MoE compaction
  pallas/launch_<opt>.py           ← CANONICAL TPU/JAX math for ALL 11 optimizers
  pallas/_pallas_models.py         ← CANONICAL TPU model fwd/bwd (decoder/vit/mamba)
  pallas/_pallas_fused.py          ← composes the 33 TPU fused cells

csrc/fused/
  sm_90/opt_components.cuh         ← apply_optimizer<OptId> → csrc/algorithms
  sm_90/model_stages.cuh           ← element-local model fwd/bwd
  sm_90/fused_megakernel.cuh       ← the composition seam (L3/L1 persistent kernel)
  gfx942/{opt_components,model_stages,fused_megakernel}.hip.hpp
  {sm_90,gfx942,tpu_v6e}/mega_<model>_<opt>.{cu,hip,py}  ← the 99 real cells
  megakernel_common*.{cuh,hip.hpp} ← task queue, %smid/HW_ID pin, GridBarrier
```

**Single-source guarantee (enforced).** The CUDA per-op path and the fused path
both `#include` `csrc/algorithms/<opt>.h` and CALL its step function — they
cannot drift. The enforced guard `scripts/check_math_single_source.py` (wired
into `--self-test` as `math_drift_guard`) *fails the build* on three triggers:
(1) a consumer stops `#include`-ing the canonical header; (2) a consumer keeps
the include but **re-inlines** the Adam moment-update/apply locally (Phase-7
re-inline detection — catches the subtle case where math is re-typed in the
`.cuh`); (3) the canonical math changes without a deliberate `--update-manifest`
(content-hash manifest). The gfx942 device transcription and the TPU JAX path
are documented, cross-referenced re-expressions (necessary: thrust/JAX toolchain
constraints), covered by the manifest. The C++ fused dispatch table is
generator-emitted (`csrc/fused/fused_wired_cells.inc`) from the same solver
enumeration that emits the 99 cells, so it cannot hand-sync-drift.

---

## 2. Per-arch story

- **sm_90 (Hopper):** inlined PTX in the owning headers (`rsqrt.approx`,
  `ex2.approx`, `fma.rn`, `redux.sync`, …); **CUTLASS Sm90 collectives**
  (TMA + WGMMA) for the model GEMMs, with a **TF32 (`tfloat32_t`) tensor-core
  path** for FP32 (scalar fallback only for untileable shapes, or forced via
  `-DSG_FORCE_SCALAR_FP32`); warp-specialized producer/consumer register split
  (`setmaxnreg`) in the fused megakernel; L2-persistence + cluster/DSMEM
  helpers.
- **gfx942 (CDNA3 / MI300X):** hand-written **AMDGCN** device kernels —
  `__builtin_amdgcn_mfma_*` (bf16 16×16) for Muon Newton-Schulz + SG2 PEER/attn,
  **DPP wave-64 reductions** for the reducing optimizers (LookSAM/Prodigy/Muon/
  SG1.1/SG1.5), FNUZ FP8, `buffer_load`→LDS, `sched_group_barrier` interleave.
  The device kernels are the **LIVE path on a hipcc build** (`#if __HIPCC__` →
  `hipLaunchKernelGGL`); ATen/rocBLAS is the `#else` **CPU fallback**. The SG2
  gfx942 bilevel adjoint + MoE compaction are real AMDGCN device code.
- **tpu_v6e:** **Pallas** programs (`pl.pallas_call` + `BlockSpec`) composed by
  `_pallas_fused.py` into one `jax.jit` fused program per cell (splash-attention
  where available, hand-tiled dense fallback otherwise; `lax.associative_scan`
  for Mamba).

---

## 3. Fused-megakernel substrate + feasibility solver

One persistent kernel (one CTA per SM/CU) runs **forward → grid-barrier →
backward → grid-barrier → optimizer** in a single launch, over a global
task-queue with `%smid`/`HW_ID` SM-pinning and a hand-built sense-reversing
GridBarrier. The feasibility solver (`grokking_optimizers/megakernel.py`,
`solve_all`) picks the highest fusion tier that fits each arch's register/smem
budget:

- **L3** (fwd+bwd+opt fused), **L1** (optimizer-only fused).
- Current solver assignment: **77 / 99 L3, 22 / 99 L1** (after the Phase-4
  register pass: SMEM staging + rematerialization + the `setmaxnreg` warp-group
  split). 🟡 **These tiers are estimates** — `ptxas -v` / `rocm-llvm` on real
  silicon is the arbiter; the per-cell `maxrregcount` autotuner sweep in
  `compile.py` selects the winner on hardware. The winning launch params
  (`block`/`vec`/`unroll`/`async_depth` + `maxrregcount`) are persisted to
  `grokking_optimizers/_kernel_tuned.json` and injected per-TU into the
  product `_ops*.so` on the next build — see §5 *Tuned-kernel linkage*.

The 99 cells are generated by `grokking_optimizers/megakernel_codegen.py`
(`--emit <model> <optimizer> <arch>` / `--write-all`); cell header comments
(tier/reg/smem) are generator-emitted from the live solver so they cannot drift.

---

## 4. Distributed training

`grokking_optimizers/distributed.py`: 3D parallelism (`ParallelConfig` +
`DistributedContext`, Megatron-style DP×TP×PP rank mesh with TP innermost) +
**ZeRO-3** sharding (DeepSpeed-or-native shim) over **NCCL (NVIDIA) / RCCL
(AMD)**. All `torch.distributed` access is guarded → a single-rank run is a
no-op with no collective launch. The fused step integrates via
`megakernel_engine.py` (the `FusedBackwardHook` / `MegakernelOptimizer` adapter
that reconciles the fused L3 launch with the framework's separate
fwd/bwd/`step()` contract).

---

## 5. Build

```bash
# NVIDIA (Hopper), with CUTLASS Sm90 collectives:
git submodule update --init third_party/cutlass
FORCE_CUDA=1 WITH_CUTLASS=1 TORCH_CUDA_ARCH_LIST="9.0a" \
  pip install -e . --no-build-isolation

# AMD (MI300X):
WITH_HIP=1 pip install -e . --no-build-isolation     # requires ROCm/hipcc

# TPU v6e:
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# CPU-only (host build; device paths take their ATen/JAX fallbacks):
pip install -e . --no-build-isolation
```

`setup.py` resolves the source set per configuration (verified 0-missing /
0-dangling): WITH_CUDA = 49 sources, WITH_HIP = 46 sources (each incl. the 33
fused cells).

### Tuned-kernel linkage (autotuner → product build)

The `compile.py` autotuner's winners are **applied to the shipped
`_ops*.so`**, not just emitted to a header. When
`grokking_optimizers/_kernel_tuned.json` exists, `setup.py`'s
`TunedBuildExtension` reads it and injects the five safe per-TU launch
parameters — `-DSG_TUNED_BLOCK_SIZE` / `_VEC_WIDTH` / `_UNROLL` /
`_ASYNC_DEPTH` (and `--maxrregcount=N` when nonzero) — onto the nvcc command
of each optimizer's CUDA translation unit (`launch_<opt>.cu` and the
`mega_<model>_<opt>.cu` megakernel cells map to `<opt>`; bindings, model-only,
and common TUs get nothing and keep the in-header defaults). The flags land in
`build.ninja` as per-build-statement `cuda_post_cflags` overrides, so a rebuilt
extension actually contains the tuned codegen.

**Absent the JSON, the build is byte-identical to before** — every kernel uses
its in-header `#ifndef SG_TUNED_*` default (block 256 / vec 4 / unroll 1 /
async_depth 2; nvcc's own register allocator). Producing winners + rebuilding +
verifying is documented in `AUTOTUNE_LINKAGE.md`; in brief:

```bash
python -m grokking_optimizers.compile --optimizer <opt> --model <model> \
    --arch sm_90 --jit-only      # per optimizer → _kernel_tuned.json
pip install -e . --no-build-isolation        # tuned flags now baked in
grep -E 'SG_TUNED|maxrregcount' build/*/build.ninja   # expect nonzero hits
```

---

## 6. Verification (this environment — no accelerator)

```bash
# The end-all-be-all: prove the modular composition compiles AND runs maximally.
python -m grokking_optimizers.verify_all                # 152/152, all phases
python -m grokking_optimizers.verify_all --phase 4      # just MAXIMALITY (fast)
python -m grokking_optimizers.verify_all --quick        # skip the 99 compiles

# Full-scale binary profiling: prove the emitted machine code is MAXIMAL.
python -m grokking_optimizers.profile_maximal           # 17/17, all tiers
python -m grokking_optimizers.profile_maximal --quick   # tier D (functional) only

# The individual gates verify_all orchestrates:
python -m grokking_optimizers.compile --self-test     # 156 passed, 0 failed
ruff check grokking_optimizers/ && ruff format --check grokking_optimizers/
python scripts/check_math_single_source.py            # drift guard (exit 0)
scripts/amdgcn_check.sh --header <gfx942 header>       # clang AMDGPU device gate
scripts/amdgcn_check.sh --cell <gfx942 mega_*.hip>     # full composed-cell gate
scripts/compile_to_object.sh <tu>.cu -DWITH_CUTLASS    # nvcc -c sm_90a
```

**`verify_all` is the single authoritative gate.** It runs six phases: (0)
toolchain probe, (1) structural inventory, (2) single-component compile gates,
(3) **MODULAR COMPOSITION** — every optimizer compiles *together with* every
model across all **99 fused cells** (33 sm_90 via `nvcc -c`, 33 gfx942 via the
AMDGCN device gate, 33 tpu_v6e via `jax` trace+lower), (4) **MAXIMALITY** —
every cell at its max feasible fusion tier, codegen idempotency (all 99 cells
byte-identical to the generator), register+smem budget, math single-source
drift, (5) cross-validation — dispatch tables match their generators, self-test,
ruff, the utilization crash-hard contract. Anything needing absent hardware is
reported `SKIP-silicon`, never a false green.

System-verified: `verify_all` **152/152, 0 fail** — self-test **156/0**; ruff
clean; **17/17** gfx942 headers + **33/33** gfx942 fused cells AMDGCN_OK;
**33/33** sm_90 cells `nvcc -c` OK; **33/33** tpu_v6e cells trace+lower OK;
**99/99** cells byte-idempotent vs the generator and 5-way consistent (canonical
file ↔ solver tier ↔ cell comment ↔ dispatch route ↔ status table); fusion-tier
map sm_90 **L3×33** / gfx942 **L3×11 + L1×22** / tpu_v6e **L3×33**; the
math-drift guard passes and **provably triggers** on injected divergence.

**Binary maximality** (`profile_maximal` **23/23**, real emitted-code numbers —
all three target archs get the SAME standard):
- **sm_90 (H100)** — SASS via `cuobjdump` + `ptxas -v`: the GEMM TUs emit Hopper
  **WGMMA tensor cores (80–176/TU) + TMA async copies (84–164/TU)** via the
  CUTLASS Sm90 collectives, with the wgmma mainloop **not serialized** (ptxas
  C7509 = 0, after `-DNDEBUG` strips the CUTLASS asserts that an extern
  `__assert_fail` otherwise forced into the pipeline) and **zero register
  spills**; the fused megakernel cells run at **30–32 real registers** (vs the
  255 budget) with **0 spills**.
- **gfx942 (MI300X)** — real AMDGCN ISA via `llvm-objdump` + `llvm-readobj`:
  the attention kernel emits **20 `v_mfma_f32_16x16x16_bf16`** matrix-core
  instructions + 36 DPP cross-lane ops; decoder/vit/mamba emit `v_mfma` in-ISA;
  real **VGPR ≤ 105 / 255** from the AMDGPU kernel descriptor.
- **tpu_v6e (Trillium)** — optimized HLO via `jax` compile + host run: every
  fused cell compiles to **`dot_general` MXU matmuls (202–618/cell) + XLA
  fusion (271–744/cell)** and **executes finite** on CPU; the v6e binding uses
  the **256-wide MXU tile**.
- **functional**: the optimizer math **provably descends** (Adam core
  32→3.6e-8, Lion 32→2.8e-12).

What remains is **silicon-only** for every arch: wall-clock latency/throughput,
achieved occupancy + bandwidth (ncu/rocprof), the autotuner's measured-latency
config selection, the gfx942 L1→L3 promotion (dynamic-LDS via rocprof), and real
MXU instruction emission on the TPU.

### Observability — device utilization across all 33 pipelines per arch

`grokking_optimizers/utilization.py` is a **live device-utilization sweep**: for
a given arch it runs each of the **33 fused pipelines** (11 optimizers × 3
models) under a sustained load while a low-overhead background poller samples the
device, then emits one structured record per pipeline (mean/peak compute % +
memory %, peak device MB) as a table + JSON.

```bash
python -m grokking_optimizers.utilization --arch tpu_v6e            # sweep all 33
python -m grokking_optimizers.utilization --arch sm_90a -O supergrok2  # one optimizer ×3
python -m grokking_optimizers.utilization --arch gfx942 --list      # enumerate, no device
```

Per-arch sampler backend: NVIDIA → `pynvml` `nvmlDeviceGetUtilizationRates`
(SM% + mem%, `nvidia-smi` fallback); AMD → `amdsmi` / `rocm-smi --showuse`
(GPU use% + VRAM%); TPU → JAX `device.memory_stats()` for live HBM utilization
(MXU compute duty-cycle is xprof-only — see `grokking_optimizers.profile`). It
complements `grokking_optimizers.profile` (one-shot ncu/rocprof/jax.profiler
dump) and `bench_backends` (wall-clock). **Failure policy: crash hard, crash
loud.** If the sampler library is missing, the device is absent, the workload
fails, or the poller can't read a counter, the module raises immediately with a
clear, attributable exception — no graceful degradation, no null-metric
fallback records. Fix the environment, don't paper over it. The enumeration,
aggregation math, and JSON/table schema are CPU-tested in `--self-test`; the
actual **numbers** are silicon-only.

---

## 7. Honest status — LIVE / FALLBACK / DORMANT and what's 🟡

| path | status |
|------|--------|
| sm_90 fused L3/L1 + TF32 model GEMM | **LIVE**, nvcc-object-verified; tiers + runtime 🟡 (ptxas/H100) |
| gfx942 device kernels (11 opt + SG2 fwd/bwd/MoE) | **LIVE on hipcc** (`#if __HIPCC__`); ATen = **FALLBACK** (CPU). clang-gate-verified; host-launch + numerics 🟡 (MI300X) |
| TPU Pallas fused (33 cells) | **LIVE**, trace+lower-verified; on-TPU runtime 🟡 (v6e) |
| SG2 bilevel adjoint | **LIVE** (ATen vendor-neutral on CPU; AMDGCN device adjoint on hipcc); numerics 🟡 |
| math-drift guard | **LIVE + enforced** in `--self-test` |

**The only remaining work class is on-silicon execution + numeric parity** — no
code is blocked on anything but real H100 / MI300X / TPU v6e hardware. Every such
item is a concrete row in the 99-cell checklist in `HARDWARE_VALIDATION.md`.

---

## 8. The grokking race

`grokking_race_v2.py` compares all 11 optimizers (AdamW baseline + 10
grokking-aware variants) head-to-head on algorithmic learning tasks under
controlled conditions — the project's namesake driver.

## 9. Deeper docs
- [`HARDWARE_VALIDATION.md`](HARDWARE_VALIDATION.md) — the 99-cell on-silicon checklist + per-stage bring-up.
- [`BUILD_REPORT.md`](archived_reports/BUILD_REPORT.md) — per-stage scope, gates, the 44-component table.
- [`RESTRUCTURE_PLAN.md`](archived_reports/RESTRUCTURE_PLAN.md) — Phase-6 inventory of the (already clean-layered) architecture. NOTE: the codebase was already clean layering, NOT parallel math trees; Phase 7 then closed the residual real gaps — deleted the dead `kernels/tpu/` duplicate, de-inlined 3 optimizers' Adam math to `algorithms/`, hardened the drift guard to catch re-inlining, made the C++ dispatch table generator-driven, and vectorized the 11 gfx942 apply-steps.
- [`archived_reports/PHASE{3,4,5,6,7}_REPORT.md`](archived_reports/) — the incremental build history (real compositions, register pass, AMD device live-wiring + vectorization, enforced drift guard, dead-tree removal).
- `csrc/algorithms/SOURCE_OF_TRUTH.md` — the optimizer-math canonical contract.

> **Implementation-maximal.** Across sm_90 / gfx942 / tpu the implementation is
> complete: single canonical math source per component (enforced), no dead
> duplicate trees, generator-driven dispatch, vectorized AMD apply-steps. The
> ONLY remaining work class is on-silicon validation (gap #7) — the
> `HARDWARE_VALIDATION.md` runbook on real H100 / MI300X / TPU v6e — to move 🟡 → ✅.
