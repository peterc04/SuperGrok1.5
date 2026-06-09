"""Pallas/JAX backend for TPU v6e (Trillium).

Algorithm launchers (one per optimizer):
  launch_adamw, launch_grokadamw, launch_grokfast, launch_lion,
  launch_looksam, launch_muon, launch_neuralgrok,
  launch_prodigy, launch_supergrok11, launch_supergrok15, launch_supergrok2

The MoE-aware variant (formerly launch_moe_adam) is folded into
launch_supergrok2 — `launch_moe_adam_step` lives at the bottom of that
file, since MoEAwareSuperGrok2 is a SuperGrok v2 subclass.

Model launchers (one per model):
  models/decoder, models/vit, models/mamba

Primitives (shared math + Pallas re-exports):
  primitives
"""
