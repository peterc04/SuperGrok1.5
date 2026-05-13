"""Pallas/JAX backend for TPU v5p.

Algorithm launchers (one per optimizer):
  launch_adamw, launch_grokadamw, launch_grokfast, launch_lion,
  launch_looksam, launch_moe_adam, launch_muon, launch_neuralgrok,
  launch_prodigy, launch_supergrok11, launch_supergrok15, launch_supergrok2

Model launchers (one per model):
  models/decoder, models/vit, models/mamba

Primitives (shared math + Pallas re-exports):
  primitives
"""
