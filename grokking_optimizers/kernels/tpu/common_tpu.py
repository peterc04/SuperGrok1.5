"""Common TPU/Pallas helpers shared by all model kernel modules.

Provides NanPolicy enum, dtype conversion helpers, and BlockSpec
utilities for Pallas kernel composition.
"""

# ----------------------------------------------------------------------------
# TPU TREE STATUS (post-consolidation): REFERENCE / SPEC.
# The AUTHORITATIVE, executed Pallas path for tpu_v5p is
# csrc/backends/pallas/launch_<opt>.py (resolved by grokking_optimizers/
# profile.py and the race driver); the live Pallas kernels live in
# csrc/backends/pallas/_pallas_kernels.py. This *_tpu.py file is the
# arch-reference specification of the same math (JAX/jnp), kept in the
# unified kernel tree alongside the sm_90/.cuh and gfx942/.hip.hpp headers.
# It is NOT imported by the production path (no duplicated executing kernel
# bodies). TPU has no inline-asm concept; lowering is via XLA/Pallas BlockSpec.
# Dependency direction is intentionally the reverse of sm_90/gfx942: there the
# kernel-tree header is canonical and csrc/backends shims to it; for Pallas the
# csrc/backends launcher stays canonical because profile.py executes it by path.
# ----------------------------------------------------------------------------

from __future__ import annotations

import enum
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


class NanPolicy(enum.IntEnum):
    IGNORE = 0
    ZERO = 1
    SKIP = 2


def apply_nan_policy(x: jnp.ndarray, policy: NanPolicy) -> jnp.ndarray:
    if policy == NanPolicy.ZERO:
        return jnp.where(jnp.isnan(x), 0.0, x)
    return x


def to_bf16(x: jnp.ndarray) -> jnp.ndarray:
    return x.astype(jnp.bfloat16)


def to_f32(x: jnp.ndarray) -> jnp.ndarray:
    return x.astype(jnp.float32)


def to_f16(x: jnp.ndarray) -> jnp.ndarray:
    return x.astype(jnp.float16)


PARAM_DTYPE = jnp.bfloat16
ACCUM_DTYPE = jnp.float32
