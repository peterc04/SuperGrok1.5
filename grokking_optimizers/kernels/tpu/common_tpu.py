"""Common TPU/Pallas helpers shared by all model kernel modules.

Provides NanPolicy enum, dtype conversion helpers, and BlockSpec
utilities for Pallas kernel composition.
"""

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
