# Apply-ready implementation spec — `grokking_optimizers/compile.py` (+ `dispatch.py`)

AREA: `grokking_optimizers/compile.py`
Drafts reconciled: `phase2/s34_and_bugs.md`, `phase2/s14_l2persist.md`, `phase2/s15_smem_carveout.md`.
All OLD blocks below are copied VERBATIM from the live tree at
`/workspace/SuperGrok1.5/` as read 2026-06-25. Apply with exact-match edits.

Gate after applying:
```
python -m grokking_optimizers.compile --self-test
ruff check grokking_optimizers/
```

## Scope decisions (read first — these bind the edits below)

1. **Bug #3 (ABI guard) lands in `grokking_optimizers/dispatch.py`**, not compile.py
   (that is where `_ops` is resolved; bindings.cpp:115-119 explicitly owes the
   assertion to dispatch.py). bindings.cpp is **NOT** touched (GROK_ABI_SCHEMA
   already exported at bindings.cpp:120/153). No new self-test case is added for it
   in compile.py, so `_SELF_TEST_EXPECTED_COUNT` is unaffected by Bug #3.

2. **Negcache default = OFF (`enable_negcache: bool = False`).** PROJECT RULE: new
   knobs must be byte-identical when OFF / at default. A `False` default means the
   canonical build, the fp64 parity lane, and the A/A/A determinism lane all run the
   exact pre-knob code path (every probe/harvest call is guarded by
   `getattr(spec, "enable_negcache", False)`). The lead flips it to `True` only after
   the keep/revert ratchet validates skip-reproducibility. (The draft left True-vs-
   False as an open question; OFF is the gate-safe choice and is what the determinism
   harness needs anyway.)

3. **Negcache harvests only DETERMINISTIC rejections** — `infeasible` (prefilter),
   `cost_model_pruned`, and `numerical_fail` (the differential gate already failed).
   Transient build/time `fail` is **NOT** cached (draft RISK #3 option (a),
   recommended): a flaky toolchain hiccup must be re-tried, never permanently
   suppressed.

4. **S1.4 / S1.5 dims are byte-identical at default by construction** (first value ==
   the in-header `#ifndef` default / OMIT sentinel) and are auto-PINNED dead by
   `_pin_dead_dims` until the kernel headers gain the `#ifndef SG_TUNED_*` guards.
   The kernel-side `.cuh` edits are owned by the kernel author (cited in the drafts,
   out of this compile.py spec's scope); this spec lands ONLY the compile.py side, so
   at apply time the new dims are inert (pinned) and the canonical build is unchanged.
   That is intentional and safe — the dims activate automatically once the headers land.

5. Self-test count: **2 new compile.py cases** are added (1 for Bug #2 gfx942
   sentinel, 1 for negcache). `_SELF_TEST_EXPECTED_COUNT` goes `265 -> 267`.
   (The negcache case is a single `run(...)` that internally exercises bloom +
   record/probe + persistence + invalidation, so it counts as one case — keeping the
   count delta minimal and unambiguous.)

================================================================================
# PART 1 — THE THREE BUG FIXES
================================================================================

## Bug #1 — inline `run_device_pgo_round` (drop the self-alias import + swallowed ImportError)

File: `grokking_optimizers/compile.py` (in `_build_aot_pgo`, ~17650).

### OLD
```python
    if spec.enable_device_pgo:
        try:
            from grokking_optimizers.device_profiling import (
                run_device_pgo_round,
            )
            device_workload_cmd = [
                sys.executable, str(workload),
                "--so",    str(so_path),
                "--opt",   OPT_CLASS[spec.optimizer],
                "--model", spec.model,
                "--arch",  spec.arch,
                "--steps", str(int(spec.pgo_steps)),
            ]
            run_device_pgo_round(
                spec, device_workload_cmd, spec.out_dir, report)
        except ImportError:
            pass

    return so_path
```

### NEW
```python
    if spec.enable_device_pgo:
        # run_device_pgo_round + collect_device_stalls live in THIS module
        # (consolidated; run_device_pgo_round is defined later in this file).
        # Call it directly: the former
        # `from grokking_optimizers.device_profiling import run_device_pgo_round`
        # only resolved via the self-referential sys.modules alias registered at
        # module-import time (_register_consolidated_module_aliases), and its bare
        # `except ImportError: pass` silently turned the whole device-PGO round
        # into a no-op if that alias was ever absent (partial import / reordering).
        # The direct call removes the ordering dependency and the swallowed-
        # ImportError trap; run_device_pgo_round's OWN try/except still contains
        # any profiler failure to a logged `[device-pgo] FAILED: ...` warning.
        device_workload_cmd = [
            sys.executable, str(workload),
            "--so",    str(so_path),
            "--opt",   OPT_CLASS[spec.optimizer],
            "--model", spec.model,
            "--arch",  spec.arch,
            "--steps", str(int(spec.pgo_steps)),
        ]
        run_device_pgo_round(
            spec, device_workload_cmd, spec.out_dir, report)

    return so_path
```

Notes:
- `run_device_pgo_round` (defined at ~32288) opens with
  `if not spec.enable_device_pgo: return None` and wraps its body in
  `try/except Exception` writing `[device-pgo] FAILED: ...`. Removing the outer
  try is therefore safe — a missing profiler degrades to a logged warning.
- The self-test at ~20305-20425 that imports through the alias is UNCHANGED (the
  alias registration is untouched), so the legacy-import path still works and the
  self-test count does not move from Bug #1.

--------------------------------------------------------------------------------
## Bug #2 — `resolve_extra_hipcc_flags` skips the `-1` (uncapped) sentinel

File: `grokking_optimizers/compile.py` (~3415-3427).

### OLD
```python
    """HIPCC analogue of resolve_extra_nvcc_flags.

    Currently emits:
      - ``maxrregcount`` -> ``-mllvm -amdgpu-max-num-vgprs=N``
      - feature macros for AMDGPU capabilities (MFMA / WMMA / fp8 / fp4 / dpp / tgsplit)
      - layout macros for tuned dtypes (``fp8_layout``, ``fp4_layout``)
    """
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.extend(["-mllvm", f"-amdgpu-max-num-vgprs={int(v)}"])
```

### NEW
```python
    """HIPCC analogue of resolve_extra_nvcc_flags.

    Currently emits:
      - ``maxrregcount`` -> ``-mllvm -amdgpu-max-num-vgprs=N`` (the sentinel
        ``_MAXRREGCOUNT_UNCAPPED`` (-1) OMITS the flag so the AMDGPU register
        allocator runs its own heuristic — mirrors the NVCC ``--maxrregcount``
        sentinel skip in resolve_extra_nvcc_flags).
      - feature macros for AMDGPU capabilities (MFMA / WMMA / fp8 / fp4 / dpp / tgsplit)
      - layout macros for tuned dtypes (``fp8_layout``, ``fp4_layout``)
    """
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            # Mirror the NVCC sibling (resolve_extra_nvcc_flags): the uncapped
            # sentinel (_MAXRREGCOUNT_UNCAPPED == -1) OMITS the flag so the AMDGPU
            # register allocator runs its own heuristic. Emitting
            # `-amdgpu-max-num-vgprs=-1` is malformed (the option is an unsigned
            # VGPR count). Any config carrying maxrregcount == -1 (e.g. a
            # sibling/transfer-seeded config from the sm_90 space, whose
            # maxrregcount list LEADS with the sentinel) would otherwise ship a
            # broken flag on the HIP path; the NVCC path already guards it.
            if v is not None and int(v) != _MAXRREGCOUNT_UNCAPPED:
                out.extend(["-mllvm", f"-amdgpu-max-num-vgprs={int(v)}"])
```

--------------------------------------------------------------------------------
## Bug #2 regression test (gfx942 maxrregcount sentinel)

File: `grokking_optimizers/compile.py`. Add this case at the END of
`_self_test_flags`, immediately after the `run(...)` for
`def_load_cache_and_register_usage_level_are_tuned_dims` and BEFORE the next
top-level `def _self_test_silent_degradation`.

### OLD
```python
        assert not any("register-usage-level" in f for f in omit), omit
        assert not any("def-load-cache" in f for f in omit), omit
    run("def_load_cache_and_register_usage_level_are_tuned_dims",
        test_def_load_cache_and_register_usage_level_are_tuned_dims)


def _self_test_silent_degradation(run) -> None:
```

### NEW
```python
        assert not any("register-usage-level" in f for f in omit), omit
        assert not any("def-load-cache" in f for f in omit), omit
    run("def_load_cache_and_register_usage_level_are_tuned_dims",
        test_def_load_cache_and_register_usage_level_are_tuned_dims)

    def test_hipcc_maxrregcount_uncapped_sentinel_omits_flag():
        """gfx942 mirror of the NVCC sentinel skip: the uncapped sentinel
        OMITS -amdgpu-max-num-vgprs (the option is an unsigned VGPR count, so
        an emitted `=-1` would be malformed); a real value DOES emit the cap."""
        sp = build_full_search_space()
        if "gfx942" not in sp:
            return  # arch not in this build's table — nothing to assert
        dims = sp["gfx942"]["dims"]
        omitted = resolve_extra_hipcc_flags(
            {"maxrregcount": _MAXRREGCOUNT_UNCAPPED}, dims, "gfx942")
        assert not any("amdgpu-max-num-vgprs" in f for f in omitted), omitted
        capped = resolve_extra_hipcc_flags(
            {"maxrregcount": 96}, dims, "gfx942")
        assert "-amdgpu-max-num-vgprs=96" in capped, capped
    run("hipcc_maxrregcount_uncapped_sentinel_omits_flag",
        test_hipcc_maxrregcount_uncapped_sentinel_omits_flag)


def _self_test_silent_degradation(run) -> None:
```

> Robustness: the test guards on `"gfx942" in sp` and passes the sentinel /
> value EXPLICITLY (it does not depend on the dim's value list), so it is
> arch-list-independent. `build_full_search_space` and `resolve_extra_hipcc_flags`
> are module-level (same scope used by the adjacent NVCC sentinel test).
>
> IMPORTANT correction to the draft: the gfx942 maxrregcount dim is
> `list(range(32, 256, 4))` (compile.py:2387) and does NOT prepend the sentinel —
> ONLY the sm_90 dim leads with `_MAXRREGCOUNT_UNCAPPED` (compile.py:2238). So the
> gfx942 CANONICAL (first-value) build today is `=32`, not the malformed `=-1`; the
> bug surfaces only when a config carries `maxrregcount == -1` (a transfer/sibling
> seed from the sm_90 space, a hand-written config, or cross-arch reuse). The fix is
> still correct and the NVCC sibling already guards the same case; the spec comment
> in the NEW block above is worded accordingly (it does NOT claim a gfx942 canonical
> first value of -1). `96` is chosen for the positive case because it is in
> `range(32, 256, 4)`.

--------------------------------------------------------------------------------
## Bug #3 — ABI-schema guard in `grokking_optimizers/dispatch.py`

File: `grokking_optimizers/dispatch.py`. bindings.cpp is NOT touched.

### Edit 3a — add the `EXPECTED_ABI_SCHEMA` constant just before `class _LazyOps`

#### OLD
```python
# that profiling / tooling code can introspect the package without a
# working build.
# ----------------------------------------------------------------------


class _LazyOps:
```

#### NEW
```python
# that profiling / tooling code can introspect the package without a
# working build.
# ----------------------------------------------------------------------


# ── Exported-ABI schema the current Python wrappers are written for ──────────
# MUST stay in lockstep with GROK_ABI_SCHEMA in csrc/bindings/bindings.cpp:120.
# Bump BOTH in the same commit that changes any exported fused-step signature
# (arg add/remove/reorder, tensor<->scalar swap, return-tuple change, symbol
# rename). _LazyOps._resolve() asserts the loaded _ops.__abi_schema__ matches
# this; a PRESENT-but-different value fails loudly (a stale .so paired with
# newer wrappers otherwise mis-marshals arguments silently — exactly what
# bindings.cpp warns about at lines 108-119). An ABSENT attribute is a
# pre-guard build: warn once and proceed (don't brick old binaries).
EXPECTED_ABI_SCHEMA: int = 1


class _LazyOps:
```

### Edit 3b — patch `_LazyOps._resolve` (note: live `__slots__` carries `_attr_cache`)

#### OLD
```python
    def _resolve(self):
        if self._real is not None:
            return self._real
        if self._error is not None:
            return None  # cached failure
        try:
            import importlib
            real = importlib.import_module("grokking_optimizers._ops")
            object.__setattr__(self, "_real", real)
            return real
        except ImportError as e:
            object.__setattr__(self, "_error", e)
            logger.debug("grokking_optimizers._ops not importable "
                         "(extension unbuilt?): %s", e)
            return None
```

#### NEW
```python
    def _resolve(self):
        if self._real is not None:
            return self._real
        if self._error is not None:
            return None  # cached failure
        try:
            import importlib
            real = importlib.import_module("grokking_optimizers._ops")
        except ImportError as e:
            object.__setattr__(self, "_error", e)
            logger.debug("grokking_optimizers._ops not importable "
                         "(extension unbuilt?): %s", e)
            return None
        # ABI-schema compatibility guard (bindings.cpp:115-119 owes this to
        # dispatch.py). A PRESENT-but-mismatched schema means the prebuilt .so
        # and these Python wrappers disagree on the fused-step signature and
        # would silently mis-marshal — fail loudly. An ABSENT attribute is a
        # pre-guard build: warn once, proceed (don't brick old binaries).
        got = getattr(real, "__abi_schema__", None)
        if got is not None and int(got) != EXPECTED_ABI_SCHEMA:
            err = UnsupportedArchError(
                f"grokking_optimizers._ops ABI schema mismatch: the compiled "
                f"extension reports __abi_schema__={int(got)} but these Python "
                f"wrappers expect {EXPECTED_ABI_SCHEMA}. The prebuilt .so is "
                f"stale relative to the wrappers (fused-step signatures may "
                f"have changed); rebuild with "
                f"`pip install -e . --force-reinstall`."
            )
            object.__setattr__(self, "_error", err)
            raise err
        if got is None:
            logger.warning(
                "grokking_optimizers._ops has no __abi_schema__ attribute "
                "(pre-guard build); skipping ABI compatibility check. Rebuild "
                "to enable it.")
        object.__setattr__(self, "_real", real)
        return real
```

### Edit 3c — make `__bool__` swallow the mismatch so `has_kernels()` stays a clean predicate

#### OLD
```python
    def __bool__(self):
        return self._resolve() is not None
```

#### NEW
```python
    def __bool__(self):
        # has_kernels()/capability probes use truthiness and must stay
        # non-raising. A hard ABI mismatch surfaces on actual attribute access
        # (the hot path) via _resolve()'s raise; here we report "not usable"
        # rather than propagating, so capability gating still returns a clean
        # False.
        try:
            return self._resolve() is not None
        except UnsupportedArchError:
            return False
```

> Direct attribute access (`_ops.fused_step` → `__getattr__` → `_resolve`) still
> raises on a present-mismatch, so the optimizers' hot path fails loudly as mandated.
> `__getattr__` calls `_resolve()` *after* checking `_attr_cache`; on a mismatch
> `_resolve` raises before any attr is cached, and `_error` is set so a *second*
> `__bool__`/`__getattr__` short-circuits via the `if self._error is not None`
> branch (returns None → `__getattr__` raises its build-hint AttributeError). That is
> acceptable: the loud raise already happened once on the hot path.

================================================================================
# PART 2 — S3.4 CROSS-RUN NEGATIVE CACHE + BLOOM DEDUP
================================================================================

## 2.0 — add `import base64`

File: `grokking_optimizers/compile.py`, top-of-file stdlib import block.

### OLD
```python
import collections
import concurrent.futures
import copy
import datetime
import hashlib
```

### NEW
```python
import base64
import collections
import concurrent.futures
import copy
import datetime
import hashlib
```

--------------------------------------------------------------------------------
## 2.1 — `_NegCacheBloom` class + `_negcache_config_hash` helper

File: `grokking_optimizers/compile.py`. Insert in the consolidated tail,
IMMEDIATELY BEFORE `def run_device_pgo_round(`.

### OLD
```python
def run_device_pgo_round(spec, workload_cmd: List[str],
                         out_dir: Path, report=None) -> Optional[Path]:
    if not spec.enable_device_pgo:
        return None
```

### NEW
```python
# ==============================================================================
# S3.4 — cross-run negative cache (bloom dedup over PROVEN-bad config hashes).
# Pure-Python / JSON / CPU-only. A pure work-SKIPPER: it never injects a config
# into the winner pool, never changes a timing, never relaxes the fp64/SAM/A-A-A
# gate. It only avoids re-evaluating configs already proven infeasible /
# numerical_fail / cost-model-pruned in a prior run for the SAME search space.
# ==============================================================================

_NEGCACHE_REASONS_CAP: int = 50_000   # bound the exact-match audit dict


class _NegCacheBloom:
    """Tiny fixed-capacity bloom filter (double-hashing, base64-JSON
    serialisable). Pure-Python, no third-party deps; CPU-only. Gives the
    autotuner an O(1) 'have I ever proven this config bad?' probe that survives
    across runs. ``m`` is a power of two so index masking is a bitwise AND."""
    __slots__ = ("m", "k", "n", "_bits")

    def __init__(self, m_bits: int = 1 << 21, k: int = 13, n: int = 0,
                 bits: Optional[bytearray] = None):
        self.m = int(m_bits)
        self.k = int(k)
        self.n = int(n)
        self._bits = bits if bits is not None else bytearray(self.m // 8)

    def _indices(self, key: str):
        d = hashlib.sha256(key.encode("utf-8")).digest()
        h1 = int.from_bytes(d[0:8], "big")
        h2 = int.from_bytes(d[8:16], "big") | 1   # odd -> full period
        mask = self.m - 1                          # m is a power of two
        for i in range(self.k):
            yield (h1 + i * h2) & mask

    def add(self, key: str) -> None:
        for idx in self._indices(key):
            self._bits[idx >> 3] |= (1 << (idx & 7))
        self.n += 1

    def __contains__(self, key: str) -> bool:
        for idx in self._indices(key):
            if not (self._bits[idx >> 3] & (1 << (idx & 7))):
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "m": self.m, "k": self.k, "n": self.n,
            "bits_b64": base64.b64encode(bytes(self._bits)).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_NegCacheBloom":
        bits = bytearray(base64.b64decode(d["bits_b64"]))
        return cls(m_bits=int(d["m"]), k=int(d["k"]), n=int(d.get("n", 0)),
                   bits=bits)


def _negcache_config_hash(config: Dict[str, Any]) -> str:
    """Hash basis for the negative cache. Uses the SAME canonical identity
    (config_key) the variant-artifact cache keys by, so two configs differing
    only in a dead/kernel-ignored dim collapse to one negative — exactly the
    dedup semantics we want. Deterministic (sorted keys via config_key)."""
    return hashlib.sha256(config_key(config).encode("utf-8")).hexdigest()


def run_device_pgo_round(spec, workload_cmd: List[str],
                         out_dir: Path, report=None) -> Optional[Path]:
    if not spec.enable_device_pgo:
        return None
```

--------------------------------------------------------------------------------
## 2.2 — `negcache` entry default (no cache-version bump)

`None` is forward/backward-inert for every existing reader, so no v5 bump. Fold a
`negcache` default into `_V4_DEFAULTS` so it rides the existing `_fresh_entry` +
`get()` defaulting paths automatically (both iterate `_V4_DEFAULTS`/`_V3_DEFAULTS`).

> NOTE: `get()` (compile.py:8038) only re-defaults `_V3_DEFAULTS`. `_fresh_entry`
> (compile.py:7389) folds `_V4_DEFAULTS` into new entries, which covers fresh
> caches. For an OLD on-disk v4 cache lacking `negcache`, the recorder methods
> (`_negcache_entry`) use `e.get("negcache")` with an `isinstance` fallback, so a
> missing key is handled at read time — no migration needed.

File: `grokking_optimizers/compile.py` (~7360).

### OLD
```python
_V4_DEFAULTS: Dict[str, Any] = {
    "trial_log_path":    None,
    "trial_log_summary": {
        "n_trials":         0,
        "best_timing_ms":   None,
        "stop_reason":      None,
        "last_updated_unix": 0.0,
    },
}
```

### NEW
```python
_V4_DEFAULTS: Dict[str, Any] = {
    "trial_log_path":    None,
    "trial_log_summary": {
        "n_trials":         0,
        "best_timing_ms":   None,
        "stop_reason":      None,
        "last_updated_unix": 0.0,
    },
    # S3.4 — cross-run negative cache sidecar (per (opt,model,arch)). None is
    # forward/backward-inert for every existing reader, so this needs no
    # cache-version bump; the recorder lazily initialises the sub-dict.
    "negcache": None,
}
```

--------------------------------------------------------------------------------
## 2.3 — CompileCache recorder + probe methods

File: `grokking_optimizers/compile.py`. Insert AFTER `record_variant` and BEFORE
`get_fresh_variant`. `self._lock` is an `RLock` (compile.py:7697), so calling
`self.get(...)` (which acquires `_lock`) while already holding `_lock` is safe.

### OLD
```python
                e["variant_artifacts"][config_key] = rec
                self._dirty = True

    def get_fresh_variant(self, opt: str, model: str, arch: str,
                          config_key: str,
                          build_sig: str) -> Optional[Path]:
```

### NEW
```python
                e["variant_artifacts"][config_key] = rec
                self._dirty = True

    # ── S3.4 negative cache ──────────────────────────────────────────────
    def _negcache_entry(self, opt: str, model: str, arch: str,
                        space_hash: str) -> dict:
        """Get-or-reset the per-(opt,model,arch) negcache sub-dict, resetting
        the bloom + reasons when the search-space hash changed (stale negatives
        from a different prefilter/space/kernel MUST NOT gate the new space).
        Caller holds self._lock (RLock; get() re-enters it safely)."""
        e = self.get(opt, model, arch)
        nc = e.get("negcache")
        if not isinstance(nc, dict) or nc.get("space_hash") != space_hash:
            nc = {"space_hash": space_hash,
                  "bloom": _NegCacheBloom().to_dict(),
                  "reasons": {}, "version": 1}
            e["negcache"] = nc
            self._dirty = True
        return nc

    def negcache_seen(self, opt: str, model: str, arch: str,
                      space_hash: str, config: Dict[str, Any]) -> bool:
        """O(1) probe: True iff this config was PROVEN bad in a prior (or this)
        run for the SAME space. Bloom gate + EXACT confirm so a bloom collision
        never reports a never-seen config as seen (conservative: a false bloom
        hit that misses the exact dict re-evaluates the config — costs time,
        never correctness)."""
        with self._lock:
            nc = self._negcache_entry(opt, model, arch, space_hash)
            h = _negcache_config_hash(config)
            bloom = _NegCacheBloom.from_dict(nc["bloom"])
            if h not in bloom:
                return False
            return h in nc.get("reasons", {})   # exact confirm

    def negcache_add(self, opt: str, model: str, arch: str,
                     space_hash: str, config: Dict[str, Any],
                     reason: str) -> None:
        """Record a config as PROVEN-bad for this space. FIFO-bounds the exact
        audit dict; the bloom never shrinks within a space_hash, so the dedup
        signal survives eviction (only the audit trail / exact-confirm is lost
        for evicted entries — that path re-evaluates, which is safe)."""
        with self._lock:
            nc = self._negcache_entry(opt, model, arch, space_hash)
            h = _negcache_config_hash(config)
            reasons = nc.setdefault("reasons", {})
            if h in reasons:
                return
            bloom = _NegCacheBloom.from_dict(nc["bloom"])
            bloom.add(h)
            nc["bloom"] = bloom.to_dict()
            if len(reasons) >= _NEGCACHE_REASONS_CAP:
                reasons.pop(next(iter(reasons)))   # cheap FIFO (insertion order)
            reasons[h] = str(reason)
            self._dirty = True

    def get_fresh_variant(self, opt: str, model: str, arch: str,
                          config_key: str,
                          build_sig: str) -> Optional[Path]:
```

--------------------------------------------------------------------------------
## 2.4 — `_merge_disk_entries`: union the negcache across concurrent writers

File: `grokking_optimizers/compile.py` (~8002-8012, inside the per-key loop, right
after the `variant_artifacts` union and before the trailing comment line).

### OLD
```python
            # Variant artefacts: union (different timer subprocesses
            # may have populated different ckeys).
            d_va = de.get("variant_artifacts")
            m_va = me.get("variant_artifacts")
            if isinstance(d_va, dict):
                if not isinstance(m_va, dict):
                    me["variant_artifacts"] = dict(d_va)
                else:
                    for vk, vv in d_va.items():
                        m_va.setdefault(vk, vv)
            # host_history: append disk's new entries to in-memory.
```

### NEW
```python
            # Variant artefacts: union (different timer subprocesses
            # may have populated different ckeys).
            d_va = de.get("variant_artifacts")
            m_va = me.get("variant_artifacts")
            if isinstance(d_va, dict):
                if not isinstance(m_va, dict):
                    me["variant_artifacts"] = dict(d_va)
                else:
                    for vk, vv in d_va.items():
                        m_va.setdefault(vk, vv)
            # S3.4 negcache: union when the space_hash matches (OR the two bloom
            # bitarrays, union the bounded reasons dicts); on a space_hash
            # mismatch keep the in-memory one (its timestamp already won above).
            # Never let a malformed negcache break the locked save.
            d_nc = de.get("negcache")
            m_nc = me.get("negcache")
            if isinstance(d_nc, dict):
                if not isinstance(m_nc, dict):
                    me["negcache"] = d_nc
                elif d_nc.get("space_hash") == m_nc.get("space_hash"):
                    try:
                        mb = _NegCacheBloom.from_dict(m_nc["bloom"])
                        db = _NegCacheBloom.from_dict(d_nc["bloom"])
                        if mb.m == db.m and len(mb._bits) == len(db._bits):
                            for i in range(len(mb._bits)):
                                mb._bits[i] |= db._bits[i]
                            mb.n = mb.n + db.n   # upper bound; exactness N/A
                            m_nc["bloom"] = mb.to_dict()
                        mr = m_nc.setdefault("reasons", {})
                        for hk, rv in (d_nc.get("reasons") or {}).items():
                            if hk not in mr and len(mr) < _NEGCACHE_REASONS_CAP:
                                mr[hk] = rv
                    except Exception as _swexc:
                        _debug_swallow('_merge_disk_entries', _swexc)
            # host_history: append disk's new entries to in-memory.
```

--------------------------------------------------------------------------------
## 2.5 — `BuildSpec`: add `enable_negcache` (default OFF)

File: `grokking_optimizers/compile.py` (~8554).

### OLD
```python
    # Stream 8 — device-side PGO (CUPTI / rocprof / XLA HLO dump)
    enable_device_pgo: bool = False
```

### NEW
```python
    # Stream 8 — device-side PGO (CUPTI / rocprof / XLA HLO dump)
    enable_device_pgo: bool = False
    # S3.4 — cross-run negative cache (skip configs proven infeasible /
    # numerical-fail / cost-model-pruned in a prior run for the SAME space).
    # Pure work-skipper; never changes a winner, never relaxes the fp64/SAM/
    # A-A-A gate. DEFAULT OFF so the canonical / parity / determinism lanes run
    # the exact pre-knob code path (every probe/harvest call is guarded by
    # getattr(spec, "enable_negcache", False)). Flip to True only after the
    # keep/revert ratchet validates that skips are reproducible.
    enable_negcache: bool = False
```

--------------------------------------------------------------------------------
## 2.6 — `run_bayesian`: optional negcache closures (gate + harvest)

File: `grokking_optimizers/compile.py`. `run_bayesian` does NOT receive `cache`/`spec`,
so it takes two optional callables; the caller (`_run_bayesian`) binds them to the
cache + keys ONLY when `enable_negcache` is on, keeping `run_bayesian` cache-free for
the self-tests that call it directly.

### Edit 2.6a — signature (add two params before the closing `)`)

#### OLD
```python
    stall_info: Optional[Dict[str, Any]] = None,
    bias_max_enqueued: int = 25,
    stream: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
```

#### NEW
```python
    stall_info: Optional[Dict[str, Any]] = None,
    bias_max_enqueued: int = 25,
    stream: Optional[Any] = None,
    negcache_probe: Optional[Any] = None,
    negcache_sink: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
```

### Edit 2.6b — gate before feasibility + harvest in the infeasible/cost-model branches

#### OLD
```python
        trial = study.ask()
        cfg = _suggest(trial, dims)
        if not is_feasible(cfg):
            # Optuna ≥ 4.0 forbids passing a value when state is
            # PRUNED / FAIL — it raises ValueError. Drop the value arg
            # (the prefilter rejection has no meaningful objective).
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "infeasible"
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg)
            continue
```

#### NEW
```python
        trial = study.ask()
        cfg = _suggest(trial, dims)
        # S3.4 — short-circuit a config PROVEN bad in a prior run for this
        # space. Mirrors the infeasible branch (PRUNED + stopper.observe(inf))
        # so convergence detection is unaffected; the real win is skipping the
        # expensive timer(cfg) compile+run for a known build/numeric failure.
        if negcache_probe is not None and negcache_probe(cfg):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "negcache_skip"
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg)
            continue
        if not is_feasible(cfg):
            # Optuna ≥ 4.0 forbids passing a value when state is
            # PRUNED / FAIL — it raises ValueError. Drop the value arg
            # (the prefilter rejection has no meaningful objective).
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "infeasible"
            if negcache_sink is not None:
                negcache_sink(cfg, "infeasible")
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg)
            continue
```

### Edit 2.6c — harvest the cost-model-pruned branch

#### OLD
```python
        if pruned_by_cost_model:
            # Optuna ≥ 4.0 forbids passing a value with state=PRUNED.
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rec = dict(raw)
            rec["trial_num"] = trial.number
            rec.setdefault("stage", "tpe")
            rec.setdefault("config_key", config_key(cfg))
            rec.setdefault("host", host)
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg, was_pruned_by_cost_model=True)
            continue
```

#### NEW
```python
        if pruned_by_cost_model:
            # Optuna ≥ 4.0 forbids passing a value with state=PRUNED.
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rec = dict(raw)
            rec["trial_num"] = trial.number
            rec.setdefault("stage", "tpe")
            rec.setdefault("config_key", config_key(cfg))
            rec.setdefault("host", host)
            if negcache_sink is not None:
                negcache_sink(cfg, "cost_model_pruned")
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg, was_pruned_by_cost_model=True)
            continue
```

> NOTE: the transient build/time `fail` branch (`rec["status"] = "fail"`) is
> deliberately NOT harvested (RISK #3 option (a)) — a flaky compile must be re-tried.

--------------------------------------------------------------------------------
## 2.7 — `_run_bayesian`: bind the closures + numerical_fail harvest

File: `grokking_optimizers/compile.py`. `_run_bayesian` already has `spec`, `cache`,
`space_hash` in scope. Pass the closures to `run_bayesian` only when negcache is on,
and harvest numerical_fail trials after the loop.

### Edit 2.7a — the `run_bayesian(...)` call site

#### OLD
```python
    try:
        tpe_trials, stop_info = run_bayesian(
            spec.arch, space, n_trials=n_trials, seed=spec.seed,
            storage=storage,
            study_name=f"sg_{spec.optimizer}_{spec.model}_{spec.arch}",
            timer=effective_timer, progress=progress1, host=_current_host(),
            prefiltered=prefiltered,
            pruner=spec.pruner,
            seed_trials=seed_trials,
            stopper=stopper,
            stall_info=stall_info_for_bias,
            bias_max_enqueued=25,
        )
    finally:
        close1()

    for t in tpe_trials:
        cache.record_trial(spec.optimizer, spec.model, spec.arch, t)
    cache.save()
```

#### NEW
```python
    # S3.4 — cross-run negative cache closures (probe + harvest), bound to this
    # (opt,model,arch,space_hash). Only constructed when enable_negcache is on,
    # so the default flow is byte-identical to today (run_bayesian sees None for
    # both and never touches the cache). Pure work-skipper; never changes a
    # winner or relaxes the fp64/SAM/A-A-A gate.
    _nc_on = bool(getattr(spec, "enable_negcache", False))
    if _nc_on:
        def _nc_probe(_cfg):
            return cache.negcache_seen(spec.optimizer, spec.model, spec.arch,
                                       space_hash, _cfg)

        def _nc_sink(_cfg, _reason):
            cache.negcache_add(spec.optimizer, spec.model, spec.arch,
                               space_hash, _cfg, _reason)
    else:
        _nc_probe = None
        _nc_sink = None

    try:
        tpe_trials, stop_info = run_bayesian(
            spec.arch, space, n_trials=n_trials, seed=spec.seed,
            storage=storage,
            study_name=f"sg_{spec.optimizer}_{spec.model}_{spec.arch}",
            timer=effective_timer, progress=progress1, host=_current_host(),
            prefiltered=prefiltered,
            pruner=spec.pruner,
            seed_trials=seed_trials,
            stopper=stopper,
            stall_info=stall_info_for_bias,
            bias_max_enqueued=25,
            negcache_probe=_nc_probe,
            negcache_sink=_nc_sink,
        )
    finally:
        close1()

    for t in tpe_trials:
        cache.record_trial(spec.optimizer, spec.model, spec.arch, t)
        # S3.4 — harvest numerical_fail trials (deterministic given space_hash:
        # the fp64/SAM differential already failed). The negcache is invalidated
        # on any space_hash change, so a numerical_fail under an OLD binary can
        # never suppress re-testing under a NEW one.
        if _nc_on and t.get("numerical_status") == "numerical_fail":
            cache.negcache_add(spec.optimizer, spec.model, spec.arch,
                               space_hash, t.get("config", {}), "numerical_fail")
    cache.save()
```

--------------------------------------------------------------------------------
## 2.8 — `_run_exhaustive`: skip known-bad + harvest numerical_fail

File: `grokking_optimizers/compile.py` (~16525-16553). `spec`, `cache`, `space_hash`
all in scope.

### OLD
```python
        for i, cfg in enumerate(configs, 1):
            ckey = config_key(cfg)
            report.write(f"\n  [{i}/{n_total}] {ckey}\n")
            report.flush()
            t0 = time.monotonic()
            result = timer(cfg)
            elapsed = time.monotonic() - t0
            # Stream 10: per-variant numerical-validation tag + origin are
            # stashed by the variant timer under config_key(cfg).
            num_status = _LAST_NUMERICAL_STATUS.get(ckey, "skipped")
            origin = _LAST_VARIANT_ORIGIN.get(ckey, ORIGIN_TEMPLATE)
            trial = {
                "trial_num":   i,
                "stage":       "exhaustive",
                "config":      cfg,
                "config_key":  ckey,
                "timing_ms":   result["timing_ms"] if result else None,
                "min_ms":      result["min_ms"]    if result else None,
                "max_ms":      result["max_ms"]    if result else None,
                "n":           result["n"]         if result else None,
                "host":        _current_host(),
                "numerical_status": num_status,
                "origin":      origin,
                "recorded_at": datetime.datetime.now().isoformat(),
                "status":      "ok" if result else "fail",
                "build_s":     elapsed,
            }
            cache.record_trial(spec.optimizer, spec.model, spec.arch, trial)
            all_trials.append(trial)
```

### NEW
```python
        _nc_on = bool(getattr(spec, "enable_negcache", False))
        for i, cfg in enumerate(configs, 1):
            ckey = config_key(cfg)
            # S3.4 — skip a config PROVEN bad in a prior run for this space
            # (deterministic rejects only: infeasible / numerical_fail /
            # cost_model_pruned). Pure work-skipper; the skipped config would
            # be rejected by the same gate anyway. OFF by default ⇒ no skips ⇒
            # byte-identical to today.
            if _nc_on and cache.negcache_seen(spec.optimizer, spec.model,
                                              spec.arch, space_hash, cfg):
                report.write(f"\n  [{i}/{n_total}] {ckey}\n"
                             f"    [negcache] skip (known-bad in prior run)\n")
                continue
            report.write(f"\n  [{i}/{n_total}] {ckey}\n")
            report.flush()
            t0 = time.monotonic()
            result = timer(cfg)
            elapsed = time.monotonic() - t0
            # Stream 10: per-variant numerical-validation tag + origin are
            # stashed by the variant timer under config_key(cfg).
            num_status = _LAST_NUMERICAL_STATUS.get(ckey, "skipped")
            origin = _LAST_VARIANT_ORIGIN.get(ckey, ORIGIN_TEMPLATE)
            trial = {
                "trial_num":   i,
                "stage":       "exhaustive",
                "config":      cfg,
                "config_key":  ckey,
                "timing_ms":   result["timing_ms"] if result else None,
                "min_ms":      result["min_ms"]    if result else None,
                "max_ms":      result["max_ms"]    if result else None,
                "n":           result["n"]         if result else None,
                "host":        _current_host(),
                "numerical_status": num_status,
                "origin":      origin,
                "recorded_at": datetime.datetime.now().isoformat(),
                "status":      "ok" if result else "fail",
                "build_s":     elapsed,
            }
            cache.record_trial(spec.optimizer, spec.model, spec.arch, trial)
            # S3.4 — harvest numerical_fail (deterministic differential reject).
            # Transient build/time `fail` is NOT cached (flaky → must re-try).
            if _nc_on and num_status == "numerical_fail":
                cache.negcache_add(spec.optimizer, spec.model, spec.arch,
                                   space_hash, cfg, "numerical_fail")
            all_trials.append(trial)
```

--------------------------------------------------------------------------------
## 2.9 — negcache self-test case

File: `grokking_optimizers/compile.py`. Add one combined case at the END of
`_self_test_cache`, after the last `run(...)`
(`header_edit_busts_source_hash_and_freshness`) and before the next top-level
`def _self_test_multi_gpu_pool`.

### OLD
```python
    run("header_edit_busts_source_hash_and_freshness",
        test_header_edit_busts_source_hash_and_freshness)


def _self_test_multi_gpu_pool(run) -> None:
```

### NEW
```python
    run("header_edit_busts_source_hash_and_freshness",
        test_header_edit_busts_source_hash_and_freshness)

    def test_negcache_bloom_record_probe_persist_invalidate():
        """S3.4 — bloom has no false negatives + round-trips; CompileCache
        records/probes a negative; the negative survives a reopen (cross-run);
        a different space_hash invalidates it; a never-added config is unseen."""
        # 1) Bloom: no false negatives, round-trip preserves membership.
        b = _NegCacheBloom()
        keys = [f"k{i}" for i in range(2000)]
        for k in keys:
            b.add(k)
        assert all(k in b for k in keys), "bloom must not have false negatives"
        b2 = _NegCacheBloom.from_dict(b.to_dict())
        assert all(k in b2 for k in keys), "round-trip must preserve membership"
        # 2) + 3) CompileCache record/probe + cross-run persistence.
        td = Path(tempfile.mkdtemp())
        try:
            cp = td / "neg.json"
            cache = CompileCache(cp)
            cfg = {"block": 256, "vec": 4, "unroll": 8}
            other = {"block": 512, "vec": 2, "unroll": 8}
            assert not cache.negcache_seen("adamw", "decoder", "sm_90a",
                                           "SS1", cfg), "fresh ⇒ unseen"
            cache.negcache_add("adamw", "decoder", "sm_90a", "SS1", cfg,
                               "numerical_fail")
            assert cache.negcache_seen("adamw", "decoder", "sm_90a",
                                       "SS1", cfg), "added ⇒ seen"
            assert not cache.negcache_seen("adamw", "decoder", "sm_90a",
                                           "SS1", other), "other cfg ⇒ unseen"
            cache.save()
            reopened = CompileCache(cp)
            assert reopened.negcache_seen("adamw", "decoder", "sm_90a",
                                          "SS1", cfg), "must survive reopen"
            # 4) space_hash change invalidates (resets the sub-dict).
            assert not reopened.negcache_seen("adamw", "decoder", "sm_90a",
                                              "SS2", cfg), \
                "different space_hash ⇒ invalidated"
        finally:
            shutil.rmtree(td, ignore_errors=True)
    run("negcache_bloom_record_probe_persist_invalidate",
        test_negcache_bloom_record_probe_persist_invalidate)


def _self_test_multi_gpu_pool(run) -> None:
```

> `tempfile`, `shutil`, `Path`, `json` are all already imported at the top of
> `_self_test_cache` (`import shutil` / `import tempfile` at its head; `Path`/`json`
> module-level). `CompileCache`, `_NegCacheBloom`, `config_key` are module-level.

--------------------------------------------------------------------------------
## 2.10 — bump `_SELF_TEST_EXPECTED_COUNT`

File: `grokking_optimizers/compile.py` (~26861). Two new compile.py cases
(Bug #2 gfx942 sentinel + negcache combined). 265 -> 267.

### OLD
```python
_SELF_TEST_EXPECTED_COUNT: int = 265
```

### NEW
```python
_SELF_TEST_EXPECTED_COUNT: int = 267
```

================================================================================
# PART 3 — S1.4 L2-PERSISTENCE TUNED DIMS (compile.py side)
================================================================================

> Kernel-side edits (primitives.cuh `#ifndef SG_TUNED_L2_*` guards + ctor; the
> `#if SG_TUNED_L2_PERSIST` RAII wrapper in fused_decoder_megakernel.cuh) are owned by
> the kernel author per s14_l2persist.md §3.5/§3.6 and are OUT OF SCOPE here. Until
> they land, `_is_dead_dim` PINS these dims to their first value (byte-identical
> canonical build, zero wasted compiles); they auto-activate when the scanner sees the
> tokens. This is the same forward-compatible contract as `dec_gemm_stages`.

## 3.1 — float-literal emission in `resolve_macros`

`l2_hit_ratio` is the project's first float-valued macro dim. Emit a C `float`
literal (trailing `f`) so the device constexpr matches `cudaAccessPolicyWindow.hitRatio`
(a `float`). `isinstance(True, int)` is True, so bool is already handled by
`_format_value`; float currently falls through to `str()` (a `double` literal). The
explicit branch below intercepts float BEFORE the scalar `else`.

File: `grokking_optimizers/compile.py` (~3299-3308).

### OLD
```python
        if isinstance(value, (tuple, list)):
            comps = list(value)
            for i, comp in enumerate(comps):
                out.append(f"-D{macro}_{i}={_format_value(comp)}")
            vol = 1
            for comp in comps:
                vol *= int(comp)
            out.append(f"-D{macro}_VOLUME={vol}")
        else:
            out.append(f"-D{macro}={_format_value(value)}")
    return out
```

### NEW
```python
        if isinstance(value, (tuple, list)):
            comps = list(value)
            for i, comp in enumerate(comps):
                out.append(f"-D{macro}_{i}={_format_value(comp)}")
            vol = 1
            for comp in comps:
                vol *= int(comp)
            out.append(f"-D{macro}_VOLUME={vol}")
        elif isinstance(value, float):
            # S1.4 — float-valued macro (l2_hit_ratio). Emit a C float literal
            # (trailing 'f') so the device constexpr is `float`, not `double`,
            # matching cudaAccessPolicyWindow.hitRatio's float field. repr()
            # gives a round-trippable shortest decimal; suffix 'f'. (bool is
            # already intercepted by _format_value above, so only genuine floats
            # reach this branch.)
            out.append(f"-D{macro}={value!r}f")
        else:
            out.append(f"-D{macro}={_format_value(value)}")
    return out
```

--------------------------------------------------------------------------------
## 3.2 — the 3 L2 dims in `_sm90_full_space`

Insert AFTER the `pipe_depth` dim and BEFORE the `]),` that closes the
`_pin_dead_dims([...])` list.

File: `grokking_optimizers/compile.py` (~2293-2295).

### OLD
```python
            _dim("pipe_depth", "int", [2, 3],
                 "SG_TUNED_PIPE_DEPTH", ["device"]),
        ]),
```

### NEW
```python
            _dim("pipe_depth", "int", [2, 3],
                 "SG_TUNED_PIPE_DEPTH", ["device"]),
            # === S1.4 L2-persistence (cudaAccessPolicyWindow) dims ===========
            # Generic, config-declared L2-residency tuning for the persistent
            # megakernel's hottest reuse (the per-step optimizer momentum
            # st.exp_avg / st.exp_avg_sq). KERNEL side: L2PersistScope in
            # csrc/backends/cuda/sm_90/primitives.cuh (RAII over
            # cudaStreamSetAttribute + cudaAccessPolicyWindow). These dims
            # parameterize it; the launch wrapper constructs the scope.
            # FIRST VALUE == today's behaviour so the canonical (no-JSON) build
            # is byte-identical: l2_persist False ⇒ the wrapper guard is dead ⇒
            # the scope is never constructed; l2_hit_ratio 1.0 == the prior
            # hardcoded hitRatio; l2_setaside_pct 100 == reserving the full span.
            # Auto-PINNED dead by _pin_dead_dims until primitives.cuh / the
            # launcher gain the `#ifndef SG_TUNED_L2_*` guards (same forward-
            # compatible contract as dec_gemm_stages above); then the source
            # scanner activates them. NEEDS-PARITY before a non-default winner
            # ships: an L2 hint cannot change numerics (cache residency only),
            # but the H100 perf gate + A/A/A determinism re-run ratify any
            # winner per keep/revert. l2_hit_ratio is the project's ONLY float-
            # valued macro dim; it emits a C float literal via resolve_macros.
            _dim("l2_persist", "bool", [False, True],
                 "SG_TUNED_L2_PERSIST", ["device"]),
            _dim("l2_hit_ratio", "float", [1.0, 0.75, 0.5, 0.25],
                 "SG_TUNED_L2_HIT_RATIO", ["device"]),
            _dim("l2_setaside_pct", "int", [100, 75, 50, 25],
                 "SG_TUNED_L2_SETASIDE_PCT", ["device"]),
        ]),
```

--------------------------------------------------------------------------------
## 3.3 — route the L2 dims through the real-TC timer (`_GEMM_TILE_DIM_NAMES`)

They change a LAUNCH-TIME stream attribute on the persistent megakernel, invisible
to the cheap `opt.step()` surrogate timer.

File: `grokking_optimizers/compile.py` (~14851-14854).

### OLD
```python
    # pipe_depth is read by tile_pipeline.cuh (the wgmma SW pipeline), also
    # invisible to the cheap timer.
    "pipe_depth",
})
```

### NEW
```python
    # pipe_depth is read by tile_pipeline.cuh (the wgmma SW pipeline), also
    # invisible to the cheap timer.
    "pipe_depth",
    # S1.4 — L2-persistence dims change a LAUNCH-TIME cudaStreamSetAttribute on
    # the real persistent megakernel (not the surrogate opt.step()), so the
    # cheap timer is blind to them — route through the real-TC timer.
    "l2_persist", "l2_hit_ratio", "l2_setaside_pct",
})
```

--------------------------------------------------------------------------------
## 3.4 — feed the L2 macros to the TC TU (`_tc_relevant_device_flags` allowlist)

THE critical wiring edit — without it the macros are dropped before the TC build.

File: `grokking_optimizers/compile.py` (~14897-14900).

### OLD
```python
    keep_macros = (
        "SG_TUNED_TILE_M", "SG_TUNED_TILE_N",
        "SG_TUNED_DEC_DW_SPLITK", "SG_TUNED_VIT_DW_SPLITK",
    )
```

### NEW
```python
    keep_macros = (
        "SG_TUNED_TILE_M", "SG_TUNED_TILE_N",
        "SG_TUNED_DEC_DW_SPLITK", "SG_TUNED_VIT_DW_SPLITK",
        # S1.4 — L2-persistence macros are honoured by the launch wrapper in
        # fused_decoder_megakernel.cuh (L2PersistScope around the persistent
        # launch), which lives IN the TC TU, so they must reach this build.
        "SG_TUNED_L2_PERSIST", "SG_TUNED_L2_HIT_RATIO", "SG_TUNED_L2_SETASIDE_PCT",
        # S1.5 — launch-attribute knobs (carveout + dyn-smem ceiling) are read
        # by the same TC launchers; keep them too (see Part 4).
        "SG_TUNED_SMEM_CARVEOUT", "SG_TUNED_MAX_DYN_SMEM_KB",
    )
```

> Combined with Part 4 so the allowlist is edited ONCE. If applying Parts 3 and 4
> separately, split this into two edits; the final `keep_macros` tuple must contain
> all four pre-existing names plus the five new ones shown above.

================================================================================
# PART 4 — S1.5 SMEM-CARVEOUT + MAX-DYN-SMEM TUNED DIMS (compile.py side)
================================================================================

> Kernel-side edits (the `opt_in` + `#if GROK_CUDA` carveout blocks in the three
> `fused_*_megakernel.cuh` launchers, and the `#ifndef SG_TUNED_*` defaults — preferably
> hoisted into `csrc/fused/megakernel_common.cuh`) are owned by the kernel author per
> s15_smem_carveout.md §4.1-§4.2 and are OUT OF SCOPE here. Both dims are plain int
> `SG_TUNED_*` macros, so `resolve_macros`'s existing `-DMACRO=VAL` path emits them with
> NO code change; they auto-activate via the scanner once the launcher text carries the
> tokens, and are PINNED to their OMIT/EXACT sentinel (byte-identical) until then.

## 4.1 — the 2 launch-attribute dims in `_sm90_full_space`

Insert AFTER the `grad_tile` dim and BEFORE the `# === L3-TC megakernel GEMM / tile
dims (#12)` comment block.

File: `grokking_optimizers/compile.py` (~2188-2190).

### OLD
```python
            _dim("grad_tile", "int", [1024, 2048],
                 "SG_TUNED_GRAD_TILE", ["device"]),
            # === L3-TC megakernel GEMM / tile dims (#12) =====================
```

### NEW
```python
            _dim("grad_tile", "int", [1024, 2048],
                 "SG_TUNED_GRAD_TILE", ["device"]),
            # === S1.5 launch-attribute tuned knobs ===========================
            # cudaFuncAttributePreferredSharedMemoryCarveout (L1-vs-shared split)
            # + a cudaFuncAttributeMaxDynamicSharedMemorySize CEILING above the
            # kernel's required sizeof. BOTH are read by the host launchers in
            # csrc/fused/sm_90/fused_{decoder,vit,mamba}_megakernel.cuh
            # (SG_TUNED_SMEM_CARVEOUT / SG_TUNED_MAX_DYN_SMEM_KB), so the scanner
            # marks them LIVE automatically once those launchers land. PARITY:
            # both only reshape the L1/shared partition + the per-func dyn-smem
            # ceiling — no math, no fp32 accumulation order, no wgmma issue order
            # change, so the HARD fp64/SAM gate is untouched (occupancy/latency
            # lever only). FIRST value is the OMIT/EXACT sentinel so the untuned
            # build is byte-AND-behaviour identical to the pre-S1.5 launcher:
            #   carveout -1 -> launcher SKIPS cudaFuncSetAttribute(carveout)
            #   max_dyn  0  -> launcher opts into EXACTLY sizeof (today's value)
            # Carveout is CUDA-only (no hipFuncAttributePreferredSharedMemory-
            # Carveout); the launcher guards it with #if GROK_CUDA, so on HIP the
            # macro is read but inert. The ceiling is clamped to the device cap
            # at launch (never a launch failure). NEEDS-PARITY before a non-
            # default winner ships for max_dyn (interacts with the ≥1-CTA/SM
            # occupancy cert); carveout is a pure hint and is parity-trivial.
            _dim("smem_carveout", "int", [-1, 100, 0, 25, 50, 75],
                 "SG_TUNED_SMEM_CARVEOUT", ["device"]),
            _dim("max_dyn_smem_kb", "int", [0, 100, 164, 200, 228],
                 "SG_TUNED_MAX_DYN_SMEM_KB", ["device"]),
            # === L3-TC megakernel GEMM / tile dims (#12) =====================
```

--------------------------------------------------------------------------------
## 4.2 — defensive `_LIVE_TUNING_DIMS` floor

Lets an installed-wheel build (no source tree visible to the scanner) still treat
the two dims as live. Optional but recommended (one-line floor; the scanner does the
real work once the launchers land).

File: `grokking_optimizers/compile.py` (~1300-1303).

### OLD
```python
_LIVE_TUNING_DIMS: frozenset = frozenset({
    "block", "vec", "unroll", "async_depth", "cluster_shape", "maxrregcount",
    "tile_m", "tile_n", "dec_dw_splitk", "vit_dw_splitk",
})
```

### NEW
```python
_LIVE_TUNING_DIMS: frozenset = frozenset({
    "block", "vec", "unroll", "async_depth", "cluster_shape", "maxrregcount",
    "tile_m", "tile_n", "dec_dw_splitk", "vit_dw_splitk",
    # S1.5 launch-attribute knobs (defensive floor for the no-source-tree case).
    "smem_carveout", "max_dyn_smem_kb",
})
```

--------------------------------------------------------------------------------
## 4.3 — `_tc_relevant_device_flags` allowlist

Already covered by the COMBINED edit in Part 3.4 (it adds the S1.5 macros alongside
the S1.4 macros). If Part 3 is NOT being applied, instead apply this standalone
variant of the allowlist edit:

### OLD (standalone — only if Part 3.4 is skipped)
```python
    keep_macros = (
        "SG_TUNED_TILE_M", "SG_TUNED_TILE_N",
        "SG_TUNED_DEC_DW_SPLITK", "SG_TUNED_VIT_DW_SPLITK",
    )
```

### NEW (standalone — only if Part 3.4 is skipped)
```python
    keep_macros = (
        "SG_TUNED_TILE_M", "SG_TUNED_TILE_N",
        "SG_TUNED_DEC_DW_SPLITK", "SG_TUNED_VIT_DW_SPLITK",
        # S1.5 — launch-attribute knobs read by the TC launchers in the TC TU.
        "SG_TUNED_SMEM_CARVEOUT", "SG_TUNED_MAX_DYN_SMEM_KB",
    )
```

--------------------------------------------------------------------------------
## 4.4 — prefilter guard rules (optional, defensive)

Mirror the kernel's clamp so a future value-list edit can't request an impossible
ceiling. They prune nothing for the value lists above.

File: `grokking_optimizers/compile.py` (~2321-2325, end of the sm_90 prefilter rules).

### OLD
```python
                {"name": "tile_m_wgmma_atom",
                 "expr": "tile_m % 64 == 0"},
                {"name": "tile_n_wgmma_atom",
                 "expr": "tile_n % 64 == 0"},
            ],
        },
    }
```

### NEW
```python
                {"name": "tile_m_wgmma_atom",
                 "expr": "tile_m % 64 == 0"},
                {"name": "tile_n_wgmma_atom",
                 "expr": "tile_n % 64 == 0"},
                # S1.5 — carveout is a 0..100 percent OR the -1 default sentinel.
                {"name": "carveout_range",
                 "expr": "smem_carveout == -1 or (0 <= smem_carveout <= 100)"},
                # S1.5 — the dyn-smem ceiling (KB) must fit the per-block smem
                # budget (the launcher ALSO clamps at runtime, so this only
                # prunes obviously-impossible value-list edits; 0 = exact-sizeof
                # sentinel always passes).
                {"name": "max_dyn_smem_fits",
                 "expr": "max_dyn_smem_kb == 0 or max_dyn_smem_kb * 1024 <= 232448"},
            ],
        },
    }
```

> HARD DEPENDENCY — §4.4 MUST be applied together with §4.1: `_validate_arch`
> (compile.py:3077-3085) raises `SearchSpaceError` at SPACE-BUILD time if a rule's
> expr references a name that is neither a declared dim nor a safe builtin. These two
> rules reference `smem_carveout` / `max_dyn_smem_kb`, which §4.1 declares as dims in
> the SAME `_sm90_full_space` block — so with §4.1 applied they validate cleanly.
> Applying §4.4 WITHOUT §4.1 would make the self-test's dry-run-all-archs space build
> raise. At SWEEP time, `cartesian()` always emits every declared dim, so configs
> always carry both keys; even if a key were ever absent, `compile_feasibility_check`
> (MED.9, compile.py:3240-3252) PASS-THROUGHs a rule that raises (never a silent
> reject). §4.4 is optional — the kernel-side runtime clamp already guarantees safety;
> skip it entirely if you prefer the minimal surface, but if you keep it, keep it
> paired with §4.1.

================================================================================
# APPLY ORDER + VERIFICATION
================================================================================

Suggested commit split (so the keep/revert ratchet can bisect):
1. Bug #2 (hipcc sentinel) + its test + count bump to 266.
2. Bug #1 (inline device-pgo).
3. Bug #3 (dispatch.py ABI guard).
4. S1.4 dims (Parts 3.1-3.3 + the S1.4 half of 3.4).
5. S1.5 dims (Parts 4.1-4.4 + the S1.5 half of 3.4).
6. S3.4 negcache (Part 2) + its test + count bump to 267.

If landing all at once, the final `_SELF_TEST_EXPECTED_COUNT` is **267**
(265 + 1 Bug#2 case + 1 negcache case). The S1.4/S1.5/S3.4 dim additions do NOT add
self-test cases on their own (no per-dim count assertion exists), so they do not move
the count.

Gate:
```
python -m grokking_optimizers.compile --self-test     # expect "267 passed, 0 failed"
ruff check grokking_optimizers/                        # expect no new findings
```
