"""DeepSeek-V4.1-Flash configuration and size presets.

Field names follow ``ModelArgs`` in the official inference reference
(``inference/model.py`` of deepseek-ai/DeepSeek-V4.1-Flash), which is also the
format of its ``inference/config.json``. :meth:`Config.from_json` reads that
file directly.

Presets
-------
``flash``
    The released architecture (552B total parameters, 196B of them in the two
    Engram tables). Useful for parameter counting; far too large to build here.
``tiny`` and ``h100`` use plain RoPE (``original_seq_len=0``): YaRN is a
context-extension setting of the released checkpoint, not part of training
from scratch.

``tiny``
    About 4M parameters at the race vocabulary (3.1M of them Engram tables),
    every mechanism present (sliding-window-only layers, compressed-sparse layers with shared KV and re-indexing, the
    decoder half with candidate blocks, Engram, MoE with a shared expert, mHC).
    Runs on CPU in tests and smoke races.
``h100``
    About 8.7B total / 1.3B active parameters (race vocabulary) with the same
    layer plan as ``flash``: the single-GPU scale the race targets. See
    ``docs/DEEPSEEK_V41.md`` for the memory budget.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace


@dataclass
class Config:
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"  # softmax | sigmoid | sqrtsoftplus
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # attention: MQA with a low-rank query and a grouped low-rank output
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    # sparse attention
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    # manifold-constrained hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # Engram n-gram memory
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_max_ngram_size: int = 4
    # each (n-gram order, head) table has ~engram_vocab_size buckets (the prime search starts there).
    # None: 160 buckets per vocabulary entry, the released model's ratio (16M / 99,092 tokens).
    engram_vocab_size: int | None = 16_000_000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    # training-only (not in the inference reference)
    max_seq_len: int = 4096
    init_std: float = 0.02
    router_bias_update_rate: float = 1e-3  # aux-loss-free balancing speed (V3/V4/V4.1)
    balance_loss_alpha: float = 1e-4  # sequence-wise balance loss weight (V4/V4.1)

    def __post_init__(self):
        self.compress_ratios = tuple(self.compress_ratios[: self.n_layers])
        self.kv_source_layers = tuple(self.kv_source_layers)
        self.index_source_layers = tuple(self.index_source_layers)
        self.engram_layer_ids = tuple(self.engram_layer_ids)
        if len(self.compress_ratios) != self.n_layers:
            raise ValueError("compress_ratios needs one entry per layer")
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.n_shared_experts != 1:
            raise ValueError("the reference architecture has exactly one shared expert")
        for layer in range(self.n_layers):
            r = self.compress_ratios[layer]
            if r and self._source(layer, self.kv_source_layers) is None:
                raise ValueError(f"layer {layer} compresses but no kv source at or before it shares ratio {r}")
            if r and self._source(layer, self.index_source_layers) is None:
                raise ValueError(f"layer {layer} compresses but has no index source at or before it")
        for layer in self.kv_source_layers:
            if not self.compress_ratios[layer] or layer not in self.index_source_layers:
                raise ValueError(f"kv source {layer} must compress and also be an index source")
        if (
            self.candidate_source_layer >= 0
            and (self.candidate_topk_blocks - 1) * self.candidate_block_size + 1 < self.index_topk
        ):
            # the pool can hold as little as (blocks - 1) full blocks plus the query's partial newest block;
            # a smaller pool would leave top-k picking positions outside it
            raise ValueError("(candidate_topk_blocks - 1) * candidate_block_size + 1 must be >= index_topk")

    def _source(self, layer: int, sources: tuple[int, ...]) -> int | None:
        """The latest source at or before ``layer`` whose ratio matches ``layer``'s."""
        r = self.compress_ratios[layer]
        cands = [s for s in sources if s <= layer and self.compress_ratios[s] == r]
        if not cands:
            return None
        src = max(cands)
        if any(self.compress_ratios[k] != r for k in range(src, layer + 1)):
            return None
        return src

    @property
    def engram_buckets(self) -> int:
        """Where each Engram table's prime search starts (~its bucket count per n-gram order and head).

        ``engram_vocab_size`` when set; otherwise 160 buckets per vocabulary entry (+1 for the
        look-back pad id), the released model's ratio, derived from the *final* vocab size.
        """
        return self.engram_vocab_size if self.engram_vocab_size is not None else 160 * (self.vocab_size + 1)

    def kv_source(self, layer: int) -> int | None:
        return self._source(layer, self.kv_source_layers) if self.compress_ratios[layer] else None

    def index_source(self, layer: int) -> int | None:
        return self._source(layer, self.index_source_layers) if self.compress_ratios[layer] else None

    @classmethod
    def from_json(cls, path: str, **overrides) -> "Config":
        """Read an inference ``config.json`` (vision / DSpark / quantization keys are ignored)."""
        with open(path) as f:
            raw = json.load(f)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        known.update(overrides)
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


def _tiny() -> Config:
    # 6 layers mirroring the flash plan: SWA-only, then a compressed (ratio 2) stretch with a kv/index
    # source and a reuse layer, then the decoder half (ratio 1) whose first layer is the kv source,
    # index source and candidate source, followed by a re-index layer.
    return Config(
        vocab_size=128,
        dim=64,
        moe_inter_dim=64,
        n_layers=6,
        n_heads=4,
        n_routed_experts=8,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        compress_ratios=(0, 2, 2, 1, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 5),
        index_n_heads=4,
        index_head_dim=16,
        index_topk=512,
        candidate_source_layer=3,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        engram_layer_ids=(1, 3),
        engram_vocab_size=None,
        engram_n_heads=2,
        engram_head_dim=16,
        original_seq_len=0,
        max_seq_len=256,
    )


def _h100() -> Config:
    # The flash layer plan (40 layers, encoder->decoder at 20, the same compression ratios, KV/index
    # sources, window and top-k) at d=2048 with 64 routed experts: ~8.7B total / ~1.3B active with the
    # race's tiny vocabulary. The routed-expert count is the memory knob: it changes total parameters
    # without changing per-token compute (16 experts: ~2.7B). Engram tables follow the released
    # buckets-per-token ratio, which is only small for small vocabularies: for full-vocabulary language
    # modelling pass an explicit engram_vocab_size (the ratio would make the tables ~64B).
    return Config(
        dim=2048,
        moe_inter_dim=512,
        n_layers=40,
        n_heads=16,
        n_routed_experts=64,
        n_activated_experts=6,
        q_lora_rank=512,
        head_dim=256,
        rope_head_dim=64,
        o_groups=4,
        o_lora_rank=512,
        index_n_heads=16,
        index_head_dim=128,
        engram_vocab_size=None,
        engram_n_heads=4,
        engram_head_dim=128,
        original_seq_len=0,
    )


PRESETS = {"flash": Config, "tiny": _tiny, "h100": _h100}


def get_config(preset: str, **overrides) -> Config:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    cfg = PRESETS[preset]()
    return replace(cfg, **overrides) if overrides else cfg


__all__ = ["Config", "PRESETS", "get_config"]
