# SuperGrok: a grokking race between optimizers

Which optimizer makes a model *grok*, that is, generalize long after memorizing its
training set, with the fewest gradient steps, seconds and FLOPs? This repository races
nine optimizers on algorithmic tasks, using **DeepSeek-V4.1-Flash** as the trained model.

It is a clean restart. The earlier code (SuperGrok 1.5/2, the CUDA/HIP/TPU kernels and the
old race scripts) is in git history at commit `19c9d39`.

## Contents

| Path | What |
|---|---|
| `grokking_optimizers/` | The optimizers, as plain `torch.optim.Optimizer` classes: AdamW, Lion, Grokfast, GrokAdamW, LookSAM, Prodigy, NeuralGrok, Muon (Kimi K3 per-head) and SuperGrok 1.1 |
| `deepseek_v41/` | A trainable pure-PyTorch port of DeepSeek-V4.1-Flash (MoE, sparse attention, mHC, Engram) with size presets |
| `grokking_race/` | The race: tasks, training loop, cost accounting, plots |
| `tests/` | Every optimizer and the model checked against published or official reference code |
| `docs/ALGORITHMS.md` | How each optimizer works, what was wrong in the old code, its cost, and whether it scales |
| `docs/DEEPSEEK_V41.md` | The model port: what is in it, how it is verified, presets, memory |

## Quick start

```bash
pip install -e ".[test]"
pytest -q                                    # about 2 minutes on CPU

# the race: DeepSeek-V4.1-Flash (tiny preset), modular division mod 97, 50% train, seeds 42/123/456
python -m grokking_race

# a subset, on a GPU
python -m grokking_race --optimizers adamw,muon,supergrok11,supergrok11_frozen --device cuda --dtype bf16

# redraw the plots from a saved run
python -m grokking_race --replot results/<run>/results.json
```

Useful options: `--tasks moddiv,chaindiv`, `--splits 10/90,50/50`, `--p 97`,
`--preset tiny|h100`, `--model-overrides '{"n_routed_experts": 16}'`,
`--hparams '{"lion": {"lr": 1e-4}}'`, `--micro-batches 4`, `--peak-tflops 989` (for MFU),
`--ntfy <topic>` (or `NTFY_TOPIC`) for phone notifications. See
`python -m grokking_race --help`.

## Race rules

* **Two splits, train and test**, as in the original race scripts: a seeded shuffle of
  every example, the first `frac_train` (default 50%) for training and the rest for
  testing. There is no validation split. Test data is only ever evaluated.
* **Grokked** = **test accuracy ≥ 95%, held for 50 consecutive evaluations** (every 10
  steps, so 500 steps). The first crossing is recorded too.
* **Cost to grok**, three ways: gradient steps, training wall-clock seconds (evaluation
  excluded) and training FLOPs. FLOPs are measured with PyTorch's `FlopCounterMode` and
  include everything an optimizer runs itself: SAM passes, meta steps, Newton–Schulz.
* **Same starting point.** Every optimizer gets the same model initialization and data
  split per seed. Adam-family methods share lr 1e-3, betas (0.9, 0.98) and weight decay 1.0,
  and add only their own mechanism.
* **Held-out data** for optimizers that need it (NeuralGrok, GrokAdamW, SuperGrok 1.1) is a
  10% slice carved out of their own training split; they train on the other 90%. Test
  data never drives training: no optimizer sees a test example, loss or accuracy.

Each run writes `results.json` plus four plots per task: the cost to grok in steps,
seconds and FLOPs side by side (`race_*.png`); test accuracy against each of the three
(`test_acc_*.png`); and per-optimizer accuracy and loss curves.

## Scale

The target is a 7–10B-parameter model on a single H100: the `h100` preset (8.69B total,
1.34B active). Plain fp32 PyTorch does not fit it with a stateful optimizer (see the memory
tables in the docs). Fitting it is the job of the CUDA phase: low-precision storage, fused
optimizer steps and activation checkpointing. FP8 and multi-GPU come after that.

## License

MIT (see `LICENSE`). `tests/reference_deepseek/` contains DeepSeek's MIT-licensed
reference code.
