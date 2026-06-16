# Tuning report — model=mamba
Objective: steps to val_acc ≥ 0.95 (cap 12000; DNF = cap + (1−peak_val)·cap). Seed 1001.

## adamw — 23 trials (0 complete, 21 pruned)

## neuralgrok — 23 trials (0 complete, 20 pruned)

## grokadamw — 21 trials (0 complete, 20 pruned)

## supergrok — 21 trials (0 complete, 20 pruned)

## supergrok15 — 22 trials (0 complete, 20 pruned)

## supergrok2 — 20 trials (0 complete, 20 pruned)

## grokfast — 21 trials (0 complete, 21 pruned)

## muon — 34 trials (4 complete, 21 pruned)
- best: trial#29 value=18000 (DNF)
- params: `{"muon_lr": 0.02745476401529831, "muon_momentum": 0.9100807737110097, "lr": 0.0005223405875454703, "weight_decay": 1.010888698598515}`

## lion — 21 trials (0 complete, 20 pruned)

## looksam — 22 trials (0 complete, 20 pruned)

## prodigy — 20 trials (0 complete, 19 pruned)
