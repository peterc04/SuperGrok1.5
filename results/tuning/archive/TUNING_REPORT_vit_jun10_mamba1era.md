# Tuning report — model=vit
Objective: steps to val_acc ≥ 0.95 (cap 6000; DNF = cap + (1−peak_val)·cap). Seed 1001.

## adamw — 28 trials (0 complete, 27 pruned)

## neuralgrok — 30 trials (0 complete, 27 pruned)

## grokadamw — 28 trials (0 complete, 27 pruned)

## supergrok — 30 trials (0 complete, 27 pruned)

## supergrok15 — 35 trials (1 complete, 26 pruned)
- best: trial#28 value=3004 (GROKKED)
- params: `{"lr": 0.0020264344137863708, "weight_decay": 1.0931241555793088, "supergrok15_alpha": 0.8801770639064841, "supergrok15_gamma": 0.02282570597919094, "supergrok15_kappa": 0.05521297804988606, "supergrok15_grad_clip": 0.5631417763065227, "supergrok15_meta_lr": 1.2386683828596414e-05, "supergrok15_sam_rho": 0.019218213054925073, "supergrok15_gate_thresh": 0.7420820149948997, "supergrok15_sam_freq_min": 4, "supergrok15_bilevel_freq_min": 4}`

## supergrok2 — 33 trials (0 complete, 29 pruned)

## grokfast — 30 trials (2 complete, 26 pruned)
- best: trial#23 value=6370 (DNF)
- params: `{"lr": 0.00020909081080225337, "grokfast_alpha": 0.9559430980623965, "grokfast_lamb": 1.9846942148901303, "weight_decay": 1.5038180241360077}`

## muon — 29 trials (7 complete, 20 pruned)
- best: trial#28 value=274 (GROKKED)
- params: `{"muon_lr": 0.025365367592727502, "muon_momentum": 0.8877089650766007, "lr": 0.0002875214721930026, "weight_decay": 0.6402088573641292}`

## lion — 31 trials (11 complete, 16 pruned)
- best: trial#29 value=3233 (GROKKED)
- params: `{"lion_lr": 0.0003382252274674619, "lion_wd": 4.987393310037697, "beta1": 0.9481554672656949, "beta2": 0.924630732792004}`

## looksam — 32 trials (10 complete, 22 pruned)
- best: trial#16 value=6613 (DNF)
- params: `{"lr": 0.0003508840588729052, "looksam_rho": 0.012332659335364591, "looksam_k": 10, "looksam_alpha": 0.47624227915481054, "weight_decay": 1.7578908548919283}`

## prodigy — 32 trials (1 complete, 31 pruned)
- best: trial#23 value=3096 (GROKKED)
- params: `{"prodigy_lr": 0.6092871731887447, "weight_decay": 1.6884349231080042, "beta1": 0.8892427177550478, "beta2": 0.9583849271273884}`
