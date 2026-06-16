# Tuning report — model=decoder
Objective: steps to val_acc ≥ 0.95 (cap 5000; DNF = cap + (1−peak_val)·cap). Seed 1001.

## adamw — 45 trials (4 complete, 37 pruned)
- best: trial#11 value=488 (GROKKED)
- params: `{"lr": 0.003872639391054715, "beta1": 0.9080579177135406, "beta2": 0.9984848390636939, "weight_decay": 1.8645260030089423}`

## neuralgrok — 40 trials (0 complete, 38 pruned)

## grokadamw — 40 trials (0 complete, 39 pruned)

## supergrok — 47 trials (3 complete, 35 pruned)
- best: trial#42 value=502 (GROKKED)
- params: `{"supergrok_lamb": 0.8401912754640304, "lr": 0.004589511955180319, "weight_decay": 1.0933940624440435, "supergrok_alpha": 0.841736197760442, "supergrok_gamma": 0.07624947174167841, "supergrok_kappa": 0.594704491625221, "supergrok_grad_clip": 2.1320389998109337, "supergrok_gate_temp": 17.03499617145306, "supergrok_meta_update_freq": 6, "supergrok_meta_lr": 9.689413853742047e-05, "supergrok_warmup": 294}`

## supergrok15 — 47 trials (5 complete, 39 pruned)
- best: trial#37 value=553 (GROKKED)
- params: `{"lr": 0.003587754508046236, "weight_decay": 1.4352057764974981, "supergrok15_alpha": 0.5343532094161323, "supergrok15_gamma": 0.18087400544896082, "supergrok15_kappa": 0.033986271474033214, "supergrok15_grad_clip": 1.5215761744176022, "supergrok15_meta_lr": 0.00022783697100988362, "supergrok15_sam_rho": 0.0502815278793043, "supergrok15_gate_thresh": 0.8698406779949901, "supergrok15_sam_freq_min": 3, "supergrok15_bilevel_freq_min": 7}`

## supergrok2 — 43 trials (0 complete, 38 pruned)

## grokfast — 42 trials (11 complete, 28 pruned)
- best: trial#37 value=462 (GROKKED)
- params: `{"lr": 0.0018104642691119345, "grokfast_alpha": 0.9414868187670322, "grokfast_lamb": 0.584391167283447, "weight_decay": 3.3679450091218053}`

## muon — 42 trials (13 complete, 27 pruned)
- best: trial#39 value=161 (GROKKED)
- params: `{"muon_lr": 0.020816145162188333, "muon_momentum": 0.8865071287939167, "lr": 0.00023469864702148018, "weight_decay": 1.3937783938002215}`

## lion — 45 trials (21 complete, 23 pruned)
- best: trial#42 value=2072 (GROKKED)
- params: `{"lion_lr": 0.0007986982233283509, "lion_wd": 3.6114961932281355, "beta1": 0.9178250862109268, "beta2": 0.9778896895794174}`

## looksam — 41 trials (17 complete, 23 pruned)
- best: trial#35 value=289 (GROKKED)
- params: `{"lr": 0.004826600856689301, "looksam_rho": 0.024025438716680766, "looksam_k": 5, "looksam_alpha": 0.35515805434485204, "weight_decay": 1.539106355830368}`

## prodigy — 42 trials (20 complete, 22 pruned)
- best: trial#36 value=608 (GROKKED)
- params: `{"prodigy_lr": 1.1416300088846958, "weight_decay": 3.9144413576588275, "beta1": 0.8507559845800987, "beta2": 0.9640958430291481}`
