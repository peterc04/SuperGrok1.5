"""
SuperGrok v1.5 — Comprehensive Test Suite

Tests all v1.5 changes:
  1. 2D SharpnessMetaNet (gradient + sharpness input)
  2. sam_meta_step (LookSAM + bilevel combined)
  3. Progressive weight decay
  + all inherited v1.1 functionality
"""

import sys, math, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, ".")
from supergrok15 import SuperGrok15, SharpnessMetaNet


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

def make_classifier():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: Construction and Defaults
# ═══════════════════════════════════════════════════════════════════════

def test_construction():
    print("\n" + "=" * 60)
    print("TEST 1: Construction and Defaults")
    print("=" * 60)

    m = make_model()
    opt = SuperGrok15(m.parameters())

    check("Default construction", opt is not None)
    check("Default wd=1.0", opt.defaults["weight_decay"] == 1.0)
    check("Meta-net is SharpnessMetaNet", isinstance(opt.meta_net, SharpnessMetaNet))
    check("sam_rho=0.05", opt.sam_rho == 0.05)
    check("sam_freq=5", opt.sam_freq == 5)
    check("wd_ramp=4.0", opt.wd_ramp == 4.0)
    check("wd_scale=20.0", opt.wd_scale == 20.0)
    check("wd_thresh=0.9", opt.wd_thresh == 0.9)

    # Custom construction
    opt2 = SuperGrok15(m.parameters(), sam_rho=0.1, sam_freq=10,
                       wd_ramp=3.0, wd_thresh=0.85)
    check("Custom sam_rho", opt2.sam_rho == 0.1)
    check("Custom sam_freq", opt2.sam_freq == 10)
    check("Custom wd_ramp", opt2.wd_ramp == 3.0)

    # Validation
    try:
        SuperGrok15(m.parameters(), sam_rho=-1)
        check("Negative sam_rho raises", False)
    except ValueError:
        check("Negative sam_rho raises", True)

    try:
        SuperGrok15(m.parameters(), sam_freq=0)
        check("sam_freq=0 raises", False)
    except ValueError:
        check("sam_freq=0 raises", True)

    try:
        SuperGrok15(m.parameters(), wd_ramp=-1)
        check("Negative wd_ramp raises", False)
    except ValueError:
        check("Negative wd_ramp raises", True)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: SharpnessMetaNet
# ═══════════════════════════════════════════════════════════════════════

def test_sharpness_meta_net():
    print("\n" + "=" * 60)
    print("TEST 2: SharpnessMetaNet")
    print("=" * 60)

    net = SharpnessMetaNet(hidden_dim=32)

    # Shape preservation
    for shape in [(10,), (5, 5), (2, 3, 4), (100,)]:
        g = torch.randn(shape)
        s = torch.randn(shape).abs()
        out = net(g, s)
        check(f"Shape {shape} preserved", out.shape == g.shape)

    # Empty tensor
    empty = torch.randn(0)
    check("Empty tensor handled", net(empty, empty).shape == empty.shape)

    # Near-identity at init (rescale=0)
    g = torch.randn(100)
    s = torch.randn(100).abs()
    out = net(g, s)
    diff = (out - g).norm().item()
    check(f"Near-identity init (diff={diff:.6f})", diff < 1e-4)

    # With zeros sharpness → should still be near-identity
    out_zero_sharp = net(g, torch.zeros(100))
    diff_zs = (out_zero_sharp - g).norm().item()
    check(f"Zero sharpness → near-identity (diff={diff_zs:.6f})", diff_zs < 1e-4)

    # Input dim is 2
    first_layer = net.net[0]
    check("First layer input_dim=2", first_layer.in_features == 2)
    check("First layer has hidden_dim outputs", first_layer.out_features == 32)

    # Gradient flow through meta-net
    g2 = torch.randn(50, requires_grad=False)
    s2 = torch.randn(50).abs()
    # Meta-net params should get gradients
    out2 = net(g2, s2)
    loss = out2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.norm() > 0 for p in net.parameters())
    check("Gradients flow through meta-net", has_grad)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: Progressive Weight Decay
# ═══════════════════════════════════════════════════════════════════════

def test_progressive_wd():
    print("\n" + "=" * 60)
    print("TEST 3: Progressive Weight Decay")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), weight_decay=1.0,
                      wd_ramp=4.0, wd_scale=20.0, wd_thresh=0.9)

    # Low accuracy → wd ≈ base
    opt._cached_train_acc = 0.5
    wd_low = opt._get_effective_wd(1.0)
    check(f"acc=0.5: wd_eff={wd_low:.3f} ≈ 1.0", abs(wd_low - 1.0) < 0.1,
          f"got {wd_low}")

    # At threshold → wd ≈ base * (1 + wd_ramp/2)
    opt._cached_train_acc = 0.9
    wd_mid = opt._get_effective_wd(1.0)
    expected_mid = 1.0 * (1.0 + 4.0 * 0.5)  # sigmoid(0) = 0.5
    check(f"acc=0.9: wd_eff={wd_mid:.3f} ≈ {expected_mid:.1f}",
          abs(wd_mid - expected_mid) < 0.1, f"got {wd_mid}")

    # High accuracy → wd ≈ base * (1 + wd_ramp)
    opt._cached_train_acc = 1.0
    wd_high = opt._get_effective_wd(1.0)
    check(f"acc=1.0: wd_eff={wd_high:.3f} ≈ 5.0", abs(wd_high - 5.0) < 0.1,
          f"got {wd_high}")

    # Monotonically increasing
    check("Monotonic: low < mid < high",
          wd_low < wd_mid < wd_high,
          f"{wd_low:.3f} < {wd_mid:.3f} < {wd_high:.3f}")

    # With different base wd
    wd_base2 = opt._get_effective_wd(2.0)
    check("Scales with base wd", wd_base2 > wd_high,
          f"base=2.0 → {wd_base2:.3f}")

    # Progressive wd during actual training step
    torch.manual_seed(42)
    m2 = nn.Linear(5, 1)
    opt2 = SuperGrok15(m2.parameters(), lr=1e-3, weight_decay=1.0,
                       wd_ramp=4.0, warmup_steps=0, warmup_ramp=1)
    x, y = torch.randn(8, 5), torch.randn(8, 1)
    crit = nn.MSELoss()

    # Step with low acc → gentle decay
    w_before = m2.weight.data.clone()
    opt2.zero_grad(); crit(m2(x), y).backward()
    opt2.step(train_loss=1.0, train_acc=0.3)
    w_after_low = m2.weight.data.clone()

    # Step with high acc → aggressive decay
    m3 = nn.Linear(5, 1)
    m3.load_state_dict(m2.state_dict())  # same starting point won't work, need fresh
    # Use a fresh optimizer to avoid state interaction
    torch.manual_seed(42)
    m3 = nn.Linear(5, 1)
    opt3 = SuperGrok15(m3.parameters(), lr=1e-3, weight_decay=1.0,
                       wd_ramp=4.0, warmup_steps=0, warmup_ramp=1)
    opt3.zero_grad(); crit(m3(x), y).backward()
    opt3.step(train_loss=0.001, train_acc=0.99)

    check("Progressive wd step completes", True)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: Basic Training Step (inherited v1.1 behavior)
# ═══════════════════════════════════════════════════════════════════════

def test_basic_step():
    print("\n" + "=" * 60)
    print("TEST 4: Basic Training Step")
    print("=" * 60)

    torch.manual_seed(42)
    m = make_model()
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=5, warmup_ramp=5)

    x, y = torch.randn(32, 10), torch.randn(32, 1)
    crit = nn.MSELoss()

    for step in range(20):
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        opt.step(train_loss=loss.item(), train_acc=min(step/20, 0.8))

    check("20 steps complete", opt.get_global_step() == 20)
    summary = opt.get_state_summary()
    check("Ramp factor=1.0 after warmup", abs(summary["ramp_factor"] - 1.0) < 1e-6)
    check("train_acc cached", summary["cached_train_acc"] > 0)
    check("effective_wd reported", summary["effective_wd"] > 0)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: Sharpness Cache (before sam_meta_step)
# ═══════════════════════════════════════════════════════════════════════

def test_sharpness_cache():
    print("\n" + "=" * 60)
    print("TEST 5: Sharpness Cache")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters())

    # Before any SAM step, sharpness should be zeros
    for p in m.parameters():
        s = opt._get_sharpness(p)
        check("Default sharpness is zeros", s.norm().item() == 0.0)
        break

    # After manually populating cache
    for p in m.parameters():
        opt._sharpness_cache[id(p)] = torch.ones_like(p.data) * 0.5
        s = opt._get_sharpness(p)
        check("Cached sharpness retrieved", s.norm().item() > 0)
        break


# ═══════════════════════════════════════════════════════════════════════
#  TEST 6: sam_meta_step
# ═══════════════════════════════════════════════════════════════════════

def test_sam_meta_step():
    print("\n" + "=" * 60)
    print("TEST 6: sam_meta_step (LookSAM + Bilevel)")
    print("=" * 60)

    torch.manual_seed(42)
    m = make_classifier()
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=0, warmup_ramp=1,
                      sam_rho=0.05, sam_freq=5)
    meta_opt = torch.optim.Adam(opt.meta_net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    x_train = torch.randn(32, 10); y_train = torch.randint(0, 5, (32,))
    x_val = torch.randn(16, 10); y_val = torch.randint(0, 5, (16,))

    # Save meta-net params before
    meta_before = {n: p.clone() for n, p in opt.meta_net.named_parameters()}

    # Forward + backward
    opt.zero_grad()
    loss = crit(m(x_train), y_train)
    loss.backward()

    # Save model params before
    model_before = {n: p.clone() for n, p in m.named_parameters()}

    # Combined SAM + bilevel step
    sam_loss, val_loss = opt.sam_meta_step(m, x_train, y_train, x_val, y_val, crit, meta_opt)

    check("Returns sam_loss (float)", isinstance(sam_loss, float))
    check("Returns val_loss (float)", isinstance(val_loss, float))
    check("sam_loss > 0", sam_loss > 0)
    check("val_loss > 0", val_loss > 0)

    # Sharpness cache should be populated
    has_sharpness = len(opt._sharpness_cache) > 0
    check("Sharpness cache populated", has_sharpness)

    # Check sharpness values are non-negative (we store abs)
    for pid, sharp in opt._sharpness_cache.items():
        check("Sharpness values ≥ 0", (sharp >= 0).all().item())
        break

    # Meta-net should have been updated
    meta_changed = False
    for n, p in opt.meta_net.named_parameters():
        if not torch.allclose(p, meta_before[n], atol=1e-8):
            meta_changed = True
            break
    check("Meta-net params updated", meta_changed)

    # Model params should be RESTORED (not perturbed)
    params_restored = True
    for n, p in m.named_parameters():
        if not torch.allclose(p.data, model_before[n], atol=1e-6):
            params_restored = False
            break
    check("Model params restored after SAM", params_restored)

    # Training gradients should be restored
    has_grads = any(p.grad is not None for p in m.parameters())
    check("Training gradients restored", has_grads)

    # Regular step should work after sam_meta_step
    opt.step(train_loss=loss.item(), train_acc=0.3)
    check("step() works after sam_meta_step", True)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 7: Sharpness flows into meta-net during step
# ═══════════════════════════════════════════════════════════════════════

def test_sharpness_in_step():
    print("\n" + "=" * 60)
    print("TEST 7: Sharpness Used During Step")
    print("=" * 60)

    torch.manual_seed(42)
    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=0, warmup_ramp=1,
                      sam_freq=5)

    x, y = torch.randn(8, 5), torch.randn(8, 1)
    crit = nn.MSELoss()

    # Step 1: no sharpness cached → uses zeros
    opt.zero_grad(); crit(m(x), y).backward()
    opt.step(train_loss=1.0, train_acc=0.3)
    w1 = m.weight.data.clone()

    # Manually inject sharpness cache
    for p in m.parameters():
        opt._sharpness_cache[id(p)] = torch.ones_like(p.data) * 10.0

    # Step 2: with sharpness → meta-net sees different input
    # Since rescale ≈ 0 at init, the difference should be tiny
    # but the code path should execute without error
    opt.zero_grad(); crit(m(x), y).backward()
    opt.step(train_loss=1.0, train_acc=0.3)
    check("Step with cached sharpness completes", True)

    summary = opt.get_state_summary()
    check("avg_sharpness_norm > 0 with cache",
          summary["avg_sharpness_norm"] > 0)
    check("sharpness_cached = True", summary["sharpness_cached"])


# ═══════════════════════════════════════════════════════════════════════
#  TEST 8: Smooth Warmup Ramp (inherited)
# ═══════════════════════════════════════════════════════════════════════

def test_warmup_ramp():
    print("\n" + "=" * 60)
    print("TEST 8: Smooth Warmup Ramp")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=10, warmup_ramp=10)
    x, y = torch.randn(4, 5), torch.randn(4, 1)
    crit = nn.MSELoss()

    ramps = []
    for step in range(25):
        opt.zero_grad(); crit(m(x), y).backward()
        opt.step(train_loss=0.5)
        ramps.append(opt._get_ramp_factor())

    check("Ramp=0 during warmup", all(r == 0.0 for r in ramps[:10]))
    ramp_phase = ramps[10:20]
    check("Ramp increases during ramp",
          all(ramp_phase[i] < ramp_phase[i+1] for i in range(len(ramp_phase)-1)))
    check("Ramp=1 after complete", abs(ramps[-1] - 1.0) < 1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 9: Memorization Fix (inherited)
# ═══════════════════════════════════════════════════════════════════════

def test_memorization_fix():
    print("\n" + "=" * 60)
    print("TEST 9: Memorization Fix")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), alpha_init=0.98, kappa=0.1,
                      alpha_update_freq=1)
    x, y = torch.randn(4, 5), torch.randn(4, 1)
    crit = nn.MSELoss()

    opt.zero_grad(); crit(m(x), y).backward()
    opt.step(train_loss=0.5)
    alpha_normal = opt.get_cached_alpha()
    check(f"Normal: alpha={alpha_normal:.4f} near init", alpha_normal > 0.9)

    opt.zero_grad(); crit(m(x), y).backward()
    opt.step(train_loss=0.01, train_acc=0.999)
    alpha_mem = opt.get_cached_alpha()
    check(f"Memorized: alpha={alpha_mem:.4f} < 0.5", alpha_mem < 0.5)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 10: Full Training Simulation
# ═══════════════════════════════════════════════════════════════════════

def test_full_training():
    print("\n" + "=" * 60)
    print("TEST 10: Full Training Simulation")
    print("=" * 60)

    torch.manual_seed(42)
    m = make_classifier()
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=10, warmup_ramp=10,
                      alpha_update_freq=5, weight_decay=1.0,
                      sam_freq=5, sam_rho=0.05,
                      wd_ramp=4.0, wd_thresh=0.9)
    meta_opt = torch.optim.Adam(opt.meta_net.parameters(), lr=1e-4)

    x_train = torch.randn(64, 10); y_train = torch.randint(0, 5, (64,))
    x_val = torch.randn(16, 10); y_val = torch.randint(0, 5, (16,))
    crit = nn.CrossEntropyLoss()

    alphas, wds = [], []
    for step in range(50):
        opt.zero_grad()
        loss = crit(m(x_train), y_train)
        loss.backward()

        acc = (m(x_train).argmax(-1) == y_train).float().mean().item()

        # Combined SAM + bilevel every sam_freq steps
        if step % opt.sam_freq == 0:
            try:
                opt.sam_meta_step(m, x_train, y_train, x_val, y_val, crit, meta_opt)
            except Exception as e:
                check(f"sam_meta_step at step {step}", False, str(e))

        # Provide val_loss periodically
        kw = {"train_loss": loss.item(), "train_acc": acc}
        if step % 5 == 0:
            with torch.no_grad():
                kw["val_loss"] = crit(m(x_val), y_val).item()
        opt.step(**kw)

        alphas.append(opt.get_cached_alpha())
        wds.append(opt.get_effective_wd())

    summary = opt.get_state_summary()
    check("50 steps complete", summary["global_step"] == 50)
    check("Ramp factor=1.0", abs(summary["ramp_factor"] - 1.0) < 1e-6)
    check("Sharpness cached", summary["sharpness_cached"])
    check("Alpha tracked", len(alphas) == 50)
    check("WD tracked", len(wds) == 50)

    print(f"  Final alpha: {alphas[-1]:.6f}")
    print(f"  Alpha range: [{min(alphas):.4f}, {max(alphas):.4f}]")
    print(f"  Final wd_eff: {wds[-1]:.3f}")
    print(f"  WD range: [{min(wds):.3f}, {max(wds):.3f}]")
    print(f"  Avg sharpness norm: {summary['avg_sharpness_norm']:.6f}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 11: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n" + "=" * 60)
    print("TEST 11: Edge Cases")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), lr=1e-3)

    # Step with no gradients
    opt.step(train_loss=0.5)
    check("Step with no grads OK", True)

    # Step with no signals
    opt.zero_grad()
    x, y = torch.randn(4, 5), torch.randn(4, 1)
    nn.MSELoss()(m(x), y).backward()
    opt.step()
    check("Step with no signals OK", True)

    # Repr
    r = repr(opt)
    check("Repr contains 'v1.5'", "v1.5" in r)
    check("Repr contains sam_rho", "sam_rho" in r)
    check("Repr contains wd_ramp", "wd_ramp" in r)

    # sam_meta_step with no gradients
    m2 = make_classifier()
    opt2 = SuperGrok15(m2.parameters())
    meta_opt2 = torch.optim.Adam(opt2.meta_net.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    x_t, y_t = torch.randn(8, 10), torch.randint(0, 5, (8,))
    x_v, y_v = torch.randn(4, 10), torch.randint(0, 5, (4,))
    # No backward called → no gradients
    sl, vl = opt2.sam_meta_step(m2, x_t, y_t, x_v, y_v, crit, meta_opt2)
    check("sam_meta_step with no grads returns (0, 0)",
          sl == 0.0 and vl == 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 12: Layer-wise β₁ and α (inherited)
# ═══════════════════════════════════════════════════════════════════════

def test_layer_wise():
    print("\n" + "=" * 60)
    print("TEST 12: Layer-wise β₁ and α")
    print("=" * 60)

    m = make_model()
    opt = SuperGrok15(m.parameters(), betas=(0.9, 0.999), gamma=0.1,
                      gamma_alpha=0.1, alpha_init=0.98)

    params = list(m.parameters())

    # β₁ decay
    for i, p in enumerate(params):
        b1 = opt._get_layer_beta1(p, 0.9, 0.1)
        expected = 0.9 * (0.9 ** i)
        check(f"Layer {i} β₁={b1:.4f} ≈ {expected:.4f}",
              abs(b1 - expected) < 1e-10)

    # α decay (early layers higher)
    alphas = [opt._get_layer_alpha(p, 0.98) for p in params]
    check("Early layers have higher α",
          alphas[0] >= alphas[-1],
          f"first={alphas[0]:.4f}, last={alphas[-1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 13: SAM Perturbation Correctness
# ═══════════════════════════════════════════════════════════════════════

def test_sam_correctness():
    print("\n" + "=" * 60)
    print("TEST 13: SAM Perturbation Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    m = nn.Linear(5, 3)
    opt = SuperGrok15(m.parameters(), lr=1e-3, sam_rho=0.1)
    meta_opt = torch.optim.Adam(opt.meta_net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    x_t = torch.randn(8, 5); y_t = torch.randint(0, 3, (8,))
    x_v = torch.randn(4, 5); y_v = torch.randint(0, 3, (4,))

    # Record params before
    w_before = m.weight.data.clone()
    b_before = m.bias.data.clone()

    # Forward + backward
    opt.zero_grad()
    loss = crit(m(x_t), y_t)
    loss.backward()

    # Do SAM step
    opt.sam_meta_step(m, x_t, y_t, x_v, y_v, crit, meta_opt)

    # Params must be exactly restored
    check("Weight restored exactly",
          torch.allclose(m.weight.data, w_before, atol=1e-6),
          f"max diff: {(m.weight.data - w_before).abs().max().item()}")
    check("Bias restored exactly",
          torch.allclose(m.bias.data, b_before, atol=1e-6))

    # Sharpness cache populated for both params
    check("Weight sharpness cached", id(m.weight) in opt._sharpness_cache)
    check("Bias sharpness cached", id(m.bias) in opt._sharpness_cache)

    # Sharpness should be non-negative
    w_sharp = opt._sharpness_cache[id(m.weight)]
    check("Weight sharpness ≥ 0", (w_sharp >= 0).all().item())
    check("Weight sharpness non-trivial", w_sharp.norm().item() > 1e-8)


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SUPERGROK v1.5 — TEST SUITE")
    print("=" * 60)

    test_construction()
    test_sharpness_meta_net()
    test_progressive_wd()
    test_basic_step()
    test_sharpness_cache()
    test_sam_meta_step()
    test_sharpness_in_step()
    test_warmup_ramp()
    test_memorization_fix()
    test_full_training()
    test_edge_cases()
    test_layer_wise()
    test_sam_correctness()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  ALL {total} CHECKS PASSED ✓")
    else:
        print(f"  {PASS}/{total} passed, {FAIL} FAILED ✗")
    print("=" * 60)
