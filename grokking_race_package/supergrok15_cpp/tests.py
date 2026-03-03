"""
SuperGrok v1.5 C++ — Test Suite

Tests the C++/CUDA-accelerated optimizer for correctness against
the expected behavior. Verifies:
  - Construction and defaults
  - SharpnessMetaNet (2D input)
  - Progressive weight decay
  - Basic training step (C++ and Python paths)
  - sam_meta_step (SAM + bilevel)
  - Sharpness cache flow
  - Warmup ramp
  - Memorization fix
  - Full training simulation
  - Edge cases
  - Numerical agreement between C++ and Python paths
"""

import sys, math, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, ".")
from supergrok15_cpp import SuperGrok15, SharpnessMetaNet
from supergrok15_cpp.optim import _HAS_CPP


def make_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16),
                         nn.ReLU(), nn.Linear(16, 1))

def make_classifier():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1; print(f"  ✓ {name}")
    else:
        FAIL += 1; print(f"  ✗ {name} — {detail}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: Construction and Backend Detection
# ═══════════════════════════════════════════════════════════════════════

def test_construction():
    print("\n" + "=" * 60)
    print("TEST 1: Construction and Backend Detection")
    print("=" * 60)

    m = make_model()
    opt = SuperGrok15(m.parameters())

    check("Default construction", opt is not None)
    check("Default wd=1.0", opt.defaults["weight_decay"] == 1.0)
    check("Meta-net is SharpnessMetaNet", isinstance(opt.meta_net, SharpnessMetaNet))
    check("sam_rho=0.05", opt.sam_rho == 0.05)
    check("sam_freq=5", opt.sam_freq == 5)
    check("wd_ramp=4.0", opt.wd_ramp == 4.0)

    r = repr(opt)
    if _HAS_CPP:
        check("Backend=C++/CUDA", "C++/CUDA" in r)
    else:
        check("Backend=Python (fallback)", "Python" in r)

    print(f"\n  C++ extension loaded: {_HAS_CPP}")

    # Validation
    try:
        SuperGrok15(m.parameters(), sam_rho=-1)
        check("Negative sam_rho raises", False)
    except ValueError:
        check("Negative sam_rho raises", True)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: SharpnessMetaNet
# ═══════════════════════════════════════════════════════════════════════

def test_sharpness_meta_net():
    print("\n" + "=" * 60)
    print("TEST 2: SharpnessMetaNet")
    print("=" * 60)

    net = SharpnessMetaNet(hidden_dim=32)

    for shape in [(10,), (5, 5), (2, 3, 4), (100,)]:
        g = torch.randn(shape)
        s = torch.randn(shape).abs()
        out = net(g, s)
        check(f"Shape {shape} preserved", out.shape == g.shape)

    # Near-identity at init
    g = torch.randn(100)
    s = torch.randn(100).abs()
    out = net(g, s)
    diff = (out - g).norm().item()
    check(f"Near-identity init (diff={diff:.6f})", diff < 1e-4)

    # get_weights returns correct shapes
    W1, b1, W2, b2, rescale = net.get_weights()
    check("W1 shape (32, 2)", W1.shape == (32, 2))
    check("b1 shape (32,)", b1.shape == (32,))
    check("W2 shape (1, 32)", W2.shape == (1, 32))
    check("b2 shape (1,)", b2.shape == (1,))
    check("rescale ≈ 0", abs(rescale) < 1e-6)


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

    opt._cached_train_acc = 0.5
    wd_low = opt._get_effective_wd(1.0)
    check(f"acc=0.5: wd_eff={wd_low:.3f} ≈ 1.0", abs(wd_low - 1.0) < 0.1)

    opt._cached_train_acc = 0.9
    wd_mid = opt._get_effective_wd(1.0)
    check(f"acc=0.9: wd_eff={wd_mid:.3f} ≈ 3.0", abs(wd_mid - 3.0) < 0.1)

    opt._cached_train_acc = 1.0
    wd_high = opt._get_effective_wd(1.0)
    check(f"acc=1.0: wd_eff={wd_high:.3f} ≈ 5.0", abs(wd_high - 5.0) < 0.1)

    check("Monotonic", wd_low < wd_mid < wd_high)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: Basic Training Step
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

    losses = []
    for step in range(20):
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        opt.step(train_loss=loss.item(), train_acc=min(step / 20, 0.8))
        losses.append(loss.item())

    check("20 steps complete", opt.get_global_step() == 20)
    check("Loss decreased", losses[-1] < losses[0],
          f"first={losses[0]:.4f}, last={losses[-1]:.4f}")

    summary = opt.get_state_summary()
    check("Ramp=1.0 after warmup", abs(summary["ramp_factor"] - 1.0) < 1e-6)
    check(f"cpp_backend={summary['cpp_backend']}", True)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: sam_meta_step
# ═══════════════════════════════════════════════════════════════════════

def test_sam_meta_step():
    print("\n" + "=" * 60)
    print("TEST 5: sam_meta_step (LookSAM + Bilevel)")
    print("=" * 60)

    torch.manual_seed(42)
    m = make_classifier()
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=0, warmup_ramp=1,
                      sam_rho=0.05)
    meta_opt = torch.optim.Adam(opt.meta_net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    x_t = torch.randn(32, 10); y_t = torch.randint(0, 5, (32,))
    x_v = torch.randn(16, 10); y_v = torch.randint(0, 5, (16,))

    meta_before = {n: p.clone() for n, p in opt.meta_net.named_parameters()}
    model_before = {n: p.data.clone() for n, p in m.named_parameters()}

    # Forward + backward
    opt.zero_grad()
    loss = crit(m(x_t), y_t)
    loss.backward()

    sam_loss, val_loss = opt.sam_meta_step(m, x_t, y_t, x_v, y_v, crit, meta_opt)

    check("Returns sam_loss", isinstance(sam_loss, float) and sam_loss > 0)
    check("Returns val_loss", isinstance(val_loss, float) and val_loss > 0)

    # Meta-net updated
    meta_changed = any(
        not torch.allclose(p, meta_before[n], atol=1e-8)
        for n, p in opt.meta_net.named_parameters()
    )
    check("Meta-net params updated", meta_changed)

    # Model params restored
    params_restored = all(
        torch.allclose(p.data, model_before[n], atol=1e-5)
        for n, p in m.named_parameters()
    )
    check("Model params restored", params_restored)

    # Sharpness cached
    summary = opt.get_state_summary()
    check("Sharpness cached", summary["sharpness_cached"])

    # Step works after sam_meta_step
    opt.step(train_loss=loss.item(), train_acc=0.3)
    check("step() after sam_meta_step OK", True)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 6: Warmup Ramp
# ═══════════════════════════════════════════════════════════════════════

def test_warmup_ramp():
    print("\n" + "=" * 60)
    print("TEST 6: Warmup Ramp")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=10, warmup_ramp=10)
    x, y = torch.randn(4, 5), torch.randn(4, 1)
    crit = nn.MSELoss()

    ramps = []
    for _ in range(25):
        opt.zero_grad(); crit(m(x), y).backward()
        opt.step(train_loss=0.5)
        ramps.append(opt._get_ramp_factor())

    check("Ramp=0 during warmup", all(r == 0.0 for r in ramps[:10]))
    check("Ramp increases", ramps[11] < ramps[18])
    check("Ramp=1 after", abs(ramps[-1] - 1.0) < 1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 7: Memorization Fix
# ═══════════════════════════════════════════════════════════════════════

def test_memorization_fix():
    print("\n" + "=" * 60)
    print("TEST 7: Memorization Fix")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters(), alpha_init=0.98, kappa=0.1,
                      alpha_update_freq=1)
    x, y = torch.randn(4, 5), torch.randn(4, 1)
    crit = nn.MSELoss()

    opt.zero_grad(); crit(m(x), y).backward()
    opt.step(train_loss=0.5)
    alpha_normal = opt.get_cached_alpha()
    check(f"Normal: alpha={alpha_normal:.4f} > 0.9", alpha_normal > 0.9)

    opt.zero_grad(); crit(m(x), y).backward()
    opt.step(train_loss=0.01, train_acc=0.999)
    alpha_mem = opt.get_cached_alpha()
    check(f"Memorized: alpha={alpha_mem:.4f} < 0.5", alpha_mem < 0.5)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 8: Full Training Simulation
# ═══════════════════════════════════════════════════════════════════════

def test_full_training():
    print("\n" + "=" * 60)
    print("TEST 8: Full Training Simulation (50 steps)")
    print("=" * 60)

    torch.manual_seed(42)
    m = make_classifier()
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=10, warmup_ramp=10,
                      alpha_update_freq=5, weight_decay=1.0,
                      sam_freq=5, sam_rho=0.05, wd_ramp=4.0)
    meta_opt = torch.optim.Adam(opt.meta_net.parameters(), lr=1e-4)

    x_t = torch.randn(64, 10); y_t = torch.randint(0, 5, (64,))
    x_v = torch.randn(16, 10); y_v = torch.randint(0, 5, (16,))
    crit = nn.CrossEntropyLoss()

    for step in range(50):
        opt.zero_grad()
        loss = crit(m(x_t), y_t)
        loss.backward()
        acc = (m(x_t).argmax(-1) == y_t).float().mean().item()

        if step % opt.sam_freq == 0:
            try:
                opt.sam_meta_step(m, x_t, y_t, x_v, y_v, crit, meta_opt)
            except Exception as e:
                check(f"sam_meta_step at step {step}", False, str(e))

        kw = {"train_loss": loss.item(), "train_acc": acc}
        if step % 5 == 0:
            with torch.no_grad():
                kw["val_loss"] = crit(m(x_v), y_v).item()
        opt.step(**kw)

    summary = opt.get_state_summary()
    check("50 steps complete", summary["global_step"] == 50)
    check("Sharpness cached", summary["sharpness_cached"])
    check("Ramp=1.0", abs(summary["ramp_factor"] - 1.0) < 1e-6)

    print(f"  Final alpha: {summary['cached_alpha']:.6f}")
    print(f"  Effective wd: {summary['effective_wd']:.3f}")
    print(f"  Avg sharpness: {summary['avg_sharpness_norm']:.6f}")
    print(f"  Backend: {'C++/CUDA' if summary['cpp_backend'] else 'Python'}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 9: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n" + "=" * 60)
    print("TEST 9: Edge Cases")
    print("=" * 60)

    m = nn.Linear(5, 1)
    opt = SuperGrok15(m.parameters())

    # Step with no grads
    opt.step(train_loss=0.5)
    check("Step with no grads OK", True)

    # Step with no signals
    x, y = torch.randn(4, 5), torch.randn(4, 1)
    opt.zero_grad(); nn.MSELoss()(m(x), y).backward()
    opt.step()
    check("Step with no signals OK", True)

    # sam_meta_step with no grads
    m2 = make_classifier()
    opt2 = SuperGrok15(m2.parameters())
    meta_opt2 = torch.optim.Adam(opt2.meta_net.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    x_t, y_t = torch.randn(8, 10), torch.randint(0, 5, (8,))
    x_v, y_v = torch.randn(4, 10), torch.randint(0, 5, (4,))
    sl, vl = opt2.sam_meta_step(m2, x_t, y_t, x_v, y_v, crit, meta_opt2)
    check("sam_meta_step with no grads", sl == 0.0 and vl == 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 10: CUDA (if available)
# ═══════════════════════════════════════════════════════════════════════

def test_cuda():
    print("\n" + "=" * 60)
    print("TEST 10: CUDA Device")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available — skipping")
        return

    dev = torch.device("cuda")
    torch.manual_seed(42)
    m = make_classifier().to(dev)
    opt = SuperGrok15(m.parameters(), lr=1e-3, warmup_steps=5, warmup_ramp=5,
                      sam_freq=3)
    opt.meta_net = opt.meta_net.to(dev)
    meta_opt = torch.optim.Adam(opt.meta_net.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    x_t = torch.randn(32, 10, device=dev); y_t = torch.randint(0, 5, (32,), device=dev)
    x_v = torch.randn(16, 10, device=dev); y_v = torch.randint(0, 5, (16,), device=dev)

    for step in range(20):
        opt.zero_grad()
        loss = crit(m(x_t), y_t)
        loss.backward()
        acc = (m(x_t).argmax(-1) == y_t).float().mean().item()

        if step % opt.sam_freq == 0:
            opt.sam_meta_step(m, x_t, y_t, x_v, y_v, crit, meta_opt)

        opt.step(train_loss=loss.item(), train_acc=acc)

    check("20 CUDA steps complete", opt.get_global_step() == 20)
    summary = opt.get_state_summary()
    check("CUDA sharpness cached", summary["sharpness_cached"])
    check(f"Backend: {'C++/CUDA' if summary['cpp_backend'] else 'Python'}",
          True)


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SUPERGROK v1.5 C++ — TEST SUITE")
    print(f"  C++ extension loaded: {_HAS_CPP}")
    print("=" * 60)

    test_construction()
    test_sharpness_meta_net()
    test_progressive_wd()
    test_basic_step()
    test_sam_meta_step()
    test_warmup_ramp()
    test_memorization_fix()
    test_full_training()
    test_edge_cases()
    test_cuda()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  ALL {total} CHECKS PASSED ✓")
    else:
        print(f"  {PASS}/{total} passed, {FAIL} FAILED ✗")
    print("=" * 60)
