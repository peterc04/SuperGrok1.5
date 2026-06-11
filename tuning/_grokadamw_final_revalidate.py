"""Final re-validation of decoder×grokadamw after the .so re-sync rebuild.

Runs in ONE process on a quiet GPU:
  (1) single-step state-aware tail_gate for grokadamw/decoder + the 3 prior cells
      (adamw/lion/grokfast) as the regression guard for the shared TC kernel.
  (2) the multi-step parity + race-path α verifier (re-run twice to OBSERVE the
      P2.5 clipped-regime reduction is bit-deterministic — the determinism that
      A/A/A at step 1 cannot see because the clip is inert there).

Run: CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python \
        tuning/_grokadamw_final_revalidate.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from tests.hw.test_l3tc_tail_gate import run_cell_gate

    print("=" * 64, flush=True)
    print("[final] (1) single-step gates (grokadamw + 3 prior regression guards)",
          flush=True)
    print("=" * 64, flush=True)
    gate_ok = True
    for cell in ("grokadamw/decoder", "adamw/decoder", "lion/decoder",
                 "grokfast/decoder"):
        ok, d = run_cell_gate(cell, verbose=False)
        gate_ok = gate_ok and ok
        print(f"  {cell:20s}: {'PASS' if ok else 'FAIL'}  "
              f"rel={d['rel_err']:.2e} det={d['det']}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("[final] (2) multi-step parity + race-path α (run A, then run B for "
          "P2.5 determinism)", flush=True)
    print("=" * 64, flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    parity = os.path.join(here, "_grokadamw_multistep_parity.py")
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="/nonexistent",
               PYTHONPATH=os.path.dirname(here))

    def _run_parity(tag):
        r = subprocess.run([sys.executable, parity], capture_output=True,
                           text=True, env=env)
        worst = [ln for ln in r.stdout.splitlines()
                 if "kernel-vs-eager" in ln or "RACE-PATH (" in ln
                 or "ALL GREEN — conversion faithful" in ln]
        print(f"  --- parity run {tag} (rc={r.returncode}) ---", flush=True)
        for ln in worst:
            print("   " + ln.strip(), flush=True)
        return r.returncode == 0, "\n".join(
            ln for ln in r.stdout.splitlines() if "param=" in ln and "WORST" not in ln)

    okA, worstA = _run_parity("A")
    okB, worstB = _run_parity("B")
    det_p2_5 = (worstA == worstB)   # bit-identical worst-errors ⇒ P2.5 deterministic
    print(f"\n  P2.5 clipped-regime reduction bit-deterministic (A==B): "
          f"{'YES' if det_p2_5 else 'NO'}", flush=True)

    all_ok = gate_ok and okA and okB and det_p2_5
    print(f"\n[final] => {'ALL GREEN — grokadamw/decoder conversion fully re-validated' if all_ok else 'NEEDS ATTENTION'}",
          flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
