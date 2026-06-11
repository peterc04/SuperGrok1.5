"""Batched decoder split-K production validation (after _ops rebuild).

Runs in ONE process on a quiet GPU:
  (1) wiring_check --models decoder  → every decoder cell still executes wgmma L3-TC
  (2) tail_gate for adamw/lion/grokfast/decoder → param+state parity vs real eager
  (3) decoder TC TF/s re-measure (B=16384, ncta=0) → the after-fix number

Run: CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python tuning/_decoder_validate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 70, flush=True)
    print("[decoder-validate] (1) wiring_check --models decoder", flush=True)
    print("=" * 70, flush=True)
    import wiring_check as wc
    rows = []
    for opt in wc.OPTS:
        r = wc.check_cell(opt, "decoder", steps=4)
        rows.append(r)
        mark = "PASS L3-TC" if r["is_l3_tc"] else "BLOCKED   "
        print(f"  [{mark}] {opt:11s}/decoder  path={r['executed_path']:26s}", flush=True)
    conv = [r for r in rows if r["is_l3_tc"]]
    print(f"  => decoder L3-TC: {len(conv)}/{len(rows)} "
          f"(converted: {[r['optimizer'] for r in conv]})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("[decoder-validate] (2) tail_gate adamw/lion/grokfast/decoder", flush=True)
    print("=" * 70, flush=True)
    from tests.hw.test_l3tc_tail_gate import run_cell_gate
    tg = {}
    for cell in ("adamw/decoder", "lion/decoder", "grokfast/decoder"):
        try:
            ok, detail = run_cell_gate(cell, verbose=True)
            tg[cell] = ok
            print(f"  => {cell}: {'PASS' if ok else 'FAIL'}\n", flush=True)
        except Exception as e:  # noqa: BLE001
            tg[cell] = False
            import traceback; traceback.print_exc()
            print(f"  => {cell}: ERROR {e}\n", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("[decoder-validate] (3) decoder TC TF/s re-measure (B=16384, ncta=0)", flush=True)
    print("=" * 70, flush=True)
    from tuning.roofline import measure_tc_cell, _data_init
    import json
    ef, eb = 4.23e10, 16384   # fallback FLOP basis (same as baseline for relative cmp)
    try:
        d = json.load(open("results/h100_grokking_race/roofline.json"))
        row = next((r for r in d["rows"] if r["optimizer"] == "adamw"
                    and r["model"] == "decoder" and r.get("path_family") == "scalar"), None)
        if row and row.get("flops_per_step"):
            ef, eb = row["flops_per_step"], row["batch"]
    except Exception:
        pass
    npar = _data_init("decoder")[2]
    trows = measure_tc_cell("decoder", ef, eb, npar, 4, 128, 2)
    for r in trows:
        print(f"  DECODER {r['path_family']:11s}: {r['wall_per_step_ms']:.3f} ms  "
              f"{r['achieved_flops_per_s']/1e12:.3f} TF/s  roof%={100*r['fraction_of_roofline']:.2f}",
              flush=True)

    print("\n[decoder-validate] SUMMARY:", flush=True)
    print(f"  wiring_check L3-TC: {len(conv)}/{len(rows)}", flush=True)
    print(f"  tail_gate: {tg}", flush=True)
    allok = len(conv) >= 3 and all(tg.values())
    print(f"  => {'ALL GREEN' if allok else 'NEEDS ATTENTION'}", flush=True)


if __name__ == "__main__":
    main()
