"""#12 acceptance driver (Lane A, 2026-06-12) — NOT part of the repo product.

Proves, with the real compile.py internals on the real GPU:
  (1) a tuner variant module built via the JIT path (_torch_load) IMPORTS
      under its own name (the #12 defect),
  (2) ONE TimingWorker smoke trial times that variant (worker subprocess
      loads the variant .so as <pkg>._ops via the fixed _load_so),
  (3) the real megakernel TC train step is timed with a tuned tile flag
      (_time_megakernel_tc_step — the timer the GEMM-dim sweep path uses).
Raw JSON evidence printed to stdout.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import grokking_optimizers.compile as C

OUT = Path("build/compiled")
REPORT = OUT / "smoke12_driver_report.txt"


def main() -> int:
    out = {}
    spec = C.BuildSpec(optimizer="adamw", model="decoder", arch="sm_90",
                       out_dir=OUT)
    report = open(REPORT, "w")

    # --- (1) build one variant via the REAL JIT path ----------------------
    sources = C._resolve_sources(spec)
    host = C._host_cflags(spec)
    device = C._device_cflags(spec)
    ld = C._ldflags(spec)
    t0 = time.monotonic()
    so = C._torch_load(spec, sources, host, device, ld, report,
                       module_suffix="_smoke12")
    build_s = time.monotonic() - t0
    report.flush()
    out["build_wall_s"] = round(build_s, 1)
    out["variant_so"] = str(so)
    if so is None:
        print(json.dumps({"verdict": "FAIL", "stage": "build", **out}))
        return 1

    # --- (1b) import it by its own name (the defect's acceptance) ---------
    mod_name = Path(so).name.split(".", 1)[0]
    spec_i = importlib.util.spec_from_file_location(mod_name, str(so))
    mod = importlib.util.module_from_spec(spec_i)
    spec_i.loader.exec_module(mod)
    out["imported_as"] = mod_name
    out["abi_schema"] = int(getattr(mod, "__abi_schema__", -1))
    out["has_fused_step"] = hasattr(mod, "fused_step")
    out["detect_arch"] = int(mod.detect_arch())

    # --- (2) ONE TimingWorker smoke trial on the variant ------------------
    w = C.TimingWorker(opt_class="AdamW", size=1024, warmup=2, iters=7,
                       enable_watchdog=False)
    try:
        ok = w.start()
        out["worker_started"] = bool(ok)
        res = w.time(str(so)) if ok else None
        out["timingworker_trial"] = res
    finally:
        w.stop()

    # --- (3) real megakernel TC step with a tuned tile flag ---------------
    tc = C._time_megakernel_tc_step(
        "decoder", ["-DSG_TUNED_TILE_M=128"], OUT, warmup=3, iters=11,
        tc_batch=2048, report=report)
    out["megakernel_tc_trial"] = tc
    report.flush(); report.close()

    ok_all = (out["has_fused_step"] and out["detect_arch"] == 90
              and isinstance(out.get("timingworker_trial"), dict)
              and "timing_ms" in (out.get("timingworker_trial") or {})
              and isinstance(out.get("megakernel_tc_trial"), dict))
    out["verdict"] = "PASS" if ok_all else "FAIL"
    print(json.dumps(out, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
