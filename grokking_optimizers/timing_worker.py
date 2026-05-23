"""grokking_optimizers.timing_worker — persistent timing subprocess.

A single subprocess holds a warm CUDA / HIP context across an entire
autotune sweep, then takes one variant ``.so`` at a time on stdin and
reports a JSON timing result on stdout.

This avoids the ~1.5 s per-variant overhead of spawning a fresh
``python -c …``, importing torch, and initialising CUDA on every
config. The worker uses CUDA graphs for sm_90 timing (via
``bench_graph.cuda_graph_median_ms``) and falls back to per-event
timing on capture failure.

Wire protocol (line-delimited JSON):

    >>>  {"op": "time", "so_path": "/path/.so", "opt_class": "Lion"}
    <<<  {"timing_ms": 0.123, "min_ms": 0.118, "max_ms": 0.135, "n": 21}
    <<<  {"error": "...", "tb": "..."}

    >>>  {"op": "ping"}
    <<<  {"ok": true}

    >>>  {"op": "shutdown"}
    <<<  {"bye": true}    # then the subprocess exits

Usage in the parent process:

    worker = TimingWorker(opt_class="Lion")
    worker.start()
    try:
        for variant_so in variants:
            result = worker.time(variant_so)
            ...
    finally:
        worker.stop()

On worker crash (returncode != 0 mid-sweep), ``time()`` returns
``None`` and the caller is expected to either restart the worker
(``worker.restart()``) or fall back to per-subprocess timing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# The script that runs *inside* the worker subprocess
# ---------------------------------------------------------------------------

_WORKER_BODY = r"""
import sys, json, importlib.util, traceback, time
try:
    import torch
except ImportError as exc:
    sys.stdout.write(json.dumps({"error": "torch import failed: " + str(exc)}) + "\n")
    sys.stdout.flush()
    sys.exit(1)


def _load_so(so_path):
    # Load the per-variant .so as ``grokking_optimizers._ops`` and
    # invalidate any previously loaded one. Multiple .so files cannot
    # both register ops with the same name in a single process, so each
    # call must drop the previous module before importing the next.
    if "grokking_optimizers._ops" in sys.modules:
        del sys.modules["grokking_optimizers._ops"]
    spec = importlib.util.spec_from_file_location(
        "grokking_optimizers._ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["grokking_optimizers._ops"] = mod
    return mod


def _time_with_graph(opt_class, size, warmup, iters):
    # Best-effort import of bench_graph; fall back if not on sys.path
    try:
        from grokking_optimizers.bench_graph import cuda_graph_median_ms
    except Exception:
        return None
    try:
        return cuda_graph_median_ms(
            opt_class, size=size, warmup=warmup, iters=iters)
    except Exception:
        return None


def _time_with_events(opt_class, size, warmup, iters):
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(size, size, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        opt.step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return {
        "timing_ms": float(times[len(times) // 2]),
        "min_ms":    float(times[0]),
        "max_ms":    float(times[-1]),
        "n":         len(times),
    }


def main():
    # The first sanity-check is that CUDA is visible. Without it the
    # worker exits early; the parent will see the error and skip the
    # sweep.
    if not torch.cuda.is_available():
        sys.stdout.write(json.dumps(
            {"error": "torch.cuda.is_available() == False"}) + "\n")
        sys.stdout.flush()
        sys.exit(2)
    # Warm the device once up-front so the very first .time() doesn't
    # pay context-creation cost.
    torch.cuda.synchronize()
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    use_graph = True
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sys.stdout.write(json.dumps(
                {"error": "bad request: " + str(exc)}) + "\n")
            sys.stdout.flush()
            continue
        op = req.get("op")
        if op == "ping":
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
            sys.stdout.flush()
            continue
        if op == "shutdown":
            sys.stdout.write(json.dumps({"bye": True}) + "\n")
            sys.stdout.flush()
            return
        if op != "time":
            sys.stdout.write(json.dumps(
                {"error": "unknown op " + str(op)}) + "\n")
            sys.stdout.flush()
            continue
        try:
            so_path = req["so_path"]
            opt_class = req["opt_class"]
            size = int(req.get("size", 4096))
            warmup = int(req.get("warmup", 5))
            iters = int(req.get("iters", 21))
            _load_so(so_path)
            result = None
            if use_graph and req.get("use_cuda_graph", True):
                result = _time_with_graph(opt_class, size, warmup, iters)
            if result is None:
                result = _time_with_events(opt_class, size, warmup, iters)
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps(
                {"error": str(exc), "tb": traceback.format_exc()}) + "\n")
            sys.stdout.flush()


main()
"""


class TimingWorker:
    """Persistent timing subprocess; one warm CUDA context for an entire sweep."""

    def __init__(self, opt_class: str, *,
                 size: int = 4096, warmup: int = 5, iters: int = 21,
                 use_cuda_graph: bool = True,
                 timeout_per_variant: int = 180,
                 env: Optional[Dict[str, str]] = None,
                 cwd: Optional[Path] = None,
                 python: Optional[str] = None):
        self.opt_class = opt_class
        self.size = size
        self.warmup = warmup
        self.iters = iters
        self.use_cuda_graph = use_cuda_graph
        self.timeout = timeout_per_variant
        self.env = env or os.environ.copy()
        self.cwd = cwd
        self.python = python or sys.executable
        self._proc: Optional[subprocess.Popen] = None
        self._error_log: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Spawn the subprocess and wait for its ``{"ready": true}`` ack."""
        if self._proc is not None and self._proc.poll() is None:
            return True
        self._proc = subprocess.Popen(
            [self.python, "-u", "-c", _WORKER_BODY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env,
            cwd=str(self.cwd) if self.cwd else None,
        )
        # First line is either {"ready": true} or {"error": "..."}
        ready = self._read_line(timeout=30)
        if not ready or ready.get("ready") is not True:
            err = ready.get("error") if ready else "no response"
            self._error_log.append(("start", err))
            self.stop()
            return False
        return True

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin:
                try:
                    self._proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def time(self, variant_so: Path) -> Optional[Dict[str, Any]]:
        """Time a variant .so. Returns the result dict or ``None`` on failure."""
        if not self.alive():
            self._error_log.append(("time", "worker not alive"))
            return None
        req = {
            "op":             "time",
            "so_path":        str(variant_so),
            "opt_class":      self.opt_class,
            "size":           self.size,
            "warmup":         self.warmup,
            "iters":          self.iters,
            "use_cuda_graph": self.use_cuda_graph,
        }
        try:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._error_log.append(("write", str(exc)))
            return None
        result = self._read_line(timeout=self.timeout)
        if result is None:
            return None
        if "error" in result:
            self._error_log.append(("time", result.get("error", "?")))
            return None
        return result

    def ping(self) -> bool:
        if not self.alive():
            return False
        try:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(json.dumps({"op": "ping"}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        line = self._read_line(timeout=5)
        return bool(line and line.get("ok"))

    @property
    def error_log(self) -> list:
        return list(self._error_log)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_line(self, timeout: float) -> Optional[Dict[str, Any]]:
        assert self._proc and self._proc.stdout
        # subprocess does not have a portable per-line timeout; use a
        # busy-wait poll. This is acceptable because timings dominate.
        deadline = _time.monotonic() + timeout
        line = ""
        while _time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # Worker died — drain any final output
                tail = self._proc.stdout.read() or ""
                if tail:
                    self._error_log.append(("died", tail.strip()[-2000:]))
                return None
            # Read one byte at a time so we can honour the timeout.
            ch = self._proc.stdout.read(1)
            if not ch:
                _time.sleep(0.01)
                continue
            if ch == "\n":
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    self._error_log.append(("decode", line[:500]))
                    return None
            line += ch
        # Timeout
        self._error_log.append(("timeout", f"after {timeout}s"))
        return None


__all__ = ["TimingWorker"]
