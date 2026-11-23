"""Python twin of kernels/harness/harness.h, for the PyTorch baselines (B1, B2).

Same protocol and the same result schema ("airgap-bench/1", kernels/harness/SCHEMA.md): groups run one
after another; per group: setup, check before any timing, warm-up until the SM clock is steady,
iteration counts calibrated once and frozen, SAMPLES samples per op in a seeded shuffle per round, NVML
read after each sample, the same throttle flags, plausibility guard and failure handling.

Differences from the C++ harness, all recorded in the file:
- event_batch_ms is CUDA-event time over a Python loop of calls, so for short kernels it includes the
  gaps Python dispatch leaves between launches; that is part of what a PyTorch baseline costs.
- mem_peak_mib: the op's own setup memory plus the peak PyTorch allocates on top during one run
  (torch's caching allocator hides frees from cudaMemGetInfo, and library attention allocates its N x N
  inside the call).
- the shuffle order comes from Python's random.Random(SHUFFLE_SEED), not std::mt19937.
- an out-of-memory error leaves the context usable (PyTorch's allocator raises before the driver does);
  other CUDA errors are treated as a lost context, as in C++.
C++ and Python ops cannot be interleaved in one process, so comparisons across the two are between
separate runs.
"""

import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pynvml
import torch

SAMPLES = 21
TARGET_SAMPLE_MS = 20.0
WARMUP_MIN_S, WARMUP_MAX_S = 1.0, 10.0
CLOCK_TOL = 0.01
THROTTLE_DROP = 0.03
RATE_SLACK = 1.05
PHASE_ITERS_MAX = 200
SHUFFLE_SEED = 20261123

_REASONS = [
    (0x1, "idle"), (0x2, "app_clocks"), (0x4, "sw_power_cap"), (0x8, "hw_slowdown"), (0x10, "sync_boost"),
    (0x20, "sw_thermal"), (0x40, "hw_thermal"), (0x80, "hw_power_brake"), (0x100, "display_clock"),
    (0x200, "board_limit"), (0x400, "reliability_voltage"),
]
THROTTLE_MASK = 0x20 | 0x40 | 0x8 | 0x80
POWER_CAP = 0x4


def reason_names(r: int) -> list[str]:
    return [name for bit, name in _REASONS if r & bit]


def read_limits(path: str | Path) -> dict:
    try:
        pk = json.loads(Path(path).read_text())
        return {"tflops": pk["fma_sustained"]["tflops_settled"], "dram_gb_per_s": pk["dram_read"]["gb_per_s"]}
    except (OSError, KeyError, ValueError):
        return {"tflops": 0.0, "dram_gb_per_s": 0.0}


@dataclass
class Op:
    name: str
    run: Callable[[], None]
    group: str = "default"
    params: dict = field(default_factory=dict)
    flops: float = 0.0
    dram_bytes: float = 0.0
    setup: Callable[[], None] | None = None
    teardown: Callable[[], None] | None = None
    phases: list = field(default_factory=list)   # [(name, callable)], together equal to run
    check: Callable[[], dict] | None = None      # returns {"ok", "max_abs_err", "max_rel_err", "note"}


@dataclass
class Sample:
    round: int
    prev: str
    event_batch_ms: float = math.nan
    host_sync_ms: float = math.nan
    phase_ms: dict = field(default_factory=dict)
    sm: int = 0
    mem: int = 0
    temp: int = 0
    reasons: int = 0
    flagged: bool = False
    flag_reason: str = ""


def stats_of(v: list[float]) -> dict:
    v = sorted(x for x in v if not math.isnan(x))
    if not v:
        return {"n": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    q = lambda p: v[round(p * (len(v) - 1))]  # noqa: E731
    return {"n": len(v), "median": q(0.5), "p10": q(0.1), "p90": q(0.9), "min": v[0], "max": v[-1]}


def flag_samples(samples: list[Sample]) -> int:
    clocks = [s.sm for s in samples]
    med = stats_of(clocks)["median"]
    n = 0
    for s in samples:
        s.flagged, s.flag_reason = False, ""
        if s.reasons & THROTTLE_MASK:
            s.flagged, s.flag_reason = True, "thermal or hardware slowdown"
        elif med is not None and s.sm < (1 - THROTTLE_DROP) * med and not (s.reasons & POWER_CAP):
            s.flagged, s.flag_reason = True, "SM clock below the op's median, not explained by the power cap"
        n += s.flagged
    return n


class State:
    def __init__(self, op: Op):
        self.op = op
        self.status, self.stage, self.error = "pending", "", ""
        self.check: dict | None = None
        self.mem_mib = math.nan
        self.mem_peak_mib = math.nan
        self.iters = self.host_iters = self.phase_iters = 0
        self.samples: list[Sample] = []
        self.clocks: dict = {}
        self.tflops = self.dram_gb_per_s = self.clock_time_corr = math.nan
        self.plausibility: list[str] = []


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError)


class Runner:
    def __init__(self, benchmark: str, limits: dict):
        self.bench = benchmark
        self.limits = limits
        self.ops: list[State] = []
        self.groups_json: list[dict] = []
        self.warnings: list[str] = []
        self.dead_group = ""
        if not (limits.get("tflops", 0) > 0 and limits.get("dram_gb_per_s", 0) > 0):
            self.warnings.append("measured limits missing (results/cuda/peak.json not found or incomplete): "
                                 "the plausibility guard is off")
            print("harness warning:", self.warnings[-1])
        pynvml.nvmlInit()
        self.dev = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.stream = torch.cuda.Stream()
        self.created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.env = self._env()

    # environment and NVML
    def _env(self) -> dict:
        p = torch.cuda.get_device_properties(0)
        procs = []
        for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                for info in fn(self.dev):
                    if info.pid == os.getpid():
                        continue
                    try:
                        name = Path(pynvml.nvmlSystemGetProcessName(info.pid)).name
                    except pynvml.NVMLError:
                        name = str(info.pid)
                    if name not in procs:
                        procs.append(name)
            except pynvml.NVMLError:
                pass
        ac = "unknown"
        for d in Path("/sys/class/power_supply").glob("*"):
            try:
                if (d / "type").read_text().strip() == "Mains":
                    ac = "on" if (d / "online").read_text().strip() == "1" else "off"
            except OSError:
                pass
        try:
            profile = Path("/sys/firmware/acpi/platform_profile").read_text().strip()
        except OSError:
            profile = ""
        return {"gpu": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
                "memory_mib": p.total_memory >> 20,
                "max_sm_mhz": pynvml.nvmlDeviceGetMaxClockInfo(self.dev, pynvml.NVML_CLOCK_SM),
                "driver": pynvml.nvmlSystemGetDriverVersion(), "cuda_runtime": torch.version.cuda,
                "cuda_driver_api": None, "nvcc": None, "host_compiler": None,
                "python": platform.python_version(), "torch": torch.__version__,
                "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32,
                         "float32_matmul_precision": torch.get_float32_matmul_precision()},
                "ac": ac, "platform_profile": profile, "other_gpu_processes": procs}

    def _snap(self) -> tuple[int, int, int, int]:
        sm = pynvml.nvmlDeviceGetClockInfo(self.dev, pynvml.NVML_CLOCK_SM)
        mem = pynvml.nvmlDeviceGetClockInfo(self.dev, pynvml.NVML_CLOCK_MEM)
        temp = pynvml.nvmlDeviceGetTemperature(self.dev, pynvml.NVML_TEMPERATURE_GPU)
        reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.dev)
        return sm, mem, temp, reasons

    # timing primitives (all on self.stream)
    def _event_batch(self, f, iters: int) -> float:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            a.record()
            for _ in range(iters):
                f()
            b.record()
        b.synchronize()
        return a.elapsed_time(b) / iters

    def _host_sync(self, f, iters: int) -> float:
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            for _ in range(iters):
                f()
                self.stream.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters

    def _phase_times(self, op: Op, iters: int) -> dict:
        acc = [0.0] * len(op.phases)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(op.phases) + 1)]
        with torch.cuda.stream(self.stream):
            for _ in range(iters):
                ev[0].record()
                for k, (_, f) in enumerate(op.phases):
                    f()
                    ev[k + 1].record()
                ev[-1].synchronize()
                for k in range(len(op.phases)):
                    acc[k] += ev[k].elapsed_time(ev[k + 1])
        return {name: acc[k] / iters for k, (name, _) in enumerate(op.phases)}

    # running
    def add(self, op: Op) -> None:
        self.ops.append(State(op))

    def run_all(self) -> int:
        groups = []
        for s in self.ops:
            if s.op.group not in groups:
                groups.append(s.op.group)
        for g in groups:
            if self.dead_group:
                for s in self.ops:
                    if s.op.group == g:
                        s.status = "skipped"
                        s.error = (f"CUDA context lost in an earlier group ({self.dead_group}); "
                                   "this process cannot recover it: rerun in a new process")
                continue
            self._run_group(g)
        return sum(s.status != "ok" for s in self.ops)

    def _fail(self, s: State, stage: str, e: BaseException) -> bool:
        """Marks s failed; returns True if the context is lost."""
        s.status, s.stage, s.error = "failed", stage, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        if _is_oom(e):
            torch.cuda.empty_cache()
            return False
        try:
            torch.cuda.synchronize()
            return False
        except Exception:  # noqa: BLE001
            return True

    def _run_group(self, g: str) -> None:
        grp = [s for s in self.ops if s.op.group == g]
        lost = False
        free0, total = torch.cuda.mem_get_info()
        used_before = (total - free0) / 2**20

        for s in grp:  # 1. setup (peak memory measured from here through one run)
            if lost:
                s.status, s.error = "skipped", "context lost earlier in the group"
                continue
            try:
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                if s.op.setup:
                    s.op.setup()
                torch.cuda.synchronize()
                s.mem_mib = (torch.cuda.memory_allocated() - base) / 2**20
                s.status = "ok"
            except Exception as e:  # noqa: BLE001
                lost = self._fail(s, "setup", e) or lost
        free1, _ = torch.cuda.mem_get_info()
        used_after = (total - free1) / 2**20

        for s in grp:  # 2. check, then one run for the peak
            if s.status != "ok" or lost:
                continue
            try:
                if s.op.check:
                    s.check = s.op.check()
                    if not s.check.get("ok", False):
                        s.status = "check_failed"
                        continue
                # peak = this op's own setup memory + what one run allocates on top (other ops' inputs,
                # set up earlier or later in the group, are not counted)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with torch.cuda.stream(self.stream):
                    s.op.run()
                self.stream.synchronize()
                s.mem_peak_mib = s.mem_mib + (torch.cuda.max_memory_allocated() - before) / 2**20
            except Exception as e:  # noqa: BLE001
                lost = self._fail(s, "check", e) or lost
        if lost:
            for s in grp:
                if s.status == "ok":
                    s.status, s.error = "skipped", "context lost during setup or check"
        active = [s for s in grp if s.status == "ok"]

        def drop(s: State, stage: str, e: BaseException) -> None:
            nonlocal lost
            if self._fail(s, stage, e):
                lost = True
                for o in grp:
                    if o is not s and o.status in ("ok", "pending"):
                        o.status, o.error = "skipped", f"context lost after {s.op.name} failed in {stage}"

        # 3. warm-up until the SM clock is steady
        warm_clocks, warm_s, t0 = [], 0.0, time.perf_counter()
        while active:
            for s in list(active):
                try:
                    t = self._event_batch(s.op.run, 1)
                    self._event_batch(s.op.run, max(1, min(100000, int(50.0 / max(t, 1e-4)))))
                except Exception as e:  # noqa: BLE001
                    drop(s, "warmup", e)
                    if lost:
                        break
            active = [s for s in active if s.status == "ok"]
            if not active:
                break
            warm_clocks.append(self._snap()[0])
            warm_s = time.perf_counter() - t0
            c = warm_clocks
            steady = len(c) >= 3 and abs(c[-1] - c[-3]) <= CLOCK_TOL * c[-1] and abs(c[-2] - c[-3]) <= CLOCK_TOL * c[-1]
            if (warm_s >= WARMUP_MIN_S and steady) or warm_s >= WARMUP_MAX_S:
                break

        # 4. calibrate, then freeze
        for s in list(active):
            try:
                t = self._event_batch(s.op.run, 1)
                probe = max(1, int(2.0 / max(t, 1e-4)))
                t = self._event_batch(s.op.run, probe)
                s.iters = min(2_000_000, max(1, math.ceil(TARGET_SAMPLE_MS / max(t, 1e-5))))
                h = self._host_sync(s.op.run, min(probe, 200))
                s.host_iters = min(200_000, max(1, math.ceil(TARGET_SAMPLE_MS / 2 / max(h, 1e-4))))
                s.phase_iters = min(s.iters, PHASE_ITERS_MAX) if s.op.phases else 0
            except Exception as e:  # noqa: BLE001
                drop(s, "calibrate", e)
                if lost:
                    break
        active = [s for s in active if s.status == "ok"]

        # 5. samples, seeded shuffle per round
        rng = random.Random(SHUFFLE_SEED)
        for r in range(SAMPLES):
            if not active:
                break
            order = list(active)
            rng.shuffle(order)
            prev = ""
            for s in order:
                if s.status != "ok":
                    continue
                try:
                    smp = Sample(round=r, prev=prev)
                    prev = s.op.name
                    smp.event_batch_ms = self._event_batch(s.op.run, s.iters)
                    smp.host_sync_ms = self._host_sync(s.op.run, s.host_iters)
                    if s.phase_iters:
                        smp.phase_ms = self._phase_times(s.op, s.phase_iters)
                    smp.sm, smp.mem, smp.temp, smp.reasons = self._snap()
                    s.samples.append(smp)
                except Exception as e:  # noqa: BLE001
                    drop(s, "timing", e)
                    if lost:
                        break
            active = [s for s in active if s.status == "ok"]

        # 6-7. flags, stats, rates
        med_clock = stats_of([smp.sm for s in active for smp in s.samples])["median"]
        for s in active:
            flagged = flag_samples(s.samples)
            ok = [x for x in s.samples if not x.flagged]
            s.clocks = {"event_batch_ms": stats_of([x.event_batch_ms for x in ok]),
                        "host_sync_ms": stats_of([x.host_sync_ms for x in ok])}
            for name, _ in s.op.phases:
                s.clocks[f"phase_{name}_ms"] = stats_of([x.phase_ms.get(name, math.nan) for x in ok])
            if flagged * 2 > len(s.samples):
                s.status = "invalid_throttled"
            xs, ys = [x.sm for x in s.samples], [x.event_batch_ms for x in s.samples]
            if len(xs) > 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
                s.clock_time_corr = cov / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
            t = (s.clocks["event_batch_ms"]["median"] or math.nan) * 1e-3
            if s.op.flops > 0 and t > 0:
                s.tflops = s.op.flops / t / 1e12
                if self.limits["tflops"] > 0 and s.tflops > RATE_SLACK * self.limits["tflops"]:
                    s.plausibility.append("FLOP rate above the measured FMA peak")
            if s.op.dram_bytes > 0 and t > 0:
                s.dram_gb_per_s = s.op.dram_bytes / t / 1e9
                if self.limits["dram_gb_per_s"] > 0 and s.dram_gb_per_s > RATE_SLACK * self.limits["dram_gb_per_s"]:
                    s.plausibility.append("DRAM rate above the measured bandwidth")

        # 8. teardown
        for s in grp:
            if s.op.teardown and not lost:
                try:
                    s.op.teardown()
                except Exception as e:  # noqa: BLE001
                    if s.status == "ok":
                        self._fail(s, "teardown", e)
        if lost:
            self.dead_group = g
        else:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.groups_json.append({"group": g, "device_used_mib_before": used_before,
                                 "device_used_mib_after_setup": used_after, "warmup_s": warm_s,
                                 "warmup_sm_mhz": warm_clocks, "median_sm_mhz": med_clock, "context_lost": lost})

    # output
    def write_json(self, path: str | Path) -> None:
        def num(x):
            return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x

        ops = []
        for s in self.ops:
            o = {"name": s.op.name, "group": s.op.group, "params": {k: str(v) for k, v in s.op.params.items()},
                 "status": s.status}
            if s.stage:
                o["stage"] = s.stage
            if s.error:
                o["error"] = s.error
            if s.check is not None:
                o["check"] = {k: num(s.check.get(k)) for k in ("ok", "max_abs_err", "max_rel_err", "note")}
            o.update({"mem_mib": num(s.mem_mib), "mem_peak_mib": num(s.mem_peak_mib),
                      "work": {"flops": s.op.flops, "dram_bytes": s.op.dram_bytes},
                      "iters": {"event_batch": s.iters, "host_sync": s.host_iters, "phases": s.phase_iters},
                      "clocks": s.clocks, "rates": {"tflops": num(s.tflops), "dram_gb_per_s": num(s.dram_gb_per_s)},
                      "clock_time_corr": num(s.clock_time_corr), "plausibility": s.plausibility,
                      "samples": [{"round": x.round, "prev": x.prev, "event_batch_ms": num(x.event_batch_ms),
                                   "host_sync_ms": num(x.host_sync_ms), **({"phase_ms": x.phase_ms} if x.phase_ms else {}),
                                   "sm_mhz": x.sm, "mem_mhz": x.mem, "temp_c": x.temp,
                                   "clock_events": reason_names(x.reasons), "flagged": x.flagged,
                                   **({"flag_reason": x.flag_reason} if x.flagged else {})} for x in s.samples]})
            ops.append(o)
        doc = {"schema": "airgap-bench/1", "benchmark": self.bench, "created": self.created, "warnings": self.warnings,
               "env": self.env, "limits": {**self.limits, "source": "results/cuda/peak.json"},
               "protocol": {"samples": SAMPLES, "target_sample_ms": TARGET_SAMPLE_MS,
                            "warmup_s": [WARMUP_MIN_S, WARMUP_MAX_S], "clock_tol": CLOCK_TOL,
                            "throttle_drop": THROTTLE_DROP, "rate_slack": RATE_SLACK,
                            "interleave": "seeded shuffle per round within a group (Python random.Random)",
                            "shuffle_seed": SHUFFLE_SEED,
                            "throttle_rule": "flag a sample on a thermal or hardware-slowdown reason, or an SM clock "
                                             "more than throttle_drop below the op's median that the power cap does "
                                             "not explain; power cap recorded, not flagged; op invalid if more than "
                                             "half its samples are flagged",
                            "python_note": "event_batch_ms is CUDA-event time over a Python loop of calls, so it "
                                           "includes dispatch gaps between short kernels"},
               "groups": self.groups_json, "ops": ops}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=1) + "\n")
