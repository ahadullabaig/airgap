"""Checks benchmark result files against the shared schema "airgap-bench/1" (kernels/harness/SCHEMA.md),
whichever harness wrote them (C++ kernels/harness/harness.h or Python bench/harness.py).

Example:
    .venv/bin/python -I bench/check_bench_schema.py results/harness/selftest.json results/attention/b1.json
"""

import json
import sys

STATUS = {"ok", "failed", "skipped", "check_failed", "invalid_throttled"}
TOP = {"schema", "benchmark", "created", "warnings", "env", "limits", "protocol", "groups", "ops"}
ENV = {"gpu", "cc", "sms", "memory_mib", "max_sm_mhz", "driver", "cuda_runtime", "nvcc", "host_compiler", "ac",
       "platform_profile", "other_gpu_processes"}
PROTOCOL = {"samples", "target_sample_ms", "warmup_s", "clock_tol", "throttle_drop", "rate_slack", "interleave",
            "shuffle_seed", "throttle_rule"}
GROUP = {"group", "device_used_mib_before", "device_used_mib_after_setup", "warmup_s", "warmup_sm_mhz",
         "median_sm_mhz", "context_lost"}
OP = {"name", "group", "params", "status", "mem_mib", "work", "iters", "clocks", "rates", "clock_time_corr",
      "plausibility", "samples"}
STATS = {"n", "median", "p10", "p90", "min", "max"}
SAMPLE = {"round", "prev", "event_batch_ms", "host_sync_ms", "sm_mhz", "mem_mhz", "temp_c", "clock_events", "flagged"}


def check(path: str) -> list[str]:
    errs = []
    d = json.load(open(path))

    def need(obj, keys, where):
        missing = keys - set(obj)
        if missing:
            errs.append(f"{where}: missing {sorted(missing)}")

    need(d, TOP, "top level")
    if d.get("schema") != "airgap-bench/1":
        errs.append(f"schema is {d.get('schema')!r}")
    need(d.get("env", {}), ENV, "env")
    need(d.get("protocol", {}), PROTOCOL, "protocol")
    for g in d.get("groups", []):
        need(g, GROUP, f"group {g.get('group')}")
    groups = {g.get("group") for g in d.get("groups", [])}
    for o in d.get("ops", []):
        w = f"op {o.get('name')}"
        need(o, OP, w)
        if o.get("status") not in STATUS:
            errs.append(f"{w}: status {o.get('status')!r}")
        if o.get("status") != "ok" and not o.get("error") and o.get("status") != "check_failed" \
                and o.get("status") != "invalid_throttled":
            errs.append(f"{w}: status {o.get('status')} without an error message")
        if not all(isinstance(v, str) for v in o.get("params", {}).values()):
            errs.append(f"{w}: params must be strings")
        if o.get("status") == "ok":
            ev = o.get("clocks", {}).get("event_batch_ms")
            if not ev:
                errs.append(f"{w}: ok without event_batch_ms")
            else:
                need(ev, STATS, f"{w} event_batch_ms")
            if not o.get("samples"):
                errs.append(f"{w}: ok without samples")
        for i, s in enumerate(o.get("samples", [])):
            need(s, SAMPLE, f"{w} sample {i}")
        if o.get("group") not in groups and o.get("status") not in ("skipped",):
            errs.append(f"{w}: group {o.get('group')!r} not in groups")
    return errs


def main() -> None:
    bad = 0
    for path in sys.argv[1:]:
        errs = check(path)
        print(f"{'ok  ' if not errs else 'FAIL'} {path}" + ("" if not errs else "\n  " + "\n  ".join(errs[:10])))
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
