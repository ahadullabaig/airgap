"""Self-test of bench/harness.py (the Python twin of the C++ harness). Writes results/harness/selftest_py.json.

- a 256 MiB device-to-device copy reproduces the copy bandwidth in results/cuda/peak.json (within 5%)
- the same op under two names agrees within 2%
- a planted wrong result is never timed; an op declaring 10x its traffic is flagged as implausible
- a 64 GiB allocation fails at setup and the rest of the group still runs
- throttle flags on made-up samples (power-cap dither not flagged; thermal and unexplained sags flagged)

Example:
    .venv/bin/python -I bench/harness_selftest.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import harness as H  # noqa: E402

fails = 0


def expect(ok: bool, what: str) -> None:
    global fails
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    fails += not ok


def test_flags() -> None:
    mk = lambda sm, r: H.Sample(round=0, prev="", sm=sm, reasons=r)  # noqa: E731
    steady = [mk(c, 0x4 | 0x400) for c in (1897, 1905, 1920, 1935, 1957, 1897, 1912)]
    expect(H.flag_samples(steady) == 0, "flags: power cap and normal spread not flagged")
    dither = [mk(s.sm, s.reasons) for s in steady]
    dither[3].sm = 1650
    expect(H.flag_samples(dither) == 0, "flags: a sag under the power cap not flagged")
    mixed = [mk(s.sm, s.reasons) for s in steady]
    mixed[2].reasons |= 0x20
    mixed[4].sm, mixed[4].reasons = 1730, 0x400
    expect(H.flag_samples(mixed) == 2 and mixed[2].flagged and mixed[4].flagged,
           "flags: thermal reason and unexplained sag flagged")


def main() -> None:
    test_flags()
    pk = json.loads((ROOT / "results" / "cuda" / "peak.json").read_text())
    run = H.Runner("harness_selftest_py", H.read_limits(ROOT / "results" / "cuda" / "peak.json"))
    buf = {}
    n = (256 << 20) // 4

    def setup_copy():
        buf["a"] = torch.zeros(n, device="cuda")
        buf["b"] = torch.empty(n, device="cuda")

    copy = lambda: buf["b"].copy_(buf["a"])  # noqa: E731
    run.add(H.Op("copy", copy, group="limits", dram_bytes=2.0 * n * 4, setup=setup_copy))
    run.add(H.Op("copy_twin", copy, group="limits", dram_bytes=2.0 * n * 4))
    run.add(H.Op("overclaimed", lambda: buf["b"][: n // 8].copy_(buf["a"][: n // 8]), group="limits",
                 dram_bytes=10 * 2.0 * n * 4 / 8))

    def bad_check():
        x = torch.ones(1024, device="cuda")
        y = x * 1.0 + (torch.arange(1024, device="cuda") == 512).float()
        err = float((y - x).abs().max())
        return {"ok": err == 0, "max_abs_err": err, "note": "multiply by 1 must leave the input unchanged"}

    run.add(H.Op("planted_error", lambda: None, group="guards", check=bad_check))

    def too_big():
        buf["huge"] = torch.empty(64 << 30, dtype=torch.uint8, device="cuda")

    run.add(H.Op("too_big", lambda: None, group="oom", setup=too_big))
    run.add(H.Op("after_oom", lambda: torch.cuda._sleep(100), group="oom"))

    bad = run.run_all()
    out = ROOT / "results" / "harness" / "selftest_py.json"
    run.write_json(out)
    by = {s.op.name: s for s in run.ops}
    ref = pk["dram_copy"]["gb_per_s"]
    print(f"copy {by['copy'].dram_gb_per_s:.1f} GB/s (peak.json copy {ref:.1f}); twin {by['copy_twin'].dram_gb_per_s:.1f}")
    expect(abs(by["copy"].dram_gb_per_s / ref - 1) < 0.05, "copy within 5% of peak.json")
    a, b = by["copy"].clocks["event_batch_ms"]["median"], by["copy_twin"].clocks["event_batch_ms"]["median"]
    expect(abs(a - b) / min(a, b) < 0.02, f"twins agree within 2% ({100 * abs(a - b) / min(a, b):.2f}%)")
    expect(by["overclaimed"].plausibility != [], "overclaimed traffic flagged")
    expect(by["planted_error"].status == "check_failed" and not by["planted_error"].samples, "planted error never timed")
    expect(by["too_big"].status == "failed" and by["too_big"].stage == "setup", "64 GiB allocation failed at setup")
    expect(by["after_oom"].status == "ok", "group runs on after the failed allocation")
    expect(bad == 2, f"exactly the two planted problems lack a timing ({bad})")
    json.loads(out.read_text())
    print("python selftest", "passed" if not fails else "FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
