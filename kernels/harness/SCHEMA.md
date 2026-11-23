# Benchmark result schema `airgap-bench/1`

Every benchmark writes one JSON file in this shape, whatever language it is written in: the C++ harness (`harness.h`) for our kernels, and its Python twin (`bench/harness.py`, self-test `bench/harness_selftest.py`) for the PyTorch baselines. `bench/check_bench_schema.py` checks any result file against the fields below. C++ and Python ops cannot be interleaved in one process, so comparisons across the two are between separate runs.

## Protocol (PLAN.md section 3.7)

| Constant | Value | Meaning |
|---|---|---|
| `samples` | 21 | timed samples per op |
| `target_sample_ms` | 20 | iteration count calibrated once so a sample's event batch lasts about this long, then frozen |
| `warmup_s` | 1 to 10 | warm up all ops of a group until three SM-clock readings agree within `clock_tol` |
| `clock_tol` | 0.01 | |
| `throttle_drop` | 0.03 | a sample is flagged if its SM clock is this far below the group median |
| `rate_slack` | 1.05 | a derived rate above this multiple of the measured limit is implausible |

- **Interleaving:** ops in one group are sampled in a fresh order each round, from a seeded shuffle (`shuffle_seed`); each sample records the op that ran just before it (`prev`). Rotation alone would keep each op behind the same neighbour, and a kernel's speed depends on what ran before it (power headroom): the same FMA kernel read 1.6% faster beside a memory-bound kernel than beside other FMA kernels. Groups run one after another, so a group's buffers must fit in device memory together. Differences between ops in different groups, or between C++ and Python runs, are only meaningful above that cross-group gap. Group composition matters more than that for compute-bound kernels: the same FMA kernel ran at 6.68 TFLOPS inside a group of short memory-bound K0 kernels (median SM clock 1717 MHz) against 7.45 alone (1897 MHz), 10% apart, because the SM clock dithers under the power cap in such a group (cause unknown). Group compute-bound kernels with comparable work, and compare across groups only with that in mind.
- **Throttling** (our reading of the plan's "discard throttled runs"): a sample is flagged on a thermal or hardware-slowdown reason, or on an SM clock more than `throttle_drop` below the op's own median when the software power cap does not explain the drop. Under the power cap the controller dithers the SM clock by about 15% for memory-heavy as well as compute-heavy work; that is the sustained behaviour being measured, so it is not flagged (K0, memory-bound, ran its fastest 4K sample at 1477 MHz). Each op records `clock_time_corr`, the correlation between its times and its SM clocks, which shows whether it is clock-sensitive at all. The clock is a single NVML reading after each sample, not an average over it, so under dither it is a weak proxy (that FMA kernel: r = -0.10 in the dithering group, -0.57 alone). The software power cap is this laptop's steady state under compute load: recorded, not flagged. An op with more than half its samples flagged is `invalid_throttled`. Statistics use unflagged samples only.
- **Correctness:** an op's check runs before any timing. An op whose check fails is never timed.
- **Failures are results:** an error in setup (for example out of memory), check, warm-up, calibration or timing marks that op `failed` with its stage and message. If the error left the CUDA context unusable (an illegal memory access), every op not yet timed is `skipped`. On this driver a process cannot create a new context after such an error, so those ops must be rerun in a new process.

## File

```
schema        "airgap-bench/1"
benchmark     name of the binary or script
created       local time, ISO 8601
warnings[]    e.g. measured limits missing (the plausibility guard is then off)
env           gpu, cc, sms, memory_mib, max_sm_mhz, driver, cuda_runtime, cuda_driver_api, nvcc,
              host_compiler, ac ("on"/"off"), platform_profile, other_gpu_processes
              Python: nvcc, host_compiler and cuda_driver_api are null; python, torch and tf32
              (matmul, cudnn, float32_matmul_precision) are added
limits        tflops, dram_gb_per_s (measured; results/cuda/peak.json)
protocol      the constants above, plus the interleave and throttle rules as text
groups[]      group, device_used_mib_before, device_used_mib_after_setup, warmup_s, warmup_sm_mhz[],
              median_sm_mhz, context_lost
ops[]         see below
```

Each op:

```
name, group, params{}            identification; params are strings (e.g. "N": "1024")
status                           ok | failed | skipped | check_failed | invalid_throttled
stage, error                     present when not ok
check{ok, max_abs_err,           present when the op has a check
      max_rel_err, note}
mem_mib                          device memory taken by this op's setup
mem_peak_mib                     (Python) mem_mib plus the peak PyTorch allocates on top during one run;
                                 library attention allocates its N x N inside the call
work{flops, dram_bytes}          declared per iteration (0 = not declared)
iters{event_batch, host_sync,    frozen iteration counts
      phases}
clocks{<name>: {n, median,       per-iteration milliseconds over unflagged samples:
        p10, p90, min, max}}       event_batch_ms   CUDA events around a back-to-back batch
                                   host_sync_ms     host clock around launch + synchronise, one at a time
                                   phase_<p>_ms     CUDA events around each declared phase
rates{tflops, dram_gb_per_s}     from the event_batch median and the declared work
clock_time_corr                  Pearson r between event_batch_ms and SM clock over all samples
plausibility[]                   reasons a rate exceeds the measured limits
samples[]                        round, prev, event_batch_ms, host_sync_ms, phase_ms{}, sm_mhz, mem_mhz,
                                 temp_c, clock_events[], flagged, flag_reason
```

One iteration is the whole operation being measured (for V1, its three launches). In the Python harness, `event_batch_ms` is event time over a Python loop of calls, so for short kernels it includes the gaps Python dispatch leaves between launches; that is part of what a PyTorch baseline costs. The two clocks differ most for small kernels: for an empty kernel, 1.7 us back to back against 4.5 us with a synchronise.
