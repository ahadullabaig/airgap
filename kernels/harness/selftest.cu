// Self-test of the benchmark harness. Run before any benchmark relies on it.
//
//   limits     reproduces results/cuda/peak.json: DRAM read, sustained FMA, empty-launch costs
//   scaling    doubling an FMA kernel's loop count doubles its time; one op registered under two
//              names gives matching medians (their gap is the smallest difference worth claiming);
//              declared phases add up to the whole iteration
//   guards     a planted wrong result is never timed; a kernel declaring 10x its real traffic is
//              flagged as implausible
//   oom        an allocation that cannot succeed is recorded as a failed setup and the run goes on
//   sticky     (only with --with-fault) an illegal memory access loses the context; everything not yet
//              timed is skipped with the reason (this driver cannot recreate a context in the same
//              process), and the result file is still written. Opt-in because it faults the GPU the
//              desktop is using (a driver Xid error each run).
//
// Writes results/harness/selftest.json and exits non-zero if an expectation fails.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "harness.h"

using harness::CheckResult;
using harness::Op;

__global__ void read_kernel(const float4* __restrict__ in, size_t n4, float* out) {
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}

// Same as kernels/microbench/peak.cu: the loop count is a compile-time constant, so the loop is fully
// unrolled 16 times, as in peak.cu (a runtime count leaves loop overhead that costs about 11% of the FMA
// rate; a full unroll of 4096 iterations overflows the instruction cache).
template <int ITERS>
__global__ void fma_kernel(float* out) {
    float a0 = 1, a1 = 2, a2 = 3, a3 = 4, a4 = 5, a5 = 6, a6 = 7, a7 = 8;
    const float m = 0.999999f, c = 1e-7f;
#pragma unroll 16
    for (int i = 0; i < ITERS; ++i) {
        a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
        a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
    }
    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void empty_kernel() {}

__global__ void scale_kernel(float* x, int n, float a, int plant_error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + ((plant_error && i == n / 2) ? 1.0f : 0.0f);
}

__global__ void bad_write_kernel(float* p) { p[(size_t)1 << 40] = 1.0f; }

static int failures = 0;
static void expect(bool ok, const std::string& what) {
    printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
    failures += !ok;
}

static const harness::Runner::State* find(const harness::Runner& r, const std::string& name) {
    for (auto& s : r.results())
        if (s.op.name == name) return &s;
    return nullptr;
}

static double med(const harness::Runner::State* s, const std::string& clock = "event_batch_ms") {
    auto it = s->clocks.find(clock);
    return it == s->clocks.end() ? NAN : it->second.median;
}

// Throttle flags on made-up samples: no real run has tripped them yet.
static void test_flags() {
    using harness::Sample;
    auto mk = [](unsigned sm, unsigned long long reasons) {
        Sample m;
        m.snap.sm = sm;
        m.snap.reasons = reasons;
        return m;
    };
    // power-capped steady state at 1897-1957 MHz: nothing flagged
    std::vector<Sample> steady;
    for (unsigned c : {1897u, 1905u, 1920u, 1935u, 1957u, 1897u, 1912u})
        steady.push_back(mk(c, nvmlClocksEventReasonSwPowerCap | nvmlClocksEventReasonReliability));
    expect(harness::flag_samples(steady) == 0, "flags: power cap and normal clock spread are not flagged");
    // power-cap dither: a 15% sag under the power cap is the steady state, not flagged
    std::vector<Sample> dither = steady;
    dither[3].snap.sm = 1650;
    expect(harness::flag_samples(dither) == 0, "flags: a clock sag under the power cap is not flagged");
    // one thermal sample, one clock sag the power cap does not explain, the rest normal
    std::vector<Sample> mixed = steady;
    mixed[2].snap.reasons |= nvmlClocksEventReasonSwThermalSlowdown;
    mixed[4].snap.sm = 1730;
    mixed[4].snap.reasons = nvmlClocksEventReasonReliability;
    int n = harness::flag_samples(mixed);
    expect(n == 2 && mixed[2].flagged && mixed[4].flagged,
           "flags: a thermal reason and an unexplained clock sag are flagged");
    expect(!harness::mostly_flagged(n, mixed.size()), "flags: 2 of 7 flagged leaves the op valid");
    // majority thermal: op invalid
    std::vector<Sample> hot = steady;
    for (size_t i = 0; i < 4; ++i) hot[i].snap.reasons |= nvmlClocksThrottleReasonHwThermalSlowdown;
    expect(harness::mostly_flagged(harness::flag_samples(hot), hot.size()), "flags: 4 of 7 thermal makes the op invalid");
}

int main(int argc, char** argv) {
    const bool with_fault = argc > 1 && std::string(argv[1]) == "--with-fault";
    test_flags();
    harness::Limits lim = harness::read_limits("../../results/cuda/peak.json");
    expect(lim.tflops > 0 && lim.dram_gb_per_s > 0, "measured limits found in peak.json");
    printf("limits from peak.json: %.2f TFLOPS, %.1f GB/s\n", lim.tflops, lim.dram_gb_per_s);
    harness::Runner run("harness_selftest", lim);
    const int sms = 16, fma_blocks = sms * 32, fma_threads = 256;

    // limits
    static float4* big = nullptr;
    static float* sink = nullptr;
    const size_t big_bytes = (size_t)512 << 20, n4 = big_bytes / 16;
    {
        Op op;
        op.name = "dram_read";
        op.group = "limits";
        op.dram_bytes = (double)big_bytes;
        op.setup = [&] {
            HCHECK(cudaMalloc(&big, big_bytes));
            HCHECK(cudaMalloc(&sink, 4 * fma_blocks * fma_threads));
            HCHECK(cudaMemset(big, 0, big_bytes));
        };
        op.run = [&](cudaStream_t st) { read_kernel<<<sms * 16, 512, 0, st>>>(big, n4, sink); };
        op.teardown = [&] { cudaFree(big); };
        run.add(op);
    }
    auto fma_op = [&](const std::string& name, const std::string& group, int iters) {
        Op op;
        op.name = name;
        op.group = group;
        op.params = {{"iters", std::to_string(iters)}};
        op.flops = 2.0 * 8 * iters * (double)fma_blocks * fma_threads;
        op.run = [=](cudaStream_t st) {
            if (iters == 1024) fma_kernel<1024><<<fma_blocks, fma_threads, 0, st>>>(sink);
            else if (iters == 2048) fma_kernel<2048><<<fma_blocks, fma_threads, 0, st>>>(sink);
            else fma_kernel<4096><<<fma_blocks, fma_threads, 0, st>>>(sink);
        };
        return op;
    };
    run.add(fma_op("fma", "limits", 4096));
    {
        Op op;
        op.name = "empty";
        op.group = "limits";
        op.run = [](cudaStream_t st) { empty_kernel<<<1, 1, 0, st>>>(); };
        run.add(op);
    }

    // scaling
    run.add(fma_op("fma_x1", "scaling", 2048));
    run.add(fma_op("fma_x2", "scaling", 4096));
    for (const char* nm : {"twin_a", "twin_b"}) run.add(fma_op(nm, "scaling", 1024));
    static float *ph_dev = nullptr, *ph_host = nullptr;
    const int ph_n = 4 << 20;  // 16 MiB
    {
        Op op;
        op.name = "phases";
        op.group = "scaling";
        op.setup = [&] {
            HCHECK(cudaMalloc(&ph_dev, ph_n * sizeof(float)));
            HCHECK(cudaMallocHost(&ph_host, ph_n * sizeof(float)));
            for (int i = 0; i < ph_n; ++i) ph_host[i] = 1.0f;
        };
        harness::Launch h2d = [&](cudaStream_t st) {
            HCHECK(cudaMemcpyAsync(ph_dev, ph_host, ph_n * sizeof(float), cudaMemcpyHostToDevice, st));
        };
        harness::Launch kern = [&](cudaStream_t st) { scale_kernel<<<(ph_n + 255) / 256, 256, 0, st>>>(ph_dev, ph_n, 2.f, 0); };
        op.phases = {{"h2d", h2d}, {"kernel", kern}};
        op.run = [=](cudaStream_t st) { h2d(st); kern(st); };
        op.teardown = [&] { cudaFree(ph_dev); cudaFreeHost(ph_host); };
        run.add(op);
    }

    // guards
    static float* g_dev = nullptr;
    const int g_n = 1 << 20;
    {
        Op op;
        op.name = "planted_error";
        op.group = "guards";
        op.setup = [&] { HCHECK(cudaMalloc(&g_dev, g_n * sizeof(float))); };
        op.run = [&](cudaStream_t st) { scale_kernel<<<(g_n + 255) / 256, 256, 0, st>>>(g_dev, g_n, 1.f, 1); };
        op.check = [&] {
            std::vector<float> h(g_n, 3.0f);
            HCHECK(cudaMemcpy(g_dev, h.data(), g_n * sizeof(float), cudaMemcpyHostToDevice));
            scale_kernel<<<(g_n + 255) / 256, 256>>>(g_dev, g_n, 1.f, 1);
            HCHECK(cudaMemcpy(h.data(), g_dev, g_n * sizeof(float), cudaMemcpyDeviceToHost));
            CheckResult c;
            for (float v : h) c.max_abs_err = std::max(c.max_abs_err, (double)std::fabs(v - 3.0f));
            c.ok = c.max_abs_err == 0;
            c.note = "scale by 1 must leave the input unchanged";
            return c;
        };
        op.teardown = [&] { cudaFree(g_dev); };
        run.add(op);
    }
    {
        Op op;
        op.name = "overclaimed_traffic";
        op.group = "guards";
        op.dram_bytes = 10.0 * big_bytes / 8;  // reads 1/8 of the big buffer, claims 10x that
        op.setup = [&] { HCHECK(cudaMalloc(&big, big_bytes)); HCHECK(cudaMemset(big, 0, big_bytes)); };
        op.run = [&](cudaStream_t st) { read_kernel<<<sms * 16, 512, 0, st>>>(big, n4 / 8, sink); };
        op.teardown = [&] { cudaFree(big); };
        run.add(op);
    }

    // oom
    {
        Op op;
        op.name = "too_big";
        op.group = "oom";
        static void* p = nullptr;
        op.setup = [&] { HCHECK(cudaMalloc(&p, (size_t)64 << 30)); };
        op.run = [](cudaStream_t) {};
        run.add(op);
        Op ok;
        ok.name = "after_oom";
        ok.group = "oom";
        ok.run = [](cudaStream_t st) { empty_kernel<<<1, 1, 0, st>>>(); };
        run.add(ok);
    }

    // sticky error, then a group that cannot run any more
    if (with_fault) {
        Op bad;
        bad.name = "illegal_access";
        bad.group = "sticky";
        static float* d = nullptr;
        bad.setup = [&] { HCHECK(cudaMalloc(&d, 1024)); };
        bad.run = [&](cudaStream_t st) { bad_write_kernel<<<1, 1, 0, st>>>(d); };
        run.add(bad);
        Op mate;
        mate.name = "same_group_as_illegal";
        mate.group = "sticky";
        mate.run = [](cudaStream_t st) { empty_kernel<<<1, 1, 0, st>>>(); };
        run.add(mate);
        Op after;
        after.name = "after_reset";
        after.group = "after_sticky";
        after.run = [](cudaStream_t st) { empty_kernel<<<1, 1, 0, st>>>(); };
        run.add(after);
    }

    int bad_ops = run.run_all();
    run.write_json(with_fault ? "../../results/harness/selftest_fault.json" : "../../results/harness/selftest.json");
    const int expected_bad = with_fault ? 5 : 2;
    printf("ops without a valid timing: %d (expected %d: planted_error, too_big%s)\n", bad_ops, expected_bad,
           with_fault ? ", illegal_access, same_group_as_illegal, after_reset" : "");
    expect(bad_ops == expected_bad, "exactly the planted problems lack a valid timing");

    auto* dr = find(run, "dram_read");
    auto* fm = find(run, "fma");
    auto* em = find(run, "empty");
    printf("limits: DRAM %.1f GB/s, FMA %.2f TFLOPS, empty launch %.2f us (events) / %.2f us (host + sync)\n",
           dr->dram_gb_per_s, fm->tflops, med(em) * 1e3, med(em, "host_sync_ms") * 1e3);
    expect(std::fabs(dr->dram_gb_per_s / lim.dram_gb_per_s - 1) < 0.05, "DRAM read within 5% of peak.json");
    expect(std::fabs(fm->tflops / lim.tflops - 1) < 0.05, "FMA within 5% of peak.json (sustained)");
    expect(med(em) * 1e3 > 1.2 && med(em) * 1e3 < 2.5, "empty launch, back to back, near 1.7 us");
    expect(med(em, "host_sync_ms") * 1e3 > 3.0 && med(em, "host_sync_ms") * 1e3 < 7.0,
           "empty launch with sync near 4.6 us");

    double ratio = med(find(run, "fma_x2")) / med(find(run, "fma_x1"));
    printf("scaling: fma x2 / x1 = %.3f\n", ratio);
    expect(std::fabs(ratio - 2) < 0.06, "doubling the work doubles the time (within 3%)");
    double a = med(find(run, "twin_a")), b = med(find(run, "twin_b"));
    double gap = std::fabs(a - b) / std::min(a, b);
    printf("twins: %.4f vs %.4f ms, gap %.2f%%\n", a, b, 100 * gap);
    expect(gap < 0.02, "the same op under two names agrees within 2%");
    double cross = std::fabs(fm->tflops / find(run, "fma_x2")->tflops - 1);
    printf("same 4096-iteration kernel in two groups: %.3f vs %.3f TFLOPS, gap %.2f%% (resolution across groups)\n",
           fm->tflops, find(run, "fma_x2")->tflops, 100 * cross);
    expect(cross < 0.05, "the same kernel in two groups agrees within 5%");
    auto* ph = find(run, "phases");
    double sum = med(ph, "phase_h2d_ms") + med(ph, "phase_kernel_ms");
    printf("phases: h2d %.3f + kernel %.3f = %.3f ms vs whole %.3f ms\n", med(ph, "phase_h2d_ms"),
           med(ph, "phase_kernel_ms"), sum, med(ph));
    expect(std::fabs(sum / med(ph) - 1) < 0.10, "phases add up to the whole iteration (within 10%)");

    auto* pe = find(run, "planted_error");
    expect(pe->status == "check_failed" && pe->samples.empty(), "planted error: check_failed and never timed");
    auto* oc = find(run, "overclaimed_traffic");
    expect(!oc->plausibility.empty(), "overclaimed traffic flagged as implausible");
    auto* tb = find(run, "too_big");
    expect(tb->status == "failed" && tb->stage == "setup", "64 GiB allocation: failed at setup");
    expect(find(run, "after_oom")->status == "ok", "same group runs on after the failed allocation");
    if (with_fault) {
        expect(find(run, "illegal_access")->status == "failed", "illegal access: failed");
        expect(find(run, "same_group_as_illegal")->status == "skipped",
               "rest of the group skipped after the context was lost");
        expect(find(run, "after_reset")->status == "skipped", "later groups skipped after the context was lost");
    }

    printf("%s\n", failures ? "SELFTEST FAILED" : "selftest passed");
    return failures ? 1 : 0;
}
