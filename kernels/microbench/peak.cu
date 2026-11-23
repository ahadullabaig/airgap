// Measured limits of this GPU, the inputs to the predicted kernel times (section 3.1 of PLAN.md).
//
//   fma        FP32 fused multiply-add throughput (8 independent chains per thread), short burst
//   fma_sustained  the same kernel back to back for about 10 s, in 1 s chunks: settled rate and clock
//   dram_read  read-only streaming bandwidth (512 MiB, float4 loads)
//   dram_copy  read + write streaming bandwidth (2 x 256 MiB)
//   l2_read    read bandwidth from a 512 KiB buffer that stays in L2 (each thread walks the whole
//              buffer at a prime offset per step, so warps stay coalesced and every line is read)
//   launch_b2b back-to-back empty-kernel launches, per launch
//   launch_rt  empty kernel + device synchronise round trip, per call
//
// Each test: warm-up, then SAMPLES timed samples (CUDA events around a batch of launches);
// median, min and max are reported. SM clock, memory clock, temperature and NVML clock-event
// (throttle) reasons are read before and after each test. Writes JSON to stdout.
// "reliability_voltage" is the normal cap on the top boost clocks; "sw_power_cap" and the thermal
// reasons are the ones that indicate a throttled measurement.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define NVCHECK(x) do { nvmlReturn_t r = (x); if (r != NVML_SUCCESS) { \
    fprintf(stderr, "%s:%d NVML %s\n", __FILE__, __LINE__, nvmlErrorString(r)); exit(1); } } while (0)

static const int SAMPLES = 21;
static const int FMA_ITERS = 4096;

__global__ void fma_kernel(float* out, float seed) {
    float a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3, a4 = seed + 4, a5 = seed + 5, a6 = seed + 6,
          a7 = seed + 7;
    const float m = 0.999999f, c = 1e-7f;
    for (int i = 0; i < FMA_ITERS; ++i) {
        a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
        a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
    }
    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // never true; keeps the work
}

__global__ void read_kernel(const float4* __restrict__ in, size_t n4, size_t mask4, float* out) {
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = in[i & mask4];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}

__global__ void l2_kernel(const float4* __restrict__ in, unsigned mask4, int reps, float* out) {
    float acc = 0.f;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 8
    for (int r = 0; r < reps; ++r) {
        float4 v = in[(tid + (unsigned)r * 7919u) & mask4];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) out[i] = in[i];
}

__global__ void empty_kernel() {}

struct Snap { unsigned sm, mem, temp; unsigned long long reasons; };

static nvmlDevice_t g_dev;
static Snap snap() {
    Snap s{};
    NVCHECK(nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_SM, &s.sm));
    NVCHECK(nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_MEM, &s.mem));
    nvmlTemperature_t t{};
    t.version = nvmlTemperature_v1;
    t.sensorType = NVML_TEMPERATURE_GPU;
    NVCHECK(nvmlDeviceGetTemperatureV(g_dev, &t));
    s.temp = (unsigned)t.temperature;
    NVCHECK(nvmlDeviceGetCurrentClocksEventReasons(g_dev, &s.reasons));
    return s;
}

// Times `batch` calls of f per sample; returns per-call milliseconds for each sample.
template <class F> static std::vector<double> sample(F f, int batch) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    for (int w = 0; w < 3; ++w) for (int b = 0; b < batch; ++b) f();  // warm-up
    CHECK(cudaDeviceSynchronize());
    std::vector<double> ms;
    for (int s = 0; s < SAMPLES; ++s) {
        CHECK(cudaEventRecord(t0));
        for (int b = 0; b < batch; ++b) f();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float m = 0;
        CHECK(cudaEventElapsedTime(&m, t0, t1));
        ms.push_back(m / batch);
    }
    CHECK(cudaGetLastError());
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

// Names of the set NVML clock-event (throttle) reason bits, e.g. ["sw_power_cap"].
static std::string reasons(unsigned long long r) {
    static const std::pair<unsigned long long, const char*> names[] = {
        {nvmlClocksEventReasonGpuIdle, "idle"},
        {nvmlClocksEventReasonApplicationsClocksSetting, "app_clocks"},
        {nvmlClocksEventReasonSwPowerCap, "sw_power_cap"},
        {nvmlClocksThrottleReasonHwSlowdown, "hw_slowdown"},
        {nvmlClocksEventReasonSyncBoost, "sync_boost"},
        {nvmlClocksEventReasonSwThermalSlowdown, "sw_thermal"},
        {nvmlClocksThrottleReasonHwThermalSlowdown, "hw_thermal"},
        {nvmlClocksThrottleReasonHwPowerBrakeSlowdown, "hw_power_brake"},
        {nvmlClocksEventReasonDisplayClockSetting, "display_clock"},
        {nvmlClocksEventReasonBoardLimit, "board_limit"},
        {nvmlClocksEventReasonReliability, "reliability_voltage"},
    };
    std::string out = "[";
    for (auto& [bit, name] : names)
        if (r & bit) out += std::string(out.size() > 1 ? ", " : "") + "\"" + name + "\"";
    return out + "]";
}

static std::string stats_json(const char* name, std::vector<double> ms, double work, const char* unit,
                              const Snap& a, const Snap& b) {
    std::sort(ms.begin(), ms.end());
    double med = ms[ms.size() / 2], lo = ms.front(), hi = ms.back();
    char buf[2048];
    // rate from the median time; best rate from the fastest sample
    snprintf(buf, sizeof buf,
             "  \"%s\": {\"ms_median\": %.6f, \"ms_min\": %.6f, \"ms_max\": %.6f, \"%s\": %.2f, \"%s_best\": %.2f,\n"
             "    \"sm_mhz\": [%u, %u], \"mem_mhz\": [%u, %u], \"temp_c\": [%u, %u], \"clock_events_before\": %s, \"clock_events_after\": %s}",
             name, med, lo, hi, unit, work / (med * 1e-3), unit, work / (lo * 1e-3), a.sm, b.sm, a.mem, b.mem,
             a.temp, b.temp, reasons(a.reasons).c_str(), reasons(b.reasons).c_str());
    return buf;
}

int main() {
    NVCHECK(nvmlInit_v2());
    NVCHECK(nvmlDeviceGetHandleByIndex_v2(0, &g_dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    unsigned max_sm = 0, max_mem = 0;
    NVCHECK(nvmlDeviceGetMaxClockInfo(g_dev, NVML_CLOCK_SM, &max_sm));
    NVCHECK(nvmlDeviceGetMaxClockInfo(g_dev, NVML_CLOCK_MEM, &max_mem));
    int rt = 0, drv = 0;
    CHECK(cudaRuntimeGetVersion(&rt));
    CHECK(cudaDriverGetVersion(&drv));
    std::vector<std::string> parts;

    // FP32 FMA
    {
        float* out;
        CHECK(cudaMalloc(&out, sizeof(float) * p.multiProcessorCount * 32 * 256));
        int blocks = p.multiProcessorCount * 32, threads = 256;
        double flops = 2.0 * 8 * FMA_ITERS * (double)blocks * threads;
        Snap a = snap();
        auto ms = sample([&] { fma_kernel<<<blocks, threads>>>(out, 1.0f); }, 20);
        Snap b = snap();
        parts.push_back(stats_json("fma", ms, flops / 1e12, "tflops", a, b));

        // sustained: 1 s chunks, back to back, for about 10 s
        int per_chunk = std::max(1, (int)(1000.0 / std::max(ms[ms.size() / 2], 1e-3)));
        std::vector<double> rates, clocks;
        cudaEvent_t c0, c1;
        CHECK(cudaEventCreate(&c0));
        CHECK(cudaEventCreate(&c1));
        Snap sa = snap();
        Snap last = sa;
        for (int chunk = 0; chunk < 10; ++chunk) {
            CHECK(cudaEventRecord(c0));
            for (int i = 0; i < per_chunk; ++i) fma_kernel<<<blocks, threads>>>(out, 1.0f);
            CHECK(cudaEventRecord(c1));
            CHECK(cudaEventSynchronize(c1));
            float m = 0;
            CHECK(cudaEventElapsedTime(&m, c0, c1));
            last = snap();
            rates.push_back(flops * per_chunk / (m * 1e-3) / 1e12);
            clocks.push_back(last.sm);
        }
        std::vector<double> tail(rates.end() - 5, rates.end());
        std::sort(tail.begin(), tail.end());
        std::string r_list, c_list;
        for (size_t i = 0; i < rates.size(); ++i) {
            char b2[32];
            snprintf(b2, sizeof b2, "%s%.2f", i ? ", " : "", rates[i]);
            r_list += b2;
            snprintf(b2, sizeof b2, "%s%.0f", i ? ", " : "", clocks[i]);
            c_list += b2;
        }
        char buf[1024];
        snprintf(buf, sizeof buf,
                 "  \"fma_sustained\": {\"tflops_settled\": %.2f, \"tflops_per_second\": [%s], \"sm_mhz_per_second\": [%s],\n"
                 "    \"temp_c\": [%u, %u], \"clock_events_before\": %s, \"clock_events_after\": %s}",
                 tail[2], r_list.c_str(), c_list.c_str(), sa.temp, last.temp, reasons(sa.reasons).c_str(),
                 reasons(last.reasons).c_str());
        parts.push_back(buf);
        cudaFree(out);
    }
    // DRAM read, DRAM copy, L2 read
    {
        size_t bytes = (size_t)512 << 20, n4 = bytes / 16;
        float4 *in, *o4;
        float* out;
        CHECK(cudaMalloc(&in, bytes));
        CHECK(cudaMalloc(&o4, bytes / 2));
        CHECK(cudaMalloc(&out, sizeof(float)));
        CHECK(cudaMemset(in, 0, bytes));
        int blocks = p.multiProcessorCount * 16, threads = 512;
        Snap a = snap();
        auto ms = sample([&] { read_kernel<<<blocks, threads>>>(in, n4, ~(size_t)0, out); }, 5);
        Snap b = snap();
        parts.push_back(stats_json("dram_read", ms, bytes / 1e9, "gb_per_s", a, b));

        size_t half4 = n4 / 2;
        a = snap();
        ms = sample([&] { copy_kernel<<<blocks, threads>>>(in, o4, half4); }, 5);
        b = snap();
        parts.push_back(stats_json("dram_copy", ms, 2.0 * half4 * 16 / 1e9, "gb_per_s", a, b));

        unsigned l2_mask4 = (unsigned)(((size_t)512 << 10) / 16 - 1);  // 512 KiB, power of two
        int reps = (int)(n4 / ((size_t)blocks * threads));            // 512 MiB of reads in total
        a = snap();
        ms = sample([&] { l2_kernel<<<blocks, threads>>>(in, l2_mask4, reps, out); }, 5);
        b = snap();
        parts.push_back(stats_json("l2_read", ms, (double)reps * blocks * threads * 16 / 1e9, "gb_per_s", a, b));
        cudaFree(in); cudaFree(o4); cudaFree(out);
    }
    // launches
    {
        Snap a = snap();
        auto ms = sample([&] { empty_kernel<<<1, 1>>>(); }, 1000);
        Snap b = snap();
        parts.push_back(stats_json("launch_b2b", ms, 1e-6, "mcalls_per_s", a, b));
        a = snap();
        std::vector<double> rt_ms;
        for (int s = 0; s < SAMPLES; ++s) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 1000; ++i) { empty_kernel<<<1, 1>>>(); CHECK(cudaDeviceSynchronize()); }
            auto t1 = std::chrono::steady_clock::now();
            rt_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000);
        }
        b = snap();
        parts.push_back(stats_json("launch_rt", rt_ms, 1e-6, "mcalls_per_s", a, b));
    }

    printf("{\n  \"gpu\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"l2_bytes\": %d, \"runtime\": %d, \"driver_api\": %d,\n"
           "  \"max_sm_mhz\": %u, \"max_mem_mhz\": %u, \"fp32_lanes_per_sm\": 128,\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize, rt, drv, max_sm, max_mem);
    for (size_t i = 0; i < parts.size(); ++i) printf("%s%s\n", parts[i].c_str(), i + 1 < parts.size() ? "," : "");
    printf("}\n");
    nvmlShutdown();
    return 0;
}
