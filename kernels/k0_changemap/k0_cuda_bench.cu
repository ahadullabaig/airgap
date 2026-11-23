// K0 GPU benchmark on the shared harness (kernels/harness).
//
// Group "kernel": the generic and vec4 kernels at 512x512 (the encoder's view), 720p, 1080p and 4K,
// 3 channels, 16 x 16 blocks, frames already in device memory. Each op cycles through a ring of frames
// at least 4x the L2 size, so every iteration reads its two frames from DRAM as a stream of new frames
// would (with one fixed pair, a 512x512 pair would sit in L2). Declared traffic: both frames + the map.
//
// Group "end_to_end": a new frame copied from pinned host memory, the vec4 kernel, and the map copied
// back, as phases, for the case where frames start on the CPU.
//
// Every op is checked bit for bit against the CPU reference (k0_changemap_serial) before timing.
//
// Usage: k0_cuda_bench [output.json]
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "harness.h"
#include "k0.h"
#include "k0_cuda.h"

using harness::CheckResult;
using harness::Op;

namespace {

struct Size {
    const char* name;
    int h, w;
};
constexpr int C = 3, BLOCK = 16;
constexpr double L2_BYTES = 1.5 * 1048576;

// Host frames: a base image plus small per-frame changes (like consecutive video frames).
std::vector<std::vector<uint8_t>> make_frames(int h, int w, int count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> px(0, 255), d(-3, 3);
    const size_t n = size_t(h) * w * C;
    std::vector<std::vector<uint8_t>> f(count, std::vector<uint8_t>(n));
    for (size_t i = 0; i < n; ++i) f[0][i] = uint8_t(px(rng));
    for (int k = 1; k < count; ++k)
        for (size_t i = 0; i < n; ++i) f[k][i] = uint8_t(std::min(255, std::max(0, int(f[k - 1][i]) + d(rng))));
    return f;
}

size_t map_len(int h, int w) { return size_t((h + BLOCK - 1) / BLOCK) * size_t((w + BLOCK - 1) / BLOCK); }

// Device-resident ring of frames shared by the ops of one size.
struct Ring {
    int h = 0, w = 0, count = 0;
    std::vector<std::vector<uint8_t>> host;
    std::vector<uint8_t*> dev;
    float* out = nullptr;
    long next = 0;
};

CheckResult check_against_cpu(Ring& r, int impl) {
    CheckResult c;
    std::vector<float> ref(map_len(r.h, r.w)), got(ref.size());
    for (int k = 0; k + 1 < r.count && k < 3; ++k) {
        k0_changemap_serial(r.host[k + 1].data(), r.host[k].data(), r.h, r.w, C, BLOCK, ref.data());
        int rc = k0_changemap_cuda_impl(r.dev[k + 1], r.dev[k], r.h, r.w, C, BLOCK, r.out, 0, impl);
        HCHECK(cudaDeviceSynchronize());
        HCHECK(cudaMemcpy(got.data(), r.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (rc != 0 || std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
            c.ok = false;
            c.note = "GPU map differs from the CPU reference (rc " + std::to_string(rc) + ")";
            for (size_t i = 0; i < ref.size(); ++i)
                c.max_abs_err = std::max(c.max_abs_err, (double)std::fabs(ref[i] - got[i]));
            return c;
        }
    }
    c.note = "bit-identical to k0_changemap_serial on 3 frame pairs";
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "../../results/k0/bench_gpu.json";
    harness::Runner run("k0_cuda_bench", harness::read_limits("../../results/cuda/peak.json"));
    const Size sizes[] = {{"512x512", 512, 512}, {"720p", 720, 1280}, {"1080p", 1080, 1920}, {"4k", 2160, 3840}};
    std::vector<std::shared_ptr<Ring>> rings;

    for (const Size& sz : sizes) {
        auto ring = std::make_shared<Ring>();
        rings.push_back(ring);
        ring->h = sz.h;
        ring->w = sz.w;
        const double frame_bytes = double(sz.h) * sz.w * C;
        ring->count = std::max(3, int(std::ceil(4 * L2_BYTES / frame_bytes)) + 1);
        const double traffic = 2 * frame_bytes + 4.0 * map_len(sz.h, sz.w);
        for (int impl : {K0_IMPL_GENERIC, K0_IMPL_VEC4}) {
            Op op;
            op.name = std::string("k0_") + (impl == K0_IMPL_VEC4 ? "vec4_" : "generic_") + sz.name;
            op.group = "kernel";
            op.params = {{"h", std::to_string(sz.h)},
                         {"w", std::to_string(sz.w)},
                         {"c", std::to_string(C)},
                         {"block", std::to_string(BLOCK)},
                         {"impl", impl == K0_IMPL_VEC4 ? "vec4" : "generic"},
                         {"ring_frames", std::to_string(ring->count)},
                         {"predicted_us_dram_bound", std::to_string(traffic / 188e9 * 1e6)}};
            op.dram_bytes = traffic;
            if (impl == K0_IMPL_GENERIC) {  // the first op of a size owns the ring
                op.setup = [ring] {
                    ring->host = make_frames(ring->h, ring->w, ring->count, 7);
                    for (auto& f : ring->host) {
                        uint8_t* d;
                        HCHECK(cudaMalloc(&d, f.size()));
                        HCHECK(cudaMemcpy(d, f.data(), f.size(), cudaMemcpyHostToDevice));
                        ring->dev.push_back(d);
                    }
                    HCHECK(cudaMalloc(&ring->out, map_len(ring->h, ring->w) * sizeof(float)));
                };
                op.teardown = [ring] {
                    for (auto* d : ring->dev) cudaFree(d);
                    ring->dev.clear();
                    cudaFree(ring->out);
                };
            }
            op.check = [ring, impl] { return check_against_cpu(*ring, impl); };
            op.run = [ring, impl](cudaStream_t st) {
                const int k = int(ring->next++ % (ring->count - 1));
                k0_changemap_cuda_impl(ring->dev[k + 1], ring->dev[k], ring->h, ring->w, C, BLOCK, ring->out, st, impl);
            };
            run.add(op);
        }
    }

    // end to end: the new frame starts in pinned host memory
    struct E2E {
        Size sz;
        uint8_t *host_new = nullptr, *dev_new = nullptr, *dev_ref = nullptr;
        float *dev_out = nullptr, *host_out = nullptr;
    };
    for (const Size& sz : sizes) {
        auto e = std::make_shared<E2E>();
        e->sz = sz;
        const size_t n = size_t(sz.h) * sz.w * C, m = map_len(sz.h, sz.w);
        Op op;
        op.name = std::string("k0_e2e_") + sz.name;
        op.group = "end_to_end";
        op.params = {{"h", std::to_string(sz.h)}, {"w", std::to_string(sz.w)}, {"c", std::to_string(C)},
                     {"block", std::to_string(BLOCK)}, {"impl", "auto"}};
        op.setup = [e, n, m] {
            auto f = make_frames(e->sz.h, e->sz.w, 2, 11);
            HCHECK(cudaMallocHost(&e->host_new, n));
            std::memcpy(e->host_new, f[1].data(), n);
            HCHECK(cudaMalloc(&e->dev_new, n));
            HCHECK(cudaMalloc(&e->dev_ref, n));
            HCHECK(cudaMemcpy(e->dev_ref, f[0].data(), n, cudaMemcpyHostToDevice));
            HCHECK(cudaMalloc(&e->dev_out, m * sizeof(float)));
            HCHECK(cudaMallocHost(&e->host_out, m * sizeof(float)));
        };
        harness::Launch h2d = [e, n](cudaStream_t st) {
            HCHECK(cudaMemcpyAsync(e->dev_new, e->host_new, n, cudaMemcpyHostToDevice, st));
        };
        harness::Launch kern = [e](cudaStream_t st) {
            k0_changemap_cuda(e->dev_new, e->dev_ref, e->sz.h, e->sz.w, C, BLOCK, e->dev_out, st);
        };
        harness::Launch d2h = [e, m](cudaStream_t st) {
            HCHECK(cudaMemcpyAsync(e->host_out, e->dev_out, m * sizeof(float), cudaMemcpyDeviceToHost, st));
        };
        op.phases = {{"h2d_frame", h2d}, {"kernel", kern}, {"d2h_map", d2h}};
        op.run = [=](cudaStream_t st) { h2d(st); kern(st); d2h(st); };
        op.check = [e, n, m] {
            CheckResult c;
            std::vector<uint8_t> ref_frame(n);
            HCHECK(cudaMemcpy(ref_frame.data(), e->dev_ref, n, cudaMemcpyDeviceToHost));
            std::vector<float> ref(m);
            k0_changemap_serial(e->host_new, ref_frame.data(), e->sz.h, e->sz.w, C, BLOCK, ref.data());
            HCHECK(cudaMemcpy(e->dev_new, e->host_new, n, cudaMemcpyHostToDevice));
            k0_changemap_cuda(e->dev_new, e->dev_ref, e->sz.h, e->sz.w, C, BLOCK, e->dev_out, 0);
            HCHECK(cudaMemcpy(e->host_out, e->dev_out, m * sizeof(float), cudaMemcpyDeviceToHost));
            c.ok = std::memcmp(ref.data(), e->host_out, m * sizeof(float)) == 0;
            c.note = c.ok ? "bit-identical to k0_changemap_serial" : "GPU map differs from the CPU reference";
            return c;
        };
        op.teardown = [e] {
            cudaFreeHost(e->host_new);
            cudaFreeHost(e->host_out);
            cudaFree(e->dev_new);
            cudaFree(e->dev_ref);
            cudaFree(e->dev_out);
        };
        run.add(op);
    }

    int bad = run.run_all();
    run.write_json(out_path);
    printf("%-22s %10s %10s %10s  %s\n", "op", "event us", "host us", "GB/s", "status");
    for (auto& s : run.results()) {
        auto ev = s.clocks.count("event_batch_ms") ? s.clocks.at("event_batch_ms").median * 1e3 : NAN;
        auto hs = s.clocks.count("host_sync_ms") ? s.clocks.at("host_sync_ms").median * 1e3 : NAN;
        printf("%-22s %10.2f %10.2f %10.1f  %s%s\n", s.op.name.c_str(), ev, hs, s.dram_gb_per_s, s.status.c_str(),
               s.plausibility.empty() ? "" : "  IMPLAUSIBLE");
        for (auto& [k, v] : s.clocks)
            if (k.rfind("phase_", 0) == 0) printf("    %-18s %10.2f us\n", k.c_str(), v.median * 1e3);
    }
    printf("wrote %s\n", out_path.c_str());
    return bad ? 1 : 0;
}
