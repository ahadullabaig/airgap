// X2 (PLAN.md section 3.4, experiments): the fixed costs of attention at the encoder's shape, 12 heads,
// N = 1024, D = 64, on the shared harness. The versions' own times come from their committed benchmarks
// (results/attention/v1.json, v2.json, v3.json: each version in its own group, at its own steady clock);
// this binary measures what those times are compared against.
//
// Group "x2_launch": empty kernels launched with exactly the grid, block size and shared-memory request of
//   each version's kernels (V1 three, V2 three, V3 one, default tile shapes), back to back: the cost of
//   launching and scheduling that many blocks with no work in them. "empty_1block" is the bare launch
//   (cf. results/cuda/peak.json launch_b2b). event_batch_ms is back to back; host_sync_ms adds a
//   synchronise per call.
// Group "x2_transfers": copying Q, K and V to the device and O back (3.1 MiB each), from pinned and from
//   pageable host memory. In the encoder Q, K and V never leave the GPU, so these are a reference for an
//   attention call fed from the host, not part of the versions' times. Pageable copies block the host:
//   read host_sync_ms for them.
//
// Usage: x2_bench [output.json]
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "attn.h"
#include "harness.h"

using harness::Op;

namespace {

constexpr int H = 12, N = 1024, D = ATTN_D;

__global__ void empty_kernel() {}

struct Launch {
    dim3 grid, block;
    size_t smem;
};

// Launch configurations of the default kernels at H = 12, N = 1024 (v1.cu, v2.cu, v3.cu).
const std::vector<Launch> V1_LAUNCHES = {
    {dim3((N + 15) / 16, (N + 15) / 16, H), dim3(16, 16), 0},  // v1_scores
    {dim3((H * N + 7) / 8), dim3(256), 0},                       // v1_softmax
    {dim3((N + 3) / 4, H), dim3(D, 4), 0},                       // v1_output
};
const std::vector<Launch> V2_LAUNCHES = {
    {dim3((N + 63) / 64, (N + 63) / 64, H), dim3(128), 2 * 16 * (64 + 4) * 4},  // v2_gemm 64x64x16/8x4, scores
    {dim3((H * N + 7) / 8), dim3(256), 0},                                       // v2_softmax_reg<32>
    {dim3(1, (N + 63) / 64, H), dim3(128), 2 * 16 * (64 + 4) * 4},               // v2_gemm 64x64x16/8x4, output
};
const std::vector<Launch> V3_LAUNCHES = {
    {dim3((N + 127) / 128, H), dim3(256), size_t(D * 132 + D * 36 + 32 * 68 + 32 * 132) * 4},  // v3 128x32/256
};

void launch(const Launch& l, cudaStream_t st) { empty_kernel<<<l.grid, l.block, l.smem, st>>>(); }

std::string grid_str(const std::vector<Launch>& ls) {
    std::string s;
    for (const auto& l : ls) {
        if (!s.empty()) s += "; ";
        s += std::to_string(l.grid.x * l.grid.y * l.grid.z) + " blocks x " + std::to_string(l.block.x * l.block.y) +
             " threads, " + std::to_string(l.smem) + " B shared";
    }
    return s;
}

Op empty_op(const std::string& name, const std::string& version, const std::vector<Launch>& ls,
            const std::vector<std::string>& phase_names) {
    Op op;
    op.name = name;
    op.group = "x2_launch";
    op.params = {{"mimics", version}, {"launches", std::to_string(ls.size())}, {"config", grid_str(ls)},
                 {"heads", std::to_string(H)}, {"n", std::to_string(N)}};
    op.run = [ls](cudaStream_t st) {
        for (const auto& l : ls) launch(l, st);
    };
    for (size_t i = 0; i < phase_names.size(); ++i) {
        const Launch l = ls[i];
        op.phases.push_back({phase_names[i], [l](cudaStream_t st) { launch(l, st); }});
    }
    return op;
}

struct Copy {
    float *host = nullptr, *dev = nullptr;
    std::vector<float> pageable;
    bool pinned = false;
};

Op copy_op(const std::string& name, bool to_device, bool pinned, int tensors) {
    auto c = std::make_shared<Copy>();
    c->pinned = pinned;
    const size_t elems = size_t(H) * N * D, bytes = elems * 4;
    Op op;
    op.name = name;
    op.group = "x2_transfers";
    op.params = {{"direction", to_device ? "host to device" : "device to host"}, {"host_memory", pinned ? "pinned" : "pageable"},
                 {"tensors", std::to_string(tensors)}, {"bytes", std::to_string(bytes * tensors)},
                 {"what", to_device ? "Q, K, V" : "O"}};
    op.setup = [c, elems, tensors] {
        const size_t total = elems * tensors;
        if (c->pinned) {
            HCHECK(cudaMallocHost(&c->host, total * 4));
        } else {
            c->pageable.assign(total, 1.f);
            c->host = c->pageable.data();
        }
        for (size_t i = 0; i < total; ++i) c->host[i] = float(i % 1000);
        HCHECK(cudaMalloc(&c->dev, total * 4));
    };
    op.run = [c, bytes, tensors, to_device](cudaStream_t st) {
        for (int t = 0; t < tensors; ++t) {
            char* h = reinterpret_cast<char*>(c->host) + t * bytes;
            char* d = reinterpret_cast<char*>(c->dev) + t * bytes;
            if (to_device) cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, st);
            else cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, st);
        }
    };
    op.teardown = [c] {
        cudaFree(c->dev);
        if (c->pinned) cudaFreeHost(c->host);
        c->pageable.clear();
        c->pageable.shrink_to_fit();
        c->dev = c->host = nullptr;
    };
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "../../results/attention/x2.json";
    // V3's launch requests more than 48 KiB of shared memory, which needs the same opt-in as the real kernel
    HCHECK(cudaFuncSetAttribute(empty_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    harness::Runner run("attention_x2_bench", harness::read_limits("../../results/cuda/peak.json"));
    run.add(empty_op("empty_1block", "a bare launch", {{dim3(1), dim3(32), 0}}, {}));
    run.add(empty_op("empty_v1", "V1", V1_LAUNCHES, {"scores", "softmax", "output"}));
    run.add(empty_op("empty_v2", "V2", V2_LAUNCHES, {"scores", "softmax", "output"}));
    run.add(empty_op("empty_v3", "V3", V3_LAUNCHES, {}));
    run.add(copy_op("h2d_qkv_pinned", true, true, 3));
    run.add(copy_op("d2h_o_pinned", false, true, 1));
    run.add(copy_op("h2d_qkv_pageable", true, false, 3));
    run.add(copy_op("d2h_o_pageable", false, false, 1));
    run.run_all();
    run.write_json(out_path);
    printf("%-18s %-8s %12s %12s\n", "op", "status", "event us", "host us");
    for (auto& s : run.results()) {
        auto med = [&](const char* k) { return s.clocks.count(k) ? 1e3 * s.clocks.at(k).median : NAN; };
        printf("%-18s %-8s %12.2f %12.2f%s%s\n", s.op.name.c_str(), s.status.c_str(), med("event_batch_ms"),
               med("host_sync_ms"), s.error.empty() ? "" : "  ", s.error.substr(0, 80).c_str());
    }
    printf("wrote %s\n", out_path.c_str());
    return 0;
}
