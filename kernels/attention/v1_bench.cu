// V1 benchmark on the shared harness (kernels/harness).
//
// Group "h12_n1024": the encoder's shape, 12 heads, N = 1024, D = 64; the three launches are timed as
//   phases. Predicted (results/cuda/predicted_times.json): 1.98-4.86 ms in total; scores and output
//   0.72-2.16 ms each (untiled matmuls at 10-30% of peak), softmax about 0.54 ms (two passes over the
//   N x N matrix at DRAM bandwidth).
// Groups "x1_n<N>": one head, N from 128 to 32,768 (X1). Each size is its own group, so a failed
//   allocation (V1 stores N x N: 4 GiB at N = 32,768) is recorded without disturbing the others.
//
// Every op is checked before timing: sampled rows (row 0, tile edges, the middle, the last row, in every
// head) against a double-precision CPU reference, error ||o - o_ref|| / ||o_ref|| below 1e-4.
//
// Usage: v1_bench [output.json]
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "attn.h"
#include "harness.h"

using harness::CheckResult;
using harness::Op;

namespace {

constexpr int D = ATTN_D;

struct Problem {
    int heads = 0, n = 0;
    std::vector<float> q, k, v;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *ws = nullptr;
};

std::vector<int> check_rows(int n) {
    std::vector<int> r = {0, 1, 15, 16, n / 2, n - 17, n - 16, n - 1};
    std::vector<int> out;
    for (int x : r)
        if (x >= 0 && x < n && (out.empty() || out.back() != x)) out.push_back(x);
    return out;
}

CheckResult check(Problem& p) {
    CheckResult c;
    HCHECK(cudaMemset(p.dout, 0xff, size_t(p.heads) * p.n * D * 4));
    if (attn_v1(p.dq, p.dk, p.dv, p.dout, p.ws, p.heads, p.n, 0) != 0) {
        c.ok = false;
        c.note = "launch failed";
        return c;
    }
    HCHECK(cudaDeviceSynchronize());
    std::vector<float> o(size_t(p.heads) * p.n * D);
    HCHECK(cudaMemcpy(o.data(), p.dout, o.size() * 4, cudaMemcpyDeviceToHost));
    const double scale = 1.0 / std::sqrt(double(D));
    std::vector<double> s(p.n);
    for (int h = 0; h < p.heads; ++h)
        for (int i : check_rows(p.n)) {
            const float* qi = &p.q[(size_t(h) * p.n + i) * D];
            double m = -INFINITY;
            for (int j = 0; j < p.n; ++j) {
                const float* kj = &p.k[(size_t(h) * p.n + j) * D];
                double acc = 0;
                for (int t = 0; t < D; ++t) acc += double(qi[t]) * kj[t];
                s[j] = acc * scale;
                m = std::max(m, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < p.n; ++j) sum += (s[j] = std::exp(s[j] - m));
            double num = 0, den = 0;
            for (int t = 0; t < D; ++t) {
                double ref = 0;
                for (int j = 0; j < p.n; ++j) ref += s[j] / sum * p.v[(size_t(h) * p.n + j) * D + t];
                const double got = o[(size_t(h) * p.n + i) * D + t];
                num += (got - ref) * (got - ref);
                den += ref * ref;
            }
            const double err = std::isfinite(num) ? std::sqrt(num / den) : INFINITY;
            c.max_rel_err = std::max(c.max_rel_err, err);
        }
    c.ok = c.max_rel_err <= 1e-4;
    c.note = "sampled rows in every head against a double-precision CPU reference";
    return c;
}

Op make_op(const std::string& group, int heads, int n, const std::vector<std::pair<std::string, std::string>>& extra) {
    auto p = std::make_shared<Problem>();
    p->heads = heads;
    p->n = n;
    Op op;
    op.name = "v1_h" + std::to_string(heads) + "_n" + std::to_string(n);
    op.group = group;
    op.params = {{"version", "V1"}, {"heads", std::to_string(heads)}, {"n", std::to_string(n)},
                 {"d", std::to_string(D)}, {"dtype", "fp32"},
                 {"workspace_mib", std::to_string(attn_v1_workspace_bytes(heads, n) / 1048576.0)}};
    for (auto& e : extra) op.params.push_back(e);
    op.flops = 4.0 * heads * double(n) * n * D;
    op.setup = [p] {
        const size_t elems = size_t(p->heads) * p->n * D;
        std::mt19937 rng(17);
        std::normal_distribution<float> g(0.f, 1.f);
        for (auto* vec : {&p->q, &p->k, &p->v}) {
            vec->resize(elems);
            for (auto& x : *vec) x = g(rng);
        }
        HCHECK(cudaMalloc(&p->dq, elems * 4));
        HCHECK(cudaMalloc(&p->dk, elems * 4));
        HCHECK(cudaMalloc(&p->dv, elems * 4));
        HCHECK(cudaMalloc(&p->dout, elems * 4));
        HCHECK(cudaMalloc(&p->ws, attn_v1_workspace_bytes(p->heads, p->n)));
        HCHECK(cudaMemcpy(p->dq, p->q.data(), elems * 4, cudaMemcpyHostToDevice));
        HCHECK(cudaMemcpy(p->dk, p->k.data(), elems * 4, cudaMemcpyHostToDevice));
        HCHECK(cudaMemcpy(p->dv, p->v.data(), elems * 4, cudaMemcpyHostToDevice));
    };
    op.check = [p] { return check(*p); };
    harness::Launch scores = [p](cudaStream_t st) { attn_v1_scores(p->dq, p->dk, p->ws, p->heads, p->n, st); };
    harness::Launch softmax = [p](cudaStream_t st) { attn_v1_softmax(p->ws, p->heads, p->n, st); };
    harness::Launch output = [p](cudaStream_t st) { attn_v1_output(p->ws, p->dv, p->dout, p->heads, p->n, st); };
    op.phases = {{"scores", scores}, {"softmax", softmax}, {"output", output}};
    op.run = [=](cudaStream_t st) { scores(st); softmax(st); output(st); };
    op.teardown = [p] {
        cudaFree(p->dq); cudaFree(p->dk); cudaFree(p->dv); cudaFree(p->dout); cudaFree(p->ws);
        p->dq = p->dk = p->dv = p->dout = p->ws = nullptr;
    };
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "../../results/attention/v1.json";
    harness::Runner run("attention_v1_bench", harness::read_limits("../../results/cuda/peak.json"));
    run.add(make_op("h12_n1024", 12, 1024,
                    {{"predicted_ms", "1.98-4.86"}, {"predicted_scores_ms", "0.72-2.16"},
                     {"predicted_softmax_ms", "0.54"}, {"predicted_output_ms", "0.72-2.16"}}));
    for (int n : {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768})
        run.add(make_op("x1_n" + std::to_string(n), 1, n, {}));
    run.run_all();
    run.write_json(out_path);

    printf("%-16s %-8s %10s %10s %8s %9s  %s\n", "op", "status", "event ms", "host ms", "TFLOPS", "mem MiB", "phases (ms)");
    for (auto& s : run.results()) {
        auto med = [&](const char* k) { return s.clocks.count(k) ? s.clocks.at(k).median : NAN; };
        printf("%-16s %-8s %10.4f %10.4f %8.3f %9.1f  scores %.4f softmax %.4f output %.4f%s%s\n", s.op.name.c_str(),
               s.status.c_str(), med("event_batch_ms"), med("host_sync_ms"), s.tflops, s.mem_mib,
               med("phase_scores_ms"), med("phase_softmax_ms"), med("phase_output_ms"),
               s.error.empty() ? "" : "  ", s.error.substr(0, 80).c_str());
    }
    printf("wrote %s\n", out_path.c_str());
    return 0;
}
