// Attention benchmark on the shared harness (kernels/harness), for any version in versions.h.
//
// Usage: attn_bench <version> [output.json] [--tune]
//
// Group "h12_n1024": the encoder's shape, 12 heads, N = 1024, D = 64; for versions with phases, each
//   phase is timed separately. Predicted times (results/cuda/predicted_times.json, per phase from the
//   same model) are written into the op's params before anything is timed.
// Groups "x1_n<N>": one head, N from 128 to 32,768 (X1). Each size is its own group, so a failed
//   allocation (the N x N workspace is 4 GiB at N = 32,768) is recorded without disturbing the others.
// --tune (V2 only): group "tune_h12_n1024" times every tile configuration of the scores and output
//   phases (each op runs that phase alone, on the encoder's shape) and the softmax paths that handle
//   N = 1024. Selection rule, fixed before the sweep: per phase, the configuration with the lowest median
//   event time among those whose full attention passes the check; it becomes the default in v2.cu.
//
// Every op is checked before timing: sampled rows (row 0, V1's 16-wide and V2's 32/64/128-wide tile edges,
// the middle, the last rows, in every head) against a double-precision CPU reference, error
// ||o - o_ref|| / ||o_ref|| below 1e-4.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "attn.h"
#include "harness.h"
#include "versions.h"

using harness::CheckResult;
using harness::Op;

namespace {

constexpr int D = ATTN_D;

struct Problem {
    int heads = 0, n = 0;
    std::vector<float> q, k, v;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *ws = nullptr;
};

// Runs a whole attention on the problem's buffers (the version's default path or a tuning variant).
using Full = std::function<int(Problem&, cudaStream_t)>;

std::vector<int> check_rows(int n) {
    std::vector<int> r = {0, 1, 15, 16, 31, 32, 63, 64, 127, 128, n / 2, n - 129, n - 128, n - 65, n - 64,
                          n - 17, n - 16, n - 1};
    std::vector<int> out;
    for (int x : r)
        if (x >= 0 && x < n) out.push_back(x);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

CheckResult check(Problem& p, const Full& full) {
    CheckResult c;
    HCHECK(cudaMemset(p.dout, 0xff, size_t(p.heads) * p.n * D * 4));
    if (full(p, 0) != 0) {
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

// Setup and teardown shared by every op: random Q, K, V (seed 17) and the version's workspace.
void add_buffers(Op& op, const std::shared_ptr<Problem>& p, const AttnVersion& ver) {
    op.setup = [p, &ver] {
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
        if (ver.workspace_bytes(p->heads, p->n)) HCHECK(cudaMalloc(&p->ws, ver.workspace_bytes(p->heads, p->n)));
        HCHECK(cudaMemcpy(p->dq, p->q.data(), elems * 4, cudaMemcpyHostToDevice));
        HCHECK(cudaMemcpy(p->dk, p->k.data(), elems * 4, cudaMemcpyHostToDevice));
        HCHECK(cudaMemcpy(p->dv, p->v.data(), elems * 4, cudaMemcpyHostToDevice));
    };
    op.teardown = [p] {
        cudaFree(p->dq); cudaFree(p->dk); cudaFree(p->dv); cudaFree(p->dout); cudaFree(p->ws);
        p->dq = p->dk = p->dv = p->dout = p->ws = nullptr;
    };
}

Op make_op(const AttnVersion& ver, const std::string& group, int heads, int n,
           const std::vector<std::pair<std::string, std::string>>& extra) {
    auto p = std::make_shared<Problem>();
    p->heads = heads;
    p->n = n;
    Op op;
    op.name = std::string(ver.name) + "_h" + std::to_string(heads) + "_n" + std::to_string(n);
    op.group = group;
    op.params = {{"version", ver.label}, {"heads", std::to_string(heads)}, {"n", std::to_string(n)},
                 {"d", std::to_string(D)}, {"dtype", "fp32"},
                 {"workspace_mib", std::to_string(ver.workspace_bytes(heads, n) / 1048576.0)}};
    if (std::strcmp(ver.name, "v2") == 0) {
        op.params.push_back({"softmax_path", attn_v2_softmax_name(attn_v2_softmax_pick(n))});
        op.params.push_back({"scores_config", attn_v2_config_name(0, attn_v2_config_default(0))});
        op.params.push_back({"output_config", attn_v2_config_name(1, attn_v2_config_default(1))});
    }
    for (auto& e : extra) op.params.push_back(e);
    op.flops = 4.0 * heads * double(n) * n * D;
    add_buffers(op, p, ver);
    const AttnVersion* vp = &ver;
    op.check = [p, vp] {
        return check(*p, [vp](Problem& q, cudaStream_t st) {
            return vp->full(q.dq, q.dk, q.dv, q.dout, q.ws, q.heads, q.n, st);
        });
    };
    if (ver.scores) {
        harness::Launch scores = [p, vp](cudaStream_t st) { vp->scores(p->dq, p->dk, p->ws, p->heads, p->n, st); };
        harness::Launch softmax = [p, vp](cudaStream_t st) { vp->softmax(p->ws, p->heads, p->n, st); };
        harness::Launch output = [p, vp](cudaStream_t st) { vp->output(p->ws, p->dv, p->dout, p->heads, p->n, st); };
        op.phases = {{"scores", scores}, {"softmax", softmax}, {"output", output}};
        op.run = [=](cudaStream_t st) { scores(st); softmax(st); output(st); };
    } else {
        op.run = [p, vp](cudaStream_t st) { vp->full(p->dq, p->dk, p->dv, p->dout, p->ws, p->heads, p->n, st); };
    }
    return op;
}

// V2 tile sweep: one op per configuration of one phase, timing that phase alone at 12 heads, N = 1024.
Op make_tune_op(const AttnVersion& ver, int phase, int cfg, int softmax_impl) {
    auto p = std::make_shared<Problem>();
    p->heads = 12;
    p->n = 1024;
    Op op;
    op.group = "tune_h12_n1024";
    const std::string rule = "per phase, the lowest median event time among configurations whose full attention "
                             "passes the check";
    int sc = attn_v2_config_default(ATTN_V2_PHASE_SCORES), oc = attn_v2_config_default(ATTN_V2_PHASE_OUTPUT);
    int sm = ATTN_V2_SOFTMAX_AUTO;
    if (phase == ATTN_V2_PHASE_SCORES) {
        sc = cfg;
        op.name = "v2tune_scores_" + std::to_string(cfg);
        op.params = {{"phase", "scores"}, {"config", attn_v2_config_name(phase, cfg)}};
        op.flops = 2.0 * p->heads * double(p->n) * p->n * D;
    } else if (phase == ATTN_V2_PHASE_OUTPUT) {
        oc = cfg;
        op.name = "v2tune_output_" + std::to_string(cfg);
        op.params = {{"phase", "output"}, {"config", attn_v2_config_name(phase, cfg)}};
        op.flops = 2.0 * p->heads * double(p->n) * p->n * D;
    } else {
        sm = softmax_impl;
        op.name = std::string("v2tune_softmax_") + attn_v2_softmax_name(softmax_impl);
        op.params = {{"phase", "softmax"}, {"config", attn_v2_softmax_name(softmax_impl)}};
        op.dram_bytes = 2.0 * p->heads * double(p->n) * p->n * 4;  // one read and one write if cached
    }
    op.params.push_back({"heads", "12"});
    op.params.push_back({"n", "1024"});
    op.params.push_back({"selection_rule", rule});
    add_buffers(op, p, ver);
    op.check = [p, sc, oc, sm] {
        return check(*p, [sc, oc, sm](Problem& q, cudaStream_t st) {
            int rc = attn_v2_scores_cfg(q.dq, q.dk, q.ws, q.heads, q.n, sc, st);
            if (!rc) rc = attn_v2_softmax_impl(q.ws, q.heads, q.n, sm, st);
            if (!rc) rc = attn_v2_output_cfg(q.ws, q.dv, q.dout, q.heads, q.n, oc, st);
            return rc;
        });
    };
    // The check leaves a valid P in the workspace, so the output phase reads real probabilities.
    if (phase == ATTN_V2_PHASE_SCORES)
        op.run = [p, sc](cudaStream_t st) { attn_v2_scores_cfg(p->dq, p->dk, p->ws, p->heads, p->n, sc, st); };
    else if (phase == ATTN_V2_PHASE_OUTPUT)
        op.run = [p, oc](cudaStream_t st) { attn_v2_output_cfg(p->ws, p->dv, p->dout, p->heads, p->n, oc, st); };
    else  // softmax in place: rerunning it on its own output keeps valid, finite rows of the same shape
        op.run = [p, sm](cudaStream_t st) { attn_v2_softmax_impl(p->ws, p->heads, p->n, sm, st); };
    return op;
}

std::vector<std::pair<std::string, std::string>> predictions(const AttnVersion& ver) {
    if (std::strcmp(ver.name, "v1") == 0)
        return {{"predicted_ms", "1.98-4.86"}, {"predicted_scores_ms", "0.72-2.16"},
                {"predicted_softmax_ms", "0.54"}, {"predicted_output_ms", "0.72-2.16"}};
    if (std::strcmp(ver.name, "v2") == 0)
        return {{"predicted_ms", "1.16-1.62"}, {"predicted_scores_ms", "0.31-0.54"},
                {"predicted_softmax_ms", "0.54"}, {"predicted_output_ms", "0.31-0.54"}};
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !attn_version(argv[1])) {
        fprintf(stderr, "usage: attn_bench <version> [output.json] [--tune]\n");
        return 2;
    }
    const AttnVersion& ver = *attn_version(argv[1]);
    bool tune = false;
    std::string out_path = std::string("../../results/attention/") + ver.name + ".json";
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tune") == 0) tune = true;
        else out_path = argv[i];
    }
    if (tune && std::strcmp(ver.name, "v2") != 0) {
        fprintf(stderr, "--tune is for v2 only\n");
        return 2;
    }
    harness::Runner run(std::string("attention_") + ver.name + (tune ? "_tune" : "_bench"),
                        harness::read_limits("../../results/cuda/peak.json"));
    if (tune) {
        for (int c = 0; c < attn_v2_config_count(ATTN_V2_PHASE_SCORES); ++c)
            run.add(make_tune_op(ver, ATTN_V2_PHASE_SCORES, c, 0));
        for (int c = 0; c < attn_v2_config_count(ATTN_V2_PHASE_OUTPUT); ++c)
            run.add(make_tune_op(ver, ATTN_V2_PHASE_OUTPUT, c, 0));
        for (int impl : {ATTN_V2_SOFTMAX_REG32, ATTN_V2_SOFTMAX_REG64, ATTN_V2_SOFTMAX_ONLINE})
            run.add(make_tune_op(ver, -1, 0, impl));
    } else {
        run.add(make_op(ver, "h12_n1024", 12, 1024, predictions(ver)));
        for (int n : {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768})
            run.add(make_op(ver, "x1_n" + std::to_string(n), 1, n, {}));
    }
    run.run_all();
    run.write_json(out_path);

    if (tune) {
        printf("%-20s %-8s %-16s %10s %8s\n", "op", "status", "config", "event ms", "TFLOPS");
        for (auto& s : run.results()) {
            std::string cfg;
            for (auto& kv : s.op.params)
                if (kv.first == "config") cfg = kv.second;
            const double ms = s.clocks.count("event_batch_ms") ? s.clocks.at("event_batch_ms").median : NAN;
            printf("%-20s %-8s %-16s %10.4f %8.3f\n", s.op.name.c_str(), s.status.c_str(), cfg.c_str(), ms, s.tflops);
        }
    } else {
        printf("%-16s %-8s %10s %10s %8s %9s  %s\n", "op", "status", "event ms", "host ms", "TFLOPS", "mem MiB",
               "phases (ms)");
        for (auto& s : run.results()) {
            auto med = [&](const char* k) { return s.clocks.count(k) ? s.clocks.at(k).median : NAN; };
            printf("%-16s %-8s %10.4f %10.4f %8.3f %9.1f  scores %.4f softmax %.4f output %.4f%s%s\n",
                   s.op.name.c_str(), s.status.c_str(), med("event_batch_ms"), med("host_sync_ms"), s.tflops, s.mem_mib,
                   med("phase_scores_ms"), med("phase_softmax_ms"), med("phase_output_ms"),
                   s.error.empty() ? "" : "  ", s.error.substr(0, 80).c_str());
        }
    }
    printf("wrote %s\n", out_path.c_str());
    return 0;
}
