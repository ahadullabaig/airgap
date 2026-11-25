// Correctness tests for the attention versions (versions.h). Exit code 0 means every check passed.
//
// Usage: attn_test [version ...]   (default: every version)
//
// For each version:
// 1. Hand-computed: two tokens, scores 1 and 0 or 0 and 0, so the outputs are (e*v0 + v1)/(e + 1) and
//    (v0 + v1)/2 in closed form.
// 2. The shared float64 reference (bench/attn_testdata.py, kernels/attention/build/testdata): every case
//    there (shapes, partial tiles, near one-hot rows, sampled rows at large N). Error per (head, row) is
//    ||o - o_ref|| / ||o_ref||; the test requires the maximum below TOL.
// 3. Invalid arguments are rejected.
// Versions with alternative internal paths (V2's softmax variants) run every case once per path that can
// handle it, so each path is tested on every shape, not only the one the dispatcher picks.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "attn.h"
#include "versions.h"

static int failures = 0;
static const double TOL = 1e-4;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++failures;                                      \
        }                                                    \
    } while (0)

#define CUDA_OK(x)                                                                              \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

// One way of running a version: returns the launch status. ws has the version's workspace size.
using Runner = std::function<int(const float*, const float*, const float*, float*, float*, int, int)>;

struct Variant {
    std::string name;
    Runner run;
    int max_n;  // the largest N this variant handles (paths limited by registers), 0 = any
};

static std::vector<Variant> variants(const AttnVersion& ver) {
    std::vector<Variant> out;
    out.push_back({ver.name, [&ver](const float* q, const float* k, const float* v, float* o, float* ws, int h,
                                    int n) { return ver.full(q, k, v, o, ws, h, n, 0); },
                   0});
    if (std::string(ver.name) == "v2") {
        for (int impl : {ATTN_V2_SOFTMAX_REG32, ATTN_V2_SOFTMAX_REG64, ATTN_V2_SOFTMAX_ONLINE}) {
            Runner r = [impl](const float* q, const float* k, const float* v, float* o, float* ws, int h, int n) {
                int rc = attn_v2_scores(q, k, ws, h, n, 0);
                if (!rc) rc = attn_v2_softmax_impl(ws, h, n, impl, 0);
                if (!rc) rc = attn_v2_output(ws, v, o, h, n, 0);
                return rc;
            };
            out.push_back({std::string("v2/softmax_") + attn_v2_softmax_name(impl), r, attn_v2_softmax_max_n(impl)});
        }
        for (int phase : {ATTN_V2_PHASE_SCORES, ATTN_V2_PHASE_OUTPUT})  // every tile configuration of the sweep
            for (int cfg = 0; cfg < attn_v2_config_count(phase); ++cfg) {
                if (cfg == attn_v2_config_default(phase)) continue;  // already covered by "v2"
                const int sc = phase == ATTN_V2_PHASE_SCORES ? cfg : attn_v2_config_default(ATTN_V2_PHASE_SCORES);
                const int oc = phase == ATTN_V2_PHASE_OUTPUT ? cfg : attn_v2_config_default(ATTN_V2_PHASE_OUTPUT);
                Runner r = [sc, oc](const float* q, const float* k, const float* v, float* o, float* ws, int h, int n) {
                    int rc = attn_v2_scores_cfg(q, k, ws, h, n, sc, 0);
                    if (!rc) rc = attn_v2_softmax(ws, h, n, 0);
                    if (!rc) rc = attn_v2_output_cfg(ws, v, o, h, n, oc, 0);
                    return rc;
                };
                out.push_back({std::string(phase == ATTN_V2_PHASE_SCORES ? "v2/scores_" : "v2/output_") +
                                   attn_v2_config_name(phase, cfg), r, 0});
            }
    }
    return out;
}

template <class T>
static std::vector<T> read_file(const std::string& path, size_t count) {
    std::vector<T> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T))) {
        std::printf("cannot read %s (run: .venv/bin/python -I bench/attn_testdata.py)\n", path.c_str());
        std::exit(2);
    }
    return v;
}

static int meta_int(const std::string& meta, const std::string& key) {
    size_t p = meta.find("\"" + key + "\"");
    return p == std::string::npos ? -1 : std::atoi(meta.c_str() + meta.find(':', p) + 1);
}

// Runs one variant on host Q, K, V ([H][N][D]); returns O.
static std::vector<float> run_on(const AttnVersion& ver, const Variant& var, const std::vector<float>& q,
                                 const std::vector<float>& k, const std::vector<float>& v, int heads, int n) {
    const size_t elems = size_t(heads) * n * ATTN_D;
    float *dq, *dk, *dv, *dout, *ws = nullptr;
    CUDA_OK(cudaMalloc(&dq, elems * 4));
    CUDA_OK(cudaMalloc(&dk, elems * 4));
    CUDA_OK(cudaMalloc(&dv, elems * 4));
    CUDA_OK(cudaMalloc(&dout, elems * 4));
    const size_t wsb = ver.workspace_bytes(heads, n);
    if (wsb) {
        CUDA_OK(cudaMalloc(&ws, wsb));
        CUDA_OK(cudaMemset(ws, 0xff, wsb));  // NaN pattern: a read of an unwritten score fails
    }
    CUDA_OK(cudaMemcpy(dq, q.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dk, k.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, v.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dout, 0xff, elems * 4));  // NaN pattern: unwritten outputs fail
    CHECK(var.run(dq, dk, dv, dout, ws, heads, n) == 0, "%s rc", var.name.c_str());
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> o(elems);
    CUDA_OK(cudaMemcpy(o.data(), dout, elems * 4, cudaMemcpyDeviceToHost));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(ws);
    return o;
}

static void test_hand_computed(const AttnVersion& ver, const Variant& var) {
    const int h = 1, n = 2, d = ATTN_D;
    std::vector<float> q(n * d, 0.f), k(n * d, 0.f), v(n * d, 0.f);
    q[0] = 8.f;           // q0 . k0 = 8, scaled by 1/8 -> score 1; q1 = 0 -> scores 0, 0
    k[0] = 1.f;           // k1 = 0
    v[0] = 1.f;  v[5] = -2.f;      // v0
    v[d] = 3.f;  v[d + 5] = 4.f;   // v1
    auto o = run_on(ver, var, q, k, v, h, n);
    const double e = std::exp(1.0);
    const double want[4] = {(e * 1 + 3) / (e + 1), (e * -2 + 4) / (e + 1), 2.0, 1.0};
    const float got[4] = {o[0], o[5], o[d], o[d + 5]};
    for (int t = 0; t < 4; ++t)
        CHECK(std::fabs(got[t] - want[t]) <= 1e-6 * std::fabs(want[t]), "%s hand-computed %d: got %.8f want %.8f",
              var.name.c_str(), t, got[t], want[t]);
    for (int c = 0; c < d; ++c)
        if (c != 0 && c != 5)
            CHECK(o[c] == 0.f && o[d + c] == 0.f, "%s hand-computed: component %d should be 0", var.name.c_str(), c);
}

static void test_case(const AttnVersion& ver, const Variant& var, const std::string& name) {
    const std::string dir = "build/testdata/" + name + "/";
    std::ifstream mf(dir + "meta.json");
    std::stringstream ms;
    ms << mf.rdbuf();
    const std::string meta = ms.str();
    const int heads = meta_int(meta, "heads"), n = meta_int(meta, "n"), nrows = meta_int(meta, "rows");
    if (heads <= 0 || n <= 0) {
        std::printf("cannot read %s (run: .venv/bin/python -I bench/attn_testdata.py)\n", (dir + "meta.json").c_str());
        std::exit(2);
    }
    if (var.max_n && n > var.max_n) return;
    const size_t elems = size_t(heads) * n * ATTN_D;
    auto q = read_file<float>(dir + "q.f32", elems), k = read_file<float>(dir + "k.f32", elems),
         v = read_file<float>(dir + "v.f32", elems);
    auto rows = read_file<int32_t>(dir + "rows.i32", nrows);
    auto oref = read_file<double>(dir + "oref.f64", size_t(heads) * nrows * ATTN_D);
    auto o = run_on(ver, var, q, k, v, heads, n);
    double worst = 0;
    int worst_h = -1, worst_r = -1;
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < nrows; ++r) {
            const float* got = &o[(size_t(h) * n + rows[r]) * ATTN_D];
            const double* want = &oref[(size_t(h) * nrows + r) * ATTN_D];
            double num = 0, den = 0;
            for (int c = 0; c < ATTN_D; ++c) {
                num += (got[c] - want[c]) * (got[c] - want[c]);
                den += want[c] * want[c];
            }
            const double err = std::isfinite(num) ? std::sqrt(num / den) : INFINITY;
            if (!(err <= worst)) worst = err, worst_h = h, worst_r = rows[r];
        }
    std::printf("  %-20s %-16s H=%-2d N=%-6d rows=%-5d max rel error %.2e (head %d, row %d)\n", var.name.c_str(),
                name.c_str(), heads, n, nrows, worst, worst_h, worst_r);
    CHECK(worst <= TOL, "%s %s: max rel error %.3e above %.0e", var.name.c_str(), name.c_str(), worst, TOL);
}

static void test_invalid_args(const AttnVersion& ver) {
    float* p = nullptr;
    CUDA_OK(cudaMalloc(&p, 4096));
    CHECK(ver.full(nullptr, p, p, p, p, 1, 16, 0) != 0, "%s: null q accepted", ver.name);
    CHECK(ver.full(p, p, p, p, p, 0, 16, 0) != 0, "%s: heads = 0 accepted", ver.name);
    CHECK(ver.full(p, p, p, p, p, 1, 0, 0) != 0, "%s: n = 0 accepted", ver.name);
    if (ver.scores)  // versions that store N x N: the workspace size must not wrap at N = 32768
        CHECK(ver.workspace_bytes(1, 32768) == (size_t)4 << 30, "%s: workspace size wraps at N = 32768", ver.name);
    cudaFree(p);
}

static std::vector<std::string> cases() {
    // every case directory written by bench/attn_testdata.py, in its listed order
    std::ifstream f("build/testdata/cases.txt");
    std::vector<std::string> out;
    for (std::string line; std::getline(f, line);)
        if (!line.empty()) out.push_back(line);
    if (out.empty()) {
        std::printf("cannot read build/testdata/cases.txt (run: .venv/bin/python -I bench/attn_testdata.py)\n");
        std::exit(2);
    }
    return out;
}

int main(int argc, char** argv) {
    std::vector<const AttnVersion*> vers;
    for (int i = 1; i < argc; ++i) {
        const AttnVersion* v = attn_version(argv[i]);
        if (!v) {
            std::printf("unknown version %s\n", argv[i]);
            return 2;
        }
        vers.push_back(v);
    }
    if (vers.empty())
        for (const auto& v : ATTN_VERSIONS) vers.push_back(&v);
    const auto names = cases();
    for (const AttnVersion* ver : vers) {
        for (const Variant& var : variants(*ver)) {
            test_hand_computed(*ver, var);
            for (const auto& name : names) test_case(*ver, var, name);
        }
        test_invalid_args(*ver);
    }
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed (");
    for (size_t i = 0; i < vers.size(); ++i) std::printf("%s%s", i ? ", " : "", vers[i]->label);
    std::printf(")\n");
    return 0;
}
