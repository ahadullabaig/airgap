// Correctness tests for V1. Exit code 0 means every check passed.
//
// 1. Hand-computed: two tokens, scores 1 and 0 or 0 and 0, so the outputs are (e*v0 + v1)/(e + 1) and
//    (v0 + v1)/2 in closed form.
// 2. The shared float64 reference (bench/attn_testdata.py, kernels/attention/build/testdata): the
//    encoder's shape (12 heads, N = 1024, every row), N not a multiple of 16, near one-hot rows (Q x40,
//    which overflows a softmax without max subtraction), and sampled rows at N = 4100 and 16384 that
//    include row 0, tile edges, the last row and every head. Error per (head, row) is
//    ||o - o_ref|| / ||o_ref||; the test requires the maximum below TOL.
// 3. Invalid arguments are rejected.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "attn.h"

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

// Runs V1 on host Q, K, V ([H][N][D]); returns O.
static std::vector<float> run_v1(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                                 int heads, int n) {
    const size_t elems = size_t(heads) * n * ATTN_D;
    float *dq, *dk, *dv, *dout, *ws;
    CUDA_OK(cudaMalloc(&dq, elems * 4));
    CUDA_OK(cudaMalloc(&dk, elems * 4));
    CUDA_OK(cudaMalloc(&dv, elems * 4));
    CUDA_OK(cudaMalloc(&dout, elems * 4));
    CUDA_OK(cudaMalloc(&ws, attn_v1_workspace_bytes(heads, n)));
    CUDA_OK(cudaMemcpy(dq, q.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dk, k.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, v.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dout, 0xff, elems * 4));  // NaN pattern: unwritten outputs fail
    CHECK(attn_v1(dq, dk, dv, dout, ws, heads, n, 0) == 0, "attn_v1 rc");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> o(elems);
    CUDA_OK(cudaMemcpy(o.data(), dout, elems * 4, cudaMemcpyDeviceToHost));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(ws);
    return o;
}

static void test_hand_computed() {
    const int h = 1, n = 2, d = ATTN_D;
    std::vector<float> q(n * d, 0.f), k(n * d, 0.f), v(n * d, 0.f);
    q[0] = 8.f;           // q0 . k0 = 8, scaled by 1/8 -> score 1; q1 = 0 -> scores 0, 0
    k[0] = 1.f;           // k1 = 0
    v[0] = 1.f;  v[5] = -2.f;      // v0
    v[d] = 3.f;  v[d + 5] = 4.f;   // v1
    auto o = run_v1(q, k, v, h, n);
    const double e = std::exp(1.0);
    const double want[4] = {(e * 1 + 3) / (e + 1), (e * -2 + 4) / (e + 1), 2.0, 1.0};
    const float got[4] = {o[0], o[5], o[d], o[d + 5]};
    for (int t = 0; t < 4; ++t)
        CHECK(std::fabs(got[t] - want[t]) <= 1e-6 * std::fabs(want[t]), "hand-computed %d: got %.8f want %.8f", t,
              got[t], want[t]);
    for (int c = 0; c < d; ++c)
        if (c != 0 && c != 5) CHECK(o[c] == 0.f && o[d + c] == 0.f, "hand-computed: component %d should be 0", c);
}

static void test_case(const std::string& name) {
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
    const size_t elems = size_t(heads) * n * ATTN_D;
    auto q = read_file<float>(dir + "q.f32", elems), k = read_file<float>(dir + "k.f32", elems),
         v = read_file<float>(dir + "v.f32", elems);
    auto rows = read_file<int32_t>(dir + "rows.i32", nrows);
    auto oref = read_file<double>(dir + "oref.f64", size_t(heads) * nrows * ATTN_D);
    auto o = run_v1(q, k, v, heads, n);
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
    std::printf("  %-16s H=%-2d N=%-6d rows=%-5d max rel error %.2e (head %d, row %d)\n", name.c_str(), heads, n,
                nrows, worst, worst_h, worst_r);
    CHECK(worst <= TOL, "%s: max rel error %.3e above %.0e", name.c_str(), worst, TOL);
}

static void test_invalid_args() {
    float* p = nullptr;
    CUDA_OK(cudaMalloc(&p, 4096));
    CHECK(attn_v1(nullptr, p, p, p, p, 1, 16, 0) != 0, "null q accepted");
    CHECK(attn_v1(p, p, p, p, p, 0, 16, 0) != 0, "heads = 0 accepted");
    CHECK(attn_v1(p, p, p, p, p, 1, 0, 0) != 0, "n = 0 accepted");
    CHECK(attn_v1_workspace_bytes(1, 32768) == (size_t)4 << 30, "workspace size must not wrap at N = 32768");
    cudaFree(p);
}

int main() {
    test_hand_computed();
    for (const char* name : {"rand_h12_n1024", "rand_h2_n1000", "peaky_h2_n256", "rand_h3_n4100", "rand_h1_n16384"})
        test_case(name);
    test_invalid_args();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all V1 checks passed\n");
    return 0;
}
