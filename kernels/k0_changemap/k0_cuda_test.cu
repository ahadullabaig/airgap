// Correctness tests for K0 on the GPU. Exit code 0 means every check passed.
//
// The GPU result must equal the CPU reference (k0_changemap_serial) bit for bit, for both kernels:
// 1. the hand-computed case from k0_test.cpp;
// 2. edge values: identical frames give 0, opposite extremes give 255;
// 3. random frames at the CPU test's sizes, channel counts and block sizes (partial edge blocks, a block
//    larger than the image, 1 x 1), symmetry K0(a, b) == K0(b, a), and frames at an odd device address,
//    which must fall back to the generic kernel;
// 4. a run of 200 random-sized cases;
// 5. dispatch and invalid arguments.
#include "k0.h"
#include "k0_cuda.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;

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

static size_t map_size(int h, int w, int block) {
    return size_t((h + block - 1) / block) * size_t((w + block - 1) / block);
}

// Runs the GPU kernel `impl` on host frames placed `offset` bytes into device buffers; returns 0 on success.
static int gpu_map(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int h, int w, int c, int block,
                   int impl, size_t offset, std::vector<float>& out) {
    const size_t n = a.size(), m = map_size(h, w, block);
    uint8_t *da, *db;
    float* dout;
    CUDA_OK(cudaMalloc(&da, n + offset));
    CUDA_OK(cudaMalloc(&db, n + offset));
    CUDA_OK(cudaMalloc(&dout, m * sizeof(float)));
    CUDA_OK(cudaMemcpy(da + offset, a.data(), n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db + offset, b.data(), n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dout, 0xff, m * sizeof(float)));  // NaN pattern: unwritten blocks fail the comparison
    int rc = k0_changemap_cuda_impl(da + offset, db + offset, h, w, c, block, dout, 0, impl);
    CUDA_OK(cudaDeviceSynchronize());
    out.resize(m);
    CUDA_OK(cudaMemcpy(out.data(), dout, m * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    return rc;
}

static void compare_all(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int h, int w, int c, int block,
                        const char* what) {
    std::vector<float> ref(map_size(h, w, block)), got;
    CHECK(k0_changemap_serial(a.data(), b.data(), h, w, c, block, ref.data()) == 0, "serial rc");
    const size_t bytes = ref.size() * sizeof(float);
    CHECK(gpu_map(a, b, h, w, c, block, K0_IMPL_GENERIC, 0, got) == 0, "generic rc %s", what);
    CHECK(std::memcmp(ref.data(), got.data(), bytes) == 0, "generic != serial: %s %dx%dx%d block %d", what, h, w, c, block);
    CHECK(gpu_map(a, b, h, w, c, block, K0_IMPL_AUTO, 0, got) == 0, "auto rc %s", what);
    CHECK(std::memcmp(ref.data(), got.data(), bytes) == 0, "auto != serial: %s %dx%dx%d block %d", what, h, w, c, block);
    if ((size_t(w) * c) % 4 == 0 && (size_t(block) * c) % 4 == 0) {
        CHECK(gpu_map(a, b, h, w, c, block, K0_IMPL_VEC4, 0, got) == 0, "vec4 rc %s", what);
        CHECK(std::memcmp(ref.data(), got.data(), bytes) == 0, "vec4 != serial: %s %dx%dx%d block %d", what, h, w, c, block);
    }
    // odd device address: auto must take the generic kernel and still agree; forced vec4 must refuse
    CHECK(gpu_map(a, b, h, w, c, block, K0_IMPL_AUTO, 1, got) == 0, "auto rc at odd address %s", what);
    CHECK(std::memcmp(ref.data(), got.data(), bytes) == 0, "auto at odd address != serial: %s", what);
    CHECK(gpu_map(a, b, h, w, c, block, K0_IMPL_VEC4, 1, got) == 2, "vec4 at odd address not refused: %s", what);
}

static void test_hand_computed() {
    const int h = 3, w = 5, c = 1, block = 2;
    std::vector<uint8_t> a(h * w, 0), b(h * w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) b[y * w + x] = uint8_t(10 * y + x);
    const float expected[6] = {5.5f, 7.5f, 9.0f, 20.5f, 22.5f, 24.0f};
    std::vector<float> got;
    CHECK(gpu_map(a, b, h, w, c, block, K0_IMPL_AUTO, 0, got) == 0, "hand-computed rc");
    for (int i = 0; i < 6; ++i) CHECK(got[i] == expected[i], "hand-computed block %d: got %f want %f", i, got[i], expected[i]);
}

static void test_extremes() {
    const int h = 64, w = 48, c = 3, block = 16;
    const size_t n = size_t(h) * w * c;
    std::vector<uint8_t> zero(n, 0), full(n, 255);
    std::vector<float> got;
    for (int impl : {K0_IMPL_GENERIC, K0_IMPL_VEC4}) {
        gpu_map(zero, zero, h, w, c, block, impl, 0, got);
        for (float v : got) CHECK(v == 0.0f, "identical frames (impl %d): %f", impl, v);
        gpu_map(zero, full, h, w, c, block, impl, 0, got);
        for (float v : got) CHECK(v == 255.0f, "opposite frames (impl %d): %f", impl, v);
    }
}

static void test_random() {
    struct Case { int h, w, c, block; };
    const Case cases[] = {
        {512, 512, 3, 16}, {720, 1280, 3, 16}, {1080, 1920, 3, 16}, {37, 53, 1, 8},
        {17, 16, 4, 16},   {5, 5, 3, 32},      {1, 1, 1, 1},
    };
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (const Case& k : cases) {
        const size_t n = size_t(k.h) * k.w * k.c;
        std::vector<uint8_t> a(n), b(n);
        for (auto& v : a) v = uint8_t(dist(rng));
        for (auto& v : b) v = uint8_t(dist(rng));
        compare_all(a, b, k.h, k.w, k.c, k.block, "fixed case");
        std::vector<float> ab, ba;
        gpu_map(a, b, k.h, k.w, k.c, k.block, K0_IMPL_AUTO, 0, ab);
        gpu_map(b, a, k.h, k.w, k.c, k.block, K0_IMPL_AUTO, 0, ba);
        CHECK(std::memcmp(ab.data(), ba.data(), ab.size() * sizeof(float)) == 0, "not symmetric %dx%d", k.h, k.w);
    }
    // random shapes: small differences (like real frames) and full-range noise
    std::uniform_int_distribution<int> side(1, 300), chan(1, 4), blk(1, 40), mode(0, 1);
    for (int t = 0; t < 200; ++t) {
        const int h = side(rng), w = side(rng), c = chan(rng), block = blk(rng);
        const size_t n = size_t(h) * w * c;
        std::vector<uint8_t> a(n), b(n);
        const bool small = mode(rng);
        for (size_t i = 0; i < n; ++i) {
            a[i] = uint8_t(dist(rng));
            const int d = small ? (dist(rng) % 7) - 3 : dist(rng) - a[i];
            b[i] = uint8_t(std::min(255, std::max(0, int(a[i]) + d)));
        }
        compare_all(a, b, h, w, c, block, small ? "random shape, small differences" : "random shape, noise");
    }
}

static void test_dispatch_and_invalid() {
    uint8_t* d;
    float* o;
    CUDA_OK(cudaMalloc(&d, 64));
    CUDA_OK(cudaMalloc(&o, 64));
    CHECK(k0_cuda_uses_vec4(d, d, 512, 3, 16) == 1, "vec4 should apply to 512x512x3, block 16");
    CHECK(k0_cuda_uses_vec4(d, d, 53, 1, 8) == 0, "vec4 must not apply when w*c is not a multiple of 4");
    CHECK(k0_cuda_uses_vec4(d + 1, d, 512, 3, 16) == 0, "vec4 must not apply at an odd address");
    CHECK(k0_changemap_cuda(nullptr, d, 1, 1, 3, 16, o, 0) != 0, "null a accepted");
    CHECK(k0_changemap_cuda(d, d, 0, 1, 3, 16, o, 0) != 0, "h = 0 accepted");
    CHECK(k0_changemap_cuda(d, d, 1, 1, 3, 0, o, 0) != 0, "block = 0 accepted");
    CHECK(k0_changemap_cuda(d, d, 1, 1, 3, 16, nullptr, 0) != 0, "null out accepted");
    CHECK(k0_changemap_cuda(d, d, 1, 1, 4, 4096, o, 0) != 0, "a block too large for 32-bit sums accepted");
    cudaFree(d);
    cudaFree(o);
}

int main() {
    test_hand_computed();
    test_extremes();
    test_random();
    test_dispatch_and_invalid();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all K0 GPU checks passed\n");
    return 0;
}
