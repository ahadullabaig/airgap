// Correctness tests for K0. Exit code 0 means every check passed.
//
// 1. A hand-computed case (catches a wrong reference, not just disagreement).
// 2. Edge values: identical frames give 0, opposite extremes give 255.
// 3. Random frames: OpenMP equals the reference bit for bit, at several
//    sizes (including partial edge blocks), channel counts, block sizes and
//    thread counts; and K0(a, b) == K0(b, a).
// 4. Invalid arguments are rejected.
#include "k0.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...)                          \
    do {                                          \
        if (!(cond)) {                            \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);             \
            std::printf("\n");                    \
            ++failures;                           \
        }                                         \
    } while (0)

static size_t map_size(int h, int w, int block) {
    return static_cast<size_t>((h + block - 1) / block) * static_cast<size_t>((w + block - 1) / block);
}

static void test_hand_computed() {
    // 3 x 5 single-channel image, block 2. a = 0, b[y][x] = 10y + x.
    const int h = 3, w = 5, c = 1, block = 2;
    std::vector<uint8_t> a(h * w, 0), b(h * w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) b[y * w + x] = static_cast<uint8_t>(10 * y + x);
    // Blocks (row, col): (0,0) {0,1,10,11} (0,1) {2,3,12,13} (0,2) {4,14}
    //                    (1,0) {20,21}     (1,1) {22,23}     (1,2) {24}
    const float expected[6] = {5.5f, 7.5f, 9.0f, 20.5f, 22.5f, 24.0f};
    float s[6], o[6];
    CHECK(k0_changemap_serial(a.data(), b.data(), h, w, c, block, s) == 0, "serial rc");
    CHECK(k0_changemap_omp(a.data(), b.data(), h, w, c, block, o) == 0, "omp rc");
    for (int i = 0; i < 6; ++i) {
        CHECK(s[i] == expected[i], "serial block %d: got %f want %f", i, s[i], expected[i]);
        CHECK(o[i] == expected[i], "omp block %d: got %f want %f", i, o[i], expected[i]);
    }
}

static void test_extremes() {
    const int h = 64, w = 48, c = 3, block = 16;
    const size_t n = static_cast<size_t>(h) * w * c, m = map_size(h, w, block);
    std::vector<uint8_t> zero(n, 0), full(n, 255);
    std::vector<float> out(m);
    k0_changemap_omp(zero.data(), zero.data(), h, w, c, block, out.data());
    for (size_t i = 0; i < m; ++i) CHECK(out[i] == 0.0f, "identical frames: block %zu = %f", i, out[i]);
    k0_changemap_serial(zero.data(), full.data(), h, w, c, block, out.data());
    for (size_t i = 0; i < m; ++i) CHECK(out[i] == 255.0f, "opposite frames (serial): block %zu = %f", i, out[i]);
    k0_changemap_omp(full.data(), zero.data(), h, w, c, block, out.data());
    for (size_t i = 0; i < m; ++i) CHECK(out[i] == 255.0f, "opposite frames (omp): block %zu = %f", i, out[i]);
}

static void test_random_agreement() {
    struct Case { int h, w, c, block; };
    const Case cases[] = {
        {512, 512, 3, 16},   // model view
        {720, 1280, 3, 16},  // 720p, exact blocks
        {1080, 1920, 3, 16}, // 1080p: partial bottom row of blocks
        {37, 53, 1, 8},      // partial blocks on both edges
        {17, 16, 4, 16},     // one partial block row
        {5, 5, 3, 32},       // block larger than the image
        {1, 1, 1, 1},
    };
    const int thread_counts[] = {1, 3, 16};
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (const Case &k : cases) {
        const size_t n = static_cast<size_t>(k.h) * k.w * k.c, m = map_size(k.h, k.w, k.block);
        std::vector<uint8_t> a(n), b(n);
        for (auto &v : a) v = static_cast<uint8_t>(dist(rng));
        for (auto &v : b) v = static_cast<uint8_t>(dist(rng));
        std::vector<float> ref(m), got(m), swapped(m);
        CHECK(k0_changemap_serial(a.data(), b.data(), k.h, k.w, k.c, k.block, ref.data()) == 0, "serial rc");
        for (int t : thread_counts) {
            k0_set_num_threads(t);
            std::fill(got.begin(), got.end(), -1.0f);
            CHECK(k0_changemap_omp(a.data(), b.data(), k.h, k.w, k.c, k.block, got.data()) == 0, "omp rc");
            CHECK(std::memcmp(ref.data(), got.data(), m * sizeof(float)) == 0,
                  "omp != serial for %dx%dx%d block %d, %d threads", k.h, k.w, k.c, k.block, t);
        }
        k0_changemap_omp(b.data(), a.data(), k.h, k.w, k.c, k.block, swapped.data());
        CHECK(std::memcmp(ref.data(), swapped.data(), m * sizeof(float)) == 0, "not symmetric for %dx%d", k.h, k.w);
    }
}

static void test_invalid_args() {
    uint8_t px[3] = {0, 0, 0};
    float out[1];
    CHECK(k0_changemap_serial(nullptr, px, 1, 1, 3, 16, out) != 0, "null a accepted");
    CHECK(k0_changemap_omp(px, px, 0, 1, 3, 16, out) != 0, "h = 0 accepted");
    CHECK(k0_changemap_omp(px, px, 1, 1, 3, 0, out) != 0, "block = 0 accepted");
    CHECK(k0_changemap_serial(px, px, 1, 1, 3, 16, nullptr) != 0, "null out accepted");
}

int main() {
    test_hand_computed();
    test_extremes();
    test_random_agreement();
    test_invalid_args();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all K0 checks passed\n");
    return 0;
}
