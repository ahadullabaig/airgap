// K0 with OpenMP: block rows are split across threads. Within a block row,
// each pixel row is swept once from left to right and its segments are added
// into per-block integer sums, so memory is read in order.
#include "k0.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

extern "C" int k0_changemap_omp(const uint8_t *a, const uint8_t *b, int h, int w, int c, int block, float *out) {
    if (!a || !b || !out || h <= 0 || w <= 0 || c <= 0 || block <= 0) return 1;
    const int by = (h + block - 1) / block;
    const int bx = (w + block - 1) / block;
    const size_t row_bytes = static_cast<size_t>(w) * c;
    const size_t seg_bytes = static_cast<size_t>(block) * c;

#pragma omp parallel
    {
        std::vector<uint64_t> sums(static_cast<size_t>(bx));
#pragma omp for schedule(static)
        for (int j = 0; j < by; ++j) {
            std::fill(sums.begin(), sums.end(), 0);
            const int y0 = j * block, y1 = std::min(y0 + block, h);
            for (int y = y0; y < y1; ++y) {
                const uint8_t *pa = a + static_cast<size_t>(y) * row_bytes;
                const uint8_t *pb = b + static_cast<size_t>(y) * row_bytes;
                for (int i = 0; i < bx; ++i) {
                    const size_t k0 = static_cast<size_t>(i) * seg_bytes;
                    const size_t k1 = std::min(k0 + seg_bytes, row_bytes);
                    uint64_t s = 0;
                    for (size_t k = k0; k < k1; ++k)
                        s += static_cast<uint64_t>(std::abs(static_cast<int>(pa[k]) - static_cast<int>(pb[k])));
                    sums[static_cast<size_t>(i)] += s;
                }
            }
            for (int i = 0; i < bx; ++i) {
                const int x0 = i * block, x1 = std::min(x0 + block, w);
                const uint64_t n = static_cast<uint64_t>(y1 - y0) * static_cast<uint64_t>(x1 - x0) * c;
                out[static_cast<size_t>(j) * bx + i] =
                    static_cast<float>(static_cast<double>(sums[static_cast<size_t>(i)]) / static_cast<double>(n));
            }
        }
    }
    return 0;
}

extern "C" void k0_set_num_threads(int n) {
    if (n > 0) omp_set_num_threads(n);
}

extern "C" int k0_get_max_threads(void) { return omp_get_max_threads(); }
