// K0 reference implementation: one block at a time.
#include "k0.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>

extern "C" int k0_changemap_serial(const uint8_t *a, const uint8_t *b, int h, int w, int c, int block, float *out) {
    if (!a || !b || !out || h <= 0 || w <= 0 || c <= 0 || block <= 0) return 1;
    const int by = (h + block - 1) / block;
    const int bx = (w + block - 1) / block;
    const size_t row_bytes = static_cast<size_t>(w) * c;

    for (int j = 0; j < by; ++j) {
        const int y0 = j * block, y1 = std::min(y0 + block, h);
        for (int i = 0; i < bx; ++i) {
            const int x0 = i * block, x1 = std::min(x0 + block, w);
            uint64_t sum = 0;
            for (int y = y0; y < y1; ++y) {
                const size_t row = static_cast<size_t>(y) * row_bytes;
                for (size_t k = row + static_cast<size_t>(x0) * c; k < row + static_cast<size_t>(x1) * c; ++k)
                    sum += static_cast<uint64_t>(std::abs(static_cast<int>(a[k]) - static_cast<int>(b[k])));
            }
            const uint64_t n = static_cast<uint64_t>(y1 - y0) * static_cast<uint64_t>(x1 - x0) * c;
            out[static_cast<size_t>(j) * bx + i] = static_cast<float>(static_cast<double>(sum) / static_cast<double>(n));
        }
    }
    return 0;
}
