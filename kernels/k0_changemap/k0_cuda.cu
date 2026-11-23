// K0 on the GPU: one warp per output block. Same contract as k0.h: integer sum of absolute differences
// per block, one double division, cast to float, so the result is bit-identical to the CPU reference
// (CUDA's double division is correctly rounded).
//
// Two kernels:
//   vec4     4-byte loads and __vsadu4 (sum of absolute differences of 4 byte pairs). Needs 4-byte aligned
//            frames, a row length (w * c) and a block segment (block * c) that are multiples of 4: true for
//            the encoder's 512 x 512 x 3 frames with 16 x 16 blocks.
//   generic  byte loads; any size, channel count, block size and alignment.
// k0_changemap_cuda picks vec4 when it applies.
#include "k0_cuda.h"

#include <cstdint>

namespace {

constexpr int WARPS_PER_CTA = 4;

__device__ __forceinline__ uint32_t warp_sum(uint32_t v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ void write_mean(float* out, int idx, uint32_t sum, int rows, int cols, int c) {
    const double n = double(rows) * double(cols) * double(c);
    out[idx] = float(double(sum) / n);
}

__global__ void k0_generic(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b, int h, int w, int c,
                           int block, int bx, int nblocks, float* __restrict__ out) {
    const int warp = blockIdx.x * WARPS_PER_CTA + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= nblocks) return;
    const int j = warp / bx, i = warp % bx;
    const int y0 = j * block, x0 = i * block;
    const int rows = min(block, h - y0), cols = min(block, w - x0);
    const int seg = cols * c;  // bytes per block row
    const size_t row_bytes = size_t(w) * c;
    const size_t base = size_t(y0) * row_bytes + size_t(x0) * c;
    uint32_t sum = 0;
    for (int k = lane; k < rows * seg; k += 32) {
        const int r = k / seg, q = k % seg;
        const size_t idx = base + size_t(r) * row_bytes + q;
        const int d = int(a[idx]) - int(b[idx]);
        sum += uint32_t(d < 0 ? -d : d);
    }
    sum = warp_sum(sum);
    if (lane == 0) write_mean(out, warp, sum, rows, cols, c);
}

__global__ void k0_vec4(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, int h, int w, int c,
                        int block, int bx, int nblocks, float* __restrict__ out) {
    const int warp = blockIdx.x * WARPS_PER_CTA + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= nblocks) return;
    const int j = warp / bx, i = warp % bx;
    const int y0 = j * block, x0 = i * block;
    const int rows = min(block, h - y0), cols = min(block, w - x0);
    const int seg4 = cols * c / 4;  // words per block row (cols * c is a multiple of 4: see dispatch)
    const size_t row4 = size_t(w) * c / 4;
    const size_t base = size_t(y0) * row4 + size_t(x0) * c / 4;
    uint32_t sum = 0;
    for (int k = lane; k < rows * seg4; k += 32) {
        const int r = k / seg4, q = k % seg4;
        const size_t idx = base + size_t(r) * row4 + q;
        sum += __vsadu4(__ldg(a + idx), __ldg(b + idx));
    }
    sum = warp_sum(sum);
    if (lane == 0) write_mean(out, warp, sum, rows, cols, c);
}

}  // namespace

extern "C" int k0_cuda_valid_args(const void* a, const void* b, int h, int w, int c, int block, const void* out) {
    if (!a || !b || !out || h <= 0 || w <= 0 || c <= 0 || block <= 0) return 0;
    // 32-bit per-block sum: 255 * block^2 * c must fit
    if (double(block) * block * c * 255.0 >= 4294967296.0) return 0;
    return 1;
}

extern "C" int k0_cuda_uses_vec4(const uint8_t* a, const uint8_t* b, int w, int c, int block) {
    const bool aligned = (reinterpret_cast<uintptr_t>(a) % 4 == 0) && (reinterpret_cast<uintptr_t>(b) % 4 == 0);
    // every block's segment and every edge segment must be whole words
    return aligned && (size_t(w) * c) % 4 == 0 && (size_t(block) * c) % 4 == 0;
}

extern "C" int k0_changemap_cuda_impl(const uint8_t* d_a, const uint8_t* d_b, int h, int w, int c, int block,
                                      float* d_out, cudaStream_t stream, int impl) {
    if (!k0_cuda_valid_args(d_a, d_b, h, w, c, block, d_out)) return 1;
    const int by = (h + block - 1) / block, bx = (w + block - 1) / block, nblocks = by * bx;
    const int ctas = (nblocks + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    bool vec4 = impl == K0_IMPL_AUTO ? k0_cuda_uses_vec4(d_a, d_b, w, c, block) : impl == K0_IMPL_VEC4;
    if (vec4 && !k0_cuda_uses_vec4(d_a, d_b, w, c, block)) return 2;  // forced vec4 on a layout it cannot handle
    if (vec4)
        k0_vec4<<<ctas, WARPS_PER_CTA * 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(d_a),
                                                          reinterpret_cast<const uint32_t*>(d_b), h, w, c, block, bx,
                                                          nblocks, d_out);
    else
        k0_generic<<<ctas, WARPS_PER_CTA * 32, 0, stream>>>(d_a, d_b, h, w, c, block, bx, nblocks, d_out);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int k0_changemap_cuda(const uint8_t* d_a, const uint8_t* d_b, int h, int w, int c, int block, float* d_out,
                                 cudaStream_t stream) {
    return k0_changemap_cuda_impl(d_a, d_b, h, w, c, block, d_out, stream, K0_IMPL_AUTO);
}
