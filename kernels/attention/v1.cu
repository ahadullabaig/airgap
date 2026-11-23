// V1: naive attention as three kernels with the N x N matrix in device memory. See attn.h for the
// layout and the thread mapping of each kernel.
#include <cmath>

#include "attn.h"

namespace {

constexpr int D = ATTN_D;

__global__ void v1_scores(const float* __restrict__ q, const float* __restrict__ k, float* __restrict__ s, int n,
                          float scale) {
    const int j = blockIdx.x * 16 + threadIdx.x;
    const int i = blockIdx.y * 16 + threadIdx.y;
    const int h = blockIdx.z;
    if (i >= n || j >= n) return;
    const float* qi = q + (size_t(h) * n + i) * D;
    const float* kj = k + (size_t(h) * n + j) * D;
    float acc = 0.f;
#pragma unroll 16
    for (int c = 0; c < D; ++c) acc = fmaf(qi[c], kj[c], acc);
    s[(size_t(h) * n + i) * n + j] = acc * scale;
}

__global__ void v1_softmax(float* __restrict__ s, int rows, int n) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    float* r = s + size_t(row) * n;
    float m = -INFINITY;
    for (int j = lane; j < n; j += 32) m = fmaxf(m, r[j]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float sum = 0.f;
    for (int j = lane; j < n; j += 32) sum += expf(r[j] - m);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const float inv = 1.f / sum;
    for (int j = lane; j < n; j += 32) r[j] = expf(r[j] - m) * inv;
}

__global__ void v1_output(const float* __restrict__ p, const float* __restrict__ v, float* __restrict__ o, int n) {
    const int c = threadIdx.x;
    const int i = blockIdx.x * 4 + threadIdx.y;
    const int h = blockIdx.y;
    if (i >= n) return;
    const float* pi = p + (size_t(h) * n + i) * n;
    const float* vh = v + size_t(h) * n * D;
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc = fmaf(pi[j], vh[size_t(j) * D + c], acc);
    o[(size_t(h) * n + i) * D + c] = acc;
}

int launched() { return cudaPeekAtLastError() == cudaSuccess ? 0 : 3; }

}  // namespace

extern "C" size_t attn_v1_workspace_bytes(int heads, int n) { return size_t(heads) * size_t(n) * size_t(n) * 4; }

extern "C" int attn_v1_scores(const float* q, const float* k, float* s, int heads, int n, cudaStream_t stream) {
    if (!q || !k || !s || heads <= 0 || n <= 0) return 1;
    dim3 grid((n + 15) / 16, (n + 15) / 16, heads), block(16, 16);
    v1_scores<<<grid, block, 0, stream>>>(q, k, s, n, 1.0f / sqrtf(float(D)));
    return launched();
}

extern "C" int attn_v1_softmax(float* s, int heads, int n, cudaStream_t stream) {
    if (!s || heads <= 0 || n <= 0) return 1;
    const int rows = heads * n;
    v1_softmax<<<(rows + 7) / 8, 256, 0, stream>>>(s, rows, n);
    return launched();
}

extern "C" int attn_v1_output(const float* p, const float* v, float* o, int heads, int n, cudaStream_t stream) {
    if (!p || !v || !o || heads <= 0 || n <= 0) return 1;
    dim3 grid((n + 3) / 4, heads), block(D, 4);
    v1_output<<<grid, block, 0, stream>>>(p, v, o, n);
    return launched();
}

extern "C" int attn_v1(const float* q, const float* k, const float* v, float* o, float* workspace, int heads, int n,
                       cudaStream_t stream) {
    int rc = attn_v1_scores(q, k, workspace, heads, n, stream);
    if (!rc) rc = attn_v1_softmax(workspace, heads, n, stream);
    if (!rc) rc = attn_v1_output(workspace, v, o, heads, n, stream);
    return rc;
}
