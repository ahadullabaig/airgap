/* Attention kernels for Part B (PLAN.md section 3.4). FP32, head dimension D = 64.
 *
 * Layout (all versions): Q, K, V, O are [H][N][D] row-major FP32 in device memory, contiguous.
 * O[h][i][:] = sum_j softmax_j(scale * Q[h][i] . K[h][j]) V[h][j], scale = 1 / sqrt(D).
 * All index arithmetic is in size_t: H * N * N reaches 2^32 at N = 32,768 for one head.
 *
 * V1 (naive, three kernels, N x N stored). Workspace S: [H][N][N] FP32; softmax runs in place, so V1
 * needs one N x N matrix per head (4 * H * N^2 bytes) besides Q, K, V, O.
 *   scores   grid (ceil(N/16), ceil(N/16), H), block 16 x 16. Thread (tx, ty) computes
 *            S[h][i][j], i = 16 * blockIdx.y + ty, j = 16 * blockIdx.x + tx, as a length-64 dot product
 *            read straight from global memory (no shared-memory tiling; rows come through L1/L2).
 *            Writes are coalesced along j.
 *   softmax  one warp per row, 8 rows per 256-thread block. Lanes stride along the row (coalesced):
 *            pass 1 row max, pass 2 sum of exp(s - max), pass 3 writes exp(s - max) / sum in place.
 *   output   grid (ceil(N/4), H), block 64 x 4. Thread (tx, ty) computes O[h][i][c], c = tx,
 *            i = 4 * blockIdx.x + ty, looping over j: P[h][i][j] is the same for the whole row (broadcast),
 *            V[h][j][c] is read coalesced along c.
 *
 * Return value of the launch functions: 0 on success, 1 invalid arguments, 3 launch error.
 */
#ifndef ATTN_H
#define ATTN_H

#include <stddef.h>
#include <cuda_runtime.h>

#define ATTN_D 64

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes of the V1 workspace (the N x N score matrix for all heads). */
size_t attn_v1_workspace_bytes(int heads, int n);

/* The three V1 kernels, separately (for phase timing) and together. */
int attn_v1_scores(const float *q, const float *k, float *s, int heads, int n, cudaStream_t stream);
int attn_v1_softmax(float *s, int heads, int n, cudaStream_t stream);
int attn_v1_output(const float *p, const float *v, float *o, int heads, int n, cudaStream_t stream);
int attn_v1(const float *q, const float *k, const float *v, float *o, float *workspace, int heads, int n,
            cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* ATTN_H */
