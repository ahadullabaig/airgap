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
 * V2 (tiled, three kernels, N x N stored). Same phases and workspace as V1; the two matrix multiplies are
 * tiled, and the softmax reads the matrix once instead of three times.
 *   scores   S = scale * Q K^T per head, a tiled matmul (v2.cu, v2_gemm): each thread block loads a BM x BK
 *            tile of Q and a BN x BK tile of K into shared memory (transposed, rows padded by 4), and each
 *            thread accumulates a TM x TN block of S in registers from float4 shared-memory reads, which do
 *            not conflict on banks. The transposing stores are conflict-free for BK = 8 and conflict 2-way
 *            for BK = 16, the chosen default (results/attention/v2_ncu.txt). Reduction length D = 64.
 *   softmax  one warp per row. Up to N = 2048 the row stays in registers (32 or 64 values per lane): one
 *            read of S and one write. Above that, a two-pass online version: one read for max and sum
 *            together, then one read and one write (V1 reads three times). This is a change from V1, so
 *            the phase timings, not the total, attribute V1-to-V2 gains to tiling or to the softmax.
 *   output   O = P V per head, the same tiled matmul with P as the left operand (reduction length N) and V
 *            as the right one, read without transposing.
 *   Tile shapes (BM, BN, BK, TM, TN) per phase were chosen by a timed sweep at 12 heads, N = 1024
 *   (results/attention/v2_tune.json); the defaults are in v2.cu.
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

/* V2: the same interface as V1 (the workspace is the same N x N matrix). */
size_t attn_v2_workspace_bytes(int heads, int n);
int attn_v2_scores(const float *q, const float *k, float *s, int heads, int n, cudaStream_t stream);
int attn_v2_softmax(float *s, int heads, int n, cudaStream_t stream);
int attn_v2_output(const float *p, const float *v, float *o, int heads, int n, cudaStream_t stream);
int attn_v2(const float *q, const float *k, const float *v, float *o, float *workspace, int heads, int n,
            cudaStream_t stream);

/* V2 internals, exposed for the tests (every softmax path on every shape) and the tile sweep. */
#define ATTN_V2_SOFTMAX_AUTO 0   /* what attn_v2_softmax does: REG32 up to N = 1024, REG64 up to 2048, ONLINE */
#define ATTN_V2_SOFTMAX_REG32 1  /* row in registers, 32 values per lane, N <= 1024 */
#define ATTN_V2_SOFTMAX_REG64 2  /* row in registers, 64 values per lane, N <= 2048 */
#define ATTN_V2_SOFTMAX_ONLINE 3 /* any N: max and sum in one pass, then normalise */
int attn_v2_softmax_impl(float *s, int heads, int n, int impl, cudaStream_t stream); /* 1 if impl can't take n */
int attn_v2_softmax_pick(int n);           /* the path ATTN_V2_SOFTMAX_AUTO takes */
const char *attn_v2_softmax_name(int impl);
int attn_v2_softmax_max_n(int impl);       /* largest N the path handles, 0 = any */
#define ATTN_V2_PHASE_SCORES 0
#define ATTN_V2_PHASE_OUTPUT 1
int attn_v2_config_count(int phase);
int attn_v2_config_default(int phase);
const char *attn_v2_config_name(int phase, int cfg); /* "BMxBNxBK/TMxTN" */
int attn_v2_scores_cfg(const float *q, const float *k, float *s, int heads, int n, int cfg, cudaStream_t stream);
int attn_v2_output_cfg(const float *p, const float *v, float *o, int heads, int n, int cfg, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* ATTN_H */
