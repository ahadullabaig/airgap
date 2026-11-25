// V2: attention as three kernels with the N x N matrix in device memory, as V1, but with tiled matrix
// multiplies and a softmax that reads the matrix once. See attn.h for the layout and the phases.
#include <cmath>

#include "attn.h"

namespace {

constexpr int D = ATTN_D;
constexpr int PAD = 4;  // shared-memory row padding: keeps rows 16-byte aligned; transposed stores are
                        // conflict-free for BK = 8, 2-way for BK = 16

// C[z][m][n] = alpha * sum_k A[z][m][k] * B(k, n), for z = blockIdx.z (one head per z).
//   A row-major, lda, batch stride sa.  B_T: B stored [n][k] (ldb), element (k, n) = B[n * ldb + k];
//   otherwise B stored [k][n], element (k, n) = B[k * ldb + n].  C row-major, ldc, batch stride sc.
// Block tile BM x BN, reduction chunk BK, each thread a TM x TN register tile (TM, TN in {4, 8}) made of
// float4 groups spaced BM / (TM / 4) rows and BN / (TN / 4) columns apart, so consecutive threads read
// consecutive 16-byte words of the B tile (no bank conflicts) and share the A word (broadcast).
// Out-of-range rows, columns and k are zero-filled on load and skipped on store.
template <int BM, int BN, int BK, int TM, int TN, bool B_T>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
v2_gemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K,
        int lda, int ldb, int ldc, size_t sa, size_t sb, size_t sc, float alpha) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int RM = TM / 4, RN = TN / 4;  // float4 groups per thread along m and n
    static_assert((BM * BK) % THREADS == 0 && (BN * BK) % THREADS == 0, "tile loads must divide evenly");
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BN / TN >= 8, "float4 groups, quarter-warp per A word");
    __shared__ __align__(16) float As[BK][BM + PAD];
    __shared__ __align__(16) float Bs[BK][BN + PAD];

    const int tid = threadIdx.x;
    const int tc = tid % (BN / TN), tr = tid / (BN / TN);
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    A += blockIdx.z * sa;
    B += blockIdx.z * sb;
    C += blockIdx.z * sc;

    float acc[TM][TN] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int t = 0; t < BM * BK / THREADS; ++t) {  // A tile, read along k, stored transposed
            const int idx = tid + t * THREADS, r = idx / BK, c = idx % BK;
            const int m = m0 + r, k = k0 + c;
            As[c][r] = (m < M && k < K) ? A[size_t(m) * lda + k] : 0.f;
        }
        if (B_T) {
#pragma unroll
            for (int t = 0; t < BN * BK / THREADS; ++t) {  // B stored [n][k]: read along k, store transposed
                const int idx = tid + t * THREADS, r = idx / BK, c = idx % BK;
                const int n = n0 + r, k = k0 + c;
                Bs[c][r] = (n < N && k < K) ? B[size_t(n) * ldb + k] : 0.f;
            }
        } else {
#pragma unroll
            for (int t = 0; t < BN * BK / THREADS; ++t) {  // B stored [k][n]: read along n, store as is
                const int idx = tid + t * THREADS, r = idx / BN, c = idx % BN;
                const int k = k0 + r, n = n0 + c;
                Bs[r][c] = (n < N && k < K) ? B[size_t(k) * ldb + n] : 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int g = 0; g < RM; ++g) {
                const float4 x = *reinterpret_cast<const float4*>(&As[kk][g * (BM / RM) + tr * 4]);
                a[g * 4 + 0] = x.x; a[g * 4 + 1] = x.y; a[g * 4 + 2] = x.z; a[g * 4 + 3] = x.w;
            }
#pragma unroll
            for (int g = 0; g < RN; ++g) {
                const float4 x = *reinterpret_cast<const float4*>(&Bs[kk][g * (BN / RN) + tc * 4]);
                b[g * 4 + 0] = x.x; b[g * 4 + 1] = x.y; b[g * 4 + 2] = x.z; b[g * 4 + 3] = x.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    const bool vec = (ldc % 4 == 0);  // float4 stores need 16-byte aligned rows of C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + (i / 4) * (BM / RM) + tr * 4 + i % 4;
        if (m >= M) continue;
        float* crow = C + size_t(m) * ldc;
#pragma unroll
        for (int g = 0; g < RN; ++g) {
            const int n = n0 + g * (BN / RN) + tc * 4;
            const float v0 = alpha * acc[i][g * 4 + 0], v1 = alpha * acc[i][g * 4 + 1];
            const float v2 = alpha * acc[i][g * 4 + 2], v3 = alpha * acc[i][g * 4 + 3];
            if (vec && n + 3 < N) {
                *reinterpret_cast<float4*>(crow + n) = make_float4(v0, v1, v2, v3);
            } else {
                if (n + 0 < N) crow[n + 0] = v0;
                if (n + 1 < N) crow[n + 1] = v1;
                if (n + 2 < N) crow[n + 2] = v2;
                if (n + 3 < N) crow[n + 3] = v3;
            }
        }
    }
}

// Softmax with the row in registers: R values per lane, N <= 32 * R. One read, one write.
template <int R>
__global__ void __launch_bounds__(256) v2_softmax_reg(float* __restrict__ s, int rows, int n) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    float* r = s + size_t(row) * n;
    float x[R];
    float m = -INFINITY;
#pragma unroll
    for (int t = 0; t < R; ++t) {
        const int j = lane + 32 * t;
        x[t] = j < n ? r[j] : -INFINITY;
        m = fmaxf(m, x[t]);
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < R; ++t) {
        x[t] = expf(x[t] - m);  // exp(-inf) = 0 for the padding
        sum += x[t];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const float inv = 1.f / sum;
#pragma unroll
    for (int t = 0; t < R; ++t) {
        const int j = lane + 32 * t;
        if (j < n) r[j] = x[t] * inv;
    }
}

// Merge two (max, sum of exp(x - max)) pairs; a lane that saw no values has (-inf, 0).
__device__ __forceinline__ void merge_ml(float& m, float& l, float mo, float lo) {
    const float mn = fmaxf(m, mo);
    if (mn == -INFINITY) return;
    l = (m == -INFINITY ? 0.f : l * expf(m - mn)) + (mo == -INFINITY ? 0.f : lo * expf(mo - mn));
    m = mn;
}

// Softmax for any N: max and sum in one read (online), then one read and one write.
__global__ void __launch_bounds__(256) v2_softmax_online(float* __restrict__ s, int rows, int n) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    float* r = s + size_t(row) * n;
    float m = -INFINITY, l = 0.f;
    for (int j = lane; j < n; j += 32) {
        const float x = r[j];
        if (x > m) {
            l = (m == -INFINITY ? 0.f : l * expf(m - x)) + 1.f;
            m = x;
        } else {
            l += expf(x - m);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float mo = __shfl_xor_sync(0xffffffffu, m, off), lo = __shfl_xor_sync(0xffffffffu, l, off);
        merge_ml(m, l, mo, lo);
    }
    const float inv = 1.f / l;
    for (int j = lane; j < n; j += 32) r[j] = expf(r[j] - m) * inv;
}

int launched() { return cudaPeekAtLastError() == cudaSuccess ? 0 : 3; }

// Tile configurations swept at 12 heads, N = 1024 (make tune -> results/attention/v2_tune.json).
// Scores: M = N = tokens, K = D.  Output: M = tokens, N = D = 64, K = tokens.
struct Cfg {
    const char* name;
    void (*scores)(const float*, const float*, float*, int, int, cudaStream_t);
    void (*output)(const float*, const float*, float*, int, int, cudaStream_t);
};

template <int BM, int BN, int BK, int TM, int TN>
void scores_launch(const float* q, const float* k, float* s, int heads, int n, cudaStream_t st) {
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM, heads), block((BM / TM) * (BN / TN));
    v2_gemm<BM, BN, BK, TM, TN, true><<<grid, block, 0, st>>>(q, k, s, n, n, D, D, D, n, size_t(n) * D,
                                                              size_t(n) * D, size_t(n) * n, 1.0f / sqrtf(float(D)));
}

template <int BM, int BN, int BK, int TM, int TN>
void output_launch(const float* p, const float* v, float* o, int heads, int n, cudaStream_t st) {
    static_assert(BN == D, "the output tile spans all D columns");
    dim3 grid(1, (n + BM - 1) / BM, heads), block((BM / TM) * (BN / TN));
    v2_gemm<BM, BN, BK, TM, TN, false><<<grid, block, 0, st>>>(p, v, o, n, D, n, n, D, D, size_t(n) * n,
                                                               size_t(n) * D, size_t(n) * D, 1.0f);
}

const Cfg SCORES_CFGS[] = {
    {"128x128x8/8x8", scores_launch<128, 128, 8, 8, 8>, nullptr},
    {"64x64x8/4x4", scores_launch<64, 64, 8, 4, 4>, nullptr},
    {"128x64x8/8x4", scores_launch<128, 64, 8, 8, 4>, nullptr},
    {"64x128x8/4x8", scores_launch<64, 128, 8, 4, 8>, nullptr},
    {"64x64x16/8x4", scores_launch<64, 64, 16, 8, 4>, nullptr},
};
const Cfg OUTPUT_CFGS[] = {
    {"64x64x8/4x8", nullptr, output_launch<64, 64, 8, 4, 8>},
    {"128x64x8/8x8", nullptr, output_launch<128, 64, 8, 8, 8>},
    {"64x64x16/4x4", nullptr, output_launch<64, 64, 16, 4, 4>},
    {"32x64x16/4x4", nullptr, output_launch<32, 64, 16, 4, 4>},
    {"64x64x16/8x4", nullptr, output_launch<64, 64, 16, 8, 4>},
};
constexpr int N_SCORES = sizeof(SCORES_CFGS) / sizeof(Cfg), N_OUTPUT = sizeof(OUTPUT_CFGS) / sizeof(Cfg);
// Defaults: the sweep's winners (selection rule in attn_bench.cu, results/attention/v2_tune.json).
constexpr int DEFAULT_SCORES = 4, DEFAULT_OUTPUT = 4;  // 64x64x16/8x4 for both (v2_tune.json)

}  // namespace

extern "C" size_t attn_v2_workspace_bytes(int heads, int n) { return size_t(heads) * size_t(n) * size_t(n) * 4; }

extern "C" int attn_v2_config_count(int phase) {
    return phase == ATTN_V2_PHASE_SCORES ? N_SCORES : phase == ATTN_V2_PHASE_OUTPUT ? N_OUTPUT : 0;
}

extern "C" int attn_v2_config_default(int phase) {
    return phase == ATTN_V2_PHASE_SCORES ? DEFAULT_SCORES : phase == ATTN_V2_PHASE_OUTPUT ? DEFAULT_OUTPUT : -1;
}

extern "C" const char* attn_v2_config_name(int phase, int cfg) {
    if (cfg < 0 || cfg >= attn_v2_config_count(phase)) return "";
    return phase == ATTN_V2_PHASE_SCORES ? SCORES_CFGS[cfg].name : OUTPUT_CFGS[cfg].name;
}

extern "C" int attn_v2_scores_cfg(const float* q, const float* k, float* s, int heads, int n, int cfg,
                                  cudaStream_t stream) {
    if (!q || !k || !s || heads <= 0 || n <= 0 || cfg < 0 || cfg >= N_SCORES) return 1;
    SCORES_CFGS[cfg].scores(q, k, s, heads, n, stream);
    return launched();
}

extern "C" int attn_v2_output_cfg(const float* p, const float* v, float* o, int heads, int n, int cfg,
                                  cudaStream_t stream) {
    if (!p || !v || !o || heads <= 0 || n <= 0 || cfg < 0 || cfg >= N_OUTPUT) return 1;
    OUTPUT_CFGS[cfg].output(p, v, o, heads, n, stream);
    return launched();
}

extern "C" int attn_v2_scores(const float* q, const float* k, float* s, int heads, int n, cudaStream_t stream) {
    return attn_v2_scores_cfg(q, k, s, heads, n, DEFAULT_SCORES, stream);
}

extern "C" int attn_v2_output(const float* p, const float* v, float* o, int heads, int n, cudaStream_t stream) {
    return attn_v2_output_cfg(p, v, o, heads, n, DEFAULT_OUTPUT, stream);
}

extern "C" int attn_v2_softmax_max_n(int impl) {
    return impl == ATTN_V2_SOFTMAX_REG32 ? 1024 : impl == ATTN_V2_SOFTMAX_REG64 ? 2048 : 0;
}

extern "C" int attn_v2_softmax_pick(int n) {
    return n <= 1024 ? ATTN_V2_SOFTMAX_REG32 : n <= 2048 ? ATTN_V2_SOFTMAX_REG64 : ATTN_V2_SOFTMAX_ONLINE;
}

extern "C" const char* attn_v2_softmax_name(int impl) {
    switch (impl) {
        case ATTN_V2_SOFTMAX_AUTO: return "auto";
        case ATTN_V2_SOFTMAX_REG32: return "reg32";
        case ATTN_V2_SOFTMAX_REG64: return "reg64";
        case ATTN_V2_SOFTMAX_ONLINE: return "online";
        default: return "";
    }
}

extern "C" int attn_v2_softmax_impl(float* s, int heads, int n, int impl, cudaStream_t stream) {
    if (!s || heads <= 0 || n <= 0) return 1;
    if (impl == ATTN_V2_SOFTMAX_AUTO) impl = attn_v2_softmax_pick(n);
    const int max_n = attn_v2_softmax_max_n(impl);
    if (max_n && n > max_n) return 1;
    const int rows = heads * n, blocks = (rows + 7) / 8;
    switch (impl) {
        case ATTN_V2_SOFTMAX_REG32: v2_softmax_reg<32><<<blocks, 256, 0, stream>>>(s, rows, n); break;
        case ATTN_V2_SOFTMAX_REG64: v2_softmax_reg<64><<<blocks, 256, 0, stream>>>(s, rows, n); break;
        case ATTN_V2_SOFTMAX_ONLINE: v2_softmax_online<<<blocks, 256, 0, stream>>>(s, rows, n); break;
        default: return 1;
    }
    return launched();
}

extern "C" int attn_v2_softmax(float* s, int heads, int n, cudaStream_t stream) {
    return attn_v2_softmax_impl(s, heads, n, ATTN_V2_SOFTMAX_AUTO, stream);
}

extern "C" int attn_v2(const float* q, const float* k, const float* v, float* o, float* workspace, int heads, int n,
                       cudaStream_t stream) {
    if (!q || !k || !v || !o || !workspace || heads <= 0 || n <= 0) return 1;
    int rc = attn_v2_scores(q, k, workspace, heads, n, stream);
    if (!rc) rc = attn_v2_softmax(workspace, heads, n, stream);
    if (!rc) rc = attn_v2_output(workspace, v, o, heads, n, stream);
    return rc;
}
