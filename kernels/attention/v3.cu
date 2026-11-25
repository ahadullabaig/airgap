// V3: fused attention in one kernel with an online softmax; nothing of size N x N is stored. See attn.h for
// the layout and the algorithm.
#include <cmath>

#include "attn.h"

namespace {

constexpr int D = ATTN_D;
constexpr int PAD = 4;  // shared-memory row padding: rows stay 16-byte aligned for float4 reads

template <int BR, int BC>
constexpr size_t smem_bytes() {  // Qs [D][BR+PAD], Ks [D][BC+PAD], Vs [BC][D+PAD], Ps [BC][BR+PAD]
    return size_t(D * (BR + PAD) + D * (BC + PAD) + BC * (D + PAD) + BC * (BR + PAD)) * sizeof(float);
}

// BR query rows per block, BC keys per tile, NT threads. Each thread owns 4 query rows (row group tr) and,
// within the TC threads of that row group (contiguous lanes, so row reductions are warp shuffles), BC / TC
// keys of each score tile and D / TC columns of the output, both as float4 groups spaced so that consecutive
// lanes read consecutive 16-byte words of shared memory.
template <int BR, int BC, int NT>
__global__ void __launch_bounds__(NT)
v3_fused(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
         float* __restrict__ o, int n, float scale) {
    constexpr int TR = BR / 4, TC = NT / TR;
    constexpr int TNS = BC / TC, RNS = TNS / 4;  // keys per thread, float4 groups
    constexpr int TNO = D / TC, RNO = TNO / 4;   // output columns per thread, float4 groups
    static_assert(BR % 4 == 0 && NT % TR == 0 && TC <= 32 && (TC & (TC - 1)) == 0, "row groups inside a warp");
    static_assert(TNS % 4 == 0 && TNO % 4 == 0, "float4 groups");
    constexpr int LQ = BR + PAD, LK = BC + PAD, LV = D + PAD, LP = BR + PAD;
    extern __shared__ __align__(16) float smem[];
    float* Qs = smem;            // [D][LQ]  the block's Q rows, transposed
    float* Ks = Qs + D * LQ;     // [D][LK]  the key tile, transposed
    float* Vs = Ks + D * LK;     // [BC][LV] the value tile, as stored
    float* Ps = Vs + BC * LV;    // [BC][LP] the tile's probabilities, transposed

    const int tid = threadIdx.x, tc = tid % TC, tr = tid / TC;
    const int r0 = blockIdx.x * BR;
    const size_t head = size_t(blockIdx.y) * n * D;
    const float* qh = q + head;
    const float* kh = k + head;
    const float* vh = v + head;
    float* oh = o + head;

    for (int idx = tid; idx < BR * D; idx += NT) {  // coalesced along d; rows past n are zeros
        const int r = idx / D, c = idx % D;
        Qs[c * LQ + r] = (r0 + r < n) ? qh[size_t(r0 + r) * D + c] : 0.f;
    }

    float acc[4][TNO] = {};
    float m[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) m[i] = -INFINITY, l[i] = 0.f;

    for (int j0 = 0; j0 < n; j0 += BC) {
        __syncthreads();  // the previous tile's K, V and P are consumed (and Q is in place, the first time)
        for (int idx = tid; idx < BC * D; idx += NT) {
            const int r = idx / D, c = idx % D;
            const bool ok = j0 + r < n;
            Ks[c * LK + r] = ok ? kh[size_t(j0 + r) * D + c] : 0.f;
            Vs[r * LV + c] = ok ? vh[size_t(j0 + r) * D + c] : 0.f;
        }
        __syncthreads();

        float s[4][TNS] = {};
#pragma unroll 8
        for (int d = 0; d < D; ++d) {
            const float4 a = *reinterpret_cast<const float4*>(&Qs[d * LQ + tr * 4]);
            float b[TNS];
#pragma unroll
            for (int g = 0; g < RNS; ++g) {
                const float4 x = *reinterpret_cast<const float4*>(&Ks[d * LK + g * (BC / RNS) + tc * 4]);
                b[g * 4 + 0] = x.x; b[g * 4 + 1] = x.y; b[g * 4 + 2] = x.z; b[g * 4 + 3] = x.w;
            }
            const float av[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < TNS; ++j) s[i][j] = fmaf(av[i], b[j], s[i][j]);
        }

        // online softmax: per row, the tile's max and sum over the TC lanes of the row group
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < TNS; ++j) {
                const int key = j0 + (j / 4) * (BC / RNS) + tc * 4 + j % 4;
                s[i][j] = key < n ? s[i][j] * scale : -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
#pragma unroll
            for (int off = TC / 2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m[i], mx);  // finite: every tile has at least one key below n
            const float alpha = m[i] == -INFINITY ? 0.f : expf(m[i] - m_new);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < TNS; ++j) {
                s[i][j] = expf(s[i][j] - m_new);  // 0 for keys past n
                sum += s[i][j];
            }
#pragma unroll
            for (int off = TC / 2; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            l[i] = l[i] * alpha + sum;
            m[i] = m_new;
#pragma unroll
            for (int c = 0; c < TNO; ++c) acc[i][c] *= alpha;
        }
#pragma unroll
        for (int j = 0; j < TNS; ++j) {  // P transposed: one float4 of this thread's 4 rows per key
            const int key = (j / 4) * (BC / RNS) + tc * 4 + j % 4;
            *reinterpret_cast<float4*>(&Ps[key * LP + tr * 4]) = make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
        }
        __syncthreads();

#pragma unroll 4
        for (int jj = 0; jj < BC; ++jj) {  // O += P V for the tile
            const float4 p = *reinterpret_cast<const float4*>(&Ps[jj * LP + tr * 4]);
            float vv[TNO];
#pragma unroll
            for (int g = 0; g < RNO; ++g) {
                const float4 x = *reinterpret_cast<const float4*>(&Vs[jj * LV + g * (D / RNO) + tc * 4]);
                vv[g * 4 + 0] = x.x; vv[g * 4 + 1] = x.y; vv[g * 4 + 2] = x.z; vv[g * 4 + 3] = x.w;
            }
            const float pv[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int c = 0; c < TNO; ++c) acc[i][c] = fmaf(pv[i], vv[c], acc[i][c]);
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + tr * 4 + i;
        if (r >= n) continue;
        const float inv = 1.f / l[i];
#pragma unroll
        for (int g = 0; g < RNO; ++g) {
            const int c = g * (D / RNO) + tc * 4;
            *reinterpret_cast<float4*>(&oh[size_t(r) * D + c]) =
                make_float4(acc[i][g * 4 + 0] * inv, acc[i][g * 4 + 1] * inv, acc[i][g * 4 + 2] * inv,
                            acc[i][g * 4 + 3] * inv);
        }
    }
}

template <int BR, int BC, int NT>
int launch(const float* q, const float* k, const float* v, float* o, int heads, int n, cudaStream_t st) {
    constexpr size_t bytes = smem_bytes<BR, BC>();
    static bool configured = false;  // above 48 KiB of shared memory needs an explicit opt-in, once
    if (!configured) {
        if (cudaFuncSetAttribute(v3_fused<BR, BC, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) !=
            cudaSuccess)
            return 3;
        configured = true;
    }
    dim3 grid((n + BR - 1) / BR, heads);
    v3_fused<BR, BC, NT><<<grid, NT, bytes, st>>>(q, k, v, o, n, 1.0f / sqrtf(float(D)));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 3;
}

struct Cfg {
    const char* name;
    int (*fn)(const float*, const float*, const float*, float*, int, int, cudaStream_t);
};

// Swept at 12 heads, N = 1024 (make tune VERSION=v3 -> results/attention/v3_tune.json).
const Cfg CFGS[] = {
    {"64x64/128", launch<64, 64, 128>},    // 69.6 KB shared: 1 block per SM
    {"64x32/128", launch<64, 32, 128>},    // 44.0 KB: 2 blocks per SM
    {"64x64/256", launch<64, 64, 256>},    // 69.6 KB, 8 warps
    {"32x64/128", launch<32, 64, 128>},    // 53.2 KB
    {"32x32/64", launch<32, 32, 64>},      // 31.7 KB: 3 blocks per SM
    {"128x32/256", launch<128, 32, 256>},  // 68.6 KB, 8 warps, half the blocks
};
constexpr int N_CFGS = sizeof(CFGS) / sizeof(Cfg);
// Default: the sweep's pick (selection rule in attn_bench.cu, results/attention/v3_tune.json).
constexpr int DEFAULT_CFG = 5;  // 128x32/256 (v3_tune.json)

}  // namespace

extern "C" size_t attn_v3_workspace_bytes(int, int) { return 0; }

extern "C" int attn_v3_config_count(void) { return N_CFGS; }

extern "C" int attn_v3_config_default(void) { return DEFAULT_CFG; }

extern "C" const char* attn_v3_config_name(int cfg) { return cfg >= 0 && cfg < N_CFGS ? CFGS[cfg].name : ""; }

extern "C" int attn_v3_cfg(const float* q, const float* k, const float* v, float* o, int heads, int n, int cfg,
                           cudaStream_t stream) {
    if (!q || !k || !v || !o || heads <= 0 || n <= 0 || cfg < 0 || cfg >= N_CFGS) return 1;
    return CFGS[cfg].fn(q, k, v, o, heads, n, stream);
}

extern "C" int attn_v3(const float* q, const float* k, const float* v, float* o, float*, int heads, int n,
                       cudaStream_t stream) {
    return attn_v3_cfg(q, k, v, o, heads, n, DEFAULT_CFG, stream);
}
