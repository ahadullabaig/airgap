// One attention version at 12 heads, N = 1024, run four times; `make profile VERSION=<v>` profiles the
// fourth run's kernels with Nsight Compute (the first three warm caches, clocks and the module).
//
// Usage: attn_profile <version> [config ...]
//   v2 [scores_cfg output_cfg]   tile configurations of the two matmuls (default: the version's defaults)
//   v3 [cfg]                     tile configuration of the fused kernel
// The non-default configurations are the sweep's alternatives (v2_tune.json, v3_tune.json); profiling them
// measures what the shapes change (occupancy, bank conflicts) instead of inferring it.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "attn.h"
#include "versions.h"

int main(int argc, char** argv) {
    const AttnVersion* ver = argc > 1 ? attn_version(argv[1]) : nullptr;
    if (!ver) {
        fprintf(stderr, "usage: attn_profile <version> [config ...]\n");
        return 2;
    }
    const int h = 12, n = 1024;
    const size_t e = size_t(h) * n * ATTN_D;
    std::vector<float> host(e);
    for (size_t i = 0; i < e; ++i) host[i] = float((i * 2654435761u) % 1000) / 500.f - 1.f;
    float *q, *k, *v, *o, *ws = nullptr;
    cudaMalloc(&q, e * 4); cudaMalloc(&k, e * 4); cudaMalloc(&v, e * 4); cudaMalloc(&o, e * 4);
    if (ver->workspace_bytes(h, n)) cudaMalloc(&ws, ver->workspace_bytes(h, n));
    cudaMemcpy(q, host.data(), e * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(k, host.data(), e * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(v, host.data(), e * 4, cudaMemcpyHostToDevice);
    const bool v2 = std::strcmp(ver->name, "v2") == 0, v3 = std::strcmp(ver->name, "v3") == 0;
    const int sc = (v2 && argc > 2) ? std::atoi(argv[2]) : (v2 ? attn_v2_config_default(ATTN_V2_PHASE_SCORES) : 0);
    const int oc = (v2 && argc > 3) ? std::atoi(argv[3]) : (v2 ? attn_v2_config_default(ATTN_V2_PHASE_OUTPUT) : 0);
    const int c3 = (v3 && argc > 2) ? std::atoi(argv[2]) : (v3 ? attn_v3_config_default() : 0);
    for (int i = 0; i < 4; ++i) {
        int rc;
        if (v2) {
            rc = attn_v2_scores_cfg(q, k, ws, h, n, sc, 0);
            if (!rc) rc = attn_v2_softmax(ws, h, n, 0);
            if (!rc) rc = attn_v2_output_cfg(ws, v, o, h, n, oc, 0);
        } else if (v3) {
            rc = attn_v3_cfg(q, k, v, o, h, n, c3, 0);
        } else {
            rc = ver->full(q, k, v, o, ws, h, n, 0);
        }
        if (rc) {
            fprintf(stderr, "launch failed (%d)\n", rc);
            return 1;
        }
    }
    cudaDeviceSynchronize();
    printf("done\n");
}
