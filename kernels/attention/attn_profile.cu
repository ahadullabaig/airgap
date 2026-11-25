// One attention version at 12 heads, N = 1024, run four times; `make profile VERSION=<v>` profiles the
// fourth run's kernels with Nsight Compute (the first three warm caches, clocks and the module).
//
// Usage: attn_profile <version>
#include <cstdio>
#include <vector>

#include "attn.h"
#include "versions.h"

int main(int argc, char** argv) {
    const AttnVersion* ver = argc > 1 ? attn_version(argv[1]) : nullptr;
    if (!ver) {
        fprintf(stderr, "usage: attn_profile <version>\n");
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
    for (int i = 0; i < 4; ++i) ver->full(q, k, v, o, ws, h, n, 0);
    cudaDeviceSynchronize();
    printf("done\n");
}
