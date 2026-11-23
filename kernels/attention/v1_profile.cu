// V1 at 12 heads, N = 1024, run four times; `make profile` profiles the second run's three kernels with ncu.
#include <cstdio>
#include <vector>
#include "attn.h"
int main() {
    const int h = 12, n = 1024; const size_t e = size_t(h) * n * ATTN_D;
    std::vector<float> host(e); for (size_t i = 0; i < e; ++i) host[i] = float((i * 2654435761u) % 1000) / 500.f - 1.f;
    float *q, *k, *v, *o, *ws;
    cudaMalloc(&q, e * 4); cudaMalloc(&k, e * 4); cudaMalloc(&v, e * 4); cudaMalloc(&o, e * 4);
    cudaMalloc(&ws, attn_v1_workspace_bytes(h, n));
    cudaMemcpy(q, host.data(), e * 4, cudaMemcpyHostToDevice); cudaMemcpy(k, host.data(), e * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(v, host.data(), e * 4, cudaMemcpyHostToDevice);
    for (int i = 0; i < 4; ++i) attn_v1(q, k, v, o, ws, h, n, 0);
    cudaDeviceSynchronize(); printf("done\n");
}
