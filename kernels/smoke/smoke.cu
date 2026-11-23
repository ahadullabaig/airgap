// Toolchain smoke test: compile, run, check and time a trivial kernel.
// Also times an empty kernel launch, the fixed-cost floor used in X2.
// Writes JSON to stdout.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void empty_kernel() {}

int main() {
    const int n = 1 << 24;  // 16M floats, 64 MiB per array
    std::vector<float> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; ++i) { ha[i] = float(i % 1000); hb[i] = 0.5f * float(i % 7); }
    float *a, *b, *c;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMalloc(&b, n * sizeof(float)));
    CHECK(cudaMalloc(&c, n * sizeof(float)));
    CHECK(cudaMemcpy(a, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256, blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(a, b, c, n);  // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    const int reps = 50;
    CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r) vec_add<<<blocks, threads>>>(a, b, c, n);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms_add = 0;
    CHECK(cudaEventElapsedTime(&ms_add, t0, t1));
    ms_add /= reps;

    CHECK(cudaMemcpy(hc.data(), c, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) if (hc[i] != ha[i] + hb[i]) ++bad;

    for (int r = 0; r < 100; ++r) empty_kernel<<<1, 1>>>();  // warm-up
    CHECK(cudaDeviceSynchronize());
    const int ereps = 10000;
    CHECK(cudaEventRecord(t0));
    for (int r = 0; r < ereps; ++r) empty_kernel<<<1, 1>>>();
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms_empty = 0;
    CHECK(cudaEventElapsedTime(&ms_empty, t0, t1));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int rt = 0, drv = 0;
    CHECK(cudaRuntimeGetVersion(&rt));
    CHECK(cudaDriverGetVersion(&drv));
    double gbps = 3.0 * n * sizeof(float) / (ms_add * 1e-3) / 1e9;
    printf("{\n \"gpu\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"runtime\": %d, \"driver_api\": %d,\n"
           " \"vec_add\": {\"n\": %d, \"ms\": %.4f, \"effective_gb_per_s\": %.1f, \"mismatches\": %d},\n"
           " \"empty_launch_us\": %.3f\n}\n",
           p.name, p.major, p.minor, p.multiProcessorCount, rt, drv, n, ms_add, gbps, bad,
           ms_empty * 1000.0 / ereps);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return bad ? 1 : 0;
}
