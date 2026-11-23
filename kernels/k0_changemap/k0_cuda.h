/* K0 on the GPU. Same contract as k0.h (bit-identical results), with device pointers and an
 * asynchronous launch on the given stream.
 *
 * d_a, d_b : two frames in device memory, h x w pixels, c interleaved 8-bit channels (HWC), contiguous.
 * d_out    : ceil(h / block) x ceil(w / block) floats in device memory.
 *
 * Return value: 0 on success; 1 invalid arguments (including 255 * block^2 * c not fitting 32 bits);
 * 2 vec4 forced on a layout it cannot handle; 3 launch error.
 */
#ifndef K0_CUDA_H
#define K0_CUDA_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { K0_IMPL_AUTO = 0, K0_IMPL_GENERIC = 1, K0_IMPL_VEC4 = 2 };

int k0_changemap_cuda(const uint8_t *d_a, const uint8_t *d_b, int h, int w, int c, int block, float *d_out,
                      cudaStream_t stream);

/* For tests and benchmarks: force a kernel. */
int k0_changemap_cuda_impl(const uint8_t *d_a, const uint8_t *d_b, int h, int w, int c, int block, float *d_out,
                           cudaStream_t stream, int impl);

/* 1 if the vec4 kernel applies to this layout (4-byte aligned frames, w*c and block*c multiples of 4). */
int k0_cuda_uses_vec4(const uint8_t *a, const uint8_t *b, int w, int c, int block);
int k0_cuda_valid_args(const void *a, const void *b, int h, int w, int c, int block, const void *out);

#ifdef __cplusplus
}
#endif

#endif /* K0_CUDA_H */
