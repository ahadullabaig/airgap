/* K0: block change map between two frames.
 *
 * a, b : two images of h x w pixels with c interleaved 8-bit channels,
 *        row-major (HWC) and contiguous.
 * out  : ceil(h / block) x ceil(w / block) floats, row-major. Each value is the
 *        mean absolute difference over the block's pixels and channels, in
 *        [0, 255]. Edge blocks average over the pixels they actually contain.
 *
 * All versions accumulate in integers per block and divide once, so they
 * return bit-identical results.
 *
 * Return value: 0 on success, nonzero on invalid arguments.
 */
#ifndef K0_H
#define K0_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reference: one block at a time, in plain loops. */
int k0_changemap_serial(const uint8_t *a, const uint8_t *b, int h, int w, int c, int block, float *out);

/* OpenMP: block rows split across threads; each thread sweeps its pixel rows
 * once, adding row segments into per-block sums (cache-friendly order). */
int k0_changemap_omp(const uint8_t *a, const uint8_t *b, int h, int w, int c, int block, float *out);

/* Thread control for the OpenMP version (wraps omp_set_num_threads). */
void k0_set_num_threads(int n);
int k0_get_max_threads(void);

#ifdef __cplusplus
}
#endif

#endif /* K0_H */
