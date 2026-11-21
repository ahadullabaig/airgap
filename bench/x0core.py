"""Shared pieces of X0: the encoder's view of a frame and the reference-mode gate.

Reference mode: for each (stride, threshold) a reference image holds every
16x16 block as it was when that block was last refreshed. Each sampled frame
is compared with the reference by K0; blocks whose mean absolute difference
exceeds the threshold are refreshed (copied from the frame) and counted as
kept. The first sample of a stream initialises the reference and is not
counted. This is what a gate pays: slow drift accumulates until it crosses
the threshold.
"""

import numpy as np

BLOCK = 16


def model_view(ip, img, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """The 512x512 RGB image the encoder sees, as C-contiguous HWC uint8.

    The SmolVLM2 processor first resizes to 2048 on the long edge, then squares to
    512x512 when not splitting (checked: a 640x640 ERA frame becomes 2048x2048,
    then 512x512), which is why it costs about 40 ms per frame on the CPU."""
    pv = ip(images=[[img]], do_image_splitting=False, return_tensors="np")["pixel_values"]
    x = np.asarray(pv)[0, 0].astype(np.float32)  # CHW, normalised
    x = (x * std[:, None, None] + mean[:, None, None]) * 255.0
    x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x.transpose(1, 2, 0))


def refresh_blocks(ref: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> None:
    """Copy the masked blocks of frame into ref, in place."""
    h, w, c = ref.shape
    rv = ref.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, c).transpose(0, 2, 1, 3, 4)
    fv = frame.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, c).transpose(0, 2, 1, 3, 4)
    rv[mask] = fv[mask]


class RefGate:
    """Reference-mode keep-rates for several strides and thresholds over one stream.

    Feed frames in order with push(i, frame). Frame i is sampled at stride s when
    i % s == 0. keep[(s, thr)] lists the share of blocks refreshed at each counted
    sample, and index[s] the frame index of each counted sample."""

    def __init__(self, k0, strides, thresholds):
        self.k0 = k0
        self.strides = list(strides)
        self.thresholds = list(thresholds)
        self.refs: dict[tuple[int, float], np.ndarray] = {}
        self.keep = {(s, t): [] for s in self.strides for t in self.thresholds}
        self.index = {s: [] for s in self.strides}
        self._refresh_checked = False

    def push(self, i: int, f: np.ndarray) -> None:
        for s in self.strides:
            if i % s:
                continue
            if (s, self.thresholds[0]) in self.refs:
                self.index[s].append(i)
            for thr in self.thresholds:
                key = (s, thr)
                if key not in self.refs:
                    self.refs[key] = f.copy()
                    continue
                m = self.k0.changemap(f, self.refs[key])
                mask = m > thr
                if mask.any():
                    refresh_blocks(self.refs[key], f, mask)
                    if not self._refresh_checked:  # the strided view assignment really writes into ref
                        jy, jx = np.argwhere(mask)[0]
                        blk = (slice(jy * BLOCK, (jy + 1) * BLOCK), slice(jx * BLOCK, (jx + 1) * BLOCK))
                        assert np.array_equal(self.refs[key][blk], f[blk]), "refresh_blocks did not write"
                        self._refresh_checked = True
                self.keep[key].append(float(mask.mean()))
