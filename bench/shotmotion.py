"""Camera motion and shot-change signals from decoded frames, NumPy only.

Used to label footage (hover / moving, single shot or not) without looking at
K0's output, so keep-rates measured later are not judged by the same pixel
difference that chose the labels.

Camera motion: each frame is resized to SIZE x SIZE grey. A GRID x GRID set of
patches is matched between two frames by phase correlation, and a similarity
transform (shift, rotation, zoom) is fitted to the patch shifts with
trimmed least squares, so a few patches with moving people or cars do not
decide the answer. Motion is reported as the RMS displacement the fitted
transform gives over the patch centres, in frame widths per second.

Shot change: two signals between frames about 0.25 s apart, (1) the share of
patches that match consistently with one transform, and (2) the L1 distance
between coarse colour histograms. A cut or dissolve drops (1) and raises (2).
"""

from dataclasses import dataclass

import av
import numpy as np

SIZE = 320
PATCH = 64
GRID = 5
PEAK_MIN = 0.08  # phase-correlation peak below this: no usable match for the patch
TOL_PX = 2.0  # a patch is an inlier if within this many working pixels of the fitted transform
HIST_SIZE = 80
HIST_BINS = 4  # per channel

_w = np.hanning(PATCH).astype(np.float32)
_HANN = np.outer(_w, _w)
_CENTRES = np.arange(GRID) * (SIZE - PATCH) / (GRID - 1) + PATCH / 2
_PTS = np.array([(x, y) for y in _CENTRES for x in _CENTRES], dtype=np.float64) - SIZE / 2  # relative to centre
_ORIGINS = [(int(y - PATCH / 2), int(x - PATCH / 2)) for y in _CENTRES for x in _CENTRES]


@dataclass
class Decoded:
    grey: np.ndarray  # (n, SIZE, SIZE) float32
    hist: np.ndarray  # (n, HIST_BINS**3) float32, sums to 1
    pts_s: np.ndarray  # (n,) presentation time in seconds (nan if missing)
    key: np.ndarray  # (n,) bool, decoder keyframe flag
    fps: float  # nominal (r_frame_rate)
    width: int
    height: int


def decode(path: str, max_frames: int | None = None) -> Decoded:
    grey, hist, pts, key = [], [], [], []
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.base_rate or st.average_rate or st.guessed_rate)
        for fr in c.decode(st):
            g = fr.reformat(width=SIZE, height=SIZE, format="gray").to_ndarray()
            rgb = fr.reformat(width=HIST_SIZE, height=HIST_SIZE, format="rgb24").to_ndarray()
            q = (rgb // (256 // HIST_BINS)).astype(np.int32)
            idx = (q[..., 0] * HIST_BINS + q[..., 1]) * HIST_BINS + q[..., 2]
            h = np.bincount(idx.ravel(), minlength=HIST_BINS**3).astype(np.float32)
            grey.append(g.astype(np.float32))
            hist.append(h / h.sum())
            pts.append(float(fr.pts * st.time_base) if fr.pts is not None else np.nan)
            key.append(bool(fr.key_frame))
            if max_frames and len(grey) >= max_frames:
                break
        w, h_ = st.width, st.height
    return Decoded(np.stack(grey), np.stack(hist), np.array(pts), np.array(key), fps, w, h_)


def _patch_fft(g: np.ndarray) -> np.ndarray:
    p = np.stack([g[y:y + PATCH, x:x + PATCH] for y, x in _ORIGINS])
    p = p - p.mean(axis=(1, 2), keepdims=True)
    return np.fft.rfft2(p * _HANN)


def patch_shifts(fa: np.ndarray, fb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift of each patch from frame a to frame b, in working pixels, and the
    phase-correlation peak height. Content at x in a appears at x + shift in b."""
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    c = np.fft.irfft2(r, s=(PATCH, PATCH))
    n = c.shape[0]
    flat = c.reshape(n, -1).argmax(axis=1)
    py, px = np.divmod(flat, PATCH)
    peak = c[np.arange(n), py, px]

    def sub(cm, c0, cp):  # parabolic peak refinement
        d = cm - 2 * c0 + cp
        safe = np.where(np.abs(d) > 1e-9, d, 1.0)
        return np.where(np.abs(d) > 1e-9, 0.5 * (cm - cp) / safe, 0.0)

    ar = np.arange(n)
    dy = sub(c[ar, (py - 1) % PATCH, px], peak, c[ar, (py + 1) % PATCH, px])
    dx = sub(c[ar, py, (px - 1) % PATCH], peak, c[ar, py, (px + 1) % PATCH])
    sy = ((py + PATCH // 2) % PATCH - PATCH // 2) + dy
    sx = ((px + PATCH // 2) % PATCH - PATCH // 2) + dx
    return np.stack([sx, sy], axis=1), peak


def fit_similarity(shifts: np.ndarray, ok: np.ndarray):
    """Trimmed least-squares fit of d = [[a, -b], [b, a]] p + t to patch shifts.
    Returns (params (a, b, tx, ty) or None, inlier mask)."""
    if ok.sum() < 4:
        return None, np.zeros(len(shifts), bool)
    x, y = _PTS[:, 0], _PTS[:, 1]
    A = np.zeros((2 * len(x), 4))
    A[0::2] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], 1)
    A[1::2] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], 1)
    d = shifts.reshape(-1)
    use = ok.copy()
    params = None
    for _ in range(3):
        rows = np.repeat(use, 2)
        if use.sum() < 4:
            return None, use
        params, *_ = np.linalg.lstsq(A[rows], d[rows], rcond=None)
        res = np.linalg.norm((A @ params - d).reshape(-1, 2), axis=1)
        new = ok & (res <= TOL_PX)
        if (new == use).all():
            break
        use = new
    return params, use


def predicted_rms(params) -> float:
    a, b, tx, ty = params
    x, y = _PTS[:, 0], _PTS[:, 1]
    dx = a * x - b * y + tx
    dy = b * x + a * y + ty
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))


@dataclass
class PairSignals:
    motion_fw_per_s: np.ndarray  # camera motion, frame widths per second (nan if no fit)
    inlier_frac: np.ndarray  # share of all patches consistent with the fitted transform
    hist_l1: np.ndarray  # half L1 distance of colour histograms, 0..1
    i0: np.ndarray  # first frame index of each pair
    k: int


def pair_signals(dec: Decoded, k: int, step: int = 1) -> PairSignals:
    """Signals for frame pairs (i, i + k), i = 0, step, 2*step, ..."""
    n = len(dec.grey)
    idx = np.arange(0, n - k, step)
    ffts = {}

    def fft_of(i):
        if i not in ffts:
            ffts[i] = _patch_fft(dec.grey[i])
        return ffts[i]

    mot = np.full(len(idx), np.nan)
    inl = np.zeros(len(idx))
    hl1 = 0.5 * np.abs(dec.hist[idx + k] - dec.hist[idx]).sum(axis=1)
    for j, i in enumerate(idx):
        sh, pk = patch_shifts(fft_of(i), fft_of(i + k))
        params, use = fit_similarity(sh, pk >= PEAK_MIN)
        inl[j] = use.sum() / len(use)
        if params is not None:
            mot[j] = predicted_rms(params) / SIZE * dec.fps / k
        for old in [x for x in ffts if x < i]:
            del ffts[old]
    return PairSignals(mot, inl, hl1, idx, k)


HIST_EVENT = 0.10  # half-L1 histogram distance over ~0.25 s that marks a possible shot change
EVENT_MERGE_S = 0.5


def shot_events(ps: PairSignals, fps: float, thr: float = HIST_EVENT) -> list[tuple[float, float, float]]:
    """Runs of pairs with hist_l1 >= thr, merged when closer than EVENT_MERGE_S.
    Returns (start_s, end_s, peak) with times from frame index / fps."""
    t0 = ps.i0 / fps
    t1 = (ps.i0 + ps.k) / fps
    ev: list[list[float]] = []
    for a, b, h in zip(t0.tolist(), t1.tolist(), ps.hist_l1.tolist()):
        if h < thr:
            continue
        if ev and a - ev[-1][1] <= EVENT_MERGE_S:
            ev[-1][1] = b
            ev[-1][2] = max(ev[-1][2], h)
        else:
            ev.append([a, b, h])
    return [tuple(e) for e in ev]
