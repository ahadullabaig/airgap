"""Frame sampling with PyAV.

We decode frames ourselves instead of using the model processor's built-in
video path (fixed 1 fps), so the same code serves the captioning windows and
the higher-rate trace export for Part B.
"""

from dataclasses import dataclass

import av
from PIL import Image


@dataclass
class VideoInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str


def probe(path: str) -> VideoInfo:
    with av.open(path) as container:
        stream = container.streams.video[0]
        if container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = float(stream.duration * stream.time_base)
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        return VideoInfo(path, duration, fps, stream.width, stream.height, stream.codec_context.name)


def sample_window(path: str, start_s: float, window_s: float, k: int) -> list[tuple[float, Image.Image]]:
    """Return k frames spread evenly over [start_s, start_s + window_s).

    Each target time t_i = start_s + (i + 0.5) * window_s / k gets the first
    decoded frame at or after it. Returns (timestamp_s, RGB image) pairs.
    """
    if k < 1 or window_s <= 0:
        raise ValueError("need k >= 1 and window_s > 0")
    targets = [start_s + (i + 0.5) * window_s / k for i in range(k)]
    out: list[tuple[float, Image.Image]] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek lands on the keyframe at or before start_s; decode forward from there.
        container.seek(int(start_s / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t >= targets[len(out)]:
                out.append((t, frame.to_image()))
                if len(out) == k:
                    break
    if len(out) < k:
        raise ValueError(f"window [{start_s}, {start_s + window_s}) yielded {len(out)} of {k} frames")
    return out
