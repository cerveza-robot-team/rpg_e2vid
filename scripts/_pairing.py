"""Shared helper for pairing E2VID-reconstructed left/right frames by timestamp.

Both stereo drivers (SGBM, RAFT-Stereo) consume the same L/R frame folders, so
the pairing logic lives here. A folder is expected to contain `frame_*.png`
files plus a `timestamps.txt` with one float (seconds) per frame, in the same
order as `sorted(glob('frame_*.png'))` — exactly the layout that
`run_reconstruction.py` writes via `utils/inference_utils.py::ImageWriter`.
"""

import os
import glob
from typing import List, NamedTuple

import numpy as np


class Pair(NamedTuple):
    idx: int        # left frame index in its sorted list
    l_path: str
    r_path: str
    ts_l: float
    ts_r: float
    dt_ms: float


def _load_folder(folder: str):
    frames = sorted(glob.glob(os.path.join(folder, 'frame_*.png')))
    ts_path = os.path.join(folder, 'timestamps.txt')
    if not os.path.isfile(ts_path):
        raise FileNotFoundError(
            'timestamps.txt missing in {} — was the folder produced by run_reconstruction.py?'
            .format(folder))
    ts = np.loadtxt(ts_path, dtype=np.float64)
    if ts.ndim == 0:
        ts = ts.reshape(1)
    if len(frames) != ts.size:
        raise ValueError(
            '{}: {} frames vs {} timestamps — folder is inconsistent.'
            .format(folder, len(frames), ts.size))
    return frames, ts


def pair_by_timestamp(left_folder: str, right_folder: str,
                      max_dt_ms: float = 10.0,
                      skip_warmup_s: float = 0.0) -> List[Pair]:
    """For each left frame, find the nearest right frame; keep pairs with |Δt| ≤ max_dt_ms.

    `skip_warmup_s` drops left frames whose timestamp is below this threshold
    (FireNet's first ~few frames are noisy from recurrent state init).
    """
    l_frames, l_ts = _load_folder(left_folder)
    r_frames, r_ts = _load_folder(right_folder)

    order = np.argsort(r_ts)
    r_ts_sorted = r_ts[order]
    pairs: List[Pair] = []

    for i, (lp, tl) in enumerate(zip(l_frames, l_ts)):
        if tl < skip_warmup_s:
            continue
        j = int(np.searchsorted(r_ts_sorted, tl))
        candidates = []
        if j < len(r_ts_sorted):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)
        best_j = min(candidates, key=lambda jj: abs(r_ts_sorted[jj] - tl))
        dt_ms = (r_ts_sorted[best_j] - tl) * 1000.0
        if abs(dt_ms) > max_dt_ms:
            continue
        rp = r_frames[order[best_j]]
        pairs.append(Pair(i, lp, rp, float(tl), float(r_ts_sorted[best_j]), float(dt_ms)))

    return pairs
