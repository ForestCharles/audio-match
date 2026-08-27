"""Constellation ("Shazam-style") fingerprinting.

Pipeline
--------
1. Decode to mono @ 11025 Hz (done by :mod:`audiomatch.audio`).
2. Streaming STFT: 1024-point FFT, 256-sample hop -> 43.07 frames/s,
   10.77 Hz/bin, log magnitude.
3. Reduce each frame to one candidate per log-spaced band (8 bands).  Keeping
   the reduction per-band is what makes the constellation survive EQ and mix
   differences: a bass-heavy transfer cannot crowd the treble out of the peak
   budget.
4. Keep band-local maxima over a +-3 frame neighbourhood, then take the top
   ``peaks_per_band_per_sec`` per band per one-second block.  This gives exact,
   deterministic control over peak density -- which is what controls database
   size.
5. Pair each anchor peak with the next ``fanout`` peaks inside the target zone
   and pack (f1, f2, dt) into a 23-bit integer hash.

Asymmetric density
------------------
The library is indexed sparsely (8 peaks/s) to keep the database small, but a
query is run densely (32 peaks/s, fan-out 16).  The dense query peak set is
very nearly a superset of the sparse one, so peaks that sat near the selection
threshold and flipped under MP3/resampling are still covered.  Density costs
disk on the index side and only CPU on the query side, so this is a free win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from . import config
from .audio import to_mono

_EPS = 1e-12

#: Peaks more than this many dB below the file's loud-ish reference level are
#: dropped.  The reference is a percentile of the file's own band energies, so
#: the rule is gain-invariant (important: some recovered files are ~50 dB down).
PEAK_FLOOR_DB = 70.0
_REF_PERCENTILE = 95.0

#: Above this seed length, query density is reduced -- a 40-minute seed does
#: not need 32 peaks/s to be identified, and the pairing loop is O(peaks).
LONG_SEED_SECONDS = 300.0


def _sliding_max(x: np.ndarray, half_width: int) -> np.ndarray:
    """Maximum over ``x[i-half_width : i+half_width+1]``, edges clamped."""
    if half_width <= 0 or x.size == 0:
        return x
    pad = np.pad(x, half_width, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(pad, 2 * half_width + 1)
    return view.max(axis=1)


def _sliding_max_left(x: np.ndarray, half_width: int) -> np.ndarray:
    """Maximum over ``x[i-half_width : i]`` (strictly before ``i``)."""
    if half_width <= 0 or x.size == 0:
        return np.full_like(x, -np.inf)
    pad = np.concatenate([np.full(half_width, -np.inf, dtype=x.dtype), x])
    view = np.lib.stride_tricks.sliding_window_view(pad[:-1], half_width)
    return view.max(axis=1)


class _StftStreamer:
    """Feed it mono blocks, get back successive log-magnitude frames."""

    def __init__(self, nfft: int = config.NFFT, hop: int = config.HOP):
        self.nfft = nfft
        self.hop = hop
        self.window = np.hanning(nfft).astype(np.float32)
        self._tail = np.zeros(0, dtype=np.float32)

    def push(self, mono: np.ndarray) -> np.ndarray:
        """Return ``(n_new_frames, nfft//2+1)`` log-magnitude, or an empty."""
        buf = np.concatenate([self._tail, mono]) if self._tail.size else mono
        n_frames = 0
        if buf.size >= self.nfft:
            n_frames = 1 + (buf.size - self.nfft) // self.hop
        if n_frames <= 0:
            self._tail = buf
            return np.zeros((0, self.nfft // 2 + 1), dtype=np.float32)
        frames = np.lib.stride_tricks.as_strided(
            buf,
            shape=(n_frames, self.nfft),
            strides=(buf.strides[0] * self.hop, buf.strides[0]),
            writeable=False,
        )
        spec = np.fft.rfft(frames * self.window, axis=1)
        mag = np.abs(spec).astype(np.float32)
        self._tail = buf[n_frames * self.hop:].copy()
        return 20.0 * np.log10(mag + _EPS)


@dataclass
class Peaks:
    """Selected constellation peaks, sorted by time."""

    frames: np.ndarray   # int32 frame index
    bins: np.ndarray     # int16 FFT bin index
    values: np.ndarray   # float32 dB
    n_frames: int        # total STFT frames in the source

    def __len__(self) -> int:
        return int(self.frames.size)

    @property
    def duration(self) -> float:
        return self.n_frames * config.FRAME_SECONDS


class BandTracker:
    """Accumulates the per-band maximum of every STFT frame.

    Memory is 8 floats + 8 ints per frame: ~8 MB for a 68-minute file, so the
    whole (reduced) spectrogram is kept and peak picking happens once, at the
    end, with no chunk-boundary artefacts.
    """

    def __init__(self) -> None:
        self._vals: list[np.ndarray] = []
        self._bins: list[np.ndarray] = []
        self.n_frames = 0

    def push_spectrogram(self, spec: np.ndarray) -> None:
        if spec.shape[0] == 0:
            return
        vals = np.empty((spec.shape[0], config.N_BANDS), dtype=np.float32)
        bins = np.empty((spec.shape[0], config.N_BANDS), dtype=np.int16)
        for b in range(config.N_BANDS):
            lo, hi = config.BAND_EDGES[b], config.BAND_EDGES[b + 1]
            sub = spec[:, lo:hi]
            idx = sub.argmax(axis=1)
            bins[:, b] = idx + lo
            vals[:, b] = sub[np.arange(sub.shape[0]), idx]
        self._vals.append(vals)
        self._bins.append(bins)
        self.n_frames += spec.shape[0]

    def pick(self, peaks_per_band_per_sec: float) -> Peaks:
        if not self._vals:
            return Peaks(np.zeros(0, np.int32), np.zeros(0, np.int16),
                         np.zeros(0, np.float32), 0)
        vals = np.concatenate(self._vals, axis=0)
        bins = np.concatenate(self._bins, axis=0)
        return _pick_peaks(vals, bins, peaks_per_band_per_sec)


def _pick_peaks(vals: np.ndarray, bins: np.ndarray,
                peaks_per_band_per_sec: float) -> Peaks:
    n_frames = vals.shape[0]
    if n_frames == 0:
        return Peaks(np.zeros(0, np.int32), np.zeros(0, np.int16),
                     np.zeros(0, np.float32), 0)

    # Gain-invariant floor, from the file's own loudness distribution.
    reference = float(np.percentile(vals, _REF_PERCENTILE))
    floor = reference - PEAK_FLOOR_DB

    block = max(1, int(round(config.FRAME_RATE)))     # ~1 second
    per_block = max(1, int(round(peaks_per_band_per_sec)))

    out_t: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_v: list[np.ndarray] = []

    for band in range(config.N_BANDS):
        v = vals[:, band]
        # Band-local maximum over +-PEAK_TIME_NEIGHBORHOOD frames.  The
        # strict-left test makes plateaus resolve to their first frame, so the
        # selection is deterministic.
        w = config.PEAK_TIME_NEIGHBORHOOD
        is_max = (v >= _sliding_max(v, w)) & (v > _sliding_max_left(v, w))
        cand = np.flatnonzero(is_max & (v > floor))
        if cand.size == 0:
            continue
        # Top `per_block` candidates per one-second block, by magnitude.
        blk = cand // block
        cv = v[cand]
        # Stable ordering: block asc, value desc, frame asc.
        order = np.lexsort((cand, -cv, blk))
        cand_s, blk_s = cand[order], blk[order]
        _, first, counts = np.unique(blk_s, return_index=True,
                                     return_counts=True)
        rank = np.arange(cand_s.size) - np.repeat(first, counts)
        keep = cand_s[rank < per_block]
        out_t.append(keep.astype(np.int32))
        out_b.append(bins[keep, band].astype(np.int16))
        out_v.append(v[keep].astype(np.float32))

    if not out_t:
        return Peaks(np.zeros(0, np.int32), np.zeros(0, np.int16),
                     np.zeros(0, np.float32), n_frames)

    t = np.concatenate(out_t)
    b = np.concatenate(out_b)
    val = np.concatenate(out_v)
    order = np.lexsort((b, t))          # time asc, then bin asc: deterministic
    return Peaks(t[order], b[order], val[order], n_frames)


def make_hashes(peaks: Peaks, fanout: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair peaks into landmark hashes.

    Returns ``(hashes int64, anchor_frames int32)``.
    """
    n = len(peaks)
    if n < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)

    t = peaks.frames
    fq = (peaks.bins >> config.FREQ_SHIFT).astype(np.int64)

    hashes = np.empty(n * fanout, dtype=np.int64)
    times = np.empty(n * fanout, dtype=np.int32)
    k = 0
    min_dt = config.TARGET_ZONE_MIN_DT
    max_dt = config.TARGET_ZONE_MAX_DT
    max_df = config.TARGET_ZONE_MAX_DF

    # `t` is sorted, so the target zone for anchor i is a contiguous slice.
    hi_idx = np.searchsorted(t, t + max_dt, side="right")
    lo_idx = np.searchsorted(t, t + min_dt, side="left")

    t_list = t.tolist()
    f_list = fq.tolist()
    lo_list = lo_idx.tolist()
    hi_list = hi_idx.tolist()

    for i in range(n):
        t1 = t_list[i]
        f1 = f_list[i]
        taken = 0
        for j in range(lo_list[i], hi_list[i]):
            f2 = f_list[j]
            if f2 - f1 > max_df or f1 - f2 > max_df:
                continue
            dt = t_list[j] - t1
            hashes[k] = (f1 << 15) | (f2 << 7) | dt
            times[k] = t1
            k += 1
            taken += 1
            if taken >= fanout:
                break
    return hashes[:k], times[:k]


def query_density(duration: float) -> tuple[float, int]:
    """Peak density and fan-out to use for a seed of the given length."""
    if duration > LONG_SEED_SECONDS:
        return (config.INDEX_PEAKS_PER_BAND_PER_SEC * 2.0, 6)
    return (config.QUERY_PEAKS_PER_BAND_PER_SEC, config.QUERY_FANOUT)


def peaks_from_blocks(blocks: Iterable[np.ndarray],
                      peaks_per_band_per_sec: float,
                      tracker: Optional[BandTracker] = None) -> Peaks:
    """Consume mono/stereo blocks and return the picked peaks."""
    tracker = tracker if tracker is not None else BandTracker()
    stft = _StftStreamer()
    for block in blocks:
        spec = stft.push(to_mono(block))
        tracker.push_spectrogram(spec)
    return tracker.pick(peaks_per_band_per_sec)


def fingerprint_array(mono: np.ndarray, peaks_per_band_per_sec: float,
                      fanout: int) -> tuple[np.ndarray, np.ndarray, Peaks]:
    """Convenience path for in-memory audio (queries, tests)."""
    peaks = peaks_from_blocks([mono], peaks_per_band_per_sec)
    h, t = make_hashes(peaks, fanout)
    return h, t, peaks
