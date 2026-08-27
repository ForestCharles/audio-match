"""Single-file analysis: one decode, both fingerprints.

This is the unit of work handed to the multiprocessing pool.  It reads each
file from disk exactly once (the 1.45 TB read is the bottleneck of the whole
tool, so reading twice would literally double the wall-clock time) and streams
the decode so that memory stays flat no matter how long the file is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import audio, config, fingerprint
from .session import RegionCollector, Signature, signature_from_regions


@dataclass
class Analysis:
    path: str
    size: int
    mtime: float
    status: str                      # 'ok' | 'error'
    error: Optional[str] = None
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bits: int = 0
    codec: str = "?"
    signature: Optional[Signature] = None
    hashes: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    decoded_bytes: int = 0

    @property
    def n_hashes(self) -> int:
        return 0 if self.hashes is None else int(self.hashes.size)


def analyze_file(path: str, *,
                 peaks_per_band_per_sec: float =
                 config.INDEX_PEAKS_PER_BAND_PER_SEC,
                 fanout: int = config.INDEX_FANOUT) -> Analysis:
    """Fingerprint + session-signature one file.  Never raises."""
    try:
        st = os.stat(path)
        size, mtime = int(st.st_size), float(st.st_mtime)
    except OSError as exc:
        return Analysis(path=path, size=0, mtime=0.0, status="error",
                        error=f"stat failed: {exc}")

    base = Analysis(path=path, size=size, mtime=mtime, status="error")
    if size < config.MIN_FILE_BYTES:
        base.error = f"file too small ({size} bytes)"
        return base

    try:
        p = audio.probe(path)
    except audio.AudioError as exc:
        base.error = f"probe failed: {exc}"
        return base
    except Exception as exc:                      # pragma: no cover - defensive
        base.error = f"probe crashed: {exc!r}"
        return base

    base.duration = p.duration
    base.sample_rate = p.sample_rate
    base.channels = p.channels
    base.bits = p.bits
    base.codec = p.codec

    tracker = fingerprint.BandTracker()
    stft = fingerprint._StftStreamer()
    regions = RegionCollector(p.duration)
    n_samples = 0
    try:
        for block in audio.decode_stream(path, rate=config.ANALYSIS_SR,
                                         channels=2):
            n_samples += block.shape[0]
            regions.push(block)
            tracker.push_spectrogram(stft.push(audio.to_mono(block)))
    except audio.AudioError as exc:
        if n_samples == 0:
            base.error = f"decode failed: {exc}"
            return base
        # Truncated/corrupt tail: keep whatever decoded.  Recovered files
        # frequently end mid-frame; partial data is still worth indexing.
        base.error = f"decode ended early: {exc}"
    except Exception as exc:                      # pragma: no cover - defensive
        base.error = f"decode crashed: {exc!r}"
        return base

    if n_samples == 0:
        base.error = base.error or "decoded to zero samples"
        return base

    decoded_seconds = n_samples / config.ANALYSIS_SR
    if p.duration <= 0:
        base.duration = decoded_seconds

    peaks = tracker.pick(peaks_per_band_per_sec)
    if decoded_seconds >= config.MIN_USEFUL_SECONDS:
        h, t = fingerprint.make_hashes(peaks, fanout)
    else:
        h = np.zeros(0, np.int64)
        t = np.zeros(0, np.int32)

    sig = signature_from_regions(
        regions.result(), rate=config.ANALYSIS_SR,
        sample_rate=p.sample_rate, channels=p.channels, bits=p.bits,
        duration=base.duration, filename=path)

    base.status = "ok"
    base.signature = sig
    base.hashes = h
    base.times = t
    base.decoded_bytes = size
    return base


def analyze_seed(path: str, *, rate_ratio: float = 1.0
                 ) -> tuple[np.ndarray, np.ndarray, Signature, float]:
    """Analyse a query seed, optionally with a sample-rate correction.

    ``rate_ratio`` < 1 decodes at a lower rate and then *interprets* the
    samples as ``ANALYSIS_SR``, which speeds the audio up by ``1/rate_ratio``.
    That is exactly the distortion you get when a 48 kHz recording is written
    with a 44.1 kHz header, so probing both ratios finds mislabelled files.
    """
    p = audio.probe(path)
    decode_rate = int(round(config.ANALYSIS_SR * rate_ratio))
    tracker = fingerprint.BandTracker()
    stft = fingerprint._StftStreamer()
    regions = RegionCollector(p.duration)
    n = 0
    for block in audio.decode_stream(path, rate=decode_rate, channels=2):
        n += block.shape[0]
        regions.push(block)
        tracker.push_spectrogram(stft.push(audio.to_mono(block)))
    if n == 0:
        raise audio.AudioError("seed decoded to zero samples")

    seed_seconds = n / config.ANALYSIS_SR
    density, fanout = fingerprint.query_density(seed_seconds)
    peaks = tracker.pick(density)
    h, t = fingerprint.make_hashes(peaks, fanout)
    sig = signature_from_regions(
        regions.result(), rate=config.ANALYSIS_SR,
        sample_rate=p.sample_rate, channels=p.channels, bits=p.bits,
        duration=p.duration, filename=path)
    return h, t, sig, seed_seconds
