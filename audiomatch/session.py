"""Session signatures: "was this recorded on the same rig, in the same room,
on the same day?"

Unlike the constellation fingerprint, this looks at everything *except* the
performance: the noise floor left by the preamps and the room, the mains hum
picked up by the cables, how the two input channels relate to each other, and
the container/filename facts.

This is a heuristic ranker.  It orders candidates for a human to audition; it
does not prove anything.  See the README's "Limitations" section.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config
from .audio import to_mono

_EPS = 1e-12

_TASCAM_RE = re.compile(
    r"(?:^|[^0-9A-Za-z])"
    r"(?P<prefix>TASCAM|DR-?40|ZOOM)?[ _-]*"
    r"(?P<take>\d{4})"
    r"(?P<role>S12|S34|_?[12]|_?[34])?"
    r"(?=\.|_|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TascamName:
    take: Optional[int] = None
    role: Optional[str] = None      # 'S12' | 'S34'


def parse_tascam_name(filename: str) -> TascamName:
    """Extract the take number and dual-record role from a DR-40 filename.

    ``TASCAM_0077S12.wav`` -> take 77, role 'S12'.
    """
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    m = re.search(r"(?P<take>\d{4})\s*(?P<role>S12|S34)?$", stem,
                  re.IGNORECASE)
    if m is None:
        m = re.search(r"(?P<take>\d{4})(?P<role>S12|S34)", stem, re.IGNORECASE)
    if m is None:
        return TascamName()
    role = m.group("role")
    return TascamName(take=int(m.group("take")),
                      role=role.upper() if role else None)


def pair_mate_role(role: Optional[str]) -> Optional[str]:
    return {"S12": "S34", "S34": "S12"}.get(role or "")


# --------------------------------------------------------------------------
# Region capture
# --------------------------------------------------------------------------


class RegionCollector:
    """Captures the first / middle / last N seconds out of a streaming decode.

    Used so that the indexing pass reads each file from disk exactly once: the
    constellation and the session signature are computed from the same decode.
    """

    def __init__(self, duration: float, rate: int = config.ANALYSIS_SR,
                 region_seconds: float = config.SESSION_REGION_SECONDS):
        self.rate = rate
        n = int(region_seconds * rate)
        self._len = n
        if duration <= 0 or duration <= region_seconds * 1.5:
            starts = [0.0]
        else:
            starts = [0.0,
                      max(0.0, (duration - region_seconds) / 2.0),
                      max(0.0, duration - region_seconds)]
        # De-duplicate overlapping regions on short files.
        self._starts = sorted({int(s * rate) for s in starts})
        self._chunks: list[list[np.ndarray]] = [[] for _ in self._starts]
        self._pos = 0

    def push(self, block: np.ndarray) -> None:
        n = block.shape[0]
        lo, hi = self._pos, self._pos + n
        for i, start in enumerate(self._starts):
            end = start + self._len
            if end <= lo or start >= hi:
                continue
            a = max(lo, start) - lo
            b = min(hi, end) - lo
            if b > a:
                self._chunks[i].append(block[a:b].copy())
        self._pos = hi

    def result(self) -> np.ndarray:
        """Concatenated stereo regions, shape ``(n, channels)``."""
        parts = [np.concatenate(c, axis=0) for c in self._chunks if c]
        parts = [p for p in parts if p.shape[0] > 0]
        if not parts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------
# Signature computation
# --------------------------------------------------------------------------


@dataclass
class Signature:
    noise: np.ndarray = field(
        default_factory=lambda: np.zeros(config.NOISE_BANDS, np.float32))
    hum: np.ndarray = field(
        default_factory=lambda: np.zeros(config.HUM_DIM, np.float32))
    chan: np.ndarray = field(
        default_factory=lambda: np.zeros(config.CHAN_DIM, np.float32))
    sample_rate: int = 0
    channels: int = 0
    bits: int = 0
    duration: float = 0.0
    take: Optional[int] = None
    role: Optional[str] = None

    @property
    def mains_hz(self) -> Optional[int]:
        """Which mains frequency dominates the hum profile, if any."""
        n50 = len(config.HUM_50_HARMONICS)
        e50 = float(np.clip(self.hum[:n50], 0, None).sum())
        e60 = float(np.clip(self.hum[n50:], 0, None).sum())
        if max(e50, e60) < 6.0:
            return None
        return 50 if e50 >= e60 else 60


def _log_band_spectrum(mag: np.ndarray, rate: int, nfft: int) -> np.ndarray:
    """Collapse a linear magnitude spectrum onto log-spaced bands (dB)."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
    edges = np.geomspace(config.NOISE_LO_HZ, config.NOISE_HI_HZ,
                         config.NOISE_BANDS + 1)
    out = np.zeros(config.NOISE_BANDS, dtype=np.float32)
    idx = np.searchsorted(freqs, edges)
    for i in range(config.NOISE_BANDS):
        lo, hi = idx[i], max(idx[i] + 1, idx[i + 1])
        hi = min(hi, mag.size)
        if lo >= hi:
            out[i] = out[i - 1] if i else -120.0
        else:
            out[i] = 20.0 * math.log10(float(mag[lo:hi].mean()) + _EPS)
    return out


def _stft_mag(mono: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    if mono.size < nfft:
        return np.zeros((0, nfft // 2 + 1), dtype=np.float32)
    n = 1 + (mono.size - nfft) // hop
    frames = np.lib.stride_tricks.as_strided(
        mono, shape=(n, nfft),
        strides=(mono.strides[0] * hop, mono.strides[0]), writeable=False)
    win = np.hanning(nfft).astype(np.float32)
    return np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32)


def _hum_profile(mono: np.ndarray, rate: int) -> np.ndarray:
    """dB prominence of mains harmonics over the local spectral background."""
    nfft = config.HUM_NFFT
    out = np.zeros(config.HUM_DIM, dtype=np.float32)
    if mono.size < nfft:
        nfft = 1 << int(math.floor(math.log2(max(256, mono.size))))
        if nfft < 4096:
            return out

    # Pick the quietest non-overlapping chunks -- hum is easiest to see when
    # the performance is not on top of it.
    n_chunks = mono.size // nfft
    if n_chunks == 0:
        return out
    trimmed = mono[:n_chunks * nfft].reshape(n_chunks, nfft)
    energy = (trimmed.astype(np.float64) ** 2).mean(axis=1)
    order = np.argsort(energy, kind="stable")[:max(1, min(3, n_chunks))]
    win = np.hanning(nfft).astype(np.float32)
    mag = np.mean([np.abs(np.fft.rfft(trimmed[i] * win))
                   for i in order], axis=0)

    hz_per_bin = rate / nfft
    bg_half = max(3, int(round(config.HUM_BACKGROUND_HZ / hz_per_bin)))
    harmonics = list(config.HUM_50_HARMONICS) + list(config.HUM_60_HARMONICS)
    for i, f in enumerate(harmonics):
        b = int(round(f / hz_per_bin))
        if b + bg_half >= mag.size:
            continue
        peak = float(mag[max(0, b - 1):b + 2].max())
        lo, hi = max(0, b - bg_half), min(mag.size, b + bg_half + 1)
        bg_bins = np.concatenate([mag[lo:max(lo, b - 2)], mag[b + 3:hi]])
        if bg_bins.size == 0:
            continue
        bg = float(np.median(bg_bins))
        out[i] = 20.0 * math.log10((peak + _EPS) / (bg + _EPS))
    return out


def _channel_stats(stereo: np.ndarray, quiet_slices: np.ndarray) -> np.ndarray:
    out = np.zeros(config.CHAN_DIM, dtype=np.float32)
    if stereo.ndim != 2 or stereo.shape[1] < 2 or stereo.shape[0] == 0:
        out[3] = 1.0        # effectively mono
        return out
    left = stereo[:, 0].astype(np.float64)
    right = stereo[:, 1].astype(np.float64)
    rms_l = math.sqrt(float((left ** 2).mean()) + _EPS)
    rms_r = math.sqrt(float((right ** 2).mean()) + _EPS)
    out[0] = 20.0 * math.log10((rms_l + _EPS) / (rms_r + _EPS))

    lc = left - left.mean()
    rc = right - right.mean()
    denom = math.sqrt(float((lc ** 2).sum()) * float((rc ** 2).sum())) + _EPS
    out[1] = float((lc * rc).sum() / denom)

    if quiet_slices.size:
        ql = left[quiet_slices]
        qr = right[quiet_slices]
        nf_l = math.sqrt(float((ql ** 2).mean()) + _EPS)
        nf_r = math.sqrt(float((qr ** 2).mean()) + _EPS)
        out[2] = 20.0 * math.log10((nf_l + _EPS) / (nf_r + _EPS))

    near_identical = out[1] > 0.9995 and abs(float(out[0])) < 0.1
    silent_side = rms_l < 1e-6 or rms_r < 1e-6
    out[3] = 1.0 if (near_identical or silent_side) else 0.0
    return out


def signature_from_regions(stereo: np.ndarray, *, rate: int,
                           sample_rate: int, channels: int, bits: int,
                           duration: float, filename: str) -> Signature:
    """Compute the full session signature from the sampled stereo regions."""
    sig = Signature(sample_rate=sample_rate, channels=channels, bits=bits,
                    duration=duration)
    name = parse_tascam_name(filename)
    sig.take, sig.role = name.take, name.role

    if stereo.shape[0] == 0:
        return sig

    mono = np.ascontiguousarray(to_mono(stereo))
    mag = _stft_mag(mono, config.SESSION_NFFT, config.SESSION_HOP)
    quiet_sample_idx = np.zeros(0, dtype=np.int64)

    if mag.shape[0] >= 4:
        energy = (mag.astype(np.float64) ** 2).sum(axis=1)
        k = max(4, int(round(mag.shape[0] * config.SESSION_QUIET_FRACTION)))
        k = min(k, mag.shape[0])
        quiet = np.argsort(energy, kind="stable")[:k]
        # Average in the log domain: robust to a single loud outlier frame.
        avg = 20.0 * np.log10(mag[quiet] + _EPS)
        band = _log_band_spectrum(
            np.power(10.0, avg.mean(axis=0) / 20.0),
            rate, config.SESSION_NFFT)
        band = band - band.mean()
        norm = float(np.linalg.norm(band))
        sig.noise = (band / norm).astype(np.float32) if norm > _EPS else band

        starts = quiet * config.SESSION_HOP
        quiet_sample_idx = np.concatenate(
            [np.arange(s, min(s + config.SESSION_NFFT, mono.size))
             for s in starts]) if starts.size else quiet_sample_idx

    sig.hum = _hum_profile(mono, rate)
    sig.chan = _channel_stats(stereo, quiet_sample_idx)
    return sig


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class SessionScore:
    total: float
    noise: float
    hum: float
    chan: float
    container: float
    notes: list[str] = field(default_factory=list)


def compare(seed: Signature, other: Signature) -> SessionScore:
    """Weighted session similarity in [0, 1], with a per-component breakdown."""
    # (a) Noise-floor spectrum shape.  Both vectors are mean-removed and
    # L2-normalised, so this is gain-invariant.  Mapped from [-1,1] to [0,1].
    noise = (_cosine(seed.noise, other.noise) + 1.0) / 2.0

    # (b) Hum harmonic profile.  Only positive prominences carry information.
    hs = np.clip(seed.hum, 0.0, None)
    ho = np.clip(other.hum, 0.0, None)
    if float(hs.sum()) < 3.0 and float(ho.sum()) < 3.0:
        hum = 0.5                      # neither has hum: uninformative
    elif float(hs.sum()) < 3.0 or float(ho.sum()) < 3.0:
        hum = 0.2                      # one hums, the other does not
    else:
        hum = max(0.0, _cosine(hs, ho))

    # (c) Channel behaviour: balance, correlation, noise-floor asymmetry.
    d_bal = abs(float(seed.chan[0]) - float(other.chan[0]))
    d_cor = abs(float(seed.chan[1]) - float(other.chan[1]))
    d_nf = abs(float(seed.chan[2]) - float(other.chan[2]))
    same_mono = 1.0 - abs(float(seed.chan[3]) - float(other.chan[3]))
    chan = (0.30 * math.exp(-d_bal / 6.0)
            + 0.25 * max(0.0, 1.0 - d_cor / 1.2)
            + 0.30 * math.exp(-d_nf / 6.0)
            + 0.15 * same_mono)

    # (d) Container facts.  A recorder writes one format for a whole session.
    container = 0.0
    container += 0.45 if seed.sample_rate == other.sample_rate else 0.0
    container += 0.25 if seed.channels == other.channels else 0.0
    container += 0.20 if seed.bits == other.bits else 0.0
    container += 0.10 if (seed.role is not None) == (other.role is not None) \
        else 0.0

    notes: list[str] = []
    if seed.take is not None and other.take is not None:
        if seed.take == other.take and seed.role and other.role \
                and seed.role != other.role:
            notes.append(
                f"dual-record pair-mate of the seed ({seed.role}<->"
                f"{other.role}, take {seed.take:04d})")
        elif abs(seed.take - other.take) <= 8:
            notes.append(
                f"adjacent take number ({other.take:04d} vs seed "
                f"{seed.take:04d})")
    if abs(float(other.chan[2])) > 12.0:
        side = "left" if float(other.chan[2]) < 0 else "right"
        notes.append(f"strongly asymmetric noise floor ({side} much quieter)")
    if float(other.chan[3]) >= 0.5:
        notes.append("effectively mono / one dead channel")
    mains = other.mains_hz
    if mains:
        notes.append(f"{mains} Hz mains hum present")

    total = (config.W_NOISE * noise + config.W_HUM * hum
             + config.W_CHAN * chan + config.W_CONTAINER * container)
    return SessionScore(total=total, noise=noise, hum=hum, chan=chan,
                        container=container, notes=notes)
