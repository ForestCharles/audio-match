"""The activity envelope: one loudness number per second, and how to align it.

Why this exists
---------------
The constellation fingerprint (mode 1) asks "is this the same *audio*?" and
answers it from the spectrum.  That question has the wrong answer when the same
performance was captured by two different rigs: different microphones in
different positions, different preamps, different gain, a different room
balance.  Almost every spectral peak moves, so almost every landmark differs.

What does *not* move is when the band played and when it stopped.  The
loudness-over-time shape of a session is a property of the performance, not of
the equipment, and it survives everything: EQ, reverb, gain, mp3, a completely
different microphone on the other side of the room.  Correlating that shape is
therefore the equipment-independent way to ask "did these two files record the
same stretch of time?".

Resolution and clock drift
--------------------------
One value per second, and that is the whole trick.  Two recorders have
independent crystals; tens of ppm of disagreement is normal.  At 1 Hz even 200
ppm over a 45-minute take is half a sample of drift, so the cross-correlation
stays a single sharp peak and no drift model is needed.  See
``config.ENVELOPE_HZ`` for the arithmetic.

Storage
-------
``10*log10(mean square)`` per second -- dBFS RMS -- linearly quantised to one
byte over [-120, 0] dB.  2.6 KB for a 45-minute file.  Correlation is done on
the dequantised dB values, because loudness *ratios* are what stay constant
between two captures at different gains, and a ratio is a difference in dB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import config

_EPS = 1e-30

#: dB per quantisation step (0.47 dB).
DB_PER_STEP = ((config.ENVELOPE_DB_CEIL - config.ENVELOPE_DB_FLOOR)
               / (config.ENVELOPE_LEVELS - 1))


def samples(seconds: float) -> int:
    """Seconds -> envelope samples.  One-to-one at the configured 1 Hz."""
    return int(round(seconds * config.ENVELOPE_HZ))


def seconds(n_samples: float) -> float:
    """Envelope samples -> seconds."""
    return float(n_samples) / config.ENVELOPE_HZ


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------


def quantize(mean_square: np.ndarray) -> np.ndarray:
    """Mean-square-per-second -> uint8 dB codes."""
    ms = np.asarray(mean_square, dtype=np.float64)
    if ms.size == 0:
        return np.zeros(0, dtype=np.uint8)
    db = 10.0 * np.log10(np.maximum(ms, 0.0) + _EPS)
    codes = np.rint((db - config.ENVELOPE_DB_FLOOR) / DB_PER_STEP)
    return np.clip(codes, 0, config.ENVELOPE_LEVELS - 1).astype(np.uint8)


def dequantize(codes: np.ndarray) -> np.ndarray:
    """uint8 dB codes -> float32 dBFS."""
    a = np.asarray(codes, dtype=np.float32)
    return (config.ENVELOPE_DB_FLOOR + a * DB_PER_STEP).astype(np.float32)


def pack(codes: np.ndarray) -> bytes:
    return np.asarray(codes, dtype=np.uint8).tobytes()


def unpack(blob: Optional[bytes]) -> np.ndarray:
    if not blob:
        return np.zeros(0, dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8)


# --------------------------------------------------------------------------
# Streaming accumulation
# --------------------------------------------------------------------------


class EnvelopeCollector:
    """Accumulates the 1 Hz envelope from a streaming decode.

    Fed from the *same* decode pass that feeds the constellation and the
    session signature -- reading a 1.45 TB library twice would double the
    wall-clock cost of the whole tool, so nothing here ever opens a file.
    """

    def __init__(self, rate: int = config.ANALYSIS_SR,
                 hz: float = config.ENVELOPE_HZ):
        self.rate = rate
        self._n = max(1, int(round(rate / hz)))
        self._vals: list[float] = []
        self._acc = 0.0
        self._cnt = 0

    def push(self, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        sq = np.square(np.asarray(mono, dtype=np.float64))
        pos = 0
        if self._cnt:
            take = min(self._n - self._cnt, sq.size)
            self._acc += float(sq[:take].sum())
            self._cnt += take
            pos = take
            if self._cnt >= self._n:
                self._vals.append(self._acc / self._n)
                self._acc, self._cnt = 0.0, 0
        rest = sq[pos:]
        full = rest.size // self._n
        if full:
            self._vals.extend(
                rest[:full * self._n].reshape(full, self._n)
                    .mean(axis=1).tolist())
        tail = rest[full * self._n:]
        if tail.size:
            self._acc += float(tail.sum())
            self._cnt += int(tail.size)

    def result(self) -> np.ndarray:
        """The quantised envelope.

        The final partial second is emitted only if it is at least half a
        second long, *or* if it is all there is.  A 0.1 s remainder is a much
        noisier level estimate than a full second and would put a spurious
        bright or dark sample at the end of every file; but a file shorter than
        one second must still produce a (one-sample) envelope, so that "this
        row has no envelope" can mean exactly one thing -- nobody has computed
        it yet -- rather than two.
        """
        vals = list(self._vals)
        if self._cnt and (self._cnt * 2 >= self._n or not vals):
            vals.append(self._acc / self._cnt)
        return quantize(np.asarray(vals, dtype=np.float64))


def envelope_of(mono: np.ndarray, rate: int = config.ANALYSIS_SR
                ) -> np.ndarray:
    """Convenience path for in-memory audio (queries, tests)."""
    collector = EnvelopeCollector(rate=rate)
    collector.push(mono)
    return collector.result()


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


@dataclass
class Alignment:
    """Best envelope alignment of a seed against one candidate."""

    ok: bool
    lag: int = 0            # samples; seed t=0 lands here in the candidate
    score: float = 0.0      # overlap-shrunk Pearson r, in [-1, 1]
    raw_r: float = 0.0      # unshrunk Pearson r over the overlap
    overlap: int = 0        # samples of overlap at the winning lag
    reason: str = ""        # why ``ok`` is False
    #: Winning score divided by the best score at any lag at least
    #: ``config.PAIR_ENVELOPE_DOMINANCE_SEPARATION_SECONDS`` away from it --
    #: see :func:`peak_dominance`.  ``inf`` means nothing else came close (or
    #: there was no far-away lag to compare against); 0.0 means the question
    #: does not arise, because there is no alignment or no positive peak.
    dominance: float = 0.0

    @property
    def peak_is_dominant(self) -> bool:
        """Is the winning lag a *distinct* answer, or one of several?

        False means the score curve is degenerate -- several unrelated lags
        score almost as well as the winner -- which is what a channel that
        hears one source rather than the room looks like.  It says nothing
        about whether the score is high.
        """
        return self.dominance >= config.PAIR_ENVELOPE_DOMINANCE_MIN

    @property
    def lag_seconds(self) -> float:
        return seconds(self.lag)

    @property
    def overlap_seconds(self) -> float:
        return seconds(self.overlap)


def required_overlap(n_seed: int, n_cand: int) -> int:
    """Minimum overlap, in samples, for a lag to be considered at all."""
    shorter = min(n_seed, n_cand)
    return max(samples(config.PAIR_MIN_ENVELOPE_SECONDS),
               int(min(samples(config.PAIR_MIN_OVERLAP_SECONDS),
                       config.PAIR_MIN_OVERLAP_FRACTION * shorter)))


def _shrink(raw_r: np.ndarray, overlap: np.ndarray,
            max_overlap: float) -> np.ndarray:
    """Penalise an overlap for being thin, in both senses.

    *Relative*: an overlap that uses only part of what this pair could have
    overlapped by is suspicious, because sliding two unrelated recordings until
    only their edges touch lines up two fade-outs and scores ~0.9.  A true
    pair's best lag uses all the overlap there is, so this costs it nothing.

    *Absolute*: a short overlap is few independent observations however large a
    fraction of the pair it is.

    See ``config.PAIR_OVERLAP_EXPONENT`` for the measurements behind both.
    """
    k = samples(config.PAIR_OVERLAP_SHRINK_SECONDS)
    relative = (overlap / max(1.0, max_overlap)) ** config.PAIR_OVERLAP_EXPONENT
    absolute = np.sqrt(overlap / (overlap + k))
    return raw_r * relative * absolute


def peak_dominance(lags: np.ndarray, score: np.ndarray,
                   best: Optional[int] = None) -> float:
    """How far the winning lag stands above the rest of the score curve.

    ``top1 / top2``, where ``top2`` is the best score at any lag at least
    ``config.PAIR_ENVELOPE_DOMINANCE_SEPARATION_SECONDS`` from the winner.
    The separation is what makes ``top2`` a *rival alignment* rather than the
    shoulder of the winning peak, which at song scale is tens of seconds wide.

    A dominant peak says "one lag explains this pair and the others do not".
    A ratio near 1 says the opposite: several unrelated lags explain it about
    equally well, so the winner is an argmax over noise and its lag cannot be
    trusted -- see ``config.PAIR_ENVELOPE_DOMINANCE_MIN`` for the measured
    separation between correct and wrong pairs.

    Returns ``inf`` when there is no rival at all (nothing far enough away, or
    nothing positive out there), and ``0.0`` when the winning score is not
    positive, because "how many times better than the runner-up" is not a
    question about a peak that is not a peak.
    """
    s = np.asarray(score, dtype=np.float64)
    if s.size == 0:
        return 0.0
    idx = int(np.argmax(s)) if best is None else int(best)
    top = float(s[idx])
    if top <= 0.0:
        return 0.0
    far = (np.abs(np.asarray(lags, dtype=np.int64) - int(lags[idx]))
           >= samples(config.PAIR_ENVELOPE_DOMINANCE_SEPARATION_SECONDS))
    if not far.any():
        return math.inf
    second = float(s[far].max())
    return top / second if second > 0.0 else math.inf


def align_at(seed_db: np.ndarray, cand_db: np.ndarray, lag: int, *,
             dominance: float = 0.0) -> Alignment:
    """Score one *given* alignment instead of searching for the best one.

    Used when something other than the envelope has decided where these two
    files line up -- the unwindowed coherence fallback -- and the envelope's
    opinion of that placement is still worth printing.  The arithmetic is
    exactly :func:`align`'s, evaluated at a single lag: Pearson over the
    overlap, then the same overlap shrinkage, so the number is comparable with
    every other score in the mode.

    The minimum-overlap rule is deliberately *not* applied.  It exists to stop
    the search wandering onto the edges of two files, and there is no search
    here; a placement backed by shared landmarks is entitled to be scored over
    whatever overlap it implies, however thin, with the usual short-overlap
    cautions doing their job downstream.
    """
    a = np.asarray(seed_db, dtype=np.float64)
    b = np.asarray(cand_db, dtype=np.float64)
    n, m = a.size, b.size
    lo, hi = max(0, -int(lag)), min(n, m - int(lag))
    if hi - lo < 2:
        return Alignment(ok=False, lag=int(lag), dominance=dominance,
                         reason="the offset leaves no overlap to score")
    x = a[lo:hi]
    y = b[lo + int(lag):hi + int(lag)]
    L = float(x.size)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float(xc @ xc) * float(yc @ yc))
    raw = 0.0 if den <= _EPS else float(np.clip((xc @ yc) / den, -1.0, 1.0))
    score = float(_shrink(np.array([raw]), np.array([L]),
                          float(min(n, m)))[0])
    return Alignment(ok=True, lag=int(lag), score=score, raw_r=raw,
                     overlap=int(L), dominance=dominance)


def _prefix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(cumsum, cumsum of squares)``, each with a leading zero."""
    c = np.zeros(x.size + 1, dtype=np.float64)
    c2 = np.zeros(x.size + 1, dtype=np.float64)
    np.cumsum(x, out=c[1:])
    np.cumsum(np.square(x), out=c2[1:])
    return c, c2


def _fft_size(n: int) -> int:
    size = 1
    while size < n:
        size <<= 1
    return size


def align(seed_db: np.ndarray, cand_db: np.ndarray) -> Alignment:
    """Best-lag normalised cross-correlation of two dB envelopes.

    The score at each lag is an *exact* Pearson correlation computed over just
    the overlapping region -- means, variances and the cross term are all taken
    over the overlap, not over the whole file.  That is what makes files of
    different lengths comparable: a 3-minute file inside a 45-minute one scores
    on its own three minutes, and is not diluted by the 42 minutes it says
    nothing about.

    The cross term for every lag comes from one FFT pair; the per-lag means and
    variances come from prefix sums.  Both are O(n log n) for the whole lag
    range, so there is no per-lag Python loop.
    """
    result = _align_many_impl(seed_db, [cand_db])
    return result[0]


def align_many(seed_db: np.ndarray,
               cands: Sequence[np.ndarray]) -> list[Alignment]:
    """:func:`align` over many candidates, batching the FFTs.

    Candidates are padded to a common length and transformed with a single
    2-D ``rfft`` per batch of ``config.PAIR_FFT_BATCH``.  For a few thousand
    45-minute candidates that is a couple of hundred transforms of a few
    thousand points each -- fractions of a second -- rather than one Python
    round trip per candidate per lag.
    """
    out: list[Alignment] = []
    batch = max(1, config.PAIR_FFT_BATCH)
    for i in range(0, len(cands), batch):
        out.extend(_align_many_impl(seed_db, cands[i:i + batch]))
    return out


def _align_many_impl(seed_db: np.ndarray,
                     cands: Sequence[np.ndarray]) -> list[Alignment]:
    n = int(np.asarray(seed_db).size)
    if n < samples(config.PAIR_MIN_ENVELOPE_SECONDS):
        reason = (f"seed is {seconds(n):.0f}s of envelope, below the "
                  f"{config.PAIR_MIN_ENVELOPE_SECONDS:.0f}s minimum")
        return [Alignment(ok=False, reason=reason) for _ in cands]

    a = np.asarray(seed_db, dtype=np.float64)
    a = a - a.mean()                       # conditioning only; r is unchanged
    if float(np.abs(a).max()) < _EPS:
        return [Alignment(ok=False, reason="seed envelope is flat (silence?)")
                for _ in cands]

    lengths = [int(np.asarray(c).size) for c in cands]
    max_m = max(lengths) if lengths else 0
    nfft = _fft_size(n + max_m + 1)

    A = np.fft.rfft(a, nfft).conj()
    B = np.zeros((len(cands), nfft), dtype=np.float64)
    for i, c in enumerate(cands):
        m = lengths[i]
        if m:
            b = np.asarray(c, dtype=np.float64)
            B[i, :m] = b - b.mean()
    corr = np.fft.irfft(np.fft.rfft(B, axis=1) * A, nfft, axis=1)

    a_sum, a_sq = _prefix(a)
    results: list[Alignment] = []
    for i, c in enumerate(cands):
        m = lengths[i]
        need = required_overlap(n, m) if m else 0
        if m == 0 or m < need or n < need:
            results.append(Alignment(
                ok=False,
                reason=f"candidate is {seconds(m):.0f}s: too short to overlap "
                       f"the seed by {seconds(need):.0f}s"))
            continue
        b = np.asarray(c, dtype=np.float64)
        b = b - b.mean()
        b_sum, b_sq = _prefix(b)

        # Lag k means: seed sample i lines up with candidate sample i+k.
        lags = np.arange(-(n - need), m - need + 1, dtype=np.int64)
        if lags.size == 0:
            results.append(Alignment(
                ok=False, reason="no lag reaches the minimum overlap"))
            continue
        lo = np.maximum(0, -lags)
        hi = np.minimum(n, m - lags)
        L = (hi - lo).astype(np.float64)

        sa = a_sum[hi] - a_sum[lo]
        saa = a_sq[hi] - a_sq[lo]
        sb = b_sum[hi + lags] - b_sum[lo + lags]
        sbb = b_sq[hi + lags] - b_sq[lo + lags]
        # Circular cross-correlation: non-negative lags sit at the front,
        # negative lags wrap to the back.
        sab = corr[i][np.where(lags >= 0, lags, lags + nfft)]

        num = sab - sa * sb / L
        var_a = np.maximum(saa - sa * sa / L, 0.0)
        var_b = np.maximum(sbb - sb * sb / L, 0.0)
        den = np.sqrt(var_a * var_b)
        raw = np.where(den > _EPS, num / np.maximum(den, _EPS), 0.0)
        raw = np.clip(raw, -1.0, 1.0)
        score = _shrink(raw, L, float(min(n, m)))

        best = int(np.argmax(score))
        results.append(Alignment(
            ok=True, lag=int(lags[best]), score=float(score[best]),
            raw_r=float(raw[best]), overlap=int(L[best]),
            dominance=peak_dominance(lags, score, best)))
    return results


# --------------------------------------------------------------------------
# Active-segment structure (presentation only -- never scoring)
# --------------------------------------------------------------------------


@dataclass
class SegmentReport:
    """How well two files' track structure lines up, in human terms."""

    seed_segments: int
    cand_segments: int
    matched: int
    total: int
    tolerance: float       # seconds

    @property
    def text(self) -> str:
        if self.total == 0:
            return "no clear track structure in the aligned region"
        return (f"segments align: {self.matched}/{self.total} boundaries "
                f"within +/-{self.tolerance:.0f}s "
                f"({self.seed_segments} active stretch(es) in the seed, "
                f"{self.cand_segments} in this file)")


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or x.size == 0:
        return x
    pad = width // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid")[:x.size]


def active_segments(env_db: np.ndarray) -> list[tuple[int, int]]:
    """Rough ``[start, end)`` second ranges where the recording is playing.

    An adaptive threshold on the smoothed envelope: the 10th percentile is the
    file's own gaps-and-room-tone level, the 90th is its playing level, and the
    boundary sits a fixed fraction of the way between them, so this works
    equally on a quiet ambient set and a loud band.  Short runs and short gaps
    are absorbed so that a rest inside a tune does not read as two tracks.
    """
    x = np.asarray(env_db, dtype=np.float64)
    if x.size < 3:
        return []
    width = max(1, samples(config.PAIR_SEGMENT_SMOOTH_SECONDS))
    s = _smooth(x, width)
    lo, hi = np.percentile(s, [10.0, 90.0])
    if hi - lo < 3.0:            # < 3 dB of range: no structure to find
        return []
    thr = lo + config.PAIR_SEGMENT_THRESHOLD_FRACTION * (hi - lo)
    active = s > thr

    runs: list[list[int]] = []
    idx = np.flatnonzero(np.diff(active.astype(np.int8)))
    edges = np.concatenate([[0], idx + 1, [active.size]])
    for start, end in zip(edges[:-1], edges[1:]):
        if active[start]:
            runs.append([int(start), int(end)])
    if not runs:
        return []

    gap = samples(config.PAIR_SEGMENT_GAP_SECONDS)
    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    keep = samples(config.PAIR_SEGMENT_MIN_SECONDS)
    return [(s0, s1) for s0, s1 in merged if s1 - s0 >= keep]


def compare_segments(seed_db: np.ndarray, cand_db: np.ndarray,
                     lag: int) -> SegmentReport:
    """Match the seed's segment boundaries against the candidate's.

    Presentation only.  The user's collaborator thinks in track starts and
    runtimes, so pair mode says "6/6 boundaries within +/-4s" as well as
    "r = 0.87"; the verdict does not depend on it.
    """
    tol = samples(config.PAIR_SEGMENT_TOLERANCE_SECONDS)
    seed_runs = active_segments(seed_db)
    cand_runs = active_segments(cand_db)
    # Only boundaries that fall inside the overlap can possibly agree.
    n, m = int(np.asarray(seed_db).size), int(np.asarray(cand_db).size)
    lo, hi = max(0, -lag), min(n, m - lag)

    seed_bounds = [b + lag for run in seed_runs for b in run
                   if lo <= b <= hi]
    cand_bounds = [b for run in cand_runs for b in run
                   if lo + lag <= b <= hi + lag]
    tol_s = config.PAIR_SEGMENT_TOLERANCE_SECONDS
    if not seed_bounds:
        return SegmentReport(len(seed_runs), len(cand_runs), 0, 0, tol_s)
    matched = 0
    remaining = list(cand_bounds)
    for b in seed_bounds:
        near = [c for c in remaining if abs(c - b) <= tol]
        if near:
            pick = min(near, key=lambda c: abs(c - b))
            remaining.remove(pick)
            matched += 1
    return SegmentReport(len(seed_runs), len(cand_runs), matched,
                         len(seed_bounds), tol_s)
