"""Searching the index.

Mode 1 (``match``)
    Landmark hashes from the seed are looked up in the index.  Every posting
    casts a vote for ``(library_file, t_library - t_seed)``.  A real match puts
    (almost) all of its votes in a single offset bin, because the whole seed is
    displaced by one constant amount; coincidental hash collisions scatter
    uniformly.  The height of that peak is the score.

Mode 2 (``session``)
    The seed's session signature is compared against every indexed file's.

Mode 3 (``pair``)
    "Which other files captured this same stretch of time?"  Two pieces of
    evidence, in order of generality:

    1. **Activity envelope** (primary, equipment-independent).  The 1 Hz
       loudness shape is a property of the performance, so it survives a
       completely different microphone, preamp, gain and room position.  Best-
       lag normalised cross-correlation finds the alignment.
    2. **Constellation coherence** (confirming, shared-clock only).  Landmarks
       that survive between the two captures should all agree on one offset --
       or, if the two recorders' clocks differ, on one straight *line* through
       (t_seed, t_lib).  Fitting that line's slope both sharpens the peak and
       measures the clock drift in ppm.

    Evidence 2 is decisive when it fires and silent when it does not: a second
    recorder with different mics may share almost no landmarks at all.  So it
    confirms, and the envelope leads.

    Because evidence 2 can only ever confirm, evidence 1 is never contradicted
    by anything -- so the *length* of the overlap has to do that job instead.
    Two unrelated recordings of the same band correlate at up to 0.838 over a
    five-minute overlap, which is above the PAIR bar, so below
    ``config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS`` of overlap the envelope
    alone tops out at TIMELINE MATCH.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from . import config, envelope as env
from .analyze import analyze_seed, analyze_seed_full
from .db import Database
from .session import Signature, SessionScore, compare, pair_mate_role

#: Guard against pathological seeds (someone seeds with a 3-hour file).  A
#: seed above this many landmarks is uniformly subsampled down to it, and the
#: caller is warned: the result is still usable but scores are no longer
#: comparable with an un-subsampled query.  At the query peak density this is
#: roughly 2.5 hours of audio, so ordinary seeds never come near it.
MAX_SEED_HASHES = 400_000

_OFFSET_BIAS = 1 << 21
_FILE_STRIDE = 1 << 22


@dataclass
class MatchHit:
    file_id: int
    path: str
    votes: int
    total_votes: int
    offset_frames: int
    probe: str
    ratio: float
    matched_seconds: float
    library_duration: float
    background: int = 0
    take: Optional[int] = None
    role: Optional[str] = None
    missing: bool = False

    @property
    def concentration(self) -> float:
        return self.votes / self.total_votes if self.total_votes else 0.0

    @property
    def sharpness(self) -> float:
        """Winning bin height divided by the file's own next-tallest bin.

        This is the single most informative number: a genuine alignment is a
        spike, so its sharpness runs from ~5 to 50+, while an unrelated file's
        histogram is flat and lands near 1.

        The background is floored at ``config.SHARPNESS_MIN_BACKGROUND``.  A
        file that shares only one offset bin with the seed has *no* measured
        noise floor, and dividing by 1 there reported an unfalsifiable "25x"
        for what was really a single lonely bin backed by no evidence at all.
        The floor makes sharpness mean "votes per plausible coincidence" with
        a fixed, documented minimum denominator.
        """
        return self.votes / max(config.SHARPNESS_MIN_BACKGROUND,
                                self.background)

    @property
    def offset_seconds(self) -> float:
        """Where in the library file the seed's t=0 lands (may be negative)."""
        return self.offset_frames * config.FRAME_SECONDS

    def is_confident(self, seed_seconds: float) -> bool:
        need = max(config.CONFIDENT_MIN_VOTES,
                   config.CONFIDENT_VOTES_PER_SEED_SECOND * seed_seconds)
        return (self.votes >= need
                and self.sharpness >= config.CONFIDENT_MIN_SHARPNESS)


@dataclass
class SessionHit:
    file_id: int
    path: str
    score: SessionScore
    duration: float
    sample_rate: int
    channels: int
    bits: int
    take: Optional[int] = None
    role: Optional[str] = None
    is_pair_mate: bool = False
    missing: bool = False


@dataclass
class Coherence:
    """Constellation agreement between the seed and one candidate, allowing
    for a linear clock drift between the two recorders."""

    votes: int = 0
    background: int = 0
    offset_frames: int = 0
    drift_ppm: float = 0.0
    slopes_tried: int = 1
    #: True when the seed was long enough for the drift grid to hold more than
    #: the zero slope.  When it is False, ``drift_ppm`` is 0 because drift was
    #: not measurable, not because it was measured to be zero.
    drift_measurable: bool = False
    #: The grid step, in ppm, and therefore the smallest drift this seed length
    #: could have seen.  ``drift_ppm == 0`` with a non-zero resolution means
    #: "nothing above this was detected", not "the clocks agree exactly".
    drift_resolution_ppm: float = 0.0

    @property
    def sharpness(self) -> float:
        return self.votes / max(config.SHARPNESS_MIN_BACKGROUND,
                                self.background)

    @property
    def offset_seconds(self) -> float:
        return self.offset_frames * config.FRAME_SECONDS

    @property
    def level(self) -> str:
        """``'strong'`` | ``'weak'`` | ``'none'``."""
        if (self.votes >= config.PAIR_COHERENCE_STRONG_VOTES
                and self.sharpness >= config.PAIR_COHERENCE_STRONG_SHARPNESS):
            return "strong"
        if (self.votes >= config.PAIR_COHERENCE_WEAK_VOTES
                and self.sharpness >= config.PAIR_COHERENCE_WEAK_SHARPNESS):
            return "weak"
        return "none"


@dataclass
class PairHit:
    file_id: int
    path: str
    duration: float
    alignment: env.Alignment
    coherence: Coherence = field(default_factory=Coherence)
    segments: Optional[env.SegmentReport] = None
    session: Optional[SessionScore] = None
    take: Optional[int] = None
    role: Optional[str] = None
    is_take_mate: bool = False
    is_seed_path: bool = False
    missing: bool = False

    @property
    def envelope_is_trusted_alone(self) -> bool:
        """Is this overlap long enough for the envelope to speak unsupported?

        Below ``config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS`` the measured
        negative distribution reaches into the PAIR range (0.838 at a 5-minute
        overlap, 0.804 at fifteen), so an envelope score on its own is not
        evidence of a pair however high it is.
        """
        return (self.alignment.ok
                and self.alignment.overlap_seconds
                >= config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS)

    @property
    def capped_by_overlap(self) -> bool:
        """True when only the length gate is keeping this hit off PAIR."""
        return (self.alignment.ok
                and self.alignment.score >= config.PAIR_R_STRONG
                and self.coherence.level != "strong"
                and not self.envelope_is_trusted_alone)

    @property
    def verdict(self) -> str:
        """``'PAIR'`` | ``'TIMELINE MATCH'`` | ``'weak'``.

        Two independent routes to PAIR, because one rig and two rigs leave
        different evidence:

        * coherence is strong -- the two files share landmarks that all agree
          on one line through (t_seed, t_lib), which essentially cannot happen
          by chance -- *and* the envelope agrees it is the same timeline; or
        * the envelope correlation alone is high enough that no measured
          unrelated pair comes near it (``config.PAIR_R_STRONG``) **and** the
          two files overlap for long enough that this is a meaningful thing to
          say (``config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS``).

        That second condition is the length gate, and it is not decoration.
        Over ~15 000 - 22 000 genuine negative pairs the envelope score of two
        *unrelated* recordings reaches 0.838 on a 5-minute overlap and 0.804 on
        a 15-minute one -- above the PAIR bar, on real audio, with the segment
        and session lines agreeing.  Since coherence can only ever confirm the
        envelope and never contradict it, nothing downstream could have caught
        those.  Below 20 minutes of overlap, therefore, the envelope alone can
        say TIMELINE MATCH and no more.

        TIMELINE MATCH is everything above ``PAIR_R_LIKELY`` that is not a
        PAIR, including a high-scoring hit held back by the gate.  It is also
        the honest verdict for a genuine different-recorder capture over a
        short overlap: the loudness timelines line up, and that is all anybody
        can tell from here.
        """
        r = self.alignment.score
        if not self.alignment.ok:
            return "weak"
        # Strong coherence is shared *audio detail*, not shared loudness, and
        # is what the gate is asking for; it therefore lifts a hit to PAIR at
        # any overlap, exactly as it did before the gate existed.
        if self.coherence.level == "strong" and r >= config.PAIR_R_LIKELY:
            return "PAIR"
        if r >= config.PAIR_R_STRONG and self.envelope_is_trusted_alone:
            return "PAIR"
        if r >= config.PAIR_R_LIKELY:
            return "TIMELINE MATCH"
        return "weak"

    @property
    def evidence(self) -> list[str]:
        """One line per piece of evidence actually behind this verdict."""
        a = self.alignment
        out: list[str] = []
        if a.ok:
            out.append(
                f"envelope r={a.raw_r:+.2f} at lag {a.lag_seconds:+.0f}s "
                f"(scored {a.score:+.2f} over a {a.overlap_seconds:.0f}s "
                f"overlap)")
        else:
            out.append(f"no envelope alignment: {a.reason}")

        c = self.coherence
        if c.level == "none":
            out.append(
                "acoustic coherence: none -- consistent with a capture on "
                "different equipment (or with no shared audio at all)")
        else:
            if not c.drift_measurable:
                drift = ", clock drift not measurable on a seed this short"
            elif c.drift_ppm:
                drift = (f", clock drift {c.drift_ppm:+.0f} ppm "
                         f"(+/-{c.drift_resolution_ppm:.0f})")
            else:
                drift = (f", no clock drift above the "
                         f"{c.drift_resolution_ppm:.0f} ppm this seed length "
                         f"can resolve")
            out.append(
                f"acoustic coherence: {c.level} ({c.votes} aligned landmark "
                f"votes, {c.sharpness:.1f}x sharpness, offset "
                f"{c.offset_seconds:+.2f}s{drift})")

        if a.ok and (a.overlap_seconds
                     < config.PAIR_UNRELIABLE_OVERLAP_SECONDS):
            # Louder than the gate caution, and unconditional: at this length
            # the correlation is unreliable in *both* directions, so it is not
            # only the high scores that need a health warning.
            out.append(
                f"very short overlap ({a.overlap_seconds:.0f}s): below "
                f"{config.PAIR_UNRELIABLE_OVERLAP_SECONDS:.0f}s this "
                f"correlation is unreliable in both directions -- true pairs "
                f"have scored negative at the correct lag and unrelated files "
                f"have produced contradictory matches -- and envelope-only "
                f"verdicts are capped at TIMELINE MATCH below "
                f"{config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS:.0f}s; seed "
                f"with the whole file")
        elif self.capped_by_overlap:
            out.append(
                f"short overlap ({a.overlap_seconds:.0f}s): envelope-only "
                f"verdicts are capped at TIMELINE MATCH below "
                f"{config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS:.0f}s, because "
                f"unrelated recordings score this high over an overlap this "
                f"short; for full confidence seed with the whole file")
        elif (c.level == "none" and self.envelope_is_trusted_alone
                and a.score >= config.PAIR_R_STRONG):
            # Informational only -- this never changes the verdict.  A
            # long-overlap envelope match with no shared landmarks at all is
            # exactly what a second recorder looks like, and exactly what a
            # mislabelled "dual-record mate" would look like too.
            out.append(
                "note: no shared-clock evidence despite a long overlap -- "
                "expected for different-recorder captures; suspicious if "
                "these files should share a recorder")

        if self.is_take_mate:
            out.append(f"filename: dual-record pair-mate of the seed "
                       f"(take {self.take:04d} {self.role})")
        elif self.take is not None:
            out.append(f"filename: Tascam take {self.take:04d}"
                       f"{' ' + self.role if self.role else ''}")
        if self.session is not None:
            out.append(f"session signature: {self.session.total:.2f} "
                       f"(mode 2's score, as supporting evidence only)")
        if self.segments is not None:
            out.append(self.segments.text)
        if self.is_seed_path:
            out.append("this is the seed file itself")
        return out


@dataclass
class QueryResult:
    seed_path: str
    seed_seconds: float
    seed_signature: Optional[Signature] = None
    matches: list[MatchHit] = field(default_factory=list)
    sessions: list[SessionHit] = field(default_factory=list)
    pairs: list[PairHit] = field(default_factory=list)
    pair_note: str = ""
    probes_run: list[str] = field(default_factory=list)
    seed_hash_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Mode 1
# --------------------------------------------------------------------------


def _histogram(file_ids: np.ndarray, deltas: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(unique_keys, counts, smoothed_counts)`` sorted by key."""
    keys = file_ids.astype(np.int64) * _FILE_STRIDE + (deltas + _OFFSET_BIAS)
    uniq, counts = np.unique(keys, return_counts=True)
    if config.OFFSET_SMOOTH <= 0:
        return uniq, counts, counts
    smoothed = counts.astype(np.int64).copy()
    for shift in range(1, config.OFFSET_SMOOTH + 1):
        for sign in (-shift, shift):
            idx = np.searchsorted(uniq, uniq + sign)
            valid = (idx < uniq.size)
            idx_c = np.clip(idx, 0, uniq.size - 1)
            hit = valid & (uniq[idx_c] == uniq + sign)
            smoothed[hit] += counts[idx_c[hit]]
    return uniq, counts, smoothed


def _search_hashes(db: Database, hashes: np.ndarray, times: np.ndarray,
                   live: set[int]) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
    """Look the seed's hashes up and return (file_id, delta, seed_t) arrays."""
    if hashes.size == 0:
        z = np.zeros(0, np.int64)
        return z, z, z

    order = np.argsort(hashes, kind="stable")
    h_sorted = hashes[order]
    t_sorted = times[order].astype(np.int64)
    uniq, starts, counts = np.unique(h_sorted, return_index=True,
                                     return_counts=True)

    # Postings arrive as numpy blocks with the "uninformatively long posting
    # list" cap (hum, hiss, room tone) already applied by sqlite, and each
    # block is filtered down to live files before it is retained -- so peak
    # memory is one block, not the whole posting set.
    live_arr = (np.fromiter(live, dtype=np.int64, count=len(live))
                if live else None)
    if live_arr is not None:
        live_arr.sort()
    blocks_h: list[np.ndarray] = []
    blocks_f: list[np.ndarray] = []
    blocks_t: list[np.ndarray] = []
    for bh, bf, bt in db.iter_postings(
            uniq, max_postings=config.MAX_POSTINGS_PER_HASH):
        if live_arr is not None:
            pos = np.searchsorted(live_arr, bf)
            keep = ((pos < live_arr.size)
                    & (live_arr[np.clip(pos, 0, live_arr.size - 1)] == bf))
            if not keep.all():
                bh, bf, bt = bh[keep], bf[keep], bt[keep]
            if bh.size == 0:
                continue
        blocks_h.append(bh)
        blocks_f.append(bf)
        blocks_t.append(bt)

    if not blocks_h:
        z = np.zeros(0, np.int64)
        return z, z, z

    lib_h = np.concatenate(blocks_h)
    lib_f = np.concatenate(blocks_f)
    lib_t = np.concatenate(blocks_t)
    del blocks_h, blocks_f, blocks_t

    # Cross-join seed occurrences x library postings, per hash.
    pos = np.searchsorted(uniq, lib_h)
    valid = (pos < uniq.size) & (uniq[np.clip(pos, 0, uniq.size - 1)] == lib_h)
    lib_h, lib_f, lib_t, pos = (lib_h[valid], lib_f[valid], lib_t[valid],
                                pos[valid])
    seed_counts = counts[pos]
    rep_f = np.repeat(lib_f, seed_counts)
    rep_t = np.repeat(lib_t, seed_counts)
    seed_idx = (np.repeat(starts[pos], seed_counts)
                + _ragged_arange(seed_counts))
    seed_t = t_sorted[seed_idx]
    return rep_f, rep_t - seed_t, seed_t


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """[0,1,..,c0-1, 0,1,..,c1-1, ...] for the given counts."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    starts = np.cumsum(counts)[:-1]
    out[0] = 0
    if starts.size:
        out[starts] = 1 - counts[:-1]
    return np.cumsum(out)


def match_search(db: Database, seed_path: str, *, top: int = 10,
                 probes: Iterable[tuple[str, float]] = config.SR_PROBES,
                 warn: Optional[Callable[[str], None]] = None,
                 ) -> tuple[list[MatchHit], float, dict[str, int],
                            Optional[Signature]]:
    """Run the constellation search across every sample-rate probe."""
    warn = warn or (lambda msg: None)
    live = db.live_ids()
    paths = db.file_paths()
    meta = {r.id: r for r in db.live_files()}

    best: dict[int, MatchHit] = {}
    seed_seconds = 0.0
    hash_counts: dict[str, int] = {}
    seed_sig: Optional[Signature] = None

    for label, ratio in probes:
        h, t, sig, secs = analyze_seed(seed_path, rate_ratio=ratio)
        if ratio == 1.0:
            seed_seconds = secs
            seed_sig = sig
        if h.size > MAX_SEED_HASHES:
            step = int(math.ceil(h.size / MAX_SEED_HASHES))
            warn(f"seed produced {h.size:,} landmarks, above the "
                 f"{MAX_SEED_HASHES:,} cap: keeping every {step}th. "
                 f"Scores are lower than an uncapped query would give; "
                 f"seed with a shorter excerpt for comparable numbers.")
            h, t = h[::step], t[::step]
        hash_counts[label] = int(h.size)

        file_ids, deltas, seed_t = _search_hashes(db, h, t, live)
        if file_ids.size == 0:
            continue
        uniq, counts, smoothed = _histogram(file_ids, deltas)
        keys_file = uniq // _FILE_STRIDE
        keys_off = (uniq % _FILE_STRIDE) - _OFFSET_BIAS

        # Group the *postings* by file_id once, so that matched_seconds can be
        # computed from a slice per file.  Testing `file_ids == fid` inside
        # the loop instead was O(reported_files x postings) and dominated the
        # query on a large index.
        p_order = np.argsort(file_ids, kind="stable")
        p_files = file_ids[p_order]
        p_deltas = deltas[p_order]
        p_seed_t = seed_t[p_order]

        # Group the (file, offset) bins by file, each group ordered by
        # descending vote count, so the winning bin of every file is first.
        order = np.lexsort((-smoothed, keys_file))
        f_sorted = keys_file[order]
        _, first, group_len = np.unique(f_sorted, return_index=True,
                                        return_counts=True)
        totals = np.zeros(int(f_sorted.max()) + 1, dtype=np.int64)
        np.add.at(totals, keys_file, counts)

        for i, glen in zip(first, group_len):
            fid = int(f_sorted[i])
            votes = int(smoothed[order[i]])
            if votes < config.REPORT_MIN_VOTES:
                continue
            off = int(keys_off[order[i]])

            # The tallest *other* bin in the same file is that file's own
            # coincidence noise floor; a true alignment towers over it.
            grp = order[i:i + glen]
            grp_off = keys_off[grp]
            away = np.abs(grp_off - off) > 2 * config.OFFSET_SMOOTH + 1
            background = int(smoothed[grp[away]].max()) if away.any() else 0

            lo = int(np.searchsorted(p_files, fid, side="left"))
            hi = int(np.searchsorted(p_files, fid, side="right"))
            sel = np.abs(p_deltas[lo:hi] - off) <= config.OFFSET_SMOOTH
            if not sel.any():
                continue
            span = p_seed_t[lo:hi][sel]
            matched = float((span.max() - span.min() + 1)
                            * config.FRAME_SECONDS)
            row = meta.get(fid)
            hit = MatchHit(
                file_id=fid, path=paths.get(fid, "?"), votes=votes,
                total_votes=int(totals[fid]), background=background,
                offset_frames=off,
                probe=label, ratio=ratio, matched_seconds=matched,
                library_duration=row.duration if row else 0.0,
                take=row.take if row else None,
                role=row.role if row else None)
            prev = best.get(fid)
            if prev is None or hit.votes > prev.votes:
                best[fid] = hit

    hits = sorted(best.values(), key=lambda x: (-x.votes, x.path))[:top]
    # Only stat the files actually reported: a library file may have been
    # deleted or renamed since it was indexed, and presenting it as live is
    # dishonest.  `audio-match index` prunes these, but the index can always
    # be older than the filesystem.
    for hit in hits:
        hit.missing = not os.path.exists(hit.path)
    return hits, seed_seconds, hash_counts, seed_sig


# --------------------------------------------------------------------------
# Mode 2
# --------------------------------------------------------------------------


def session_search(db: Database, seed_sig: Signature, *, top: int = 10,
                   exclude_paths: Iterable[str] = (),
                   ignore_filenames: bool = False) -> list[SessionHit]:
    excl = set(exclude_paths)
    mate = pair_mate_role(seed_sig.role) if not ignore_filenames else None
    hits: list[SessionHit] = []
    for row in db.live_files():
        if row.path in excl:
            continue
        other = row.signature()
        score = compare(seed_sig, other,
                        ignore_filenames=ignore_filenames)
        is_mate = bool(
            seed_sig.take is not None and row.take == seed_sig.take
            and mate and row.role == mate)
        hits.append(SessionHit(
            file_id=row.id, path=row.path, score=score, duration=row.duration,
            sample_rate=row.sample_rate, channels=row.channels,
            bits=row.bits, take=row.take, role=row.role,
            is_pair_mate=is_mate))
    hits.sort(key=lambda h: (-h.score.total, h.path))
    hits = hits[:top]
    for hit in hits:
        hit.missing = not os.path.exists(hit.path)
    return hits


# --------------------------------------------------------------------------
# Mode 3
# --------------------------------------------------------------------------


def drift_slopes(seed_frames: int) -> np.ndarray:
    """Clock-drift slopes (ppm) worth trying for a seed of this length.

    Two slopes are only distinguishable when they move votes by more than one
    *smoothed* histogram bin (``2 * OFFSET_SMOOTH + 1`` frames) across the
    whole seed, so the grid step is the larger of ``PAIR_DRIFT_MIN_STEP_PPM``
    and that.  When the step is so coarse that fewer than two of them fit
    inside ``PAIR_MAX_DRIFT_PPM``, no drift is fitted at all and the caller
    reports "not measurable": announcing "-300 ppm" off a 150-second seed
    would be invention.  A 45-minute seed gets the full +/-300 ppm grid.

    Ordered by increasing magnitude, so ties resolve toward "no drift" -- the
    more conservative claim.
    """
    return drift_grid(seed_frames)[0]


def drift_grid(seed_frames: int) -> tuple[np.ndarray, float]:
    """``(slopes_ppm, step_ppm)``.  ``step_ppm`` is 0 when no fit is possible.

    The step is also the honest resolution to quote alongside any estimate:
    a drift smaller than one step is, at this seed length, not a thing this
    tool can see.
    """
    if seed_frames <= 0:
        return np.zeros(1, dtype=np.float64), 0.0
    bin_frames = float(2 * config.OFFSET_SMOOTH + 1)
    ppm_per_bin = 1e6 * bin_frames / float(seed_frames)
    step = max(config.PAIR_DRIFT_MIN_STEP_PPM, ppm_per_bin)
    n = int(config.PAIR_MAX_DRIFT_PPM // step)
    n = min(n, (config.PAIR_MAX_DRIFT_SLOPES - 1) // 2)
    if n < 2:
        return np.zeros(1, dtype=np.float64), 0.0
    mags = np.arange(1, n + 1, dtype=np.float64) * step
    slopes = np.concatenate([[0.0],
                             np.stack([mags, -mags], axis=1).ravel()])
    return slopes, step


def fit_coherence(seed_t: np.ndarray, deltas: np.ndarray, *,
                  seed_frames: int, center_frames: int) -> Coherence:
    """Sharpen the offset histogram by fitting a clock-drift slope.

    A single recorder's landmarks all land in one offset bin.  Two recorders
    whose sample clocks differ by ``s`` ppm put theirs on the *line*
    ``t_lib = (1 + s) * t_seed + offset``, which smears that bin across
    ``s * duration`` frames.  Compensating the seed timestamps by each
    candidate slope and keeping the sharpest histogram both recovers the votes
    and measures ``s``.

    Only the window ``+/- PAIR_COHERENCE_WINDOW_SECONDS`` around the lag the
    envelope proposed is examined: this is *confirmation* of that specific
    alignment, not an independent search, and bounding the window is what keeps
    a 45-minute seed's drift fit cheap.  The sharpness denominator is therefore
    the tallest competing bin inside the window, which is a stricter reference
    than mode 1's whole-file background, not a looser one.

    A non-zero slope is only *believed* if it beats the zero slope by
    ``PAIR_DRIFT_MIN_GAIN``.  Compensating for a drift that was never there
    still wins the occasional vote by luck, and without that bar the fit
    cheerfully reported "+77 ppm" for two files recorded off the same crystal.
    """
    half = int(round(config.PAIR_COHERENCE_WINDOW_SECONDS
                     / config.FRAME_SECONDS))
    width = 2 * half + 1
    slopes, step = drift_grid(seed_frames)
    guard = 2 * config.OFFSET_SMOOTH + 1
    kernel = np.ones(guard, dtype=np.int64)
    blank = Coherence(slopes_tried=int(slopes.size),
                      drift_measurable=bool(slopes.size > 1),
                      drift_resolution_ppm=step)
    if deltas.size == 0:
        return blank

    seed_f = seed_t.astype(np.float64)
    best = blank
    zero_votes = 0
    # ``slopes`` is ordered by magnitude with 0.0 first, so the zero slope's
    # vote count is known before any other slope is judged against it.
    for ppm in slopes:
        shifted = deltas - (ppm * 1e-6) * seed_f
        d = np.rint(shifted).astype(np.int64) - center_frames
        sel = (d >= -half) & (d <= half)
        if not sel.any():
            continue
        counts = np.bincount(d[sel] + half, minlength=width)
        smoothed = (counts if config.OFFSET_SMOOTH <= 0
                    else np.convolve(counts, kernel, mode="same"))
        peak = int(np.argmax(smoothed))
        votes = int(smoothed[peak])
        if ppm == 0.0:
            zero_votes = votes
        elif votes < zero_votes * config.PAIR_DRIFT_MIN_GAIN:
            continue                  # not enough gain to be a measurement
        if votes <= best.votes:
            continue                  # ties keep the smaller |ppm|
        lo, hi = max(0, peak - guard), min(width, peak + guard + 1)
        away = np.concatenate([smoothed[:lo], smoothed[hi:]])
        best = Coherence(
            votes=votes,
            background=int(away.max()) if away.size else 0,
            offset_frames=peak - half + center_frames,
            drift_ppm=float(ppm),
            slopes_tried=int(slopes.size),
            drift_measurable=bool(slopes.size > 1),
            drift_resolution_ppm=step)
    return best


def pair_search(db: Database, seed_path: str, *, top: int = 10,
                ignore_filenames: bool = False,
                warn: Optional[Callable[[str], None]] = None,
                ) -> tuple[list[PairHit], float, Optional[Signature], str]:
    """Find files that captured the same session timeline as ``seed_path``.

    Returns ``(hits, seed_seconds, seed_signature, note)``; ``note`` is a
    human-readable explanation when the mode could not run at all.
    """
    warn = warn or (lambda msg: None)
    seed = analyze_seed_full(seed_path)
    seed_codes = seed.envelope
    seed_db = env.dequantize(seed_codes)

    missing = db.count_missing_envelopes()
    if missing:
        warn(f"{missing:,} indexed file(s) have no activity envelope and are "
             f"invisible to pair mode -- run 'audio-match backfill' once to "
             f"fill them in (no re-fingerprinting, no re-index)")

    if seed_codes.size < env.samples(config.PAIR_MIN_ENVELOPE_SECONDS):
        note = (f"seed is {seed.seconds:.0f}s long; pair mode needs at least "
                f"{config.PAIR_MIN_ENVELOPE_SECONDS:.0f}s of audio, because "
                f"below that the 1 Hz envelope has too few points for a "
                f"correlation to mean anything")
        return [], seed.seconds, seed.signature, note

    rows = list(db.iter_envelopes())
    if not rows:
        note = ("no indexed file has an activity envelope; run "
                "'audio-match backfill' (or re-index) before using pair mode")
        return [], seed.seconds, seed.signature, note

    cand_db = [env.dequantize(r.envelope) for r in rows]
    alignments = env.align_many(seed_db, cand_db)
    order = sorted(range(len(rows)),
                   key=lambda i: (-alignments[i].score, rows[i].path))

    n_report = max(top, config.PAIR_COHERENCE_CANDIDATES)
    chosen = [i for i in order[:n_report] if alignments[i].ok]
    if not chosen:
        reasons = {a.reason for a in alignments if a.reason}
        note = ("no indexed file overlaps the seed enough to be compared"
                + (f" ({sorted(reasons)[0]})" if reasons else ""))
        return [], seed.seconds, seed.signature, note

    # -- evidence 2: constellation coherence, for the top candidates only.
    #
    # "Top candidates" means *every candidate that will be reported* --
    # ``chosen``, which is already capped at ``max(top,
    # PAIR_COHERENCE_CANDIDATES)``.  Capping the coherence set at 20 while
    # ``--top 25`` reported 25 made the 21st hit say "acoustic coherence:
    # none -- consistent with a capture on different equipment", which is a
    # claim about landmarks that were never looked at.  A verdict line may
    # only report evidence that was actually sought.
    #
    # The posting lookup restricts *results* to the candidate ids but still
    # applies MAX_POSTINGS_PER_HASH across the whole library, exactly as mode 1
    # does.  That is deliberate: a hash that occurs in thousands of files is
    # hum or hiss, and it says nothing about these two files just because only
    # two of them are being looked at right now.
    coherence: dict[int, Coherence] = {}
    ids = {rows[i].id for i in chosen}
    seed_frames = int(round(seed.seconds * config.FRAME_RATE))
    if seed.hashes.size and ids:
        h, t = seed.hashes, seed.times
        if h.size > MAX_SEED_HASHES:
            step = int(math.ceil(h.size / MAX_SEED_HASHES))
            warn(f"seed produced {h.size:,} landmarks, above the "
                 f"{MAX_SEED_HASHES:,} cap: keeping every {step}th for the "
                 f"coherence check.")
            h, t = h[::step], t[::step]
        file_ids, deltas, seed_t = _search_hashes(db, h, t, ids)
        if file_ids.size:
            p_order = np.argsort(file_ids, kind="stable")
            p_files = file_ids[p_order]
            p_deltas = deltas[p_order]
            p_seed_t = seed_t[p_order]
            lag_frames = {rows[i].id: int(round(alignments[i].lag_seconds
                                                / config.FRAME_SECONDS))
                          for i in chosen}
            for fid in ids:
                lo = int(np.searchsorted(p_files, fid, side="left"))
                hi = int(np.searchsorted(p_files, fid, side="right"))
                if hi <= lo:
                    continue
                coherence[fid] = fit_coherence(
                    p_seed_t[lo:hi], p_deltas[lo:hi],
                    seed_frames=seed_frames,
                    center_frames=lag_frames.get(fid, 0))

    # -- supporting evidence, for every candidate that reached this far.
    seed_path_abs = os.path.abspath(seed_path)
    mate_role = (pair_mate_role(seed.signature.role)
                 if not ignore_filenames else None)
    hits: list[PairHit] = []
    for i in chosen:
        row = rows[i]
        a = alignments[i]
        meta = db.file_row(row.id)
        session = (compare(seed.signature, meta.signature(),
                           ignore_filenames=ignore_filenames)
                   if meta is not None else None)
        hits.append(PairHit(
            file_id=row.id, path=row.path, duration=row.duration,
            alignment=a, coherence=coherence.get(row.id, Coherence()),
            segments=env.compare_segments(seed_db, cand_db[i], a.lag),
            session=session, take=row.take, role=row.role,
            is_take_mate=bool(
                not ignore_filenames and seed.signature.take is not None
                and row.take == seed.signature.take
                and mate_role and row.role == mate_role),
            is_seed_path=os.path.abspath(row.path) == seed_path_abs,
            missing=not os.path.exists(row.path)))

    # Candidates were *generated* by envelope score alone, but they are
    # *reported* verdict-first.  A file whose landmarks line up on a drifting
    # line with the seed is as close to proof as this tool gets, and burying
    # it under a slightly higher envelope score with no second evidence behind
    # it would be the wrong way round.
    rank = {"PAIR": 0, "TIMELINE MATCH": 1, "weak": 2}
    hits.sort(key=lambda h: (rank[h.verdict], -h.alignment.score, h.path))
    return hits[:top], seed.seconds, seed.signature, ""


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------


def run_query(db: Database, seed_path: str, *, mode: str = "both",
              top: int = 10, try_rates: bool = False,
              ignore_filenames: bool = False) -> QueryResult:
    """Run the requested query mode(s).

    ``mode='both'`` deliberately means *match + session* and nothing else.
    Pair matching is a separate, explicitly requested mode: it is a different
    question ("what else recorded this hour?") with a different answer shape,
    and quietly bolting it onto the default would have changed the output of
    every existing invocation.

    ``try_rates`` (the ``--try-rates`` flag) additionally re-decodes the seed
    at the 44.1/48 kHz ratios.  It is **off by default**: it triples the seed
    decode cost and only helps when the seed's own header may be wrong.
    """
    result = QueryResult(seed_path=seed_path, seed_seconds=0.0)
    probes = config.SR_PROBES if try_rates else (config.SR_PROBES[0],)

    if mode == "pair":
        hits, secs, sig, note = pair_search(
            db, seed_path, top=top, ignore_filenames=ignore_filenames,
            warn=result.warnings.append)
        result.pairs = hits
        result.seed_seconds = secs
        result.seed_signature = sig
        result.pair_note = note
        return result

    if mode in ("match", "both"):
        hits, secs, counts, sig = match_search(
            db, seed_path, top=top, probes=probes,
            warn=result.warnings.append)
        result.matches = hits
        result.seed_seconds = secs
        result.seed_hash_counts = counts
        result.seed_signature = sig
        result.probes_run = [p[0] for p in probes]

    if mode in ("session", "both"):
        sig = result.seed_signature
        if sig is None:
            _h, _t, sig, secs = analyze_seed(seed_path, rate_ratio=1.0)
            result.seed_signature = sig
            if not result.seed_seconds:
                result.seed_seconds = secs
        result.sessions = session_search(
            db, sig, top=top, ignore_filenames=ignore_filenames)

    return result
