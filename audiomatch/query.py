"""Searching the index.

Mode 1 (``match``)
    Landmark hashes from the seed are looked up in the index.  Every posting
    casts a vote for ``(library_file, t_library - t_seed)``.  A real match puts
    (almost) all of its votes in a single offset bin, because the whole seed is
    displaced by one constant amount; coincidental hash collisions scatter
    uniformly.  The height of that peak is the score.

Mode 2 (``session``)
    The seed's session signature is compared against every indexed file's.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from . import config
from .analyze import analyze_seed
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
class QueryResult:
    seed_path: str
    seed_seconds: float
    seed_signature: Optional[Signature] = None
    matches: list[MatchHit] = field(default_factory=list)
    sessions: list[SessionHit] = field(default_factory=list)
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
# Combined
# --------------------------------------------------------------------------


def run_query(db: Database, seed_path: str, *, mode: str = "both",
              top: int = 10, try_rates: bool = False,
              ignore_filenames: bool = False) -> QueryResult:
    """Run one or both query modes.

    ``try_rates`` (the ``--try-rates`` flag) additionally re-decodes the seed
    at the 44.1/48 kHz ratios.  It is **off by default**: it triples the seed
    decode cost and only helps when the seed's own header may be wrong.
    """
    result = QueryResult(seed_path=seed_path, seed_seconds=0.0)
    probes = config.SR_PROBES if try_rates else (config.SR_PROBES[0],)

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
