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
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from . import config
from .analyze import analyze_seed
from .db import Database
from .session import Signature, SessionScore, compare, pair_mate_role

#: Guard against pathological seeds (someone seeds with a 3-hour file).
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

    @property
    def concentration(self) -> float:
        return self.votes / self.total_votes if self.total_votes else 0.0

    @property
    def sharpness(self) -> float:
        """Winning bin height divided by the file's own next-tallest bin.

        This is the single most informative number: a genuine alignment is a
        spike, so its sharpness runs from ~5 to 50+, while an unrelated file's
        histogram is flat and lands near 1.
        """
        return self.votes / max(1, self.background)

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


@dataclass
class QueryResult:
    seed_path: str
    seed_seconds: float
    seed_signature: Optional[Signature] = None
    matches: list[MatchHit] = field(default_factory=list)
    sessions: list[SessionHit] = field(default_factory=list)
    probes_run: list[str] = field(default_factory=list)
    seed_hash_counts: dict[str, int] = field(default_factory=dict)


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

    rows_h: list[int] = []
    rows_f: list[int] = []
    rows_t: list[int] = []
    for h, fid, t in db.lookup(uniq.tolist()):
        rows_h.append(h)
        rows_f.append(fid)
        rows_t.append(t)
    if not rows_h:
        z = np.zeros(0, np.int64)
        return z, z, z

    lib_h = np.asarray(rows_h, dtype=np.int64)
    lib_f = np.asarray(rows_f, dtype=np.int64)
    lib_t = np.asarray(rows_t, dtype=np.int64)

    if live:
        keep = np.isin(lib_f, np.fromiter(live, dtype=np.int64,
                                          count=len(live)))
        lib_h, lib_f, lib_t = lib_h[keep], lib_f[keep], lib_t[keep]
        if lib_h.size == 0:
            z = np.zeros(0, np.int64)
            return z, z, z

    # Drop hashes whose posting list is uninformatively long (hum, hiss,
    # silence): they cost the most and contribute only noise.
    lib_uniq, lib_counts = np.unique(lib_h, return_counts=True)
    too_common = lib_uniq[lib_counts > config.MAX_POSTINGS_PER_HASH]
    if too_common.size:
        keep = ~np.isin(lib_h, too_common)
        lib_h, lib_f, lib_t = lib_h[keep], lib_f[keep], lib_t[keep]
        if lib_h.size == 0:
            z = np.zeros(0, np.int64)
            return z, z, z

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
                 ) -> tuple[list[MatchHit], float, dict[str, int],
                            Optional[Signature]]:
    """Run the constellation search across every sample-rate probe."""
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
            h, t = h[::step], t[::step]
        hash_counts[label] = int(h.size)

        file_ids, deltas, seed_t = _search_hashes(db, h, t, live)
        if file_ids.size == 0:
            continue
        uniq, counts, smoothed = _histogram(file_ids, deltas)
        keys_file = uniq // _FILE_STRIDE
        keys_off = (uniq % _FILE_STRIDE) - _OFFSET_BIAS

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

            sel = ((file_ids == fid)
                   & (np.abs(deltas - off) <= config.OFFSET_SMOOTH))
            if not sel.any():
                continue
            span = seed_t[sel]
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

    hits = sorted(best.values(), key=lambda x: (-x.votes, x.path))
    return hits[:top], seed_seconds, hash_counts, seed_sig


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
    return hits[:top]


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------


def run_query(db: Database, seed_path: str, *, mode: str = "both",
              top: int = 10, sr_probes: bool = True,
              ignore_filenames: bool = False) -> QueryResult:
    result = QueryResult(seed_path=seed_path, seed_seconds=0.0)
    probes = config.SR_PROBES if sr_probes else (config.SR_PROBES[0],)

    if mode in ("match", "both"):
        hits, secs, counts, sig = match_search(db, seed_path, top=top,
                                               probes=probes)
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
