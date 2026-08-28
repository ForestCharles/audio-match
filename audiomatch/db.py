"""SQLite storage.

Two versions, deliberately
--------------------------
``config.SCHEMA_VERSION`` means "the stored fingerprints themselves changed":
the only cure is deleting the database and re-indexing, and that is what the
error says.  ``config.STORAGE_VERSION`` means "the table layout grew something
new that the existing fingerprints are still valid under".  Storage upgrades
are applied in place by :meth:`Database._migrate` -- an additive
``ALTER TABLE`` -- because telling somebody to re-read 1.45 TB to gain a column
that a 3x cheaper decode could fill would be a lie.  ``audio-match backfill``
fills the new column for the rows that predate it.

Layout notes
------------
``hashes`` is a ``WITHOUT ROWID`` table whose primary key *is* the whole row,
so SQLite stores exactly one B-tree instead of a table plus a covering index.
That roughly halves the bytes per landmark (~16 B instead of ~30 B) and is the
single biggest factor in keeping a 2500-hour library under a couple of GB.

There is deliberately **no index on ``hashes.file_id``**.  Adding one would
double the database size, and it would only ever be used to delete a file's
landmarks.  Instead, re-indexing a changed file marks the old ``files`` row
dead and inserts a new one with a fresh ``file_id``; queries skip dead ids, and
``audio-match purge`` reclaims the space in a single sequential rewrite.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

from . import config
from .envelope import pack as envelope_blob, unpack as envelope_unblob
from .session import Signature

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    alive        INTEGER NOT NULL DEFAULT 1,
    size         INTEGER,
    mtime        REAL,
    status       TEXT,            -- 'ok' | 'error'
    error        TEXT,
    duration     REAL,
    sample_rate  INTEGER,
    channels     INTEGER,
    bits         INTEGER,
    codec        TEXT,
    take         INTEGER,
    role         TEXT,
    n_hashes     INTEGER DEFAULT 0,
    noise        BLOB,
    hum          BLOB,
    chan         BLOB,
    envelope     BLOB,            -- uint8 1 Hz activity envelope; NULL = never
                                  -- computed (pre-v2 row) -> run 'backfill'
    indexed_at   REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_files_path
    ON files(path) WHERE alive = 1;

CREATE TABLE IF NOT EXISTS hashes (
    hash    INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    t       INTEGER NOT NULL,
    PRIMARY KEY (hash, file_id, t)
) WITHOUT ROWID;
"""


@dataclass
class FileRow:
    id: int
    path: str
    size: int
    mtime: float
    status: str
    error: Optional[str]
    duration: float
    sample_rate: int
    channels: int
    bits: int
    codec: str
    take: Optional[int]
    role: Optional[str]
    n_hashes: int
    noise: np.ndarray
    hum: np.ndarray
    chan: np.ndarray

    def signature(self) -> Signature:
        return Signature(
            noise=self.noise, hum=self.hum, chan=self.chan,
            sample_rate=self.sample_rate, channels=self.channels,
            bits=self.bits, duration=self.duration,
            take=self.take, role=self.role,
        )


@dataclass
class EnvelopeRow:
    """A library file's activity envelope, plus what pair mode needs with it."""

    id: int
    path: str
    duration: float
    take: Optional[int]
    role: Optional[str]
    envelope: np.ndarray        # uint8 codes; dequantise for correlation


def _blob(a: Optional[np.ndarray]) -> bytes:
    if a is None:
        return b""
    return np.asarray(a, dtype="<f4").tobytes()


def _unblob(b: Optional[bytes], n: int) -> np.ndarray:
    if not b:
        return np.zeros(n, dtype=np.float32)
    a = np.frombuffer(b, dtype="<f4")
    if a.size < n:
        a = np.concatenate([a, np.zeros(n - a.size, np.float32)])
    return np.array(a[:n], dtype=np.float32)


class Database:
    """Thin wrapper over the sqlite connection used by index and query."""

    def __init__(self, path: str, *, create: bool = True):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if create and parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if not create and not os.path.exists(path):
            raise FileNotFoundError(
                f"no database at {path!r} -- run 'audio-match index' first")
        self.conn = sqlite3.connect(path, timeout=120.0)
        self.conn.executescript(SCHEMA)
        self._check_version()
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def _meta(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))

    def _check_version(self) -> None:
        stored = self._meta("schema_version")
        if stored is None:
            self._set_meta("schema_version", str(config.SCHEMA_VERSION))
            self.conn.commit()
            return
        if int(stored) != config.SCHEMA_VERSION:
            raise RuntimeError(
                f"database {self.path!r} was built by fingerprint schema "
                f"v{stored}, this build is v{config.SCHEMA_VERSION}. "
                f"Delete it and re-index.")

    def _migrate(self) -> None:
        """Bring an older *storage* layout up to date, in place.

        Additive only.  Every migration here must leave the landmarks and the
        session signatures byte-identical -- if a change cannot promise that,
        it belongs behind ``SCHEMA_VERSION`` and a re-index instead.
        """
        stored = self._meta("storage_version")
        version = int(stored) if stored is not None else 1
        if version > config.STORAGE_VERSION:
            raise RuntimeError(
                f"database {self.path!r} uses storage layout v{version}, "
                f"newer than this build's v{config.STORAGE_VERSION}. "
                f"Upgrade audio-match, or delete the database and re-index.")

        if version < 2:
            # v2: the 1 Hz activity envelope used by pair mode.  Existing rows
            # get NULL, which is exactly what `audio-match backfill` looks for.
            columns = {r[1] for r in
                       self.conn.execute("PRAGMA table_info(files)")}
            if "envelope" not in columns:
                self.conn.execute("ALTER TABLE files ADD COLUMN envelope BLOB")

        if stored is None or version != config.STORAGE_VERSION:
            self._set_meta("storage_version", str(config.STORAGE_VERSION))
            self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def begin_bulk(self) -> None:
        """Loosen durability for the bulk index pass."""
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -131072")   # 128 MB page cache

    def existing_stamp(self, path: str) -> Optional[tuple[int, float, str]]:
        cur = self.conn.execute(
            "SELECT size, mtime, status FROM files "
            "WHERE alive = 1 AND path = ?", (path,))
        row = cur.fetchone()
        return (int(row[0] or 0), float(row[1] or 0.0), str(row[2] or ""))\
            if row else None

    def all_stamps(self) -> dict[str, tuple[int, float, str]]:
        """Every live file's (size, mtime, status), for fast resume checks."""
        cur = self.conn.execute(
            "SELECT path, size, mtime, status FROM files WHERE alive = 1")
        return {r[0]: (int(r[1] or 0), float(r[2] or 0.0), str(r[3] or ""))
                for r in cur}

    def alive_paths_under(self, root: str) -> list[str]:
        """Every live file path at or below ``root``.

        Matching is done in Python on an absolute, normalised prefix rather
        than with SQL ``LIKE``, which has no usable escaping for the ``%`` and
        ``_`` that occur constantly in real filenames.
        """
        root = os.path.abspath(root)
        prefix = root.rstrip(os.sep) + os.sep
        cur = self.conn.execute("SELECT path FROM files WHERE alive = 1")
        return [r[0] for r in cur
                if r[0] == root or r[0].startswith(prefix)]

    def retire(self, path: str) -> None:
        """Mark any live row for ``path`` dead (its landmarks become garbage)."""
        self.conn.execute(
            "UPDATE files SET alive = 0 WHERE alive = 1 AND path = ?", (path,))

    def add_file(self, *, path: str, size: int, mtime: float, status: str,
                 error: Optional[str], probe_duration: float,
                 sample_rate: int, channels: int, bits: int, codec: str,
                 sig: Optional[Signature],
                 hashes: Optional[np.ndarray] = None,
                 times: Optional[np.ndarray] = None,
                 envelope: Optional[np.ndarray] = None) -> int:
        self.retire(path)
        n_hashes = int(hashes.size) if hashes is not None else 0
        # ``None`` stores SQL NULL, which means "nobody has computed this
        # file's envelope yet" and is what `backfill` selects on.  An empty
        # array stores a zero-length blob, which means "computed, and there
        # was nothing there" -- a different fact, and not a backfill candidate.
        env_blob = None if envelope is None else envelope_blob(envelope)
        cur = self.conn.execute(
            "INSERT INTO files(path, alive, size, mtime, status, error, "
            " duration, sample_rate, channels, bits, codec, take, role, "
            " n_hashes, noise, hum, chan, envelope, indexed_at) "
            "VALUES(?,1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (path, size, mtime, status, error, probe_duration, sample_rate,
             channels, bits, codec,
             sig.take if sig else None, sig.role if sig else None,
             n_hashes,
             _blob(sig.noise if sig else None),
             _blob(sig.hum if sig else None),
             _blob(sig.chan if sig else None),
             env_blob,
             time.time()))
        file_id = int(cur.lastrowid)
        if hashes is not None and times is not None and hashes.size:
            self.conn.executemany(
                "INSERT OR IGNORE INTO hashes(hash, file_id, t) "
                "VALUES(?,?,?)",
                zip(hashes.tolist(),
                    [file_id] * int(hashes.size),
                    times.tolist()))
        return file_id

    def set_envelope(self, file_id: int, envelope: np.ndarray) -> None:
        """Fill in one row's envelope and touch *nothing else*.

        Backfill must not disturb ``size``, ``mtime``, ``n_hashes`` or a single
        landmark: a database that has been backfilled has to be byte-identical
        to one that was indexed with the envelope from the start, apart from
        this column.  Hence a targeted UPDATE rather than a re-``add_file``.
        """
        self.conn.execute("UPDATE files SET envelope = ? WHERE id = ?",
                          (envelope_blob(envelope), int(file_id)))

    def commit(self) -> None:
        self.conn.commit()

    # -- reads -------------------------------------------------------------

    _FILE_COLUMNS = ("id, path, size, mtime, status, error, duration, "
                     "sample_rate, channels, bits, codec, take, role, "
                     "n_hashes, noise, hum, chan")

    @staticmethod
    def _file_row(r: tuple) -> FileRow:
        return FileRow(
            id=int(r[0]), path=r[1], size=int(r[2] or 0),
            mtime=float(r[3] or 0.0), status=r[4], error=r[5],
            duration=float(r[6] or 0.0), sample_rate=int(r[7] or 0),
            channels=int(r[8] or 0), bits=int(r[9] or 0),
            codec=r[10] or "?", take=r[11], role=r[12],
            n_hashes=int(r[13] or 0),
            noise=_unblob(r[14], config.NOISE_BANDS),
            hum=_unblob(r[15], config.HUM_DIM),
            chan=_unblob(r[16], config.CHAN_DIM))

    def live_files(self, status: str = "ok") -> Iterator[FileRow]:
        cur = self.conn.execute(
            f"SELECT {self._FILE_COLUMNS} FROM files "
            f"WHERE alive = 1 AND status = ?", (status,))
        for r in cur:
            yield self._file_row(r)

    def file_row(self, file_id: int) -> Optional[FileRow]:
        """One file by id.  A primary-key lookup, for the handful of rows a
        pair query actually reports -- a full ``live_files()`` scan to read
        twenty signatures would be silly on a 2500-hour index."""
        r = self.conn.execute(
            f"SELECT {self._FILE_COLUMNS} FROM files WHERE id = ?",
            (int(file_id),)).fetchone()
        return None if r is None else self._file_row(r)

    def files_missing_envelope(self) -> list[tuple[int, str, int, float]]:
        """``(id, path, size, mtime)`` for every live, OK row with no envelope.

        Only ``status='ok'`` rows: a file that could not be decoded when it was
        indexed will not decode now either, and re-attempting it on every
        backfill run would turn a resumable pass into an unbounded one.
        ``--retry-errors`` on ``index`` is the right tool for those.
        """
        cur = self.conn.execute(
            "SELECT id, path, size, mtime FROM files "
            "WHERE alive = 1 AND status = 'ok' AND envelope IS NULL "
            "ORDER BY path")
        return [(int(r[0]), r[1], int(r[2] or 0), float(r[3] or 0.0))
                for r in cur]

    def count_missing_envelopes(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE alive = 1 AND status = 'ok' "
            "AND envelope IS NULL").fetchone()
        return int(row[0] or 0)

    def iter_envelopes(self) -> Iterator["EnvelopeRow"]:
        """Every live, OK file that has an envelope, with it decoded to dB.

        One byte per second means a 2500-hour library is ~9 MB of envelope in
        total, so pair mode can afford to hold all of them at once and hand the
        whole set to one batched FFT.
        """
        cur = self.conn.execute(
            "SELECT id, path, duration, take, role, envelope FROM files "
            "WHERE alive = 1 AND status = 'ok' AND envelope IS NOT NULL")
        for r in cur:
            codes = envelope_unblob(r[5])
            if codes.size == 0:
                continue
            yield EnvelopeRow(id=int(r[0]), path=r[1],
                              duration=float(r[2] or 0.0),
                              take=r[3], role=r[4], envelope=codes)

    def file_paths(self) -> dict[int, str]:
        cur = self.conn.execute(
            "SELECT id, path FROM files WHERE alive = 1")
        return {int(r[0]): r[1] for r in cur}

    def live_ids(self) -> set[int]:
        cur = self.conn.execute("SELECT id FROM files WHERE alive = 1")
        return {int(r[0]) for r in cur}

    def lookup(self, hash_values: Sequence[int],
               chunk: int = 900) -> Iterator[tuple[int, int, int]]:
        """Yield ``(hash, file_id, t)`` for every posting of the given hashes.

        Row-at-a-time; :meth:`iter_postings` is what queries actually use.
        """
        values = list(hash_values)
        for i in range(0, len(values), chunk):
            part = values[i:i + chunk]
            q = ("SELECT hash, file_id, t FROM hashes WHERE hash IN ("
                 + ",".join("?" * len(part)) + ")")
            yield from self.conn.execute(q, part)

    #: Rows pulled out of sqlite per fetch before conversion to numpy.  Bounds
    #: the number of Python integers alive at any one moment.
    POSTING_BLOCK = 32768

    def iter_postings(self, hash_values: Sequence[int], *,
                      max_postings: Optional[int] = None,
                      chunk: int = 900
                      ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(hash, file_id, t)`` numpy blocks for the given hashes.

        ``max_postings`` skips *entirely* any hash with more postings than
        that -- hum, hiss and room tone, which carry almost no information but
        dominate both the cost and the noise floor.  The cap is applied by a
        ``GROUP BY ... HAVING COUNT(*) >`` pre-filter **inside sqlite**, so an
        overfull posting list is never materialised in Python; the previous
        code read every posting into a list and only then discarded it, which
        on a hum-heavy seed meant hundreds of MB of transient Python ints.
        """
        values = sorted({int(h) for h in hash_values})
        for i in range(0, len(values), chunk):
            part = values[i:i + chunk]
            if max_postings is not None:
                marks = ",".join("?" * len(part))
                over = {int(r[0]) for r in self.conn.execute(
                    f"SELECT hash FROM hashes WHERE hash IN ({marks}) "
                    f"GROUP BY hash HAVING COUNT(*) > ?",
                    (*part, int(max_postings)))}
                if over:
                    part = [h for h in part if h not in over]
            if not part:
                continue
            marks = ",".join("?" * len(part))
            cur = self.conn.execute(
                f"SELECT hash, file_id, t FROM hashes WHERE hash IN ({marks})",
                part)
            while True:
                rows = cur.fetchmany(self.POSTING_BLOCK)
                if not rows:
                    break
                block = np.fromiter(
                    (v for row in rows for v in row), dtype=np.int64,
                    count=3 * len(rows)).reshape(len(rows), 3)
                del rows
                yield block[:, 0], block[:, 1], block[:, 2]

    def db_bytes(self) -> int:
        """On-disk size, WAL included (and checkpointed first so it is real)."""
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

    def stats(self) -> dict[str, int]:
        c = self.conn
        def one(sql: str, *a) -> int:
            r = c.execute(sql, a).fetchone()
            return int(r[0] or 0)
        return {
            "files_live": one("SELECT COUNT(*) FROM files WHERE alive=1"),
            "files_ok": one("SELECT COUNT(*) FROM files "
                            "WHERE alive=1 AND status='ok'"),
            "files_error": one("SELECT COUNT(*) FROM files "
                               "WHERE alive=1 AND status='error'"),
            "files_dead": one("SELECT COUNT(*) FROM files WHERE alive=0"),
            "files_no_envelope": one(
                "SELECT COUNT(*) FROM files WHERE alive=1 AND status='ok' "
                "AND envelope IS NULL"),
            "hashes": one("SELECT COUNT(*) FROM hashes"),
            "seconds": int(one(
                "SELECT CAST(COALESCE(SUM(duration),0) AS INTEGER) "
                "FROM files WHERE alive=1 AND status='ok'")),
            "bytes": self.db_bytes(),
        }

    def purge(self) -> tuple[int, int]:
        """Drop landmarks belonging to dead files, then VACUUM.

        Returns ``(rows_removed, files_removed)``.
        """
        live = self.live_ids()
        before = self.conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
        if live:
            # A parameter per live file overflows SQLITE_MAX_VARIABLE_NUMBER
            # (999 before sqlite 3.32, i.e. Ubuntu 20.04) on any real library,
            # so the id set goes through a temp table instead.
            self.conn.execute("DROP TABLE IF EXISTS temp._live_ids")
            self.conn.execute(
                "CREATE TEMP TABLE _live_ids (id INTEGER PRIMARY KEY)")
            self.conn.executemany(
                "INSERT INTO temp._live_ids(id) VALUES(?)",
                ((int(i),) for i in live))
            self.conn.execute(
                "DELETE FROM hashes WHERE file_id NOT IN "
                "(SELECT id FROM temp._live_ids)")
            self.conn.execute("DROP TABLE temp._live_ids")
        else:
            self.conn.execute("DELETE FROM hashes")
        n_dead = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE alive=0").fetchone()[0]
        self.conn.execute("DELETE FROM files WHERE alive = 0")
        self.conn.commit()
        self.conn.execute("VACUUM")
        after = self.conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
        return int(before - after), int(n_dead)


def open_db(path: str, *, create: bool = True) -> Database:
    return Database(path, create=create)
