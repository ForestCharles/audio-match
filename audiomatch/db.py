"""SQLite storage.

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

    # -- lifecycle ---------------------------------------------------------

    def _check_version(self) -> None:
        cur = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(config.SCHEMA_VERSION),))
            self.conn.commit()
            return
        if int(row[0]) != config.SCHEMA_VERSION:
            raise RuntimeError(
                f"database {self.path!r} was built by fingerprint schema "
                f"v{row[0]}, this build is v{config.SCHEMA_VERSION}. "
                f"Delete it and re-index.")

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
                 times: Optional[np.ndarray] = None) -> int:
        self.retire(path)
        n_hashes = int(hashes.size) if hashes is not None else 0
        cur = self.conn.execute(
            "INSERT INTO files(path, alive, size, mtime, status, error, "
            " duration, sample_rate, channels, bits, codec, take, role, "
            " n_hashes, noise, hum, chan, indexed_at) "
            "VALUES(?,1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (path, size, mtime, status, error, probe_duration, sample_rate,
             channels, bits, codec,
             sig.take if sig else None, sig.role if sig else None,
             n_hashes,
             _blob(sig.noise if sig else None),
             _blob(sig.hum if sig else None),
             _blob(sig.chan if sig else None),
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

    def commit(self) -> None:
        self.conn.commit()

    # -- reads -------------------------------------------------------------

    def live_files(self, status: str = "ok") -> Iterator[FileRow]:
        cur = self.conn.execute(
            "SELECT id, path, size, mtime, status, error, duration, "
            "sample_rate, channels, bits, codec, take, role, n_hashes, "
            "noise, hum, chan FROM files WHERE alive = 1 AND status = ?",
            (status,))
        for r in cur:
            yield FileRow(
                id=int(r[0]), path=r[1], size=int(r[2] or 0),
                mtime=float(r[3] or 0.0), status=r[4], error=r[5],
                duration=float(r[6] or 0.0), sample_rate=int(r[7] or 0),
                channels=int(r[8] or 0), bits=int(r[9] or 0),
                codec=r[10] or "?", take=r[11], role=r[12],
                n_hashes=int(r[13] or 0),
                noise=_unblob(r[14], config.NOISE_BANDS),
                hum=_unblob(r[15], config.HUM_DIM),
                chan=_unblob(r[16], config.CHAN_DIM))

    def file_paths(self) -> dict[int, str]:
        cur = self.conn.execute(
            "SELECT id, path FROM files WHERE alive = 1")
        return {int(r[0]): r[1] for r in cur}

    def live_ids(self) -> set[int]:
        cur = self.conn.execute("SELECT id FROM files WHERE alive = 1")
        return {int(r[0]) for r in cur}

    def lookup(self, hash_values: Sequence[int],
               chunk: int = 900) -> Iterator[tuple[int, int, int]]:
        """Yield ``(hash, file_id, t)`` for every posting of the given hashes."""
        values = list(hash_values)
        for i in range(0, len(values), chunk):
            part = values[i:i + chunk]
            q = ("SELECT hash, file_id, t FROM hashes WHERE hash IN ("
                 + ",".join("?" * len(part)) + ")")
            yield from self.conn.execute(q, part)

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
