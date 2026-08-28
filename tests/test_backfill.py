"""Storage v2 migration and the ``backfill`` pass.

The contract these tests exist to defend: a database built before pair mode
must gain the envelope column *without* re-fingerprinting anything.  Landmarks,
session signatures, sizes and mtimes have to come out byte-identical.
"""

from __future__ import annotations

import os
import shutil
import sqlite3

import numpy as np
import pytest

from audiomatch import config
from audiomatch.cli import main
from audiomatch.db import open_db
from audiomatch.indexer import run_backfill, run_index

from conftest import requires_ffmpeg
from test_index import synthetic_library

pytestmark = requires_ffmpeg


#: The ``files`` table exactly as storage v1 wrote it -- no ``envelope``.
V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    alive        INTEGER NOT NULL DEFAULT 1,
    size         INTEGER,
    mtime        REAL,
    status       TEXT,
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
CREATE UNIQUE INDEX ix_files_path ON files(path) WHERE alive = 1;
CREATE TABLE hashes (
    hash INTEGER NOT NULL, file_id INTEGER NOT NULL, t INTEGER NOT NULL,
    PRIMARY KEY (hash, file_id, t)
) WITHOUT ROWID;
"""


def _landmarks(db_path: str) -> list[tuple]:
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT hash, file_id, t FROM hashes "
                           "ORDER BY hash, file_id, t").fetchall()
    finally:
        con.close()


def _files_without_envelope(db_path: str) -> list[tuple]:
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(files)")
                if r[1] != "envelope"]
        return con.execute(
            f"SELECT {', '.join(cols)} FROM files ORDER BY id").fetchall()
    finally:
        con.close()


@pytest.fixture
def tone_db(tmp_path) -> tuple[str, str]:
    lib = synthetic_library(tmp_path / "lib", n=3, seconds=4.0)
    db_path = str(tmp_path / "tones.db")
    with open_db(db_path) as db:
        summary = run_index(db, lib, workers=1, progress_stream=None)
    assert summary["indexed"] == 3 and summary["errors"] == 0
    return lib, db_path


# --------------------------------------------------------------------------
# Indexing writes the envelope
# --------------------------------------------------------------------------


def test_indexing_stores_an_envelope_for_every_file(tone_db):
    _lib, db_path = tone_db
    with open_db(db_path, create=False) as db:
        assert db.count_missing_envelopes() == 0
        assert db.stats()["files_no_envelope"] == 0
        rows = list(db.iter_envelopes())
        assert len(rows) == 3
        for row in rows:
            # 4-second files: 4 one-second samples.
            assert row.envelope.size == 4
            assert row.envelope.dtype == np.uint8


def test_backfill_does_nothing_when_nothing_is_missing(tone_db):
    _lib, db_path = tone_db
    before = _landmarks(db_path)
    logged: list[str] = []
    with open_db(db_path, create=False) as db:
        summary = run_backfill(db, workers=1, progress_stream=None,
                               log=logged.append)
    assert summary == {"filled": 0, "errors": 0, "changed": 0, "missing": 0,
                       "bytes": 0, "seconds": 0.0, "aborted": False,
                       "considered": 0}
    assert any("already has an activity envelope" in m for m in logged)
    assert _landmarks(db_path) == before


# --------------------------------------------------------------------------
# Backfill fills only what is missing, and touches nothing else
# --------------------------------------------------------------------------


def test_backfill_fills_only_the_missing_rows_and_rewrites_nothing_else(
        tone_db):
    _lib, db_path = tone_db
    with open_db(db_path, create=False) as db:
        original_envelopes = {r.path: r.envelope.tobytes()
                              for r in db.iter_envelopes()}
    hashes_before = _landmarks(db_path)
    files_before = _files_without_envelope(db_path)

    # Simulate a database written before the envelope existed, for two of the
    # three files only.
    con = sqlite3.connect(db_path)
    victims = [r[0] for r in con.execute(
        "SELECT path FROM files ORDER BY path LIMIT 2")]
    con.execute("UPDATE files SET envelope = NULL WHERE path IN (?, ?)",
                victims)
    con.commit()
    con.close()

    with open_db(db_path, create=False) as db:
        assert db.count_missing_envelopes() == 2
        assert [p for _i, p, _s, _m in db.files_missing_envelope()] == \
            sorted(victims)
        summary = run_backfill(db, workers=2, progress_stream=None)

    assert summary["filled"] == 2
    assert summary["errors"] == 0 and summary["changed"] == 0
    assert summary["considered"] == 2

    # Landmarks byte-identical, every other files column byte-identical.
    assert _landmarks(db_path) == hashes_before
    assert _files_without_envelope(db_path) == files_before

    # And the refilled envelopes equal what the original index computed.
    with open_db(db_path, create=False) as db:
        assert db.count_missing_envelopes() == 0
        refilled = {r.path: r.envelope.tobytes()
                    for r in db.iter_envelopes()}
    assert refilled == original_envelopes


def test_backfill_is_resumable(tone_db):
    _lib, db_path = tone_db
    con = sqlite3.connect(db_path)
    con.execute("UPDATE files SET envelope = NULL")
    con.commit()
    con.close()

    with open_db(db_path, create=False) as db:
        # Fill one by hand, as an interrupted run would have.
        fid, _p, _s, _m = db.files_missing_envelope()[0]
        db.set_envelope(fid, np.array([1, 2, 3], dtype=np.uint8))
        db.commit()
        assert db.count_missing_envelopes() == 2
        summary = run_backfill(db, workers=1, progress_stream=None)
    assert summary["considered"] == 2 and summary["filled"] == 2
    with open_db(db_path, create=False) as db:
        assert db.count_missing_envelopes() == 0


def test_backfill_skips_a_file_that_changed_since_it_was_indexed(tone_db):
    lib, db_path = tone_db
    con = sqlite3.connect(db_path)
    con.execute("UPDATE files SET envelope = NULL")
    con.commit()
    con.close()

    victim = os.path.join(lib, "tone01.wav")
    with open(victim, "ab") as fh:
        fh.write(b"\0" * 8192)

    logged: list[str] = []
    with open_db(db_path, create=False) as db:
        summary = run_backfill(db, workers=1, progress_stream=None,
                               log=logged.append)
    assert summary["changed"] == 1
    assert summary["filled"] == 2
    assert any("changed since it was indexed" in m for m in logged)
    # The stale row keeps its NULL: an envelope computed from audio the
    # landmarks were not computed from would be worse than none.
    with open_db(db_path, create=False) as db:
        assert [p for _i, p, _s, _m in db.files_missing_envelope()] == [victim]


def test_backfill_skips_a_vanished_file(tone_db, tmp_path):
    lib, db_path = tone_db
    con = sqlite3.connect(db_path)
    con.execute("UPDATE files SET envelope = NULL")
    con.commit()
    con.close()

    shutil.move(os.path.join(lib, "tone02.wav"), str(tmp_path / "stash.wav"))
    logged: list[str] = []
    with open_db(db_path, create=False) as db:
        summary = run_backfill(db, workers=1, progress_stream=None,
                               log=logged.append)
    assert summary["missing"] == 1 and summary["filled"] == 2
    assert any("no longer exists" in m for m in logged)


def test_backfill_records_an_error_for_a_file_it_cannot_decode(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "ok.wav").write_bytes(b"")
    good = synthetic_library(lib, n=1, seconds=4.0)
    db_path = str(tmp_path / "x.db")
    with open_db(db_path) as db:
        run_index(db, good, workers=1, progress_stream=None)
        # A row that indexed fine, then became garbage in place, with the
        # stamp updated so the "changed" guard does not catch it first.
        path = os.path.join(good, "tone00.wav")
        with open(path, "wb") as fh:
            fh.write(b"not audio at all" * 500)
        st = os.stat(path)
        db.conn.execute(
            "UPDATE files SET envelope = NULL, size = ?, mtime = ? "
            "WHERE path = ?", (st.st_size, st.st_mtime, path))
        db.commit()
        logged: list[str] = []
        summary = run_backfill(db, workers=1, progress_stream=None,
                               log=logged.append)
    assert summary["errors"] == 1 and summary["filled"] == 0
    assert any("ERROR" in m for m in logged)


# --------------------------------------------------------------------------
# The storage-version migration
# --------------------------------------------------------------------------


def test_a_v1_database_migrates_in_place_and_does_not_demand_a_reindex(
        tmp_path):
    db_path = str(tmp_path / "v1.db")
    con = sqlite3.connect(db_path)
    con.executescript(V1_SCHEMA)
    con.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(config.SCHEMA_VERSION),))
    con.execute(
        "INSERT INTO files(path, alive, size, mtime, status, duration, "
        " sample_rate, channels, bits, codec, n_hashes) "
        "VALUES('/lib/a.wav', 1, 10, 1.0, 'ok', 60.0, 48000, 2, 24, 'pcm', 2)")
    con.executemany("INSERT INTO hashes VALUES(?,?,?)",
                    [(11, 1, 0), (12, 1, 5)])
    con.commit()
    con.close()

    # Opening it must succeed -- no "delete it and re-index".
    with open_db(db_path, create=False) as db:
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(files)")}
        assert "envelope" in cols
        assert db._meta("storage_version") == str(config.STORAGE_VERSION)
        # The pre-existing row is a backfill candidate, and its landmarks are
        # exactly where they were.
        assert db.count_missing_envelopes() == 1
        assert _landmarks(db_path) == [(11, 1, 0), (12, 1, 5)]
        assert list(db.iter_envelopes()) == []

    # Re-opening is idempotent.
    with open_db(db_path, create=False) as db:
        assert db.count_missing_envelopes() == 1


def test_two_processes_migrating_the_same_v1_database_do_not_collide(
        tmp_path, monkeypatch):
    """Opening a pre-v2 database twice at once must not crash either opener.

    Realistically: ``audio-match query`` in one terminal while ``index`` or
    ``backfill`` is starting up in another, on a database that has not been
    migrated yet.  Both run ``PRAGMA table_info`` (no envelope), both then run
    the ``ALTER``, and the loser used to die with ``sqlite3.OperationalError:
    duplicate column name: envelope``.  The column the winner added is the
    column the loser wanted, so that is success.

    The race is reproduced deterministically: the other process's ALTER is
    injected after *this* process has already read the column list.
    """
    import audiomatch.db as dbmod

    db_path = str(tmp_path / "v1.db")
    con = sqlite3.connect(db_path)
    con.executescript(V1_SCHEMA)
    con.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(config.SCHEMA_VERSION),))
    con.commit()
    con.close()

    real_connect = sqlite3.connect

    class _LosesTheRace:
        """Proxies the connection, letting the *other* process win once."""

        def __init__(self, conn):
            self._conn = conn
            self._raced = False

        def execute(self, sql, *args):
            cur = self._conn.execute(sql, *args)
            if "PRAGMA table_info(files)" in sql and not self._raced:
                self._raced = True
                rows = cur.fetchall()          # our probe: no envelope column
                other = real_connect(db_path)
                other.execute("ALTER TABLE files ADD COLUMN envelope BLOB")
                other.commit()
                other.close()
                return iter(rows)
            return cur

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(
        dbmod.sqlite3, "connect",
        lambda *a, **k: _LosesTheRace(real_connect(*a, **k)))

    with open_db(db_path, create=False) as db:     # must not raise
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(files)")}
        assert "envelope" in cols
        assert db._meta("storage_version") == str(config.STORAGE_VERSION)


def test_a_future_storage_version_is_refused_with_a_useful_message(tmp_path):
    db_path = str(tmp_path / "future.db")
    open_db(db_path).close()
    con = sqlite3.connect(db_path)
    con.execute("UPDATE meta SET value = '999' WHERE key = 'storage_version'")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="newer than this build"):
        open_db(db_path, create=False)


def test_a_stale_fingerprint_schema_still_demands_a_reindex(tmp_path):
    """The two versions must not have been conflated: a fingerprint-schema
    bump still means re-index, and says so."""
    db_path = str(tmp_path / "stale.db")
    open_db(db_path).close()
    con = sqlite3.connect(db_path)
    con.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="re-index"):
        open_db(db_path, create=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_backfill_subcommand(tone_db, capsys):
    _lib, db_path = tone_db
    con = sqlite3.connect(db_path)
    con.execute("UPDATE files SET envelope = NULL")
    con.commit()
    con.close()

    assert main(["--db", db_path, "backfill", "--workers", "2"]) == 0
    err = capsys.readouterr().err
    assert "3 indexed file(s) need an activity envelope" in err
    assert "filled 3 envelope(s)" in err

    with open_db(db_path, create=False) as db:
        assert db.count_missing_envelopes() == 0


def test_stats_and_index_point_at_backfill_when_envelopes_are_missing(
        tone_db, capsys):
    lib, db_path = tone_db
    con = sqlite3.connect(db_path)
    con.execute("UPDATE files SET envelope = NULL")
    con.commit()
    con.close()

    assert main(["--db", db_path, "stats"]) == 0
    assert "run 'audio-match backfill'" in capsys.readouterr().out

    assert main(["--db", db_path, "index", lib, "--quiet"]) == 0
    assert "run 'audio-match backfill'" in capsys.readouterr().err
