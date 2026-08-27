"""Indexing: resumability, error handling, determinism, storage."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import numpy as np
import pytest

from audiomatch import config
from audiomatch.analyze import analyze_file
from audiomatch.db import open_db
from audiomatch.fingerprint import fingerprint_array
from audiomatch.indexer import plan_index, run_index, walk_library

from conftest import Corpus, cut, requires_corpus, requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture
def small_library(corpus: Corpus, tmp_path) -> str:
    """Two short excerpts, copied so tests may freely mutate them."""
    lib = tmp_path / "lib"
    lib.mkdir()
    for session, role in (("0072", "S12"), ("0072", "S34")):
        src = cut(corpus.source(session, role),
                  corpus.seed(f"small_{session}{role}.wav"),
                  corpus.start(session), 20.0)
        shutil.copy(src, lib / os.path.basename(src))
    return str(lib)


@requires_corpus
def test_rerun_skips_unchanged_files(small_library, tmp_path):
    db_path = str(tmp_path / "resume.db")
    with open_db(db_path) as db:
        first = run_index(db, small_library, workers=1, progress_stream=None)
    assert first["indexed"] == 2 and first["skipped"] == 0

    with open_db(db_path) as db:
        second = run_index(db, small_library, workers=1, progress_stream=None)
    assert second["indexed"] == 0
    assert second["skipped"] == 2


@requires_corpus
def test_only_the_touched_file_is_reindexed(small_library, tmp_path):
    db_path = str(tmp_path / "touch.db")
    with open_db(db_path) as db:
        run_index(db, small_library, workers=1, progress_stream=None)

    victim = sorted(os.listdir(small_library))[0]
    path = os.path.join(small_library, victim)
    # Move the mtime somewhere unambiguous rather than relying on clock skew.
    os.utime(path, (time.time() + 120, time.time() + 120))

    with open_db(db_path) as db:
        plan = plan_index(db, small_library)
        assert [os.path.basename(p) for p in plan.todo] == [victim]
        assert plan.skipped == 1
        summary = run_index(db, small_library, workers=1,
                            progress_stream=None)
    assert summary["indexed"] == 1 and summary["skipped"] == 1

    # The superseded record is retired, not duplicated, and its landmarks are
    # no longer reachable from a query.
    with open_db(db_path) as db:
        stats = db.stats()
        assert stats["files_live"] == 2
        assert stats["files_dead"] == 1
        live = db.live_ids()
        reachable = {r[0] for r in db.conn.execute(
            "SELECT DISTINCT file_id FROM hashes")}
        assert reachable - live, "expected orphaned landmarks before purge"
        removed, files = db.purge()
        assert files == 1 and removed > 0
        reachable = {r[0] for r in db.conn.execute(
            "SELECT DISTINCT file_id FROM hashes")}
        assert reachable <= db.live_ids()


@requires_corpus
def test_size_change_forces_reindex(small_library, tmp_path):
    db_path = str(tmp_path / "size.db")
    with open_db(db_path) as db:
        run_index(db, small_library, workers=1, progress_stream=None)
    victim = os.path.join(small_library, sorted(os.listdir(small_library))[0])
    with open(victim, "ab") as fh:
        fh.write(b"\0" * 4096)
    with open_db(db_path) as db:
        plan = plan_index(db, small_library)
    assert len(plan.todo) == 1 and plan.skipped == 1


def test_corrupt_files_are_logged_and_skipped(tmp_path):
    lib = tmp_path / "bad"
    lib.mkdir()
    (lib / "text.wav").write_text(
        "This is a text file wearing a .wav extension.\n" * 500)
    (lib / "noise.wav").write_bytes(os.urandom(200_000))
    (lib / "empty.wav").write_bytes(b"")
    (lib / "truncated.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    db_path = str(tmp_path / "bad.db")
    logged: list[str] = []
    with open_db(db_path) as db:
        summary = run_index(db, str(lib), workers=1, progress_stream=None,
                            log=logged.append)
        stats = db.stats()
        errors = db.conn.execute(
            "SELECT path, error FROM files WHERE status='error'").fetchall()

    # Nothing raised, nothing indexed, every failure recorded with a reason.
    assert summary["indexed"] == 0
    assert summary["errors"] >= 2
    assert stats["files_ok"] == 0
    assert all(err for _path, err in errors)
    assert any("ERROR" in line for line in logged)


def test_corrupt_file_analysis_never_raises(tmp_path):
    p = tmp_path / "garbage.wav"
    p.write_bytes(b"not audio" * 1000)
    result = analyze_file(str(p))
    assert result.status == "error"
    assert result.error
    assert result.n_hashes == 0


def test_missing_file_analysis_never_raises(tmp_path):
    result = analyze_file(str(tmp_path / "does-not-exist.wav"))
    assert result.status == "error"
    assert "stat failed" in result.error


def test_extension_allowlist_and_all_files_override(tmp_path):
    lib = tmp_path / "mixed"
    (lib / "sub").mkdir(parents=True)
    for name in ("a.wav", "b.flac", "c.txt", "sub/d.mp3", "sub/e.pdf",
                 ".hidden.wav"):
        (lib / name).write_bytes(b"x" * 10)

    default = {os.path.basename(p) for p in walk_library(str(lib))}
    assert default == {"a.wav", "b.flac", "d.mp3"}

    everything = {os.path.basename(p)
                  for p in walk_library(str(lib), all_files=True)}
    assert everything == {"a.wav", "b.flac", "c.txt", "d.mp3", "e.pdf"}


@requires_corpus
def test_hashing_is_deterministic(corpus):
    src = cut(corpus.source("0072", "S12"),
              corpus.seed("determinism_20s.wav"),
              corpus.start("0072"), 20.0)
    a = analyze_file(src)
    b = analyze_file(src)
    assert a.status == b.status == "ok"
    assert a.n_hashes > 0
    np.testing.assert_array_equal(a.hashes, b.hashes)
    np.testing.assert_array_equal(a.times, b.times)
    np.testing.assert_allclose(a.signature.noise, b.signature.noise)
    np.testing.assert_allclose(a.signature.chan, b.signature.chan)


def test_hashing_is_deterministic_on_synthetic_audio():
    """Determinism without needing the corpus: same input, same hashes."""
    rng = np.random.default_rng(20260827)
    seconds = 8
    t = np.arange(seconds * config.ANALYSIS_SR) / config.ANALYSIS_SR
    tone = sum(np.sin(2 * np.pi * f * t) for f in (220, 440, 933, 1710))
    mono = (0.2 * tone + 0.02 * rng.standard_normal(t.size)).astype(np.float32)

    h1, t1, p1 = fingerprint_array(mono, 1.0, config.INDEX_FANOUT)
    h2, t2, p2 = fingerprint_array(mono.copy(), 1.0, config.INDEX_FANOUT)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(t1, t2)
    assert h1.size > 0
    # Hashes must fit the documented 23-bit packing.
    assert int(h1.max()) < (1 << config.HASH_BITS)
    assert int(h1.min()) >= 0
    # And density must land near the configured target.
    peaks_per_second = len(p1) / seconds
    assert 4 <= peaks_per_second <= 12


@requires_corpus
def test_landmark_density_stays_within_the_documented_budget(indexed_db):
    """~1 MB of database per hour of audio is the number in the README."""
    with open_db(indexed_db, create=False) as db:
        stats = db.stats()
    hours = stats["seconds"] / 3600.0
    assert hours > 0
    bytes_per_hour = stats["bytes"] / hours
    hashes_per_second = stats["hashes"] / stats["seconds"]
    assert 8 <= hashes_per_second <= 25, hashes_per_second
    assert bytes_per_hour < 2.0 * 1024 * 1024, bytes_per_hour


@requires_corpus
def test_stale_schema_is_rejected(tmp_path, small_library):
    db_path = str(tmp_path / "stale.db")
    with open_db(db_path) as db:
        db.conn.execute("UPDATE meta SET value='999' "
                        "WHERE key='schema_version'")
        db.conn.commit()
    with pytest.raises(RuntimeError, match="re-index"):
        open_db(db_path)


def _corrupt_mp3(tmp_path) -> str:
    """An MP3 whose every frame is damaged: ffmpeg emits >64 KB of stderr."""
    import subprocess
    clean = tmp_path / "clean.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=240", "-c:a", "libmp3lame",
         "-b:a", "320k", str(clean)], check=True, capture_output=True)
    data = bytearray(clean.read_bytes())
    for i in range(2000, len(data), 120):
        data[i] ^= 0xFF
    bad = tmp_path / "corrupt.mp3"
    bad.write_bytes(bytes(data))
    return str(bad)


def test_decode_stream_does_not_deadlock_on_a_stderr_flood(tmp_path):
    """Regression: a file that floods ffmpeg's stderr must still decode.

    ffmpeg's stderr is a pipe with a ~64 KB buffer.  If it is not drained
    while stdout is being read, ffmpeg blocks writing errors, stops producing
    audio, and the decode hangs forever -- unrecoverably, in a worker, in the
    middle of an unattended 1.45 TB index.
    """
    import threading

    from audiomatch import audio

    bad = _corrupt_mp3(tmp_path)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", bad, "-map", "0:a:0",
         "-ac", "2", "-ar", str(config.ANALYSIS_SR), "-f", "f32le", "-"],
        capture_output=True)
    assert len(proc.stderr) > 65536, (
        f"fixture is not hostile enough: only {len(proc.stderr)} B of stderr")

    result: dict = {}

    def decode():
        try:
            result["frames"] = sum(
                b.shape[0] for b in audio.decode_stream(bad))
        except Exception as exc:                        # noqa: BLE001
            result["exc"] = repr(exc)

    th = threading.Thread(target=decode, daemon=True)
    th.start()
    th.join(120)
    assert not th.is_alive(), "decode_stream deadlocked on ffmpeg's stderr"
    assert result.get("frames", 0) > config.ANALYSIS_SR, result


def test_analyze_file_survives_a_stderr_flood(tmp_path):
    an = analyze_file(_corrupt_mp3(tmp_path))
    assert an.status == "ok"
    assert an.n_hashes > 0
