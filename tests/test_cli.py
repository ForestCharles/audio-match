"""End-to-end CLI behaviour."""

from __future__ import annotations

import pytest

from audiomatch.cli import fmt_clock, main

from conftest import Corpus, cut, requires_corpus, requires_ffmpeg

pytestmark = requires_ffmpeg


def test_version_and_help():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


@pytest.mark.parametrize("seconds,expected", [
    (0.0, "0:00.0"),
    (59.94, "0:59.9"),
    (60.0, "1:00.0"),
    (3661.5, "1:01:01.5"),
    (-12.25, "-0:12.2"),
])
def test_fmt_clock(seconds, expected):
    assert fmt_clock(seconds) == expected


def test_query_against_missing_database(tmp_path, capsys):
    rc = main(["--db", str(tmp_path / "nope.db"), "query", __file__])
    assert rc == 2
    assert "run 'audio-match index'" in capsys.readouterr().err


def test_query_with_missing_seed(tmp_path, capsys):
    db = str(tmp_path / "x.db")
    main(["--db", db, "index", str(tmp_path)])
    rc = main(["--db", db, "query", str(tmp_path / "nothere.wav")])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


@requires_corpus
def test_full_index_then_query_round_trip(corpus: Corpus, tmp_path, capsys):
    """The exact flow from the README, run for real."""
    db = str(tmp_path / "cli.db")

    assert main(["--db", db, "index", corpus.lib, "--workers", "4"]) == 0
    out = capsys.readouterr()
    assert "indexed 8 file(s), 0 error(s)" in out.err

    seed = cut(corpus.lib_file("0077", "S12"),
               corpus.seed("cli_seed.mp3"), 60.0, 30.0,
               extra=["-ar", "44100", "-af", "volume=-8dB",
                      "-c:a", "libmp3lame", "-b:a", "128k"])
    assert main(["--db", db, "query", seed, "--top", "5"]) == 0
    out = capsys.readouterr().out

    assert "MODE 1: same audio" in out
    assert "MODE 2: same session" in out
    assert "TASCAM_0077S12.wav" in out
    assert "[MATCH  ]" in out

    # The winning line must be the first result listed.
    first_result = [ln for ln in out.splitlines()
                    if ln.startswith(" 1. [")][0]
    assert "MATCH" in first_result

    assert main(["--db", db, "stats"]) == 0
    stats = capsys.readouterr().out
    assert "files indexed   8" in stats
    assert "landmarks" in stats


@requires_corpus
def test_query_modes_are_independent(corpus: Corpus, tmp_path, capsys):
    db = str(tmp_path / "modes.db")
    main(["--db", db, "index", corpus.lib, "--workers", "4"])
    capsys.readouterr()
    seed = corpus.lib_file("0072", "S12")

    main(["--db", db, "query", seed, "--mode", "match"])
    out = capsys.readouterr().out
    assert "MODE 1" in out and "MODE 2" not in out

    main(["--db", db, "query", seed, "--mode", "session"])
    out = capsys.readouterr().out
    assert "MODE 2" in out and "MODE 1" not in out


def test_errors_subcommand(tmp_path, capsys):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.wav").write_bytes(b"nope" * 2000)
    db = str(tmp_path / "e.db")
    main(["--db", db, "index", str(lib), "--quiet"])
    capsys.readouterr()
    assert main(["--db", db, "errors"]) == 0
    out = capsys.readouterr().out
    assert "broken.wav" in out
    assert "1 file(s) failed to index." in out


def test_stale_schema_exits_cleanly_without_a_traceback(tmp_path, capsys):
    """A DB from an older fingerprint schema must be an error, not a crash."""
    import sqlite3

    from audiomatch.db import open_db

    db = str(tmp_path / "stale.db")
    open_db(db).close()
    con = sqlite3.connect(db)
    con.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    con.commit()
    con.close()

    rc = main(["--db", db, "stats"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "re-index" in err
    assert "Traceback" not in err
