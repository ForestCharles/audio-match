"""Scoring edge cases, honest output, and the opt-in rate probes.

These need ffmpeg but not the DR-40 corpus: the library is a handful of
synthetic noise+tone WAVs generated on the fly.
"""

from __future__ import annotations

import os
import shutil

import pytest

from audiomatch import config
from audiomatch.cli import main
from audiomatch.db import open_db
from audiomatch.indexer import run_index
from audiomatch.query import MatchHit, run_query

from conftest import requires_ffmpeg
from test_index import synthetic_library

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def tone_library(tmp_path_factory) -> tuple[str, str]:
    """(library_dir, indexed_db_path) built from synthetic audio."""
    root = tmp_path_factory.mktemp("tones")
    lib = synthetic_library(root / "lib", n=3, seconds=8.0)
    db_path = str(root / "tones.db")
    with open_db(db_path) as db:
        summary = run_index(db, lib, workers=1, progress_stream=None)
    assert summary["indexed"] == 3 and summary["errors"] == 0
    return lib, db_path


# --------------------------------------------------------------------------
# FIX 6: a lone offset bin is not infinitely sharp
# --------------------------------------------------------------------------


def _hit(votes: int, background: int) -> MatchHit:
    return MatchHit(file_id=1, path="/lib/x.wav", votes=votes,
                    total_votes=votes, offset_frames=0, probe="native",
                    ratio=1.0, matched_seconds=1.0, library_duration=60.0,
                    background=background)


def test_a_lone_offset_bin_does_not_report_unbounded_sharpness():
    """No second bin means no measured background, not a perfect match.

    Before the floor, a file whose only shared bin held 25 votes reported
    '25.0x' and was labelled [MATCH] on the strength of nothing at all.
    """
    lone = _hit(votes=25, background=0)
    assert lone.sharpness == 25 / config.SHARPNESS_MIN_BACKGROUND
    assert lone.sharpness < 25.0

    # The floor only ever bites below itself; a real measured background is
    # used unchanged, so existing scores are untouched.
    measured = _hit(votes=600, background=6)
    assert measured.sharpness == 100.0
    assert _hit(votes=50, background=config.SHARPNESS_MIN_BACKGROUND
                ).sharpness == 50 / config.SHARPNESS_MIN_BACKGROUND


def test_the_sharpness_floor_gates_the_weakest_lone_bins():
    """A tiny lone bin can no longer clear the confidence sharpness bar."""
    seed_seconds = 20.0
    # 25 votes clears the vote floor; 25/3 = 8.3x still clears sharpness.
    assert _hit(25, 0).is_confident(seed_seconds)
    # But a lone bin below 4 x 3 = 12 votes cannot, whatever else it does.
    assert not _hit(11, 0).is_confident(seed_seconds)


# --------------------------------------------------------------------------
# FIX 2: deleted library files are labelled, not presented as live
# --------------------------------------------------------------------------


def test_query_annotates_results_whose_file_has_vanished(tone_library,
                                                         tmp_path, capsys):
    lib, db_path = tone_library
    src = os.path.join(lib, "tone00.wav")
    seed = str(tmp_path / "seed.wav")
    shutil.copy(src, seed)

    stash = str(tmp_path / "stashed.wav")
    shutil.move(src, stash)
    try:
        rc = main(["--db", db_path, "query", seed, "--mode", "match"])
        out = capsys.readouterr().out
    finally:
        shutil.move(stash, src)

    assert rc == 0
    assert "tone00.wav" in out
    assert "[missing]" in out
    assert "no longer exists" in out


def test_query_does_not_annotate_files_that_are_still_there(tone_library,
                                                            capsys):
    lib, db_path = tone_library
    assert main(["--db", db_path, "query",
                 os.path.join(lib, "tone00.wav"), "--mode", "match"]) == 0
    assert "[missing]" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# FEATURE: --try-rates is opt-in
# --------------------------------------------------------------------------


def test_rate_probes_are_off_by_default_and_opt_in_with_try_rates(
        tone_library):
    lib, db_path = tone_library
    seed = os.path.join(lib, "tone00.wav")
    with open_db(db_path, create=False) as db:
        default = run_query(db, seed, mode="match")
        opted_in = run_query(db, seed, mode="match", try_rates=True)
    assert default.probes_run == ["native"]
    assert list(default.seed_hash_counts) == ["native"]
    assert opted_in.probes_run == [p[0] for p in config.SR_PROBES]
    assert len(opted_in.probes_run) == 3


def test_try_rates_flag_is_accepted_by_the_cli(tone_library, capsys):
    lib, db_path = tone_library
    rc = main(["--db", db_path, "query", os.path.join(lib, "tone00.wav"),
               "--mode", "match", "--try-rates"])
    assert rc == 0
    assert "MODE 1" in capsys.readouterr().out
