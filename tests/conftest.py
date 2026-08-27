"""Shared fixtures.

The suite runs against the *real* recovered Tascam DR-40 corpus in
``/mnt/host/projects/audio-recovery/recovered/`` (read-only).  Those files are
up to 1 GB, so nothing here ever loads a whole one: every excerpt is cut with
``ffmpeg -ss/-t`` into a scratch directory, and the cut excerpts are reused
across the whole session.

If the corpus is not present the corpus-dependent tests skip; the pure unit
tests still run anywhere ffmpeg is installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import pytest

RECOVERED = "/mnt/host/projects/audio-recovery/recovered"

SCRATCH = os.environ.get(
    "AUDIOMATCH_TEST_DIR",
    "/tmp/claude-1000/-mnt-host-projects-audio-recovery/"
    "360ff73d-ae88-48bb-8efb-a3985c503c40/scratchpad/amtest")

#: One source file per recording session, with a time offset that is known to
#: contain audio.  0076/0079 are only a few seconds long and 0077 is from the
#: same card and day as they are, so 0077 represents that session.
SESSIONS = {
    "0077": ("TASCAM_0077S12.wav", "TASCAM_0077S34.wav", 600.0),
    "0048": ("TASCAM_0048S12.wav", "TASCAM_0048S34.wav", 400.0),
    "0072": ("TASCAM_0072S12.wav", "TASCAM_0072S34.wav", 300.0),
    "pak":  ("pakDR40_S12.wav", "pakDR40_S34.wav", 500.0),
}

LIB_EXCERPT_SECONDS = 150.0


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and \
        shutil.which("ffprobe") is not None


def have_corpus() -> bool:
    return all(os.path.exists(os.path.join(RECOVERED, f))
               for pair in SESSIONS.values() for f in pair[:2])


requires_ffmpeg = pytest.mark.skipif(
    not have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")
requires_corpus = pytest.mark.skipif(
    not (have_ffmpeg() and have_corpus()),
    reason=f"recovered DR-40 corpus not available at {RECOVERED}")


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-800:])


def cut(src: str, dst: str, start: float, length: float,
        extra: list[str] | None = None) -> str:
    """Cut an excerpt.  Cached: never re-cuts a file that already exists."""
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    args = ["-ss", f"{start:.3f}", "-i", src, "-t", f"{length:.3f}"]
    args += extra if extra is not None else ["-c", "copy"]
    run_ffmpeg([*args, dst])
    return dst


@dataclass
class Corpus:
    root: str
    lib: str
    seeds: str

    def source(self, session: str, role: str) -> str:
        s12, s34, _ = SESSIONS[session]
        return os.path.join(RECOVERED, s12 if role == "S12" else s34)

    def start(self, session: str) -> float:
        return SESSIONS[session][2]

    def lib_file(self, session: str, role: str) -> str:
        return os.path.join(self.lib,
                            os.path.basename(self.source(session, role)))

    def seed(self, name: str) -> str:
        return os.path.join(self.seeds, name)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    """A small library of 2.5-minute excerpts, one pair per session."""
    if not (have_ffmpeg() and have_corpus()):
        pytest.skip("corpus unavailable")
    c = Corpus(root=SCRATCH,
               lib=os.path.join(SCRATCH, "lib"),
               seeds=os.path.join(SCRATCH, "seeds"))
    os.makedirs(c.lib, exist_ok=True)
    os.makedirs(c.seeds, exist_ok=True)
    for session in SESSIONS:
        for role in ("S12", "S34"):
            cut(c.source(session, role), c.lib_file(session, role),
                c.start(session), LIB_EXCERPT_SECONDS)
    return c


@pytest.fixture(scope="session")
def indexed_db(corpus: Corpus, tmp_path_factory) -> str:
    """The excerpt library, indexed once and shared by every test."""
    from audiomatch.db import open_db
    from audiomatch.indexer import run_index

    db_path = str(tmp_path_factory.mktemp("index") / "library.db")
    with open_db(db_path) as db:
        summary = run_index(db, corpus.lib, workers=4, progress_stream=None)
    assert summary["errors"] == 0
    assert summary["indexed"] == 8
    return db_path
