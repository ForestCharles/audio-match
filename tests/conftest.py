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


# --------------------------------------------------------------------------
# Pair mode needs a second, longer library
# --------------------------------------------------------------------------

#: Pair mode correlates a *1 Hz* envelope, so its excerpts are measured in
#: minutes, not the 150 s that is plenty for the constellation.  Ten minutes
#: is the shortest length at which all four sessions' true pairs clear the
#: PAIR threshold on the envelope alone (measured: 0.820, 0.883, 0.896, 0.921
#: at 600 s; 0.698 .. 0.886 at 300 s, where two of the four fall back on
#: coherence for their verdict).
PAIR_EXCERPT_SECONDS = 600.0


@dataclass
class PairCorpus:
    """A library holding **one** member of each dual-record pair.

    The seeds are the other member, cut from the same wall-clock range.  If
    both members were in the library, a seed would always rank its own
    recording first and "does the mate come top?" could not be asked.
    """

    lib: str
    seeds: str

    def lib_file(self, session: str) -> str:
        return os.path.join(self.lib, SESSIONS[session][0])       # the S12

    def seed_file(self, session: str) -> str:
        return os.path.join(self.seeds, SESSIONS[session][1])     # the S34


@pytest.fixture(scope="session")
def pair_corpus() -> PairCorpus:
    if not (have_ffmpeg() and have_corpus()):
        pytest.skip("corpus unavailable")
    root = os.path.join(SCRATCH, "pair")
    c = PairCorpus(lib=os.path.join(root, "lib"),
                   seeds=os.path.join(root, "seeds"))
    os.makedirs(c.lib, exist_ok=True)
    os.makedirs(c.seeds, exist_ok=True)
    for session, (s12, s34, start) in SESSIONS.items():
        cut(os.path.join(RECOVERED, s12), c.lib_file(session),
            start, PAIR_EXCERPT_SECONDS)
        cut(os.path.join(RECOVERED, s34), c.seed_file(session),
            start, PAIR_EXCERPT_SECONDS)
    return c


#: The false-PAIR repro, from the adversarial review of pair mode.
#:
#: ``pakDR40_earlier`` is a *different recording* from ``pakDR40_S12/S34`` --
#: an earlier single-stream take from the same card, with no shared audio at
#: all -- and these two ranges do not overlap in wall-clock time either.  Yet
#: the seed's first five minutes correlate with 15:00-25:00 of the earlier file
#: at raw r = 0.94, scored 0.86 over a 300 s overlap: above ``PAIR_R_STRONG``,
#: from a pair of files that share nothing.  That is the measurement behind
#: ``config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS``.
EARLIER_WAV = "pakDR40_earlier.wav"
FALSE_PAIR_SEED_SECONDS = 300.0
FALSE_PAIR_LIB_SECONDS = 600.0
FALSE_PAIR_LIB_START = 900.0


def have_earlier() -> bool:
    return os.path.exists(os.path.join(RECOVERED, EARLIER_WAV))


requires_earlier = pytest.mark.skipif(
    not (have_ffmpeg() and have_corpus() and have_earlier()),
    reason=f"{EARLIER_WAV} not available in {RECOVERED}")


@dataclass
class FalsePairCorpus:
    lib: str          # directory holding the one library file
    seed: str         # the five-minute seed


@pytest.fixture(scope="session")
def false_pair_corpus() -> FalsePairCorpus:
    """A five-minute seed and a ten-minute library file that share no audio."""
    if not (have_ffmpeg() and have_corpus() and have_earlier()):
        pytest.skip("corpus unavailable")
    root = os.path.join(SCRATCH, "falsepair")
    c = FalsePairCorpus(
        lib=os.path.join(root, "lib"),
        seed=os.path.join(root, "seeds", "pakDR40_S34_0-5.wav"))
    cut(os.path.join(RECOVERED, EARLIER_WAV),
        os.path.join(c.lib, "pakDR40_earlier_15-25.wav"),
        FALSE_PAIR_LIB_START, FALSE_PAIR_LIB_SECONDS)
    cut(os.path.join(RECOVERED, SESSIONS["pak"][1]), c.seed,
        0.0, FALSE_PAIR_SEED_SECONDS)
    return c


@pytest.fixture(scope="session")
def false_pair_db(false_pair_corpus: FalsePairCorpus, tmp_path_factory) -> str:
    from audiomatch.db import open_db
    from audiomatch.indexer import run_index

    db_path = str(tmp_path_factory.mktemp("falsepair") / "falsepair.db")
    with open_db(db_path) as db:
        summary = run_index(db, false_pair_corpus.lib, workers=2,
                            progress_stream=None)
    assert summary["errors"] == 0
    assert summary["indexed"] == 1
    return db_path


@pytest.fixture(scope="session")
def pair_db(pair_corpus: PairCorpus, tmp_path_factory) -> str:
    from audiomatch.db import open_db
    from audiomatch.indexer import run_index

    db_path = str(tmp_path_factory.mktemp("pair") / "pair.db")
    with open_db(db_path) as db:
        summary = run_index(db, pair_corpus.lib, workers=4,
                            progress_stream=None)
    assert summary["errors"] == 0
    assert summary["indexed"] == len(SESSIONS)
    return db_path
