"""Mode 2: does the session signature rank same-session files higher?

These assertions are deliberately about *ordering* and *aggregate separation*,
never about absolute score values, because the score is a heuristic blend
whose weights are expected to be tuned.
"""

from __future__ import annotations

import os
import statistics

import pytest

from audiomatch.analyze import analyze_seed
from audiomatch.db import open_db
from audiomatch.query import session_search
from audiomatch.session import compare, parse_tascam_name

from conftest import Corpus, cut, requires_corpus

pytestmark = requires_corpus

#: Which library file belongs to which session.
FILE_SESSION = {
    "TASCAM_0077S12.wav": "0077", "TASCAM_0077S34.wav": "0077",
    "TASCAM_0048S12.wav": "0048", "TASCAM_0048S34.wav": "0048",
    "TASCAM_0072S12.wav": "0072", "TASCAM_0072S34.wav": "0072",
    "pakDR40_S12.wav": "pak", "pakDR40_S34.wav": "pak",
}


def _held_out_seed(corpus: Corpus, session: str, role: str, name: str) -> str:
    """An excerpt from a part of the recording the library does not contain."""
    offsets = {"0077": 1800.0, "0048": 1200.0, "0072": 600.0, "pak": 1400.0}
    return cut(corpus.source(session, role), corpus.seed(name),
               offsets[session], 90.0)


def _rank(hits, session: str) -> list[int]:
    return [i for i, h in enumerate(hits)
            if FILE_SESSION[os.path.basename(h.path)] == session]


@pytest.mark.parametrize("session,role", [("0077", "S12"), ("0048", "S34")])
def test_same_session_outranks_other_sessions(indexed_db, corpus,
                                              session, role, capsys):
    seed = _held_out_seed(corpus, session, role, f"sess_{session}{role}.wav")
    _h, _t, sig, _s = analyze_seed(seed)
    with open_db(indexed_db, create=False) as db:
        hits = session_search(db, sig, top=20)

    own, other = [], []
    for h in hits:
        (own if FILE_SESSION[os.path.basename(h.path)] == session
         else other).append(h.score.total)

    lines = ["", f"session seed {session}{role}:"]
    lines += [f"  {i + 1}. {h.score.total:.3f}  "
              f"{os.path.basename(h.path)}" for i, h in enumerate(hits)]
    print("\n".join(lines))

    assert len(own) == 2 and len(other) == 6
    # 1. The top-ranked file comes from the seed's own session.
    assert FILE_SESSION[os.path.basename(hits[0].path)] == session, \
        f"top hit {hits[0].path} is not from session {session}"
    # 2. Same-session files score higher on average than everything else.
    assert statistics.mean(own) > statistics.mean(other)
    # 3. Both same-session files (i.e. including the S-pair) land in the
    #    better-scoring half of the ranking.
    assert max(_rank(hits, session)) < len(hits) // 2


def test_pair_mate_is_annotated(indexed_db, corpus):
    """A seed named ...0048S12 should flag ...0048S34 as its pair-mate."""
    seed = cut(corpus.source("0048", "S12"),
               corpus.seed("TASCAM_0048S12_excerpt.wav"),
               1200.0, 60.0)
    _h, _t, sig, _s = analyze_seed(seed)
    assert sig.take == 48 and sig.role == "S12"
    with open_db(indexed_db, create=False) as db:
        hits = session_search(db, sig, top=20)
    mates = [h for h in hits if h.is_pair_mate]
    assert [os.path.basename(h.path) for h in mates] == \
        ["TASCAM_0048S34.wav"]
    notes = " ".join(mates[0].score.notes)
    assert "pair-mate" in notes


def test_ignore_filenames_removes_the_take_number_evidence(indexed_db,
                                                           corpus):
    seed = _held_out_seed(corpus, "0077", "S12", "sess_0077S12.wav")
    _h, _t, sig, _s = analyze_seed(seed)
    with open_db(indexed_db, create=False) as db:
        with_names = {os.path.basename(h.path): h.score.total
                      for h in session_search(db, sig, top=20)}
        without = {os.path.basename(h.path): h.score.total
                   for h in session_search(db, sig, top=20,
                                           ignore_filenames=True)}
    # The same-take file loses ground once filenames are ignored; a file with
    # no parseable take number is unaffected either way.
    assert without["TASCAM_0077S12.wav"] < with_names["TASCAM_0077S12.wav"]
    assert without["pakDR40_S12.wav"] == pytest.approx(
        with_names["pakDR40_S12.wav"])


def test_signature_is_gain_invariant(corpus, tmp_path):
    """A -12 dB copy must look like the same session, not a different one."""
    src = corpus.lib_file("0072", "S12")
    quiet = str(tmp_path / "quiet.wav")
    cut(src, quiet, 0.0, 60.0, extra=["-af", "volume=-12dB"])
    loud = str(tmp_path / "loud.wav")
    cut(src, loud, 0.0, 60.0, extra=["-c", "copy"])

    _h, _t, a, _s = analyze_seed(loud)
    _h, _t, b, _s = analyze_seed(quiet)
    score = compare(a, b, ignore_filenames=True)
    assert score.noise > 0.9, f"noise floor shape moved with gain: {score}"


@pytest.mark.parametrize("name,take,role", [
    ("TASCAM_0077S12.wav", 77, "S12"),
    ("TASCAM_0077S34.wav", 77, "S34"),
    ("/some/dir/TASCAM_0048S34.WAV", 48, "S34"),
    ("sess_0048.wav", 48, None),
    ("pakDR40_S12.wav", None, None),
    ("random name.wav", None, None),
])
def test_tascam_filename_parsing(name, take, role):
    parsed = parse_tascam_name(name)
    assert parsed.take == take
    assert parsed.role == role
