"""Mode 3: pair matching, against the real recovered DR-40 corpus.

The library holds one member of each dual-record pair (the S12 of each of the
four sessions, ten minutes each); the seeds are the other member (the S34),
cut from the same wall-clock range.  So "did the mate come top?" is a real
question, not one the seed's own recording could answer for it.
"""

from __future__ import annotations

import io
import os
import re

import numpy as np
import pytest

from audiomatch import config, envelope as E
from audiomatch.cli import main
from audiomatch.db import open_db
from audiomatch.indexer import run_index
from audiomatch.query import (Coherence, PairHit, drift_grid, drift_slopes,
                              fit_coherence, fit_coherence_global, pair_search,
                              run_query)

from conftest import (CLOSE_MIC_SECONDS, FALSE_PAIR_LIB_SECONDS,
                      FALSE_PAIR_SEED_SECONDS, PAIR_EXCERPT_SECONDS, SESSIONS,
                      CloseMicCorpus, FalsePairCorpus, PairCorpus, cut,
                      requires_corpus, requires_earlier, requires_ffmpeg,
                      run_ffmpeg)

pytestmark = requires_ffmpeg


def _search(db_path: str, seed: str, **kw):
    with open_db(db_path, create=False) as db:
        return pair_search(db, seed, **kw)


@pytest.fixture(scope="module")
def pair_results(pair_db, pair_corpus) -> dict:
    """Every session's pair query, run once and shared by the tests below.

    Each one decodes and fingerprints a ten-minute seed, which is by far the
    most expensive thing in this file; running them per-test would triple the
    suite's wall clock for no extra coverage.
    """
    out = {}
    for session in sorted(SESSIONS):
        hits, seconds, sig, note = _search(
            pair_db, pair_corpus.seed_file(session), top=8)
        assert not note, note
        out[session] = (hits, seconds, sig)
    return out


def _report(hits) -> str:
    return "\n".join(
        f"  {i}. [{h.verdict:14s}] {os.path.basename(h.path):22s} "
        f"score {h.alignment.score:.3f} raw {h.alignment.raw_r:.3f} "
        f"lag {h.alignment.lag:+4d}s ov {h.alignment.overlap:4d}s  "
        f"coherence {h.coherence.level:6s} "
        f"v={h.coherence.votes:4d} sharp={h.coherence.sharpness:5.1f} "
        f"ppm={h.coherence.drift_ppm:+7.1f}"
        for i, h in enumerate(hits, 1))


# --------------------------------------------------------------------------
# 1. Ground truth: four real S12/S34 pairs
# --------------------------------------------------------------------------


@requires_corpus
@pytest.mark.parametrize("session", sorted(SESSIONS))
def test_the_dual_record_mate_is_ranked_first(pair_results, pair_corpus,
                                              session):
    """The headline claim, on real audio, for every session in the corpus.

    Measured, on ten-minute excerpts, for the true mate:

        session   score   raw r   lag   coherence votes / sharpness
        0048      0.896   0.940    0s   134 /  9.6x
        0072      0.820   0.860    0s   438 / 11.8x
        0077      0.883   0.926    0s   289 / 26.3x
        pak       0.921   0.966    0s   115 /  6.4x

    Every one of them clears ``PAIR_R_STRONG`` on the envelope alone; the best
    unrelated candidate in any of the four queries reaches 0.620.  The 0072
    margin over 0.80 is the thinnest at 0.02, which is worth knowing if these
    thresholds are ever retuned.

    The *verdict*, though, does not rest on that score: a ten-minute excerpt
    overlaps by 600 s, well under ``PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS``, so
    the envelope alone would be capped at TIMELINE MATCH here.  These four come
    out PAIR because they are one recorder's two microphone pairs and their
    coherence is strong -- which is the gate working as designed, not around
    it.  See ``test_strong_coherence_lifts_a_short_overlap_to_pair``.
    """
    hits, seconds, _sig = pair_results[session]
    assert seconds == pytest.approx(PAIR_EXCERPT_SECONDS, abs=2.0)

    best = hits[0]
    want = os.path.basename(pair_corpus.lib_file(session))
    report = f"\npair seed {session}S34:\n" + _report(hits)
    print(report)

    assert os.path.basename(best.path) == want, report
    assert best.verdict == "PAIR"
    # The envelope alone must clear the PAIR bar; the length gate is what then
    # asks for a second opinion, and coherence supplies it.
    assert best.alignment.score >= config.PAIR_R_STRONG
    # Same wall-clock range, so the two files line up at lag 0.  pakDR40_S12
    # has 0.74 s of silence prepended by the recovery, which is sub-sample at
    # 1 Hz and must not move the answer.
    assert abs(best.alignment.lag) <= 2
    # Same recorder, same clock: coherence must fire.
    assert best.coherence.level == "strong"
    assert abs(best.coherence.offset_seconds) < 1.0
    # And every unrelated session must stay below TIMELINE MATCH.
    for other in hits[1:]:
        assert other.alignment.score < config.PAIR_R_LIKELY, report
        assert other.verdict == "weak"


@requires_corpus
def test_the_evidence_lines_say_what_is_behind_the_verdict(pair_results):
    hits, _s, _sig = pair_results["0048"]
    text = "\n".join(hits[0].evidence)
    assert "envelope r=" in text and "overlap" in text
    assert "acoustic coherence: strong" in text
    assert "dual-record pair-mate of the seed (take 0048 S12)" in text
    assert "session signature:" in text
    assert "segments align:" in text
    # A file that is not the mate must not claim to be one.
    assert not any("pair-mate" in line for line in hits[-1].evidence)


@requires_corpus
def test_take_numbers_are_evidence_not_a_gate(pair_results):
    """pakDR40 has no parseable Tascam take number and must still win."""
    hits, _s, _sig = pair_results["pak"]
    assert os.path.basename(hits[0].path) == "pakDR40_S12.wav"
    assert hits[0].take is None and not hits[0].is_take_mate
    assert hits[0].verdict == "PAIR"


# --------------------------------------------------------------------------
# 2. A simulated second recorder
# --------------------------------------------------------------------------

#: Sample-rate ratio injected by the "other recorder" filter chain.  ffmpeg's
#: asetrate takes an integer, so the drift is 48005/48000 rather than a round
#: 100 ppm.
DRIFT_NUM, DRIFT_DEN = 48005, 48000
DRIFT_PPM = (DRIFT_NUM / DRIFT_DEN - 1.0) * 1e6          # 104.2 ppm


@pytest.fixture(scope="module")
def other_recorder(pair_corpus: PairCorpus) -> str:
    """The 0077 S34 excerpt as if a different rig had captured it.

    Unsynchronised clock (+104 ppm), a different microphone's tilt (-6 dB
    below 200 Hz, +3 dB above 4 kHz), a different position in the room (a
    short echo), 6 dB less gain, and 128 kbps mp3 on the way out.
    """
    dst = os.path.join(os.path.dirname(pair_corpus.seeds), "other_rig.mp3")
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return dst
    run_ffmpeg([
        "-i", pair_corpus.seed_file("0077"),
        "-af", f"asetrate={DRIFT_NUM},aresample={DRIFT_DEN},"
               f"bass=g=-6:f=200,treble=g=3:f=4000,"
               f"aecho=0.8:0.7:60:0.25,volume=-6dB",
        "-c:a", "libmp3lame", "-b:a", "128k", dst])
    return dst


@requires_corpus
def test_a_different_recorder_still_finds_the_counterpart(
        pair_db, pair_corpus, other_recorder, capsys):
    """The whole point of leading with the envelope.

    The seed has been resampled, re-EQ'd, echoed, attenuated and squeezed
    through 128 kbps mp3, and its clock does not agree with the library's.
    Measured result: score 0.871 (raw r 0.913) at lag 0, ranking the true
    counterpart first with the runner-up at 0.407.

    Coherence is *allowed* to collapse here and the assertions do not require
    it -- that is the case the envelope exists for.  On this corpus it in fact
    survives (220 aligned votes), because the transform left the transient
    structure the constellation keys on largely intact.  A real second rig in
    a different part of the room would be harsher.
    """
    hits, _s, _sig, note = _search(pair_db, other_recorder, top=8)
    assert not note, note
    best = hits[0]
    report = "\n".join(
        f"  [{h.verdict:14s}] {os.path.basename(h.path):22s} "
        f"score {h.alignment.score:.3f} raw {h.alignment.raw_r:.3f} "
        f"lag {h.alignment.lag:+4d}s coherence {h.coherence.level} "
        f"({h.coherence.votes} votes, ppm {h.coherence.drift_ppm:+.0f})"
        for h in hits)
    print("\nsimulated other recorder ->\n" + report)

    assert os.path.basename(best.path) == "TASCAM_0077S12.wav", report
    assert abs(best.alignment.lag) <= 2, report
    assert best.verdict in ("PAIR", "TIMELINE MATCH"), report
    assert best.alignment.score >= config.PAIR_R_LIKELY, report
    for other in hits[1:]:
        assert other.alignment.score < config.PAIR_R_LIKELY, report


@requires_corpus
def test_drift_below_the_resolution_floor_is_reported_as_such(
        pair_corpus, other_recorder, tmp_path):
    """A ten-minute seed cannot see 104 ppm, and must not pretend to.

    Measured vote counts for this exact pair, per candidate slope (the library
    here holds the transformed file's own untransformed source, so the
    landmarks survive in bulk -- 3189 votes at zero drift):

        -232: 2715   -116: 2172   0: 3189   +116: 3322   +232: 2715

    (measured over the full 600 s of library; this test indexes 300 s of it,
    which halves every count and changes nothing else).

    The injected drift is +104 ppm, and compensating for it recovers 4% more
    votes.  That is not a measurement, it is a coin flip: 104 ppm over 600 s
    is 65 ms, less than one smoothed histogram bin, so there is nothing to
    recover.  An earlier version of this fit reported "+77 ppm" here with
    total confidence -- and reported "+77 ppm" just as confidently for two
    files off the *same* crystal.

    So the assertion is that the tool declines: coherence strong, offset
    right, and drift reported as "nothing above what this seed can resolve".
    """
    lib = tmp_path / "clean"
    lib.mkdir()
    # Half the seed's length is plenty of library file: the drift resolution
    # is set by how long the *seed* is, and 300 s still overlaps in full.
    cut(pair_corpus.seed_file("0077"), str(lib / "clean_0077S34.wav"),
        0.0, 300.0)
    db_path = str(tmp_path / "clean.db")
    with open_db(db_path) as db:
        assert run_index(db, str(lib), workers=1,
                         progress_stream=None)["indexed"] == 1

    hits, _s, _sig, note = _search(db_path, other_recorder)
    assert not note, note
    best = hits[0]
    c = best.coherence
    print(f"\ndrift fit: injected {DRIFT_PPM:+.1f} ppm, reported "
          f"{c.drift_ppm:+.1f} ppm (resolution {c.drift_resolution_ppm:.0f} "
          f"ppm) from {c.votes} votes at {c.sharpness:.1f}x over "
          f"{c.slopes_tried} candidate slopes")

    assert best.verdict == "PAIR"
    assert c.level == "strong"
    assert abs(c.offset_seconds) < 1.0
    assert c.drift_measurable and c.slopes_tried > 1
    assert c.drift_resolution_ppm > DRIFT_PPM, (
        "this seed is long enough to resolve the injected drift, so the test "
        "no longer demonstrates what it claims to")
    assert c.drift_ppm == 0.0, (
        f"claimed {c.drift_ppm:+.1f} ppm from a seed that cannot see it")
    assert "no clock drift above" in "\n".join(best.evidence)


@requires_corpus
def test_same_recorder_pairs_are_not_credited_with_a_drift(pair_results):
    """Two files off one crystal have no drift, and must be told so.

    Regression: with a one-frame slope grid and no gain requirement, the pak
    pair's 115-vote histogram peaked at +77 ppm rather than 0 -- by a single
    vote -- and the output announced it as a measurement.
    """
    for session, (hits, _s, _sig) in pair_results.items():
        c = hits[0].coherence
        assert c.level == "strong", session
        assert c.drift_ppm == 0.0, (session, c.drift_ppm)


# --------------------------------------------------------------------------
# 3. Negatives
# --------------------------------------------------------------------------


@requires_corpus
def test_unrelated_sessions_never_reach_timeline_match(pair_results,
                                                       pair_corpus):
    """The negative distribution of *these twelve pairs*, asserted.

    Every seed against every library file it is *not* related to.  Measured
    here (12 unrelated pairs, 10-minute excerpts): 0.141, 0.182, 0.259, 0.267,
    0.304, 0.338, 0.347, 0.396, 0.417, 0.439, 0.568, 0.620 -- median 0.34,
    ceiling 0.620.

    Twelve pairs is nowhere near enough to calibrate a threshold on, and this
    test is not the calibration: a later sweep of ~15 000 - 22 000 negative
    excerpt pairs found ceilings of 0.838 at a 5-minute overlap and 0.804 at
    fifteen, which is what ``PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS`` exists for.
    What this test asserts is narrower and still worth having: on this corpus,
    at this excerpt length, no unrelated file reaches even TIMELINE MATCH.
    """
    scores = []
    for session, (hits, _s, _sig) in sorted(pair_results.items()):
        want = os.path.basename(pair_corpus.lib_file(session))
        for h in hits:
            if os.path.basename(h.path) == want:
                continue
            scores.append((h.alignment.score, session,
                           os.path.basename(h.path), h.verdict))
    assert len(scores) == 12
    scores.sort()
    print("\nunrelated-pair envelope scores:")
    for s, seed, other, verdict in scores:
        print(f"  {s:.3f}  [{verdict}]  {seed} vs {other}")
    assert all(v == "weak" for _s, _a, _b, v in scores)
    assert scores[-1][0] < config.PAIR_R_LIKELY, scores[-1]


@requires_earlier
def test_a_short_overlap_cannot_print_pair_on_the_envelope_alone(
        false_pair_db, false_pair_corpus: FalsePairCorpus):
    """The false-PAIR repro from the review of this mode, on real audio.

    Seed: the first five minutes of ``pakDR40_S34``.  Library: 15:00-25:00 of
    ``pakDR40_earlier`` -- a different recording, from a different part of the
    card, with no shared audio and no shared wall-clock time.  Mode 1 on the
    same two files shows 1.1x sharpness, i.e. nothing.

    Measured before the length gate existed::

        [PAIR] envelope r=+0.94 at lag +24s (scored +0.86 over a 300s overlap)
               acoustic coherence: none

    Every downstream line agreed with it, because coherence can only confirm
    and the mode-2 score is supporting evidence.  Nothing could have caught it
    except the one fact that was already on the page: 300 seconds is not
    enough overlap for a loudness correlation to mean this.
    """
    hits, seconds, _sig, note = _search(false_pair_db,
                                        false_pair_corpus.seed, top=5)
    assert not note, note
    assert seconds == pytest.approx(FALSE_PAIR_SEED_SECONDS, abs=2.0)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.duration == pytest.approx(FALSE_PAIR_LIB_SECONDS, abs=2.0)
    print("\nfalse-pair repro ->\n" + _report(hits))

    # The measurement this test is built on: still a high envelope score, still
    # no coherence, still a short overlap.  If any of these stop holding, the
    # test has stopped exercising the gate and should be re-derived.
    assert hit.alignment.score >= config.PAIR_R_STRONG, _report(hits)
    assert hit.coherence.level == "none", _report(hits)
    assert (hit.alignment.overlap_seconds
            < config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS)
    assert hit.alignment.overlap_seconds == pytest.approx(
        FALSE_PAIR_SEED_SECONDS, abs=2.0)

    assert hit.verdict != "PAIR", _report(hits)
    assert hit.verdict == "TIMELINE MATCH", _report(hits)
    assert hit.capped_by_overlap
    text = "\n".join(hit.evidence)
    assert "capped at TIMELINE MATCH" in text, text
    assert "seed with the whole file" in text, text
    # And the soft "no coherence over a long overlap" note must NOT appear:
    # this overlap is not long, and the caution already says the useful thing.
    assert "note: no shared-clock evidence" not in text, text


@requires_earlier
def test_the_false_pair_repro_reads_the_same_through_the_cli(
        false_pair_db, false_pair_corpus: FalsePairCorpus, capsys):
    """The verdict and both cautions have to survive to the printed page."""
    assert main(["--db", false_pair_db, "query", false_pair_corpus.seed,
                 "--mode", "pair"]) == 0
    out = capsys.readouterr().out
    print(out)
    assert re.search(r"^ 1\. \[ *TIMELINE MATCH *\] ", out, re.M), out
    assert "[     PAIR" not in out
    # The per-hit caution, and the header one that fires on the seed length.
    assert "short overlap (300s): envelope-only verdicts are capped" in out
    assert "short seed (5:00.0): envelope-only verdicts are capped" in out
    assert "20 minutes of overlap" in out
    assert "seed with the whole file" in out
    assert "TIMELINE MATCH needs r >= 0.65 and means the loudness timelines "\
           "align" in out


@requires_corpus
@pytest.mark.parametrize("session", ["0048"])
def test_strong_coherence_lifts_a_short_overlap_to_pair(pair_results,
                                                        pair_corpus, session):
    """The gate's bypass, on the real corpus.

    A ten-minute S12/S34 excerpt pair overlaps by 600 s -- half of what the
    envelope needs to speak on its own -- so this hit is PAIR only because the
    two files share landmarks that agree on one offset.  That is exactly the
    second opinion the gate asks for, and it must still be enough.
    """
    hits, _s, _sig = pair_results[session]
    best = hits[0]
    assert os.path.basename(best.path) == os.path.basename(
        pair_corpus.lib_file(session))
    assert (best.alignment.overlap_seconds
            < config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS), _report(hits)
    assert not best.envelope_is_trusted_alone
    assert best.coherence.level == "strong", _report(hits)
    assert best.verdict == "PAIR", _report(hits)
    # It was never capped, so it must not carry the caution.
    assert not best.capped_by_overlap
    assert not any("capped at TIMELINE MATCH" in line
                   for line in best.evidence)


@requires_corpus
def test_every_reported_hit_gets_the_coherence_pass(pair_db, pair_corpus,
                                                    tmp_path):
    """``--top`` above ``PAIR_COHERENCE_CANDIDATES`` must not report an
    absence of evidence that was never sought.

    The coherence set used to be capped at 20 while ``--top 25`` reported 25,
    so the 21st hit printed "acoustic coherence: none -- consistent with a
    capture on different equipment" for a file whose landmarks had not been
    looked at once.  Here the seed's own envelope is planted in 20 decoy rows
    so that they outrank the real dual-record mate *by envelope score*, which
    then has to be reported with its (strong) coherence anyway.

    The decoys score ~1.0 with no coherence at all, so the length gate demotes
    every one of them to TIMELINE MATCH and the real mate is printed first --
    which is the point of the gate, and the reason the "did it reach the
    coherence pass?" assertion below is made against the envelope-score order
    rather than the reported order.
    """
    import shutil
    import sqlite3

    from audiomatch.analyze import analyze_seed_full

    db_path = str(tmp_path / "padded.db")
    shutil.copy(pair_db, db_path)
    seed = pair_corpus.seed_file("0077")
    codes = analyze_seed_full(seed).envelope

    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT INTO files(path, alive, size, mtime, status, error, duration,"
        " sample_rate, channels, bits, codec, take, role, n_hashes, noise,"
        " hum, chan, envelope, indexed_at) "
        "VALUES(?,1,0,0,'ok',NULL,?,44100,2,24,'pcm_s24le',NULL,NULL,0,"
        "?,?,?,?,0)",
        [(f"/decoy/pad{i:02d}.wav", float(codes.size), b"", b"", b"",
          codes.tobytes())
         for i in range(config.PAIR_COHERENCE_CANDIDATES)])
    con.commit()
    con.close()

    hits, _s, _sig, note = _search(db_path, seed,
                                   top=config.PAIR_COHERENCE_CANDIDATES + 5)
    assert not note, note
    mate = os.path.basename(pair_corpus.lib_file("0077"))
    found = [h for h in hits if os.path.basename(h.path) == mate]
    assert found, _report(hits)
    assert found[0].coherence.level == "strong", _report(hits)

    by_score = sorted(hits, key=lambda h: -h.alignment.score)
    place = [i for i, h in enumerate(by_score)
             if os.path.basename(h.path) == mate][0]
    assert place >= config.PAIR_COHERENCE_CANDIDATES, _report(by_score)
    # The decoys are envelope-only, so none of them may claim to be a pair.
    assert all(h.verdict == "TIMELINE MATCH"
               for h in hits if h.path.startswith("/decoy/")), _report(hits)


# --------------------------------------------------------------------------
# 4. Short and degenerate seeds
# --------------------------------------------------------------------------


@requires_corpus
def test_a_thirty_second_seed_is_refused_with_a_reason(pair_db, pair_corpus,
                                                       tmp_path):
    seed = str(tmp_path / "short.wav")
    cut(pair_corpus.seed_file("0072"), seed, 0.0, 30.0)
    hits, seconds, sig, note = _search(pair_db, seed)
    assert hits == []
    assert seconds == pytest.approx(30.0, abs=1.0)
    assert sig is not None, "the seed was still analysed"
    assert "at least" in note and "60s" in note


def test_a_silent_seed_does_not_crash(tmp_path):
    """Silence has no envelope shape to align, and must say so."""
    lib = tmp_path / "lib"
    lib.mkdir()
    run_ffmpeg(["-f", "lavfi", "-i", "anoisesrc=d=150:c=pink:r=16000:a=0.4",
                "-ac", "2", "-c:a", "pcm_s16le", str(lib / "noise.wav")])
    seed = str(tmp_path / "silence.wav")
    run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=stereo", "-t", "150",
                "-c:a", "pcm_s16le", seed])

    db_path = str(tmp_path / "s.db")
    with open_db(db_path) as db:
        run_index(db, str(lib), workers=1, progress_stream=None)
    hits, _s, _sig, note = _search(db_path, seed)
    assert hits == []
    assert "flat" in note or "overlap" in note, note


def test_a_seed_longer_than_every_candidate_still_works(tmp_path):
    """A 10-minute seed against a 3-minute library file: the comparison is
    scored on the candidate's own three minutes, not diluted by the rest.

    The loudness modulation is deliberately aperiodic -- two incommensurate
    sines -- so that exactly one lag can be right.  A repeating pattern would
    align equally well at several lags and the test would prove nothing.
    """
    lib = tmp_path / "lib"
    lib.mkdir()
    long_wav = str(tmp_path / "long.wav")
    run_ffmpeg(["-f", "lavfi",
                "-i", "anoisesrc=d=600:c=pink:r=16000:a=0.4:seed=1",
                "-filter_complex",
                "[0:a]volume="
                "'0.05+0.95*gt(sin(2*PI*t/97)+sin(2*PI*t/53),0.35)'"
                ":eval=frame,aformat=channel_layouts=stereo",
                "-c:a", "pcm_s16le", long_wav])
    run_ffmpeg(["-ss", "200", "-t", "180", "-i", long_wav,
                "-c", "copy", str(lib / "excerpt.wav")])
    db_path = str(tmp_path / "l.db")
    with open_db(db_path) as db:
        run_index(db, str(lib), workers=1, progress_stream=None)

    hits, _s, _sig, note = _search(db_path, long_wav)
    assert not note, note
    assert len(hits) == 1
    a = hits[0].alignment
    assert a.ok
    assert a.lag == pytest.approx(-200, abs=2)
    assert a.overlap == pytest.approx(180, abs=2)
    assert a.raw_r > 0.9


def test_an_index_with_no_envelopes_at_all_says_so(tmp_path):
    import sqlite3
    from test_index import synthetic_library

    lib = synthetic_library(tmp_path / "lib", n=2, seconds=70.0)
    db_path = str(tmp_path / "n.db")
    with open_db(db_path) as db:
        run_index(db, lib, workers=1, progress_stream=None)
    con = sqlite3.connect(db_path)
    con.execute("UPDATE files SET envelope = NULL")
    con.commit()
    con.close()

    warnings: list[str] = []
    hits, _s, _sig, note = _search(db_path, os.path.join(lib, "tone00.wav"),
                                   warn=warnings.append)
    assert hits == []
    assert "backfill" in note
    assert any("backfill" in w for w in warnings)


# --------------------------------------------------------------------------
# 5. The drift fit, in isolation
# --------------------------------------------------------------------------


def test_drift_slopes_grow_with_the_seed_length():
    """A short seed must not pretend to have measured a drift."""
    assert list(drift_slopes(0)) == [0.0]
    assert drift_grid(0)[1] == 0.0
    short = drift_slopes(int(150 * config.FRAME_RATE))
    assert list(short) == [0.0], "150 s cannot resolve any drift"

    ten_min, ten_step = drift_grid(int(600 * config.FRAME_RATE))
    long, long_step = drift_grid(int(2700 * config.FRAME_RATE))
    assert 1 < ten_min.size < long.size <= config.PAIR_MAX_DRIFT_SLOPES
    assert ten_step > long_step >= config.PAIR_DRIFT_MIN_STEP_PPM
    for grid in (ten_min, long):
        assert grid[0] == 0.0
        assert np.all(np.abs(grid) <= config.PAIR_MAX_DRIFT_PPM)
        # Ordered by magnitude, so ties resolve toward "no drift".
        assert np.all(np.diff(np.abs(grid)) >= 0)
    assert min(np.abs(long[1:])) == pytest.approx(long_step)
    # The step is one smoothed histogram bin's worth of total drift.
    frames = int(2700 * config.FRAME_RATE)
    assert long_step == pytest.approx(
        max(config.PAIR_DRIFT_MIN_STEP_PPM,
            1e6 * (2 * config.OFFSET_SMOOTH + 1) / frames))


def test_a_drift_that_wins_by_luck_is_not_reported_as_a_measurement():
    """The gain bar, in isolation.

    A perfectly aligned, non-drifting pair plus coincidence noise.  Some
    non-zero slope will always scrape a vote or two more than zero; it must
    not be dressed up as +N ppm.
    """
    rng = np.random.default_rng(31)
    seed_frames = int(2700 * config.FRAME_RATE)
    t = rng.integers(0, seed_frames, 1500).astype(np.int64)
    delta = np.full(t.size, 17, dtype=np.int64)
    noise_t = rng.integers(0, seed_frames, 6000).astype(np.int64)
    noise_d = rng.integers(-1200, 1200, 6000).astype(np.int64)

    c = fit_coherence(np.concatenate([t, noise_t]),
                      np.concatenate([delta, noise_d]),
                      seed_frames=seed_frames, center_frames=17)
    assert c.level == "strong"
    assert c.drift_ppm == 0.0
    assert c.drift_measurable and c.drift_resolution_ppm > 0


@pytest.mark.parametrize("ppm", [0.0, 120.0, -120.0, 240.0])
def test_fit_coherence_recovers_an_injected_drift(ppm):
    """Synthetic landmark votes on a line, plus a flat coincidence floor."""
    rng = np.random.default_rng(4)
    seed_frames = int(2700 * config.FRAME_RATE)
    offset = 43                                   # ~1 s of constant offset
    t = np.sort(rng.integers(0, seed_frames, 4000)).astype(np.int64)
    delta = np.rint(offset + ppm * 1e-6 * t).astype(np.int64)
    # Coincidence noise: three times as many votes, scattered over the window.
    noise_t = rng.integers(0, seed_frames, 12000).astype(np.int64)
    noise_d = rng.integers(-1200, 1200, 12000).astype(np.int64)

    c = fit_coherence(np.concatenate([t, noise_t]),
                      np.concatenate([delta, noise_d]),
                      seed_frames=seed_frames, center_frames=offset)
    assert c.drift_measurable
    assert abs(c.drift_ppm - ppm) <= c.drift_resolution_ppm
    assert abs(c.offset_frames - offset) <= 2
    assert c.level == "strong"
    assert c.votes > 500


def test_fit_coherence_on_pure_noise_reports_none():
    rng = np.random.default_rng(9)
    seed_frames = int(2700 * config.FRAME_RATE)
    t = rng.integers(0, seed_frames, 3000).astype(np.int64)
    d = rng.integers(-1200, 1200, 3000).astype(np.int64)
    c = fit_coherence(t, d, seed_frames=seed_frames, center_frames=0)
    assert c.level == "none", (c.votes, c.sharpness)


def test_fit_coherence_with_no_postings_is_empty_not_an_error():
    z = np.zeros(0, np.int64)
    c = fit_coherence(z, z, seed_frames=1000, center_frames=0)
    assert c.votes == 0 and c.level == "none"


# --------------------------------------------------------------------------
# 6. Verdicts and CLI
# --------------------------------------------------------------------------


#: Long enough for the envelope to be trusted on its own.
LONG = int(config.PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS)


def _hit(score: float, coherence: str = "none", overlap: int = LONG,
         dominance: float = 2.0, **kw) -> PairHit:
    """A synthetic hit.  The default alignment has a *dominant* envelope peak,
    which is the ordinary case: 56 of the 56 correct pairs in the measured set
    clear the bar and none of them is what the degenerate-peak path is for."""
    levels = {
        "strong": Coherence(votes=200, background=4),
        "weak": Coherence(votes=12, background=4),
        "none": Coherence(),
    }
    assert levels[coherence].level == coherence
    return PairHit(file_id=1, path="/lib/x.wav", duration=float(overlap),
                   alignment=E.Alignment(ok=True, lag=0, score=score,
                                         raw_r=score, overlap=overlap,
                                         dominance=dominance),
                   coherence=levels[coherence], **kw)


def test_the_two_routes_to_a_pair_verdict():
    # Envelope alone, when it is decisive *and* there is enough of it.
    assert _hit(0.90).verdict == "PAIR"
    # Coherence promotes a merely-plausible envelope score.
    assert _hit(0.70).verdict == "TIMELINE MATCH"
    assert _hit(0.70, "strong").verdict == "PAIR"
    # But coherence cannot rescue an envelope that disagrees.
    assert _hit(0.40, "strong").verdict == "weak"
    # Weak coherence is not a promotion.
    assert _hit(0.70, "weak").verdict == "TIMELINE MATCH"
    assert _hit(0.64).verdict == "weak"
    # An unusable alignment is never a pair.
    bad = _hit(0.99)
    bad.alignment = E.Alignment(ok=False, reason="too short")
    assert bad.verdict == "weak"


def test_the_envelope_alone_needs_a_long_overlap_to_say_pair():
    """The length gate, at its boundary.

    Below ``PAIR_ENVELOPE_TRUST_OVERLAP_SECONDS`` the measured negative
    distribution reaches 0.838, so a high envelope score with nothing behind it
    is not a pair verdict however high it goes.
    """
    assert _hit(0.99, overlap=LONG - 1).verdict == "TIMELINE MATCH"
    assert _hit(0.99, overlap=LONG).verdict == "PAIR"
    assert _hit(0.86, overlap=300).verdict == "TIMELINE MATCH"
    # Weak coherence is not the second opinion the gate is asking for.
    assert _hit(0.86, "weak", overlap=300).verdict == "TIMELINE MATCH"
    # Strong coherence is, at any length: it needs genuinely shared audio.
    assert _hit(0.86, "strong", overlap=300).verdict == "PAIR"
    assert _hit(0.86, "strong", overlap=60).verdict == "PAIR"
    # The gate cannot promote anything: it only ever holds a verdict back.
    assert _hit(0.50, overlap=LONG * 4).verdict == "weak"


def test_a_capped_hit_says_so_and_says_what_to_do_about_it():
    hit = _hit(0.86, overlap=300)
    assert hit.capped_by_overlap and not hit.envelope_is_trusted_alone
    line = [ln for ln in hit.evidence if ln.startswith("short overlap")]
    assert len(line) == 1, hit.evidence
    assert "300s" in line[0]
    assert "capped at TIMELINE MATCH" in line[0]
    assert "1200s" in line[0]
    assert "seed with the whole file" in line[0]
    # No caution once the overlap is long enough...
    assert not any(ln.startswith("short overlap")
                   for ln in _hit(0.86).evidence)
    # ...nor when coherence has already lifted the hit past the gate.
    short_but_confirmed = _hit(0.86, "strong", overlap=300)
    assert not short_but_confirmed.capped_by_overlap
    assert not any(ln.startswith("short overlap")
                   for ln in short_but_confirmed.evidence)
    # ...nor for a score that was never near PAIR anyway.
    assert not any(ln.startswith("short overlap")
                   for ln in _hit(0.70, overlap=300).evidence)


def test_a_very_short_overlap_is_flagged_whatever_the_verdict():
    """Below five minutes the correlation fails in *both* directions.

    Measured against ground truth on a five-recorder live set: a 61-second seed
    scored r = -0.52 against its own true dual-record mate (ranked 15th of 16),
    and a 112-second seed returned seven mutually contradictory matches at
    implied lags from 919 s to 2473 s.  So the warning cannot be limited to
    hits the PAIR gate holds back -- a *missed* mate looks like a weak hit, and
    it needs the same health warning.
    """
    for score in (0.90, 0.70, 0.30):
        hit = _hit(score, overlap=112)
        line = [ln for ln in hit.evidence
                if ln.startswith("very short overlap")]
        assert len(line) == 1, (score, hit.evidence)
        assert "112s" in line[0]
        assert "unreliable in both directions" in line[0]
        assert "capped at TIMELINE MATCH below 1200s" in line[0]
        assert "seed with the whole file" in line[0]
        # One caution, not two: this line already names the cap.
        assert not any(ln.startswith("short overlap") for ln in hit.evidence)

    # It is about the overlap, not the verdict: strong coherence buys a PAIR
    # here and the warning stays.
    confirmed = _hit(0.90, "strong", overlap=112)
    assert confirmed.verdict == "PAIR"
    assert any(ln.startswith("very short overlap")
               for ln in confirmed.evidence)

    # Just above the line, the milder gate caution takes over.
    boundary = _hit(0.90, overlap=int(config.PAIR_UNRELIABLE_OVERLAP_SECONDS))
    assert not any(ln.startswith("very short") for ln in boundary.evidence)
    assert any(ln.startswith("short overlap") for ln in boundary.evidence)


def test_a_long_overlap_with_no_coherence_at_all_is_annotated():
    """Soft, informational, and never a verdict change.

    A long envelope match with no shared landmarks whatsoever is the normal
    reading for a second recorder -- and also what a pair of files that are
    *supposed* to be one recorder's two microphone pairs would look like if
    they were not.  The tool cannot tell which, so it says so and leaves the
    verdict alone.
    """
    hit = _hit(0.86)
    assert hit.verdict == "PAIR"
    note = [ln for ln in hit.evidence if ln.startswith("note:")]
    assert len(note) == 1, hit.evidence
    assert "no shared-clock evidence despite a long overlap" in note[0]
    assert "suspicious if these files should share a recorder" in note[0]
    # Not on a short overlap (that hit gets the caution instead), not when
    # coherence fired, and not below the PAIR bar.
    for other in (_hit(0.86, overlap=300), _hit(0.86, "strong"),
                  _hit(0.86, "weak"), _hit(0.70)):
        assert not any(ln.startswith("note:") for ln in other.evidence), other


def test_a_different_recorder_reads_as_timeline_match_not_a_failure():
    hit = _hit(0.72, "none")
    assert hit.verdict == "TIMELINE MATCH"
    text = "\n".join(hit.evidence)
    assert "coherence: none" in text
    assert "different equipment" in text


@requires_corpus
def test_pair_mode_end_to_end_through_the_cli(pair_db, pair_corpus, capsys):
    seed = pair_corpus.seed_file("0077")
    assert main(["--db", pair_db, "query", seed, "--mode", "pair",
                 "--top", "4"]) == 0
    out = capsys.readouterr().out
    assert "MODE 3: pair mates" in out
    assert "MODE 1" not in out and "MODE 2" not in out
    assert "TASCAM_0077S12.wav" in out
    assert re.search(r"^ 1\. \[ *PAIR *\] .*TASCAM_0077S12\.wav",
                     out, re.M), out
    assert "envelope r=" in out
    assert "acoustic coherence: strong" in out
    assert "segments align:" in out
    # Every result carries a verdict label.
    numbered = [ln for ln in out.splitlines() if ln.startswith(" 1. [")]
    assert len(numbered) == 1


@requires_corpus
def test_mode_both_is_unchanged_by_the_addition_of_pair_mode(pair_db,
                                                             pair_corpus,
                                                             tmp_path,
                                                             capsys):
    """`--mode both` must still mean match + session and nothing else."""
    seed = str(tmp_path / "short_both.wav")
    cut(pair_corpus.lib_file("0072"), seed, 30.0, 45.0)
    with open_db(pair_db, create=False) as db:
        res = run_query(db, seed, mode="both", top=3)
    assert res.pairs == [] and res.pair_note == ""
    assert res.matches and res.sessions

    main(["--db", pair_db, "query", seed, "--mode", "both", "--top", "3"])
    out = capsys.readouterr().out
    assert "MODE 1" in out and "MODE 2" in out and "MODE 3" not in out


# --------------------------------------------------------------------------
# 7. The channel that hears one source instead of the room
# --------------------------------------------------------------------------


def test_peak_dominance_tells_one_answer_from_several():
    """A dominant peak and a degenerate flat top, in isolation."""
    lags = np.arange(-600, 601)

    # One answer: a single song-scale bump, nothing else anywhere.
    single = 0.9 * np.exp(-(lags / 20.0) ** 2)
    assert E.peak_dominance(lags, single) > 100.0

    # A rival 300 s away that is nearly as good: several answers, no winner.
    twin = single + 0.87 * np.exp(-((lags - 300) / 20.0) ** 2)
    assert E.peak_dominance(lags, twin) == pytest.approx(0.9 / 0.87, rel=1e-3)
    assert E.peak_dominance(lags, twin) < config.PAIR_ENVELOPE_DOMINANCE_MIN

    # A flat top: the argmax is whichever bin noise happened to favour.
    flat = 0.5 + 0.004 * np.cos(lags / 31.0)
    assert E.peak_dominance(lags, flat) < 1.02

    # The rival has to be a *different alignment*, not the winning peak's own
    # shoulder, which is what the 60 s separation buys: a peak wide enough to
    # still be at 78% of its height a minute out is dominant, not degenerate.
    broad = 0.9 * np.exp(-(lags / 120.0) ** 2)
    assert broad[lags == 60][0] / broad.max() == pytest.approx(0.78, abs=0.01)
    assert E.peak_dominance(lags, broad) > config.PAIR_ENVELOPE_DOMINANCE_MIN

    # Degenerate inputs answer rather than divide by zero.
    assert E.peak_dominance(lags, np.zeros_like(lags, dtype=float)) == 0.0
    assert E.peak_dominance(lags, -np.abs(single)) == 0.0
    assert E.peak_dominance(np.arange(3), np.array([0.1, 0.9, 0.2])) == np.inf


#: Recorded peak geometry from the five-recorder live set, measured through
#: ``align()``'s own score curve against hand-made (sesx) ground truth.
#:
#: One row per unordered pair of the nine files whose true offsets are known:
#: ``(top1, top2, envelope lag error in seconds)``, where ``top2`` is the best
#: score at least 60 s from ``top1`` -- the same rule
#: :func:`envelope.peak_dominance` applies.  The reverse ordering of each pair
#: measures identically, so the 36 rows here are the 72 ordered pairs.
#:
#: The eight rows with a lag error are every pair involving one direct-input
#: channel; they are wrong by 103 s to 1644 s.
MEASURED_PEAKS = (
    (0.4565, 0.4507, 632), (0.4443, 0.4384, 452), (0.4915, 0.4781, 479),
    (0.4964, 0.4823, 477), (0.6609, 0.6418, 271), (0.5256, 0.5103, 106),
    (0.2629, 0.2472, 1644), (0.2789, 0.2542, 103),
    (0.6476, 0.5417, 0), (0.6619, 0.5452, 0), (0.8224, 0.6652, 0),
    (0.8813, 0.7008, 0), (0.9169, 0.7211, 0), (0.8017, 0.6250, 0),
    (0.9402, 0.7287, 0), (0.8554, 0.6624, 0), (0.8992, 0.6954, 0),
    (0.7802, 0.5963, 0), (0.9198, 0.6929, 0), (0.9176, 0.6888, 0),
    (0.7316, 0.5448, 0), (0.6736, 0.4972, 0), (0.8979, 0.6605, 0),
    (0.8714, 0.6379, 0), (0.9484, 0.6878, 0), (0.8527, 0.6182, 0),
    (0.8888, 0.6414, 0), (0.6980, 0.4984, 0), (0.9530, 0.6609, 0),
    (0.5539, 0.3796, 0), (0.5236, 0.3556, 0), (0.9321, 0.6235, 0),
    (0.9203, 0.6127, 0), (0.8819, 0.5830, 0), (0.8911, 0.5879, 0),
    (0.9130, 0.5779, 0),
)


def test_the_dominance_bar_sits_between_the_two_measured_populations():
    """The trigger, on the geometry it was derived from.

    This is the calibration, asserted: on real recordings with known true
    offsets, every pair whose envelope lag is *wrong* has a degenerate peak
    (1.013 .. 1.097) and every pair whose lag is *right* has a dominant one
    (1.196 .. 1.580).  The two populations do not touch, and
    ``PAIR_ENVELOPE_DOMINANCE_MIN`` sits in the gap.

    If a future change to the score curve moves either population across the
    bar, this fails -- which is the point.
    """
    lags = np.array([0, 100])          # 100 s apart: a genuine rival lag
    wrong, right = [], []
    for top1, top2, err in MEASURED_PEAKS:
        dom = E.peak_dominance(lags, np.array([top1, top2]))
        assert dom == pytest.approx(top1 / top2)
        (wrong if err > 1 else right).append(dom)

    assert len(wrong) == 8 and len(right) == 28          # 16 and 56, ordered
    print(f"\nmeasured dominance: wrong lags "
          f"{min(wrong):.3f}..{max(wrong):.3f}  correct lags "
          f"{min(right):.3f}..{max(right):.3f}")
    caught = [d for d in wrong if d < config.PAIR_ENVELOPE_DOMINANCE_MIN]
    disturbed = [d for d in right if d < config.PAIR_ENVELOPE_DOMINANCE_MIN]
    assert len(caught) == len(wrong), "a wrong envelope lag would go unchecked"
    assert not disturbed, "a correct pair would be sent down the fallback"
    assert max(wrong) < config.PAIR_ENVELOPE_DOMINANCE_MIN < min(right)


def test_a_degenerate_envelope_peak_is_flagged_when_nothing_settles_it():
    """The fallback ran (or could not run) and did not reach 'strong'.

    The hit keeps its envelope verdict -- there is nothing better to replace it
    with -- but the lag behind it is a coin flip, and the line says so.
    """
    hit = _hit(0.86, dominance=1.02)
    assert hit.envelope_peak_is_degenerate
    line = [ln for ln in hit.evidence if ln.startswith("envelope peak is not")]
    assert len(line) == 1, hit.evidence
    assert "score curve is degenerate" in line[0]
    assert "lag unreliable" in line[0]
    assert "close-mic/direct-input" in line[0]

    # Not when the peak stands out...
    assert not any(ln.startswith("envelope peak is not")
                   for ln in _hit(0.86).evidence)
    # ...nor when the landmarks confirmed the lag anyway...
    confirmed = _hit(0.86, "strong", dominance=1.02)
    assert not confirmed.envelope_peak_is_degenerate
    assert not any(ln.startswith("envelope peak is not")
                   for ln in confirmed.evidence)
    # ...nor on a hit that is claiming nothing in the first place.
    assert not any(ln.startswith("envelope peak is not")
                   for ln in _hit(0.30, dominance=1.02).evidence)


def test_a_fingerprint_placed_hit_never_prints_the_envelope_lag():
    """The overridden lag must not survive anywhere in the output."""
    hit = _hit(0.21, "strong", overlap=300, dominance=1.00,
               fingerprint_placed=True,
               superseded=E.Alignment(ok=True, lag=-1644, score=0.26,
                                      raw_r=0.28, overlap=300,
                                      dominance=1.00))
    hit.alignment = E.Alignment(ok=True, lag=100, score=0.21, raw_r=0.21,
                                overlap=300, dominance=1.00)
    hit.coherence.offset_frames = int(round(100 / config.FRAME_SECONDS))

    assert hit.verdict == "PAIR", "strong coherence placed it; that is enough"
    assert not hit.capped_by_overlap
    assert not hit.envelope_peak_is_degenerate
    text = "\n".join(hit.evidence)
    print("\n" + text)
    assert text.startswith("placed by acoustic fingerprint despite an "
                           "uninformative envelope -- typical of close-mic or "
                           "direct-input channels")
    assert "acoustic coherence: strong" in text
    # The rejected lag appears only as the thing that was rejected.
    assert "the envelope's own best lag was -1644s" in text
    assert "peak dominance of 1.00" in text
    assert not any(ln.startswith("envelope r=") for ln in hit.evidence)
    # And the length cautions, which are all about trusting the envelope, are
    # not printed over a verdict that does not rest on it.
    assert "capped at TIMELINE MATCH" not in text
    assert "very short overlap" not in text


def test_the_unwindowed_search_is_sized_to_the_deltas_it_has():
    """Range-sized, not window-sized -- the difference is 8.6 M bins.

    Two hour-long files whose landmark deltas spread over three hours: the
    histogram may be as big as what was actually observed and no bigger.  A
    fixed half-window wide enough for any pair of files in a library (say a day
    of possible offsets) is millions of bins per slope and seconds per
    candidate, which is what made an unwindowed search look unaffordable.
    """
    rng = np.random.default_rng(11)
    seed_frames = int(3600 * config.FRAME_RATE)
    offset = int(round(1500.0 / config.FRAME_SECONDS))     # 25 minutes out
    t = rng.integers(0, seed_frames, 600).astype(np.int64)
    d = np.full(t.size, offset, dtype=np.int64)
    span = int(round(3600.0 / config.FRAME_SECONDS))
    noise_t = rng.integers(0, seed_frames, 40_000).astype(np.int64)
    noise_d = rng.integers(-span, 2 * span, 40_000).astype(np.int64)
    seed_t = np.concatenate([t, noise_t])
    deltas = np.concatenate([d, noise_d])

    import time
    start = time.perf_counter()
    c = fit_coherence_global(seed_t, deltas, seed_frames=seed_frames)
    elapsed = time.perf_counter() - start
    print(f"\nunwindowed search over {deltas.size:,} matched landmark pairs "
          f"and {c.bins:,} bins ({c.slopes_tried} drift slopes): "
          f"{elapsed * 1000:.0f} ms")

    assert c.unwindowed and c.level == "strong"
    assert abs(c.offset_frames - offset) <= 2
    assert c.bins == int(deltas.max()) - int(deltas.min()) + 1
    assert c.bins <= 3 * span + 2, "wider than the deltas that were observed"

    naive = 2 * int(round(100_000.0 / config.FRAME_SECONDS)) + 1
    assert naive > 8_000_000            # the allocation this test forbids
    assert c.bins < naive / 10          # measured: 5.4% of it


def test_the_unwindowed_search_declines_rather_than_allocating(monkeypatch):
    """Above the bin ceiling it returns nothing instead of a huge array."""
    monkeypatch.setattr(config, "PAIR_GLOBAL_COHERENCE_MAX_BINS", 1000)
    rng = np.random.default_rng(5)
    seed_frames = int(600 * config.FRAME_RATE)
    t = rng.integers(0, seed_frames, 500).astype(np.int64)
    d = rng.integers(0, 50_000, 500).astype(np.int64)
    c = fit_coherence_global(t, d, seed_frames=seed_frames)
    assert c.votes == 0 and c.level == "none" and c.bins == 0
    # And a windowed fit of the same postings is unaffected by the ceiling.
    assert fit_coherence(t, d, seed_frames=seed_frames,
                         center_frames=0).bins > 1000


def test_the_unwindowed_search_matches_the_windowed_one_where_they_overlap():
    """Same machinery, same answer -- the window is the only difference."""
    rng = np.random.default_rng(21)
    seed_frames = int(2700 * config.FRAME_RATE)
    offset = 43
    t = np.sort(rng.integers(0, seed_frames, 3000)).astype(np.int64)
    d = np.rint(offset + 120.0 * 1e-6 * t).astype(np.int64)
    noise_t = rng.integers(0, seed_frames, 9000).astype(np.int64)
    noise_d = rng.integers(-1200, 1200, 9000).astype(np.int64)
    seed_t = np.concatenate([t, noise_t])
    deltas = np.concatenate([d, noise_d])

    win = fit_coherence(seed_t, deltas, seed_frames=seed_frames,
                        center_frames=offset)
    glob = fit_coherence_global(seed_t, deltas, seed_frames=seed_frames)
    assert glob.level == win.level == "strong"
    assert glob.votes == win.votes
    assert glob.offset_frames == win.offset_frames
    assert glob.drift_ppm == win.drift_ppm == pytest.approx(120.0, abs=30.0)


@requires_ffmpeg
def test_the_fingerprint_places_a_close_mic_seed_the_envelope_cannot(
        close_mic_db, close_mic_corpus: CloseMicCorpus):
    """The whole feature, end to end, on audio built to be this case.

    The seed is a synthetic direct-input channel: an independent loud source
    playing a repeating figure, with the room 25 dB down as bleed (see
    ``close_mic_corpus``).  Its envelope therefore describes its own player,
    not the performance, and proposes a lag ~100 s from the truth off a
    completely degenerate score curve.  The landmarks in the bleed still know
    the answer, and the unwindowed search is what asks them.
    """
    hits, seconds, _sig, note = _search(close_mic_db, close_mic_corpus.seed,
                                        top=3)
    assert not note, note
    assert seconds == pytest.approx(CLOSE_MIC_SECONDS, abs=2.0)
    assert len(hits) == 1
    hit = hits[0]
    a, sup, c = hit.alignment, hit.superseded, hit.coherence
    assert hit.fingerprint_placed and sup is not None, (
        "the envelope was believed here, so nothing exercised the fallback: "
        + _report(hits))
    print(f"\nclose-mic seed -> {os.path.basename(hit.path)} [{hit.verdict}]\n"
          f"  envelope proposed lag {sup.lag_seconds:+.0f}s at dominance "
          f"{sup.dominance:.3f} (score {sup.score:.3f})\n"
          f"  unwindowed search: {c.votes} votes, {c.sharpness:.1f}x, offset "
          f"{c.offset_seconds:+.2f}s over {c.bins:,} bins\n"
          f"  adopted lag {a.lag_seconds:+.0f}s, envelope there "
          f"r={a.raw_r:+.2f}")

    # The premise: the envelope's answer is degenerate *and* wrong.
    assert sup.dominance < config.PAIR_ENVELOPE_DOMINANCE_MIN, (
        "the envelope peak is dominant here, so this test no longer "
        "exercises the trigger")
    assert abs(sup.lag - close_mic_corpus.true_lag) > \
        config.PAIR_COHERENCE_WINDOW_SECONDS, (
        "the envelope lag is close enough that the windowed pass would have "
        "confirmed it, so nothing here needed the fallback")

    # The finding: the landmarks placed it, to the second.
    assert c.unwindowed and c.level == "strong"
    assert c.offset_seconds == pytest.approx(close_mic_corpus.true_lag,
                                             abs=1.0)
    assert a.lag == pytest.approx(close_mic_corpus.true_lag, abs=1)
    assert hit.verdict == "PAIR"

    text = "\n".join(hit.evidence)
    assert "placed by acoustic fingerprint despite an uninformative " \
           "envelope -- typical of close-mic or direct-input channels" in text
    assert f"the envelope's own best lag was {sup.lag_seconds:+.0f}s" in text
    assert "acoustic coherence: strong" in text
    assert not any(ln.startswith("envelope r=") for ln in hit.evidence)

    # And it reads that way on the printed page, with the adopted lag.
    from audiomatch.cli import print_pair_results
    from audiomatch.query import QueryResult
    out = io.StringIO()
    print_pair_results(QueryResult(seed_path=close_mic_corpus.seed,
                                   seed_seconds=seconds, pairs=hits), out)
    printed = out.getvalue()
    print(printed)
    assert re.search(r"^ 1\. \[ *PAIR *\] ", printed, re.M), printed
    assert "the seed's 0:00 lands at 1:40.0 in this file" in printed
    assert "placed by acoustic fingerprint" in printed
