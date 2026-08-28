"""Mode 3: pair matching, against the real recovered DR-40 corpus.

The library holds one member of each dual-record pair (the S12 of each of the
four sessions, ten minutes each); the seeds are the other member (the S34),
cut from the same wall-clock range.  So "did the mate come top?" is a real
question, not one the seed's own recording could answer for it.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from audiomatch import config, envelope as E
from audiomatch.cli import main
from audiomatch.db import open_db
from audiomatch.indexer import run_index
from audiomatch.query import (Coherence, PairHit, drift_grid, drift_slopes,
                              fit_coherence, pair_search, run_query)

from conftest import (PAIR_EXCERPT_SECONDS, SESSIONS, PairCorpus, cut,
                      requires_corpus, requires_ffmpeg, run_ffmpeg)

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
        f"  {i}. [{h.verdict:11s}] {os.path.basename(h.path):22s} "
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

    Every one of them clears ``PAIR_R_STRONG`` on the envelope alone, before
    coherence is consulted; the best unrelated candidate in any of the four
    queries reaches 0.620.  The 0072 margin over 0.80 is the thinnest at 0.02,
    which is worth knowing if these thresholds are ever retuned.
    """
    hits, seconds, _sig = pair_results[session]
    assert seconds == pytest.approx(PAIR_EXCERPT_SECONDS, abs=2.0)

    best = hits[0]
    want = os.path.basename(pair_corpus.lib_file(session))
    report = f"\npair seed {session}S34:\n" + _report(hits)
    print(report)

    assert os.path.basename(best.path) == want, report
    assert best.verdict == "PAIR"
    # The envelope alone must clear the PAIR bar -- coherence is a bonus here,
    # not a crutch.
    assert best.alignment.score >= config.PAIR_R_STRONG
    # Same wall-clock range, so the two files line up at lag 0.  pakDR40_S12
    # has 0.74 s of silence prepended by the recovery, which is sub-sample at
    # 1 Hz and must not move the answer.
    assert abs(best.alignment.lag) <= 2
    # Same recorder, same clock: coherence must fire.
    assert best.coherence.level == "strong"
    assert abs(best.coherence.offset_seconds) < 1.0
    # And every unrelated session must stay below LIKELY PAIR.
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
        f"  [{h.verdict:11s}] {os.path.basename(h.path):22s} "
        f"score {h.alignment.score:.3f} raw {h.alignment.raw_r:.3f} "
        f"lag {h.alignment.lag:+4d}s coherence {h.coherence.level} "
        f"({h.coherence.votes} votes, ppm {h.coherence.drift_ppm:+.0f})"
        for h in hits)
    print("\nsimulated other recorder ->\n" + report)

    assert os.path.basename(best.path) == "TASCAM_0077S12.wav", report
    assert abs(best.alignment.lag) <= 2, report
    assert best.verdict in ("PAIR", "LIKELY PAIR"), report
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
def test_unrelated_sessions_never_reach_likely_pair(pair_results,
                                                    pair_corpus):
    """The measured negative distribution, asserted.

    Every seed against every library file it is *not* related to.  These are
    the hardest negatives available: the same band, the same recorder, the
    same room, sets of similar length and shape, recorded on different days.

    Measured here (12 unrelated pairs, 10-minute excerpts): 0.141, 0.182,
    0.259, 0.267, 0.304, 0.338, 0.347, 0.396, 0.417, 0.439, 0.568, 0.620 --
    median 0.34, ceiling 0.620.  Measured on the *whole* files rather than
    excerpts (48 unrelated ordered pairs): 0.402 .. 0.584, median 0.50,
    against true pairs at 0.887 .. 0.937.

    ``PAIR_R_LIKELY`` = 0.65 sits above both ceilings, but only by 0.03 over
    the excerpt one, which is the tightest margin anywhere in this mode.
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


def _hit(score: float, coherence: str = "none", **kw) -> PairHit:
    levels = {
        "strong": Coherence(votes=200, background=4),
        "weak": Coherence(votes=12, background=4),
        "none": Coherence(),
    }
    assert levels[coherence].level == coherence
    return PairHit(file_id=1, path="/lib/x.wav", duration=600.0,
                   alignment=E.Alignment(ok=True, lag=0, score=score,
                                         raw_r=score, overlap=600),
                   coherence=levels[coherence], **kw)


def test_the_two_routes_to_a_pair_verdict():
    # Envelope alone, when it is decisive.
    assert _hit(0.90).verdict == "PAIR"
    # Coherence promotes a merely-likely envelope score.
    assert _hit(0.70).verdict == "LIKELY PAIR"
    assert _hit(0.70, "strong").verdict == "PAIR"
    # But coherence cannot rescue an envelope that disagrees.
    assert _hit(0.40, "strong").verdict == "weak"
    # Weak coherence is not a promotion.
    assert _hit(0.70, "weak").verdict == "LIKELY PAIR"
    assert _hit(0.64).verdict == "weak"
    # An unusable alignment is never a pair.
    bad = _hit(0.99)
    bad.alignment = E.Alignment(ok=False, reason="too short")
    assert bad.verdict == "weak"


def test_a_different_recorder_reads_as_likely_pair_not_a_failure():
    hit = _hit(0.72, "none")
    assert hit.verdict == "LIKELY PAIR"
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
