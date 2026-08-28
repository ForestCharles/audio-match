"""The activity envelope: quantisation, streaming, alignment, segments.

Pure unit tests -- no corpus, no ffmpeg.  The corpus-level behaviour of pair
mode lives in ``test_pair.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from audiomatch import config, envelope as E


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------


def test_quantisation_round_trips_within_one_step():
    db = np.linspace(config.ENVELOPE_DB_FLOOR, config.ENVELOPE_DB_CEIL, 400)
    mean_square = 10.0 ** (db / 10.0)
    back = E.dequantize(E.quantize(mean_square))
    assert np.max(np.abs(back - db)) <= E.DB_PER_STEP / 2 + 1e-4
    assert E.DB_PER_STEP == pytest.approx(120.0 / 255.0)


def test_quantisation_clamps_instead_of_wrapping():
    """Digital silence and above-full-scale must saturate, not wrap around."""
    codes = E.quantize(np.array([0.0, 1e-30, 1.0, 100.0]))
    assert codes[0] == 0 and codes[1] == 0
    assert codes[2] == config.ENVELOPE_LEVELS - 1
    assert codes[3] == config.ENVELOPE_LEVELS - 1
    assert codes.dtype == np.uint8


def test_blob_round_trip():
    codes = np.array([0, 7, 128, 255], dtype=np.uint8)
    np.testing.assert_array_equal(E.unpack(E.pack(codes)), codes)
    assert E.unpack(None).size == 0
    assert E.unpack(b"").size == 0


def test_one_byte_per_second():
    """The storage claim in config.py, checked."""
    rng = np.random.default_rng(7)
    mono = rng.standard_normal(45 * 60 * config.ANALYSIS_SR).astype(np.float32)
    assert len(E.pack(E.envelope_of(mono))) == 45 * 60


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def test_streaming_matches_a_single_push_whatever_the_block_size():
    """The indexer feeds ragged decode blocks; the answer must not depend on
    where ffmpeg happened to split them."""
    rng = np.random.default_rng(11)
    mono = (rng.standard_normal(37 * config.ANALYSIS_SR) * 0.1).astype(
        np.float32)
    reference = E.envelope_of(mono)
    for block in (1, 999, 11025, 11026, 65536, 1 << 18):
        c = E.EnvelopeCollector()
        for i in range(0, mono.size, block):
            c.push(mono[i:i + block])
        np.testing.assert_array_equal(c.result(), reference,
                                      err_msg=f"block={block}")


def test_a_short_tail_is_dropped_but_a_short_file_still_gets_a_sample():
    rate = config.ANALYSIS_SR
    # 10.1 s: the 0.1 s tail is too thin to be a level estimate, so 10 samples.
    assert E.envelope_of(np.ones(int(10.1 * rate), np.float32)).size == 10
    # 10.6 s: the tail is over half a second, so it is kept.
    assert E.envelope_of(np.ones(int(10.6 * rate), np.float32)).size == 11
    # Under a second, the tail is all there is -- and a row with an envelope
    # must never be indistinguishable from a row with none.
    assert E.envelope_of(np.ones(rate // 4, np.float32)).size == 1
    assert E.envelope_of(np.zeros(0, np.float32)).size == 0


def test_the_envelope_tracks_loudness():
    rate = config.ANALYSIS_SR
    quiet = np.full(5 * rate, 0.001, np.float32)
    loud = np.full(5 * rate, 0.5, np.float32)
    db = E.dequantize(E.envelope_of(np.concatenate([quiet, loud, quiet])))
    assert db.size == 15
    assert db[:5].max() < db[5:10].min()
    # 0.001 -> -60 dBFS, 0.5 -> -6 dBFS, both well inside the stored range.
    assert db[0] == pytest.approx(-60.0, abs=1.0)
    assert db[7] == pytest.approx(-6.0, abs=1.0)


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def _structured(n: int, seed: int = 3) -> np.ndarray:
    """A plausible dB envelope: quiet gaps between loud stretches."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -55.0)
    pos = 0
    while pos < n:
        run = int(rng.integers(90, 260))
        x[pos:pos + run] = -18.0 + rng.normal(0, 2.5, size=min(run, n - pos))
        pos += run + int(rng.integers(20, 60))
    return (x + rng.normal(0, 0.6, size=n)).astype(np.float32)


def test_alignment_finds_a_known_lag_in_both_directions():
    a = _structured(1200)
    for lag in (0, 1, 37, -37, 200, -200):
        lo = max(0, lag)
        b = np.concatenate([np.full(lo, -55.0, np.float32),
                            a[max(0, -lag):]]).astype(np.float32)
        al = E.align(a, b)
        assert al.ok
        assert al.lag == lag, f"wanted {lag}, got {al.lag}"
        assert al.raw_r > 0.95


def test_alignment_is_gain_invariant():
    """A copy 12 dB down is the same envelope shape, so r must not move."""
    a = _structured(900)
    al = E.align(a, a - 12.0)
    assert al.lag == 0 and al.raw_r == pytest.approx(1.0, abs=1e-6)


def test_unrelated_envelopes_score_far_below_the_pair_thresholds():
    scores = [E.align(_structured(1200, s), _structured(1200, s + 100)).score
              for s in range(20)]
    assert max(scores) < config.PAIR_R_LIKELY, sorted(scores)[-3:]


def test_an_edge_only_overlap_is_penalised_into_irrelevance():
    """The failure mode the relative-overlap penalty exists for.

    Every recording starts quiet, gets loud and ends quiet, so sliding two
    unrelated files until only their edges touch lines up two fades and scores
    almost perfectly on a sliver of the available overlap.  Measured on the
    corpus, that reached raw r = 0.911 between files with nothing whatever in
    common -- higher than some *true* pairs.

    Here the artefact is made exact: the seed's first 300 seconds and the
    candidate's last 300 seconds are literally identical, and nothing else is.
    The raw correlation is therefore 1.0 and must not be believed.
    """
    rng = np.random.default_rng(5)
    shared = rng.normal(-25.0, 6.0, 300)
    a = np.concatenate([shared, rng.normal(-25.0, 6.0, 900)]).astype(
        np.float32)
    b = np.concatenate([rng.normal(-25.0, 6.0, 900), shared]).astype(
        np.float32)

    al = E.align(a, b)
    assert al.ok
    assert al.lag == 900 and al.overlap == 300
    assert al.raw_r == pytest.approx(1.0, abs=1e-6), (
        "fixture is not reproducing the artefact")
    assert al.score < config.PAIR_R_LIKELY, al


def test_short_overlaps_are_excluded_outright():
    a = _structured(1200)
    b = _structured(1200, seed=42)
    # 50% of the shorter file is 600, above the 300 s cap.
    assert E.required_overlap(a.size, b.size) == 300
    al = E.align(a, b)
    assert al.overlap >= 300
    # And the fraction rule wins on shorter files, floored at the minimum
    # envelope length so it can never evaporate on a tiny candidate.
    assert E.required_overlap(1200, 400) == 200
    assert E.required_overlap(1200, 80) == int(
        config.PAIR_MIN_ENVELOPE_SECONDS)


def test_a_seed_shorter_than_the_minimum_is_refused_not_guessed():
    short = _structured(30)
    al = E.align(short, _structured(1200))
    assert not al.ok
    assert "minimum" in al.reason
    assert al.score == 0.0


def test_a_silent_seed_is_refused_not_correlated():
    flat = np.full(600, -120.0, np.float32)
    al = E.align(flat, _structured(1200))
    assert not al.ok and "flat" in al.reason


def test_a_candidate_too_short_to_overlap_is_refused():
    """A five-second clip is not evidence about a 20-minute seed.

    Regression: with the overlap requirement expressed purely as a *fraction*
    of the shorter file, the requirement shrank with the candidate and this
    comparison went ahead over 2.5 seconds of overlap.
    """
    al = E.align(_structured(1200), _structured(5, seed=99))
    assert not al.ok and "too short" in al.reason
    assert E.align(_structured(1200), np.zeros(0, np.float32)).ok is False


def test_different_lengths_are_normalised_over_the_overlap_only():
    """A short excerpt inside a long file must score on its own minutes.

    Naively correlating over the whole array would dilute a perfect 300 s
    match by the 900 s the excerpt says nothing about.
    """
    long = _structured(1200)
    excerpt = long[400:700]
    al = E.align(excerpt, long)
    assert al.ok and al.lag == 400
    assert al.raw_r == pytest.approx(1.0, abs=1e-6)
    assert al.overlap == 300


def test_align_many_matches_align_one_by_one():
    seed = _structured(900)
    cands = [_structured(900, s) for s in range(1, 40)] + [seed]
    batched = E.align_many(seed, cands)
    assert len(batched) == len(cands)
    for c, got in zip(cands, batched):
        want = E.align(seed, c)
        assert got.lag == want.lag
        assert got.score == pytest.approx(want.score, abs=1e-9)


def test_align_many_batches_and_still_agrees(monkeypatch):
    monkeypatch.setattr(config, "PAIR_FFT_BATCH", 3)
    seed = _structured(900)
    cands = [_structured(900, s) for s in range(10)]
    batched = E.align_many(seed, cands)
    for c, got in zip(cands, batched):
        assert got.score == pytest.approx(E.align(seed, c).score, abs=1e-9)


def test_clock_drift_at_the_configured_resolution_is_negligible():
    """Why the envelope is 1 Hz, demonstrated.

    Resample a 45-minute envelope by 200 ppm -- a worse clock mismatch than
    two consumer recorders would realistically show -- and the correlation is
    essentially untouched, because 200 ppm over 2700 s is half a sample.
    """
    n = 2700
    a = _structured(n, seed=17)
    t = np.arange(n) * (1.0 + 200e-6)
    drifted = np.interp(np.arange(n), t, a).astype(np.float32)
    al = E.align(a, drifted)
    assert al.lag == 0
    assert al.raw_r > 0.99, al


# --------------------------------------------------------------------------
# Segment structure (presentation only)
# --------------------------------------------------------------------------


def test_active_segments_finds_the_loud_stretches():
    x = np.full(900, -55.0, np.float32)
    for lo, hi in ((60, 260), (320, 560), (640, 880)):
        x[lo:hi] = -18.0
    runs = E.active_segments(x)
    assert len(runs) == 3
    for (got_lo, got_hi), (lo, hi) in zip(runs, ((60, 260), (320, 560),
                                                 (640, 880))):
        assert abs(got_lo - lo) <= 4 and abs(got_hi - hi) <= 4


def test_a_short_rest_inside_a_tune_is_not_a_track_boundary():
    x = np.full(900, -55.0, np.float32)
    x[100:800] = -18.0
    x[400:404] = -55.0            # four seconds of rest, mid-tune
    assert len(E.active_segments(x)) == 1


def test_featureless_audio_has_no_segments():
    assert E.active_segments(np.full(900, -30.0, np.float32)) == []
    assert E.active_segments(np.zeros(2, np.float32)) == []


def test_segment_comparison_reports_agreement_after_the_lag_is_applied():
    x = np.full(900, -55.0, np.float32)
    for lo, hi in ((60, 260), (320, 560), (640, 880)):
        x[lo:hi] = -18.0
    shifted = np.concatenate([np.full(50, -55.0, np.float32), x[:-50]])
    report = E.compare_segments(x, shifted.astype(np.float32), lag=50)
    assert report.matched == report.total > 0
    assert "within +/-" in report.text

    unrelated = np.full(900, -55.0, np.float32)
    unrelated[10:200] = -18.0
    bad = E.compare_segments(x, unrelated, lag=0)
    assert bad.matched < bad.total
