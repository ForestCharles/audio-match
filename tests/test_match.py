"""Mode 1: does the constellation actually find the same audio?"""

from __future__ import annotations

import os

import pytest

from audiomatch import config
from audiomatch.db import open_db
from audiomatch.query import match_search

from conftest import Corpus, cut, requires_corpus

pytestmark = requires_corpus


def _search(db_path: str, seed: str, **kw):
    with open_db(db_path, create=False) as db:
        return match_search(db, seed, top=10, **kw)


@pytest.fixture(scope="module")
def hostile_seed(corpus: Corpus) -> str:
    """The full obstacle course, all at once.

    30 s from the middle of an indexed 48 kHz excerpt, resampled to 44.1 kHz,
    attenuated 8 dB and squeezed through 128 kbps MP3.
    """
    return cut(corpus.lib_file("0077", "S12"),
               corpus.seed("hostile_30s.mp3"), 60.0, 30.0,
               extra=["-ar", "44100", "-af", "volume=-8dB",
                      "-c:a", "libmp3lame", "-b:a", "128k"])


def test_transcoded_excerpt_ranks_first_with_a_decisive_margin(
        indexed_db, corpus, hostile_seed):
    hits, seed_seconds, _counts, _sig = _search(indexed_db, hostile_seed)

    assert hits, "no candidates at all for a seed cut from the library"
    best = hits[0]
    assert os.path.basename(best.path) == "TASCAM_0077S12.wav"
    assert best.is_confident(seed_seconds)

    # Decisive: the winner must tower over the runner-up, and over its own
    # file's second-tallest offset bin.
    runner_up = hits[1].votes if len(hits) > 1 else 0
    assert best.votes >= 10 * max(1, runner_up), (
        f"margin too thin: {best.votes} vs {runner_up}")
    assert best.sharpness >= 10.0


def test_reported_offset_is_approximately_correct(
        indexed_db, corpus, hostile_seed):
    """The seed was cut at 60 s into the library excerpt; say so."""
    hits, _s, _c, _sig = _search(indexed_db, hostile_seed)
    assert abs(hits[0].offset_seconds - 60.0) < 0.5
    # And it should know roughly how much of the seed lined up.
    assert hits[0].matched_seconds > 20.0


def test_audio_absent_from_the_library_returns_no_confident_match(
        indexed_db, corpus):
    """A different part of an indexed *file* is still absent audio.

    The library holds 0077S12 from 600-750 s; this seed comes from 1500 s, so
    every landmark it shares with the index is a coincidence.
    """
    seed = cut(corpus.source("0077", "S12"), corpus.seed("absent_30s.wav"),
               1500.0, 30.0)
    hits, seed_seconds, _c, _sig = _search(indexed_db, seed)
    for hit in hits:
        assert not hit.is_confident(seed_seconds), (
            f"false positive: {hit.path} scored {hit.votes} "
            f"(sharpness {hit.sharpness:.1f}x)")
    if hits:
        threshold = max(config.CONFIDENT_MIN_VOTES,
                        config.CONFIDENT_VOTES_PER_SEED_SECOND * seed_seconds)
        assert hits[0].votes < threshold


def test_sample_rate_mislabelled_file_is_found_and_flagged(
        indexed_db, corpus):
    """A 48 kHz recording carrying a 44.1 kHz header.

    Reproduced by decoding to raw PCM and re-wrapping it with the wrong rate,
    which is exactly what a damaged or hand-patched WAV header does.
    """
    import subprocess

    src = corpus.lib_file("0072", "S12")          # genuine 48 kHz
    dst = corpus.seed("mislabelled_44k.wav")
    if not os.path.exists(dst):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "30", "-t", "40", "-i", src,
             "-f", "s24le", "-ac", "2", "-"],
            capture_output=True, check=True).stdout
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "s24le", "-ar", "44100",
             "-ac", "2", "-i", "-", dst],
            input=raw, capture_output=True, check=True)

    hits, seed_seconds, _c, _sig = _search(indexed_db, dst)
    best = hits[0]
    assert os.path.basename(best.path) == "TASCAM_0072S12.wav"
    assert best.is_confident(seed_seconds)
    assert best.probe != "native", (
        "a mislabelled file matched on the native probe, so the mislabel "
        "detection never fired")
    assert best.probe == "seed-is-48k-labelled-44.1k"
    assert abs(best.ratio - 44100.0 / 48000.0) < 1e-6
    # Offset is reported on the library timeline, so it is the true 30 s.
    assert abs(best.offset_seconds - 30.0) < 0.5


def test_sr_probes_can_be_disabled(indexed_db, corpus, hostile_seed):
    hits, _s, counts, _sig = _search(indexed_db, hostile_seed,
                                     probes=(config.SR_PROBES[0],))
    assert list(counts) == ["native"]
    assert os.path.basename(hits[0].path) == "TASCAM_0077S12.wav"


def test_s12_vs_s34_cross_microphone_behaviour(indexed_db, corpus, capsys):
    """DOCUMENTS behaviour, does not gate it.

    S12 and S34 of one take are the same performance captured through
    different microphones: same timing, different spectrum.  The constellation
    is spectral, so most landmarks differ -- but the ones that survive (sharp
    broadband transients heard by both mic pairs) all agree on the same time
    offset.  The expected signature is therefore a *low score at a correct and
    reasonably sharp offset*, which is qualitatively different from noise.
    """
    seed = cut(corpus.source("0048", "S12"), corpus.seed("cross_0048S12.wav"),
               corpus.start("0048") + 60.0, 45.0)
    hits, seed_seconds, _c, _sig = _search(indexed_db, seed)
    by_name = {os.path.basename(h.path): h for h in hits}

    same = by_name.get("TASCAM_0048S12.wav")
    assert same is not None and same.is_confident(seed_seconds), \
        "sanity: the seed's own file must still match"

    mate = by_name.get("TASCAM_0048S34.wav")
    report = ["", "S12 -> S34 cross-microphone behaviour:",
              f"  same file (S12): {same.votes} votes, "
              f"{same.sharpness:.1f}x sharpness, "
              f"offset {same.offset_seconds:.2f}s"]
    if mate is None:
        report.append("  pair-mate (S34): no bin above the reporting floor")
    else:
        report.append(
            f"  pair-mate (S34): {mate.votes} votes, "
            f"{mate.sharpness:.1f}x sharpness, "
            f"offset {mate.offset_seconds:.2f}s "
            f"(delta from S12 offset: "
            f"{mate.offset_seconds - same.offset_seconds:+.2f}s)")
        report.append(
            f"  confident by the tool's own threshold: "
            f"{mate.is_confident(seed_seconds)}")
    print("\n".join(report))
    with capsys.disabled():
        print("\n".join(report))
