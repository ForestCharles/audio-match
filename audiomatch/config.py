"""Central tuning constants for audio-match.

Every number that affects on-disk fingerprints lives here.  Changing any value
marked ``FINGERPRINT_AFFECTING`` invalidates existing databases; the schema
version below is bumped when that happens so that stale DBs are detected.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

#: Bumped whenever the meaning of stored fingerprints/signatures changes.
#: A bump here means "everything on disk is now wrong": the only cure is a
#: full re-index, and that is what the error message says.
SCHEMA_VERSION = 1

#: Bumped whenever the *table layout* grows something new that the existing
#: fingerprints remain valid under.  Deliberately separate from
#: ``SCHEMA_VERSION``: adding the activity envelope (v2) did not change a
#: single landmark or session signature, so forcing a full 1.45 TB re-index for
#: it would have been a lie.  Storage upgrades are applied in place by
#: ``Database._migrate`` (an additive ``ALTER TABLE``), and the new column is
#: then filled by ``audio-match backfill``, which decodes only the files that
#: are missing it.
#:
#: v1 -> v2: ``files.envelope``, the 1 Hz activity envelope used by pair mode.
STORAGE_VERSION = 2

#: Default database location.  Documented in the README.
DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".audio-match.db")


# --------------------------------------------------------------------------
# Decoding (FINGERPRINT_AFFECTING)
# --------------------------------------------------------------------------

#: Analysis sample rate.  11025 Hz keeps the whole fingerprint band (<5.5 kHz)
#: well inside what survives 128 kbps MP3, and is an exact 1/4 of 44100 and a
#: clean ratio of 48000, so 44.1k and 48k sources land on identical grids.
ANALYSIS_SR = 11025

#: Number of samples ffmpeg is asked for per read (per channel, both channels).
DECODE_BLOCK = 1 << 18


# --------------------------------------------------------------------------
# STFT / constellation (FINGERPRINT_AFFECTING)
# --------------------------------------------------------------------------

#: FFT size.  1024 @ 11025 Hz -> 92.9 ms window, 10.77 Hz per bin.
NFFT = 1024
#: Hop.  256 @ 11025 Hz -> 23.22 ms per frame, 43.07 frames/second.
HOP = 256

FRAME_RATE = ANALYSIS_SR / HOP          # 43.066 frames per second
FRAME_SECONDS = HOP / ANALYSIS_SR       # 0.02322 s

#: Log-spaced band edges (FFT bin indices) used for peak picking.  Spanning
#: ~43 Hz .. ~5000 Hz in 8 bands.  Picking a fixed number of peaks *per band*
#: (rather than globally) makes the constellation tolerant of EQ/mix changes:
#: a bass-heavy remix cannot starve the treble bands of peaks.
BAND_EDGES = (4, 8, 16, 30, 55, 100, 180, 300, 465)
N_BANDS = len(BAND_EDGES) - 1

#: Peaks per band per second when *indexing* the library.
INDEX_PEAKS_PER_BAND_PER_SEC = 1.0
#: Peaks per band per second when *querying*.  Deliberately denser: the query
#: peak set should be a superset of whatever the library pass chose, so that
#: near-threshold peaks that flipped under transcoding are still covered.
QUERY_PEAKS_PER_BAND_PER_SEC = 4.0

#: A peak must be the maximum of its band within +/- this many frames.
PEAK_TIME_NEIGHBORHOOD = 3

#: Pairing target zone, in frames.
TARGET_ZONE_MIN_DT = 2      # 0.046 s
TARGET_ZONE_MAX_DT = 127    # 2.95 s
#: Maximum absolute frequency distance between the two peaks of a pair, in
#: quantised (8-bit) frequency units.
TARGET_ZONE_MAX_DF = 63

#: Fan-out: how many partners each anchor peak is paired with.
INDEX_FANOUT = 2
#: Query fan-out.  Must comfortably exceed INDEX_FANOUT * (query density /
#: index density) so that the library's chosen pairs are re-emitted.
QUERY_FANOUT = 16

#: Frequency quantisation: FFT bin >> FREQ_SHIFT.  21.5 Hz per unit, 256 units.
FREQ_SHIFT = 1
FREQ_BITS = 8
DT_BITS = 7
#: hash = f1q << 15 | f2q << 7 | dt   -> 23 bits, ~8.4 M buckets.
HASH_BITS = FREQ_BITS * 2 + DT_BITS


# --------------------------------------------------------------------------
# Query / scoring
# --------------------------------------------------------------------------

#: Hashes whose posting list is longer than this are skipped at query time.
#: They carry (almost) no information -- mains hum, tape hiss, room tone --
#: but dominate the cost and the noise floor.
MAX_POSTINGS_PER_HASH = 400

#: Offset histogram votes are summed over a +/- this many frame window, which
#: absorbs the ~10-30 ms encoder delay introduced by MP3/AAC round-trips.
OFFSET_SMOOTH = 1

#: A hit is called "confident" when the aligned vote count clears BOTH an
#: absolute floor and a duration-proportional floor, AND the winning offset
#: bin towers over the same file's next-tallest bin by this factor.  The
#: sharpness test is what separates a real alignment (a spike) from a file
#: that merely shares a lot of common hashes (a flat histogram).
CONFIDENT_MIN_VOTES = 25
CONFIDENT_VOTES_PER_SEED_SECOND = 0.5
CONFIDENT_MIN_SHARPNESS = 4.0

#: Floor applied to the sharpness denominator.  When a file shares exactly one
#: offset bin with the seed there is no second bin to measure, and dividing by
#: 1 reported an unfalsifiable "25x" for a single lonely bin with no evidence
#: behind it.  Treating the background as at least this many votes keeps the
#: ratio meaningful: a file must show >= 4 x 3 = 12 aligned votes before it can
#: clear CONFIDENT_MIN_SHARPNESS on no measured background at all.
SHARPNESS_MIN_BACKGROUND = 3

#: Below this, results are not printed at all.
REPORT_MIN_VOTES = 6

#: Sample-rate mislabel probes: (label, decode_rate).  Only the first entry
#: (the native decode) runs unless ``query --try-rates`` is given.  Decoding at
#: ANALYSIS_SR * r and then *interpreting* the samples as ANALYSIS_SR speeds
#: the seed up by 1/r, which is exactly what happens when a 48 kHz file is
#: played back as if it were 44.1 kHz (and vice versa).
SR_PROBES = (
    ("native", 1.0),
    ("seed-is-48k-labelled-44.1k", 44100.0 / 48000.0),
    ("seed-is-44.1k-labelled-48k", 48000.0 / 44100.0),
)


# --------------------------------------------------------------------------
# Session signature (FINGERPRINT_AFFECTING)
# --------------------------------------------------------------------------

#: Length of each sampled region (seconds) and where they are taken from.
SESSION_REGION_SECONDS = 60.0
SESSION_N_REGIONS = 3  # first / middle / last

#: Noise-floor spectrum.
SESSION_NFFT = 2048
SESSION_HOP = 1024
#: Per-bin percentile used as the noise-floor estimate (minimum statistics).
SESSION_FLOOR_PERCENTILE = 5.0
#: Fraction of frames (quietest) used for the per-channel noise comparison.
SESSION_QUIET_FRACTION = 0.05
#: Number of log-spaced bands in the stored noise-floor vector, and the range
#: they span.  Deliberately restricted to 40-900 Hz: below ~1 kHz the noise
#: floor is room modes, HVAC, traffic rumble, stand coupling and preamp 1/f
#: noise -- all specific to one rig in one room.  Above ~1 kHz it is mostly
#: converter and preamp hiss, which looks near-identical on every recording
#: from the same model of device and so dilutes the signal.  Measured on the
#: DR-40 corpus: restricting to this band raised same-session/other-session
#: separation from +0.41 to +0.65 cosine.
NOISE_BANDS = 48
NOISE_LO_HZ = 40.0
NOISE_HI_HZ = 900.0

#: Hum profile: high-resolution FFT over the quietest contiguous chunk.
HUM_NFFT = 32768  # 0.336 Hz per bin at 11025 Hz
#: Per-bin percentile across chunks.  Hum is stationary and survives it;
#: music is transient and does not.
HUM_PERCENTILE = 20.0
HUM_50_HARMONICS = tuple(50.0 * k for k in range(1, 13))   # 50 .. 600
HUM_60_HARMONICS = tuple(60.0 * k for k in range(1, 11))   # 60 .. 600
HUM_DIM = len(HUM_50_HARMONICS) + len(HUM_60_HARMONICS)
#: Half-width, in Hz, of the local background used as the hum reference.
HUM_BACKGROUND_HZ = 6.0
#: Total harmonic prominence (dB, summed over the harmonic series) below which
#: a file is treated as having no mains hum at all.  Calibrated on the DR-40
#: corpus, which is battery powered: its "hum" readings sum to 23-46 dB of
#: pure measurement noise, so the gate has to sit above that.
HUM_PRESENCE_DB = 45.0

#: Channel statistics vector:
#: [balance_dB, correlation, noisefloor_diff_dB, is_effectively_mono,
#:  dc_offset_left_ppm, dc_offset_right_ppm].
#:
#: The DC offset is the surprise star of this feature set.  A converter's DC
#: bias is a property of the specific input path and gain setting, it is
#: rock-steady for a whole session, and it is unrelated to the performance.
#: On the DR-40 corpus it separates the two rig configurations cleanly
#: (-174 ppm / +30 ppm / -53 ppm clusters).  Caveat: any high-pass filter,
#: normalisation or lossy encode destroys it, so it only helps when comparing
#: original recorder output -- which is exactly the intended use.
CHAN_DIM = 6
#: Tolerance (ppm) for calling two DC offsets "the same".
DC_TOLERANCE_PPM = 15.0

#: Take-number proximity decay: DR-40 take numbers increment through a
#: session, so nearby numbers are weak positive evidence.  e-folding distance.
TAKE_PROXIMITY_DECAY = 6.0
#: Score used when either file has no parseable take number.
TAKE_PROXIMITY_NEUTRAL = 0.40

#: Session-similarity component weights (must sum to 1.0).
W_NOISE = 0.35
W_HUM = 0.10
W_CHAN = 0.30
W_CONTAINER = 0.25


# --------------------------------------------------------------------------
# Activity envelope (mode 3 / pair matching)
# --------------------------------------------------------------------------

#: Envelope resolution: one value per second.
#:
#: This number is doing more work than it looks.  Two recorders capturing the
#: same performance have independent crystals, so their sample clocks drift
#: apart by tens of ppm.  At 200 ppm -- already a bad pair of consumer
#: recorders -- a 45-minute take drifts by 200e-6 * 2700 s = 0.54 s, i.e. *half
#: an envelope sample*.  So at 1 Hz a cross-correlation of two unsynchronised
#: captures still lines up as a single sharp peak, with no drift model at all.
#: Push the envelope to 10 Hz and the same drift smears the peak over five
#: samples and the score collapses.  1 Hz is the resolution at which clock
#: drift stops mattering, which is exactly what pair matching needs.
#:
#: The cost is storage-trivial: one byte per second is 2.6 KB for a 45-minute
#: file, ~9 MB for a 2500-hour library.
ENVELOPE_HZ = 1.0

#: Quantisation of the stored envelope.  Each second holds ``10*log10`` of the
#: mean square of the mono downmix -- i.e. dBFS RMS -- linearly quantised into
#: one unsigned byte over ``[ENVELOPE_DB_FLOOR, ENVELOPE_DB_CEIL]``, giving
#: 120/255 = 0.47 dB per step.
#:
#: Why log-quantised bytes rather than float16 linear amplitude: the score is a
#: correlation of *log* envelopes, so dB is the domain the numbers are actually
#: compared in, and quantising there spends the bits evenly across the whole
#: dynamic range.  0.47 dB of quantisation noise against the tens of dB that
#: separate a loud passage from a gap between songs is nothing.  The floor is
#: -120 dBFS so that even the pathologically quiet recovered files in this
#: corpus (one is ~50 dB below normal level) sit well inside the range rather
#: than clipping to the bottom of it.
ENVELOPE_DB_FLOOR = -120.0
ENVELOPE_DB_CEIL = 0.0
ENVELOPE_LEVELS = 256


# --------------------------------------------------------------------------
# Pair matching (mode 3)
# --------------------------------------------------------------------------

#: A seed shorter than this has too few envelope samples to align honestly.
#: At 1 Hz a 30-second seed is 30 numbers; correlated against a few hundred
#: candidate lags, r > 0.8 happens by chance routinely.  Pair mode refuses
#: rather than reporting a number it does not believe.
PAIR_MIN_ENVELOPE_SECONDS = 60.0

#: Required overlap between seed and candidate:
#: ``max(PAIR_MIN_ENVELOPE_SECONDS, min(300 s, 50% of the shorter file))``.
#: Lags that overlap by less than this are not considered at all -- two files
#: that share ten seconds at the very edge are not evidence of anything,
#: however well those ten seconds correlate.
#:
#: The outer ``max`` matters: without it, "50% of the shorter file" made the
#: requirement vanish exactly when it was most needed, so a five-second
#: candidate was compared to a 45-minute seed over a 2.5-second overlap.  An
#: envelope comparison needs a minute of shared timeline to mean anything,
#: whoever is the shorter party, which is the same number that gates the seed.
PAIR_MIN_OVERLAP_SECONDS = 300.0
PAIR_MIN_OVERLAP_FRACTION = 0.5

#: Two soft penalties on top of that hard minimum.  Between them they are the
#: single most important tuning decision in pair mode, so here is the
#: measurement (all 9 recovered corpus files, whole, all 72 ordered pairs):
#:
#:     statistic                       true pairs      unrelated pairs
#:     raw best-lag Pearson r          0.913 .. 0.948  up to 0.911  (!)
#:     r * sqrt(L / L_max)             0.913 .. 0.948  up to 0.602
#:
#: The raw correlation does not separate at all.  The reason is a specific and
#: entirely predictable artefact: *every* recording starts quiet, gets loud and
#: ends quiet, so sliding two unrelated files until only their edges touch
#: lines up two fades and scores 0.9 on 300 seconds of overlap out of the 2400
#: available.  Penalising the overlap *relative to the maximum overlap this
#: pair could have had* kills that dead -- and costs a true pair nothing,
#: because a true pair's best lag is the one that uses all the overlap there
#: is.  ``L_max`` is ``min(len(seed), len(candidate))``.
#:
#: The exponent is not delicate: 0.25, 0.35 and 0.5 all produce byte-identical
#: rankings on the corpus, because the penalty's job is to move the argmax off
#: the edge, after which the factor is 1.  0.5 is the plain square root.
PAIR_OVERLAP_EXPONENT = 0.5

#: The second penalty is absolute rather than relative, and guards the other
#: end: a *short seed* against a long library file overlaps fully at hundreds
#: of different lags, so the relative term is 1 everywhere and cannot help.
#: ``sqrt(L / (L + PAIR_OVERLAP_SHRINK_SECONDS))`` discounts small samples:
#: 0.71 at one minute of overlap, 0.91 at five, 0.99 at forty-five.  The
#: envelope's own autocorrelation time is song-scale (tens of seconds), so a
#: one-minute overlap really is only a handful of independent observations and
#: really does deserve to be halved.
PAIR_OVERLAP_SHRINK_SECONDS = 60.0

#: Envelope alignment is run in batches of this many candidates through one
#: 2-D FFT, rather than one transform per candidate.
PAIR_FFT_BATCH = 256

#: How many top envelope candidates get the (much more expensive)
#: constellation-coherence pass.
PAIR_COHERENCE_CANDIDATES = 20

#: Coherence is evaluated only within this many seconds of the lag the
#: envelope proposed.  Coherence is *confirming* evidence for that specific
#: alignment, not an independent search, and restricting the window keeps the
#: drift fit cheap enough to run on a 45-minute seed.
PAIR_COHERENCE_WINDOW_SECONDS = 30.0

#: Clock-drift search.  A drifting capture puts its matched landmark pairs on a
#: *line* in (t_seed, t_lib) rather than in one offset bin, so the offset
#: histogram is recomputed with the seed timestamps pre-compensated by each
#: candidate slope and the sharpest histogram wins.
#:
#: The grid is adaptive.  Two slopes are only distinguishable if they disagree
#: by more than one STFT frame (23 ms) across the seed, so the step is
#: ``max(PAIR_DRIFT_MIN_STEP_PPM, 2 frames worth)``: a 150-second seed gets
#: only the zero slope (correctly -- 300 ppm over 150 s is 45 ms, two frames,
#: nothing to fit), while a 45-minute seed gets the full 25 ppm grid.  The grid
#: is capped at ``PAIR_MAX_DRIFT_SLOPES`` entries so the cost is bounded.
PAIR_MAX_DRIFT_PPM = 300.0
PAIR_DRIFT_MIN_STEP_PPM = 25.0
PAIR_MAX_DRIFT_SLOPES = 25

#: Coherence verdicts.  The bars are the same shape as mode 1's (an absolute
#: vote floor plus a sharpness floor) but lower, because a pair-mate is the
#: same performance through *different microphones*: most landmarks differ and
#: only broadband transients heard by both capsules survive.  Measured on the
#: corpus, a real S12/S34 pair over a 45-second excerpt scores ~19 votes at
#: 3.2x sharpness at exactly the right offset, where an unrelated file scores
#: nothing at all -- so 'weak' has to start below that.
PAIR_COHERENCE_STRONG_VOTES = 30
PAIR_COHERENCE_STRONG_SHARPNESS = 4.0
PAIR_COHERENCE_WEAK_VOTES = 8
PAIR_COHERENCE_WEAK_SHARPNESS = 2.0

#: Verdict thresholds on the penalised envelope correlation.  Measured on the
#: whole recovered corpus (see ``PAIR_OVERLAP_EXPONENT`` for the method):
#:
#:     true S12/S34 pairs      0.887 .. 0.937   (n=8)
#:     unrelated pairs         0.402 .. 0.585   (n=48, p90 0.54)
#:
#: The unrelated set is the *hardest* available negative class -- the same
#: band, the same rig, the same room, sets of similar length and shape on
#: different days -- so 0.585 is a realistic ceiling for "no relationship",
#: not a soft one.  LIKELY PAIR at 0.65 clears it with room; PAIR at 0.80 sits
#: in the empty middle of a 0.30-wide gap.
#:
#: A degraded different-recorder capture lands between the two, which is
#: exactly what LIKELY PAIR is for: the simulated second rig in the test suite
#: (100 ppm resample, EQ tilt, echo, -6 dB, 128k mp3) still scores ~0.9 on the
#: envelope while sharing almost no landmarks at all.
PAIR_R_STRONG = 0.80
PAIR_R_LIKELY = 0.65

#: Segment display only -- never scoring.  The aligned envelopes are smoothed
#: over ``SMOOTH`` seconds and thresholded at ``FRACTION`` of the way from the
#: 10th to the 90th percentile of the file's own envelope; runs shorter than
#: ``MIN`` seconds and gaps shorter than ``GAP`` seconds are absorbed, so a
#: bar of rest inside a tune does not read as two tracks.  Boundaries that land
#: within ``TOLERANCE`` seconds of each other after alignment are called
#: agreeing.
PAIR_SEGMENT_SMOOTH_SECONDS = 5.0
PAIR_SEGMENT_THRESHOLD_FRACTION = 0.35
PAIR_SEGMENT_MIN_SECONDS = 20.0
PAIR_SEGMENT_GAP_SECONDS = 8.0
PAIR_SEGMENT_TOLERANCE_SECONDS = 4.0


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

AUDIO_EXTENSIONS = frozenset(
    """
    .wav .wave .bwf .w64 .rf64 .aif .aiff .aifc .flac .mp3 .m4a .mp4 .aac
    .ogg .oga .opus .wma .caf .au .snd .ape .wv .mpc .mka .dsf .amr .3gp
    """.split()
)

#: Files smaller than this are ignored (an empty/stub WAV cannot be matched).
MIN_FILE_BYTES = 4096
#: Files whose decoded duration is shorter than this get a signature but no
#: useful fingerprint; they are still indexed and flagged.
MIN_USEFUL_SECONDS = 1.0

#: Number of finished files buffered before a COMMIT.  A crash loses at most
#: this many files' work (plus whatever is in flight in the workers).
COMMIT_EVERY = 20
