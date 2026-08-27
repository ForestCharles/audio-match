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
SCHEMA_VERSION = 1

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
