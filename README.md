# audio-match

Find, in a very large library of audio files, both:

1. **The same audio** — other transfers, mixes, excerpts or lossy encodes of
   the same performance, even at a different gain, sample rate or bitrate.
2. **The same recording session** — files recorded on the same rig, in the same
   room, on the same day. Different music, so content matching cannot find
   them; this mode works from the noise floor, the mains hum, the converter's
   DC bias and the channel behaviour instead.

Built for a ~1.45 TB library of Tascam DR-40 WAVs on a Linux server. One
indexing pass, one SQLite file, two query modes.

---

## Install

```bash
sudo apt install ffmpeg          # or your distro's equivalent
pip install numpy                # the only Python dependency
git clone <this repo> && cd audio-match
pip install -e .                 # gives you the `audio-match` command
```

You can skip the install entirely and run `python3 -m audiomatch ...` from the
checkout. Requires Python 3.10+, numpy, and the `ffmpeg` **and** `ffprobe`
binaries on `PATH`. No scipy, no compiled extensions, no network access.

All decoding is done by shelling out to ffmpeg and reading raw `f32le` from its
stdout. That is deliberate: this library is full of carved, recovered and
hand-patched WAV files whose headers lie about their length and sometimes about
their sample rate. ffmpeg copes; a pure-Python WAV reader does not.

---

## Quick start

```bash
# One-time pass over the library (hours -- see "Indexing 1.45 TB" below).
audio-match index /mnt/library

# Who else has this performance?  And what else came from this session?
audio-match query ~/seed.wav
```

The database defaults to **`~/.audio-match.db`**. Override it with a global
`--db PATH`, which must come *before* the subcommand:

```bash
audio-match --db /var/lib/audio-match.db index /mnt/library --workers 24
audio-match --db /var/lib/audio-match.db query ~/seed.wav --mode match --top 20
```

### Commands

| Command | What it does |
| --- | --- |
| `index <dir>` | Scan and fingerprint a library. Resumable. |
| `query <file>` | Search with a seed file. `--mode match\|session\|both` (default `both`). |
| `stats` | Index size, hours of audio, landmark count, bytes per hour. |
| `errors` | List every file that failed to decode, with the reason. |
| `purge` | Reclaim space left by re-indexed (superseded) files. |

Useful flags:

* `index --workers N` — worker processes, default = `os.cpu_count()`.
* `index --all-files` — try every file, not just the audio extension allowlist.
* `index --force` — re-index everything, ignoring the resume stamps.
* `index --retry-errors` — retry files that previously failed to decode.
* `query --top N` — how many results per mode (default 10).
* `query --no-sr-probes` — skip the 44.1/48 kHz mislabel probes (3x faster).
* `query --ignore-filenames` — score mode 2 on audio evidence only.

---

## Mode 1 — same audio

A constellation ("Shazam-style") fingerprint.

**How it works.** Every file is decoded to mono at 11025 Hz and run through a
1024-point STFT with a 256-sample hop (92.9 ms window, 23.22 ms per frame,
10.77 Hz per bin). The spectrum is split into 8 log-spaced bands, and in each
band the tool keeps the strongest local maximum per one-second block. Each peak
is paired with the next few peaks inside a target zone, and each pair is packed
into a 23-bit integer hash of `(freq1, freq2, Δt)`.

At query time the seed produces the same kind of hashes. Every hash the index
recognises casts one vote for `(library_file, t_library − t_seed)`. If the seed
really is inside a library file, *every* matching landmark is displaced by the
same constant, so the votes pile into a single time-offset bin. Coincidental
hash collisions scatter across every bin instead. So the shape of the histogram,
not the number of shared hashes, is what tells you it is a match.

**Reading the output.**

```
 1. [MATCH  ] score    372   sharpness   53.1x   concentration  16.2%
      /mnt/library/2019/TASCAM_0077S12.wav
      seed 0:00 lands at 1:00.0 in this file (length 2:30.0); ~29s of seed aligned
```

* **score** — votes in the winning offset bin. Grows with seed length and with
  how much of the seed is really present.
* **sharpness** — that bin divided by the same file's *next-tallest* bin. This
  is the number to trust. On real data a true match runs **40–100x**, while
  unrelated files sit at **1.0–1.9x**. The gap is enormous and does not depend
  on how long your seed is.
* **concentration** — the winning bin as a fraction of that file's total votes.
  Informative but noisy on small indexes; sharpness supersedes it.
* **seed 0:00 lands at …** — where in the library file your seed begins. A
  negative value means the seed starts *before* the library file does (the
  library file is the excerpt, not the seed).

A result is labelled `[MATCH  ]` when it clears **both**:

* `score ≥ max(25, 0.5 × seed_seconds)` votes, and
* `sharpness ≥ 4.0x`.

Anything else prints as `[weak   ]`. If nothing clears the bar, the tool says
so explicitly rather than handing you a best-of-the-noise answer.

**What it survives.** Verified against the real corpus (see `tests/`):
44.1 ↔ 48 kHz resampling, ±10 dB gain changes, 128 and 320 kbps MP3, and a 30 s
excerpt taken from the middle of a longer file. The worst case — all of those
at once, a 30 s 128 kbps MP3 at 44.1 kHz and −8 dB cut from a 48 kHz original —
scores 372 against a noise floor of 10, i.e. a 37x margin.

### Sample-rate mislabelling

Some of this library's files were recovered without a trustworthy header, so
their true sample rate is uncertain. A 48 kHz recording carrying a 44.1 kHz
header plays 8.8% slow, and every frequency in it shifts by a factor of 0.919 —
which destroys a constellation match outright.

So every query is run three times: once natively, once with the seed decoded at
`11025 × 44100/48000` Hz and once at `11025 × 48000/44100` Hz, then reinterpreted
as 11025 Hz. That is a pure ffmpeg `-ar` change, so it costs a re-decode of the
seed and nothing else. A hit found on one of the shifted probes is reported as:

```
      *** MATCH AT WRONG SAMPLE RATE *** seed-is-48k-labelled-44.1k: the seed
      had to be resampled by 1.0884x to line up, i.e. the two files disagree
      about the true sample rate of this recording.
```

The reported time offset is always on the **library file's** timeline.

Use `--no-sr-probes` if you know your seed's rate is right and want the query
three times faster.

### Interpreting S12 / S34 dual-record pairs

In the DR-40's 4-channel mode, `TASCAM_0077S12.wav` and `TASCAM_0077S34.wav`
are the *same performance through different microphones*, recorded
simultaneously. Their spectra differ, so most landmarks differ — but the sharp
broadband transients that both mic pairs heard produce landmarks that survive,
and those all agree on the same time offset.

The behaviour you should expect, measured on `TASCAM_0048`:

| Seed | Match against | Score | Sharpness | Offset |
| --- | --- | ---: | ---: | ---: |
| 45 s of `0048S12` | `0048S12` (itself) | 593 | 98.8x | 60.00 s (correct) |
| 45 s of `0048S12` | `0048S34` (pair-mate) | 19 | 3.2x | 60.00 s (correct) |
| 45 s of `0048S12` | unrelated files | 6–10 | 1.0–1.2x | random |

So the pair-mate lands *between* a true match and the noise: too low to be
called a confident match from a 45 s excerpt, but with a visibly sharper and
demonstrably **correct** offset. With a longer seed it clears the bar outright —
a full 2.5-minute `0072S12` seed matches its `0072S34` mate at 130 votes and
8.1x sharpness, flagged `[MATCH  ]`.

Practical reading: **a weak hit at 2–4x sharpness whose offset is a round number
(often exactly 0:00) and whose filename differs only in `S12`/`S34` is almost
certainly the pair-mate, not a coincidence.** Mode 2 annotates these explicitly.

---

## Mode 2 — same session

This mode is **heuristic**. It ranks candidates for you to audition. It does not
prove that two files came from the same session, and it can be wrong. Read the
component columns, not just the total.

Four components, computed once per file at index time and stored as a few
hundred bytes:

**(a) Noise-floor spectrum** *(weight 0.35)* — the per-frequency-bin 5th
percentile of the magnitude spectrogram, sampled from the first, middle and last
60 seconds of the file. Taking a low percentile *per bin* (minimum statistics)
rather than averaging whole quiet frames matters: in 150 seconds of continuous
playing there are no silent frames, so the naive estimator just measures music.
The vector covers **40–900 Hz** in 48 log-spaced bands, mean-removed and
L2-normalised so it is gain-invariant. The band limit is deliberate — below
1 kHz the floor is room modes, HVAC, traffic rumble, stand coupling and 1/f
preamp noise, all specific to one rig in one room; above 1 kHz it is generic
converter hiss that looks the same on every recording from that model of
device. On this corpus, restricting to 40–900 Hz raised same-session vs
other-session separation from +0.41 to +0.65 cosine.

**(b) Mains hum profile** *(weight 0.10)* — a 32768-point FFT (0.34 Hz per bin)
of the same regions, again taking a low percentile per bin across many chunks so
that the stationary hum line survives and the transient music does not. Reports
the dB prominence of 50 Hz and 60 Hz harmonics up to 600 Hz over their local
spectral background. Gated: below 45 dB of total harmonic prominence the file is
treated as having no hum at all, and the component returns a neutral 0.5. The
DR-40 corpus is battery-powered and lands below that gate — ungated, this
component was actively misranking results. It will earn its weight on
mains-powered material.

**(c) Channel statistics** *(weight 0.30)* — L/R RMS balance in dB, inter-channel
correlation, the per-channel noise-floor difference (this is what catches the
session with a dead left input), a mono/dead-channel flag, and **the per-channel
DC offset in ppm**. The DC offset is the surprise star of this feature set: a
converter's DC bias is a property of one specific input path at one gain
setting, it is rock-steady for a whole session, and it has nothing to do with
the performance. On this corpus it separates the rig configurations cleanly
(−174 / +30 / −53 ppm clusters). Caveat: any high-pass filter, normalisation or
lossy encode destroys it, so it only helps when comparing original recorder
output — which is the intended use.

**(d) Container and filename facts** *(weight 0.25)* — sample rate, channel count,
bit depth, and Tascam take-number proximity parsed from the filename (a DR-40
increments its take number through a session, so nearby numbers are real, if
weak, evidence). Pass `--ignore-filenames` to drop the filename half and score
on audio evidence alone.

**Reading the output.**

```
  rank  total  noise  hum   chan  cont.  file
    1.  0.913  0.968 0.500 0.915 1.000  /mnt/library/TASCAM_0077S12.wav
        #########.  48000 Hz/2ch/24-bit  2:30.0
        - same converter DC bias (-52 ppm)
        - same take number (0077)
```

There is no threshold and no "confident" label here, on purpose — the scores are
only meaningful *relative to each other within one query*. Look for a step down
in `total` between the plausible group and the rest, and check whether `noise`
and `chan` (the physical evidence) agree with `cont.` (the bookkeeping
evidence). A high total driven entirely by `cont.` means "same file format and a
nearby take number", which is much weaker than it looks.

Results that are the seed's plausible dual-record pair-mate — same take number,
the other `Sxx` suffix — are called out as
`LIKELY DUAL-RECORD PAIR-MATE OF THE SEED`.

**Measured performance.** Seeding with a held-out 90-second excerpt from a part
of the recording that is *not* in the index, over a library holding two files
from each of four sessions:

| Seed | Rank of own-session files (of 8) | Top hit |
| --- | --- | --- |
| `TASCAM_0077S12` @ 30:00 | 1, 2 | correct |
| `TASCAM_0048S34` @ 20:00 | 1, 4 | correct |

The `0048` seed places two `pakDR40` files at ranks 2 and 3, above its own
session's `S12`. That is an honest result, not a bug: those two sessions share a
converter DC bias and a very similar low-frequency noise floor, so they may well
be the same rig in the same room on different days. The tool is telling you
something true and leaving the judgement to you.

---

## Indexing 1.45 TB

**One pass, and the disk is the bottleneck.** Each file is read exactly once;
the constellation and the session signature are computed from the same streaming
decode. Decoding is streamed in blocks, so worker memory stays flat (tens of MB)
no matter how long the file is — a 1 GB WAV costs the same RAM as a 10 MB one.

**Expected wall clock.** Measured on this development VM: **45 MB/s with 4
workers**, i.e. ~11 MB/s per worker, which is about 38x realtime for 24-bit
48 kHz stereo. So:

```
wall clock  ≈  1.45 TB / min(disk sequential MB/s, workers × 11 MB/s)
```

* 4 workers, any disk: ~9 hours (CPU-bound).
* 16+ workers on a disk that streams 200 MB/s: ~2 hours (disk-bound).
* 16+ workers on a disk that streams 100 MB/s: ~4 hours.

Past about 16–20 workers you are almost certainly waiting on the disk. Start
with `--workers $(nproc)` and watch whether the reported MB/s plateaus.

**Expected database size.** Measured **~1.0 MB of database per hour of audio**
(16 landmarks/second, ~17 bytes per landmark including index overhead):

| Library | Database |
| --- | --- |
| 1.45 TB of 24-bit/48 kHz stereo (≈ 1 400 h) | **~1.4 GB** |
| 2 500 h of mixed formats | **~2.4 GB** |

That budget is why the landmark table is a `WITHOUT ROWID` table whose primary
key *is* the whole row: SQLite then stores one B-tree instead of a table plus a
covering index, roughly halving bytes per landmark.

**Resumability.** The index is keyed on `(path, size, mtime)`.

* Rerunning skips every file whose size and mtime are unchanged.
* A file that changed is re-indexed automatically.
* Results are committed in batches of 20, so a crash or a `Ctrl-C` loses at most
  that batch plus whatever the workers had in flight. Just run the same command
  again.
* Files that fail to decode are recorded with their error and *not* retried on
  the next run (they would fail again). `audio-match errors` lists them;
  `index --retry-errors` retries them.

**Superseded files and `purge`.** There is deliberately no index on
`hashes.file_id` — adding one would roughly double the database, and it would
only ever be used for deletion. Instead, re-indexing a changed file marks the old
record dead and writes a new one with a fresh id. Queries skip dead ids, so
results stay correct, but the old landmarks still occupy space. `audio-match
index` tells you when this has happened; `audio-match purge` reclaims it in a
single sequential rewrite. You will rarely need it.

**Progress output** goes to stderr and shows files done/total, bytes done/total,
throughput, elapsed time, ETA and a running error count. Piped output degrades
to one line every 50 files instead of a redrawing status line.

---

## Limitations

* **Mode 2 is a ranker, not a classifier.** There is no threshold above which
  two files are "the same session". Two different sessions on the same rig in
  the same room will score high, correctly and unhelpfully. Audition the top
  results.
* **Mode 2 wants original recorder output.** The DC-offset and noise-floor
  evidence is destroyed by high-pass filtering, normalisation, mastering or
  lossy encoding. Comparing a mastered MP3 to a raw WAV will mostly measure the
  processing, not the session.
* **Mode 1 needs the audio to be genuinely the same performance.** A different
  take of the same song will not match, and should not — that is what mode 2 is
  for.
* **Mode 1 needs a few seconds.** Below about 10 seconds of seed the vote counts
  get thin and the `max(25, …)` floor starts rejecting real matches. 30 seconds
  is comfortable.
* **Heavy time-stretching is not handled.** The two 44.1/48 kHz probes are the
  only speed variations tested. A genuine tempo change, a varispeed transfer or
  a pitch-shift will not match.
* **Silence and near-silence produce no landmarks.** A file of room tone cannot
  be matched by mode 1 at all (mode 2 will still rank it).
* **The peak-density budget is a real trade-off.** 16 landmarks/second is sparse
  by the standards of commercial fingerprinting, chosen so that 2 500 hours fits
  in ~2.4 GB. Queries compensate by fingerprinting the *seed* four times more
  densely, which costs only query CPU. If you have disk to spare and want more
  sensitivity for very short excerpts, raise
  `INDEX_PEAKS_PER_BAND_PER_SEC` in `audiomatch/config.py` and re-index —
  database size scales with it roughly linearly.
* **Changing anything in `config.py` marked `FINGERPRINT_AFFECTING` invalidates
  the database.** Bump `SCHEMA_VERSION` when you do; existing databases will
  then refuse to open with a message telling you to re-index, rather than
  silently returning garbage.

---

## Layout

| File | Contents |
| --- | --- |
| `audiomatch/config.py` | Every tuning constant, with the reasoning behind it. |
| `audiomatch/audio.py` | ffmpeg/ffprobe subprocess decoding, streaming and one-shot. |
| `audiomatch/fingerprint.py` | Streaming STFT, peak picking, landmark hashing. |
| `audiomatch/session.py` | Noise floor, hum, channel stats, filename parsing, similarity. |
| `audiomatch/analyze.py` | One decode per file, feeding both fingerprints. |
| `audiomatch/db.py` | SQLite schema, writes, lookups, purge. |
| `audiomatch/indexer.py` | Directory walk, resume planning, worker pool, progress. |
| `audiomatch/query.py` | Offset histograms, scoring, session ranking. |
| `audiomatch/cli.py` | Argument parsing and all human-readable output. |

### Schema

```sql
CREATE TABLE files (
    id INTEGER PRIMARY KEY, path TEXT, alive INTEGER,
    size INTEGER, mtime REAL, status TEXT, error TEXT,
    duration REAL, sample_rate INTEGER, channels INTEGER,
    bits INTEGER, codec TEXT, take INTEGER, role TEXT,
    n_hashes INTEGER,
    noise BLOB, hum BLOB, chan BLOB,     -- float32 session signature
    indexed_at REAL
);
CREATE UNIQUE INDEX ix_files_path ON files(path) WHERE alive = 1;

CREATE TABLE hashes (
    hash INTEGER, file_id INTEGER, t INTEGER,
    PRIMARY KEY (hash, file_id, t)
) WITHOUT ROWID;
```

---

## Tests

```bash
pip install pytest
python3 -m pytest tests/          # ~90 seconds
```

The suite runs against the **real** recovered DR-40 corpus in
`/mnt/host/projects/audio-recovery/recovered/`, cutting short excerpts with
ffmpeg into a scratch directory (it never loads a whole 1 GB file). If that
corpus is not present those tests skip and the pure unit tests still run.
