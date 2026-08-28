# audio-match

Find, in a very large library of audio files:

1. **The same audio** — other transfers, mixes, excerpts or lossy encodes of
   the same performance, even at a different gain, sample rate or bitrate.
2. **The same recording session** — files recorded on the same rig, in the same
   room, on the same day. Different music, so content matching cannot find
   them; this mode works from the noise floor, the mains hum, the converter's
   DC bias and the channel behaviour instead.
3. **Pair mates** — other files that captured *the same stretch of time*,
   including captures made on entirely different equipment: another recorder,
   other microphones, an unsynchronised clock and a different start time. This
   mode works from the loudness envelope of the performance, which every
   microphone in the room heard the same way.

Built for a ~1.45 TB library of Tascam DR-40 WAVs on a Linux server. One
indexing pass, one SQLite file, three query modes.

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

# What else was recording while this was recorded?
audio-match query ~/seed.wav --mode pair
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
| `index <dir>` | Scan and fingerprint a library. Resumable; prunes files that have vanished from the root. |
| `query <file>` | Search with a seed file. `--mode match\|session\|both\|pair` (default `both`). |
| `backfill` | Compute the activity envelope for files indexed before mode 3 existed. Needed **once** per pre-existing database; decodes only those files and rewrites no fingerprints. |
| `stats` | Index size, hours of audio, landmark count, bytes per hour. |
| `errors` | List every file that failed to decode, with the reason. |
| `purge` | Reclaim space left by re-indexed (superseded) files. |

Useful flags:

* `index --workers N` — worker processes, default = `os.cpu_count()`.
* `index --all-files` — try every file, not just the audio extension allowlist.
* `index --force` — re-index everything, ignoring the resume stamps.
* `index --retry-errors` — retry files that previously failed to decode.
* `index --no-prune` — keep records for files that have vanished from the
  indexed root (see [Pruning vanished files](#pruning-vanished-files)).
* `query --top N` — how many results per mode (default 10).
* `query --try-rates` — also try the 44.1/48 kHz mislabel probes (3x slower,
  **off by default**; see [Sample-rate mislabelling](#sample-rate-mislabelling)).
* `query --ignore-filenames` — score modes 2 and 3 on audio evidence only,
  ignoring Tascam take numbers parsed from filenames.
* `backfill --workers N` — worker processes for the envelope pass.

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

**The sharpness denominator has a floor.** Some files share exactly *one*
offset bin with the seed — there is no second bin, so there is nothing to
measure the winner against. Dividing by 1 there produced an unfalsifiable
`25.0x` (and a `[MATCH  ]`) for a lone bin backed by no evidence at all, so the
background is floored at **3 votes**: sharpness is `votes / max(background, 3)`.
Consequences worth knowing:

* A file with a genuinely measured background (anything ≥ 3) scores exactly as
  it always did — this changes no real result.
* A lone bin can never report more than `votes / 3`, and needs at least
  `4 × 3 = 12` aligned votes before it can clear the sharpness bar at all.
* So a lone bin still has to clear the *vote* floor on its own merits. Sharpness
  is a ratio, not proof; a 25-vote match against a silent background is exactly
  as trustworthy as 25 votes, no more.

Results whose library file has since been deleted or moved are annotated
`[missing]` — the index can always be older than the filesystem. Re-run
`audio-match index` on that library root to prune them (below).

**Uninformative hashes are skipped.** Mains hum, tape hiss and room tone
produce a handful of hashes that occur in nearly every file in the library.
They carry almost no information but would dominate both the cost and the noise
floor, so any hash with more than `MAX_POSTINGS_PER_HASH` (400) postings is
dropped from the query entirely. The cap is applied *inside SQLite*, by a
`GROUP BY … HAVING COUNT(*)` pre-filter, so an overfull posting list is never
read into memory — on a hum-heavy seed that is the difference between tens of
megabytes of transient Python objects and effectively none.

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

Pass **`--try-rates`** and the query is run three times: once natively, once
with the seed decoded at `11025 × 44100/48000` Hz and once at
`11025 × 48000/44100` Hz, then reinterpreted as 11025 Hz. That is a pure ffmpeg
`-ar` change, so it costs a re-decode of the seed and nothing else — but it is
still three decodes instead of one, which is why it is **off by default**.

**When to use it.** Turn it on when the *seed's own* header may be a
reconstructed guess rather than something the recorder wrote: carved files,
files recovered from a damaged card, anything hand-patched. In this project's
corpus `pakDR40_earlier.wav` is exactly that case — its 44.1 kHz rate was
assumed from provenance, not read from an intact header, so it is a seed worth
querying with `--try-rates`. If your seed came straight off a recorder, or you
have otherwise confirmed its rate, the extra two probes cannot tell you anything
and you are paying 3x the seed decode for nothing.

A hit found on one of the shifted probes is reported as:

```
      *** MATCH AT WRONG SAMPLE RATE *** seed-is-48k-labelled-44.1k: the seed
      had to be resampled by 1.0884x to line up, i.e. the two files disagree
      about the true sample rate of this recording.
```

The reported time offset is always on the **library file's** timeline.

Note that this probe only compensates for a wrong header on the **seed**. A
*library* file with a mislabelled header is found by seeding with a file you
trust and passing `--try-rates`, which shifts the seed to meet it.

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

## Mode 3 — pair mates (same session timeline)

> **New databases get this automatically. An index built before mode 3 existed
> needs one `audio-match backfill` pass first — not a re-index.** See
> [Backfilling an older index](#backfilling-an-older-index).

Modes 1 and 2 answer "is this the same audio?" and "is this the same rig?".
Mode 3 answers a third question: **which other files captured this same stretch
of time?**

That covers two quite different situations, and the difference matters:

* **Dual-record.** The DR-40 in 4-channel mode writes `…S12.wav` and
  `…S34.wav` for one take: the same performance through two different
  microphone pairs, on one recorder, on one clock.
* **A multi-recorder rig.** Extra channels captured on entirely separate
  equipment — different microphones, different gain, a different position in
  the room, a sample clock that agrees with nobody, and a start time that is
  whenever somebody pressed record.

The first case is easy for mode 1. The second is impossible for it: change the
microphone and its position and almost every spectral peak moves, so almost
every landmark differs. Mode 3 leads with evidence that does not care what the
equipment was.

### Evidence 1 — the activity envelope (primary)

Every file carries a **loudness envelope at one value per second**: the RMS of
the mono downmix over each second, in dBFS. Two files are compared by
best-lag normalised cross-correlation of those envelopes.

This works across rigs because the envelope is a property of *the performance*,
not of the capture. Songs start and stop, applause happens, somebody talks
between numbers — and every microphone in the room hears that same shape,
however differently it hears the timbre. EQ, reverb, 6 dB less gain and a
128 kbps mp3 leave it essentially untouched.

**Why one value per second, specifically.** Two recorders have independent
crystals. Tens of ppm of disagreement is normal; 200 ppm would be a bad pair of
consumer machines. Over a 45-minute take, 200 ppm is
`200e-6 × 2700 s = 0.54 s` — *half an envelope sample*. So at 1 Hz an
unsynchronised second recorder still produces a single sharp correlation peak
and no drift model is needed at all. At 10 Hz the same drift would smear that
peak across five samples and the score would collapse. 1 Hz is the resolution
at which clock drift stops mattering.

It is also free: one byte per second is 2.6 KB for a 45-minute file, about 9 MB
for a 2500-hour library, and it is accumulated during the single decode pass
that already feeds modes 1 and 2.

**The overlap penalty, and why it is the whole ballgame.** Correlating two
envelopes at their best lag, naively, does not work. Measured over the whole
recovered corpus (8 files, 56 ordered pairs -- 8 true, 48 unrelated):

| statistic | true S12/S34 pairs | unrelated pairs |
| --- | --- | --- |
| raw best-lag Pearson r | 0.913 … 0.948 | up to **0.911** |
| penalised score | 0.887 … 0.937 | up to **0.584** |

The raw correlation does not separate at all, and the reason is completely
predictable once you see it: every recording starts quiet, gets loud and ends
quiet. Slide two unrelated files until only their *edges* touch and you are
correlating two fades. The worst real instance: `pakDR40_S34` against
`TASCAM_0077S34`, two entirely unrelated sessions, raw r = 0.878 over 307
seconds of overlap out of the 2437 available.

So the reported score penalises an overlap for being thin in both senses:

```
score = r × (L / L_max)^0.5 × sqrt(L / (L + 60 s))
```

where `L` is the overlap at the winning lag and `L_max = min(len(seed),
len(candidate))`. The **relative** term is what kills the edge artefact, and it
costs a true pair nothing — a true pair's best lag is the one that uses all the
overlap there is, so its factor is 1. The **absolute** term guards the opposite
end, where a short seed overlaps a long library file fully at hundreds of
different lags and the relative term cannot help; one minute of overlap scores
at 0.71 of face value, five minutes at 0.91, forty-five at 0.99.

On top of that, lags overlapping by less than
`max(60 s, min(5 min, 50% of the shorter file))` are not considered at all.

The output reports both numbers — `envelope r=+0.89 … (scored +0.81 over a
300s overlap)` — so you can always see what the penalty did.

### Evidence 2 — constellation coherence (confirming)

For the top envelope candidates, mode 3 also runs the mode-1 landmark search,
restricted to those files. Landmarks that survive between two captures should
all agree on one time offset — or, if the two recorders' clocks differ, on one
straight **line** through `(t_seed, t_library)`. Fitting that line's slope both
recovers votes that a plain offset histogram would have smeared away and
measures the clock drift in ppm.

**Drift is much harder to measure than it looks, and the tool says so.** Two
slopes are distinguishable only when they move votes by more than a smoothed
histogram bin (3 frames, 70 ms) across the whole seed:

| seed length | grid step | slopes tried | total drift at 100 ppm |
| --- | ---: | ---: | --- |
| 150 s | — | 1 (zero) | 6.5 ms — 0.3 bins, **no fit attempted** |
| 10 min | 116 ppm | 5 | 65 ms — 0.9 bins |
| 45 min | 26 ppm | 23 | 270 ms — 3.9 bins |

The last column is what actually matters. Measured on the corpus: a genuine
+104 ppm on a *ten-minute* seed recovers only **4% more votes** when
compensated — 3189 at zero drift against 3322 at the nearest grid point. That
is a coin flip, not a measurement, and an earlier version of this fit duly
reported a confident `+77 ppm` for it — and reported `+77 ppm` just as
confidently for two files recorded off *the same crystal*.

So a non-zero slope has to beat the zero slope by 10% of its votes before it is
believed. Below that the output says **"no clock drift above the 116 ppm this
seed length can resolve"**, which is a different statement from "the clocks
agree"; and a seed too short for any grid says *drift not measurable*. In
practice the drift figure earns its keep on full-length seeds — which is the
normal way to use this mode — and is honestly silent on excerpts.

Coherence is evaluated only within ±30 s of the lag the envelope proposed: it
is confirmation of that specific alignment, not an independent search. That
also means its sharpness figure is measured against the tallest competing bin
*inside that window*, which is a stricter reference than mode 1's whole-file
background, not a looser one.

**Coherence is decisive when it fires and silent when it does not.** A second
recorder with different microphones may share no usable landmarks at all. So:

> **`acoustic coherence: none` is not evidence against a pair.** It is the
> expected reading for a genuine different-equipment capture.

### The verdict

| Verdict | Requires |
| --- | --- |
| **PAIR** | score ≥ 0.80, **or** coherence `strong` with score ≥ 0.65 |
| **LIKELY PAIR** | score ≥ 0.65 |
| weak | anything below |

Two routes to PAIR, because one rig and two rigs leave different evidence.
Coherence `strong` (≥ 30 aligned votes at ≥ 4× sharpness) essentially cannot
happen by chance, so it promotes a merely-plausible envelope score; but it can
never rescue an envelope that disagrees about the timeline.

Candidates are *generated* by envelope score alone — everything is correlated,
the top 20 get the coherence pass — but they are *reported* verdict first, then
by score. A file whose landmarks fall on a drifting line with the seed is as
close to proof as this tool gets, and burying it under a slightly higher
envelope score with nothing behind it would be the wrong way round.

The thresholds come from the measurements above. The unrelated set is the
*hardest* negative class available — the same band, the same recorder, the same
room, sets of similar length and shape on different days — so its 0.584 ceiling
is realistic rather than flattering. LIKELY PAIR at 0.65 clears it; PAIR at
0.80 sits in the empty middle of a gap 0.30 wide.

**LIKELY PAIR is the honest verdict for a different-recorder capture**, and is
not a lesser result. The test suite's simulated second rig (+104 ppm resample,
−6 dB below 200 Hz, +3 dB above 4 kHz, an echo, −6 dB gain, 128 kbps mp3)
scores **0.871** on the envelope (raw r 0.913) at exactly the right lag, with
the runner-up at 0.407.

### What the output says

Every hit lists the evidence actually behind it, one line each. This is a real
run, seeded with `TASCAM_0077S34` against a library holding the four sessions'
`S12` files (lines wrapped here for width):

```
== MODE 3: pair mates (same session timeline) ==
seed: .../seeds/TASCAM_0077S34.wav
seed length: 10:00.0
seed filename parsed as Tascam take 0077 S34

 1. [   PAIR    ] .../lib/TASCAM_0077S12.wav
      length 10:00.0; the seed's 0:00 lands at 0:00.0 in this file
      - envelope r=+0.93 at lag +0s (scored +0.88 over a 600s overlap)
      - acoustic coherence: strong (289 aligned landmark votes, 26.3x
        sharpness, offset +0.00s, no clock drift above the 116 ppm this
        seed length can resolve)
      - filename: dual-record pair-mate of the seed (take 0077 S12)
      - session signature: 0.81 (mode 2's score, as supporting evidence only)
      - segments align: 8/8 boundaries within +/-4s (4 active stretch(es) in
        the seed, 4 in this file)

 2. [   weak    ] .../lib/pakDR40_S12.wav
      length 10:00.0; the seed's 0:00 lands at -3:23.0 in this file
      - envelope r=+0.55 at lag -203s (scored +0.42 over a 397s overlap)
      - acoustic coherence: none -- consistent with a capture on different
        equipment (or with no shared audio at all)
      - session signature: 0.49 (mode 2's score, as supporting evidence only)
      - segments align: 2/5 boundaries within +/-4s (4 active stretch(es) in
        the seed, 4 in this file)
```

Note what the runner-up shows: a respectable-looking raw `r=+0.55`, found at a
lag that only overlaps by 397 of the available 600 seconds, penalised to 0.42 —
and 2 of 5 track boundaries lining up, which is what "no relationship" looks
like next to the winner's 8 of 8.

Take numbers and the mode-2 session score appear as **supporting evidence
lines, never as gates** — `pakDR40_S12.wav` has no parseable take number at all
and still wins its own query on the audio.

### The segment view

The last line is for reading in terms of tracks rather than correlations. The
aligned envelopes are smoothed over 5 s and thresholded 35% of the way from the
10th to the 90th percentile of each file's own levels; runs under 20 s and gaps
under 8 s are absorbed so that a bar of rest inside a tune does not read as two
tracks. The surviving boundaries are then matched across the alignment.

`segments align: 6/6 boundaries within ±4s` means the two files agree about
where six track starts and ends fall. **This is presentation only and does not
affect the verdict** — it is a sanity check you can hear, and a way to talk
about the result with somebody who thinks in set lists.

### Backfilling an older index

The envelope lives in a new `files.envelope` column. This is a *storage*
change, not a fingerprint change, so an existing database is **not** invalidated
and does **not** need re-indexing — its landmarks and session signatures are
still exactly right. Opening it migrates the table in place; one pass then
fills the new column:

```bash
audio-match backfill                     # or --db PATH, --workers N
```

Backfill decodes only the rows whose envelope is missing, and computes only the
envelope — no STFT, no peak picking, no hashing — which makes it roughly three
times cheaper per byte than indexing. It writes one column with a targeted
`UPDATE`: landmarks, signatures, sizes and mtimes come out byte-identical.
Like `index`, it is resumable — the work list is "rows with no envelope", which
shrinks as the run commits — and a file whose size or mtime no longer matches
its row is skipped rather than filled, because an envelope computed from audio
the landmarks were *not* computed from would be worse than none. Run
`audio-match index` on those.

Files indexed after this change always get an envelope; `stats`, `index` and
`query --mode pair` all tell you if any are still missing.

### Limitations of mode 3

* **Anything that recorded the same hour will match.** The soundboard feed, the
  house recording, another band's punter with a phone in his pocket — if it
  captured the same timeline, its envelope correlates and it is a true positive
  by mode 3's definition. That is usually exactly what you want. When it is
  not, the coherence line and the take number are what single out one
  recorder's own dual-record mate.
* **Mode 3 does not need, and does not check, that the audio is the same.**
  Envelope agreement means "these were recording at the same time", nothing
  more. Use mode 1 when you need "this is the same audio".
* **Below a minute of seed it refuses.** At 1 Hz, a 30-second seed is 30
  numbers; correlated against a few hundred candidate lags, r > 0.8 happens by
  chance routinely. Pair mode says so rather than reporting a number it does
  not believe. Ten minutes or more is comfortable.
* **Silence has no shape.** A file of room tone has a flat envelope and cannot
  be aligned to anything; mode 3 reports that rather than correlating noise.
* **Drift beyond ±300 ppm is not searched**, and drift is not measurable at all
  on short seeds (see the table above). The envelope alignment itself is
  unaffected either way — that is the point of 1 Hz.
* **A performance with no dynamics is hard.** The envelope carries information
  only where the loudness *changes*; a continuous, evenly-loud 45 minutes gives
  the correlation little to hold on to. Sets with gaps between numbers are the
  easy case, and are also the normal one.
* **The thresholds are calibrated on four sessions, not four hundred.** The
  gap between true pairs and unrelated ones is wide (0.887 … 0.937 against
  0.402 … 0.584 on whole files), but the *closest* negative measured — two
  different sessions by the same band, compared over ten-minute excerpts —
  reached 0.620 against a LIKELY PAIR bar of 0.65. That 0.03 is the thinnest
  margin anywhere in this mode, and it is on excerpts rather than whole files.
  Seed with as much of the recording as you have; the separation improves with
  length, and it is what the numbers above were measured on.
* **`--mode both` is deliberately still match + session.** Pair matching is a
  different question with a different answer shape, so it is requested
  explicitly with `--mode pair` rather than bolted onto the default output.

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

**If a worker dies.** A decode that gets the worker process killed outright —
the OOM killer picking it off, a segfaulting ffmpeg — is not the same as a file
that fails to decode, because there is no exception to catch and no error to
record. The run prints

```
ERROR: a worker died (likely OOM or a crashing decode); progress is saved --
re-run `audio-match index` to resume
```

and exits **nonzero**. Everything committed so far is intact, so the fix is
simply to run the same command again: the resume stamps skip everything already
done and the run picks up where it stopped. If it dies again in the same place,
the culprit is one specific file — `--workers 1` will make the run stop *on*
that file so you can identify and exclude it, and lowering `--workers` reduces
peak memory if the OOM killer is the cause.

(This is why the pool is a `ProcessPoolExecutor` rather than a
`multiprocessing.Pool`: `Pool.imap_unordered` simply blocks forever when a
worker disappears, which turns a nine-hour unattended run into a silent hang
with no output and no error.)

### Pruning vanished files

Deleting or renaming a library file does not, on its own, remove it from the
index — so without a prune pass, queries go on ranking paths that no longer
exist. After each scan, `audio-match index` tombstones every live record **under
the root it was given** whose path has vanished, and reports the count:

```
indexed 12 file(s), 0 error(s), 8,431 unchanged, 3 vanished files pruned
```

A renamed file therefore produces two changes in one run: the old path is pruned
and the new path is indexed fresh. Pruning uses the same tombstone mechanism as
re-indexing, so the vanished file's landmarks become unreachable immediately and
are reclaimed by `audio-match purge` along with everything else.

**Records outside the indexed root are never touched.** One database may hold
several roots — a second library, a removable drive — and `index /mnt/library`
must not delete the records for `/media/archive`. The prune pass is scoped
strictly to paths at or below the root you passed on this run, so each root is
pruned only by a run that actually scanned it.

The corollary is a real hazard: if a root is a **mount point and it is not
mounted**, the directory looks empty and every record under it is a vanished
file. Use `--no-prune` for those roots, or make the mount a precondition of the
indexing job.

**Superseded files and `purge`.** There is deliberately no index on
`hashes.file_id` — adding one would roughly double the database, and it would
only ever be used for deletion. Instead, re-indexing a changed file marks the old
record dead and writes a new one with a fresh id. Queries skip dead ids, so
results stay correct, but the old landmarks still occupy space. `audio-match
index` tells you when this has happened; `audio-match purge` reclaims it in a
single sequential rewrite. Pruned (vanished) files use the same mechanism, so a
library that churns will accumulate reclaimable space the same way. You will
rarely need it.

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
* **Heavy time-stretching is not handled.** The two 44.1/48 kHz probes
  (`--try-rates`) are the only speed variations tested, and they are off by
  default. A genuine tempo change, a varispeed transfer or a pitch-shift will
  not match.
* **Very long seeds are subsampled.** Above 400 000 landmarks — roughly 2.5
  hours of audio at the query peak density — the seed is uniformly subsampled
  down to the cap and the query prints a warning. The result is still usable,
  but its scores are lower than an uncapped query would give and so are not
  comparable with those of a normal-length seed. Seed with an excerpt instead.
* **Silence and near-silence produce no landmarks.** A file of room tone cannot
  be matched by mode 1 at all (mode 2 will still rank it).
* **The peak-density budget is a real trade-off.** 16 landmarks/second is sparse
  by the standards of commercial fingerprinting, chosen so that 2 500 hours fits
  in ~2.4 GB. Queries compensate by fingerprinting the *seed* four times more
  densely, which costs only query CPU. If you have disk to spare and want more
  sensitivity for very short excerpts, raise
  `INDEX_PEAKS_PER_BAND_PER_SEC` in `audiomatch/config.py` and re-index —
  database size scales with it roughly linearly.
* **Mode 3 finds co-recordings, not the same audio.** Envelope agreement means
  "these two files were recording at the same time" — a soundboard feed, the
  house recording or somebody else's machine all qualify, correctly. See
  [Limitations of mode 3](#limitations-of-mode-3) for the rest.
* **Mode 3 needs an envelope, and old databases do not have one.** Run
  `audio-match backfill` once. `stats`, `index` and `query --mode pair` all say
  so if any files are still missing it.
* **Changing anything in `config.py` marked `FINGERPRINT_AFFECTING` invalidates
  the database.** Bump `SCHEMA_VERSION` when you do; existing databases will
  then refuse to open with a message telling you to re-index, rather than
  silently returning garbage. Adding a *column* is different — bump
  `STORAGE_VERSION`, add a migration, and leave the fingerprints alone.

---

## Layout

| File | Contents |
| --- | --- |
| `audiomatch/config.py` | Every tuning constant, with the reasoning behind it. |
| `audiomatch/audio.py` | ffmpeg/ffprobe subprocess decoding, streaming and one-shot. |
| `audiomatch/fingerprint.py` | Streaming STFT, peak picking, landmark hashing. |
| `audiomatch/session.py` | Noise floor, hum, channel stats, filename parsing, similarity. |
| `audiomatch/envelope.py` | 1 Hz activity envelope: quantisation, streaming, FFT alignment, segments. |
| `audiomatch/analyze.py` | One decode per file, feeding all three signatures. |
| `audiomatch/db.py` | SQLite schema, writes, lookups, purge. |
| `audiomatch/indexer.py` | Directory walk, resume planning, worker pool, progress, backfill. |
| `audiomatch/query.py` | Offset histograms, scoring, session ranking, drift fitting, pair verdicts. |
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
    envelope BLOB,                       -- uint8 dBFS/second activity envelope
                                         -- NULL = never computed -> 'backfill'
    indexed_at REAL
);
CREATE UNIQUE INDEX ix_files_path ON files(path) WHERE alive = 1;

CREATE TABLE hashes (
    hash INTEGER, file_id INTEGER, t INTEGER,
    PRIMARY KEY (hash, file_id, t)
) WITHOUT ROWID;
```

Two version numbers live in `meta`, and they mean different things:

| Key | Meaning | On mismatch |
| --- | --- | --- |
| `schema_version` | The stored **fingerprints** changed. | Refuse to open: delete and re-index. |
| `storage_version` | The **table layout** grew something the existing fingerprints are still valid under. | Migrate in place (additive `ALTER TABLE`), then `audio-match backfill` fills the new column. |

Splitting them is the reason adding mode 3 did not invalidate anybody's index.
A `storage_version` *newer* than the running build is refused, since this build
cannot know what the columns it does not have are supposed to contain.

---

## Tests

```bash
pip install pytest
python3 -m pytest tests/          # 115 tests, ~6 minutes warm
```

The suite runs against the **real** recovered DR-40 corpus in
`/mnt/host/projects/audio-recovery/recovered/`, cutting short excerpts with
ffmpeg into a scratch directory (it never loads a whole 1 GB file). If that
corpus is not present those tests skip and the pure unit tests still run.

The first run is much slower than later ones: it cuts every excerpt it needs
(including four ten-minute pairs for mode 3, ~1.4 GB) and caches them. Point
`AUDIOMATCH_TEST_DIR` somewhere with a couple of gigabytes free.

Mode 3's tests are the expensive half, and unavoidably so: the envelope is a
1 Hz signal, so demonstrating anything about it needs ten-minute seeds rather
than the 150 s that is plenty for the constellation. The four ground-truth
queries share one set of seed analyses (a module-scoped fixture) for that
reason.
