"""audio-match: find the same audio, and the same recording session, in a
very large library of WAV files.

Two independent search modes share one index:

``match``
    Constellation (Shazam-style) landmark hashing.  Finds other transfers,
    mixes, excerpts and lossy encodes of the *same performance*, and reports
    where in the library file the seed lines up.

``session``
    A compact per-file signature built from the noise floor, mains hum,
    inter-channel behaviour and container facts.  Ranks files that were
    plausibly recorded on the same rig, in the same room, on the same day --
    even though their audio content is completely different.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
