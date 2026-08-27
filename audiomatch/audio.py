"""All decoding goes through the ffmpeg/ffprobe binaries.

Rationale: the target library is full of recovered / carved / hand-patched WAV
files whose headers lie about length (and sometimes about sample rate).  A
pure-Python WAV reader chokes on those; ffmpeg does not.  Everything is decoded
to raw ``f32le`` on stdout and read straight into numpy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from . import config


class AudioError(RuntimeError):
    """Raised when a file cannot be probed or decoded."""


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioError(
            f"{name!r} was not found on PATH. audio-match requires ffmpeg "
            f"(package 'ffmpeg' on Debian/Ubuntu, 'ffmpeg' on RHEL/EPEL)."
        )
    return path


def require_ffmpeg() -> None:
    """Fail fast, with a useful message, if the ffmpeg toolchain is missing."""
    _binary("ffmpeg")
    _binary("ffprobe")


@dataclass(frozen=True)
class Probe:
    """Container facts, read from the header only (no full-file read)."""

    duration: float
    sample_rate: int
    channels: int
    bits: int
    codec: str

    @property
    def is_usable(self) -> bool:
        return self.sample_rate > 0 and self.channels > 0


def probe(path: str, timeout: float = 60.0) -> Probe:
    """Read stream/container facts with ffprobe.  Cheap: header access only."""
    cmd = [
        _binary("ffprobe"), "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=sample_rate,channels,bits_per_raw_sample,bits_per_sample,"
        "codec_name,duration:format=duration",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffprobe timed out after {timeout:g}s") from exc
    if out.returncode != 0:
        msg = out.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(msg[-1] if msg else f"ffprobe exit {out.returncode}")
    try:
        data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise AudioError(f"ffprobe produced unparseable JSON: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise AudioError("no audio stream found")
    st = streams[0]

    def _num(*keys, default=0.0):
        for k in keys:
            v = st.get(k)
            if v not in (None, "", "N/A"):
                try:
                    return float(v)
                except ValueError:
                    continue
        return default

    duration = _num("duration")
    if duration <= 0:
        try:
            duration = float((data.get("format") or {}).get("duration") or 0.0)
        except ValueError:
            duration = 0.0

    p = Probe(
        duration=max(0.0, duration),
        sample_rate=int(_num("sample_rate")),
        channels=int(_num("channels")),
        bits=int(_num("bits_per_raw_sample", "bits_per_sample")),
        codec=str(st.get("codec_name") or "?"),
    )
    if not p.is_usable:
        raise AudioError("audio stream has no sample rate or channels")
    return p


def _decode_cmd(path: str, rate: int, channels: int,
                start: Optional[float], length: Optional[float]) -> list[str]:
    cmd = [_binary("ffmpeg"), "-v", "error", "-nostdin"]
    if start:
        # Before -i so ffmpeg can seek instead of decoding-and-discarding.
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", path]
    if length:
        cmd += ["-t", f"{length:.6f}"]
    cmd += [
        "-map", "0:a:0",
        "-ac", str(channels),
        "-ar", str(rate),
        "-f", "f32le",
        "-",
    ]
    return cmd


def decode(path: str, *, rate: int = config.ANALYSIS_SR, channels: int = 2,
           start: Optional[float] = None, length: Optional[float] = None,
           timeout: float = 3600.0) -> np.ndarray:
    """Decode (a slice of) a file fully into memory.

    Returns an array of shape ``(n_frames, channels)``.  Use only for short
    excerpts; see :func:`decode_stream` for whole-file work.
    """
    cmd = _decode_cmd(path, rate, channels, start, length)
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout,
                             check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffmpeg timed out after {timeout:g}s") from exc
    if out.returncode != 0 and not out.stdout:
        msg = out.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(msg[-1] if msg else f"ffmpeg exit {out.returncode}")
    buf = np.frombuffer(out.stdout, dtype="<f4")
    usable = (buf.size // channels) * channels
    if usable == 0:
        raise AudioError("decoded to zero samples")
    return buf[:usable].reshape(-1, channels)


def decode_stream(path: str, *, rate: int = config.ANALYSIS_SR,
                  channels: int = 2,
                  block_frames: int = config.DECODE_BLOCK,
                  ) -> Iterator[np.ndarray]:
    """Yield successive ``(n, channels)`` blocks of a full decode.

    Streaming keeps worker memory flat (tens of MB) regardless of file length,
    which matters when 32 workers are chewing through 1 GB WAVs.
    """
    cmd = _decode_cmd(path, rate, channels, None, None)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    # Drain stderr concurrently.  A corrupt file can make ffmpeg emit far more
    # than a pipe buffer's worth of error lines; if nobody reads them ffmpeg
    # blocks on write(2), stops producing stdout, and the decode deadlocks
    # forever.  Recovered/carved files do exactly this.
    err_chunks: list[bytes] = []
    err_thread = threading.Thread(
        target=lambda p, out: out.append(p.read()),
        args=(proc.stderr, err_chunks), daemon=True)
    err_thread.start()
    nbytes = block_frames * channels * 4
    stderr_tail = b""
    try:
        assert proc.stdout is not None
        carry = b""
        while True:
            chunk = proc.stdout.read(nbytes)
            if not chunk:
                break
            if carry:
                chunk = carry + chunk
                carry = b""
            extra = len(chunk) % (channels * 4)
            if extra:
                carry = chunk[len(chunk) - extra:]
                chunk = chunk[:len(chunk) - extra]
            if not chunk:
                continue
            block = np.frombuffer(chunk, dtype="<f4").reshape(-1, channels)
            yield block
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        err_thread.join(timeout=30.0)
        stderr_tail = b"".join(err_chunks)[-2000:]
        if proc.stderr is not None:
            proc.stderr.close()
        proc.wait()
    if proc.returncode not in (0, None):
        msg = stderr_tail.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(msg[-1] if msg else f"ffmpeg exit {proc.returncode}")


def to_mono(block: np.ndarray) -> np.ndarray:
    """Downmix an ``(n, channels)`` block to a 1-D float32 array."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32, copy=False)
    return block.mean(axis=1, dtype=np.float32)
