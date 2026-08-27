"""The library indexing pass.

Design
------
* One *writer* (the main process) owns the sqlite connection; N *workers*
  decode and fingerprint.  sqlite does not like concurrent writers, and the
  work is overwhelmingly CPU+IO in the workers anyway.
* Work is handed out with ``imap_unordered`` so a single 1 GB file cannot stall
  the pool.
* Results are committed in batches of ``COMMIT_EVERY``; a crash therefore loses
  at most that batch plus whatever the workers had in flight.
* Resumability is keyed on ``(path, size, mtime)``.  A rerun re-stats every
  candidate file (cheap) and skips anything whose stamp is unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from . import config
from .analyze import Analysis, analyze_file
from .db import Database


@dataclass
class Plan:
    todo: list[str]
    skipped: int
    todo_bytes: int


def walk_library(root: str, *, all_files: bool = False,
                 follow_symlinks: bool = False) -> Iterator[str]:
    """Yield candidate audio files under ``root``, sorted for determinism."""
    root = os.path.abspath(root)
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root,
                                                followlinks=follow_symlinks):
        dirnames.sort()
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if not all_files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in config.AUDIO_EXTENSIONS:
                    continue
            yield os.path.join(dirpath, name)


def plan_index(db: Database, root: str, *, all_files: bool = False,
               force: bool = False,
               retry_errors: bool = False) -> Plan:
    """Decide which files need (re)indexing."""
    stamps = db.all_stamps()
    todo: list[str] = []
    skipped = 0
    total_bytes = 0
    for path in walk_library(root, all_files=all_files):
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not force:
            prev = stamps.get(path)
            if prev is not None:
                size, mtime, status = prev
                unchanged = (size == st.st_size
                             and abs(mtime - st.st_mtime) < 1e-6)
                if unchanged and (status == "ok" or not retry_errors):
                    skipped += 1
                    continue
        todo.append(path)
        total_bytes += st.st_size
    return Plan(todo=todo, skipped=skipped, todo_bytes=total_bytes)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def _fmt_hms(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Progress:
    """Single-line progress with throughput and ETA."""

    def __init__(self, total_files: int, total_bytes: int,
                 stream=sys.stderr, min_interval: float = 0.5):
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.stream = stream
        self.min_interval = min_interval
        self.done_files = 0
        self.done_bytes = 0
        self.errors = 0
        self.t0 = time.time()
        self._last = 0.0
        self.enabled = stream is not None and stream.isatty()

    def update(self, an: Analysis, force: bool = False) -> None:
        self.done_files += 1
        self.done_bytes += an.size
        if an.status != "ok":
            self.errors += 1
        now = time.time()
        if not force and now - self._last < self.min_interval:
            return
        self._last = now
        self.render()

    def render(self) -> None:
        if self.stream is None:
            return
        elapsed = max(1e-6, time.time() - self.t0)
        rate = self.done_bytes / elapsed
        remaining = max(0, self.total_bytes - self.done_bytes)
        eta = remaining / rate if rate > 0 else float("inf")
        pct = (100.0 * self.done_bytes / self.total_bytes
               if self.total_bytes else 100.0)
        line = (f"  {self.done_files:,}/{self.total_files:,} files  "
                f"{_fmt_bytes(self.done_bytes)}/{_fmt_bytes(self.total_bytes)}"
                f" ({pct:5.1f}%)  {_fmt_bytes(rate)}/s  "
                f"elapsed {_fmt_hms(elapsed)}  ETA {_fmt_hms(eta)}"
                f"  errors {self.errors}")
        if self.enabled:
            self.stream.write("\r\x1b[2K" + line)
            self.stream.flush()
        elif self.done_files % 50 == 0 or self.done_files == self.total_files:
            self.stream.write(line + "\n")
            self.stream.flush()

    def finish(self) -> None:
        if self.stream is None:
            return
        self.render()
        if self.enabled:
            self.stream.write("\n")
        self.stream.flush()


def _worker(path: str) -> Analysis:
    try:
        return analyze_file(path)
    except BaseException as exc:                  # pragma: no cover
        return Analysis(path=path, size=0, mtime=0.0, status="error",
                        error=f"worker crashed: {exc!r}")


def run_index(db: Database, root: str, *, workers: int = 0,
              all_files: bool = False, force: bool = False,
              retry_errors: bool = False,
              progress_stream=sys.stderr,
              log: Optional[Callable[[str], None]] = None) -> dict:
    """Index ``root`` into ``db``.  Returns a summary dict."""
    log = log or (lambda msg: None)
    plan = plan_index(db, root, all_files=all_files, force=force,
                      retry_errors=retry_errors)
    log(f"scan: {len(plan.todo):,} file(s) to index, "
        f"{plan.skipped:,} unchanged and skipped, "
        f"{_fmt_bytes(plan.todo_bytes)} to read")
    summary = {"indexed": 0, "errors": 0, "skipped": plan.skipped,
               "bytes": plan.todo_bytes, "seconds": 0.0, "hashes": 0}
    if not plan.todo:
        return summary

    n_workers = workers if workers and workers > 0 else (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(plan.todo)))
    db.begin_bulk()
    progress = Progress(len(plan.todo), plan.todo_bytes, progress_stream)
    t0 = time.time()
    pending = 0

    def store(an: Analysis) -> None:
        nonlocal pending
        db.add_file(path=an.path, size=an.size, mtime=an.mtime,
                    status=an.status, error=an.error,
                    probe_duration=an.duration, sample_rate=an.sample_rate,
                    channels=an.channels, bits=an.bits, codec=an.codec,
                    sig=an.signature, hashes=an.hashes, times=an.times)
        summary["hashes"] += an.n_hashes
        if an.status == "ok":
            summary["indexed"] += 1
        else:
            summary["errors"] += 1
            log(f"  ERROR {an.path}: {an.error}")
        if an.status == "ok" and an.error:
            log(f"  WARN  {an.path}: {an.error}")
        pending += 1
        if pending >= config.COMMIT_EVERY:
            db.commit()
            pending = 0

    try:
        if n_workers == 1:
            for path in plan.todo:
                an = _worker(path)
                store(an)
                progress.update(an)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for an in pool.imap_unordered(_worker, plan.todo,
                                              chunksize=1):
                    store(an)
                    progress.update(an)
    except KeyboardInterrupt:
        log("\ninterrupted -- committing what is done; rerun to resume")
    finally:
        db.commit()
        progress.finish()

    summary["seconds"] = time.time() - t0
    return summary
