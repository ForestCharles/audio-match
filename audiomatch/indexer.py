"""The library indexing pass.

Design
------
* One *writer* (the main process) owns the sqlite connection; N *workers*
  decode and fingerprint.  sqlite does not like concurrent writers, and the
  work is overwhelmingly CPU+IO in the workers anyway.
* Work is handed out as individual futures with a bounded in-flight window, so
  a single 1 GB file cannot stall the pool and the queue cannot grow to the
  size of the library.
* ``concurrent.futures.ProcessPoolExecutor`` is used rather than
  ``multiprocessing.Pool`` **deliberately**: if a worker is SIGKILLed (the OOM
  killer, a segfaulting ffmpeg) ``Pool.imap_unordered`` simply blocks forever,
  which turns an unattended multi-terabyte index run into a silent hang.  The
  executor raises ``BrokenProcessPool`` instead, which we turn into a clear
  message and a nonzero exit.
* Results are committed in batches of ``COMMIT_EVERY``; a crash therefore loses
  at most that batch plus whatever the workers had in flight.
* Resumability is keyed on ``(path, size, mtime)``.  A rerun re-stats every
  candidate file (cheap) and skips anything whose stamp is unchanged.
* After the scan, alive rows *under the indexed root* whose file has vanished
  are tombstoned so that queries stop ranking paths that no longer exist.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from . import config
from .analyze import Analysis, analyze_file, envelope_file
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


#: Message shown when a worker process dies outright.
WORKER_DIED_MESSAGE = (
    "\nERROR: a worker died (likely OOM or a crashing decode); progress is "
    "saved -- re-run `audio-match index` to resume")

#: Test-only hook.  When this environment variable is set to a file's
#: *basename*, the worker that picks that file up SIGKILLs itself, which is the
#: only portable way to reproduce an OOM-killed worker.  Workers are spawned,
#: so a monkeypatch in the parent process would not reach them; an environment
#: variable is inherited.  Never set in normal operation.
KILL_ENV = "AUDIOMATCH_TEST_KILL_WORKER_ON"


def _worker(path: str) -> Analysis:
    victim = os.environ.get(KILL_ENV)
    if victim and os.path.basename(path) == victim:
        os.kill(os.getpid(), signal.SIGKILL)      # pragma: no cover
    try:
        return analyze_file(path)
    except BaseException as exc:                  # pragma: no cover
        return Analysis(path=path, size=0, mtime=0.0, status="error",
                        error=f"worker crashed: {exc!r}")


def _envelope_worker(path: str) -> Analysis:
    victim = os.environ.get(KILL_ENV)
    if victim and os.path.basename(path) == victim:
        os.kill(os.getpid(), signal.SIGKILL)      # pragma: no cover
    try:
        return envelope_file(path)
    except BaseException as exc:                  # pragma: no cover
        return Analysis(path=path, size=0, mtime=0.0, status="error",
                        error=f"worker crashed: {exc!r}")


class WorkerDied(RuntimeError):
    """Raised internally when the process pool breaks."""


def prune_vanished(db: Database, root: str,
                   log: Optional[Callable[[str], None]] = None) -> int:
    """Tombstone alive rows under ``root`` whose file no longer exists.

    Only rows *under the indexed root* are considered: one database may hold
    several roots (removable media, a second library), and a root that is not
    currently mounted must not have its records deleted.
    """
    log = log or (lambda msg: None)
    gone = [p for p in db.alive_paths_under(root) if not os.path.exists(p)]
    for path in gone:
        db.retire(path)
        log(f"  PRUNE {path}")
    if gone:
        db.commit()
    return len(gone)


def run_index(db: Database, root: str, *, workers: int = 0,
              all_files: bool = False, force: bool = False,
              retry_errors: bool = False, prune: bool = True,
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
               "bytes": plan.todo_bytes, "seconds": 0.0, "hashes": 0,
               "pruned": 0, "aborted": False}

    if prune:
        summary["pruned"] = prune_vanished(db, os.path.abspath(root), log)
        if summary["pruned"]:
            log(f"prune: {summary['pruned']:,} vanished file(s) pruned")

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
                    sig=an.signature, hashes=an.hashes, times=an.times,
                    envelope=an.envelope)
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

    def consume(an: Analysis) -> None:
        store(an)
        progress.update(an)

    try:
        if n_workers == 1:
            for path in plan.todo:
                consume(_worker(path))
        else:
            _run_pool(plan.todo, n_workers, consume)
    except KeyboardInterrupt:
        log("\ninterrupted -- committing what is done; rerun to resume")
    except WorkerDied:
        summary["aborted"] = True
    finally:
        db.commit()
        progress.finish()

    if summary["aborted"]:
        log(WORKER_DIED_MESSAGE)

    summary["seconds"] = time.time() - t0
    return summary


#: In-flight futures per worker.  Enough to keep every worker fed across a
#: burst of tiny files, small enough that a library of millions of paths does
#: not become millions of live future objects.
INFLIGHT_PER_WORKER = 4


def _run_pool(todo: list[str], n_workers: int,
              consume: Callable[[Analysis], None],
              fn: Callable[[str], Analysis] = _worker) -> None:
    """Fan ``todo`` out over a process pool, feeding results to ``consume``.

    ``fn`` must be a module-level function: workers are *spawned*, so the task
    is pickled by qualified name and a closure or a lambda would not survive
    the trip.

    Raises :class:`WorkerDied` if a worker process disappears (SIGKILL, OOM
    killer, segfault) instead of hanging forever, which is what
    ``multiprocessing.Pool`` would do.
    """
    ctx = mp.get_context("spawn")
    pending = iter(todo)
    try:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=ctx) as pool:
            window = n_workers * INFLIGHT_PER_WORKER
            futures = {pool.submit(fn, p)
                       for p in itertools.islice(pending, window)}
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                broken = None
                for fut in done:
                    try:
                        consume(fut.result())
                    except BrokenProcessPool as exc:
                        # Keep draining: futures that *did* finish still hold
                        # real results, and committing them is free progress.
                        broken = exc
                if broken is not None:
                    for fut in futures:
                        fut.cancel()
                    raise WorkerDied(str(broken)) from broken
                futures |= {pool.submit(fn, p)
                            for p in itertools.islice(pending, len(done))}
    except BrokenProcessPool as exc:
        # The executor's own shutdown can surface the breakage too.
        raise WorkerDied(str(exc)) from exc


# --------------------------------------------------------------------------
# Backfill: fill in the activity envelope on a database that predates it
# --------------------------------------------------------------------------


def run_backfill(db: Database, *, workers: int = 0,
                 progress_stream=sys.stderr,
                 log: Optional[Callable[[str], None]] = None) -> dict:
    """Compute the 1 Hz envelope for every indexed file that lacks one.

    Reuses the index pass's worker pool, progress line and commit batching, so
    it is interruptible and resumable in exactly the same way: the work list is
    "rows whose ``envelope`` is NULL", which shrinks as the run commits, so a
    rerun picks up where the last one stopped with no extra bookkeeping.

    Files that already have an envelope are never opened.  Landmarks, session
    signatures, sizes and mtimes are never rewritten -- only the one column.

    A file whose ``(size, mtime)`` no longer matches the row is *skipped*, not
    filled: its audio has changed since it was fingerprinted, so an envelope
    computed now would describe different audio from the landmarks beside it.
    ``audio-match index`` is the command that resolves that, and the summary
    says so.
    """
    log = log or (lambda msg: None)
    todo = db.files_missing_envelope()
    summary = {"filled": 0, "errors": 0, "changed": 0, "missing": 0,
               "bytes": 0, "seconds": 0.0, "aborted": False,
               "considered": len(todo)}
    if not todo:
        log("backfill: every indexed file already has an activity envelope")
        return summary

    by_path = {path: (fid, size, mtime) for fid, path, size, mtime in todo}
    paths: list[str] = []
    total_bytes = 0
    for fid, path, size, mtime in todo:
        try:
            st = os.stat(path)
        except OSError:
            summary["missing"] += 1
            log(f"  GONE  {path}: no longer exists; run 'audio-match index' "
                f"on its library root to prune it")
            continue
        if st.st_size != size or abs(st.st_mtime - mtime) >= 1e-6:
            summary["changed"] += 1
            log(f"  STALE {path}: changed since it was indexed; re-run "
                f"'audio-match index' instead")
            continue
        paths.append(path)
        total_bytes += st.st_size

    log(f"backfill: {len(paths):,} file(s) to decode, "
        f"{_fmt_bytes(total_bytes)} to read "
        f"({summary['considered'] - len(paths):,} skipped)")
    if not paths:
        return summary

    n_workers = workers if workers and workers > 0 else (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(paths)))
    db.begin_bulk()
    progress = Progress(len(paths), total_bytes, progress_stream)
    t0 = time.time()
    pending = 0

    def consume(an: Analysis) -> None:
        nonlocal pending
        entry = by_path.get(an.path)
        if an.status == "ok" and an.envelope is not None and entry is not None:
            db.set_envelope(entry[0], an.envelope)
            summary["filled"] += 1
            if an.error:
                log(f"  WARN  {an.path}: {an.error}")
        else:
            summary["errors"] += 1
            log(f"  ERROR {an.path}: {an.error}")
        summary["bytes"] += an.size
        pending += 1
        if pending >= config.COMMIT_EVERY:
            db.commit()
            pending = 0
        progress.update(an)

    try:
        if n_workers == 1:
            for path in paths:
                consume(_envelope_worker(path))
        else:
            _run_pool(paths, n_workers, consume, fn=_envelope_worker)
    except KeyboardInterrupt:
        log("\ninterrupted -- committing what is done; rerun to resume")
    except WorkerDied:
        summary["aborted"] = True
    finally:
        db.commit()
        progress.finish()

    if summary["aborted"]:
        log(WORKER_DIED_MESSAGE.replace("`audio-match index`",
                                        "`audio-match backfill`"))
    summary["seconds"] = time.time() - t0
    return summary
