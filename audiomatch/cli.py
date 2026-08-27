"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from . import __version__, config
from .audio import AudioError, require_ffmpeg
from .db import open_db
from .indexer import _fmt_bytes, _fmt_hms, run_index
from .query import QueryResult, run_query
from .session import SessionScore


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def fmt_clock(seconds: float) -> str:
    """m:ss / h:mm:ss, with a sign for negative offsets."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{sign}{h}:{m:02d}:{s:04.1f}"
    return f"{sign}{m}:{s:04.1f}"


def _short(path: str, width: int = 78) -> str:
    if len(path) <= width:
        return path
    return "..." + path[-(width - 3):]


def print_match_results(res: QueryResult, out=None) -> None:
    w = (out or sys.stdout).write
    w("\n== MODE 1: same audio (constellation match) ==\n")
    threshold = max(config.CONFIDENT_MIN_VOTES,
                    config.CONFIDENT_VOTES_PER_SEED_SECOND * res.seed_seconds)
    w(f"seed: {res.seed_path}\n")
    w(f"seed length: {fmt_clock(res.seed_seconds)}   "
      f"landmarks: {res.seed_hash_counts.get('native', 0):,}   "
      f"confident-match threshold: {threshold:.0f} aligned votes\n")
    if not res.matches:
        w("\n  no candidate scored above the reporting floor "
          f"({config.REPORT_MIN_VOTES} votes).\n")
        w("  -> this audio does not appear to be in the index.\n")
        return

    confident = [m for m in res.matches if m.is_confident(res.seed_seconds)]
    w("\n")
    for i, m in enumerate(res.matches, 1):
        mark = "MATCH  " if m.is_confident(res.seed_seconds) else "weak   "
        w(f"{i:2d}. [{mark}] score {m.votes:6,}   "
          f"sharpness {m.sharpness:6.1f}x   "
          f"concentration {m.concentration * 100:5.1f}%\n")
        w(f"      {_short(m.path)}\n")
        w(f"      seed 0:00 lands at {fmt_clock(m.offset_seconds)} in this "
          f"file (length {fmt_clock(m.library_duration)}); "
          f"~{m.matched_seconds:.0f}s of seed aligned\n")
        if m.probe != "native":
            if m.is_confident(res.seed_seconds):
                w(f"      *** MATCH AT WRONG SAMPLE RATE *** {m.probe}: the "
                  f"seed had to be resampled by {1.0 / m.ratio:.4f}x to line "
                  f"up,\n      i.e. the two files disagree about the true "
                  f"sample rate of this recording.\n")
            else:
                w(f"      (best bin came from the {m.probe} probe)\n")
        if m.role:
            w(f"      Tascam take {m.take:04d} {m.role}\n")
        w("\n")

    if confident:
        w(f"  {len(confident)} confident match(es).\n"
          "  'score'       = votes in the winning time-offset bin.\n"
          "  'sharpness'   = that bin divided by the same file's next-tallest "
          "bin;\n"
          f"                  >= {config.CONFIDENT_MIN_SHARPNESS:.0f}x means "
          "the alignment is a spike, not a smear.\n")
    else:
        w("  no result cleared the confident-match threshold "
          f"({threshold:.0f} votes and "
          f"{config.CONFIDENT_MIN_SHARPNESS:.0f}x sharpness);\n"
          "  treat the above as coincidence unless you can hear the "
          "similarity.\n")


def _bar(x: float, width: int = 10) -> str:
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "#" * n + "." * (width - n)


def print_session_results(res: QueryResult, out=None) -> None:
    w = (out or sys.stdout).write
    w("\n== MODE 2: same session (heuristic signature ranking) ==\n")
    sig = res.seed_signature
    if sig is not None:
        mains = sig.mains_hz
        w(f"seed: {sig.sample_rate} Hz / {sig.channels} ch / {sig.bits}-bit"
          f"   L-R balance {sig.chan[0]:+.1f} dB"
          f"   L/R correlation {sig.chan[1]:+.2f}"
          f"   noise-floor L-R {sig.chan[2]:+.1f} dB"
          f"   DC bias {sig.chan[4]:+.0f}/{sig.chan[5]:+.0f} ppm"
          f"   mains: {str(mains) + ' Hz' if mains else 'none detected'}\n")
        if sig.take is not None:
            w(f"seed filename parsed as Tascam take {sig.take:04d}"
              f"{' ' + sig.role if sig.role else ''}\n")
    if not res.sessions:
        w("\n  index is empty.\n")
        return
    w("\n  rank  total  noise  hum   chan  cont.  file\n")
    for i, h in enumerate(res.sessions, 1):
        s: SessionScore = h.score
        w(f"  {i:3d}.  {s.total:.3f}  {s.noise:.3f} {s.hum:.3f} "
          f"{s.chan:.3f} {s.container:.3f}  {_short(h.path, 58)}\n")
        w(f"        {_bar(s.total)}  {h.sample_rate} Hz/{h.channels}ch/"
          f"{h.bits}-bit  {fmt_clock(h.duration)}\n")
        notes = list(s.notes)
        if h.is_pair_mate:
            notes.insert(0, "LIKELY DUAL-RECORD PAIR-MATE OF THE SEED")
        for note in notes:
            w(f"        - {note}\n")
    w("\n  Mode 2 is heuristic: it ranks candidates for you to audition.\n"
      "  A high score means 'same recorder, room and settings are plausible',\n"
      "  not 'same session, proven'.  Compare the component columns: 'noise'\n"
      "  and 'hum' are the physical fingerprints; 'cont.' only says the file\n"
      "  formats agree, which is weak evidence on its own.\n")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    require_ffmpeg()
    root = os.path.abspath(args.library)
    if not os.path.exists(root):
        print(f"error: {root!r} does not exist", file=sys.stderr)
        return 2
    with open_db(args.db) as db:
        print(f"indexing {root}\n     db: {args.db}", file=sys.stderr)
        summary = run_index(
            db, root, workers=args.workers, all_files=args.all_files,
            force=args.force, retry_errors=args.retry_errors,
            progress_stream=None if args.quiet else sys.stderr,
            log=(lambda m: None) if args.quiet
            else (lambda m: print(m, file=sys.stderr)))
        stats = db.stats()
    secs = summary["seconds"]
    print(f"\nindexed {summary['indexed']:,} file(s), "
          f"{summary['errors']:,} error(s), "
          f"{summary['skipped']:,} unchanged", file=sys.stderr)
    print(f"read {_fmt_bytes(summary['bytes'])} in {_fmt_hms(secs)}"
          f" ({_fmt_bytes(summary['bytes'] / secs) if secs else '0 B'}/s)",
          file=sys.stderr)
    print(f"database: {_fmt_bytes(stats['bytes'])}, "
          f"{stats['hashes']:,} landmarks, "
          f"{stats['files_ok']:,} indexed files, "
          f"{_fmt_hms(stats['seconds'])} of audio", file=sys.stderr)
    if stats["seconds"] > 0:
        per_hour = stats["bytes"] / (stats["seconds"] / 3600.0)
        print(f"          ~{_fmt_bytes(per_hour)} of database per hour "
              f"of audio", file=sys.stderr)
    if stats["files_dead"]:
        print(f"note: {stats['files_dead']:,} superseded file record(s) still "
              "hold landmarks; run 'audio-match purge' to reclaim the space",
              file=sys.stderr)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    require_ffmpeg()
    if not os.path.exists(args.seed):
        print(f"error: seed {args.seed!r} does not exist", file=sys.stderr)
        return 2
    with open_db(args.db, create=False) as db:
        stats = db.stats()
        if stats["files_ok"] == 0:
            print(f"error: {args.db} contains no indexed files",
                  file=sys.stderr)
            return 2
        try:
            res = run_query(db, os.path.abspath(args.seed), mode=args.mode,
                            top=args.top, sr_probes=not args.no_sr_probes,
                            ignore_filenames=args.ignore_filenames)
        except AudioError as exc:
            print(f"error: could not analyse seed: {exc}", file=sys.stderr)
            return 2
    print(f"index: {args.db}  ({stats['files_ok']:,} files, "
          f"{_fmt_hms(stats['seconds'])} of audio, "
          f"{stats['hashes']:,} landmarks)")
    if args.mode in ("match", "both"):
        print_match_results(res)
    if args.mode in ("session", "both"):
        print_session_results(res)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with open_db(args.db, create=False) as db:
        s = db.stats()
    print(f"database        {args.db}")
    print(f"size            {_fmt_bytes(s['bytes'])}")
    print(f"files indexed   {s['files_ok']:,}")
    print(f"files errored   {s['files_error']:,}")
    print(f"files superseded{s['files_dead']:,}")
    print(f"audio indexed   {_fmt_hms(s['seconds'])}")
    print(f"landmarks       {s['hashes']:,}")
    if s["seconds"]:
        print(f"db per hour     "
              f"{_fmt_bytes(s['bytes'] / (s['seconds'] / 3600.0))}")
    return 0


def cmd_errors(args: argparse.Namespace) -> int:
    with open_db(args.db, create=False) as db:
        rows = db.conn.execute(
            "SELECT path, error FROM files WHERE alive=1 AND status='error' "
            "ORDER BY path").fetchall()
    for path, err in rows:
        print(f"{path}\n    {err}")
    print(f"\n{len(rows):,} file(s) failed to index.")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    with open_db(args.db, create=False) as db:
        before = db.stats()["bytes"]
        rows, files = db.purge()
        after = db.stats()["bytes"]
    print(f"removed {rows:,} landmark(s) from {files:,} superseded file "
          f"record(s); {_fmt_bytes(before)} -> {_fmt_bytes(after)}")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-match",
        description="Find the same audio, and the same recording session, "
                    "in a large library of audio files.")
    p.add_argument("--version", action="version",
                   version=f"audio-match {__version__}")
    p.add_argument("--db", default=config.DEFAULT_DB,
                   help=f"index database (default: {config.DEFAULT_DB})")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="scan a library and build the index")
    pi.add_argument("library", help="directory (or single file) to index")
    pi.add_argument("--workers", type=int, default=0,
                    help="worker processes (default: number of CPUs)")
    pi.add_argument("--all-files", action="store_true",
                    help="try every file, not just known audio extensions")
    pi.add_argument("--force", action="store_true",
                    help="re-index even files whose size and mtime are "
                         "unchanged")
    pi.add_argument("--retry-errors", action="store_true",
                    help="retry files that previously failed to decode")
    pi.add_argument("--quiet", action="store_true",
                    help="suppress progress output")
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="search the index with a seed file")
    pq.add_argument("seed", help="audio file to search for")
    pq.add_argument("--mode", choices=("match", "session", "both"),
                    default="both")
    pq.add_argument("--top", type=int, default=10,
                    help="results per mode (default: 10)")
    pq.add_argument("--no-sr-probes", action="store_true",
                    help="skip the 44.1/48 kHz mislabel probes (faster)")
    pq.add_argument("--ignore-filenames", action="store_true",
                    help="score mode 2 on audio evidence only, ignoring "
                         "Tascam take numbers parsed from filenames")
    pq.set_defaults(func=cmd_query)

    ps = sub.add_parser("stats", help="show index statistics")
    ps.set_defaults(func=cmd_stats)

    pe = sub.add_parser("errors", help="list files that failed to index")
    pe.set_defaults(func=cmd_errors)

    pp = sub.add_parser("purge",
                        help="reclaim space from superseded file records")
    pp.set_defaults(func=cmd_purge)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
