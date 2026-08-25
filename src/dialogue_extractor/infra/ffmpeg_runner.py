"""Locating and running ffmpeg / ffprobe.

Discovery is **capability-based, not PATH-order-based**. A machine can easily
have several ffmpeg builds installed, and the first one on PATH is not
necessarily a usable one -- an ancient build will happily answer `-version` and
then fail on `-fps_mode`, which the whole CFR guarantee depends on. So every
candidate is probed by actually running the operations we need, and the first
one that passes is used.

This also means the user never has to reorder their PATH, and the failure mode
when no usable build exists is a clear message naming what was tried.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dialogue_extractor.domain.errors import FFmpegError, FFmpegNotFoundError
from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = 3600.0
_PROBE_TIMEOUT = 30.0

# Cached per (name, configured) so the capability probe runs once per process.
_resolved: dict[tuple[str, str | None], Path] = {}


@dataclass(frozen=True, slots=True)
class BinaryReport:
    """Why a candidate binary was accepted or rejected -- used in error text."""

    path: str
    version: str
    usable: bool
    reason: str


def _run_quiet(
    args: list[str], timeout: float = _PROBE_TIMEOUT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _candidates(name: str, configured: str | None) -> list[Path]:
    """Every plausible binary, most-preferred first, de-duplicated."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(candidate: str | Path | None) -> None:
        if not candidate:
            return
        p = Path(candidate)
        key = str(p).lower()
        if key not in seen and p.exists():
            seen.add(key)
            found.append(p)

    # An explicitly configured path always gets first refusal.
    if configured and configured not in (name,):
        add(shutil.which(configured) or configured)

    # Every match on PATH, not just the first -- that is the whole point.
    exts = os.environ.get("PATHEXT", ".EXE").split(os.pathsep) if os.name == "nt" else [""]
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for ext in exts:
            add(Path(directory) / f"{name}{ext.lower()}")

    return found


def _probe_ffmpeg(path: Path) -> BinaryReport:
    """Reject builds that cannot do what the pipeline actually requires."""
    version_proc = _run_quiet([str(path), "-hide_banner", "-version"])
    if version_proc.returncode != 0:
        return BinaryReport(str(path), "unknown", False, "did not respond to -version")
    version = version_proc.stdout.splitlines()[0] if version_proc.stdout else "unknown"

    encoders = _run_quiet([str(path), "-hide_banner", "-encoders"])
    if "libx264" not in encoders.stdout:
        return BinaryReport(str(path), version, False, "no libx264 encoder")

    # The decisive check. -fps_mode arrived in ffmpeg 5.1; forcing CFR is what
    # makes `timestamp == frame / fps` true, so a build without it is unusable
    # no matter what else it supports.
    cfr = _run_quiet(
        [
            str(path), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=0.1:rate=25",
            "-fps_mode", "cfr", "-c:v", "libx264", "-f", "null", "-",
        ]
    )
    if cfr.returncode != 0:
        return BinaryReport(str(path), version, False, "does not support -fps_mode (needs >= 5.1)")

    return BinaryReport(str(path), version, True, "ok")


def _probe_ffprobe(path: Path) -> BinaryReport:
    proc = _run_quiet([str(path), "-hide_banner", "-version"])
    if proc.returncode != 0:
        return BinaryReport(str(path), "unknown", False, "did not respond to -version")
    version = proc.stdout.splitlines()[0] if proc.stdout else "unknown"

    # Must accept the JSON flags the validator relies on. Probing against a
    # deliberately absent file is enough: an unsupported *flag* is reported
    # differently from an unreadable *input*, and only the former disqualifies
    # the binary.
    probe = _run_quiet(
        [
            str(path), "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            "-i", "__dialogue_extractor_probe_nonexistent__",
        ]
    )
    combined = (probe.stdout + probe.stderr).lower()
    if "unrecognized option" in combined or "option not found" in combined:
        return BinaryReport(str(path), version, False, "does not support -print_format json")

    return BinaryReport(str(path), version, True, "ok")


def resolve_binary(name: str, configured: str | None = None) -> Path:
    """Find a *usable* ffmpeg/ffprobe, probing capabilities rather than trusting PATH.

    Raises FFmpegNotFoundError listing every candidate tried and why each failed.
    """
    cache_key = (name, configured)
    if cache_key in _resolved:
        return _resolved[cache_key]

    reports: list[BinaryReport] = []
    probe = _probe_ffmpeg if name == "ffmpeg" else _probe_ffprobe

    for candidate in _candidates(name, configured):
        try:
            report = probe(candidate)
        except (subprocess.TimeoutExpired, OSError) as exc:
            report = BinaryReport(str(candidate), "unknown", False, f"{type(exc).__name__}")
        reports.append(report)
        if report.usable:
            log.info("ffmpeg.resolved", binary=name, path=report.path, version=report.version)
            _resolved[cache_key] = candidate
            return candidate
        log.debug("ffmpeg.rejected", binary=name, path=report.path, reason=report.reason)

    tried = "; ".join(f"{r.path} ({r.reason})" for r in reports) or "nothing on PATH"
    raise FFmpegNotFoundError(
        f"{name}_not_usable",
        f"No usable {name} found. Tried: {tried}",
        remediation=(
            f"Install a current FFmpeg (5.1 or newer) and make sure {name} is on PATH. "
            "On Windows: winget install --id=Gyan.FFmpeg -e"
        ),
    )


def reset_binary_cache() -> None:
    """Drop resolution caching. Used by tests."""
    _resolved.clear()


def run_ffmpeg(
    args: list[str],
    *,
    binary: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Run ffmpeg with `args`, returning stderr (where ffmpeg reports progress).

    Raises FFmpegError carrying the tail of stderr -- ffmpeg's own diagnostics
    are far more useful than a generic non-zero-exit message.
    """
    exe = resolve_binary("ffmpeg", binary)
    cmd = [str(exe), "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg.run", cmd=" ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    try:
        proc = _run_quiet(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            "ffmpeg_timeout", f"ffmpeg exceeded {timeout:.0f}s and was killed."
        ) from exc

    if proc.returncode != 0:
        raise FFmpegError(
            "ffmpeg_failed",
            f"ffmpeg exited {proc.returncode}: {proc.stderr.strip()[-2000:]}",
        )
    return proc.stderr


def run_ffprobe(args: list[str], *, binary: str | None = None, timeout: float = 120.0) -> str:
    exe = resolve_binary("ffprobe", binary)
    cmd = [str(exe), "-hide_banner", *args]
    try:
        proc = _run_quiet(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            "ffprobe_timeout", f"ffprobe exceeded {timeout:.0f}s and was killed."
        ) from exc

    if proc.returncode != 0:
        raise FFmpegError(
            "ffprobe_failed",
            f"ffprobe exited {proc.returncode}: {proc.stderr.strip()[-2000:]}",
        )
    return proc.stdout


def probe_json(media_path: Path, *, binary: str | None = None) -> dict[str, Any]:
    """Full ffprobe stream+format dump as a dict."""
    raw = run_ffprobe(
        [
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(media_path),
        ],
        binary=binary,
    )
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FFmpegError(
            "ffprobe_bad_json", f"ffprobe returned unparseable JSON for {media_path.name}."
        ) from exc
    return parsed
