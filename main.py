"""Command-line entry point.

Thin by design: parse arguments, build settings, hand off to the composition
root, print what came back. Every decision that matters lives behind
`Orchestrator.run`, so the CLI and the HTTP API cannot answer the same question
differently.

    python main.py --url URL --dialogue "the line to find"
    python main.py --url URL --max-candidates 10

Exit codes:

    0   a frame was produced (success, or ambiguous with its reason stated)
    2   the pipeline ran and answered "no": not_found, or not_on_screen
    1   the pipeline failed, and a structured error document says where

2 is separated from 1 deliberately. "This line is spoken off camera" is a
correct answer that a script may well want to act on, and collapsing it into the
same code as a crashed download would make the two indistinguishable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_extractor.composition import build_orchestrator
from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.enums import Status
from dialogue_extractor.infra.logging import configure_logging
from dialogue_extractor.pipeline.orchestrator import PipelineOutcome, PipelineRequest
from dialogue_extractor.vision.providers.registry import available_providers

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ANSWERED_NO = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-extractor",
        description="Find the exact frame where a line of dialogue is spoken on screen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --url URL --dialogue \"my mind rebels at stagnation\"\n"
            "  python main.py --url URL --max-candidates 10\n"
            "  python main.py --url ./local/file.mp4 --dialogue \"...\"\n"
        ),
    )

    parser.add_argument(
        "--url",
        required=True,
        help="Public video URL, or a path to a local file.",
    )
    parser.add_argument(
        "--dialogue",
        default=None,
        help=(
            "The line to find. Supplying it selects target mode; omitting it "
            "sweeps the whole video for every on-screen line."
        ),
    )

    budget = parser.add_argument_group("discovery budget (ignored in target mode)")
    budget.add_argument(
        "--max-candidates",
        type=int,
        default=constants.DEFAULT_MAX_CANDIDATES,
        help="Stop after this many candidates. 0 means no limit.",
    )
    budget.add_argument(
        "--time-budget",
        type=float,
        default=constants.DEFAULT_TIME_BUDGET_SECONDS,
        help="Stop after this many seconds. 0 means no limit.",
    )

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument(
        "--match-floor",
        type=float,
        default=constants.MATCH_SCORE_FLOOR,
        help="Fuzzy match score below which a transcript hit is not a candidate.",
    )
    tuning.add_argument(
        "--pad",
        type=float,
        default=constants.WINDOW_PAD_SECONDS,
        help="Seconds of padding around a matched window, absorbing ASR jitter.",
    )
    tuning.add_argument(
        "--min-speaking-ms",
        type=float,
        default=constants.MIN_SPEAKING_MS,
        help=(
            "Shortest run of speaking frames that counts. Converted to a frame "
            "count using the video's own fps, so the rule means the same thing "
            "at 24 and at 60."
        ),
    )
    tuning.add_argument(
        "--no-scene-cuts",
        action="store_true",
        help="Do not split candidate windows at scene cuts.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--asd-provider",
        default=None,
        choices=available_providers(),
        help="Active-speaker-detection backend.",
    )
    runtime.add_argument("--output-dir", type=Path, default=None)
    runtime.add_argument("--work-dir", type=Path, default=None)
    runtime.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the result JSON. Defaults to <output-dir>/result.json.",
    )
    runtime.add_argument(
        "--device",
        default=None,
        choices=("auto", "cuda", "cpu"),
        help="Compute device for transcription and diarization.",
    )
    runtime.add_argument(
        "--dns",
        default="auto",
        choices=("auto", "doh", "system"),
        help=(
            "DNS strategy for downloads. auto falls back to DNS-over-HTTPS when "
            "interference is detected."
        ),
    )
    runtime.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached transcripts and diarization, and write none.",
    )
    runtime.add_argument("--log-level", default=None)
    runtime.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the result JSON, nothing else.",
    )
    return parser


def settings_from(args: argparse.Namespace) -> Settings:
    """Layer the CLI flags over the environment.

    Only flags the user actually passed are forwarded: an unset flag must leave
    the environment value standing rather than overwrite it with a default.
    """
    overrides: dict[str, object] = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.work_dir is not None:
        overrides["work_dir"] = args.work_dir
    if args.device is not None:
        overrides["device"] = args.device
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.asd_provider is not None:
        overrides["asd_provider"] = args.asd_provider
    if args.no_cache:
        overrides["use_cache"] = False
    return Settings(**overrides)  # type: ignore[arg-type]


def exit_code_for(outcome: PipelineOutcome) -> int:
    if outcome.ok:
        return EXIT_OK
    if outcome.status == Status.ERROR.value:
        return EXIT_FAILED
    return EXIT_ANSWERED_NO


def report(
    outcome: PipelineOutcome, *, quiet: bool, out_path: Path, settings: Settings
) -> None:
    """Print the result, and say where the frame image landed."""
    document = json.dumps(outcome.payload, indent=2)

    if quiet:
        print(document)
    else:
        if outcome.diagnostics:
            print("\n--- pipeline ---")
            for key, value in outcome.diagnostics.items():
                print(f"  {key:<20} {value}")
        print("\n--- result ---")
        print(document)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    if not quiet:
        print(f"\nwritten to {out_path}")
        image = outcome.payload.get("image")
        if isinstance(image, str):
            print(f"frame image: {settings.output_dir / 'frames' / image}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = settings_from(args)
    except Exception as exc:  # configuration is not yet inside the boundary
        print(
            json.dumps(
                {
                    "status": Status.ERROR.value,
                    "stage": "config",
                    "reason": "invalid_configuration",
                    "detail": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return EXIT_FAILED

    configure_logging("ERROR" if args.quiet else settings.log_level)

    request = PipelineRequest(
        url=args.url,
        dialogue=args.dialogue,
        max_candidates=args.max_candidates,
        time_budget_seconds=args.time_budget,
        match_score_floor=args.match_floor,
        window_pad_seconds=args.pad,
        min_speaking_ms=args.min_speaking_ms,
        detect_scene_cuts=not args.no_scene_cuts,
    )

    orchestrator = build_orchestrator(
        settings, provider_name=args.asd_provider, dns_mode=args.dns
    )
    outcome = orchestrator.run(request)

    out_path = args.out or (settings.output_dir / "result.json")
    report(outcome, quiet=args.quiet, out_path=out_path, settings=settings)
    return exit_code_for(outcome)


if __name__ == "__main__":
    raise SystemExit(main())
