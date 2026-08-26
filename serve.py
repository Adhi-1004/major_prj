"""HTTP entry point.

    python serve.py --port 8000

Separate from `main.py` so the CLI stays usable on a machine with no server
dependencies resolved, and so the container has one obvious command to run.
"""

from __future__ import annotations

import argparse

from dialogue_extractor.api.app import create_app
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.infra.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-extractor-api",
        description="Serve the dialogue frame extractor over HTTP.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default=None)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes. Development only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides: dict[str, object] = {}
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging(settings.log_level)

    import uvicorn

    if args.reload:
        # Reload needs an import string rather than an instance, since the
        # worker is a fresh process that has to build the app itself.
        uvicorn.run(
            "dialogue_extractor.api.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
    else:
        uvicorn.run(
            create_app(settings),
            host=args.host,
            port=args.port,
            log_level=settings.log_level.lower(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
