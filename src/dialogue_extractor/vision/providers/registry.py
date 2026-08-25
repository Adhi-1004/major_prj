"""Selecting an active-speaker-detection backend by name.

The indirection is what makes the modularity claim concrete: the orchestrator
asks for a provider by name and receives something satisfying the ASDProvider
port, with no knowledge of which one answered.
"""

from __future__ import annotations

from collections.abc import Callable

from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import ConfigError
from dialogue_extractor.interfaces.ports import ASDProvider
from dialogue_extractor.vision.providers.light_asd import LightASDProvider
from dialogue_extractor.vision.providers.nvidia_asd import NvidiaASDProvider


def _build_nvidia(settings: Settings) -> ASDProvider:
    return NvidiaASDProvider(
        api_key=settings.require_nvidia_api_key(),
        function_id=settings.require_asd_function_id(),
        target=settings.asd_target,
    )


def _build_light(settings: Settings) -> ASDProvider:
    return LightASDProvider()


BUILDERS: dict[str, Callable[[Settings], ASDProvider]] = {
    "nvidia": _build_nvidia,
    "light-asd": _build_light,
    "light": _build_light,
}


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(BUILDERS))


def get_provider(name: str, settings: Settings) -> ASDProvider:
    """Build the named provider.

    Credentials are resolved here, so a missing key fails with a remediation
    string at selection time rather than midway through a long run.
    """
    builder = BUILDERS.get(name.strip().lower())
    if builder is None:
        raise ConfigError(
            "unknown_asd_provider",
            f"No active-speaker-detection provider named {name!r}.",
            remediation=f"Available: {', '.join(available_providers())}",
        )
    return builder(settings)
