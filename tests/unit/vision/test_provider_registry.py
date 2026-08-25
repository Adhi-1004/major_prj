"""Provider selection.

The registry is what makes "the detection backend is swappable" checkable
rather than merely asserted: the pipeline names a provider and receives
something satisfying the port, knowing nothing about which one answered.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import ConfigError, MissingCredentialError
from dialogue_extractor.interfaces.ports import ASDProvider
from dialogue_extractor.vision.providers.light_asd import LightASDProvider
from dialogue_extractor.vision.providers.nvidia_asd import NvidiaASDProvider
from dialogue_extractor.vision.providers.registry import (
    available_providers,
    get_provider,
)


def isolated(**overrides: object) -> Settings:
    """Settings built without reading the developer's real .env.

    Without `_env_file=None` these tests pass or fail depending on whether the
    machine running them happens to have credentials configured -- which is the
    opposite of what a test asserting "missing credential" should depend on.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def configured() -> Settings:
    return isolated(nvidia_api_key="nvapi-x", asd_function_id="fid-1")


class TestSelection:
    def test_nvidia_by_name(self) -> None:
        assert isinstance(get_provider("nvidia", configured()), NvidiaASDProvider)

    @pytest.mark.parametrize("name", ["light-asd", "light"])
    def test_light_by_name(self, name: str) -> None:
        assert isinstance(get_provider(name, configured()), LightASDProvider)

    @pytest.mark.parametrize("name", ["NVIDIA", " nvidia ", "Nvidia"])
    def test_name_matching_is_forgiving(self, name: str) -> None:
        assert isinstance(get_provider(name, configured()), NvidiaASDProvider)

    def test_unknown_name_lists_the_options(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            get_provider("magic", configured())
        assert excinfo.value.reason == "unknown_asd_provider"
        assert "nvidia" in excinfo.value.to_dict()["remediation"]

    def test_every_advertised_provider_can_be_built(self) -> None:
        for name in available_providers():
            assert isinstance(get_provider(name, configured()), ASDProvider)


class TestSubstitutability:
    def test_both_satisfy_the_port(self) -> None:
        """The pipeline must not be able to tell them apart by type."""
        for name in ("nvidia", "light-asd"):
            assert isinstance(get_provider(name, configured()), ASDProvider)

    def test_each_reports_its_own_name(self) -> None:
        assert get_provider("nvidia", configured()).name == "nvidia"
        assert get_provider("light-asd", configured()).name == "light-asd"


class TestCredentialsResolveAtSelection:
    def test_missing_key_fails_immediately(self) -> None:
        """Better here than midway through a long run."""
        with pytest.raises(MissingCredentialError) as excinfo:
            get_provider("nvidia", isolated(asd_function_id="fid"))
        assert excinfo.value.reason == "missing_nvidia_api_key"

    def test_missing_function_id_fails_immediately(self) -> None:
        with pytest.raises(MissingCredentialError) as excinfo:
            get_provider("nvidia", isolated(nvidia_api_key="nvapi-x"))
        assert excinfo.value.reason == "missing_asd_function_id"

    def test_light_needs_no_credentials(self) -> None:
        get_provider("light-asd", isolated())  # must not raise


class TestLightAsdIsHonestlyUnimplemented:
    def test_health_check_says_so(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            LightASDProvider().health_check()
        assert "not implemented" in str(excinfo.value).lower()

    def test_detect_says_so(self) -> None:
        from pathlib import Path

        with pytest.raises(NotImplementedError):
            LightASDProvider().detect(Path("x.mp4"), None, {}, 0, 25.0)

    def test_message_points_at_the_working_backend(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            LightASDProvider().health_check()
        assert "--asd-provider nvidia" in str(excinfo.value)
