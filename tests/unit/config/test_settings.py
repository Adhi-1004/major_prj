"""Settings precedence, credential handling, and secret hygiene."""

from __future__ import annotations

import pytest

from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings, get_settings, reset_settings
from dialogue_extractor.domain.errors import MissingCredentialError

ENV_KEYS = [
    "NVIDIA_API_KEY",
    "ASD_FUNCTION_ID",
    "ASD_TARGET",
    "HF_TOKEN",
    "DEVICE",
    "LOG_LEVEL",
    "WHISPER_MODEL",
    "USE_CACHE",
    "MIN_SPEAKING_MS",
    "LANGUAGE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """Isolate from the developer's real .env and exported variables."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env to discover here
    reset_settings()
    yield
    reset_settings()


def test_defaults_apply_when_nothing_is_set() -> None:
    settings = Settings()
    assert settings.device == "auto"
    assert settings.asd_provider == "nvidia"
    assert settings.whisper_model == constants.DEFAULT_WHISPER_MODEL
    assert settings.min_speaking_ms == constants.MIN_SPEAKING_MS


def test_environment_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("DEVICE", "cpu")
    settings = Settings()
    assert settings.whisper_model == "medium"
    assert settings.device == "cpu"


def test_explicit_argument_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """This is how the CLI layers its flags on top of .env."""
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    assert Settings(whisper_model="large-v3").whisper_model == "large-v3"


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMETHING_UNRELATED", "x")
    Settings()  # must not raise


class TestLogLevel:
    @pytest.mark.parametrize("value", ["debug", "Info", "WARNING"])
    def test_is_normalized_to_upper(self, value: str) -> None:
        assert Settings(log_level=value).log_level == value.upper()

    def test_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError):
            Settings(log_level="LOUD")


class TestDeviceChoice:
    @pytest.mark.parametrize("value", ["auto", "cuda", "cpu"])
    def test_accepts_known_devices(self, value: str) -> None:
        assert Settings(device=value).device == value  # type: ignore[arg-type]

    def test_rejects_unknown_device(self) -> None:
        with pytest.raises(ValueError):
            Settings(device="tpu")  # type: ignore[arg-type]


class TestLanguage:
    """Any video, not only English: the language is configurable."""

    def test_english_is_the_default(self) -> None:
        assert Settings().whisper_language() == "en"

    @pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", ""])
    def test_auto_means_detect(self, value: str) -> None:
        assert Settings(language=value).whisper_language() is None

    def test_explicit_code_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGUAGE", "Hi")
        assert Settings().whisper_language() == "hi"


class TestRequiredCredentials:
    """A missing key must fail with something the user can act on, not a None
    propagating into a gRPC call and surfacing as an opaque auth error."""

    def test_nvidia_key_missing_raises_with_remediation(self) -> None:
        with pytest.raises(MissingCredentialError) as excinfo:
            Settings().require_nvidia_api_key()
        payload = excinfo.value.to_dict()
        assert payload["reason"] == "missing_nvidia_api_key"
        assert "build.nvidia.com" in payload["remediation"]

    def test_empty_string_counts_as_missing(self) -> None:
        with pytest.raises(MissingCredentialError):
            Settings(nvidia_api_key="").require_nvidia_api_key()

    def test_function_id_missing_raises_with_remediation(self) -> None:
        with pytest.raises(MissingCredentialError) as excinfo:
            Settings().require_asd_function_id()
        assert excinfo.value.reason == "missing_asd_function_id"

    def test_hf_token_missing_names_the_gated_repo(self) -> None:
        """The most common setup failure is having a token but not having
        accepted the gated repo's terms, so the remediation must say so."""
        with pytest.raises(MissingCredentialError) as excinfo:
            Settings().require_hf_token()
        remediation = excinfo.value.to_dict()["remediation"]
        assert constants.PYANNOTE_PIPELINE in remediation
        assert "accept the terms" in remediation.lower()

    def test_present_credentials_are_returned(self) -> None:
        settings = Settings(nvidia_api_key="nvapi-abc", asd_function_id="fid", hf_token="hf_xyz")
        assert settings.require_nvidia_api_key() == "nvapi-abc"
        assert settings.require_asd_function_id() == "fid"
        assert settings.require_hf_token() == "hf_xyz"


class TestSecretHygiene:
    def test_secrets_are_not_in_repr(self) -> None:
        settings = Settings(nvidia_api_key="nvapi-supersecret", hf_token="hf_supersecret")
        assert "supersecret" not in repr(settings)

    def test_safe_dump_redacts_secrets(self) -> None:
        dumped = Settings(nvidia_api_key="nvapi-supersecret", hf_token="hf_x").safe_dump()
        assert dumped["nvidia_api_key"] == "***set***"
        assert dumped["hf_token"] == "***set***"
        assert "supersecret" not in str(dumped)

    def test_safe_dump_distinguishes_unset_from_set(self) -> None:
        dumped = Settings(nvidia_api_key="nvapi-x").safe_dump()
        assert dumped["nvidia_api_key"] == "***set***"
        assert dumped["hf_token"] is None

    def test_safe_dump_is_json_serializable(self) -> None:
        import json

        json.dumps(Settings(nvidia_api_key="nvapi-x").safe_dump())


class TestSingleton:
    def test_returns_the_same_instance(self) -> None:
        assert get_settings() is get_settings()

    def test_overrides_rebuild_it(self) -> None:
        first = get_settings()
        second = get_settings(whisper_model="tiny")
        assert second is not first
        assert second.whisper_model == "tiny"

    def test_reset_clears_it(self) -> None:
        first = get_settings()
        reset_settings()
        assert get_settings() is not first


def test_ensure_dirs_creates_them(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "out" / "work",
        cache_dir=tmp_path / "cache",
    )
    settings.ensure_dirs()
    assert (tmp_path / "out" / "work").is_dir()
    assert (tmp_path / "cache").is_dir()


def test_ensure_dirs_is_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        output_dir=tmp_path / "o", work_dir=tmp_path / "w", cache_dir=tmp_path / "c"
    )
    settings.ensure_dirs()
    settings.ensure_dirs()  # must not raise
