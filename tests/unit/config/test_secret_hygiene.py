"""Secrets must not reach a log, an error message, or the output directory.

Three real API credentials pass through this system. The failure this guards
against is not dramatic -- nobody prints a key on purpose -- it is a key riding
along inside something else that gets printed: a settings dump at startup, an
exception message that interpolated its arguments, a result file committed to a
repository. Each of those has happened in real projects, so each is tested.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import MissingCredentialError

REPO_ROOT = Path(__file__).resolve().parents[3]

FAKE_NVIDIA = "nvapi-0123456789abcdefghijklmnopqrstuvwxyzABCD"
FAKE_HF = "hf_0123456789abcdefghijklmnopqrstuvwx"


@pytest.fixture
def loaded(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key=FAKE_NVIDIA,
        hf_token=FAKE_HF,
        asd_function_id="fn-1234",
        output_dir=tmp_path,
    )


class TestSecretsAreOpaqueByDefault:
    def test_repr_does_not_reveal_the_key(self, loaded: Settings) -> None:
        """The accidental path: something logs the settings object."""
        assert FAKE_NVIDIA not in repr(loaded)
        assert FAKE_HF not in repr(loaded)

    def test_str_does_not_reveal_the_key(self, loaded: Settings) -> None:
        assert FAKE_NVIDIA not in str(loaded)

    def test_the_field_itself_is_opaque(self, loaded: Settings) -> None:
        assert loaded.nvidia_api_key is not None
        assert FAKE_NVIDIA not in str(loaded.nvidia_api_key)

    def test_reading_it_deliberately_still_works(self, loaded: Settings) -> None:
        """Opaque by accident, readable on purpose -- that is the whole point."""
        assert loaded.require_nvidia_api_key() == FAKE_NVIDIA


class TestSafeDump:
    def test_it_marks_secrets_as_set_without_naming_them(
        self, loaded: Settings
    ) -> None:
        dumped = loaded.safe_dump()
        assert dumped["nvidia_api_key"] == "***set***"
        assert dumped["hf_token"] == "***set***"
        assert FAKE_NVIDIA not in json.dumps(dumped)
        assert FAKE_HF not in json.dumps(dumped)

    def test_absent_secrets_read_as_none_not_as_set(self, tmp_path: Path) -> None:
        """"Set to nothing" and "not set" are different diagnoses."""
        bare = Settings(_env_file=None, output_dir=tmp_path)  # type: ignore[call-arg]
        assert bare.safe_dump()["nvidia_api_key"] is None

    def test_it_is_json_serializable(self, loaded: Settings) -> None:
        """A dump that cannot be logged as JSON invites logging the raw object."""
        assert json.loads(json.dumps(loaded.safe_dump()))["device"] == "auto"

    def test_non_secret_config_survives(self, loaded: Settings) -> None:
        assert loaded.safe_dump()["asd_function_id"] == "fn-1234"


class TestErrorsDoNotCarrySecrets:
    def test_a_missing_credential_error_names_the_variable_not_a_value(self) -> None:
        error = MissingCredentialError(
            "missing_nvidia_api_key", "NVIDIA_API_KEY is not set."
        )
        document = json.dumps(error.to_dict())
        assert "NVIDIA_API_KEY" in document
        assert "nvapi-" not in document

    def test_the_remediation_points_at_where_to_get_one(self, tmp_path: Path) -> None:
        bare = Settings(_env_file=None, output_dir=tmp_path)  # type: ignore[call-arg]
        with pytest.raises(MissingCredentialError) as excinfo:
            bare.require_nvidia_api_key()
        assert "build.nvidia.com" in (excinfo.value.remediation or "")

    def test_an_unexpected_pipeline_error_does_not_echo_its_message(
        self, tmp_path: Path
    ) -> None:
        """The orchestrator reports the exception *type*, never its text.

        A library that interpolates its arguments into an exception message is
        the realistic way a key ends up in a response body.
        """
        from tests.unit.pipeline.test_error_boundary import REQUEST, build

        outcome = build(
            transcriber=RuntimeError(f"auth failed for {FAKE_NVIDIA}"),
            tmp_path=tmp_path,
        ).run(REQUEST)

        assert FAKE_NVIDIA not in json.dumps(outcome.payload)
        assert "RuntimeError" in json.dumps(outcome.payload)


class TestTheRepositoryIsClean:
    """A committed key is the failure that survives every other precaution."""

    PATTERNS = (
        re.compile(r"hf_[A-Za-z0-9]{34}"),
        re.compile(r"nvapi-[A-Za-z0-9_-]{40,}"),
        re.compile(r"sk-[A-Za-z0-9]{32,}"),
    )

    def _looks_like_a_placeholder(self, text: str) -> bool:
        lowered = text.lower()
        return "x" * 8 in lowered or "0123456789abcdef" in lowered

    def test_no_tracked_file_contains_a_real_looking_key(self) -> None:
        import subprocess

        listing = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if listing.returncode != 0:
            pytest.skip("not a git repository")

        offenders: list[str] = []
        for name in listing.stdout.splitlines():
            path = REPO_ROOT / name
            if not path.is_file() or path.suffix in {".png", ".jpg", ".mp4", ".wav"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in self.PATTERNS:
                for hit in pattern.findall(text):
                    if not self._looks_like_a_placeholder(hit):
                        offenders.append(f"{name}: {hit[:12]}...")

        assert not offenders, f"possible secrets committed: {offenders}"

    def test_the_env_file_is_ignored(self) -> None:
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in gitignore

    def test_the_example_env_holds_only_placeholders(self) -> None:
        example = REPO_ROOT / ".env.example"
        if not example.is_file():
            pytest.skip("no .env.example")
        text = example.read_text(encoding="utf-8")
        for pattern in self.PATTERNS:
            for hit in pattern.findall(text):
                assert self._looks_like_a_placeholder(hit), hit
