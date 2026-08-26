"""The wiring between the abstract pipeline and its concrete parts.

Two things are worth asserting and neither shows up in a unit test of any single
module. First, that every provider and mode combination actually constructs --
a typo in a factory is invisible until a run that has already spent five minutes
downloading and transcribing. Second, that the expensive imports really are
deferred: the claim that `--help` and `/health` do not pay for torch is easy to
break with one convenient top-level import, and nothing else would notice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dialogue_extractor.composition import (
    FfmpegNormalizer,
    build_cache,
    build_stages,
    source_for,
)
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import ConfigError, MissingCredentialError
from dialogue_extractor.infra.disk_cache import DiskCache, NullCache
from dialogue_extractor.ingestion.local_file_source import LocalFileSource
from dialogue_extractor.ingestion.ytdlp_source import YtDlpSource
from dialogue_extractor.interfaces.ports import MediaNormalizer
from dialogue_extractor.pipeline.orchestrator import Orchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",
        asd_function_id="00000000-0000-0000-0000-000000000000",
        hf_token="hf_test",
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
    )


class TestStagesConstruct:
    def test_every_dependency_is_supplied(self, settings: Settings) -> None:
        stages = build_stages(settings)
        assert stages.source_for is not None
        assert isinstance(stages.normalizer, MediaNormalizer)
        assert callable(stages.transcriber)
        assert callable(stages.diarizer)
        assert callable(stages.asd)
        assert stages.extractor is not None

    def test_an_orchestrator_can_be_built(self, settings: Settings) -> None:
        from dialogue_extractor.composition import build_orchestrator

        assert isinstance(build_orchestrator(settings), Orchestrator)

    @pytest.mark.parametrize("provider", ["nvidia", "light", "light-asd"])
    def test_every_provider_name_is_wired(
        self, provider: str, settings: Settings
    ) -> None:
        """A typo here would surface only after a five-minute transcription."""
        stages = build_stages(settings, provider_name=provider)
        assert stages.asd().name

    def test_the_unimplemented_backend_is_still_selectable(
        self, settings: Settings
    ) -> None:
        """It constructs and fails on *use*, which is the honest arrangement.

        Refusing at construction would make `--asd-provider light` a config
        error rather than a working seam, and the modularity claim rests on the
        seam being real. The message names what implementing it would take.
        """
        provider = build_stages(settings, provider_name="light").asd()
        with pytest.raises(NotImplementedError) as excinfo:
            provider.health_check()
        assert "Light-ASD" in str(excinfo.value)

    def test_an_unknown_provider_is_a_config_error(self, settings: Settings) -> None:
        with pytest.raises(ConfigError):
            build_stages(settings, provider_name="nonesuch").asd()

    def test_a_missing_key_fails_with_remediation_not_none(
        self, tmp_path: Path
    ) -> None:
        """Better to fail at selection than midway through a long run."""
        bare = Settings(_env_file=None, output_dir=tmp_path)  # type: ignore[call-arg]
        with pytest.raises(MissingCredentialError) as excinfo:
            build_stages(bare, provider_name="nvidia").asd()
        assert excinfo.value.remediation


class TestSourceSelection:
    @pytest.mark.parametrize(
        "uri",
        [
            "https://www.youtube.com/watch?v=abc",
            "https://ok.ru/video/248244667877",
            "http://example.com/v.mp4",
        ],
    )
    def test_remote_urls_get_the_downloader(self, uri: str) -> None:
        assert isinstance(source_for(uri), YtDlpSource)

    @pytest.mark.parametrize(
        "uri", ["C:/videos/clip.mp4", "./clip.mp4", "file:///tmp/clip.mp4"]
    )
    def test_local_paths_get_the_file_source(self, uri: str) -> None:
        assert isinstance(source_for(uri), LocalFileSource)

    def test_classification_happens_per_uri(self) -> None:
        """One process may be asked for a URL and a path in turn."""
        assert type(source_for("https://e.com/v")) is not type(source_for("./x.mp4"))


class TestCacheSelection:
    def test_enabled_by_default(self, settings: Settings) -> None:
        assert isinstance(build_cache(settings), DiskCache)

    def test_disabled_gives_a_cache_that_forgets(self, tmp_path: Path) -> None:
        off = Settings(_env_file=None, use_cache=False, cache_dir=tmp_path)  # type: ignore[call-arg]
        cache = build_cache(off)
        assert isinstance(cache, NullCache)
        cache.put("k", {"v": 1})
        assert cache.get("k") is None


class TestNormalizerAdapter:
    def test_it_satisfies_the_port(self, settings: Settings) -> None:
        assert isinstance(FfmpegNormalizer(settings), MediaNormalizer)

    def test_it_carries_the_configured_binaries(self, tmp_path: Path) -> None:
        custom = Settings(
            _env_file=None,  # type: ignore[call-arg]
            ffmpeg_path="/opt/ffmpeg",
            ffprobe_path="/opt/ffprobe",
            output_dir=tmp_path,
        )
        adapter = FfmpegNormalizer(custom)
        assert adapter._ffmpeg == "/opt/ffmpeg"
        assert adapter._ffprobe == "/opt/ffprobe"


class TestHeavyImportsStayDeferred:
    """The claim that costs the most if it quietly stops being true.

    One convenient top-level `import torch` in a composition module would add
    seconds and gigabytes to every `--help`, and nothing else in the suite would
    notice. Run in a subprocess because the test session has almost certainly
    imported torch already.
    """

    @pytest.mark.parametrize(
        "module",
        [
            "dialogue_extractor.composition",
            "dialogue_extractor.pipeline.orchestrator",
            "dialogue_extractor.api.app",
            "main",
        ],
    )
    def test_importing_it_does_not_pull_in_torch(self, module: str) -> None:
        code = (
            "import sys;"
            f"import {module};"
            "heavy=[m for m in ('torch','faster_whisper','pyannote') if m in sys.modules];"
            "print(','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "", f"{module} pulled in {result.stdout.strip()}"


class TestCliEntryPoint:
    def test_help_works_without_credentials_or_models(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "--dialogue" in result.stdout
        assert "--max-candidates" in result.stdout

    def test_a_missing_url_is_refused_by_the_parser(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 2
        assert "--url" in result.stderr

    def test_a_bad_url_exits_with_a_structured_document(self, tmp_path: Path) -> None:
        """No traceback reaches stdout, whatever goes wrong."""
        import json

        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--url",
                "javascript:alert(1)",
                "--output-dir",
                str(tmp_path),
                "--quiet",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 1
        assert "Traceback" not in result.stdout
        document = json.loads(result.stdout)
        assert document["status"] == "error"
        assert document["stage"] == "ingestion"


class TestExitCodes:
    """0 / 1 / 2 are a contract a calling script can act on."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("success", 0),
            ("ambiguous", 0),
            ("not_on_screen", 2),
            ("not_found", 2),
            ("error", 1),
        ],
    )
    def test_a_correct_no_is_not_a_failure(self, status: str, expected: int) -> None:
        from main import exit_code_for

        from dialogue_extractor.pipeline.orchestrator import PipelineOutcome

        assert exit_code_for(PipelineOutcome(payload={"status": status})) == expected
