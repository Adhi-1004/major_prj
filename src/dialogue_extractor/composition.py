"""Where the abstract pipeline meets its concrete parts.

The orchestrator names only ports. This is the one module allowed to know which
implementation satisfies each of them, so that "swap the ASD backend" or "run
without a cache" is a change here rather than a change threaded through the
pipeline. Both entry points -- the CLI and the API -- build their dependencies
from this, which is what stops the two from drifting into different pipelines.

Three of the factories import lazily, inside the function body. `torch`,
`faster_whisper` and `pyannote.audio` cost seconds and gigabytes to import, and
an API process answering /health, or a CLI run that fails URL validation, should
pay neither.
"""

from __future__ import annotations

from pathlib import Path

from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.models import MediaManifest, RawMedia
from dialogue_extractor.infra.disk_cache import DiskCache, NullCache
from dialogue_extractor.ingestion.local_file_source import LocalFileSource
from dialogue_extractor.ingestion.normalizer import normalize
from dialogue_extractor.ingestion.url_validator import SourceKind, classify_source
from dialogue_extractor.ingestion.ytdlp_source import YtDlpSource
from dialogue_extractor.interfaces.ports import (
    ASDProvider,
    Cache,
    Diarizer,
    MediaNormalizer,
    Transcriber,
    VideoSource,
)
from dialogue_extractor.pipeline.orchestrator import Orchestrator, Stages
from dialogue_extractor.vision.providers.registry import get_provider
from dialogue_extractor.vision.pyav_extractor import PyAVFrameExtractor


class FfmpegNormalizer(MediaNormalizer):
    """Adapts the `normalize` function to the port.

    The function came first and is used directly by the stage-check tools; this
    exists so the orchestrator can depend on the interface rather than on a
    module-level function, which is what makes the stage substitutable in tests.
    """

    def __init__(self, settings: Settings) -> None:
        self._ffmpeg = settings.ffmpeg_path
        self._ffprobe = settings.ffprobe_path

    def normalize(self, raw: RawMedia, work_dir: Path) -> MediaManifest:
        return normalize(
            raw,
            work_dir,
            ffmpeg_binary=self._ffmpeg,
            ffprobe_binary=self._ffprobe,
        )


def build_cache(settings: Settings) -> Cache:
    """A real cache, or one that forgets everything, by configuration."""
    if not settings.use_cache:
        return NullCache()
    return DiskCache(settings.cache_dir)


def source_for(uri: str, *, dns_mode: str = "auto") -> VideoSource:
    """Pick the acquisition strategy for one URI.

    Classification happens per request rather than once at startup because a
    single process may be asked for a remote URL and a local path in turn.
    """
    if classify_source(uri) is SourceKind.LOCAL:
        return LocalFileSource()
    return YtDlpSource(dns_mode=dns_mode)


def build_transcriber(settings: Settings, cache: Cache) -> Transcriber:
    """A caching Whisper transcriber.

    The fingerprint has to name every setting that changes the output, not just
    the model: quantization alone moved the word count by more than a dozen on
    the sample video, and a key that ignored it would serve one run's transcript
    to a differently-configured one.
    """
    from dialogue_extractor.audio.cached_transcriber import (
        CachedTranscriber,
        whisper_fingerprint,
    )
    from dialogue_extractor.audio.whisper_transcriber import WhisperTranscriber

    inner = WhisperTranscriber(settings.whisper_model, device=settings.device)
    return CachedTranscriber(
        inner,
        cache,
        fingerprint=whisper_fingerprint(
            settings.whisper_model,
            None,
            constants.WHISPER_BEAM_SIZE,
            inner.compute_type,
        ),
    )


def build_diarizer(settings: Settings, cache: Cache) -> Diarizer:
    """A caching pyannote diarizer."""
    from dialogue_extractor.audio.cached_diarizer import (
        CachedDiarizer,
        pyannote_fingerprint,
    )
    from dialogue_extractor.audio.pyannote_diarizer import PyannoteDiarizer

    return CachedDiarizer(
        PyannoteDiarizer(settings.require_hf_token(), device=settings.device),
        cache,
        fingerprint=pyannote_fingerprint(constants.PYANNOTE_PIPELINE),
    )


def build_asd(settings: Settings, provider_name: str | None = None) -> ASDProvider:
    return get_provider(provider_name or settings.asd_provider, settings)


def build_stages(
    settings: Settings,
    *,
    provider_name: str | None = None,
    dns_mode: str = "auto",
) -> Stages:
    """Assemble every dependency the orchestrator needs."""
    cache = build_cache(settings)
    return Stages(
        source_for=lambda uri: source_for(uri, dns_mode=dns_mode),
        normalizer=FfmpegNormalizer(settings),
        transcriber=lambda: build_transcriber(settings, cache),
        diarizer=lambda: build_diarizer(settings, cache),
        asd=lambda: build_asd(settings, provider_name),
        extractor=PyAVFrameExtractor(),
    )


def build_orchestrator(
    settings: Settings,
    *,
    provider_name: str | None = None,
    dns_mode: str = "auto",
) -> Orchestrator:
    """The composition root proper: one call, one ready pipeline."""
    return Orchestrator(
        build_stages(settings, provider_name=provider_name, dns_mode=dns_mode),
        work_dir=settings.work_dir,
        output_dir=settings.output_dir,
    )
