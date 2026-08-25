"""The ASD client against the real hosted service.

Opt-in (`-m e2e`), because it needs credentials and spends credits.

It runs against **NVIDIA's own sample assets**, not our generated output, and
that is deliberate. If this passes and a run on our own clip fails, the fault is
in what we produced -- the clip, the diarization payload, the offsets. If this
fails too, the fault is in the client or the credentials. Without that
separation, a failure on our own data has two plausible causes and no way to
tell them apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import credential

from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import ASDAuthError
from dialogue_extractor.tools.asd_check import fetch_asset
from dialogue_extractor.vision.providers.nvidia_asd import NvidiaASDProvider

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not credential("NVIDIA_API_KEY") or not Settings().asd_function_id,
        reason="needs NVIDIA_API_KEY and ASD_FUNCTION_ID",
    ),
]

ASSETS = {
    "sample_video_streamable.mp4": "video",
    "sample_audio.wav": "audio",
    "sample_diarization.json": "diarization",
}


@pytest.fixture(scope="module")
def nvidia_assets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """NVIDIA's own sample video, audio, and diarization.

    Fetched via the tool's own helper rather than a second copy of the download
    logic. The first version here duplicated it, and when the tool learned that
    the media is stored in Git LFS -- so the raw host serves a 132-byte pointer
    rather than the file -- this copy did not, and every test failed with
    "not a valid MP4 atom" against a perfectly good service.
    """
    directory = tmp_path_factory.mktemp("nvidia_assets")
    paths: dict[str, Path] = {}

    for filename, kind in ASSETS.items():
        try:
            paths[kind] = fetch_asset(filename, directory / filename)
        except Exception as exc:
            pytest.skip(f"could not download {filename}: {exc}")

    return paths


@pytest.fixture(scope="module")
def provider() -> NvidiaASDProvider:
    settings = Settings()
    return NvidiaASDProvider(
        api_key=settings.require_nvidia_api_key(),
        function_id=settings.require_asd_function_id(),
        target=settings.asd_target,
    )


class TestAgainstNvidiasOwnAssets:
    def test_detects_a_speaking_face(
        self, provider: NvidiaASDProvider, nvidia_assets: dict[str, Path]
    ) -> None:
        """The isolating check: their video, their diarization, our client."""
        diarization = json.loads(nvidia_assets["diarization"].read_text(encoding="utf-8"))

        result = provider.detect(
            clip_path=nvidia_assets["video"],
            audio_path=nvidia_assets["audio"],
            diarization=diarization,
            clip_start_frame=0,
            fps=25.0,
        )

        assert result.frames, "the service returned no frames at all"
        assert result.any_face_detected, "no face was detected in a sample clip of faces"

        speaking = [f for f in result.frames if f.speaking_faces]
        assert speaking, "no frame reported anyone speaking"

    def test_speaker_association_comes_back(
        self, provider: NvidiaASDProvider, nvidia_assets: dict[str, Path]
    ) -> None:
        """If this fails, diarization is not reaching the session -- which the
        service reports by returning unassociated results, not by erroring."""
        diarization = json.loads(nvidia_assets["diarization"].read_text(encoding="utf-8"))

        result = provider.detect(
            nvidia_assets["video"], nvidia_assets["audio"], diarization, 0, 25.0
        )
        assert result.any_speaker_id_present

    def test_confidences_are_in_range(
        self, provider: NvidiaASDProvider, nvidia_assets: dict[str, Path]
    ) -> None:
        diarization = json.loads(nvidia_assets["diarization"].read_text(encoding="utf-8"))
        result = provider.detect(
            nvidia_assets["video"], nvidia_assets["audio"], diarization, 0, 25.0
        )

        confidences = [
            face.face_detection_confidence for frame in result.frames for face in frame.faces
        ]
        assert confidences
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_frame_ids_are_contiguous_from_zero(
        self, provider: NvidiaASDProvider, nvidia_assets: dict[str, Path]
    ) -> None:
        """Frame ids must index the clip, which is what the offset arithmetic
        downstream assumes."""
        diarization = json.loads(nvidia_assets["diarization"].read_text(encoding="utf-8"))
        result = provider.detect(
            nvidia_assets["video"], nvidia_assets["audio"], diarization, 0, 25.0
        )

        ids = [f.frame_id for f in result.frames]
        assert ids[0] == 0
        assert ids == sorted(ids)


class TestCredentialFailures:
    def test_a_bad_key_is_reported_as_an_auth_error(
        self, nvidia_assets: dict[str, Path]
    ) -> None:
        """Not as a generic failure -- the user needs to know it is the key."""
        settings = Settings()
        bad = NvidiaASDProvider(
            api_key="nvapi-definitely-not-a-real-key",
            function_id=settings.require_asd_function_id(),
            target=settings.asd_target,
            deadline=60.0,
        )
        diarization = json.loads(nvidia_assets["diarization"].read_text(encoding="utf-8"))

        with pytest.raises(ASDAuthError):
            bad.detect(nvidia_assets["video"], nvidia_assets["audio"], diarization, 0, 25.0)
