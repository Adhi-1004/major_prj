"""The ASD client against a local gRPC server.

This is the only test that exercises the real client end to end -- the actual
generated stubs, the actual streaming call, the actual response iteration --
without touching the network or spending credits. The live test costs money and
needs credentials; this one runs on every commit.

The server also *records what it received*, which is how the ordering guarantee
becomes verifiable rather than merely intended: a real server is what would
observe diarization arriving late, so a stand-in for it is where to assert that
it does not.
"""

from __future__ import annotations

import json
import sys
from concurrent import futures
from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.audio.diarization_export import SpeakerCodec, to_nvidia_diarization
from dialogue_extractor.domain.errors import ASDAuthError, ASDError, DiarizationDroppedError
from dialogue_extractor.domain.models import Word
from dialogue_extractor.vision.providers.nvidia_asd import NvidiaASDProvider

VENDOR_ROOT = Path("src/dialogue_extractor/vision/providers/nim")

pytestmark = pytest.mark.skipif(
    not (VENDOR_ROOT / "interfaces").exists(),
    reason="vendored NVIDIA client not present",
)


def _protos() -> tuple[Any, Any]:
    for path in (VENDOR_ROOT / "interfaces", VENDOR_ROOT):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    from nvidia.ai4m.activespeakerdetection.v1 import (  # type: ignore[import-not-found]
        activespeakerdetection_pb2 as pb2,
    )
    from nvidia.ai4m.activespeakerdetection.v1 import (  # type: ignore[import-not-found]
        activespeakerdetection_pb2_grpc as pb2_grpc,
    )
    return pb2, pb2_grpc


class RecordingServicer:
    """Stands in for the service, recording the order it received things in."""

    def __init__(
        self,
        *,
        frames: int = 10,
        faces_per_frame: int = 1,
        speaker_id: int = 1,
        fail_with: Any = None,
    ) -> None:
        self.pb2, _ = _protos()
        self.frames = frames
        self.faces_per_frame = faces_per_frame
        self.speaker_id = speaker_id
        self.fail_with = fail_with

        self.received: list[str] = []
        self.segments_received = 0
        self.metadata: dict[str, str] = {}

    def DetectActiveSpeaker(self, request_iterator: Any, context: Any) -> Any:
        self.metadata = dict(context.invocation_metadata())

        if self.fail_with is not None:
            context.abort(self.fail_with, "stub failure")

        for request in request_iterator:
            if request.HasField("config"):
                self.received.append("config")
            elif request.data.HasField("diarization_info"):
                self.received.append("diarization")
                self.segments_received += len(request.data.diarization_info.segments)
            elif request.data.video_data:
                self.received.append("video")
            elif request.data.audio_data:
                self.received.append("audio")

        yield self.pb2.DetectActiveSpeakerResponse(
            config=self.pb2.ActiveSpeakerDetectionConfig()
        )

        for frame_id in range(self.frames):
            result = self.pb2.ActiveSpeakerDetectionResult(frame_id=frame_id)
            for face_index in range(self.faces_per_frame):
                info = result.speaker_data.add()
                info.face_id = face_index + 1
                info.is_speaking = frame_id >= 3  # silent, then speaking
                info.face_detection_confidence = 0.9
                # -1 is the service's real "no association" sentinel.
                info.diarized_speaker_id = self.speaker_id
                info.speaker_bbox.x = 0.1
                info.speaker_bbox.y = 0.2
                info.speaker_bbox.width = 0.3
                info.speaker_bbox.height = 0.4
            yield self.pb2.DetectActiveSpeakerResponse(active_speaker_detection_result=result)


@pytest.fixture
def stub_server():  # type: ignore[no-untyped-def]
    """Start a servicer on a free port; yield (servicer, target)."""
    import grpc

    _, pb2_grpc = _protos()
    servers: list[Any] = []

    def start(servicer: RecordingServicer) -> str:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        pb2_grpc.add_ActiveSpeakerDetectionServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return f"127.0.0.1:{port}"

    yield start

    for server in servers:
        server.stop(grace=0)


class InsecureProvider(NvidiaASDProvider):
    """The real provider over an insecure channel, so no TLS setup is needed."""

    def detect(self, clip_path, audio_path, diarization, clip_start_frame, fps):  # type: ignore[no-untyped-def]
        import grpc


        real_secure = grpc.secure_channel
        grpc.secure_channel = lambda target, credentials, *a, **k: grpc.insecure_channel(target)
        try:
            return super().detect(clip_path, audio_path, diarization, clip_start_frame, fps)
        finally:
            grpc.secure_channel = real_secure


@pytest.fixture
def clip_and_payload(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"v" * (64 * 1024 * 2))
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a" * 4096)

    words = tuple(
        Word(f"w{i}", i * 0.2, i * 0.2 + 0.15, speaker_id="SPEAKER_04") for i in range(150)
    )
    payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))
    return clip, audio, payload


class TestRoundTrip:
    def test_returns_mapped_frames(self, stub_server, clip_and_payload) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        servicer = RecordingServicer(frames=10)
        target = stub_server(servicer)

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        result = provider.detect(clip, audio, payload, clip_start_frame=300, fps=25.0)

        assert len(result.frames) == 10
        assert result.clip_start_frame == 300
        assert result.provider == "nvidia"

    def test_speaking_flags_survive(self, stub_server, clip_and_payload) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        target = stub_server(RecordingServicer(frames=10))

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        result = provider.detect(clip, audio, payload, 0, 25.0)

        speaking = [f.frame_id for f in result.frames if f.speaking_faces]
        assert speaking == [3, 4, 5, 6, 7, 8, 9]

    def test_frame_ids_stay_clip_relative(self, stub_server, clip_and_payload) -> None:  # type: ignore[no-untyped-def]
        """Applying the offset twice would be an easy and invisible mistake."""
        clip, audio, payload = clip_and_payload
        target = stub_server(RecordingServicer(frames=5))

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        result = provider.detect(clip, audio, payload, clip_start_frame=1000, fps=25.0)

        assert [f.frame_id for f in result.frames] == [0, 1, 2, 3, 4]

    def test_multiple_faces_per_frame(self, stub_server, clip_and_payload) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        target = stub_server(RecordingServicer(frames=3, faces_per_frame=3))

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        result = provider.detect(clip, audio, payload, 0, 25.0)

        assert all(len(f.faces) == 3 for f in result.frames)


class TestOrderingObservedByTheServer:
    """What the service actually receives, in the order it receives it."""

    def test_diarization_arrives_before_any_media(
        self, stub_server, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        servicer = RecordingServicer(frames=2)
        target = stub_server(servicer)

        InsecureProvider(api_key="k", function_id="f", target=target).detect(
            clip, audio, payload, 0, 25.0
        )

        kinds = servicer.received
        assert kinds[0] == "config"
        last_diarization = max(i for i, k in enumerate(kinds) if k == "diarization")
        first_media = min(i for i, k in enumerate(kinds) if k in ("video", "audio"))
        assert last_diarization < first_media

    def test_every_word_reaches_the_server(
        self, stub_server, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        servicer = RecordingServicer(frames=2)
        target = stub_server(servicer)

        InsecureProvider(api_key="k", function_id="f", target=target).detect(
            clip, audio, payload, 0, 25.0
        )
        assert servicer.segments_received == 150

    def test_credentials_are_sent_as_metadata(
        self, stub_server, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        servicer = RecordingServicer(frames=2)
        target = stub_server(servicer)

        InsecureProvider(api_key="nvapi-secret", function_id="fid-9", target=target).detect(
            clip, audio, payload, 0, 25.0
        )

        assert servicer.metadata["authorization"] == "Bearer nvapi-secret"
        assert servicer.metadata["function-id"] == "fid-9"


class TestDiarizationDroppedIsRefused:
    def test_response_with_no_speaker_association_raises(
        self, stub_server, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        """The silent-wrong-answer case: ordinary-looking detections carrying no
        speaker association at all."""
        clip, audio, payload = clip_and_payload
        target = stub_server(RecordingServicer(frames=5, speaker_id=-1))

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        with pytest.raises(DiarizationDroppedError):
            provider.detect(clip, audio, payload, 0, 25.0)


class TestServerErrors:
    def test_unauthenticated_becomes_an_auth_error(
        self, stub_server, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        import grpc

        clip, audio, payload = clip_and_payload
        target = stub_server(
            RecordingServicer(fail_with=grpc.StatusCode.UNAUTHENTICATED)
        )

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        with pytest.raises(ASDAuthError):
            provider.detect(clip, audio, payload, 0, 25.0)

    def test_resource_exhausted_becomes_a_quota_error(
        self, stub_server, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        import grpc

        from dialogue_extractor.domain.errors import ASDQuotaError

        clip, audio, payload = clip_and_payload
        target = stub_server(
            RecordingServicer(fail_with=grpc.StatusCode.RESOURCE_EXHAUSTED)
        )

        provider = InsecureProvider(api_key="k", function_id="f", target=target)
        with pytest.raises(ASDQuotaError):
            provider.detect(clip, audio, payload, 0, 25.0)

    def test_unreachable_server_is_reported_clearly(
        self, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        clip, audio, payload = clip_and_payload
        provider = InsecureProvider(
            api_key="k", function_id="f", target="127.0.0.1:1", deadline=2.0
        )
        with pytest.raises(ASDError):
            provider.detect(clip, audio, payload, 0, 25.0)


class TestPayloadIsWhatWeThink:
    def test_the_payload_sent_is_the_documented_shape(
        self, clip_and_payload
    ) -> None:  # type: ignore[no-untyped-def]
        _, _, payload = clip_and_payload
        assert json.loads(json.dumps(payload))["words"][0]["speaker_id"].startswith("speaker_")
