"""The NVIDIA ASD provider: request shape, response mapping, error handling.

No network. The gRPC stub is faked, so what gets exercised is the part we wrote:
the ordering of the request stream, the translation of responses into domain
objects, and the refusal to interpret a response that shows diarization was
dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.domain.errors import (
    ASDAuthError,
    ASDError,
    ASDQuotaError,
    ASDTimeoutError,
    ASDUnavailableError,
    DiarizationDroppedError,
    MalformedDiarizationError,
)
from dialogue_extractor.vision.providers.nvidia_asd import (
    NvidiaASDProvider,
    assert_diarization_reached_service,
    classify_grpc_error,
    lacks_speaker_association,
    map_response_frames,
    read_file_chunks,
)


def face_dict(
    *,
    face_id: int = 1,
    speaking: bool = True,
    confidence: float = 0.93,
    speaker: int | None = 1,
) -> dict[str, Any]:
    """One face, in the shape the vendored client's collection loop produces."""
    return {
        "diarized_speaker_id": speaker,
        "face_id": face_id,
        "is_speaking": speaking,
        "confidence_score": confidence,
        "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    }


class TestMapResponseFrames:
    def test_maps_a_single_frame(self) -> None:
        result = map_response_frames({0: [face_dict()]}, clip_start_frame=0, fps=25.0)

        assert len(result.frames) == 1
        face = result.frames[0].faces[0]
        assert face.face_id == 1
        assert face.is_speaking is True
        assert face.face_detection_confidence == pytest.approx(0.93)
        assert face.diarized_speaker_id == 1

    def test_renames_confidence_score_to_the_proto_field_name(self) -> None:
        """The vendored client stores the proto's `face_detection_confidence`
        under the key `confidence_score`. Our model keeps the proto's name."""
        result = map_response_frames(
            {0: [face_dict(confidence=0.77)]}, clip_start_frame=0, fps=25.0
        )
        assert result.frames[0].faces[0].face_detection_confidence == pytest.approx(0.77)

    def test_bbox_is_carried_through(self) -> None:
        bbox = map_response_frames({0: [face_dict()]}, 0, 25.0).frames[0].faces[0].bbox
        assert (bbox.x, bbox.y, bbox.width, bbox.height) == pytest.approx((0.1, 0.2, 0.3, 0.4))

    def test_frames_are_sorted_by_id(self) -> None:
        """Frame selection walks these in order looking for a consecutive run."""
        result = map_response_frames(
            {5: [face_dict()], 1: [face_dict()], 3: [face_dict()]}, 0, 25.0
        )
        assert [f.frame_id for f in result.frames] == [1, 3, 5]

    def test_multiple_faces_per_frame(self) -> None:
        frame = map_response_frames(
            {0: [face_dict(face_id=1), face_dict(face_id=2, speaking=False)]}, 0, 25.0
        ).frames[0]
        assert len(frame.faces) == 2
        assert len(frame.speaking_faces) == 1

    def test_frame_with_no_faces(self) -> None:
        result = map_response_frames({0: []}, 0, 25.0)
        assert result.frames[0].faces == ()
        assert result.any_face_detected is False

    def test_empty_response(self) -> None:
        result = map_response_frames({}, 0, 25.0)
        assert result.frames == ()

    def test_carries_the_offset_and_rate(self) -> None:
        result = map_response_frames({0: [face_dict()]}, clip_start_frame=2456, fps=23.976)
        assert result.clip_start_frame == 2456
        assert result.fps == pytest.approx(23.976)
        assert result.provider == "nvidia"


class TestFrameOffsetInvariant:
    """Invariant 1: global_frame == clip_start_frame + response frame_id.

    The service numbers frames from the start of the clip it was sent. Forgetting
    the offset yields a real frame from the right video at the wrong moment,
    which is entirely plausible-looking and therefore hard to notice.
    """

    @pytest.mark.parametrize(
        ("clip_start_frame", "frame_id"),
        [(0, 0), (0, 42), (300, 0), (300, 17), (86_400, 88)],
    )
    def test_offset_is_recoverable(self, clip_start_frame: int, frame_id: int) -> None:
        from dialogue_extractor.domain.timing import clip_frame_to_global

        result = map_response_frames({frame_id: [face_dict()]}, clip_start_frame, 25.0)
        recovered = clip_frame_to_global(result.frames[0].frame_id, result.clip_start_frame)

        assert recovered == clip_start_frame + frame_id

    def test_response_ids_stay_clip_relative(self) -> None:
        """The mapper must not silently pre-apply the offset, or it would be
        applied twice downstream."""
        result = map_response_frames({7: [face_dict()]}, clip_start_frame=300, fps=25.0)
        assert result.frames[0].frame_id == 7


class TestNoSpeakerAssociation:
    """A window where every face carries -1 is *not* a failure.

    -1 is the documented "tracked but unassigned" state. A whole window of them
    means the visible faces matched none of the diarized audio -- which is the
    ordinary signal for dialogue spoken off camera, the very thing this system
    exists to identify. Raising here would turn the system's core answer into
    an error, and it did: a real discovery run reported `diarization_dropped`
    for a line whose speaker was simply not in shot.
    """

    def test_detects_the_condition(self) -> None:
        result = map_response_frames(
            {0: [face_dict(speaker=-1)], 1: [face_dict(speaker=-1)]}, 0, 25.0
        )
        assert lacks_speaker_association(result) is True

    def test_one_associated_face_is_enough_to_clear_it(self) -> None:
        result = map_response_frames(
            {0: [face_dict(speaker=-1)], 1: [face_dict(speaker=2)]}, 0, 25.0
        )
        assert lacks_speaker_association(result) is False

    def test_no_faces_at_all_is_not_the_same_condition(self) -> None:
        assert lacks_speaker_association(map_response_frames({0: [], 1: []}, 0, 25.0)) is False

    def test_empty_response(self) -> None:
        assert lacks_speaker_association(map_response_frames({}, 0, 25.0)) is False


class TestSessionLevelDiarizationGuard:
    """The failure this guards against is systematic by nature: a session whose
    first request lacked diarization has none for its whole duration. Checking
    per-window cannot distinguish it from one off-camera line."""

    def test_every_window_unassociated_raises(self) -> None:
        with pytest.raises(DiarizationDroppedError) as excinfo:
            assert_diarization_reached_service(unassociated=8, processed=8)
        assert excinfo.value.reason == "diarization_dropped"

    def test_one_unassociated_window_among_many_is_fine(self) -> None:
        """The real observed case: 1 of 8 windows had no association because
        that speaker was off camera."""
        assert_diarization_reached_service(unassociated=1, processed=8)

    def test_most_but_not_all_is_fine(self) -> None:
        assert_diarization_reached_service(unassociated=7, processed=8)

    def test_a_short_run_is_not_enough_evidence(self) -> None:
        """A one-candidate run of genuinely off-camera dialogue must not be
        reported as a plumbing failure."""
        assert_diarization_reached_service(unassociated=1, processed=1)
        assert_diarization_reached_service(unassociated=2, processed=2)

    def test_the_message_distinguishes_the_two_cases(self) -> None:
        with pytest.raises(DiarizationDroppedError) as excinfo:
            assert_diarization_reached_service(unassociated=5, processed=5)
        assert "off camera" in excinfo.value.detail

    def test_nothing_processed(self) -> None:
        assert_diarization_reached_service(unassociated=0, processed=0)


class TestGrpcErrorClassification:
    def _rpc_error(self, code: Any, details: str = "boom") -> Any:
        import grpc

        class FakeRpcError(grpc.RpcError):
            def code(self) -> Any:
                return code

            def details(self) -> str:
                return details

        return FakeRpcError()

    def test_unauthenticated(self) -> None:
        import grpc

        error = classify_grpc_error(self._rpc_error(grpc.StatusCode.UNAUTHENTICATED))
        assert isinstance(error, ASDAuthError)
        assert "NVIDIA_API_KEY" in error.to_dict()["remediation"]

    def test_permission_denied_is_also_auth(self) -> None:
        import grpc

        assert isinstance(
            classify_grpc_error(self._rpc_error(grpc.StatusCode.PERMISSION_DENIED)),
            ASDAuthError,
        )

    def test_resource_exhausted(self) -> None:
        import grpc

        error = classify_grpc_error(self._rpc_error(grpc.StatusCode.RESOURCE_EXHAUSTED))
        assert isinstance(error, ASDQuotaError)

    def test_deadline_exceeded(self) -> None:
        import grpc

        assert isinstance(
            classify_grpc_error(self._rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)),
            ASDTimeoutError,
        )

    def test_unavailable(self) -> None:
        import grpc

        assert isinstance(
            classify_grpc_error(self._rpc_error(grpc.StatusCode.UNAVAILABLE)),
            ASDUnavailableError,
        )

    def test_invalid_argument_does_not_blame_diarization_by_default(self) -> None:
        """INVALID_ARGUMENT covers any malformed input. The first real one was
        "not a valid MP4 atom", which had nothing to do with diarization --
        blaming it unconditionally sends the reader to the wrong place."""
        import grpc

        from dialogue_extractor.domain.errors import ASDInvalidInputError

        error = classify_grpc_error(
            self._rpc_error(grpc.StatusCode.INVALID_ARGUMENT, "not a valid MP4 atom")
        )
        assert isinstance(error, ASDInvalidInputError)
        assert not isinstance(error, MalformedDiarizationError)
        assert "MP4" in error.to_dict()["remediation"]

    def test_invalid_argument_blames_diarization_when_the_service_does(self) -> None:
        import grpc

        error = classify_grpc_error(
            self._rpc_error(
                grpc.StatusCode.INVALID_ARGUMENT, "diarization segments are invalid"
            )
        )
        assert isinstance(error, MalformedDiarizationError)

    def test_unknown_code_is_still_a_stage_error(self) -> None:
        import grpc

        error = classify_grpc_error(self._rpc_error(grpc.StatusCode.UNKNOWN))
        assert isinstance(error, ASDError)
        assert error.to_dict()["stage"] == "active_speaker_detection"


class TestHealthCheck:
    def test_missing_api_key(self) -> None:
        with pytest.raises(ASDAuthError) as excinfo:
            NvidiaASDProvider(api_key="", function_id="fid").health_check()
        assert excinfo.value.reason == "missing_api_key"

    def test_missing_function_id(self) -> None:
        with pytest.raises(ASDAuthError) as excinfo:
            NvidiaASDProvider(api_key="key", function_id="").health_check()
        assert excinfo.value.reason == "missing_function_id"

    def test_valid_credentials_pass(self) -> None:
        NvidiaASDProvider(api_key="key", function_id="fid").health_check()


class TestMetadata:
    def test_carries_bearer_token_and_function_id(self) -> None:
        provider = NvidiaASDProvider(api_key="nvapi-secret", function_id="fid-123")
        metadata = dict(provider._metadata())
        assert metadata["authorization"] == "Bearer nvapi-secret"
        assert metadata["function-id"] == "fid-123"


class TestReadFileChunks:
    def test_yields_the_whole_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.bin"
        path.write_bytes(b"x" * 1000)
        assert b"".join(read_file_chunks(path, chunk_size=128)) == b"x" * 1000

    def test_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert list(read_file_chunks(path)) == []


class TestMissingClip:
    def test_detect_rejects_a_missing_clip(self, tmp_path: Path) -> None:
        provider = NvidiaASDProvider(api_key="key", function_id="fid")
        with pytest.raises(ASDError) as excinfo:
            provider.detect(tmp_path / "absent.mp4", None, {"words": []}, 0, 25.0)
        assert excinfo.value.reason == "clip_missing"


class TestRequestOrdering:
    """Diarization must reach the service before any media.

    The service treats a session whose first inference request lacks diarization
    as having none at all, and then returns confidently wrong results rather
    than an error. Upstream's sample client interleaves the streams, so its
    first video chunk precedes any diarization -- this asserts we do not.
    """

    def _requests(self, tmp_path: Path, with_audio: bool = True) -> list[Any]:
        from dialogue_extractor.audio.diarization_export import (
            SpeakerCodec,
            to_nvidia_diarization,
        )
        from dialogue_extractor.domain.models import Word

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"v" * (64 * 1024 * 3))  # three chunks
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"a" * (64 * 1024 * 2))

        words = tuple(
            Word(f"w{i}", i * 0.1, i * 0.1 + 0.08, speaker_id="SPEAKER_00")
            for i in range(250)  # more than one diarization batch
        )
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))

        provider = NvidiaASDProvider(api_key="key", function_id="fid")
        return list(
            provider._build_requests(clip, audio if with_audio else None, payload)
        )

    def test_config_is_first(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path)
        assert requests[0].HasField("config")

    def test_all_diarization_precedes_all_media(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path)

        kinds = []
        for request in requests:
            if request.HasField("config"):
                kinds.append("config")
            elif request.data.HasField("diarization_info"):
                kinds.append("diarization")
            elif request.data.video_data:
                kinds.append("video")
            elif request.data.audio_data:
                kinds.append("audio")

        last_diarization = max(i for i, k in enumerate(kinds) if k == "diarization")
        first_media = min(i for i, k in enumerate(kinds) if k in ("video", "audio"))

        assert last_diarization < first_media, (
            f"media reached the service before diarization finished: {kinds[:8]}"
        )

    def test_diarization_is_batched(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path)
        batches = [r for r in requests if not r.HasField("config")
                   and r.data.HasField("diarization_info")]
        assert len(batches) == 3, "250 words at 100 per batch"

    def test_transcript_rides_on_the_final_batch_only(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path)
        batches = [r for r in requests if not r.HasField("config")
                   and r.data.HasField("diarization_info")]
        assert batches[-1].data.diarization_info.transcript != ""
        assert all(b.data.diarization_info.transcript == "" for b in batches[:-1])

    def test_every_word_is_sent(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path)
        total = sum(
            len(r.data.diarization_info.segments)
            for r in requests
            if not r.HasField("config") and r.data.HasField("diarization_info")
        )
        assert total == 250

    def test_video_and_audio_are_both_sent(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path)
        video = sum(1 for r in requests if not r.HasField("config") and r.data.video_data)
        audio = sum(1 for r in requests if not r.HasField("config") and r.data.audio_data)
        assert video == 3
        assert audio == 2

    def test_embedded_audio_mode_sends_no_audio_stream(self, tmp_path: Path) -> None:
        requests = self._requests(tmp_path, with_audio=False)
        assert not any(
            r.data.audio_data for r in requests if not r.HasField("config")
        )


class TestDiarizationPayloadRejection:
    def test_unencodable_speaker_label_is_reported_clearly(self, tmp_path: Path) -> None:
        """A label with no digits cannot be encoded, and the message must say
        how to avoid it rather than surfacing a bare regex failure."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"v" * 100)

        provider = NvidiaASDProvider(api_key="key", function_id="fid")
        payload = {
            "words": [{"text": "x", "start": 0.0, "end": 1.0, "speaker_id": "UNKNOWN"}]
        }

        with pytest.raises(MalformedDiarizationError) as excinfo:
            list(provider._build_requests(clip, None, payload))

        assert excinfo.value.reason == "diarization_rejected"
        assert "SpeakerCodec" in excinfo.value.to_dict()["remediation"]


class TestWireSpeakerIdSentinel:
    """What the service actually means by each speaker code.

    Values below are the ones observed running against NVIDIA's own sample
    assets: their diarization contains speaker_0 and speaker_1, and the response
    carried -1 on the untracked face and 0 on the speaking one.

    The natural assumption -- that 0 means "unassociated", since an unset proto
    int32 defaults to 0 -- is wrong, and wrong in the damaging direction. It
    nulls the real speaker on every *correctly associated* face, so the
    on-screen check would fail to match on exactly the frames it exists to
    confirm and report "not on screen" for dialogue plainly on screen.
    """

    def test_negative_one_means_unassociated(self) -> None:
        result = map_response_frames({0: [face_dict(speaker=-1)]}, 0, 25.0)
        assert result.frames[0].faces[0].diarized_speaker_id is None

    def test_zero_is_a_real_speaker_not_a_sentinel(self) -> None:
        result = map_response_frames({0: [face_dict(speaker=0)]}, 0, 25.0)
        assert result.frames[0].faces[0].diarized_speaker_id == 0

    def test_positive_ids_are_preserved(self) -> None:
        result = map_response_frames({0: [face_dict(speaker=3)]}, 0, 25.0)
        assert result.frames[0].faces[0].diarized_speaker_id == 3

    def test_the_observed_mix_decodes_correctly(self) -> None:
        """Exactly what the live run returned: -1 on one face, 0 on the other."""
        result = map_response_frames(
            {0: [face_dict(face_id=1, speaking=False, speaker=-1),
                 face_dict(face_id=2, speaking=True, speaker=0)]},
            0,
            25.0,
        )
        faces = result.frames[0].faces
        assert faces[0].diarized_speaker_id is None
        assert faces[1].diarized_speaker_id == 0
        assert faces[1].is_speaking is True

    def test_a_zero_speaker_counts_as_association(self) -> None:
        """So it must not be mistaken for an unassociated window."""
        result = map_response_frames({0: [face_dict(speaker=0)]}, 0, 25.0)
        assert result.any_speaker_id_present is True
        assert lacks_speaker_association(result) is False

    def test_all_negative_reads_as_unassociated(self) -> None:
        result = map_response_frames(
            {0: [face_dict(speaker=-1)], 1: [face_dict(speaker=-1)]}, 0, 25.0
        )
        assert result.any_speaker_id_present is False
        assert lacks_speaker_association(result) is True

    def test_non_numeric_speaker_id_degrades_to_none(self) -> None:
        result = map_response_frames({0: [face_dict(speaker="bogus")]}, 0, 25.0)  # type: ignore[arg-type]
        assert result.frames[0].faces[0].diarized_speaker_id is None
