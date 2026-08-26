"""Active speaker detection via NVIDIA's hosted Maxine NIM.

Adapted from `NVIDIA-Maxine/nim-clients` rather than reimplemented -- the
request construction and response shape are theirs, and the vendored copy under
`nim/` is the authority. See `nim/VENDORED.md` for provenance and for the two
places the published documentation disagrees with the actual wire format.

Two behaviours here are deliberately not what the sample client does:

**Diarization goes out first.** Upstream interleaves video, audio, and
diarization in one loop, so the first video chunk precedes any diarization. The
service treats a session whose first inference request lacks diarization as
having none at all, and then returns *confidently wrong* results rather than an
error. All diarization batches are therefore sent immediately after the config
message, before any media.

**Results are returned, not drawn.** Upstream's `process_responses()` ends by
rendering bounding boxes into an output video. The same collection loop is used
here, minus that call, yielding the structured per-frame detections the pipeline
actually needs.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import (
    ASDAuthError,
    ASDError,
    ASDInvalidInputError,
    ASDQuotaError,
    ASDTimeoutError,
    ASDUnavailableError,
    DiarizationDroppedError,
    MalformedDiarizationError,
)
from dialogue_extractor.domain.models import ASDFrame, ASDResult, BoundingBox, Face
from dialogue_extractor.infra.logging import get_logger, stage
from dialogue_extractor.infra.retry import retry
from dialogue_extractor.interfaces.ports import ASDProvider

log = get_logger(__name__)

_VENDOR_ROOT = Path(__file__).parent / "nim"


def _ensure_vendor_on_path() -> None:
    """Put the vendored client on sys.path.

    Its generated stubs import each other as top-level `nvidia.*`, which is how
    protoc emitted them, so the `interfaces/` directory has to be importable as
    a root rather than as a subpackage.
    """
    for path in (_VENDOR_ROOT / "interfaces", _VENDOR_ROOT):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _load_protos() -> tuple[Any, Any, Any, Any]:
    _ensure_vendor_on_path()
    try:
        from nvidia.ai4m.activespeakerdetection.v1 import (
            activespeakerdetection_pb2 as asd_pb2,
        )
        from nvidia.ai4m.activespeakerdetection.v1 import (
            activespeakerdetection_pb2_grpc as asd_grpc,
        )
        from nvidia.ai4m.audio.v1 import audio_pb2
        from nvidia.ai4m.video.v1 import video_pb2
    except ImportError as exc:
        raise ASDError(
            "vendored_client_missing",
            f"Could not import the vendored NVIDIA client: {exc}",
            remediation="Check src/dialogue_extractor/vision/providers/nim/ is intact.",
        ) from exc
    return asd_pb2, asd_grpc, audio_pb2, video_pb2


DATA_CHUNK_SIZE = 64 * 1024
"""Matches the vendored client's chunk size."""

DIARIZATION_BATCH_SIZE = 100
"""Words per diarization message, matching the vendored client."""


def read_file_chunks(path: Path, chunk_size: int = DATA_CHUNK_SIZE) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            yield chunk


def classify_grpc_error(exc: BaseException) -> ASDError:
    """Map a gRPC failure to a specific, correctly-retryable error."""
    import grpc

    code = exc.code() if isinstance(exc, grpc.RpcError) and hasattr(exc, "code") else None
    detail = getattr(exc, "details", lambda: str(exc))() or str(exc)

    if code == grpc.StatusCode.UNAUTHENTICATED or code == grpc.StatusCode.PERMISSION_DENIED:
        return ASDAuthError(
            "auth_rejected",
            f"The detection service rejected the credentials: {detail[:300]}",
            remediation=(
                "Check NVIDIA_API_KEY and ASD_FUNCTION_ID in .env. The function id is on "
                "the API tab of https://build.nvidia.com/nvidia/active-speaker-detection"
            ),
        )
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return ASDQuotaError(
            "quota_or_payload",
            f"Request rejected as too large or over quota: {detail[:300]}",
            remediation="The clip may be too long; the pipeline will retry with a shorter one.",
        )
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return ASDTimeoutError(
            "deadline_exceeded",
            f"The detection service did not respond in time: {detail[:300]}",
        )
    if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL):
        return ASDUnavailableError(
            "service_unavailable",
            f"Could not reach the detection service: {detail[:300]}",
            remediation="This may be transient. Check network access to grpc.nvcf.nvidia.com.",
        )
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        # INVALID_ARGUMENT covers anything malformed -- a bad container, an
        # unsupported codec, a broken diarization payload. Blaming diarization
        # unconditionally sends the reader looking in the wrong place; the first
        # real failure here was "not a valid MP4 atom", which had nothing to do
        # with it. So only claim diarization when the service says so.
        lowered = detail.lower()
        if "diariz" in lowered or "speaker" in lowered:
            return MalformedDiarizationError(
                "malformed_diarization",
                f"The service rejected the diarization payload: {detail[:300]}",
            )
        return ASDInvalidInputError(
            "invalid_input",
            f"The service rejected the input as malformed: {detail[:300]}",
            remediation=(
                "The clip must be H.264 in a streamable MP4. Check the clip actually "
                "downloaded or encoded rather than being a stub or pointer file."
            ),
        )
    return ASDError(
        "detection_failed",
        f"Active speaker detection failed ({code}): {detail[:300]}",
    )


UNASSOCIATED_WIRE_CODE = -1
"""What the service returns for a face it tracked but associated with no speaker.

Established by running against NVIDIA's own sample assets, whose diarization
contains `speaker_0` and `speaker_1`: the response carried **-1** on the
untracked face and **0** on the one that was speaking. So zero is a legitimate
speaker id, and the sentinel is negative.

This was worth finding the hard way. Assuming zero meant "unassociated" -- the
natural reading, since an unset proto int32 defaults to 0 -- nulls out the real
speaker on every *correctly associated* face. The on-screen check compares that
speaker against the diarized one, so it would have failed to match on precisely
the frames it exists to confirm, and reported "not on screen" for dialogue
plainly on screen.
"""


def _speaker_or_none(raw: Any) -> int | None:
    """Normalize the wire speaker id, treating "no association" as None.

    Negative means unassociated. Zero does not -- see UNASSOCIATED_WIRE_CODE.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return None if value < 0 else value


def map_response_frames(
    frame_detections: dict[int, list[dict[str, Any]]],
    clip_start_frame: int,
    fps: float,
    *,
    provider: str = "nvidia",
) -> ASDResult:
    """Turn collected per-frame dicts into the domain result.

    Note the field rename: the vendored client stores the proto's
    `face_detection_confidence` under the key `confidence_score`. Our domain
    model keeps the proto's name, since that is the contract of record.
    """
    frames = tuple(
        ASDFrame(
            frame_id=frame_id,
            faces=tuple(
                Face(
                    face_id=int(face["face_id"]),
                    bbox=BoundingBox(
                        x=float(face["bbox"]["x"]),
                        y=float(face["bbox"]["y"]),
                        width=float(face["bbox"]["width"]),
                        height=float(face["bbox"]["height"]),
                    ),
                    is_speaking=bool(face["is_speaking"]),
                    face_detection_confidence=float(face["confidence_score"]),
                    diarized_speaker_id=_speaker_or_none(face.get("diarized_speaker_id")),
                )
                for face in faces
            ),
        )
        for frame_id, faces in sorted(frame_detections.items())
    )

    return ASDResult(
        frames=frames,
        clip_start_frame=clip_start_frame,
        fps=fps,
        provider=provider,
    )


class NvidiaASDProvider(ASDProvider):
    """Hosted NVIDIA Active Speaker Detection over gRPC."""

    name = "nvidia"

    def __init__(
        self,
        api_key: str,
        function_id: str,
        *,
        target: str = "grpc.nvcf.nvidia.com:443",
        deadline: float = constants.ASD_DEADLINE_SECONDS,
        speaker_detection_threshold: float | None = None,
    ) -> None:
        self._api_key = api_key
        self._function_id = function_id
        self._target = target
        self._deadline = deadline
        self._threshold = speaker_detection_threshold

    def health_check(self) -> None:
        if not self._api_key:
            raise ASDAuthError("missing_api_key", "No NVIDIA API key was provided.")
        if not self._function_id:
            raise ASDAuthError("missing_function_id", "No ASD function id was provided.")
        _load_protos()

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (
            ("authorization", f"Bearer {self._api_key}"),
            ("function-id", self._function_id),
        )

    def _build_requests(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
    ) -> Iterator[Any]:
        """Yield the request stream: config, then diarization, then media.

        The ordering is the point. See the module docstring.
        """
        asd_pb2, _, audio_pb2, video_pb2 = _load_protos()

        # 1. Configuration.
        detection_config = asd_pb2.ActiveSpeakerDetectionConfig(
            input_video_config=video_pb2.VideoConfig(codec=video_pb2.VIDEO_CODEC_H264),
        )
        if audio_path is not None:
            detection_config.input_audio_config.CopyFrom(
                audio_pb2.AudioConfig(encoding=audio_pb2.AUDIO_CODEC_WAV)
            )
            detection_config.audio_source_config = (
                asd_pb2.AUDIO_SOURCE_CONFIG_SEPARATE_STREAM
            )
        else:
            detection_config.input_audio_config.CopyFrom(
                audio_pb2.AudioConfig(encoding=audio_pb2.AUDIO_CODEC_WAV)
            )
            detection_config.audio_source_config = (
                asd_pb2.AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO
            )
        if self._threshold is not None:
            detection_config.speaker_detection_threshold = self._threshold

        yield asd_pb2.DetectActiveSpeakerRequest(config=detection_config)

        # 2. Diarization, in full, before any media reaches the service.
        segments, transcript = self._parse_diarization(diarization)
        for index in range(0, len(segments), DIARIZATION_BATCH_SIZE):
            batch = segments[index : index + DIARIZATION_BATCH_SIZE]
            is_last = index + DIARIZATION_BATCH_SIZE >= len(segments)
            yield asd_pb2.DetectActiveSpeakerRequest(
                data=asd_pb2.ActiveSpeakerDetectionData(
                    diarization_info=asd_pb2.AudioDiarizationInfo(
                        segments=batch,
                        transcript=transcript if is_last else "",
                    )
                )
            )

        # 3. Media.
        for chunk in read_file_chunks(clip_path):
            yield asd_pb2.DetectActiveSpeakerRequest(
                data=asd_pb2.ActiveSpeakerDetectionData(video_data=chunk)
            )
        if audio_path is not None:
            for chunk in read_file_chunks(audio_path):
                yield asd_pb2.DetectActiveSpeakerRequest(
                    data=asd_pb2.ActiveSpeakerDetectionData(audio_data=chunk)
                )

    def _parse_diarization(self, diarization: dict[str, Any]) -> tuple[list[Any], str]:
        """Build proto segments using the vendored parser.

        Their parser rather than our own construction: it is what the service's
        own client uses, so it is the definition of the format, including the
        seconds-to-milliseconds conversion and the integer speaker ids.
        """
        import json

        _ensure_vendor_on_path()
        try:
            from diarization import SampleDiarizationParser  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ASDError(
                "vendored_parser_missing",
                f"Could not import the vendored diarization parser: {exc}",
            ) from exc

        try:
            segments, transcript = SampleDiarizationParser().parse(json.dumps(diarization))
        except (ValueError, KeyError, TypeError) as exc:
            raise MalformedDiarizationError(
                "diarization_rejected",
                f"The diarization payload could not be converted: {exc}",
                remediation=(
                    "Speaker labels must contain digits -- build the payload with a "
                    "SpeakerCodec so unattributed words do not become 'UNKNOWN'."
                ),
            ) from exc

        return list(segments), transcript or ""

    def _collect(self, response_iter: Iterator[Any]) -> dict[int, list[dict[str, Any]]]:
        """Collect per-frame detections.

        This is upstream's `process_responses()` loop with its trailing
        `write_output_video()` call removed.
        """
        frame_detections: dict[int, list[dict[str, Any]]] = {}

        for response in response_iter:
            if response.HasField("config") or response.HasField("keepalive"):
                continue
            if not response.HasField("active_speaker_detection_result"):
                continue

            result = response.active_speaker_detection_result
            frame_detections[result.frame_id] = [
                {
                    "diarized_speaker_id": s.diarized_speaker_id,
                    "face_id": s.face_id,
                    "is_speaking": s.is_speaking,
                    "confidence_score": s.face_detection_confidence,
                    "bbox": {
                        "x": s.speaker_bbox.x,
                        "y": s.speaker_bbox.y,
                        "width": s.speaker_bbox.width,
                        "height": s.speaker_bbox.height,
                    },
                }
                for s in result.speaker_data
            ]

        return frame_detections

    @retry(
        attempts=3,
        base_delay=2.0,
        # Only the failures that another attempt can actually fix. A hosted
        # endpoint returns the occasional INTERNAL or UNAVAILABLE for reasons
        # that have nothing to do with the request -- and without a retry a
        # single one of those ends a run that has already spent minutes
        # downloading, transcribing and diarizing. Everything else is left
        # alone: retrying rejected credentials or a malformed clip just fails
        # three times as slowly.
        retry_on=(ASDUnavailableError, ASDTimeoutError),
    )
    def _detect_with_retry(
        self,
        asd_grpc: Any,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        """One attempt at the streaming RPC.

        The whole channel is built inside the retry rather than outside it: a
        connection that failed once is not one to reuse, and the request stream
        is a generator that cannot be replayed.
        """
        import grpc

        credentials = grpc.ssl_channel_credentials()
        try:
            with grpc.secure_channel(self._target, credentials) as channel:
                stub = asd_grpc.ActiveSpeakerDetectionServiceStub(channel)
                responses = stub.DetectActiveSpeaker(
                    self._build_requests(clip_path, audio_path, diarization),
                    metadata=self._metadata(),
                    timeout=self._deadline,
                )
                return self._collect(responses)
        except ASDError:
            raise
        except grpc.RpcError as exc:
            raise classify_grpc_error(exc) from exc
        except Exception as exc:
            raise ASDError(
                "detection_failed",
                f"Active speaker detection failed: {type(exc).__name__}: {str(exc)[:300]}",
            ) from exc

    def detect(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
        clip_start_frame: int,
        fps: float,
    ) -> ASDResult:

        self.health_check()

        if not clip_path.exists():
            raise ASDError("clip_missing", f"Clip does not exist: {clip_path}")

        _, asd_grpc, _, _ = _load_protos()

        with stage(
            "active_speaker_detection",
            clip=clip_path.name,
            clip_start_frame=clip_start_frame,
        ) as report:
            frame_detections = self._detect_with_retry(
                asd_grpc, clip_path, audio_path, diarization
            )

            report["frames"] = len(frame_detections)

        result = map_response_frames(
            frame_detections, clip_start_frame, fps, provider=self.name
        )
        if result.any_face_detected and not result.any_speaker_id_present:
            # Not an error. Every face carrying -1 means the service tracked
            # faces but matched none of them to the diarized audio -- which is
            # precisely the off-camera case this system exists to detect. It
            # flows on to frame selection, which reports it as not-on-screen
            # with a reason.
            log.debug(
                "asd.no_speaker_association",
                clip=clip_path.name,
                frames=len(result.frames),
            )
        return result


def lacks_speaker_association(result: ASDResult) -> bool:
    """True when faces were tracked but none matched a diarized speaker.

    On its own this is *not* a failure. A face carrying -1 is the documented
    "tracked but unassigned" state, and a whole window of them is the ordinary
    signal for dialogue spoken off camera -- the thing this system is built to
    identify. Raising here would turn the system's core answer into an error.

    It only means something is wrong when it holds for *every* window in a run;
    see `assert_diarization_reached_service`.
    """
    return result.any_face_detected and not result.any_speaker_id_present


def assert_diarization_reached_service(
    unassociated: int, processed: int, *, minimum_sample: int = 3
) -> None:
    """Raise when a whole run shows no speaker association anywhere.

    This is the session-level version of the check, and it is the level the
    problem actually lives at. A session whose first inference request lacked
    diarization is treated as having none for its entire duration -- so the
    symptom is systematic, not per-window.

    Requiring a minimum sample keeps it from firing on a short run that happens
    to contain only off-camera dialogue, which is a legitimate answer.
    """
    if processed < minimum_sample or unassociated < processed:
        return

    raise DiarizationDroppedError(
        "diarization_dropped",
        f"All {processed} windows came back with no speaker association at all. "
        "One such window is ordinary -- it means the speaker was off camera -- "
        "but every window means the diarization payload never reached the service.",
        remediation=(
            "Diarization must be sent before the first inference request. This "
            "points at request ordering, not at configuration."
        ),
    )
