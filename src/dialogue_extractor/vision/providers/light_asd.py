"""Placeholder for a locally-run active speaker detector.

Not implemented in this submission. It exists as a real class rather than as a
note in the documentation so that the claim "the detection backend is
swappable" is checkable: this satisfies the same ASDProvider port, the registry
selects it by name, and the pipeline would not need changing to use it.

The intended implementation is Light-ASD (CVPR 2023), which ships pretrained
weights and a documented single-command inference path. It would remove the
network dependency entirely, which matters because the hosted service is the
only part of this pipeline that cannot run offline.

If it is built: use the TalkSet-fine-tuned weights, not the AVA-only
checkpoint. The authors note that AVA-only training does not generalize to
footage outside that dataset, and generalizing to an arbitrary video is the
whole requirement here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dialogue_extractor.domain.models import ASDResult
from dialogue_extractor.interfaces.ports import ASDProvider


class LightASDProvider(ASDProvider):
    """Local active speaker detection. Declared, not implemented."""

    name = "light-asd"

    _MESSAGE = (
        "The local Light-ASD provider is not implemented in this submission. "
        "Use --asd-provider nvidia, which is the supported backend. "
        "To implement it: clone https://github.com/Junhua-Liao/Light-ASD, use the "
        "TalkSet-fine-tuned weights (not the AVA-only checkpoint), and return an "
        "ASDResult from this class -- nothing else in the pipeline needs to change."
    )

    def health_check(self) -> None:
        raise NotImplementedError(self._MESSAGE)

    def detect(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
        clip_start_frame: int,
        fps: float,
    ) -> ASDResult:
        raise NotImplementedError(self._MESSAGE)
