# Vendored from NVIDIA-Maxine/nim-clients

Everything in this directory is third-party code, copied verbatim. It is **not**
ours to reformat, and it is excluded from ruff, mypy, and coverage in
`pyproject.toml` so it stays byte-identical to upstream and therefore diffable
when NVIDIA change it.

| | |
|---|---|
| Source | https://github.com/NVIDIA-Maxine/nim-clients |
| Commit | `7ada3461f8eec7ccdb5e4670a46aed4867faac00` |
| Date | 2026-07-15 |
| Licence | MIT -- full text in `LICENSE.md` beside this file |

## What was taken, and why

| Path | Reason |
|---|---|
| `LICENSE.md` | NVIDIA's MIT licence, vendored alongside the code it covers. |
| `interfaces/nvidia/ai4m/**/*_pb2.py`, `*_pb2_grpc.py` | Pre-compiled protobuf stubs. Upstream ships these compiled, so there is no protoc step; `protos/linux/compile_protos.sh` exists only for regeneration or other languages. |
| `diarization.py` | `SampleDiarizationParser`, the code that will actually consume our payload. Reused rather than reimplemented so the contract is exercised by its real consumer. |
| `nim_constants.py` | `DATA_CHUNK_SIZE` (64 KB), `DIARIZATION_WORDS_BATCH_SIZE` (100), and the codec enum maps. Renamed from `constants.py` to avoid colliding with our own. |

## A note on licence headers

`diarization.py` and `nim_constants.py` carry NVIDIA's SPDX header inline.
The eight generated `*_pb2.py` / `*_pb2_grpc.py` files do not -- protoc does not
emit one, and upstream ships them the same way. `LICENSE.md` beside this file
covers all of them.

## What was deliberately not taken

`scripts/active_speaker_detection.py` is the sample CLI. Its request-building and
response-collection logic is adapted in `vision/providers/nvidia_asd.py` rather
than imported, for three reasons:

1. Its `process_responses()` ends by calling `write_output_video()`, which
   renders bounding boxes with OpenCV. We want the structured per-frame results,
   not an annotated video, and we do not want the OpenCV dependency.
2. It reads configuration from `argparse` namespaces and prints progress to
   stdout, neither of which suits use as a library.
3. **Its request ordering is unsafe for our purposes.** See below.

## The one deliberate behavioural difference

Upstream interleaves the three streams in a single loop:

```python
while not (video_done and audio_done and diarization_done):
    if not video_done:        yield <video chunk>
    if not audio_done:        yield <audio chunk>
    if not diarization_done:  yield <diarization batch>
```

so the first video chunk is sent **before** any diarization. The service's own
documentation warns:

> In streaming mode, diarization must be available at the time of the first
> inference request. If diarization is not supplied for that request, the
> session is treated as having no diarization, which can result in incorrect
> outputs.

and offers two remedies -- interleave it, or send it before the first inference
call. We take the second: all diarization batches go out immediately after the
config message, before any media. That failure mode is silent (confidently wrong
results rather than an error), so the ordering that cannot race is worth the
trivial cost.

## Contract details the published documentation gets wrong

Both were found by reading these files, and both are pinned by
`tests/integration/test_diarization_contract.py`, which runs our payload through
the parser above rather than against a contract we wrote down ourselves.

- **`speaker_id` is an `int32` on the wire.** The documented JSON shows
  `"speaker_id": "speaker_0"`; the parser recovers the integer with `\d+`. A
  label containing no digits raises, so `"UNKNOWN"` -- our unassigned
  placeholder -- would fail every request containing an unattributed word. Real
  output routinely contains them. Handled by `SpeakerCodec`.
- **The proto carries milliseconds, the JSON carries seconds.** The parser
  multiplies by 1000. Emitting milliseconds would describe a clip a thousand
  times too long, and both are valid JSON.

## Refreshing

Re-download the files listed above at a newer commit, update the SHA and date in
this file, and run:

```bash
pytest tests/integration/test_diarization_contract.py tests/unit/vision -q
```

Those tests exercise the parser and the response mapping directly, so a
contract change upstream shows up as a failure rather than as wrong results.
