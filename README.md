# On-Screen Dialogue Frame Extractor

Locates the exact video frame in which a given line of dialogue is spoken **on-screen**, and returns its timestamp, frame number, extracted text, and the frame image itself. If no target line is provided, the system discovers and returns every verified on-screen dialogue in the video.

"On-screen" specifically means the speaking character is visibly in frame, with mouth movement in sync with the audio at that moment — as distinct from voice-over, off-camera narration, or any case where the audio is heard but the visible person is not the one producing it.

This project was built in response to a technical problem statement requiring exactly this capability, working on an arbitrary public video URL without manual inspection, and generalizing to a different video or target line at evaluation time.

## Overview

Rather than a single end-to-end model, the system treats the problem as three fused sub-problems: what was said and when (speech recognition), who said it (speaker diarization), and whether the speaker was visibly on camera while saying it (audio-visual active speaker detection). Each stage is independently verifiable, testable, and swappable behind a stable interface.

Two modes share one pipeline:

- **Target search (Mode A):** given a target phrase, locate it in the transcript and verify on-screen status only for the small time window around it. The full video is never visually processed.
- **Discovery (Mode B):** with no target phrase, every detected speech segment is treated as a candidate, verified the same way, and every confirmed on-screen dialogue is returned as a list.

## Key Features

- Accurate, word-level dialogue localization via `faster-whisper`, with fuzzy matching tolerant of transcription noise
- Speaker diarization via `pyannote-audio`, aligned to transcript words by greatest temporal overlap
- On-screen verification via NVIDIA's Active Speaker Detection service, cross-checked against the diarized speaker rather than accepting any visible face
- Duration-based temporal smoothing, so a single flickering frame cannot produce a confident wrong answer
- A documented, deterministic confidence formula, with results routed to success, ambiguous, or not-on-screen rather than a bare true or false
- Forced constant frame rate normalization, so frame numbers and timestamps correspond exactly regardless of the source's original frame rate
- Structured error reporting at every stage; a failure in ingestion, transcription, or detection never crashes the process
- A modular architecture built around SOLID principles, so the active-speaker-detection backend, transcriber, or diarizer can each be replaced independently

## How It Works

![System Flow](docs/diagrams/system-flow.png)

The full architecture diagram, user flow, edge-case handling, assumptions, and the reasoning behind each design decision — including alternative approaches considered and why they were or were not adopted — are documented in [`docs/approach.md`](docs/approach.md).

## Tech Stack

| Purpose | Technology |
|---|---|
| Video acquisition | `yt-dlp` |
| Media normalization | `FFmpeg` / `FFprobe` |
| Speech transcription | `faster-whisper` |
| Speaker diarization | `pyannote-audio` |
| On-screen verification | NVIDIA Active Speaker Detection (NIM, hosted) |
| Frame extraction | `PyAV` / OpenCV |
| Text matching | `rapidfuzz` |
| API layer | `FastAPI` |
| Configuration | `pydantic-settings` |
| Web interface | `React` + `Vite` + `TypeScript` |

## Project Structure

```
quest1-dialogueextractor/
├── main.py                 # CLI entry point
├── serve.py                # HTTP entry point (uvicorn + FastAPI, also serves the UI)
├── Dockerfile               # Multi-stage, non-root, CPU-only torch by default
├── prompts.txt               # AI/LLM prompts used during development
├── frontend/                 # React + Vite + TypeScript interface
│   ├── src/api/               # Contract mirror, error normalizer, fetch client
│   ├── src/model/              # Wire shapes -> render shapes, label tables
│   ├── src/hooks/               # Submit/poll state machine, health preflight
│   ├── src/components/           # Form, progress, result views
│   └── src/styles/                # Design tokens and stylesheet
├── src/dialogue_extractor/
│   ├── domain/              # Pure data models, timing math, error types
│   ├── interfaces/           # Abstract ports (video source, transcriber, diarizer, ASD provider)
│   ├── ingestion/             # URL validation, acquisition, normalization
│   ├── audio/                  # Transcription, diarization, alignment
│   ├── matching/                 # Text normalization and fuzzy phrase matching
│   ├── vision/                     # Clip building, scene detection, active speaker detection, frame extraction
│   ├── pipeline/                     # Orchestration, both modes, confidence scoring, result building
│   ├── api/                            # FastAPI app, routes, request schemas, in-process job store
│   ├── infra/                            # Logging, retry, caching, FFmpeg wrapper
│   └── composition.py                     # Composition root -- wires concrete stages behind the ports
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── approach.md          # Full design and architecture document
│   └── diagrams/
└── output/                  # Generated results and extracted frame images
```

The full, per-module folder structure and its SOLID rationale are in [`docs/approach.md`](docs/approach.md).

## Getting Started

### Prerequisites

- Python 3.11 or later
- A current build of FFmpeg and FFprobe on the system path
- An NVIDIA Build API key and the Active Speaker Detection function ID
- A Hugging Face access token, with the terms of the gated `pyannote` pipeline repositories accepted
- An NVIDIA GPU is optional; it accelerates local transcription and diarization but is not required, since active speaker detection runs on a hosted endpoint

### Installation

```bash
git clone <this-repository-url>
cd quest1-dialogueextractor
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Fill in NVIDIA_API_KEY, ASD_FUNCTION_ID, HF_TOKEN
```

### Usage

Target search, returning the first verified on-screen occurrence:

```bash
python main.py --url "https://ok.ru/video/248244667877" \
                --dialogue "My mind rebels at stagnation"
```

Discovery mode, returning every verified on-screen dialogue:

```bash
python main.py --url "https://ok.ru/video/248244667877"
```

Discovery mode with a candidate budget, for long videos:

```bash
python main.py --url "https://ok.ru/video/248244667877" --max-candidates 10
```

Results are written as JSON to `output/`, alongside the extracted frame images.

Example result:

```json
{
  "status": "success",
  "timestamp": "00:01:42.350",
  "frame_number": 2456,
  "text": "My mind rebels at stagnation",
  "confidence": 0.92,
  "face_detection_confidence": 0.91,
  "image": "frame_2456.png"
}
```

The process exit code tells a calling script what happened without parsing JSON: `0` for a produced frame (`success` or `ambiguous`), `2` for a correct "no" (`not_found` or `not_on_screen`), `1` for a pipeline failure. Run `python main.py --help` for the full flag list -- tuning (`--pad`, `--min-speaking-ms`, `--match-floor`), Mode B budgets (`--max-candidates`, `--time-budget`), and runtime options (`--asd-provider`, `--output-dir`, `--no-cache`, `--quiet`).

### Running the API

```bash
python serve.py --port 8000
```

`create_app()` is a factory rather than a module-level `app`, since tests build
one with the pipeline stubbed out -- so `uvicorn dialogue_extractor.api.app:app`
will not work; `serve.py` (or `uvicorn ...:create_app --factory` for
auto-reload during development) is the entry point. Interactive docs are at
`http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /extract` | Submit a job; returns `202` with a `job_id` immediately -- a real run takes minutes and cannot answer inside the request |
| `GET /jobs/{id}` | Poll a job's stage and, once finished, its result. A correct `not_on_screen` or `not_found` answer is still `200`; only an unknown job id is `404` |
| `GET /jobs` | List submitted jobs (without their results, to keep the listing light) |
| `GET /frames/{filename}` | Fetch an extracted frame image by name |
| `GET /health` | Liveness, plus whether each credential is *configured* (never the value) and the resolved ffmpeg version -- answers even with no credentials set, since that is the one thing it exists to report |

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"url": "https://ok.ru/video/248244667877", "dialogue": "My mind rebels at stagnation"}'
# {"job_id": "...", "state": "running", "mode": "target", "poll": "/jobs/..."}

curl http://localhost:8000/jobs/<job_id>
```

Jobs run on an in-process thread pool and are not persisted -- restarting the
server drops any job in flight. That is a deliberate, stated trade-off (see
`docs/approach.md`), not an oversight: the brief's own scope calls for a single
evaluator-run service, not a job queue with its own infrastructure. The CLI
remains the primary interface; the API is a thin layer over the same
orchestrator, so the two cannot answer the same request differently.

### Running the web interface

The API serves the built frontend itself, so one process gives you both:

```bash
cd frontend && npm install && npm run build
```
```bash
python serve.py --port 8000
```

Open <http://localhost:8000>. Enter a URL and a line of dialogue and it returns
the frame; leave the dialogue empty and it sweeps the whole video, reporting each
candidate as it goes.

`frontend/dist/` is mounted at `/` **only when it exists**, so a backend-only
checkout still starts normally and `/docs` keeps working either way. For
frontend development with hot reload, see [`frontend/README.md`](frontend/README.md).

### Running with Docker

```bash
docker build -t dialogue-extractor .
```

The image's default entrypoint runs the API server:

```bash
docker run --rm --env-file .env -p 8000:8000 \
  -v "$(pwd)/output:/app/output" dialogue-extractor
```

For a one-shot CLI run, override the entrypoint:

```bash
docker run --rm --env-file .env -v "$(pwd)/output:/app/output" \
  --entrypoint python dialogue-extractor main.py \
  --url "https://ok.ru/video/248244667877" \
  --dialogue "My mind rebels at stagnation"
```

The container runs as a non-root user with ffmpeg preinstalled, and installs a
CPU-only build of torch by default -- mount a GPU and swap the base image if
you want local transcription/diarization accelerated; the hosted active-speaker
step is unaffected either way.

### Running Tests

```bash
pytest tests/unit tests/integration -q --cov=src/dialogue_extractor --cov-report=term-missing
```

End-to-end tests, which require live credentials and network access, are opt-in:

```bash
pytest tests/e2e -m e2e -v
```

## Documentation

- [`docs/approach.md`](docs/approach.md) — full design document: problem framing, evaluated approaches with accept/reject reasoning, system architecture, edge cases, assumptions, and customer impact
- [`prompts.txt`](prompts.txt) — AI and LLM prompts used during development
- [`docs/diagrams/`](docs/diagrams/) — architecture, user flow, and system flow diagrams

## Known Limitations and Future Scope

The system assumes a single publicly accessible video per request, the primary audio track, and English by default. A local, fully offline active-speaker-detection fallback (`--asd-provider light`) is wired into the same interface the hosted provider uses -- selectable today, but it raises `NotImplementedError` naming exactly what plugging in real weights would take, rather than being hidden or half-built. The HTTP API's job store is in-process and does not survive a restart, a deliberate scope choice rather than an oversight. The hosted active-speaker service also has an undocumented minimum clip length (~3.5s), discovered empirically; short candidates are padded to it automatically rather than silently dropped. The complete list of assumptions, edge-case handling, and everything discovered during implementation that the original design didn't anticipate is in [`docs/approach.md`](docs/approach.md#implementation-notes-what-the-plan-didnt-know-yet).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ffmpeg` errors on ingestion, or normalization silently produces a broken file | An old system `ffmpeg` (Debian/Ubuntu's default package can be years behind) is shadowing a newer one on PATH | Install a current build and confirm with `ffmpeg -version`; the tool resolves binaries by capability probing, not just by name, but it still needs a modern one present |
| `MissingCredentialError` for `NVIDIA_API_KEY`, `ASD_FUNCTION_ID`, or `HF_TOKEN` | `.env` wasn't copied from `.env.example`, or a key is blank | Check `GET /health` (or the CLI error's `remediation` field) -- it reports which credentials are configured without ever printing their values |
| `GatedModelError` from diarization | The Hugging Face token's account hasn't accepted the `pyannote` pipeline's model terms | Visit the gated repo listed in the error and accept the terms; a token alone is not sufficient |
| Diarization or transcription re-runs from scratch after a code change | The cache fingerprint changed (a different model, quantization, or pipeline version invalidates it on purpose) | Expected behaviour -- a stale cache silently serving an old configuration's output is worse than a slower re-run |
| `POST /extract` seems to hang | It doesn't -- it returns `202` in milliseconds with a `job_id`. The *job* takes minutes | Poll `GET /jobs/{id}`, don't wait on the `POST` call itself |
| A short line of dialogue (one or two words) is never found | Below the detection service's ~3.5s minimum, though candidates are padded automatically -- check `--pad` if using a much shorter value than the default | Increase `--pad`, or leave it at the default |
| `Sign in to confirm you're not a bot` from YouTube | The site has temporarily blocked this address after repeated downloads | Wait a few minutes. Re-running the same video does **not** download again -- acquisition is cached by URL, so only the first run fetches. A local file path also bypasses it entirely |
| A run re-downloads a video you already processed | The cache key includes the normalization settings, so changing resolution or audio rate is a deliberate miss; `--no-cache` also forces a fresh fetch | Expected. Drop `--no-cache`, or clear `.cache/` if you want to start clean |
| A new video returns a previous video's dialogue | Fixed. Downloads once shared one filename and yt-dlp reused the existing file. If you have a `.cache/` from before the fix, it may hold media stored under the wrong key | Delete `.cache/` once; entries created since the fix are correct |
| `docker build` fails installing torch | No network access to `download.pytorch.org`, or a firewall blocking it | Confirm connectivity to that host; the CPU wheel index is required even for a CPU-only build |

## Author

**Rahul V S**

GitHub: [github.com/techieRahul17](https://github.com/techieRahul17)
LinkedIn: [https://www.linkedin.com/in/rahul-v-s/](https://www.linkedin.com/in/rahul-v-s/)

## Contributors

- Rahul V S — design, architecture, and implementation

## Acknowledgments

This project builds on NVIDIA's Active Speaker Detection NIM, `faster-whisper`, `pyannote-audio`, and `yt-dlp`. Full attribution and links to each are listed in the references section of `docs/approach.md`.