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

## Project Structure

```
quest1-dialogueextractor/
├── main.py                 # CLI entry point
├── src/dialogue_extractor/
│   ├── domain/              # Pure data models, timing math, error types
│   ├── interfaces/           # Abstract ports (video source, transcriber, diarizer, ASD provider)
│   ├── ingestion/             # URL validation, acquisition, normalization
│   ├── audio/                  # Transcription, diarization, alignment
│   ├── matching/                 # Text normalization and fuzzy phrase matching
│   ├── vision/                     # Clip building, scene detection, active speaker detection, frame extraction
│   ├── pipeline/                     # Orchestration, both modes, confidence scoring, result building
│   ├── api/                            # FastAPI application and routes
│   └── infra/                            # Logging, retry, caching, FFmpeg wrapper
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── approach.md          # Full design and architecture document
│   ├── PROMPTS.md           # AI/LLM prompts used during development
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

### Running the API

```bash
uvicorn dialogue_extractor.api.app:app --reload
```

Exposes `POST /extract`, `GET /jobs/{id}`, and `GET /health`. The CLI remains the primary interface; the API is a thin layer over the same pipeline.

### Running with Docker

```bash
docker build -t dialogue-extractor .
docker run --rm --env-file .env -v "$(pwd)/output:/app/output" \
  dialogue-extractor --url "https://ok.ru/video/248244667877" \
                      --dialogue "My mind rebels at stagnation"
```

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
- [`docs/PROMPTS.md`](docs/PROMPTS.md) — AI and LLM prompts used during development
- [`docs/diagrams/`](docs/diagrams/) — architecture, user flow, and system flow diagrams

## Known Limitations and Future Scope

The system assumes a single publicly accessible video per request, the primary audio track, and English by default. A local, fully offline active-speaker-detection fallback is documented as viable but not implemented in this submission. The complete list of assumptions, edge-case handling, and future scope is in `docs/approach.md`.

## Author

**Rahul V S**

GitHub: [github.com/techieRahul17](https://github.com/techieRahul17)
LinkedIn: [https://www.linkedin.com/in/rahul-v-s/](https://www.linkedin.com/in/rahul-v-s/)

## Contributors

- Rahul V S — design, architecture, and implementation

## Acknowledgments

This project builds on NVIDIA's Active Speaker Detection NIM, `faster-whisper`, `pyannote-audio`, and `yt-dlp`. Full attribution and links to each are listed in the references section of `docs/approach.md`.