# Frontend

A React + Vite + TypeScript interface for the Dialogue Frame Extractor. It
submits a job, follows it while it runs, and renders every result state the API
can produce.

Runtime dependencies are `react` and `react-dom`. There is no router, no state
library, no component kit and no CSS framework — the styling is plain CSS with
custom properties in `src/styles/`.

## Running it

**Production** — the API serves the built assets itself, so there is one process
and one origin:

```bash
npm install && npm run build
```
```bash
cd .. && .venv/Scripts/python.exe serve.py --port 8000
```

Then open <http://localhost:8000>. The API mounts `frontend/dist/` at `/` when
it exists, and only when it exists, so a backend-only checkout still starts.

**Development** — two processes, with Vite proxying the API:

```bash
.venv/Scripts/python.exe serve.py --port 8000
```
```bash
cd frontend && npm run dev
```

Then open <http://localhost:5173>. The proxy in `vite.config.ts` forwards
`/extract`, `/jobs`, `/frames` and `/health` to `127.0.0.1:8000`, so the browser
still only talks to one origin and no CORS configuration is needed.

If you would rather point the dev server at an API on another host, set
`cors_origins` in the backend's `.env`:

```
CORS_ORIGINS=http://localhost:5173
```

That is the only situation where CORS is involved. The middleware is not
installed at all when the setting is empty.

## Layout

```
src/
├── api/
│   ├── types.ts      1:1 mirror of the JSON the API sends
│   ├── errors.ts     normalizes the three different error envelopes
│   └── client.ts     fetch wrappers, frame URLs
├── model/
│   ├── view.ts       wire shapes -> what components render
│   └── labels.ts     machine slugs -> prose
├── hooks/
│   ├── useExtraction.ts   submit, poll, and the phase machine
│   └── useHealth.ts       one preflight check on load
├── components/
└── styles/
```

The interesting file is `model/view.ts`. The API's `status` field does not
determine a result's shape — a `not_on_screen` verdict can arrive with a frame
and a full confidence breakdown (a verified frame that scored below 0.45) or
with neither. So `view.ts` splits every result along two independent axes:

- **verdict** — what the system concluded, from `status` alone
- **evidence** — whether there is a frame to show, from the fields present

Components then never infer one from the other, which is the mistake that would
render a real frame as an empty state or vice versa.

Two related details it also handles:

- The `ambiguous` array can contain items whose own status is `not_on_screen`.
  Rows are labelled from `item.status`; the array only decides which section they
  sit in.
- In discovery mode the confidence weights sum to **0.60**, not 1.0, because
  there is no target phrase to score a match against. Contributions are
  `value * weight / sum(weights)` rather than assuming a unit total.

## Notes

- `npm run typecheck` runs TypeScript alone. `strict`,
  `exactOptionalPropertyTypes` and `noUncheckedIndexedAccess` are all on, which
  is what makes the type mirror worth having.
- The one hero frame loads eagerly; grid thumbnails are lazy and their cards use
  `content-visibility: auto`. A discovery sweep can produce a dozen 1920×1080
  stills, and the off-camera bucket — usually the largest — references no images
  at all, because no frame was ever selected for those lines.
- Frame URLs carry `?j=<job_id>`. The server ignores it; it scopes the browser
  cache per job, since frames are named only by frame number and two runs can
  produce the same filename.
