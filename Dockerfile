# syntax=docker/dockerfile:1

# Two stages, for one reason: the build needs compilers to install wheels that
# have no manylinux build, and shipping those compilers would roughly double the
# image while adding tools an inference service has no business carrying.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CPU-only torch. The default index would pull the CUDA build -- several
# gigabytes of driver libraries that cannot work without a GPU on the host, and
# are dead weight when there is one, since the base image has no CUDA runtime.
# Mount a GPU and use the CUDA base image instead if you want acceleration.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


FROM python:3.11-slim-bookworm AS runtime

# ffmpeg is not optional and not a detail: the whole frame-accuracy argument
# rests on forcing CFR at ingestion, and a stale build silently fails on modern
# codecs. Debian bookworm ships 5.1.x, which is new enough for every filter and
# flag this pipeline uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The service downloads files named by a remote server and hands
# filenames back over HTTP, so the process should own as little as possible.
RUN useradd --create-home --uid 10001 extractor

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # Model weights land here. Without an explicit home the caches scatter into
    # the layer and are re-downloaded on every container start.
    HF_HOME=/home/extractor/.cache/huggingface \
    OUTPUT_DIR=/app/output \
    WORK_DIR=/app/output/work \
    CACHE_DIR=/app/.cache

WORKDIR /app

COPY --chown=extractor:extractor src/ ./src/
COPY --chown=extractor:extractor main.py serve.py ./

RUN mkdir -p /app/output /app/.cache /home/extractor/.cache \
    && chown -R extractor:extractor /app /home/extractor

USER extractor

EXPOSE 8000

# Hits the one endpoint that answers without credentials, a model, or a GPU --
# anything heavier would report unhealthy on a correctly configured service that
# simply has no API key yet.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Defaults to the API. Override the entrypoint for a one-shot CLI run:
#   docker run --rm --env-file .env -v "$PWD/output:/app/output" \
#     --entrypoint python IMAGE main.py --url URL --dialogue "a line"
ENTRYPOINT ["python", "serve.py"]
