"""Runtime configuration: environment and .env, with CLI overrides on top.

Secrets live only here, sourced from the environment. They are never logged,
never embedded in an error message, and never written to the output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import MissingCredentialError

DeviceChoice = Literal["auto", "cuda", "cpu"]


class Settings(BaseSettings):
    """Application settings.

    Precedence, highest first: explicit constructor argument (the CLI passes
    these) > environment variable > .env file > the default declared here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Credentials ---------------------------------------------------------
    nvidia_api_key: SecretStr | None = None
    asd_function_id: str | None = None
    asd_target: str = "grpc.nvcf.nvidia.com:443"
    hf_token: SecretStr | None = None

    # --- Compute -------------------------------------------------------------
    device: DeviceChoice = "auto"
    whisper_model: str = constants.DEFAULT_WHISPER_MODEL
    language: str = "en"
    """Spoken language for transcription: an ISO code ("en", "hi", "fr", ...) or
    "auto" to let Whisper detect it.

    English is the default because naming the language stops a short English
    clip being misdetected. Set "auto" (or the right code) for any other video:
    left at "en", Whisper transcribes foreign speech as poor English, and the
    phrase match fails for a line that is really there.
    """

    # --- Paths ---------------------------------------------------------------
    output_dir: Path = Path("output")
    work_dir: Path = Path("output/work")
    cache_dir: Path = Path(".cache")

    # --- Behaviour -----------------------------------------------------------
    asd_provider: str = "nvidia"
    log_level: str = "INFO"
    use_cache: bool = True

    cors_origins: str = ""
    """Comma-separated origins allowed to call the API cross-origin.

    Empty by default, and empty means the middleware is never installed at all
    rather than installed permissively. In production the built frontend is
    served by this same process, so requests are same-origin and no CORS is
    involved; this exists only for `npm run dev`, where Vite serves the UI from
    its own port. A wildcard is deliberately not supported -- an origin list
    that has to be named is one that has to be thought about.
    """

    frontend_dist: Path = Path("frontend/dist")
    """Where the built frontend lives, served at / when it exists."""

    min_speaking_ms: float = constants.MIN_SPEAKING_MS
    window_pad_seconds: float = constants.WINDOW_PAD_SECONDS
    chunk_max_seconds: float = constants.CHUNK_MAX_SECONDS
    match_score_floor: float = constants.MATCH_SCORE_FLOOR
    max_candidates: int = constants.DEFAULT_MAX_CANDIDATES
    time_budget_seconds: float = constants.DEFAULT_TIME_BUDGET_SECONDS

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # --- Getting past YouTube's bot wall -------------------------------------
    # YouTube refuses anonymous downloads from an address it has decided is
    # automated ("Sign in to confirm you're not a bot"). Two ways through, in
    # increasing order of reliability and inconvenience.

    ytdlp_player_clients: str = "default,android_vr,android"
    """Which YouTube player clients to try, in order.

    Clients are gated differently, and the ones YouTube blocks first are the
    ones yt-dlp reaches for first. Measured against a video that was being
    refused: the default set alone failed, `android_vr` and `android` got
    through, and this list yields the full 1080p stream when the default set
    works while still falling back to the Android clients (360p) when it does
    not. Quality is preferred, availability is guaranteed.

    Set empty to use yt-dlp's own defaults with no fallback.
    """

    ytdlp_cookies_from_browser: str = ""
    """Take cookies from a local browser profile, e.g. "chrome" or "firefox".

    The reliable answer when the client trick stops working, because a signed-in
    request is not anonymous. Note the browser must be **closed**: Chrome holds
    a lock on its cookie database and yt-dlp cannot copy it while it runs.
    Optionally `browser:profile`, e.g. "firefox:default-release".
    """

    ytdlp_cookie_file: Path | None = None
    """Path to a cookies.txt export, for when the browser cannot be closed."""

    def whisper_language(self) -> str | None:
        """The language to pass to Whisper; None means auto-detect."""
        value = self.language.strip().lower()
        return None if value in ("", "auto") else value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log level {v!r}")
        return level

    # --- Credential access ---------------------------------------------------
    # Deliberately explicit accessors rather than bare attribute reads, so a
    # missing key fails with a remediation string instead of a None propagating
    # into a gRPC call and surfacing as an opaque auth error.

    def require_nvidia_api_key(self) -> str:
        if self.nvidia_api_key is None or not self.nvidia_api_key.get_secret_value():
            raise MissingCredentialError(
                "missing_nvidia_api_key",
                "NVIDIA_API_KEY is not set.",
                remediation=(
                    "Create a key at https://build.nvidia.com/nvidia/active-speaker-detection "
                    "and put it in .env as NVIDIA_API_KEY."
                ),
            )
        return self.nvidia_api_key.get_secret_value()

    def require_asd_function_id(self) -> str:
        if not self.asd_function_id:
            raise MissingCredentialError(
                "missing_asd_function_id",
                "ASD_FUNCTION_ID is not set.",
                remediation=(
                    "Copy the function id from the API tab of "
                    "https://build.nvidia.com/nvidia/active-speaker-detection "
                    "into .env as ASD_FUNCTION_ID."
                ),
            )
        return self.asd_function_id

    def require_hf_token(self) -> str:
        if self.hf_token is None or not self.hf_token.get_secret_value():
            raise MissingCredentialError(
                "missing_hf_token",
                "HF_TOKEN is not set.",
                remediation=(
                    "Create a read token at https://huggingface.co/settings/tokens, accept the "
                    f"terms on https://huggingface.co/{constants.PYANNOTE_PIPELINE}, "
                    "then put the token in .env as HF_TOKEN."
                ),
            )
        return self.hf_token.get_secret_value()

    def cors_origin_list(self) -> list[str]:
        """The configured origins, parsed. Empty list means "install nothing"."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.work_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def safe_dump(self) -> dict[str, object]:
        """Config with every secret redacted -- safe to log."""
        data = self.model_dump()
        for key in ("nvidia_api_key", "hf_token"):
            data[key] = "***set***" if data.get(key) is not None else None
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


_settings: Settings | None = None


def get_settings(**overrides: object) -> Settings:
    """Process-wide settings singleton.

    Passing overrides rebuilds it -- that is how the CLI layers its flags on top
    of the environment.
    """
    global _settings
    if overrides or _settings is None:
        _settings = Settings(**overrides)  # type: ignore[arg-type]
    return _settings


def reset_settings() -> None:
    """Drop the singleton. Used by tests to isolate environment mutations."""
    global _settings
    _settings = None
