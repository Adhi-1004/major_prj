"""How the app is assembled: CORS, and serving a built frontend.

Both are conditional, and the condition is the interesting part. A CORS policy
that installs itself by default is a policy nobody chose, and a static mount
that assumes its directory exists turns "I haven't built the frontend yet" into
a server that will not start.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dialogue_extractor.api.app import create_app
from dialogue_extractor.config.settings import Settings


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    """Settings that cannot see the developer's real .env."""
    base: dict[str, object] = {
        "_env_file": None,
        "output_dir": tmp_path / "output",
        "work_dir": tmp_path / "work",
        "cache_dir": tmp_path / "cache",
        "frontend_dist": tmp_path / "dist",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class TestCorsIsOptIn:
    """Unset means the middleware is never installed, not installed permissively."""

    def test_absent_by_default(self, tmp_path: Path) -> None:
        client = TestClient(create_app(settings_for(tmp_path)))
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_present_when_an_origin_is_named(self, tmp_path: Path) -> None:
        client = TestClient(
            create_app(settings_for(tmp_path, cors_origins="http://localhost:5173"))
        )
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_an_unnamed_origin_is_not_allowed(self, tmp_path: Path) -> None:
        """The point of naming origins is that unnamed ones are refused."""
        client = TestClient(
            create_app(settings_for(tmp_path, cors_origins="http://localhost:5173"))
        )
        response = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers

    def test_several_origins_parse(self, tmp_path: Path) -> None:
        settings = settings_for(
            tmp_path, cors_origins="http://localhost:5173, http://127.0.0.1:5173"
        )
        assert settings.cors_origin_list() == [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    @pytest.mark.parametrize("raw", ["", "   ", ",", " , "])
    def test_blank_configurations_install_nothing(
        self, raw: str, tmp_path: Path
    ) -> None:
        """A stray comma must not be read as an origin."""
        assert settings_for(tmp_path, cors_origins=raw).cors_origin_list() == []

    def test_the_preflight_is_answered_when_enabled(self, tmp_path: Path) -> None:
        client = TestClient(
            create_app(settings_for(tmp_path, cors_origins="http://localhost:5173"))
        )
        response = client.options(
            "/extract",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200


class TestFrontendMount:
    def test_a_missing_dist_does_not_break_startup(self, tmp_path: Path) -> None:
        """Every backend test, and every CLI-only user, is in this case."""
        client = TestClient(create_app(settings_for(tmp_path)))
        assert client.get("/health").status_code == 200

    def test_a_built_frontend_is_served_at_the_root(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>UI</title>")

        client = TestClient(create_app(settings_for(tmp_path)))
        response = client.get("/")
        assert response.status_code == 200
        assert "UI" in response.text

    def test_assets_are_served(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html>")
        (dist / "assets" / "app.js").write_text("export default 1;")

        client = TestClient(create_app(settings_for(tmp_path)))
        assert client.get("/assets/app.js").status_code == 200

    def test_api_routes_are_not_shadowed_by_the_catch_all(
        self, tmp_path: Path
    ) -> None:
        """A mount at "/" is a catch-all. Registered before the router it would
        swallow every endpoint, which is why it is mounted last."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>UI</title>")

        client = TestClient(create_app(settings_for(tmp_path)))

        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/jobs").json() == {"jobs": [], "count": 0}
        assert client.get("/jobs/nope").status_code == 404
        assert client.get("/openapi.json").status_code == 200

    def test_docs_still_work_behind_the_mount(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html>")

        client = TestClient(create_app(settings_for(tmp_path)))
        assert client.get("/docs").status_code == 200

    def test_a_file_that_is_not_a_directory_is_ignored(self, tmp_path: Path) -> None:
        """`frontend/dist` pointing at a file must not crash the app."""
        stray = tmp_path / "dist"
        stray.write_text("not a directory")

        client = TestClient(create_app(settings_for(tmp_path)))
        assert client.get("/health").status_code == 200
