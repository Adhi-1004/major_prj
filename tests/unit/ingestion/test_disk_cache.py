"""Disk cache behaviour.

The governing rule: a cache is an optimization, so no cache failure may ever
fail the pipeline. Corruption, an interrupted write, an unserializable value --
all degrade to a miss.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_extractor.infra.disk_cache import DiskCache, NullCache, make_key


class TestMakeKey:
    def test_is_deterministic(self) -> None:
        assert make_key("a", "b") == make_key("a", "b")

    def test_differs_by_content(self) -> None:
        assert make_key("a", "b") != make_key("a", "c")

    def test_parts_are_unambiguously_separated(self) -> None:
        """Without a separator, ("ab","c") and ("a","bc") would collide."""
        assert make_key("ab", "c") != make_key("a", "bc")

    def test_is_filename_safe(self) -> None:
        key = make_key("https://ok.ru/video/1?x=/../y")
        assert key.isalnum()
        assert len(key) == 32


class TestGetPut:
    def test_round_trips_a_value(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("k1", {"words": [{"text": "hi", "start": 0.0}]})
        assert cache.get("k1") == {"words": [{"text": "hi", "start": 0.0}]}

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        assert DiskCache(tmp_path).get("absent") is None

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("k", {"v": 1})
        cache.put("k", {"v": 2})
        assert cache.get("k") == {"v": 2}

    def test_keys_are_independent(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2


class TestFailureIsAlwaysAMiss:
    def test_corrupt_entry_is_evicted_not_raised(self, tmp_path: Path) -> None:
        """A half-written file from an interrupted run must not fail a later one."""
        cache = DiskCache(tmp_path)
        cache.put("k", {"v": 1})
        (tmp_path / "k.json").write_text("{not valid json", encoding="utf-8")

        assert cache.get("k") is None
        assert not (tmp_path / "k.json").exists(), "corrupt entry should be removed"

    def test_truncated_entry_is_a_miss(self, tmp_path: Path) -> None:
        (tmp_path / "k.json").write_text('{"words": [', encoding="utf-8")
        assert DiskCache(tmp_path).get("k") is None

    def test_unserializable_value_does_not_raise(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("k", {"handle": object()})  # not JSON-serializable
        assert cache.get("k") is None

    def test_no_temp_file_is_left_behind_after_a_failed_write(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("k", {"handle": object()})
        assert list(tmp_path.glob("*.tmp")) == []


class TestAtomicity:
    def test_write_is_atomic(self, tmp_path: Path) -> None:
        """No .tmp survives a successful write either."""
        cache = DiskCache(tmp_path)
        cache.put("k", {"v": 1})
        assert list(tmp_path.glob("*.tmp")) == []
        assert (tmp_path / "k.json").exists()


class TestFileArtefacts:
    def test_path_for_is_stable(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        assert cache.path_for("k", ".wav") == cache.path_for("k", "wav")

    def test_stores_and_reports_a_file(self, tmp_path: Path) -> None:
        source = tmp_path / "in.wav"
        source.write_bytes(b"audio")
        cache = DiskCache(tmp_path / "cache")

        stored = cache.store_file("k", source, ".wav")
        assert stored.exists()
        assert stored.read_bytes() == b"audio"
        assert cache.has_file("k", ".wav") is True

    def test_has_file_is_false_for_absent_and_empty(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        assert cache.has_file("k", ".wav") is False
        cache.path_for("k", ".wav").write_bytes(b"")
        assert cache.has_file("k", ".wav") is False


class TestEviction:
    def test_evict_removes_value_and_artefacts(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("k", {"v": 1})
        cache.path_for("k", ".wav").write_bytes(b"audio")

        cache.evict("k")
        assert cache.get("k") is None
        assert not cache.path_for("k", ".wav").exists()

    def test_clear_empties_the_cache(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.root.is_dir()


class TestDisabled:
    def test_disabled_cache_never_stores(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path, enabled=False)
        cache.put("k", {"v": 1})
        assert cache.get("k") is None

    def test_disabled_store_file_returns_the_original(self, tmp_path: Path) -> None:
        source = tmp_path / "in.wav"
        source.write_bytes(b"audio")
        assert DiskCache(tmp_path, enabled=False).store_file("k", source, ".wav") == source


class TestNullCache:
    def test_satisfies_the_same_shape(self) -> None:
        cache = NullCache()
        cache.put("k", {"v": 1})
        assert cache.get("k") is None
        assert cache.path_for("k", ".wav").name == "k.wav"


class TestVersioning:
    def test_version_is_part_of_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bumping the version must invalidate entries whose meaning changed."""
        import dialogue_extractor.infra.disk_cache as module

        before = make_key("audio-hash")
        monkeypatch.setattr(module, "_CACHE_VERSION", "v2")
        assert make_key("audio-hash") != before
