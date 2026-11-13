import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from codoxear import media_preview
from codoxear.media_preview import ensure_media_preview
from codoxear.media_preview import prune_media_preview_cache


class TestVideoPreviewCache(unittest.TestCase):
    def setUp(self) -> None:
        with media_preview._PREVIEW_LOCKS_GUARD:
            media_preview._PREVIEW_LOCKS.clear()
        with media_preview._PREVIEW_FAILURES_GUARD:
            media_preview._PREVIEW_FAILURES.clear()

    def test_prune_removes_oldest_previews_by_file_cap_and_keeps_current(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old1 = root / "old1.mp4"
            old2 = root / "old2.mp4"
            keep = root / "keep.mp4"
            newest = root / "newest.mp4"
            for idx, path in enumerate([old1, old2, keep, newest], start=1):
                path.write_bytes(bytes([idx]) * 4)
                os.utime(path, (1000 + idx, 1000 + idx))

            prune_media_preview_cache(root, keep=keep, max_files=2, max_bytes=0)

            self.assertFalse(old1.exists())
            self.assertFalse(old2.exists())
            self.assertTrue(keep.exists())
            self.assertTrue(newest.exists())

    def test_prune_removes_oldest_previews_by_byte_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "old.mp4"
            keep = root / "keep.mp4"
            fresh = root / "fresh.mp4"
            old.write_bytes(b"o" * 7)
            keep.write_bytes(b"k" * 7)
            fresh.write_bytes(b"f" * 7)
            os.utime(old, (1000, 1000))
            os.utime(keep, (1001, 1001))
            os.utime(fresh, (1002, 1002))

            prune_media_preview_cache(root, keep=keep, max_files=0, max_bytes=14)

            self.assertFalse(old.exists())
            self.assertTrue(keep.exists())
            self.assertTrue(fresh.exists())

    def test_concurrent_preview_generation_singleflights_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "clip.mkv"
            src.write_bytes(b"video")
            preview_dir = root / "previews"
            started = threading.Event()
            release = threading.Event()
            calls: list[list[str]] = []
            results: list[Path] = []
            errors: list[BaseException] = []

            def fake_generate(path: Path, out: Path, *, kind: str, preview_dir: Path) -> Path:
                calls.append([str(path), str(out), str(preview_dir)])
                started.set()
                self.assertTrue(release.wait(timeout=5), "test did not release fake ffmpeg")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"preview")
                return out

            def worker() -> None:
                try:
                    results.append(ensure_media_preview(src, kind="video", preview_dir=preview_dir))
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
            with mock.patch.object(media_preview, "_generate_media_preview", side_effect=fake_generate):
                for thread in threads:
                    thread.start()
                self.assertTrue(started.wait(timeout=5), "fake ffmpeg did not start")
                time.sleep(0.05)
                release.set()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], results[1])
            self.assertEqual(len(calls), 1)
            self.assertTrue(results[0].exists())
            with media_preview._PREVIEW_LOCKS_GUARD:
                self.assertEqual(media_preview._PREVIEW_LOCKS, {})

    def test_failed_preview_generation_is_throttled_until_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "clip.mkv"
            src.write_bytes(b"video")
            preview_dir = root / "previews"
            old_ttl = media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS
            calls = 0

            def failing_generate(path: Path, out: Path, *, kind: str, preview_dir: Path) -> Path:
                nonlocal calls
                calls += 1
                raise RuntimeError("bad codec")

            try:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = 60
                with mock.patch.object(media_preview, "_generate_media_preview", side_effect=failing_generate):
                    with self.assertRaisesRegex(RuntimeError, "bad codec"):
                        ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                    with self.assertRaisesRegex(RuntimeError, "recent media preview generation failed: bad codec"):
                        ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                    self.assertEqual(calls, 1)
                    src.write_bytes(b"video changed")
                    os.utime(src, ns=(time.time_ns() + 1_000_000, time.time_ns() + 1_000_000))
                    with self.assertRaisesRegex(RuntimeError, "bad codec"):
                        ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                    self.assertEqual(calls, 2)
            finally:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = old_ttl

    def test_expired_preview_failures_are_pruned_on_later_activity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview_dir = root / "previews"
            old_ttl = media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS
            old_cap = media_preview.MEDIA_PREVIEW_FAILURE_MAX_ENTRIES
            calls = 0

            def failing_generate(path: Path, out: Path, *, kind: str, preview_dir: Path) -> Path:
                nonlocal calls
                calls += 1
                raise RuntimeError("bad codec")

            try:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = 1
                media_preview.MEDIA_PREVIEW_FAILURE_MAX_ENTRIES = 512
                with mock.patch.object(media_preview, "_generate_media_preview", side_effect=failing_generate):
                    for idx in range(3):
                        src = root / f"old-{idx}.mkv"
                        src.write_bytes(f"video-{idx}".encode())
                        with self.assertRaisesRegex(RuntimeError, "bad codec"):
                            ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                    with media_preview._PREVIEW_FAILURES_GUARD:
                        self.assertEqual(len(media_preview._PREVIEW_FAILURES), 3)
                    time.sleep(1.1)
                    fresh = root / "fresh.mkv"
                    fresh.write_bytes(b"fresh")
                    with self.assertRaisesRegex(RuntimeError, "bad codec"):
                        ensure_media_preview(fresh, kind="video", preview_dir=preview_dir)
                    with media_preview._PREVIEW_FAILURES_GUARD:
                        self.assertEqual(len(media_preview._PREVIEW_FAILURES), 1)
                    self.assertEqual(calls, 4)
            finally:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = old_ttl
                media_preview.MEDIA_PREVIEW_FAILURE_MAX_ENTRIES = old_cap

    def test_preview_failure_throttle_preserves_permission_error_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "clip.mkv"
            src.write_bytes(b"video")
            preview_dir = root / "previews"
            old_ttl = media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS
            calls = 0

            def permission_denied(path: Path, out: Path, *, kind: str, preview_dir: Path) -> Path:
                nonlocal calls
                calls += 1
                raise PermissionError("denied")

            try:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = 60
                with mock.patch.object(media_preview, "_generate_media_preview", side_effect=permission_denied):
                    with self.assertRaises(PermissionError):
                        ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                    with self.assertRaises(PermissionError):
                        ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                self.assertEqual(calls, 2)
                with media_preview._PREVIEW_FAILURES_GUARD:
                    self.assertEqual(media_preview._PREVIEW_FAILURES, {})
            finally:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = old_ttl

    def test_preview_failure_cache_respects_entry_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview_dir = root / "previews"
            old_ttl = media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS
            old_cap = media_preview.MEDIA_PREVIEW_FAILURE_MAX_ENTRIES

            def failing_generate(path: Path, out: Path, *, kind: str, preview_dir: Path) -> Path:
                raise RuntimeError("bad codec")

            try:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = 60
                media_preview.MEDIA_PREVIEW_FAILURE_MAX_ENTRIES = 2
                with mock.patch.object(media_preview, "_generate_media_preview", side_effect=failing_generate):
                    for idx in range(4):
                        src = root / f"clip-{idx}.mkv"
                        src.write_bytes(f"video-{idx}".encode())
                        with self.assertRaisesRegex(RuntimeError, "bad codec"):
                            ensure_media_preview(src, kind="video", preview_dir=preview_dir)
                with media_preview._PREVIEW_FAILURES_GUARD:
                    self.assertLessEqual(len(media_preview._PREVIEW_FAILURES), 2)
            finally:
                media_preview.MEDIA_PREVIEW_FAILURE_TTL_SECONDS = old_ttl
                media_preview.MEDIA_PREVIEW_FAILURE_MAX_ENTRIES = old_cap

    def test_prune_ignores_temp_mp4_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tmp = root / ".abc.tmp.mp4"
            keep = root / "keep.mp4"
            tmp.write_bytes(b"tmp")
            keep.write_bytes(b"keep")

            prune_media_preview_cache(root, keep=keep, max_files=1, max_bytes=1)

            self.assertTrue(tmp.exists())
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()


def test_audio_preview_real_ffmpeg(tmp_path):
    import json
    import shutil
    import subprocess
    import pytest
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    source = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", str(source)], check=True)
    result = ensure_media_preview(source, kind="audio", preview_dir=tmp_path / "cache")
    assert result.suffix == ".mp3" and result.stat().st_size > 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(result)], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)["streams"]
    assert [(stream["codec_type"], stream["codec_name"]) for stream in streams] == [("audio", "mp3")]


def test_prune_combines_audio_and_video_files(tmp_path):
    audio = tmp_path / "a.mp3"
    video = tmp_path / "b.mp4"
    audio.write_bytes(b"a")
    video.write_bytes(b"b")
    os.utime(audio, (10, 10))
    os.utime(video, (20, 20))
    prune_media_preview_cache(tmp_path, max_files=1)
    assert not audio.exists() and video.exists()


def test_media_cache_env_uses_legacy_only_when_new_unset(monkeypatch):
    new = "CODEX_WEB_MEDIA_PREVIEW_MAX_FILES"
    old = "CODEX_WEB_VIDEO_PREVIEW_MAX_FILES"
    monkeypatch.setenv(old, "17")
    monkeypatch.delenv(new, raising=False)
    assert media_preview._positive_int_env(new, 256) == 17
    monkeypatch.setenv(new, "23")
    assert media_preview._positive_int_env(new, 256) == 23
