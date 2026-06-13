import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from codoxear import video_preview
from codoxear.video_preview import ensure_video_preview
from codoxear.video_preview import prune_video_preview_cache


class TestVideoPreviewCache(unittest.TestCase):
    def setUp(self) -> None:
        with video_preview._PREVIEW_LOCKS_GUARD:
            video_preview._PREVIEW_LOCKS.clear()
        with video_preview._PREVIEW_FAILURES_GUARD:
            video_preview._PREVIEW_FAILURES.clear()

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

            prune_video_preview_cache(root, keep=keep, max_files=2, max_bytes=0)

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

            prune_video_preview_cache(root, keep=keep, max_files=0, max_bytes=14)

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

            def fake_generate(path: Path, out: Path, *, preview_dir: Path) -> Path:
                calls.append([str(path), str(out), str(preview_dir)])
                started.set()
                self.assertTrue(release.wait(timeout=5), "test did not release fake ffmpeg")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"preview")
                return out

            def worker() -> None:
                try:
                    results.append(ensure_video_preview(src, preview_dir=preview_dir))
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
            with mock.patch.object(video_preview, "_generate_video_preview", side_effect=fake_generate):
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
            with video_preview._PREVIEW_LOCKS_GUARD:
                self.assertEqual(video_preview._PREVIEW_LOCKS, {})

    def test_failed_preview_generation_is_throttled_until_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "clip.mkv"
            src.write_bytes(b"video")
            preview_dir = root / "previews"
            old_ttl = video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS
            calls = 0

            def failing_generate(path: Path, out: Path, *, preview_dir: Path) -> Path:
                nonlocal calls
                calls += 1
                raise RuntimeError("bad codec")

            try:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = 60
                with mock.patch.object(video_preview, "_generate_video_preview", side_effect=failing_generate):
                    with self.assertRaisesRegex(RuntimeError, "bad codec"):
                        ensure_video_preview(src, preview_dir=preview_dir)
                    with self.assertRaisesRegex(RuntimeError, "recent video preview generation failed: bad codec"):
                        ensure_video_preview(src, preview_dir=preview_dir)
                    self.assertEqual(calls, 1)
                    src.write_bytes(b"video changed")
                    os.utime(src, ns=(time.time_ns() + 1_000_000, time.time_ns() + 1_000_000))
                    with self.assertRaisesRegex(RuntimeError, "bad codec"):
                        ensure_video_preview(src, preview_dir=preview_dir)
                    self.assertEqual(calls, 2)
            finally:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = old_ttl

    def test_expired_preview_failures_are_pruned_on_later_activity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview_dir = root / "previews"
            old_ttl = video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS
            old_cap = video_preview.VIDEO_PREVIEW_FAILURE_MAX_ENTRIES
            calls = 0

            def failing_generate(path: Path, out: Path, *, preview_dir: Path) -> Path:
                nonlocal calls
                calls += 1
                raise RuntimeError("bad codec")

            try:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = 1
                video_preview.VIDEO_PREVIEW_FAILURE_MAX_ENTRIES = 512
                with mock.patch.object(video_preview, "_generate_video_preview", side_effect=failing_generate):
                    for idx in range(3):
                        src = root / f"old-{idx}.mkv"
                        src.write_bytes(f"video-{idx}".encode())
                        with self.assertRaisesRegex(RuntimeError, "bad codec"):
                            ensure_video_preview(src, preview_dir=preview_dir)
                    with video_preview._PREVIEW_FAILURES_GUARD:
                        self.assertEqual(len(video_preview._PREVIEW_FAILURES), 3)
                    time.sleep(1.1)
                    fresh = root / "fresh.mkv"
                    fresh.write_bytes(b"fresh")
                    with self.assertRaisesRegex(RuntimeError, "bad codec"):
                        ensure_video_preview(fresh, preview_dir=preview_dir)
                    with video_preview._PREVIEW_FAILURES_GUARD:
                        self.assertEqual(len(video_preview._PREVIEW_FAILURES), 1)
                    self.assertEqual(calls, 4)
            finally:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = old_ttl
                video_preview.VIDEO_PREVIEW_FAILURE_MAX_ENTRIES = old_cap

    def test_preview_failure_throttle_preserves_permission_error_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "clip.mkv"
            src.write_bytes(b"video")
            preview_dir = root / "previews"
            old_ttl = video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS
            calls = 0

            def permission_denied(path: Path, out: Path, *, preview_dir: Path) -> Path:
                nonlocal calls
                calls += 1
                raise PermissionError("denied")

            try:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = 60
                with mock.patch.object(video_preview, "_generate_video_preview", side_effect=permission_denied):
                    with self.assertRaises(PermissionError):
                        ensure_video_preview(src, preview_dir=preview_dir)
                    with self.assertRaises(PermissionError):
                        ensure_video_preview(src, preview_dir=preview_dir)
                self.assertEqual(calls, 2)
                with video_preview._PREVIEW_FAILURES_GUARD:
                    self.assertEqual(video_preview._PREVIEW_FAILURES, {})
            finally:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = old_ttl

    def test_preview_failure_cache_respects_entry_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview_dir = root / "previews"
            old_ttl = video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS
            old_cap = video_preview.VIDEO_PREVIEW_FAILURE_MAX_ENTRIES

            def failing_generate(path: Path, out: Path, *, preview_dir: Path) -> Path:
                raise RuntimeError("bad codec")

            try:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = 60
                video_preview.VIDEO_PREVIEW_FAILURE_MAX_ENTRIES = 2
                with mock.patch.object(video_preview, "_generate_video_preview", side_effect=failing_generate):
                    for idx in range(4):
                        src = root / f"clip-{idx}.mkv"
                        src.write_bytes(f"video-{idx}".encode())
                        with self.assertRaisesRegex(RuntimeError, "bad codec"):
                            ensure_video_preview(src, preview_dir=preview_dir)
                with video_preview._PREVIEW_FAILURES_GUARD:
                    self.assertLessEqual(len(video_preview._PREVIEW_FAILURES), 2)
            finally:
                video_preview.VIDEO_PREVIEW_FAILURE_TTL_SECONDS = old_ttl
                video_preview.VIDEO_PREVIEW_FAILURE_MAX_ENTRIES = old_cap

    def test_prune_ignores_temp_mp4_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tmp = root / ".abc.tmp.mp4"
            keep = root / "keep.mp4"
            tmp.write_bytes(b"tmp")
            keep.write_bytes(b"keep")

            prune_video_preview_cache(root, keep=keep, max_files=1, max_bytes=1)

            self.assertTrue(tmp.exists())
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
