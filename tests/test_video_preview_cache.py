import os
import tempfile
import unittest
from pathlib import Path

from codoxear.video_preview import prune_video_preview_cache


class TestVideoPreviewCache(unittest.TestCase):
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
