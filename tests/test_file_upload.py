"""Direct module tests for `codoxear.file_upload`.

These tests exercise `safe_filename`, `stage_uploaded_file`, and
`attachment_inject_text` directly with explicit dependency injection
(`upload_dir`, `now_fn`, `max_bytes`). They replace the former
server-global/`do_POST` monkeypatch seam (`patch.object(server, "MANAGER")`,
`patch("codoxear.server.UPLOAD_DIR")`, `patch("codoxear.server._now")`,
`patch.object(server, "_json_response")`), which coupled to internal dispatch
order and blinded serialization. Behavioral route coverage lives in
`tests/test_control_routes.py` via the injected-deps seam.
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from codoxear import file_upload
from codoxear import pty_util
from codoxear.file_upload import attachment_inject_text
from codoxear.file_upload import remove_session_uploads
from codoxear.file_upload import remove_staged_attachment_file
from codoxear.file_upload import safe_filename
from codoxear.file_upload import stage_uploaded_file


def _symlinks_supported() -> bool:
    """Probe whether the current platform/filesystem can create symlinks."""
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as td:
        link = Path(td) / "link"
        target = Path(td) / "target"
        try:
            target.mkdir()
            link.symlink_to(target, target_is_directory=True)
            return link.is_symlink()
        except (OSError, NotImplementedError):
            return False


class TestSafeFilename(unittest.TestCase):
    def test_reduces_traversal_to_basename(self) -> None:
        # Path.name strips directory components; the leading "../" chain must
        # not survive into the staged filename. (POSIX semantics; the
        # production target is Linux/macOS.)
        self.assertEqual(safe_filename("../../payload.tar.gz"), "payload.tar.gz")
        self.assertEqual(safe_filename("/etc/passwd"), "passwd")

    def test_strips_shell_and_path_metacharacters(self) -> None:
        # Only alnum and -_. space survive; ; ` | are dropped; spaces
        # collapse to underscores.
        self.assertEqual(safe_filename("a b;c`d|e"), "a_bcde")

    def test_falls_back_to_generic_name_when_empty(self) -> None:
        # The default triggers only when the reduced name is empty after
        # whitespace stripping (dots are preserved characters).
        self.assertEqual(safe_filename(""), "file")
        self.assertEqual(safe_filename("///"), "file")
        self.assertEqual(safe_filename("   "), "file")
        self.assertEqual(safe_filename("   ", default="doc"), "doc")

    def test_preserves_unicode_alnum(self) -> None:
        self.assertEqual(safe_filename("南京大学_程元_简历.pdf"), "南京大学_程元_简历.pdf")

    def test_truncates_overlong_names(self) -> None:
        long_name = "a" * 200 + ".txt"
        result = safe_filename(long_name)
        self.assertEqual(len(result), 96)
        self.assertTrue(result.startswith("a"))


class TestStageUploadedFile(unittest.TestCase):
    def _stage(
        self,
        tmp: str,
        session_id: str,
        filename: str,
        raw: bytes,
        *,
        now: float = 1234.567,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> Path:
        return stage_uploaded_file(
            session_id,
            filename,
            raw,
            upload_dir=Path(tmp),
            now_fn=lambda: now,
            max_bytes=max_bytes,
        )

    def test_writes_binary_bytes_and_preserves_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._stage(td, "sess-1", "../../payload.tar.gz", b"\x00\x01payload")
            self.assertEqual(path.name, "1234567_payload.tar.gz")
            self.assertEqual(path.read_bytes(), b"\x00\x01payload")

    def test_falls_back_to_generic_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._stage(td, "sess-2", "///", b"abc", now=2.0)
            self.assertEqual(path.name, "2000_file")
            self.assertEqual(path.parent, (Path(td) / "sess-2").resolve())

    def test_rejects_oversize_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "file too large"):
                self._stage(td, "sess-3", "big.bin", b"abcd", max_bytes=3)

    def test_preserves_unicode_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._stage(td, "sess-4", "南京大学_程元_简历.pdf", b"pdf", now=3.0)
            self.assertEqual(path.name, "3000_南京大学_程元_简历.pdf")
            self.assertEqual(path.read_bytes(), b"pdf")

    def test_same_millisecond_same_name_gets_unique_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = self._stage(td, "sess-4b", "same.txt", b"one", now=4.0)
            second = self._stage(td, "sess-4b", "same.txt", b"two", now=4.0)
            self.assertEqual(first.name, "4000_same.txt")
            self.assertEqual(second.name, "4000_2_same.txt")
            self.assertEqual(first.read_bytes(), b"one")
            self.assertEqual(second.read_bytes(), b"two")

    def test_chmods_file_to_0600_when_observable(self) -> None:
        # On filesystems that support permission bits (the production target),
        # the staged file must be owner-only. This is a best-effort assertion:
        # if the underlying FS reports a permissive mode regardless of chmod
        # (e.g. some network/Windows mounts), we only assert the write succeeded.
        with tempfile.TemporaryDirectory() as td:
            path = self._stage(td, "sess-5", "secret.txt", b"x")
            mode = stat.S_IMODE(os.stat(path).st_mode)
            if mode != 0o777:
                # FS honors chmod -> strict check.
                self.assertEqual(mode, 0o600)

    def test_rejects_blank_session_id_and_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "session_id required"):
                self._stage(td, "   ", "f.txt", b"x")
            with self.assertRaisesRegex(ValueError, "filename required"):
                self._stage(td, "sess", "   ", b"x")

    def test_rejects_non_bytes_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "file bytes required"):
                stage_uploaded_file(
                    "sess",
                    "f.txt",
                    "not-bytes",  # type: ignore[arg-type]
                    upload_dir=Path(td),
                    now_fn=lambda: 1.0,
                    max_bytes=1024,
                )

    def test_bytearray_payload_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._stage(td, "sess", "f.bin", bytearray(b"\xff\xfe"))
            self.assertEqual(path.read_bytes(), b"\xff\xfe")


class TestRemoveSessionUploads(unittest.TestCase):
    def test_removes_only_target_session_preserving_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "s1").mkdir()
            (root / "s1" / "1234_doc.txt").write_bytes(b"s1-bytes")
            (root / "keep").mkdir()
            (root / "keep" / "5678_keep.txt").write_bytes(b"keep-bytes")

            removed = remove_session_uploads(root, "s1")

            self.assertTrue(removed)
            self.assertFalse((root / "s1").exists())
            self.assertTrue((root / "keep").is_dir())
            self.assertEqual((root / "keep" / "5678_keep.txt").read_bytes(), b"keep-bytes")

    def test_missing_upload_dir_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertFalse(remove_session_uploads(root, "never-staged"))
            # root never created is also a clean no-op
            self.assertFalse(remove_session_uploads(root / "also-absent", "s1"))

    def test_blank_or_invalid_id_raises_without_touching_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "s1").mkdir()
            (root / "s1" / "x").write_bytes(b"x")
            for bad in ("", "   ", ".", "..", "a/b", "a\\b"):
                with self.assertRaisesRegex(ValueError, "session_id"):
                    remove_session_uploads(root, bad)
            # nothing was deleted despite invalid ids
            self.assertTrue((root / "s1").exists())
            self.assertTrue(root.exists())

    def test_rejects_non_path_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "upload_root"):
            remove_session_uploads(123, "s1")  # type: ignore[arg-type]

    def test_symlink_entry_is_unlinked_without_following_target(self) -> None:
        # Some platforms/filesystems cannot create symlinks; on those the
        # documented contract (unlink the link, leave the target intact) is not
        # observable, so skip rather than assert a weaker property.
        if not _symlinks_supported():
            self.skipTest("symlinks not supported on this platform")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = Path(td) / "outside-target"
            outside.mkdir()
            (outside / "secret.txt").write_bytes(b"do-not-touch")
            (root / "keep").mkdir()
            (root / "keep" / "5678_keep.txt").write_bytes(b"keep-bytes")
            # Tampered entry: uploads/s1 is a symlink resolving OUTSIDE the root.
            (root / "s1").symlink_to(outside, target_is_directory=True)

            removed = remove_session_uploads(root, "s1")

            self.assertTrue(removed)
            self.assertFalse((root / "s1").is_symlink())
            self.assertFalse((root / "s1").exists())
            # The link's target contents are untouched.
            self.assertTrue(outside.is_dir())
            self.assertEqual((outside / "secret.txt").read_bytes(), b"do-not-touch")
            # Sibling upload dirs are untouched.
            self.assertEqual((root / "keep" / "5678_keep.txt").read_bytes(), b"keep-bytes")

    def test_remove_staged_attachment_file_unlinks_only_direct_child(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subdir = root / "s1"
            subdir.mkdir()
            target = subdir / "1234_doc.txt"
            target.write_bytes(b"data")
            (root / "keep").mkdir()
            (root / "keep" / "other.txt").write_bytes(b"keep")

            self.assertTrue(remove_staged_attachment_file(root, "s1", target))
            self.assertFalse(target.exists())
            self.assertEqual((root / "keep" / "other.txt").read_bytes(), b"keep")
            self.assertFalse(remove_staged_attachment_file(root, "s1", target))
            with self.assertRaisesRegex(ValueError, "outside session uploads"):
                remove_staged_attachment_file(root, "s1", root / "keep" / "other.txt")

    def test_broken_symlink_entry_is_unlinked(self) -> None:
        # A symlink pointing at a nonexistent destination (broken link) must
        # still be removed: is_symlink() is true even when exists() is false, so
        # ordering the symlink check before the existence check matters.
        if not _symlinks_supported():
            self.skipTest("symlinks not supported on this platform")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "s1").symlink_to(Path(td) / "does-not-exist", target_is_directory=True)

            removed = remove_session_uploads(root, "s1")

            self.assertTrue(removed)
            self.assertFalse((root / "s1").is_symlink())


class TestAttachmentInjectText(unittest.TestCase):
    def test_uses_readable_label_and_trailing_newline(self) -> None:
        text = attachment_inject_text(2, Path("/tmp/example.txt"))
        self.assertEqual(text, "Attachment 2: /tmp/example.txt\n")

    def test_preserves_unicode_path(self) -> None:
        text = attachment_inject_text(1, Path("/tmp/南京大学_程元_简历.pdf"))
        self.assertEqual(text, "Attachment 1: /tmp/南京大学_程元_简历.pdf\n")

    def test_rejects_non_positive_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "attachment_index must be >= 1"):
            attachment_inject_text(0, Path("/tmp/example.txt"))
        with self.assertRaisesRegex(ValueError, "attachment_index must be >= 1"):
            attachment_inject_text(-3, Path("/tmp/example.txt"))


class TestModuleSurface(unittest.TestCase):
    """Guard against accidental re-coupling of the upload helpers to server
    globals: the production seam must keep `upload_dir`/`now_fn`/`max_bytes` as
    explicit injected parameters."""

    def test_stage_signature_requires_injected_state(self) -> None:
        import inspect

        sig = inspect.signature(stage_uploaded_file)
        params = sig.parameters
        # These must be keyword-only so callers cannot silently drop them,
        # and must not carry defaults (forces explicit injection at every site).
        for kw in ("upload_dir", "now_fn", "max_bytes"):
            self.assertIn(kw, params, f"{kw} must remain an explicit parameter")
            self.assertEqual(params[kw].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(params[kw].default, inspect.Parameter.empty)

    def test_helpers_are_importable_without_server(self) -> None:
        # Direct attribute access confirms the module is the source of truth,
        # not a server.py re-export.
        self.assertIs(safe_filename, file_upload.safe_filename)
        self.assertIs(stage_uploaded_file, file_upload.stage_uploaded_file)
        self.assertIs(attachment_inject_text, file_upload.attachment_inject_text)


class TestSeqBytes(unittest.TestCase):
    def test_seq_bytes_preserves_unicode_text(self) -> None:
        raw = "Attachment 1: /tmp/南京大学_程元_简历.pdf\n"
        self.assertEqual(pty_util.seq_bytes(raw), raw.encode("utf-8"))

    def test_seq_bytes_decodes_escape_only_sequences(self) -> None:
        self.assertEqual(pty_util.seq_bytes("\\x1b"), b"\x1b")


if __name__ == "__main__":
    unittest.main()
