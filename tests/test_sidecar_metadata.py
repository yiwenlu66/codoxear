from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from codoxear import sidecar_metadata


class TestSidecarMetadata(unittest.TestCase):
    def test_read_metadata_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta_path = root / "fixture.json"
            sock = root / "fixture.sock"
            meta_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid metadata json"):
                sidecar_metadata.read_metadata(meta_path, sock=sock)

    def test_required_int_rejects_bool(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid broker_pid"):
            sidecar_metadata.required_int({"broker_pid": True}, "broker_pid", sock=Path("fixture.sock"))

    def test_log_path_allows_null_and_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertIsNone(sidecar_metadata.log_path({"log_path": None}, sock=root / "fixture.sock"))
            with self.assertRaisesRegex(ValueError, "invalid log_path"):
                sidecar_metadata.log_path({"log_path": str(root)}, sock=root / "fixture.sock")

    def test_start_ts_rejects_non_finite_numbers(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "invalid start_ts"):
                    sidecar_metadata.start_ts({"start_ts": value}, sock=Path("fixture.sock"))

    def test_ignored_rollout_paths_requires_nonempty_strings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            self.assertEqual(
                sidecar_metadata.ignored_rollout_paths({"ignored_rollout_paths": [str(old_log)]}, sock=root / "fixture.sock"),
                {old_log},
            )
            with self.assertRaisesRegex(ValueError, "invalid ignored_rollout_paths entry"):
                sidecar_metadata.ignored_rollout_paths({"ignored_rollout_paths": [""]}, sock=root / "fixture.sock")

    def test_capabilities_require_protocol_v2_true_flags(self) -> None:
        good = {"control_protocol_version": 2, "control_capabilities": {"sync_send": True, "key_write_errors": True}}
        self.assertTrue(sidecar_metadata.sync_send_supported(good))
        self.assertTrue(sidecar_metadata.key_write_errors_supported(good))
        self.assertFalse(sidecar_metadata.sync_send_supported({"control_protocol_version": 1, "control_capabilities": {"sync_send": True}}))
        self.assertFalse(sidecar_metadata.key_write_errors_supported({"control_protocol_version": 2, "control_capabilities": {"key_write_errors": False}}))

    def test_detaches_current_log_treats_blank_session_id_as_absent(self) -> None:
        current_log = Path("current.jsonl")
        self.assertTrue(sidecar_metadata.detaches_current_log({"session_id": "  ", "log_path": None}, current_log))
        self.assertFalse(sidecar_metadata.detaches_current_log({"session_id": "thread", "log_path": None}, current_log))
        self.assertFalse(sidecar_metadata.detaches_current_log({"session_id": "  ", "log_path": None}, None))

    def test_log_invalid_preserves_diagnostic_prefix(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            sidecar_metadata.log_invalid("discover", Path("fixture.sock"), ValueError("bad"))
        self.assertIn("error: discover: invalid sidecar metadata for fixture.sock: bad", stderr.getvalue())

    def test_required_live_pid_uses_current_process_liveness(self) -> None:
        self.assertEqual(sidecar_metadata.required_live_pid({"broker_pid": os.getpid()}, "broker_pid", sock=Path("fixture.sock")), os.getpid())
        with self.assertRaisesRegex(ValueError, "invalid broker_pid"):
            sidecar_metadata.required_live_pid({"broker_pid": -1}, "broker_pid", sock=Path("fixture.sock"))


if __name__ == "__main__":
    unittest.main()
