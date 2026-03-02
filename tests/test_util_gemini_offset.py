from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.util import read_jsonl_from_offset


def _write_gemini_session(path: Path, messages: list[dict]) -> None:
    payload = {
        "sessionId": "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb",
        "messages": messages,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestUtilGeminiOffset(unittest.TestCase):
    def test_read_jsonl_from_offset_supports_gemini_chat_json(self) -> None:
        with TemporaryDirectory() as td:
            gem_home = Path(td) / ".gemini"
            chats = gem_home / "tmp" / "proj" / "chats"
            chats.mkdir(parents=True, exist_ok=True)
            log = chats / "session-2026-03-02T00-00-abcd1234.json"
            _write_gemini_session(
                log,
                [
                    {"type": "user", "timestamp": "2026-03-02T00:00:00.000Z", "content": [{"text": "hello"}]},
                    {"type": "gemini", "timestamp": "2026-03-02T00:00:01.000Z", "content": "done"},
                ],
            )

            with patch.dict(os.environ, {"GEMINI_HOME": str(gem_home)}, clear=False):
                objs, off = read_jsonl_from_offset(log, 0, max_bytes=1_000_000)
                self.assertEqual(len(objs), 2)
                self.assertEqual(objs[0].get("type"), "user")
                self.assertEqual(objs[1].get("type"), "assistant")
                self.assertGreater(off, 0)

                _write_gemini_session(
                    log,
                    [
                        {"type": "user", "timestamp": "2026-03-02T00:00:00.000Z", "content": [{"text": "hello"}]},
                        {"type": "gemini", "timestamp": "2026-03-02T00:00:01.000Z", "content": "done"},
                        {"type": "user", "timestamp": "2026-03-02T00:00:03.000Z", "content": [{"text": "next"}]},
                    ],
                )
                objs2, off2 = read_jsonl_from_offset(log, off, max_bytes=1_000_000)
                self.assertEqual(len(objs2), 1)
                self.assertEqual(objs2[0].get("type"), "user")
                self.assertGreater(off2, off)

    def test_read_jsonl_from_offset_normalizes_large_physical_offset(self) -> None:
        with TemporaryDirectory() as td:
            gem_home = Path(td) / ".gemini"
            chats = gem_home / "tmp" / "proj" / "chats"
            chats.mkdir(parents=True, exist_ok=True)
            log = chats / "session-2026-03-02T00-00-abcd1234.json"
            _write_gemini_session(
                log,
                [
                    {"type": "user", "timestamp": "2026-03-02T00:00:00.000Z", "content": [{"text": "hello"}]},
                    {"type": "gemini", "timestamp": "2026-03-02T00:00:01.000Z", "content": "done"},
                ],
            )

            with patch.dict(os.environ, {"GEMINI_HOME": str(gem_home)}, clear=False):
                physical_off = int(log.stat().st_size)
                objs, off = read_jsonl_from_offset(log, physical_off, max_bytes=1_000_000)
                self.assertEqual(objs, [])
                self.assertNotEqual(off, physical_off)

                _write_gemini_session(
                    log,
                    [
                        {"type": "user", "timestamp": "2026-03-02T00:00:00.000Z", "content": [{"text": "hello"}]},
                        {"type": "gemini", "timestamp": "2026-03-02T00:00:01.000Z", "content": "done"},
                        {"type": "user", "timestamp": "2026-03-02T00:00:03.000Z", "content": [{"text": "next"}]},
                    ],
                )
                objs2, off2 = read_jsonl_from_offset(log, off, max_bytes=1_000_000)
                self.assertEqual(len(objs2), 1)
                self.assertEqual(objs2[0].get("type"), "user")
                self.assertGreater(off2, off)


if __name__ == "__main__":
    unittest.main()
