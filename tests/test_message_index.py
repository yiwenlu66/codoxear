import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import codoxear.rollout_log as rollout_log

from codoxear.rollout_log import _read_chat_history_page
from codoxear.rollout_log import _read_chat_live_delta
from codoxear.rollout_log import _read_chat_tail_page


def _write_assistant_rows(path: Path, count: int) -> None:
    rows = []
    for i in range(count):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"a{i}"}],
                    "phase": "final_answer",
                },
                "ts": float(i),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _assistant_text(text: str, ts: float) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
            "phase": "final_answer",
        },
        "ts": ts,
    }


def _user_text(text: str, ts: float) -> dict:
    return {"type": "event_msg", "payload": {"type": "user_message", "message": text}, "ts": ts}


class TestMessageIndex(unittest.TestCase):
    def test_tail_and_history_pages_reach_bof_in_order(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 200)

            page1, before1, after1, has_older1 = _read_chat_tail_page(path, limit=80)
            self.assertEqual([ev.get("text") for ev in page1[:2]], ["a120", "a121"])
            self.assertEqual(page1[-1].get("text"), "a199")
            self.assertIsInstance(page1[0].get("_before_byte"), int)
            self.assertEqual(page1[0].get("_before_byte"), before1)
            self.assertTrue(has_older1)
            self.assertGreater(before1, 0)
            self.assertGreater(after1, before1)

            page2, before2, has_older2 = _read_chat_history_page(path, before_byte=before1, limit=80)
            self.assertEqual([ev.get("text") for ev in page2[:2]], ["a40", "a41"])
            self.assertEqual(page2[-1].get("text"), "a119")
            self.assertIsInstance(page2[0].get("_before_byte"), int)
            self.assertEqual(page2[0].get("_before_byte"), before2)
            self.assertLess(page2[-1].get("_before_byte"), before1)
            self.assertTrue(has_older2)
            self.assertGreater(before2, 0)
            self.assertLess(before2, before1)

            page3, before3, has_older3 = _read_chat_history_page(path, before_byte=before2, limit=80)
            self.assertEqual(page3[0].get("text"), "a0")
            self.assertEqual(page3[-1].get("text"), "a39")
            self.assertFalse(has_older3)
            self.assertEqual(before3, 0)

    def test_tail_page_bounds_sparse_recent_history_and_caches_unchanged_identity(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            old = _assistant_text("old", 1.0)
            newest = _assistant_text("new", 2.0)
            filler = {"type": "debug", "payload": "x" * (64 * 1024)}
            with path.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(old) + "\n")
                for _ in range(130):
                    stream.write(json.dumps(filler) + "\n")
                stream.write(json.dumps(newest) + "\n")

            with mock.patch.object(rollout_log, "_read_chat_page_reverse", wraps=rollout_log._read_chat_page_reverse) as reader:
                tail1, before1, after1, has_older1 = rollout_log._read_chat_tail_page(path, limit=80)
                tail2, before2, after2, has_older2 = rollout_log._read_chat_tail_page(path, limit=80)

            self.assertEqual(reader.call_count, 1)
            self.assertEqual([event.get("text") for event in tail1], ["new"])
            self.assertEqual(tail2, tail1)
            self.assertTrue(has_older1)
            self.assertTrue(has_older2)
            self.assertGreater(before1, 0)
            self.assertEqual((before2, after2), (before1, after1))

            history, _next_before, _has_older = _read_chat_history_page(path, before_byte=before1, limit=80)
            self.assertEqual([event.get("text") for event in history], ["old"])

        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            rows = [
                _user_text("first", 1.0),
                _assistant_text("same final text", 2.0),
                _assistant_text("same final text", 2.4),
                _user_text("second", 3.0),
                _assistant_text("same final text", 4.0),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            events, _before, _after, _has_older = _read_chat_tail_page(path, limit=10)

        self.assertEqual([ev.get("role") for ev in events], ["user", "assistant", "user", "assistant"])
        self.assertEqual([ev.get("text") for ev in events], ["first", "same final text", "second", "same final text"])

    def test_live_delta_dedupes_adjacent_assistant_duplicate_texts(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            rows = [
                _user_text("first", 1.0),
                _assistant_text("same final text", 2.0),
                _assistant_text("same final text", 2.4),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            events, _next_after, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)

        self.assertEqual([ev.get("role") for ev in events], ["user", "assistant"])
        self.assertEqual([ev.get("text") for ev in events], ["first", "same final text"])

    def test_stale_live_delta_does_not_affect_history_order(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 200)

            _tail_page, before1, _after1, _has_older1 = _read_chat_tail_page(path, limit=60)
            live_events, next_after, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)
            self.assertEqual(live_events[0].get("text"), "a0")
            self.assertEqual(live_events[-1].get("text"), "a199")
            self.assertIsInstance(live_events[0].get("_before_byte"), int)
            self.assertLess(live_events[0].get("_before_byte"), live_events[-1].get("_before_byte"))
            self.assertGreater(next_after, 0)

            history_page, before2, has_older2 = _read_chat_history_page(path, before_byte=before1, limit=60)
            texts = [ev.get("text") for ev in history_page]
            self.assertEqual(texts, sorted(texts, key=lambda value: int(value[1:])))
            self.assertEqual(texts[0], "a80")
            self.assertEqual(texts[-1], "a139")
            self.assertTrue(has_older2)
            self.assertGreater(before1, before2)


if __name__ == "__main__":
    unittest.main()
