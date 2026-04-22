import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


class TestMessageIndex(unittest.TestCase):
    def test_tail_and_history_pages_reach_bof_in_order(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 200)

            page1, before1, after1, has_older1 = _read_chat_tail_page(path, limit=80)
            self.assertEqual([ev.get("text") for ev in page1[:2]], ["a120", "a121"])
            self.assertEqual(page1[-1].get("text"), "a199")
            self.assertTrue(has_older1)
            self.assertGreater(before1, 0)
            self.assertGreater(after1, before1)

            page2, before2, has_older2 = _read_chat_history_page(path, before_byte=before1, limit=80)
            self.assertEqual([ev.get("text") for ev in page2[:2]], ["a40", "a41"])
            self.assertEqual(page2[-1].get("text"), "a119")
            self.assertTrue(has_older2)
            self.assertGreater(before2, 0)
            self.assertLess(before2, before1)

            page3, before3, has_older3 = _read_chat_history_page(path, before_byte=before2, limit=80)
            self.assertEqual(page3[0].get("text"), "a0")
            self.assertEqual(page3[-1].get("text"), "a39")
            self.assertFalse(has_older3)
            self.assertEqual(before3, 0)

    def test_stale_live_delta_does_not_affect_history_order(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 200)

            _tail_page, before1, _after1, _has_older1 = _read_chat_tail_page(path, limit=60)
            live_events, next_after, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)
            self.assertEqual(live_events[0].get("text"), "a0")
            self.assertEqual(live_events[-1].get("text"), "a199")
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
