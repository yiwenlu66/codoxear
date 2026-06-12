import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.queue_store import QueueStore


class TestQueueStore(unittest.TestCase):
    def test_load_migrates_legacy_strings_and_duplicate_ids(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "queues.json"
            path.write_text(json.dumps({"s1": ["one", {"id": "dup", "text": "two"}, {"id": "dup", "text": "three"}]}), encoding="utf-8")
            queues = QueueStore(path).load()

        self.assertEqual([item["text"] for item in queues["s1"]], ["one", "two", "three"])
        ids = [item["id"] for item in queues["s1"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sending_item_cannot_be_mutated_and_success_removes_same_id_only(self) -> None:
        store = QueueStore(Path("/tmp/unused.json"))
        queues = {"s1": [{"id": "a", "text": "dup", "created_ts": 1}, {"id": "b", "text": "dup", "created_ts": 2}]}

        with self.assertRaisesRegex(ValueError, "item is already sending"):
            store.update(queues, "s1", "a", "edit", sending_item_id="a")
        with self.assertRaisesRegex(ValueError, "item is already sending"):
            store.delete(queues, "s1", "a", sending_item_id="a")
        with self.assertRaisesRegex(ValueError, "item is already sending"):
            store.move(queues, "s1", "a", 1, sending_item_id="a")

        store.pop_sent(queues, "s1", "a")
        self.assertEqual([item["id"] for item in queues["s1"]], ["b"])
        self.assertEqual([item["text"] for item in queues["s1"]], ["dup"])

    def test_commit_unknown_state_survives_load_list_and_save(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "queues.json"
            path.write_text(json.dumps({"s1": [{"id": "a", "text": "maybe sent", "created_ts": 1, "commit_unknown": True, "commit_unknown_ts": 2}]}), encoding="utf-8")
            store = QueueStore(path)
            queues = store.load()
            listed = store.list_items(queues, "s1")
            store.save(queues)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(queues["s1"][0]["commit_unknown"])
        self.assertEqual(queues["s1"][0]["commit_unknown_ts"], 2.0)
        self.assertTrue(listed[0]["commit_unknown"])
        self.assertFalse(store.list_items(queues, "s1", sending_item_id="a")[0]["commit_unknown"])
        self.assertTrue(saved["s1"][0]["commit_unknown"])
        self.assertEqual(saved["s1"][0]["commit_unknown_ts"], 2.0)

    def test_commit_unknown_item_blocks_reordering_past_it(self) -> None:
        store = QueueStore(Path("/tmp/unused.json"))
        queues = {
            "s1": [
                {"id": "a", "text": "maybe", "created_ts": 1, "commit_unknown": True},
                {"id": "b", "text": "later", "created_ts": 2},
            ]
        }

        with self.assertRaisesRegex(ValueError, "commit-unknown item blocks reordering"):
            store.move(queues, "s1", "b", 0)
        with self.assertRaisesRegex(ValueError, "commit status is unknown"):
            store.move(queues, "s1", "a", 1)
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            store.delete(queues, "s1", "a")
        with self.assertRaisesRegex(ValueError, "commit status is unknown"):
            store.update(queues, "s1", "a", "changed")
        self.assertEqual([item["id"] for item in queues["s1"]], ["a", "b"])

        self.assertEqual(store.delete(queues, "s1", "a", allow_commit_unknown=True), 1)
        self.assertEqual([item["id"] for item in queues["s1"]], ["b"])

    def test_drop_missing_sessions_and_save_omit_empty_queues(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "queues.json"
            store = QueueStore(path)
            queues = {"live": [{"id": "a", "text": "one", "created_ts": 1}], "dead": [{"id": "b", "text": "two", "created_ts": 2}], "empty": []}

            self.assertTrue(store.drop_missing_sessions(queues, {"live", "empty"}))
            store.save(queues)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(saved), {"live"})


if __name__ == "__main__":
    unittest.main()
