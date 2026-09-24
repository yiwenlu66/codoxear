import json
import math
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.draft_store import DRAFT_MAX_BYTES
from codoxear.draft_store import DraftStore


class TestDraftStore(unittest.TestCase):
    def test_load_save_round_trip_preserves_text_exactly(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session_drafts.json"
            path.write_text(
                json.dumps(
                    {
                        "s1": {"text": "hello draft", "updated_ts": 12.5},
                        "s2": {"text": "  padded  ", "updated_ts": 1.0},
                        "s3": {"text": "multi\nline ✓ unicode", "updated_ts": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            drafts = DraftStore(path).load()
            DraftStore(path).save(drafts)
            reloaded = DraftStore(path).load()

        self.assertEqual(drafts, reloaded)
        self.assertEqual(drafts["s1"]["text"], "hello draft")
        self.assertEqual(drafts["s2"]["text"], "  padded  ")
        self.assertEqual(drafts["s3"]["text"], "multi\nline ✓ unicode")
        self.assertEqual(drafts["s1"]["updated_ts"], 12.5)

    def test_load_missing_file_returns_empty(self) -> None:
        with TemporaryDirectory() as td:
            self.assertEqual(DraftStore(Path(td) / "none.json").load(), {})

    def test_load_rejects_non_object_root(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session_drafts.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                DraftStore(path).load()

    def test_load_drops_malformed_entries_and_keeps_tombstones(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session_drafts.json"
            path.write_text(
                json.dumps(
                    {
                        "ok": {"text": "keep", "updated_ts": 5.0},
                        "non-dict": "plain string",
                        "list": [{"text": "a", "updated_ts": 1.0}],
                        "bad-text": {"text": 42, "updated_ts": 5.0},
                        "missing-text": {"updated_ts": 5.0},
                        "empty-text": {"text": "", "updated_ts": 5.0},
                        "ws-text": {"text": "   \n\t ", "updated_ts": 5.0},
                        "": {"text": "empty sid", "updated_ts": 5.0},
                    }
                ),
                encoding="utf-8",
            )
            drafts = DraftStore(path).load()

        # Empty/whitespace-only text entries are tombstones — first-class
        # deletion events that must survive load so they propagate through
        # the last-writer-wins channel. Whitespace-only normalizes to "".
        self.assertEqual(set(drafts), {"ok", "empty-text", "ws-text"})
        self.assertEqual(drafts["empty-text"], {"text": "", "updated_ts": 5.0})
        self.assertEqual(drafts["ws-text"], {"text": "", "updated_ts": 5.0})

    def test_load_coerces_invalid_timestamps_to_finite_now(self) -> None:
        before = time.time()
        with TemporaryDirectory() as td:
            path = Path(td) / "session_drafts.json"
            path.write_text(
                json.dumps(
                    {
                        "str-ts": {"text": "a", "updated_ts": "not-a-number"},
                        "none-ts": {"text": "b", "updated_ts": None},
                        "nan-ts": {"text": "c", "updated_ts": float("nan")},
                        "inf-ts": {"text": "d", "updated_ts": float("inf")},
                        "neg-ts": {"text": "e", "updated_ts": -3.0},
                        "bool-ts": {"text": "f", "updated_ts": True},
                        "num-str-ts": {"text": "g", "updated_ts": "7.5"},
                    }
                ),
                encoding="utf-8",
            )
            drafts = DraftStore(path).load()
        after = time.time()

        self.assertEqual(
            set(drafts),
            {"str-ts", "none-ts", "nan-ts", "inf-ts", "neg-ts", "bool-ts", "num-str-ts"},
        )
        self.assertEqual(drafts["num-str-ts"]["updated_ts"], 7.5)
        for sid in ("str-ts", "none-ts", "nan-ts", "inf-ts", "neg-ts", "bool-ts"):
            ts = drafts[sid]["updated_ts"]
            self.assertTrue(math.isfinite(ts), f"{sid}: {ts}")
            self.assertTrue(before <= ts <= after, f"{sid}: {ts}")

    def test_set_writes_timestamped_entry_and_tombstones_on_blank(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))
        drafts: dict[str, dict[str, object]] = {}

        ts = store.set(drafts, "s1", "draft text", now_ts=99.5)
        self.assertEqual(ts, 99.5)
        self.assertEqual(drafts["s1"], {"text": "draft text", "updated_ts": 99.5})

        # Blank clears are timestamped deletion events, not entry removal:
        # the tombstone must outrank every earlier edit timestamp.
        for blank in ("", "   ", "\n\t "):
            self.assertEqual(store.set(drafts, "s1", blank, now_ts=100.0), 100.0)
            self.assertEqual(drafts["s1"], {"text": "", "updated_ts": 100.0})

        self.assertEqual(store.set(drafts, "s1", "again", now_ts=101.0), 101.0)
        self.assertEqual(drafts["s1"], {"text": "again", "updated_ts": 101.0})

    def test_tombstone_method_writes_timestamped_empty_entry(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))
        drafts: dict[str, dict[str, object]] = {"s1": {"text": "old", "updated_ts": 1.0}}

        ts = store.tombstone(drafts, "s1", now_ts=77.25)

        self.assertEqual(ts, 77.25)
        self.assertEqual(drafts["s1"], {"text": "", "updated_ts": 77.25})

    def test_set_blank_wins_over_byte_cap(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))
        drafts: dict[str, dict[str, object]] = {"s1": {"text": "keep", "updated_ts": 1.0}}

        # Clearing a draft can never exceed the cap: the blank check runs
        # before the byte-cap check and records a tombstone.
        self.assertEqual(store.set(drafts, "s1", " " * (DRAFT_MAX_BYTES + 1), now_ts=2.0), 2.0)
        self.assertEqual(drafts["s1"], {"text": "", "updated_ts": 2.0})

    def test_set_enforces_byte_cap(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))
        drafts: dict[str, dict[str, object]] = {}

        self.assertEqual(store.set(drafts, "s1", "a" * DRAFT_MAX_BYTES, now_ts=1.0), 1.0)
        self.assertEqual(store.set(drafts, "s2", "é" * (DRAFT_MAX_BYTES // 2), now_ts=1.0), 1.0)

        with self.assertRaisesRegex(ValueError, "byte limit"):
            store.set(drafts, "s3", "a" * (DRAFT_MAX_BYTES + 1), now_ts=1.0)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            # Two UTF-8 bytes per character: 131073 chars exceed the cap.
            store.set(drafts, "s4", "é" * (DRAFT_MAX_BYTES // 2 + 1), now_ts=1.0)

        self.assertEqual(set(drafts), {"s1", "s2"})

    def test_set_rejects_non_utf8_encodable_text(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))
        drafts: dict[str, dict[str, object]] = {}

        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            store.set(drafts, "s1", "lone \ud800 surrogate", now_ts=1.0)
        self.assertNotIn("s1", drafts)

    def test_public_entry_projection(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))

        self.assertEqual(store.public_entry(None), {"text": "", "updated_ts": 0.0})
        self.assertEqual(store.public_entry("junk"), {"text": "", "updated_ts": 0.0})
        self.assertEqual(store.public_entry({"text": "hi", "updated_ts": 3.5}), {"text": "hi", "updated_ts": 3.5})
        self.assertEqual(store.public_entry({"text": "hi", "updated_ts": True}), {"text": "hi", "updated_ts": 0.0})
        self.assertEqual(store.public_entry({"text": 5, "updated_ts": 3.5}), {"text": "", "updated_ts": 3.5})
        self.assertEqual(store.public_entry({"text": "", "updated_ts": 3.5}), {"text": "", "updated_ts": 3.5})
        self.assertEqual(store.public_entry({"text": "hi", "updated_ts": float("nan")}), {"text": "hi", "updated_ts": 0.0})
        self.assertEqual(store.public_entry({"text": "hi", "updated_ts": float("inf")}), {"text": "hi", "updated_ts": 0.0})
        self.assertEqual(store.public_entry({"text": "hi", "updated_ts": -1.0}), {"text": "hi", "updated_ts": 0.0})
        self.assertEqual(store.public_entry({"text": "hi"}), {"text": "hi", "updated_ts": 0.0})

    def test_updated_ts_helper(self) -> None:
        store = DraftStore(Path("/tmp/unused.json"))
        drafts = {
            "s1": {"text": "x", "updated_ts": 4.5},
            "tomb": {"text": "", "updated_ts": 8.25},
        }

        self.assertEqual(store.updated_ts(drafts, "s1"), 4.5)
        self.assertEqual(store.updated_ts(drafts, "tomb"), 8.25)
        self.assertEqual(store.updated_ts(drafts, "missing"), 0.0)
        self.assertEqual(store.updated_ts(None, "s1"), 0.0)
        self.assertEqual(store.updated_ts({"s1": "junk"}, "s1"), 0.0)

    def test_save_skips_unpersistable_entries_and_normalizes_tombstones(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session_drafts.json"
            DraftStore(path).save(
                {
                    "s1": {"text": "keep", "updated_ts": 1.0},
                    "s2": {"text": "   ", "updated_ts": 2.0},
                    "s3": "junk",
                    "": {"text": "empty sid", "updated_ts": 3.0},
                }
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved, {"s1": {"text": "keep", "updated_ts": 1.0}, "s2": {"text": "", "updated_ts": 2.0}})

    def test_tombstone_persists_across_reload(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session_drafts.json"
            store = DraftStore(path)
            drafts: dict[str, dict[str, object]] = {}
            store.set(drafts, "s1", "sent from another client", now_ts=10.0)
            clear_ts = store.set(drafts, "s1", "", now_ts=11.0)
            store.save(drafts)

            reloaded = DraftStore(path).load()

        self.assertEqual(clear_ts, 11.0)
        # Pruning empty-text entries on load would lose delete propagation.
        self.assertEqual(reloaded, {"s1": {"text": "", "updated_ts": 11.0}})

    def test_none_path_disables_io(self) -> None:
        store = DraftStore(None)
        self.assertEqual(store.load(), {})
        store.save({"s1": {"text": "x", "updated_ts": 1.0}})


if __name__ == "__main__":
    unittest.main()
