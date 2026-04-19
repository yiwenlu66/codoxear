from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codoxear.page_state_sqlite import DurableSessionRecord
from codoxear.page_state_sqlite import PageStateDB
from codoxear.page_state_sqlite import import_legacy_app_dir_to_db


class TestPageStateSQLite(unittest.TestCase):
    def test_roundtrip_core_page_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = PageStateDB(Path(td) / "state.sqlite")
            db.save_session_ui_state(
                aliases={("pi", "sess-a"): "Alpha"},
                sidebar_meta={
                    ("pi", "sess-a"): {
                        "priority_offset": 0.5,
                        "snooze_until": 123.0,
                        "dependency_session_id": "sess-b",
                        "focused": True,
                    }
                },
                hidden_keys={"thread:pi:sess-a"},
            )
            db.save_files({("pi", "sess-a"): ["a.txt", "b.txt"]})
            db.save_queues({("pi", "sess-a"): ["first", "second"]})
            db.save_recent_cwds({"/tmp/project": 11.0})
            db.save_cwd_groups({"/tmp/project": {"label": "Project", "collapsed": True}})

            aliases, sidebar_meta, hidden_keys = db.load_session_ui_state()
            self.assertEqual(aliases, {("pi", "sess-a"): "Alpha"})
            self.assertEqual(sidebar_meta[("pi", "sess-a")]["priority_offset"], 0.5)
            self.assertEqual(sidebar_meta[("pi", "sess-a")]["snooze_until"], 123.0)
            self.assertEqual(sidebar_meta[("pi", "sess-a")]["dependency_session_id"], "sess-b")
            self.assertTrue(sidebar_meta[("pi", "sess-a")]["focused"])
            self.assertEqual(hidden_keys, {"thread:pi:sess-a"})
            self.assertEqual(db.load_files(), {("pi", "sess-a"): ["a.txt", "b.txt"]})
            self.assertEqual(db.load_queues(), {("pi", "sess-a"): ["first", "second"]})
            self.assertEqual(db.load_recent_cwds(), {"/tmp/project": 11.0})
            self.assertEqual(
                db.load_cwd_groups(),
                {"/tmp/project": {"label": "Project", "collapsed": True}},
            )
            db.close()

    def test_roundtrip_durable_sessions_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = PageStateDB(Path(td) / "state.sqlite")
            db.save_sessions(
                {
                    ("pi", "sess-a"): DurableSessionRecord(
                        backend="pi",
                        session_id="sess-a",
                        cwd="/tmp/project",
                        source_path="/tmp/project/session.jsonl",
                        title="alpha",
                        first_user_message="hello",
                        created_at=10.0,
                        updated_at=12.0,
                        pending_startup=True,
                    )
                }
            )

            rows = db.load_sessions()
            self.assertEqual(rows[("pi", "sess-a")].cwd, "/tmp/project")
            self.assertEqual(rows[("pi", "sess-a")].source_path, "/tmp/project/session.jsonl")
            self.assertEqual(rows[("pi", "sess-a")].title, "alpha")
            self.assertEqual(rows[("pi", "sess-a")].first_user_message, "hello")
            self.assertEqual(rows[("pi", "sess-a")].updated_at, 12.0)
            self.assertTrue(rows[("pi", "sess-a")].pending_startup)
            self.assertEqual(db.known_session_refs(), {("pi", "sess-a")})
            db.close()

    def test_imports_legacy_json_with_runtime_to_durable_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "legacy"
            source.mkdir()
            socks = source / "socks"
            socks.mkdir()
            (socks / "rt-1.json").write_text(
                json.dumps({"agent_backend": "pi", "session_id": "durable-1"}),
                encoding="utf-8",
            )
            (source / "session_aliases.json").write_text(
                json.dumps({"rt-1": "Alpha"}),
                encoding="utf-8",
            )
            (source / "session_sidebar.json").write_text(
                json.dumps({"rt-1": {"priority_offset": 0.25, "snooze_until": 10.0}}),
                encoding="utf-8",
            )
            (source / "session_files.json").write_text(
                json.dumps({"sid:rt-1": ["docs/a.md"]}),
                encoding="utf-8",
            )
            (source / "session_queues.json").write_text(
                json.dumps({"sid:rt-1": ["hello"]}),
                encoding="utf-8",
            )
            (source / "hidden_sessions.json").write_text(
                json.dumps(["thread:pi:durable-1"]),
                encoding="utf-8",
            )
            (source / "recent_cwds.json").write_text(
                json.dumps({"/tmp/project": 99.0}),
                encoding="utf-8",
            )
            (source / "cwd_groups.json").write_text(
                json.dumps({"/tmp/project": {"label": "Proj", "collapsed": False}}),
                encoding="utf-8",
            )
            (source / "voice_settings.json").write_text(
                json.dumps({"enabled": True}),
                encoding="utf-8",
            )
            db_path = Path(td) / "state.sqlite"

            report = import_legacy_app_dir_to_db(source_app_dir=source, db_path=db_path)
            self.assertEqual(report.unmapped_rows, 0)
            self.assertEqual(report.imported_aliases, 1)
            self.assertEqual(report.imported_sidebar_meta, 1)
            self.assertEqual(report.imported_files, 1)
            self.assertEqual(report.imported_queues, 1)

            db = PageStateDB(db_path)
            aliases, sidebar_meta, hidden_keys = db.load_session_ui_state()
            self.assertEqual(aliases, {("pi", "durable-1"): "Alpha"})
            self.assertEqual(sidebar_meta[("pi", "durable-1")]["priority_offset"], 0.25)
            self.assertEqual(sidebar_meta[("pi", "durable-1")]["snooze_until"], 10.0)
            self.assertEqual(hidden_keys, {"thread:pi:durable-1"})
            self.assertEqual(db.load_files(), {("pi", "durable-1"): ["docs/a.md"]})
            self.assertEqual(db.load_queues(), {("pi", "durable-1"): ["hello"]})
            self.assertEqual(db.load_recent_cwds(), {"/tmp/project": 99.0})
            self.assertEqual(db.load_cwd_groups(), {"/tmp/project": {"label": "Proj", "collapsed": False}})
            self.assertEqual(db.load_app_kv("voice_settings"), {"enabled": True})
            db.close()

    def test_records_unmapped_legacy_rows_without_dropping_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "legacy"
            source.mkdir()
            (source / "session_aliases.json").write_text(
                json.dumps({"missing-runtime": "Alpha"}),
                encoding="utf-8",
            )
            db_path = Path(td) / "state.sqlite"

            report = import_legacy_app_dir_to_db(source_app_dir=source, db_path=db_path)
            self.assertEqual(report.unmapped_rows, 1)

            db = PageStateDB(db_path)
            row = db._conn.execute(
                "SELECT source_name, legacy_key FROM legacy_import_unmapped"
            ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["source_name"], "session_aliases.json")
            self.assertEqual(row["legacy_key"], "missing-runtime")
            db.close()


if __name__ == "__main__":
    unittest.main()
