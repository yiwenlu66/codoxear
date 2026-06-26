import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from codoxear.unattended import UnattendedStore
from codoxear.unattended import disable_unattended_if_exhausted
from codoxear.unattended import clean_unattended_cooldown_minutes
from codoxear.unattended import clean_unattended_remaining_injections
from codoxear.unattended import record_unattended_success
from codoxear.unattended import render_unattended_prompt
from codoxear.unattended import unattended_cooldown_blocked
from codoxear.unattended import unattended_prompt_decision
from codoxear.unattended import unattended_scope_key
from codoxear.unattended import unattended_tail_allows_injection


class TestUnattendedStore(unittest.TestCase):
    def test_load_cleans_entries_and_preserves_file_format(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "unattended.json"
            path.write_text(
                json.dumps(
                    {
                        "sid-a": {"enabled": True, "request": "go", "cooldown_minutes": 2, "remaining_injections": 0},
                        "sid-b": {"enabled": False},
                        "sid-c": "bad",
                        "": {"enabled": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = UnattendedStore(path=path, default_idle_minutes=5, default_max_injections=10)

            self.assertEqual(
                store.load(),
                {
                    "sid-a": {"enabled": True, "request": "go", "cooldown_minutes": 2, "remaining_injections": 0},
                    "sid-b": {"enabled": False, "request": "", "cooldown_minutes": 5, "remaining_injections": 10},
                },
            )

    def test_save_writes_sorted_pretty_json(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "unattended.json"
            store = UnattendedStore(path=path, default_idle_minutes=5, default_max_injections=10)
            store.save({"sid-b": {"enabled": False}, "sid-a": {"enabled": True}})
            text = path.read_text(encoding="utf-8")

        self.assertTrue(text.endswith("\n"))
        self.assertLess(text.index('"sid-a"'), text.index('"sid-b"'))

    def test_legacy_text_field_fails_loudly(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "unattended.json"
            path.write_text('{"sid":{"text":"old"}}\n', encoding="utf-8")
            store = UnattendedStore(path=path, default_idle_minutes=5, default_max_injections=10)
            with self.assertRaisesRegex(ValueError, "use 'request', not 'text'"):
                store.load()

    def test_unattended_policy_helpers_cover_sweep_decisions(self) -> None:
        self.assertEqual(unattended_scope_key(thread_id="t1", log_path=Path("/tmp/log.jsonl")), "thread:t1")
        self.assertEqual(unattended_scope_key(thread_id="", log_path=Path("/tmp/log.jsonl")), "log:/tmp/log.jsonl")
        self.assertTrue(unattended_cooldown_blocked(now_ts=100.0, cooldown_seconds=60.0, session_last_ts=50.0, scope_last_ts=0.0))
        self.assertTrue(unattended_cooldown_blocked(now_ts=100.0, cooldown_seconds=60.0, session_last_ts=0.0, scope_last_ts=50.0))
        self.assertFalse(unattended_cooldown_blocked(now_ts=100.0, cooldown_seconds=60.0, session_last_ts=10.0, scope_last_ts=0.0))
        self.assertTrue(unattended_tail_allows_injection(("assistant", 10.0), now_ts=100.0, cooldown_seconds=60.0))
        self.assertFalse(unattended_tail_allows_injection(("user", 10.0), now_ts=100.0, cooldown_seconds=60.0))
        self.assertFalse(unattended_tail_allows_injection(("assistant", 90.0), now_ts=100.0, cooldown_seconds=60.0))

        disabled, changed = disable_unattended_if_exhausted(
            {"enabled": True, "remaining_injections": 0},
            default_max_injections=10,
        )
        self.assertTrue(changed)
        self.assertEqual(disabled["enabled"], False)
        self.assertEqual(disabled["remaining_injections"], 0)

        decision = unattended_prompt_decision(
            {"enabled": True, "request": "next", "cooldown_minutes": 1, "remaining_injections": 2},
            now_ts=100.0,
            session_last_ts=0.0,
            scope_last_ts=0.0,
            prompt_prefix="Base",
            default_idle_minutes=5,
            default_max_injections=10,
        )
        self.assertEqual(decision.prompt, "Base\n\n---\n\nAdditional request from user: next\n")
        self.assertEqual(decision.cooldown_seconds, 60.0)
        self.assertFalse(decision.disabled_exhausted)

        update = record_unattended_success(
            {"enabled": True, "remaining_injections": 1},
            default_max_injections=10,
        )
        self.assertEqual(update.remaining_injections, 0)
        self.assertFalse(update.enabled)
        self.assertFalse(update.config["enabled"])

    def test_cleaners_and_prompt_match_server_contract(self) -> None:
        self.assertEqual(clean_unattended_cooldown_minutes(None, default_idle_minutes=5), 5)
        self.assertEqual(clean_unattended_remaining_injections(None, default_max_injections=10, allow_zero=True), 10)
        with self.assertRaisesRegex(ValueError, "cooldown_minutes must be an integer"):
            clean_unattended_cooldown_minutes(True, default_idle_minutes=5)
        self.assertEqual(render_unattended_prompt("next", prompt_prefix="Base\n"), "Base\n\n---\n\nAdditional request from user: next\n")


if __name__ == "__main__":
    unittest.main()
