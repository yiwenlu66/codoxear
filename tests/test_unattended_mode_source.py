import unittest

from codoxear.unattended import (
    UNATTENDED_PROMPT_PREFIX,
    clean_unattended_cooldown_minutes,
    clean_unattended_remaining_injections,
    record_unattended_success,
    render_unattended_prompt,
    unattended_prompt_decision,
)


class TestUnattendedModeBehavior(unittest.TestCase):
    def test_prompt_rendering_preserves_constitution_and_user_request(self) -> None:
        rendered = render_unattended_prompt("  inspect the failing test  ", prompt_prefix=UNATTENDED_PROMPT_PREFIX)
        self.assertTrue(rendered.startswith(UNATTENDED_PROMPT_PREFIX.rstrip()))
        self.assertTrue(rendered.endswith("Additional request from user: inspect the failing test\n"))
        self.assertEqual(render_unattended_prompt("", prompt_prefix="prefix\n"), "prefix\n")

    def test_config_validation_rejects_non_integer_and_out_of_range_values(self) -> None:
        self.assertEqual(clean_unattended_cooldown_minutes(None, default_idle_minutes=5), 5)
        self.assertEqual(clean_unattended_remaining_injections(None, default_max_injections=3, allow_zero=True), 3)
        for value in (True, "5", 1.5):
            with self.assertRaisesRegex(ValueError, "unattended cooldown_minutes must be an integer"):
                clean_unattended_cooldown_minutes(value, default_idle_minutes=5)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            clean_unattended_cooldown_minutes(0, default_idle_minutes=5)
        for value in (False, "2", 2.0):
            with self.assertRaisesRegex(ValueError, "unattended remaining_injections must be an integer"):
                clean_unattended_remaining_injections(value, default_max_injections=3, allow_zero=True)
        with self.assertRaisesRegex(ValueError, "at least 0"):
            clean_unattended_remaining_injections(-1, default_max_injections=3, allow_zero=True)

    def test_decision_respects_cooldown_and_disables_exhausted_config(self) -> None:
        enabled = {"enabled": True, "request": "continue", "cooldown_minutes": 2, "remaining_injections": 2}
        decision = unattended_prompt_decision(
            enabled, now_ts=1000, session_last_ts=0, scope_last_ts=0,
            prompt_prefix="prefix", default_idle_minutes=5, default_max_injections=3,
        )
        self.assertEqual(decision.cooldown_seconds, 120)
        self.assertIn("Additional request from user: continue", decision.prompt)
        blocked = unattended_prompt_decision(
            enabled, now_ts=1000, session_last_ts=950, scope_last_ts=0,
            prompt_prefix="prefix", default_idle_minutes=5, default_max_injections=3,
        )
        self.assertEqual(blocked.prompt, "")
        exhausted = unattended_prompt_decision(
            {**enabled, "remaining_injections": 0}, now_ts=1000, session_last_ts=0, scope_last_ts=0,
            prompt_prefix="prefix", default_idle_minutes=5, default_max_injections=3,
        )
        self.assertTrue(exhausted.disabled_exhausted)
        self.assertEqual(exhausted.config, {**enabled, "remaining_injections": 0, "enabled": False})
        update = record_unattended_success({**enabled, "remaining_injections": 1}, default_max_injections=3)
        self.assertEqual((update.remaining_injections, update.enabled), (0, False))


if __name__ == "__main__":
    unittest.main()
