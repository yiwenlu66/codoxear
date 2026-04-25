import unittest
from pathlib import Path


VOICE_PUSH = Path(__file__).resolve().parents[1] / "codoxear" / "voice_push.py"
SERVICE_WORKER = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "service-worker.js"


class TestVoicePushSource(unittest.TestCase):
    def test_summary_prompts_cover_final_and_narration_targets(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("compress assistant final responses", source)
        self.assertIn("Use at most 30 words", source)
        self.assertIn("compress assistant progress narration", source)
        self.assertIn("Use at most 15 words", source)

    def test_keepalive_silence_is_enabled(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("HLS_KEEPALIVE_SECONDS", source)
        self.assertIn("append_silence", source)
        self.assertIn("anullsrc", source)

    def test_voice_pool_includes_full_verified_set(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn('"cedar"', source)
        self.assertIn('"marin"', source)
        self.assertIn('"verse"', source)

    def test_notification_text_is_canonical_backend_field(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn('"notification_text"', source)
        self.assertIn('row.get("notification_text")', source)
        self.assertNotIn('"preview_text": notification_text', source)
        sw_source = SERVICE_WORKER.read_text(encoding="utf-8")
        self.assertIn("payload.notification_text", sw_source)


if __name__ == "__main__":
    unittest.main()
