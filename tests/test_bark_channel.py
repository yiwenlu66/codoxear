"""BarkChannel out-of-band push tests."""

import json
import unittest
from unittest.mock import MagicMock, patch

from codoxear.voice_push import BarkChannel, _clean_voice_settings


class BarkChannelTests(unittest.TestCase):
    def test_disabled_when_token_empty(self):
        ch = BarkChannel(endpoint="https://api.day.app", token="")
        self.assertFalse(ch.is_enabled())
        # send() must short-circuit and not perform an HTTP call.
        with patch("codoxear.voice_push.urllib.request.urlopen") as mock_open:
            res = ch.send(
                session_id="s1",
                session_display_name="Sess",
                message_id="m1",
                notification_text="hi",
                timestamp=None,
            )
        self.assertFalse(mock_open.called)
        self.assertEqual(res["status"], "skipped")

    def test_send_posts_correct_body(self):
        ch = BarkChannel(endpoint="https://api.day.app", token="abc123", base_url="https://example/codoxear")

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            captured["data"] = req.data
            captured["method"] = req.get_method()
            resp = MagicMock()
            resp.read = lambda: b'{"code":200,"message":"success"}'
            resp.__enter__ = lambda self_: self_
            resp.__exit__ = lambda *a: False
            resp.status = 200
            return resp

        with patch("codoxear.voice_push.urllib.request.urlopen", side_effect=fake_urlopen):
            res = ch.send(
                session_id="broker-1",
                session_display_name="My Session",
                message_id="m1",
                notification_text="agent finished",
                timestamp=1000.0,
            )

        self.assertEqual(res["status"], "sent")
        self.assertEqual(captured["url"], "https://api.day.app/abc123")
        self.assertEqual(captured["method"], "POST")
        body = json.loads(captured["data"].decode("utf-8"))
        self.assertEqual(body["title"], "My Session")
        self.assertEqual(body["body"], "agent finished")
        self.assertEqual(body["group"], "codoxear")
        self.assertEqual(body["url"], "https://example/codoxear/#session=broker-1")

    def test_send_marks_error_on_exception(self):
        ch = BarkChannel(endpoint="https://api.day.app", token="abc")
        with patch("codoxear.voice_push.urllib.request.urlopen", side_effect=RuntimeError("network down")):
            res = ch.send(
                session_id="s1",
                session_display_name="Sess",
                message_id="m1",
                notification_text="hi",
                timestamp=None,
            )
        self.assertEqual(res["status"], "error")
        self.assertIn("network down", res["error"])


class VoiceSettingsBarkFieldsTests(unittest.TestCase):
    def test_settings_round_trip(self):
        cleaned = _clean_voice_settings({"bark_enabled": True, "bark_token": "tok"})
        self.assertTrue(cleaned["bark_enabled"])
        self.assertEqual(cleaned["bark_endpoint"], "https://api.day.app")
        self.assertEqual(cleaned["bark_token"], "tok")

    def test_settings_default_when_missing(self):
        cleaned = _clean_voice_settings({})
        self.assertFalse(cleaned["bark_enabled"])
        self.assertEqual(cleaned["bark_endpoint"], "https://api.day.app")
        self.assertEqual(cleaned["bark_token"], "")

    def test_settings_rejects_invalid_endpoint(self):
        cleaned = _clean_voice_settings({"bark_endpoint": "ftp://nope"})
        self.assertEqual(cleaned["bark_endpoint"], "https://api.day.app")

    def test_settings_base_url_round_trip(self):
        cleaned = _clean_voice_settings({"bark_base_url": "http://192.168.1.10:13780/"})
        self.assertEqual(cleaned["bark_base_url"], "http://192.168.1.10:13780")

    def test_settings_base_url_rejects_non_http(self):
        cleaned = _clean_voice_settings({"bark_base_url": "ftp://nope"})
        self.assertEqual(cleaned["bark_base_url"], "")

    def test_settings_base_url_default_empty(self):
        cleaned = _clean_voice_settings({})
        self.assertEqual(cleaned["bark_base_url"], "")


class BarkCoordinatorWiringTests(unittest.TestCase):
    def test_coordinator_passes_base_url_into_bark_channel(self):
        import threading
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from codoxear.voice_push import VoicePushCoordinator

        with TemporaryDirectory() as td:
            stop_event = threading.Event()
            stop_event.set()
            coord = VoicePushCoordinator(
                app_dir=Path(td),
                stop_event=stop_event,
                settings_path=Path(td) / "voice_settings.json",
                subscriptions_path=Path(td) / "push_subscriptions.json",
                delivery_ledger_path=Path(td) / "voice_delivery_ledger.json",
                vapid_private_key_path=Path(td) / "vapid.pem",
            )
            coord.set_settings(
                {
                    "bark_enabled": True,
                    "bark_token": "tok",
                    "bark_base_url": "http://192.168.1.10:13780",
                }
            )
            bark = [c for c in coord._channels if getattr(c, "kind", "") == "bark"]
            self.assertEqual(len(bark), 1)
            self.assertEqual(bark[0]._base_url, "http://192.168.1.10:13780")  # type: ignore[attr-defined]

    def test_disabled_bark_not_in_channels(self):
        import threading
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from codoxear.voice_push import VoicePushCoordinator

        with TemporaryDirectory() as td:
            stop_event = threading.Event()
            stop_event.set()
            coord = VoicePushCoordinator(
                app_dir=Path(td),
                stop_event=stop_event,
                settings_path=Path(td) / "voice_settings.json",
                subscriptions_path=Path(td) / "push_subscriptions.json",
                delivery_ledger_path=Path(td) / "voice_delivery_ledger.json",
                vapid_private_key_path=Path(td) / "vapid.pem",
            )
            coord.set_settings({"bark_enabled": False, "bark_token": "tok"})
            kinds = [getattr(c, "kind", "") for c in coord._channels]
            self.assertNotIn("bark", kinds)
            self.assertIn("webpush", kinds)


if __name__ == "__main__":
    unittest.main()
