import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from unittest.mock import patch
from codoxear.rollout_log import _extract_delivery_messages
from codoxear.voice_push import AnnouncementTask
from codoxear.voice_push import GeneratedAnnouncement
from codoxear.voice_push import MergedHLSStream
from codoxear.voice_push import _default_vapid_subject
from codoxear.voice_push import VoicePushCoordinator


class _FakeClient:
    def __init__(self) -> None:
        self.summary_calls = []
        self.speech_calls = []

    def summarize(self, **kwargs):
        self.summary_calls.append(kwargs)
        if kwargs.get("target_words") == 15:
            return "short narration summary"
        return "short final summary"

    def synthesize(self, **kwargs):
        self.speech_calls.append(kwargs)
        return b"fake-audio"


class _FakeHLS:
    def __init__(self) -> None:
        self.append_calls = []
        self.last_error = ""
        self.reset_calls = 0

    def append_audio(self, **kwargs):
        self.append_calls.append(kwargs)
        return 1.0

    def append_silence(self, **kwargs):
        return True

    def set_last_error(self, message: str) -> None:
        self.last_error = message

    def reset(self) -> None:
        self.reset_calls += 1

    def snapshot(self):
        return {"segment_count": len(self.append_calls), "last_error": self.last_error, "media_sequence": 1}


class TestDeliveryExtraction(unittest.TestCase):
    def test_dedupes_event_msg_and_response_item_with_same_text(self) -> None:
        rows = [
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "Working on it"}, "ts": 1.25},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Working on it"}],
                },
                "ts": 1.25,
            },
        ]
        messages = _extract_delivery_messages(rows)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_class, "narration")
        self.assertEqual(messages[0].text, "Working on it")

    def test_marks_final_response_from_phase_and_end_turn(self) -> None:
        rows = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "done"}],
                },
                "ts": 2.0,
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "end_turn": True,
                    "content": [{"type": "output_text", "text": "done again"}],
                },
                "ts": 3.0,
            },
        ]
        messages = _extract_delivery_messages(rows)
        self.assertEqual([item.message_class for item in messages], ["final_response", "final_response"])

    def test_dedupes_adjacent_assistant_messages_with_same_text_but_different_timestamps(self) -> None:
        rows = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "same final text"}],
                },
                "ts": 2.0,
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "same final text"}],
                },
                "ts": 2.2,
            },
        ]
        messages = _extract_delivery_messages(rows)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_class, "final_response")
        self.assertEqual(messages[0].text, "same final text")

    def test_dedupes_final_response_when_response_item_only_adds_memory_citation(self) -> None:
        rows = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "phase": "final_answer",
                    "message": "done",
                },
                "ts": 2.0,
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "done\n\n"
                                "<oai-mem-citation>\n"
                                "<citation_entries>\n"
                                "MEMORY.md:1-2|note=[demo]\n"
                                "</citation_entries>\n"
                                "<rollout_ids>\n"
                                "019d1f15-1cc0-7181-853a-b7a0ceadf3c1\n"
                                "</rollout_ids>\n"
                                "</oai-mem-citation>\n"
                            ),
                        }
                    ],
                },
                "ts": 2.001,
            },
        ]
        messages = _extract_delivery_messages(rows)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_class, "final_response")
        self.assertEqual(messages[0].text, "done")


class TestVoicePushCoordinator(unittest.TestCase):
    def test_final_response_summarizes_and_enqueues_audio(self) -> None:
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
            fake_client = _FakeClient()
            fake_hls = _FakeHLS()
            coord._client = fake_client  # type: ignore[assignment]
            coord._hls = fake_hls  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": False,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            coord.observe_messages(
                session_id="sid-1",
                session_display_name="Repo",
                messages=_extract_delivery_messages(
                    [
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "phase": "final_answer",
                                "content": [{"type": "output_text", "text": "Longer final answer body"}],
                            },
                            "ts": 4.0,
                        }
                    ]
                ),
            )
            with coord._lock:
                task = coord._queue[0]
            coord._process_task(task)
            with coord._lock:
                prepared = coord._prepared
            self.assertIsNotNone(prepared)
            assert prepared is not None
            coord._append_prepared(prepared)
            self.assertEqual(len(fake_client.summary_calls), 1)
            self.assertEqual(len(fake_client.speech_calls), 1)
            self.assertEqual(len(fake_hls.append_calls), 1)
            self.assertEqual(fake_client.summary_calls[0]["target_words"], 30)
            spoken = fake_client.speech_calls[0]["text"]
            self.assertEqual(spoken, "Turn summary from Repo. short final summary")
            ledger = coord._delivery_ledger[task.source_message_ids[0]]
            self.assertEqual(ledger["notification_text"], "short final summary")
            self.assertEqual(ledger["summary_status"], "sent")
            self.assertEqual(ledger["narrated_status"], "sent")
            self.assertEqual(ledger["push_status"], "skipped")

    def test_narration_is_summarized_on_dequeue_before_tts(self) -> None:
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
            fake_client = _FakeClient()
            fake_hls = _FakeHLS()
            coord._client = fake_client  # type: ignore[assignment]
            coord._hls = fake_hls  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": False,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            message = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Long and verbose narration body"}],
                        },
                        "ts": 4.0,
                    }
                ]
            )[0]
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[message])
            with coord._lock:
                task = coord._queue[0]
            self.assertEqual(task.summary_word_target, 15)
            self.assertEqual(task.spoken_text, "")
            coord._process_task(task)
            with coord._lock:
                prepared = coord._prepared
            self.assertIsNotNone(prepared)
            assert prepared is not None
            coord._append_prepared(prepared)
            self.assertEqual(len(fake_client.summary_calls), 1)
            self.assertEqual(fake_client.summary_calls[0]["target_words"], 15)
            self.assertEqual(fake_client.summary_calls[0]["source_label"], "Narration updates")
            self.assertEqual(len(fake_client.speech_calls), 1)
            self.assertEqual(fake_client.speech_calls[0]["text"], "From Repo. short narration summary")
            row = coord._delivery_ledger[message.message_id]
            self.assertEqual(row["summary_status"], "sent")
            self.assertEqual(row["summary_text"], "short narration summary")
            self.assertEqual(row["narrated_status"], "sent")

    def test_notification_text_falls_back_to_raw_when_no_api_key(self) -> None:
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
                    "tts_enabled_for_narration": False,
                    "tts_enabled_for_final_response": False,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "",
                }
            )
            coord.observe_messages(
                session_id="sid-1",
                session_display_name="Repo",
                messages=_extract_delivery_messages(
                    [
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "phase": "final_answer",
                                "content": [{"type": "output_text", "text": "Line one.\n\nLine two."}],
                            },
                            "ts": 4.0,
                        }
                    ]
                ),
            )
            row = next(iter(coord._delivery_ledger.values()))
            self.assertEqual(row["summary_status"], "skipped")
            self.assertEqual(row["notification_text"], "Line one. Line two.")

    def test_final_response_with_memory_citation_is_summarized_once(self) -> None:
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
            fake_client = _FakeClient()
            coord._client = fake_client  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": False,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            coord.observe_messages(
                session_id="sid-1",
                session_display_name="Repo",
                messages=_extract_delivery_messages(
                    [
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "agent_message",
                                "phase": "final_answer",
                                "message": "Longer final answer body",
                            },
                            "ts": 4.0,
                        },
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "phase": "final_answer",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            "Longer final answer body\n\n"
                                            "<oai-mem-citation>\n"
                                            "<citation_entries>\n"
                                            "MEMORY.md:1-2|note=[demo]\n"
                                            "</citation_entries>\n"
                                            "<rollout_ids>\n"
                                            "019d1f15-1cc0-7181-853a-b7a0ceadf3c1\n"
                                            "</rollout_ids>\n"
                                            "</oai-mem-citation>\n"
                                        ),
                                    }
                                ],
                            },
                            "ts": 4.001,
                        },
                    ]
                ),
            )
            self.assertEqual(len(fake_client.summary_calls), 1)
            with coord._lock:
                self.assertEqual(len(coord._delivery_ledger), 1)
                self.assertEqual(len(coord._queue), 1)

    def test_subscription_upsert_and_toggle_round_trip(self) -> None:
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
            payload = {
                "endpoint": "https://push.example.test/device/1",
                "keys": {"p256dh": "abc", "auth": "def"},
            }
            snapshot = coord.upsert_subscription(
                subscription=payload,
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X)",
                device_label="phone",
                device_class="mobile",
            )
            self.assertEqual(len(snapshot["subscriptions"]), 1)
            self.assertTrue(snapshot["subscriptions"][0]["notifications_enabled"])
            self.assertEqual(snapshot["subscriptions"][0]["device_class"], "mobile")
            endpoint = snapshot["subscriptions"][0]["endpoint"]
            snapshot = coord.toggle_subscription(endpoint=endpoint, enabled=False)
            self.assertFalse(snapshot["subscriptions"][0]["notifications_enabled"])
            snapshot = coord.upsert_subscription(
                subscription=payload,
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X)",
                device_label="phone",
                device_class="mobile",
            )
            self.assertTrue(snapshot["subscriptions"][0]["notifications_enabled"])

    def test_narration_is_dropped_immediately_without_listener(self) -> None:
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
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": False,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            message = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Long and verbose narration body"}],
                        },
                        "ts": 4.0,
                    }
                ]
            )[0]
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[message])
            with coord._lock:
                self.assertEqual(coord._queue, [])
            row = coord._delivery_ledger[message.message_id]
            self.assertEqual(row["summary_status"], "skipped")
            self.assertEqual(row["narrated_status"], "skipped")
            self.assertEqual(row["last_error"], "no active listener")

    def test_final_response_without_listener_skips_voice_but_keeps_summary(self) -> None:
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
            fake_client = _FakeClient()
            coord._client = fake_client  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": False,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            message = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": "Longer final answer body"}],
                        },
                        "ts": 4.0,
                    }
                ]
            )[0]
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[message])
            with coord._lock:
                self.assertEqual(coord._queue, [])
            row = coord._delivery_ledger[message.message_id]
            self.assertEqual(row["summary_status"], "sent")
            self.assertEqual(row["notification_text"], "short final summary")
            self.assertEqual(row["narrated_status"], "skipped")
            self.assertEqual(row["push_status"], "skipped")
            self.assertEqual(row["last_error"], "no active listener")

    def test_voice_mapping_is_stable_for_session_id(self) -> None:
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
            self.assertEqual(coord._voice_for_session("sid-1", "alpha"), coord._voice_for_session("sid-1", "beta"))
            self.assertNotEqual(coord._voice_for_session("sid-1", "alpha"), "")

    def test_voice_pool_contains_all_verified_audio_speech_voices(self) -> None:
        from codoxear.voice_push import DEFAULT_VOICES

        self.assertEqual(
            DEFAULT_VOICES,
            ("alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse"),
        )

    def test_default_vapid_subject_prefers_tailscale_dns(self) -> None:
        with patch("codoxear.voice_push.subprocess.check_output", return_value='{"Self":{"DNSName":"demo.tail.ts.net."}}'):
            self.assertEqual(_default_vapid_subject(), "https://demo.tail.ts.net")

    def test_latest_narration_merges_pending_message_for_same_session(self) -> None:
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
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            older = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "older narration"}],
                        },
                        "ts": 1.0,
                    }
                ]
            )[0]
            newer = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "newer narration"}],
                        },
                        "ts": 2.0,
                    }
                ]
            )[0]
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[older])
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[newer])
            with coord._lock:
                self.assertEqual(len(coord._queue), 1)
                current = coord._queue[0]
                self.assertEqual(current.message_class, "narration")
                self.assertEqual(current.source_message_ids, (older.message_id, newer.message_id))
                self.assertEqual(current.source_text, "older narration\n\nnewer narration")
            older_row = coord._delivery_ledger[older.message_id]
            newer_row = coord._delivery_ledger[newer.message_id]
            self.assertEqual(older_row["summary_status"], "pending")
            self.assertEqual(older_row["narrated_status"], "pending")
            self.assertEqual(older_row["last_error"], "")
            self.assertEqual(newer_row["summary_status"], "pending")
            self.assertEqual(newer_row["narrated_status"], "pending")
            self.assertEqual(newer_row["last_error"], "")

    def test_generating_message_is_protected_from_queue_eviction(self) -> None:
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
            fake_client = _FakeClient()
            fake_hls = _FakeHLS()
            coord._client = fake_client  # type: ignore[assignment]
            coord._hls = fake_hls  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            coord.observe_messages(
                session_id="sid-1",
                session_display_name="Repo",
                messages=_extract_delivery_messages(
                    [
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "older narration"}],
                            },
                            "ts": 1.0,
                        }
                    ]
                ),
            )
            with coord._lock:
                old_task = coord._queue.pop(0)
                coord._generating_task = old_task
            newer = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "newer narration"}],
                        },
                        "ts": 2.0,
                    }
                ]
            )[0]
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[newer])
            coord._process_task(old_task)
            with coord._lock:
                prepared = coord._prepared
            self.assertIsNotNone(prepared)
            assert prepared is not None
            coord._append_prepared(prepared)
            self.assertEqual(len(fake_hls.append_calls), 1)
            with coord._lock:
                current = coord._queue[0]
                self.assertEqual(current.source_message_ids, (newer.message_id,))
                self.assertEqual(current.source_text, "newer narration")
            old_row = [row for row in coord._delivery_ledger.values() if row["preview_text"] == "older narration"][0]
            self.assertEqual(old_row["summary_status"], "sent")
            self.assertEqual(old_row["narrated_status"], "sent")

    def test_listener_drop_clears_pending_voice_state_and_resets_hls(self) -> None:
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
            fake_hls = _FakeHLS()
            coord._hls = fake_hls  # type: ignore[assignment]
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            queued = AnnouncementTask(
                message_id="queue-1",
                source_message_ids=("queue-1",),
                session_id="sid-1",
                session_display_name="Repo",
                message_class="narration",
                source_text="queued narration",
                spoken_text="",
                notification_text="",
                voice="alloy",
                ts=1.0,
                summary_word_target=15,
                listener_epoch=coord._listener_epoch,
            )
            generating = AnnouncementTask(
                message_id="gen-1",
                source_message_ids=("gen-1",),
                session_id="sid-2",
                session_display_name="Repo 2",
                message_class="narration",
                source_text="generating narration",
                spoken_text="",
                notification_text="",
                voice="ash",
                ts=2.0,
                summary_word_target=15,
                listener_epoch=coord._listener_epoch,
            )
            prepared = AnnouncementTask(
                message_id="prep-1",
                source_message_ids=("prep-1",),
                session_id="sid-3",
                session_display_name="Repo 3",
                message_class="final_response",
                source_text="prepared final",
                spoken_text="Turn summary from Repo 3. prepared final",
                notification_text="prepared final",
                voice="nova",
                ts=3.0,
                summary_word_target=None,
                listener_epoch=coord._listener_epoch,
            )
            now_ts = 10.0
            with coord._lock:
                for task in (queued, generating, prepared):
                    coord._delivery_ledger[task.message_id] = {
                        "message_id": task.message_id,
                        "session_id": task.session_id,
                        "session_display_name": task.session_display_name,
                        "message_class": task.message_class,
                        "preview_text": task.source_text,
                        "notification_text": task.notification_text,
                        "summary_text": "",
                        "summary_status": "pending",
                        "narrated_status": "pending",
                        "push_status": "skipped",
                        "voice": task.voice,
                        "created_ts": now_ts,
                        "updated_ts": now_ts,
                        "last_error": "",
                    }
                coord._queue = [queued]
                coord._generating_task = generating
                coord._prepared = GeneratedAnnouncement(task=prepared, audio_bytes=b"ready")
            payload = coord.listener_heartbeat(client_id="listener-1", enabled=False)
            self.assertEqual(payload["active_listener_count"], 0)
            self.assertEqual(fake_hls.reset_calls, 1)
            with coord._lock:
                self.assertEqual(coord._queue, [])
                self.assertIsNone(coord._generating_task)
                self.assertIsNone(coord._prepared)
                self.assertIsNone(coord._playing_task)
            for task in (queued, generating, prepared):
                row = coord._delivery_ledger[task.message_id]
                self.assertEqual(row["narrated_status"], "skipped")
                self.assertEqual(row["last_error"], "no active listener")

    def test_old_epoch_generation_is_dropped_after_listener_returns(self) -> None:
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
            fake_client = _FakeClient()
            fake_hls = _FakeHLS()
            coord._client = fake_client  # type: ignore[assignment]
            coord._hls = fake_hls  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": False,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            message = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Long and verbose narration body"}],
                        },
                        "ts": 4.0,
                    }
                ]
            )[0]
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[message])
            with coord._lock:
                task = coord._queue.pop(0)
            coord.listener_heartbeat(client_id="listener-1", enabled=False)
            coord.listener_heartbeat(client_id="listener-2", enabled=True)
            coord._process_task(task)
            with coord._lock:
                self.assertIsNone(coord._prepared)
            self.assertEqual(fake_hls.append_calls, [])
            row = coord._delivery_ledger[message.message_id]
            self.assertEqual(row["narrated_status"], "skipped")
            self.assertEqual(row["last_error"], "no active listener")

    def test_queued_final_response_coexists_with_newer_narration(self) -> None:
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
            fake_client = _FakeClient()
            coord._client = fake_client  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            older = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": "older final response"}],
                        },
                        "ts": 1.0,
                    }
                ]
            )[0]
            newer = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "newer narration"}],
                        },
                        "ts": 2.0,
                    }
                ]
            )
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=[older])
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=newer)
            with coord._lock:
                self.assertEqual(len(coord._queue), 2)
                self.assertEqual([task.message_class for task in coord._queue], ["final_response", "narration"])
                self.assertEqual(coord._queue[1].source_text, "newer narration")
            old_row = coord._delivery_ledger[older.message_id]
            self.assertEqual(old_row["summary_status"], "sent")
            self.assertEqual(old_row["narrated_status"], "pending")
            self.assertEqual(old_row["last_error"], "")

    def test_queue_replacement_does_not_evict_generating_or_playing_items(self) -> None:
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
            generating = AnnouncementTask(
                message_id="gen-1",
                source_message_ids=("gen-1",),
                session_id="sid-1",
                session_display_name="Repo",
                message_class="narration",
                source_text="protected generating",
                spoken_text="",
                notification_text="",
                voice="alloy",
                ts=1.0,
                summary_word_target=15,
                listener_epoch=0,
            )
            playing = AnnouncementTask(
                message_id="play-1",
                source_message_ids=("play-1",),
                session_id="sid-2",
                session_display_name="Repo 2",
                message_class="narration",
                source_text="protected playing",
                spoken_text="",
                notification_text="",
                voice="ash",
                ts=1.5,
                summary_word_target=15,
                listener_epoch=0,
            )
            queued_old = AnnouncementTask(
                message_id="queue-1",
                source_message_ids=("queue-1",),
                session_id="sid-1",
                session_display_name="Repo",
                message_class="narration",
                source_text="old queued",
                spoken_text="",
                notification_text="",
                voice="alloy",
                ts=2.0,
                summary_word_target=15,
                listener_epoch=0,
            )
            queued_new = AnnouncementTask(
                message_id="queue-2",
                source_message_ids=("queue-2",),
                session_id="sid-1",
                session_display_name="Repo",
                message_class="narration",
                source_text="new queued",
                spoken_text="",
                notification_text="",
                voice="alloy",
                ts=3.0,
                summary_word_target=15,
                listener_epoch=0,
            )
            now_ts = 10.0
            with coord._lock:
                for task in (generating, playing, queued_old, queued_new):
                    coord._delivery_ledger[task.message_id] = {
                        "message_id": task.message_id,
                        "session_id": task.session_id,
                        "session_display_name": task.session_display_name,
                        "message_class": task.message_class,
                        "preview_text": task.source_text,
                        "summary_text": "",
                        "summary_status": "pending",
                        "narrated_status": "pending",
                        "push_status": "skipped",
                        "voice": task.voice,
                        "created_ts": now_ts,
                        "updated_ts": now_ts,
                        "last_error": "",
                    }
                coord._generating_task = generating
                coord._playing_task = playing
                coord._queue = [queued_old]
                coord._enqueue_task_locked(queued_new)
            with coord._lock:
                self.assertEqual(coord._generating_task.message_id, "gen-1")
                self.assertEqual(coord._playing_task.message_id, "play-1")
                self.assertEqual(len(coord._queue), 1)
                self.assertEqual(coord._queue[0].source_message_ids, ("queue-1", "queue-2"))
                self.assertEqual(coord._queue[0].source_text, "old queued\n\nnew queued")
            old_row = coord._delivery_ledger["queue-1"]
            self.assertEqual(old_row["summary_status"], "pending")
            self.assertEqual(old_row["narrated_status"], "pending")
            self.assertEqual(old_row["last_error"], "")

    def test_mixed_batch_preserves_one_queue_slot_per_class(self) -> None:
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
            fake_client = _FakeClient()
            coord._client = fake_client  # type: ignore[assignment]
            coord.set_settings(
                {
                    "tts_enabled_for_narration": True,
                    "tts_enabled_for_final_response": True,
                    "tts_base_url": "https://api.openai.com/v1",
                    "tts_api_key": "test-key",
                }
            )
            coord.listener_heartbeat(client_id="listener-1", enabled=True)
            mixed = _extract_delivery_messages(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": "older final response"}],
                        },
                        "ts": 1.0,
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "newer narration"}],
                        },
                        "ts": 2.0,
                    },
                ]
            )
            coord.observe_messages(session_id="sid-1", session_display_name="Repo", messages=mixed)
            with coord._lock:
                self.assertEqual(len(coord._queue), 2)
                self.assertEqual([task.message_class for task in coord._queue], ["final_response", "narration"])
                self.assertEqual(coord._queue[1].source_text, "newer narration")
            final_row = [row for row in coord._delivery_ledger.values() if row["message_class"] == "final_response"][0]
            self.assertEqual(final_row["summary_status"], "sent")
            self.assertEqual(final_row["narrated_status"], "pending")
            self.assertEqual(final_row["last_error"], "")

    def test_hls_segments_are_written_in_sequence_order(self) -> None:
        with TemporaryDirectory() as td:
            stream = MergedHLSStream(root_dir=Path(td))
            stream._store_segment(seq=2, segment_name="000002-b.ts", segment_path=Path(td) / "000002-b.ts", duration=1.0)
            stream._store_segment(seq=1, segment_name="000001-a.ts", segment_path=Path(td) / "000001-a.ts", duration=1.0)
            playlist = stream.playlist_bytes().decode("utf-8")
            self.assertLess(playlist.index("000001-a.ts"), playlist.index("000002-b.ts"))

    def test_hls_target_duration_tracks_longest_segment(self) -> None:
        with TemporaryDirectory() as td:
            stream = MergedHLSStream(root_dir=Path(td))
            stream._store_segment(seq=1, segment_name="000001-a.ts", segment_path=Path(td) / "000001-a.ts", duration=27.2)
            playlist = stream.playlist_bytes().decode("utf-8")
            self.assertIn("#EXT-X-TARGETDURATION:28", playlist)

    def test_hls_reset_clears_live_playlist(self) -> None:
        with TemporaryDirectory() as td:
            stream = MergedHLSStream(root_dir=Path(td))
            segment_path = Path(td) / "000001-a.ts"
            segment_path.write_bytes(b"audio")
            stream._store_segment(seq=1, segment_name="000001-a.ts", segment_path=segment_path, duration=1.0)
            self.assertIn("000001-a.ts", stream.playlist_bytes().decode("utf-8"))
            stream.reset()
            playlist = stream.playlist_bytes().decode("utf-8")
            self.assertNotIn("000001-a.ts", playlist)
            self.assertEqual(stream.snapshot()["segment_count"], 0)

    def test_append_audio_skips_invalid_na_segment_instead_of_crashing(self) -> None:
        with TemporaryDirectory() as td:
            stream = MergedHLSStream(root_dir=Path(td))
            prefix = "abc123def456"

            def fake_run(*args, **kwargs):
                segdir = Path(td) / "segments"
                (segdir / f"{prefix}-part-000.ts").write_bytes(b"good")
                (segdir / f"{prefix}-part-001.ts").write_bytes(b"bad")
                return subprocess.CompletedProcess(args[0], 0, b"", b"")

            def fake_duration(path: Path) -> float:
                if path.name.endswith("000.ts"):
                    return 1.25
                raise RuntimeError("invalid ffprobe duration: N/A")

            with patch("codoxear.voice_push.shutil.which", return_value="/usr/bin/fake"), \
                patch("codoxear.voice_push.subprocess.run", side_effect=fake_run), \
                patch.object(stream, "_segment_duration_seconds", side_effect=fake_duration):
                total = stream.append_audio(message_id=prefix, audio_bytes=b"fake-audio")

            self.assertEqual(total, 1.25)
            playlist = stream.playlist_bytes().decode("utf-8")
            self.assertIn(".ts", playlist)
            self.assertEqual(playlist.count("#EXTINF:"), 1)


if __name__ == "__main__":
    unittest.main()
