from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from codoxear import rollout_log
from codoxear.activity_peek import scan_window_activity
from codoxear.broker_turn_state import State, _apply_rollout_obj_to_state
from codoxear.cc_log import cc_message_role, cc_user_text
from codoxear.message_routes import _read_chat_export_events
from codoxear.rollout_chat_events import _single_chat_event, _chat_assistant_dedupe_key
from codoxear.transcript_search import search_chat_logs_bounded, read_chat_window_around

FIXTURES = Path(__file__).parent / "fixtures" / "compaction"


def rows(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


def markers(events):
    return [event for event in events if event.get("kind") == "compaction"]


def write_log(path, objects):
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objects))
    return path


def test_real_pi_window_metadata_and_no_summary(tmp_path):
    objects = rows("pi-real.jsonl")
    events, _, _, _ = rollout_log._extract_chat_events(objects)
    boundary, = markers(events)
    assert boundary["role"] == "system"
    assert boundary["tokens_before"] == 123460
    assert boundary["tokens_after"] == 62227  # not the summarizer's 20834
    assert boundary["kept_messages"] == 2
    assert boundary["ts"] > 0
    assert not {"text", "summary", "details", "usage", "trigger", "fromHook"} & boundary.keys()
    path = write_log(tmp_path / "pi.jsonl", objects)
    tail, before, after, has_older = rollout_log._read_chat_tail_page(path, limit=20)
    tail_marker, = markers(tail)
    assert tail_marker["tokens_after"] == 62227
    assert tail_marker["kept_messages"] == 2
    assert tail_marker["message_id"].startswith("compaction:")
    assert "_before_byte" in tail_marker
    assert before == 0 and after == path.stat().st_size and not has_older
    exported = _read_chat_export_events(path, max_bytes=100000)
    assert markers(exported) == markers(tail)
    live = rollout_log._read_chat_live_delta(path, after_byte=0)[0]
    assert markers(live) == markers(tail)
    history = rollout_log._read_chat_history_page(path, before_byte=after, limit=20)[0]
    assert markers(history) == markers(tail)


def test_pi_partial_window_omits_unprovable_metadata():
    objects = rows("pi-real.jsonl")
    compact = next(obj for obj in objects if obj["type"] == "compaction")
    boundary, = markers(rollout_log._extract_chat_events([compact])[0])
    assert boundary["tokens_before"] == 123460
    assert "tokens_after" not in boundary and "kept_messages" not in boundary
    assert "trigger" not in boundary
    # A hook is not an auto/manual discriminator.
    assert "trigger" not in _single_chat_event({**compact, "fromHook": True})
    assert "tokens_before" not in _single_chat_event({**compact, "tokensBefore": True})


def test_tail_enrichment_does_not_add_pi_file_reads(tmp_path):
    path = write_log(tmp_path / "pi.jsonl", rows("pi-real.jsonl"))
    with patch.object(rollout_log, "_iter_jsonl_records_reverse", wraps=rollout_log._iter_jsonl_records_reverse) as read:
        rollout_log._read_chat_tail_page(path, limit=30)
    # Existing prior-open-turn scans run independently; no compaction-specific
    # lookbehind is allowed. Compare with identical rows minus the compaction.
    compaction_reads = read.call_count
    plain = write_log(tmp_path / "plain.jsonl", [o for o in rows("pi-real.jsonl") if o["type"] != "compaction"])
    with patch.object(rollout_log, "_iter_jsonl_records_reverse", wraps=rollout_log._iter_jsonl_records_reverse) as read:
        rollout_log._read_chat_tail_page(plain, limit=30)
    assert compaction_reads == read.call_count


def test_cc_boundary_and_summary_exclusion_everywhere(tmp_path):
    objects = rows("cc.jsonl")
    events, _, _, _ = rollout_log._extract_chat_events(objects)
    boundary, = markers(events)
    assert boundary["tokens_before"] == 167189 and boundary["trigger"] == "auto"
    assert len([ev for ev in events if ev["role"] == "user"]) == 2
    summary = next(o for o in objects if o.get("isCompactSummary"))
    assert cc_user_text(summary) is None and cc_message_role(summary) is None
    assert _single_chat_event(summary) is None
    assert not rollout_log._extract_chat_events([summary])[2]["turn_start"]
    path = write_log(tmp_path / "cc.jsonl", objects)
    assert search_chat_logs_bounded([path], "REDACTED")[0] == 0
    count, users, _ = search_chat_logs_bounded([path], "*", role="user", match_all=True)
    assert count == 2 and [u["text"] for u in users] == ["Before compaction", "After compaction"]
    assert all("REDACTED" not in str(e) for e in _read_chat_export_events(path, max_bytes=100000))
    assert _single_chat_event({**objects[2], "compactMetadata": {"trigger": "manual"}})["trigger"] == "manual"


@pytest.mark.parametrize("name", ["pi-real.jsonl", "cc.jsonl", "codex.jsonl"])
def test_boundaries_are_neutral_to_turn_dedupe_voice_and_activity(name, tmp_path):
    objects = [o for o in rows(name) if _single_chat_event(o) and _single_chat_event(o).get("kind") == "compaction" or o.get("isCompactSummary")]
    events, meta, flags, _ = rollout_log._extract_chat_events(objects)
    assert meta == {"thinking": 0, "thinking_tokens": 0, "tool": 0, "system": 0}
    assert flags == {"turn_start": False, "turn_end": False, "turn_aborted": False}
    assert rollout_log._extract_delivery_messages(objects) == []
    assert scan_window_activity(objects) == (False, None)
    assert all(_chat_assistant_dedupe_key(ev) is None for ev in events)
    st = State(codex_pid=123, pty_master_fd=-1, cwd=str(tmp_path), start_ts=0, codex_home=tmp_path, sessions_dir=tmp_path)
    for obj in objects:
        _apply_rollout_obj_to_state(st, obj, 10)
    assert not st.busy and not st.turn_open and not st.backend_turn_activity_seen
    assert st.last_turn_activity_ts == 0


@pytest.mark.parametrize("variant", ["checkpoint", "notification", "wrapped", "paired"])
def test_codex_forms_and_pair_dedupe(variant, tmp_path):
    objects = rows("codex.jsonl")
    checkpoint, notification = objects[1:3]
    variants = {"checkpoint": [checkpoint], "notification": [notification], "wrapped": [{**checkpoint, "type": "response_item", "payload": {"type": "compacted", "message": "REDACTED"}}], "paired": [checkpoint, notification]}
    selected = variants[variant]
    assert len(markers(rollout_log._extract_chat_events(selected)[0])) == 1
    path = write_log(tmp_path / f"{variant}.jsonl", selected)
    assert len(markers(rollout_log._read_chat_tail_page(path, limit=10)[0])) == 1
    assert len(markers(_read_chat_export_events(path, max_bytes=100000))) == 1


@pytest.mark.parametrize("summary_bytes", [0, 100000])
def test_codex_pair_split_across_live_and_history_boundaries(tmp_path, summary_bytes):
    objects = rows("codex.jsonl")
    if summary_bytes:
        objects[1]["payload"]["message"] = "x" * summary_bytes
    path = write_log(tmp_path / "codex.jsonl", objects[:2])
    first, after, *_ = rollout_log._read_chat_live_delta(path, after_byte=0)
    first_marker, = markers(first)
    with path.open("a") as file:
        file.write(json.dumps(objects[2]) + "\n")
    second, *_ = rollout_log._read_chat_live_delta(path, after_byte=after)
    assert markers(second)[0]["message_id"] == first_marker["message_id"]
    # A one-row tail canonicalizes to the checkpoint even when it initially
    # reads only the notification. Its older cursor cannot re-read the pair.
    tail, before, _, older = rollout_log._read_chat_tail_page(path, limit=1)
    assert markers(tail)[0]["message_id"] == first_marker["message_id"]
    history = rollout_log._read_chat_history_page(path, before_byte=before, limit=10)[0]
    assert not markers(history)
    assert older
    window, *_ = read_chat_window_around(path, position_byte=after, before_limit=10, after_limit=10)
    assert len(markers(window)) == 1
    assert markers(window)[0]["message_id"] == first_marker["message_id"]
