from __future__ import annotations

import json

import pytest

from codoxear.activity_peek import format_tool_label, scan_window_activity


def _pi(*parts):
    return {"type": "message", "message": {"role": "assistant", "content": list(parts)}}


def _cc(*parts):
    return {"type": "assistant", "message": {"role": "assistant", "content": list(parts)}}


def _codex(kind, **values):
    return {"type": "response_item", "payload": {"type": kind, **values}}


@pytest.mark.parametrize("call,result", [
    (
        _pi({"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "git diff --stat"}}),
        {"type": "message", "message": {"role": "toolResult", "toolCallId": "t1"}},
    ),
    (
        _cc({"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "git diff --stat"}}),
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]}},
    ),
    (
        _codex("function_call", call_id="t1", name="shell", arguments=json.dumps({"command": ["git", "diff", "--stat"]})),
        _codex("function_call_output", call_id="t1", output="done"),
    ),
    (
        _codex("custom_tool_call", call_id="t1", name="bash", arguments=json.dumps({"command": "git diff --stat"})),
        _codex("custom_tool_call_output", call_id="t1", output="done"),
    ),
    (
        {"type": "event_msg", "payload": {"type": "exec_command_begin", "call_id": "t1", "command": ["git", "diff", "--stat"]}},
        {"type": "event_msg", "payload": {"type": "exec_command_end", "call_id": "t1"}},
    ),
    (
        _codex("local_shell_call", call_id="t1", action={"type": "exec", "command": ["git", "diff", "--stat"]}),
        _codex("function_call_output", call_id="t1", output="done"),
    ),
])
def test_backend_calls_and_results(call, result):
    assert scan_window_activity([call]) == (True, {"label": "bash: git diff --stat", "running": True})
    # A fast tool finishes inside one read window: the result flips the call
    # to not-running instead of erasing it before it is ever displayed.
    assert scan_window_activity([call, result]) == (True, {"label": "bash: git diff --stat", "running": False})
    # A result whose call is outside the window emits the flip-only signal so
    # the client drops the running marker from the call it is already showing.
    assert scan_window_activity([result]) == (True, {"label": None, "running": False})
    assert scan_window_activity([result, call]) == (True, {"label": "bash: git diff --stat", "running": True})


def test_last_call_wins_and_older_results_do_not_erase_it():
    older = _codex("function_call", call_id="old", name="read", arguments='{"path": "old.py"}')
    newer = _codex("function_call", call_id="new", name="read", arguments='{"path": "new.py"}')
    old_result = _codex("function_call_output", call_id="old")
    new_result = _codex("function_call_output", call_id="new")
    assert scan_window_activity([older, newer, old_result]) == (True, {"label": "read: new.py", "running": True})
    # The newer call's own result flips it; an older parallel call's result
    # leaves the running newer call untouched.
    assert scan_window_activity([older, newer, old_result, new_result]) == (True, {"label": "read: new.py", "running": False})
    assert scan_window_activity([newer, old_result]) == (True, {"label": "read: new.py", "running": True})


def test_order_with_multiple_parts_and_results():
    row = _pi(
        {"type": "thinking", "thinking": "inspect the code"},
        {"type": "toolCall", "id": "a", "name": "Read", "arguments": {"path": "a.py"}},
        {"type": "toolCall", "id": "b", "name": "Write", "arguments": {"path": "b.py"}},
    )
    assert scan_window_activity([row]) == (True, {"label": "write: b.py", "running": True})


def test_result_after_intervening_thinking_keeps_the_thinking_label():
    call = _codex("function_call", call_id="t1", name="read", arguments='{"path": "a.py"}')
    thinking = _pi({"type": "thinking", "thinking": "waiting"})
    result = _codex("function_call_output", call_id="t1")
    # The thinking row is newer than the call; the result only flips a line
    # that is still marked running, it never resurrects the older call.
    assert scan_window_activity([call, thinking, result]) == (True, {"label": "thinking: waiting", "running": False})


@pytest.mark.parametrize("row,text", [
    (_pi({"type": "thinking", "thinking": "\n  inspect\t the code\nsecond line"}), "thinking: inspect the code"),
    (_cc({"type": "thinking", "thinking": "checking the model\nsecond"}), "thinking: checking the model"),
    (_pi({"type": "thinking"}), "thinking"),
    (_cc({"type": "thinking", "thinking": {}}), "thinking"),
    (_codex("reasoning", summary=[{"type": "summary_text", "text": ""}, {"type": "summary_text", "text": "check evidence\nmore"}]), "thinking: check evidence"),
    (_codex("reasoning", summary=[]), "thinking"),
    ({"type": "event_msg", "payload": {"type": "agent_reasoning", "text": "compare alternatives\nmore"}}, "thinking: compare alternatives"),
    ({"type": "event_msg", "payload": {"type": "agent_reasoning", "message": "compare alternatives"}}, "thinking: compare alternatives"),
])
def test_backend_thinking(row, text):
    assert scan_window_activity([row]) == (True, {"label": text, "running": False})


@pytest.mark.parametrize("name,args,expected", [
    ("Bash", {"command": "git\t diff  --stat\nignored"}, "bash: git diff --stat"),
    ("functions.shell", '{"command": ["git", "diff", "--stat"]}', "bash: git diff --stat"),
    ("exec", {"cmd": "ls -la"}, "bash: ls -la"),
    ("local_shell_call", {"command": "pwd"}, "bash: pwd"),
    ("Read", {"path": "src/app.py"}, "read: src/app.py"),
    ("Write", {"file_path": "src/app.py", "content": "lots of content"}, "write: src/app.py"),
    ("Edit", {"file_path": "src/app.py"}, "edit: src/app.py"),
    ("str_replace", {"path": "app.py"}, "str_replace: app.py"),
    ("apply_patch", {"path": "app.py"}, "apply_patch: app.py"),
    ("Grep", {"pattern": "find me"}, "grep: find me"),
    ("search", {"query": "find me"}, "search: find me"),
    ("glob", {"pattern": "*.py"}, "glob: *.py"),
    ("web_fetch", {"url": "https://example.org"}, "web_fetch: https://example.org"),
    ("web_search", {"query": "current docs"}, "web_search: current docs"),
    ("Task", {"description": "inspect the module\nsecond line"}, "subagent: inspect the module"),
    ("subagents", {"task": "inspect the module"}, "subagent: inspect the module"),
    ("custom", {"nested": {}, "query": "short value"}, "custom: short value"),
    ("custom", {"nested": {}, "limit": 3}, "custom: 3"),
    ("custom", {"nested": []}, "custom"),
    ("Bash", "malformed JSON", "bash"),
    ("Read", None, "read"),
    ("Bash", {"command": ["ls", {}]}, "bash"),
])
def test_tool_labels(name, args, expected):
    assert format_tool_label(name, args) == expected


def test_cap_and_whitespace_are_applied_to_all_labels():
    for label in [
        format_tool_label("BASH", {"command": "x" * 200}),
        scan_window_activity([_pi({"type": "thinking", "thinking": "x" * 200})])[1]["label"],
        format_tool_label("x" * 200, None),
    ]:
        assert len(label) == 120
        assert label.endswith("…")
        assert label.count("…") == 1
    assert len(format_tool_label("read", {"path": "x" * 114})) == 120
    assert not format_tool_label("read", {"path": "x" * 114}).endswith("…")


@pytest.mark.parametrize("row", [
    None, 1, "text", [], {}, {"type": []},
    {"type": "message", "message": []},
    {"type": "assistant", "message": {"role": "assistant", "content": [None, 2, {}, {"type": []}]}},
    _pi({"type": "toolCall", "name": []}),
    _pi({"type": "text", "text": "ordinary response"}),
    {"type": "response_item", "payload": []},
    _codex([], arguments=[]),
    _codex("function_call", name=None),
    {"type": "event_msg", "payload": {"type": "token_count"}},
    {"type": "message", "message": {"role": "toolResult"}},
])
def test_unrecognized_or_malformed_rows_are_neutral(row):
    assert scan_window_activity([row]) == (False, None)


def test_malformed_arguments_still_expose_a_recognized_call():
    row = _pi({"type": "toolCall", "name": "Read", "arguments": [None]})
    assert scan_window_activity([row]) == (True, {"label": "read", "running": True})


def test_transcript_excluded_pi_rows_never_surface_in_the_peek():
    # Harness/intercom plumbing is filtered from every transcript projection;
    # the peek is transcript chrome and must apply the same boundary.
    excluded_call = {
        "type": "message",
        "metadata": {"tags": ["intercom"]},
        "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "t9", "name": "bash", "arguments": {"command": "secret internal"}}],
        },
    }
    visible_call = _pi({"type": "toolCall", "id": "t1", "name": "read", "arguments": {"path": "a.py"}})
    assert scan_window_activity([excluded_call]) == (False, None)
    assert scan_window_activity([visible_call, excluded_call]) == (True, {"label": "read: a.py", "running": True})
