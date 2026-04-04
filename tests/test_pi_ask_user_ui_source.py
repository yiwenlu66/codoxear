import json
import subprocess
import textwrap
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def extract_js_function(source: str, needle: str) -> str:
    start = source.index(needle)
    paren_start = source.index("(", start)
    paren_depth = 0
    body_start = -1
    for idx in range(paren_start, len(source)):
        ch = source[idx]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
            if paren_depth == 0:
                body_start = source.index("{", idx)
                break
    if body_start < 0:
        raise ValueError(f"could not locate body for {needle!r}")
    depth = 0
    for idx in range(body_start, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise ValueError(f"could not extract function block for {needle!r}")


def eval_fetch_session_ui_state_with_html_404() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "async function api(")
        + "\n"
        + extract_js_function(source, "async function fetchSessionUiState(")
        + "\n"
        + "globalThis.__test_fetchSessionUiState = fetchSessionUiState;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          performance: {{ now: () => 0 }},
          pushPerfSample: () => {{}},
          resolveAppUrl: (path) => String(path || ""),
          fetch: async () => ({{
            ok: false,
            status: 404,
            text: async () => "<!DOCTYPE html><html><body>not found</body></html>",
          }}),
          latestSessions: [{{ session_id: "pi-session", backend: "pi" }}],
          sessionAgentBackend: (s) => String(s && s.backend || "").trim().toLowerCase() === "pi" ? "pi" : "codex",
          console: {{ error: () => {{}} }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        (async () => {{
          const result = await ctx.__test_fetchSessionUiState("pi-session");
          process.stdout.write(JSON.stringify(result));
        }})().catch((err) => {{
          process.stderr.write(String(err && err.stack || err));
          process.exit(1);
        }});
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_normalize_ask_user_options(options: list[object]) -> list[dict[str, str]]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "function normalizeAskUserOptions(")
        + "\n"
        + "globalThis.__test_normalizeAskUserOptions = normalizeAskUserOptions;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const result = ctx.__test_normalizeAskUserOptions({json.dumps(options)});
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_normalize_cache_event(event: dict[str, object]) -> dict[str, object] | None:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "function normalizeAskUserOptions(")
        + "\n"
        + extract_js_function(source, "function normalizeCacheEvent(")
        + "\n"
        + "globalThis.__test_normalizeCacheEvent = normalizeCacheEvent;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const result = ctx.__test_normalizeCacheEvent({json.dumps(event)});
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_clear_resolved_ask_user_pending_responses(
    pending_by_session: dict[str, dict[str, dict[str, object]]],
    events: list[dict[str, object]],
    sid: str,
) -> dict[str, dict[str, dict[str, object]]]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        "const uiResponsePendingBySession = new Map();\n"
        + "const uiResponseDraftBySession = new Map();\n"
        + extract_js_function(source, "function askUserRequestId(")
        + "\n"
        + extract_js_function(source, "function askUserPendingResponseMap(")
        + "\n"
        + extract_js_function(source, "function askUserDraftMap(")
        + "\n"
        + extract_js_function(source, "function forgetPendingUiResponse(")
        + "\n"
        + extract_js_function(source, "function forgetAskUserDraft(")
        + "\n"
        + extract_js_function(source, "function clearResolvedAskUserPendingResponses(")
        + "\n"
        + "globalThis.__test_clearResolvedAskUserPendingResponses = clearResolvedAskUserPendingResponses;\n"
        + "globalThis.__test_uiResponsePendingBySession = uiResponsePendingBySession;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        for (const [sessionId, entries] of Object.entries({json.dumps(pending_by_session)})) {{
          ctx.__test_uiResponsePendingBySession.set(sessionId, new Map(Object.entries(entries)));
        }}
        ctx.__test_clearResolvedAskUserPendingResponses({json.dumps(events)}, {json.dumps(sid)});
        const out = Object.fromEntries(
          Array.from(ctx.__test_uiResponsePendingBySession.entries(), ([sessionId, entries]) => [
            sessionId,
            Object.fromEntries(entries),
          ])
        );
        process.stdout.write(JSON.stringify(out));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_synthetic_ask_user_event_for_request(req: dict[str, object], sid: str) -> dict[str, object] | None:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "function normalizeAskUserOptions(")
        + "\n"
        + extract_js_function(source, "function isAskUserUiRequest(")
        + "\n"
        + extract_js_function(source, "function askUserBooleanFlag(")
        + "\n"
        + extract_js_function(source, "function syntheticAskUserEventForRequest(")
        + "\n"
        + "globalThis.__test_syntheticAskUserEventForRequest = syntheticAskUserEventForRequest;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const result = ctx.__test_syntheticAskUserEventForRequest({json.dumps(req)}, {json.dumps(sid)});
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_synthetic_ask_user_event_for_pending_response(
    request_id: str, pending: dict[str, object], sid: str
) -> dict[str, object] | None:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "function normalizeAskUserOptions(")
        + "\n"
        + extract_js_function(source, "function isAskUserUiRequest(")
        + "\n"
        + extract_js_function(source, "function askUserBooleanFlag(")
        + "\n"
        + extract_js_function(source, "function normalizedAskUserAnswerValues(")
        + "\n"
        + extract_js_function(source, "function syntheticAskUserEventForPendingResponse(")
        + "\n"
        + "globalThis.__test_syntheticAskUserEventForPendingResponse = syntheticAskUserEventForPendingResponse;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const result = ctx.__test_syntheticAskUserEventForPendingResponse(
          {json.dumps(request_id)},
          {json.dumps(pending)},
          {json.dumps(sid)}
        );
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_build_ask_user_submission_payload(ev: dict[str, object], draft: dict[str, object]) -> dict[str, object] | None:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "function askUserRequestId(")
        + "\n"
        + extract_js_function(source, "function askUserBooleanFlag(")
        + "\n"
        + extract_js_function(source, "function normalizedAskUserAnswerValues(")
        + "\n"
        + extract_js_function(source, "function buildAskUserSubmissionPayload(")
        + "\n"
        + "globalThis.__test_buildAskUserSubmissionPayload = buildAskUserSubmissionPayload;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const result = ctx.__test_buildAskUserSubmissionPayload({json.dumps(ev)}, {json.dumps(draft)});
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def eval_merge_ask_user_event_update(
    existing_event: dict[str, object] | None,
    next_event: dict[str, object] | None,
) -> dict[str, object] | None:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = (
        extract_js_function(source, "function normalizeAskUserOptions(")
        + "\n"
        + extract_js_function(source, "function askUserRequestId(")
        + "\n"
        + extract_js_function(source, "function askUserBooleanFlag(")
        + "\n"
        + extract_js_function(source, "function askUserHasPromptMetadata(")
        + "\n"
        + extract_js_function(source, "function mergeAskUserEventUpdate(")
        + "\n"
        + "globalThis.__test_mergeAskUserEventUpdate = mergeAskUserEventUpdate;\n"
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const result = ctx.__test_mergeAskUserEventUpdate(
          {json.dumps(existing_event)},
          {json.dumps(next_event)}
        );
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def test_app_js_exposes_fetch_submit_and_row_helpers() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "async function fetchSessionUiState(sid)" in source
    assert "function normalizeAskUserOptions(rawOptions)" in source
    assert "function makeAskUserRow(ev)" in source
    assert "submitSessionUiResponse(" in source
    assert "/ui_response" in source
    assert "resolved-awaiting-log" in source


def test_top_level_ui_state_helper_does_not_capture_render_local_session_index() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    pre_render_app = source[: source.index("function renderApp() {")]

    assert "sessionIndex" not in pre_render_app


def test_fetch_session_ui_state_tolerates_html_404_from_older_server() -> None:
    result = eval_fetch_session_ui_state_with_html_404()

    assert result == {"requests": []}


def test_render_pipeline_handles_ask_user_rows() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'if (ev.type === "ask_user") return makeAskUserRow(ev).row;' in source


def test_normalize_ask_user_options_preserves_structured_entries() -> None:
    options = eval_normalize_ask_user_options(
        [
            "Simple",
            {"title": "Structured", "description": "Shown beneath the title"},
            {"title": "No description"},
            {"description": "ignored"},
            42,
        ]
    )

    assert options == [
        {"title": "Simple", "description": ""},
        {"title": "Structured", "description": "Shown beneath the title"},
        {"title": "No description", "description": ""},
    ]


def test_normalize_cache_event_preserves_structured_ask_user_options() -> None:
    event = eval_normalize_cache_event(
        {
            "type": "ask_user",
            "tool_call_id": "ask-structured",
            "question": "Where should it go?",
            "context": "Need a placement decision.",
            "options": [
                "Sidebar",
                {"title": "Details", "description": "Inline with the main content"},
                {"title": "Dock"},
                {"description": "ignored"},
            ],
            "allow_freeform": True,
            "allow_multiple": False,
            "resolved": True,
            "answer": ["Sidebar", 7, None],
            "cancelled": False,
            "was_custom": True,
            "ts": 12.5,
        }
    )

    assert event == {
        "type": "ask_user",
        "tool_call_id": "ask-structured",
        "question": "Where should it go?",
        "context": "Need a placement decision.",
        "options": [
            {"title": "Sidebar", "description": ""},
            {"title": "Details", "description": "Inline with the main content"},
            {"title": "Dock", "description": ""},
        ],
        "allow_freeform": True,
        "allow_multiple": False,
        "resolved": True,
        "answer": ["Sidebar"],
        "was_custom": True,
        "ts": 12.5,
    }


def test_normalize_cache_event_defaults_missing_allow_freeform_to_true() -> None:
    event = eval_normalize_cache_event(
        {
            "type": "ask_user",
            "tool_call_id": "ask-default-freeform",
            "question": "Where should it go?",
            "options": ["Sidebar", "Details"],
            "resolved": False,
        }
    )

    assert event == {
        "type": "ask_user",
        "tool_call_id": "ask-default-freeform",
        "question": "Where should it go?",
        "context": "",
        "options": [
            {"title": "Sidebar", "description": ""},
            {"title": "Details", "description": ""},
        ],
        "allow_freeform": True,
        "allow_multiple": False,
        "resolved": False,
    }


def test_clear_resolved_ask_user_pending_responses_drops_only_stale_entries() -> None:
    remaining = eval_clear_resolved_ask_user_pending_responses(
        pending_by_session={
            "sid-a": {
                "ask-open": {"value": "Sidebar"},
                "ask-done": {"value": "Details"},
            },
            "sid-b": {
                "ask-other": {"value": "Dock"},
            },
        },
        events=[
            {"type": "ask_user", "tool_call_id": "ask-open", "resolved": False},
            {"type": "ask_user", "tool_call_id": "ask-done", "resolved": True},
            {"type": "ask_user", "tool_call_id": "ask-done", "resolved": True},
            {"type": "ask_user", "tool_call_id": None, "resolved": True},
            {"type": "assistant", "text": "Not an ask_user event"},
        ],
        sid="sid-a",
    )

    assert remaining == {
        "sid-a": {
            "ask-open": {"value": "Sidebar"},
        },
        "sid-b": {
            "ask-other": {"value": "Dock"},
        },
    }


def test_synthetic_live_request_preserves_ask_user_modes_and_structured_options() -> None:
    event = eval_synthetic_ask_user_event_for_request(
        {
            "id": "ui-req-42",
            "interaction_kind": "ask_user",
            "method": "select",
            "question": "Choose one or more destinations",
            "context": "You can combine options or type a custom answer.",
            "options": [
                {"title": "Details", "description": "Main pane"},
                "Sidebar",
            ],
            "allow_freeform": True,
            "allow_multiple": True,
        },
        sid="pi-session",
    )

    assert event == {
        "type": "ask_user",
        "tool_call_id": "ui-req-42",
        "question": "Choose one or more destinations",
        "context": "You can combine options or type a custom answer.",
        "options": [
            {"title": "Details", "description": "Main pane"},
            {"title": "Sidebar", "description": ""},
        ],
        "allow_freeform": True,
        "allow_multiple": True,
        "resolved": False,
        "sid": "pi-session",
        "live_only": True,
    }


def test_synthetic_live_request_defaults_missing_allow_freeform_to_true() -> None:
    event = eval_synthetic_ask_user_event_for_request(
        {
            "id": "ui-req-default-freeform",
            "tool_name": "ask_user",
            "method": "select",
            "question": "Choose a destination",
            "options": ["Details", "Sidebar"],
        },
        sid="pi-session",
    )

    assert event == {
        "type": "ask_user",
        "tool_call_id": "ui-req-default-freeform",
        "question": "Choose a destination",
        "context": "",
        "options": [
            {"title": "Details", "description": ""},
            {"title": "Sidebar", "description": ""},
        ],
        "allow_freeform": True,
        "allow_multiple": False,
        "resolved": False,
        "sid": "pi-session",
        "live_only": True,
    }


def test_synthetic_live_request_ignores_generic_select_requests() -> None:
    event = eval_synthetic_ask_user_event_for_request(
        {
            "id": "ui-generic-select",
            "method": "select",
            "question": "Pick a destination",
            "options": ["Details", "Sidebar"],
        },
        sid="pi-session",
    )

    assert event is None


def test_synthetic_pending_response_ignores_unmarked_generic_requests() -> None:
    event = eval_synthetic_ask_user_event_for_pending_response(
        request_id="ui-generic-select",
        pending={
            "question": "Pick a destination",
            "options": ["Details", "Sidebar"],
            "value": "Details",
        },
        sid="pi-session",
    )

    assert event is None


def test_synthetic_pending_response_preserves_multi_select_answer_shape() -> None:
    event = eval_synthetic_ask_user_event_for_pending_response(
        request_id="ask-live-multi",
        pending={
            "interaction_kind": "ask_user",
            "question": "Pick every place that applies",
            "context": "Multiple destinations are allowed.",
            "options": [
                "Details",
                {"title": "Sidebar", "description": "Secondary column"},
            ],
            "allow_freeform": True,
            "allow_multiple": True,
            "value": ["Details", "Sidebar", 7],
        },
        sid="pi-session",
    )

    assert event == {
        "type": "ask_user",
        "tool_call_id": "ask-live-multi",
        "question": "Pick every place that applies",
        "context": "Multiple destinations are allowed.",
        "options": [
            {"title": "Details", "description": ""},
            {"title": "Sidebar", "description": "Secondary column"},
        ],
        "allow_freeform": True,
        "allow_multiple": True,
        "resolved": True,
        "answer": ["Details", "Sidebar"],
        "cancelled": False,
        "resolved_awaiting_log": True,
        "sid": "pi-session",
        "live_only": True,
    }


def test_build_ask_user_submission_payload_supports_freeform_and_multi_select() -> None:
    multi_payload = eval_build_ask_user_submission_payload(
        {
            "tool_call_id": "ask-live-multi",
            "allow_freeform": True,
            "allow_multiple": True,
        },
        {
            "selected": ["Details", "Sidebar"],
            "freeform": "Custom dock",
        },
    )
    single_payload = eval_build_ask_user_submission_payload(
        {
            "tool_call_id": "ask-live-freeform",
            "allow_freeform": True,
            "allow_multiple": False,
        },
        {
            "selected": ["Sidebar"],
            "freeform": "Custom dock",
        },
    )

    assert multi_payload == {"id": "ask-live-multi", "value": ["Details", "Sidebar", "Custom dock"]}
    assert single_payload == {"id": "ask-live-freeform", "value": "Custom dock"}


def test_merge_ask_user_event_update_preserves_prompt_metadata_for_late_resolution() -> None:
    event = eval_merge_ask_user_event_update(
        {
            "type": "ask_user",
            "tool_call_id": "ask-late-resolution",
            "question": "Where should the todo live?",
            "context": "Need a placement decision.",
            "options": [
                "Sidebar",
                {"title": "Details", "description": "Main pane"},
            ],
            "allow_freeform": False,
            "allow_multiple": True,
            "resolved": False,
            "ts": 12.0,
        },
        {
            "type": "ask_user",
            "tool_call_id": "ask-late-resolution",
            "question": "",
            "context": "",
            "options": [],
            "allow_freeform": True,
            "allow_multiple": False,
            "resolved": True,
            "answer": ["Sidebar"],
            "ts": 18.0,
        },
    )

    assert event == {
        "type": "ask_user",
        "tool_call_id": "ask-late-resolution",
        "question": "Where should the todo live?",
        "context": "Need a placement decision.",
        "options": [
            {"title": "Sidebar", "description": ""},
            {"title": "Details", "description": "Main pane"},
        ],
        "allow_freeform": False,
        "allow_multiple": True,
        "resolved": True,
        "answer": ["Sidebar"],
        "ts": 18.0,
    }


def test_app_js_renders_structured_ask_user_options_and_clears_stale_pending_history() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'class: "askUserFreeformInput"' in source
    assert 'class: "askUserSubmit"' in source
    assert '"askUserOption is-multiple"' in source
    assert "buildAskUserSubmissionPayload(ev, draft)" in source
    assert 'class: "askUserOptionTitle"' in source
    assert 'class: "askUserOptionDescription"' in source
    assert "clearResolvedAskUserPendingResponses(msgs);" in source
    assert "mergeAskUserEventUpdate(existingEv, ev)" in source


def test_app_css_includes_ask_user_card_styles() -> None:
    source = APP_CSS.read_text(encoding="utf-8")

    assert ".askUserCard" in source
    assert ".askUserOption" in source
    assert ".askUserOptionTitle" in source
    assert ".askUserOptionDescription" in source
    assert ".askUserComposer" in source
    assert ".askUserFreeformInput" in source
    assert ".askUserSubmit" in source
    assert ".askUserAnswer" in source
