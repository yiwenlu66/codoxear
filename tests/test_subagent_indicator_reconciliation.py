from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

from codoxear.agent_backend import get_agent_backend
from codoxear.cc_subagents import handle_hook_event
from codoxear.rollout_chat_batch import _extract_chat_events
from codoxear.session_listing import build_active_session_rows_snapshot
from codoxear.session_model import Session
from codoxear.session_store import SessionStore, SessionStorePaths
from codoxear.util import scan_active_cc_subagents


ROOT = Path(__file__).resolve().parents[1]
APP_SESSIONS_JS = ROOT / "codoxear" / "static" / "app_sessions.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_FLOW_JS = ROOT / "codoxear" / "static" / "app_message_flow.js"
CC_SESSION_ID = "11111111-2222-3333-4444-555555555555"


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=tmp_path / "aliases.json",
            sidebar_meta=tmp_path / "sidebar.json",
            hidden_sessions=tmp_path / "hidden.json",
            files=tmp_path / "files.json",
            queues=tmp_path / "queues.json",
            pending_attachments=tmp_path / "pending.json",
            commit_unknown_sends=tmp_path / "unknown.json",
            recent_cwds=tmp_path / "recent.json",
            unattended=tmp_path / "unattended.json",
        ),
        file_history_max=5,
        recent_cwd_max=5,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda value: value if isinstance(value, str) else "",
        clean_priority_offset=lambda value: float(value or 0.0),
        clean_snooze_until=lambda value: float(value) if value not in (None, "", 0) else None,
        clean_dependency_session_id=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_recent_cwd=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_surface_projection(rows: list[dict]) -> dict:
    sessions_source = APP_SESSIONS_JS.read_text(encoding="utf-8")
    transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
    message_flow_source = APP_MESSAGE_FLOW_JS.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          window: {{
            CodoxearPolling: {{
              messagePollDelayMs: () => 1000,
              normalizeMessagePollKickDelay: ({{ requested }}) => requested,
              browserOffline: () => false,
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(transcript_source)}, ctx);
        vm.runInContext({json.dumps(message_flow_source)}, ctx);
        vm.runInContext({json.dumps(sessions_source)}, ctx);

        function node(attrs = {{}}, children = []) {{
          const out = {{ ...attrs, children: [], dataset: {{}}, style: {{}}, isConnected: false, parentNode: null, parentElement: null }};
          out.classList = {{ add: (name) => {{ out.class = `${{out.class || ""}} ${{name}}`.trim(); }}, remove: () => {{}} }};
          out.appendChild = (child) => {{
            if (!child) return child;
            out.children.push(child);
            child.parentNode = out;
            child.parentElement = out;
            return child;
          }};
          out.insertBefore = (child, before) => {{
            if (!child) return child;
            const index = out.children.indexOf(before);
            if (index < 0) return out.appendChild(child);
            out.children.splice(index, 0, child);
            child.parentNode = out;
            child.parentElement = out;
            child.isConnected = true;
            return child;
          }};
          out.remove = () => {{
            out.isConnected = false;
            if (out.parentNode) out.parentNode.children = out.parentNode.children.filter((child) => child !== out);
            out.parentNode = null;
            out.parentElement = null;
          }};
          out.addEventListener = () => {{}};
          Object.defineProperty(out, "childElementCount", {{ get: () => out.children.length }});
          Object.defineProperty(out, "innerHTML", {{ set: () => {{ out.children = []; }} }});
          for (const child of children) out.appendChild(child);
          return out;
        }}

        const rows = {json.dumps(rows)};
        const wrap = node();
        const empty = node();
        const sidebar = ctx.window.CodoxearSessions.createSessionsController({{
          sessionsWrap: wrap,
          sidebarEmptyHint: empty,
          el: (_tag, attrs, children) => node(attrs, children),
          iconSvg: () => "",
          sidebarRenderSignature: () => "reconciliation",
          sidebarSessionEntries: (sessions) => sessions.map((session) => ({{ type: "session", session }})),
          sessionDisplayName: (session) => session.session_id,
          sessionLaunchFailed: () => false,
          sessionLaunchPending: () => false,
          redactedLaunchErrorText: () => "",
          fmtRelativeAge: () => "now",
          sidebarEffortCode: () => "",
          sidebarModelText: () => "",
          baseName: () => "repo",
          sessionIsFast: () => false,
          agentBackendLogoPath: () => "",
          agentBackendDisplayName: () => "backend",
          sessionAgentBackend: (session) => session.agent_backend,
          sessionLaunchIcon: () => "",
          sessionLaunchLabel: () => "",
          confirmAction: async () => false,
          api: async () => ({{}}),
          clearDeletedSessionClientState: () => {{}},
          refreshSessions: async () => [],
          setToast: () => {{}},
          openEditSession: () => {{}},
          duplicateSession: async () => {{}},
          selectSession: async () => {{}},
          setSidebarOpen: () => {{}},
          now: () => 1000,
          performanceNow: () => 0,
        }});
        sidebar.renderSessions(rows, {{ selectedId: "", swipeActions: false }});

        function findText(current, className, found = []) {{
          if (current && current.class === className) found.push(current.text);
          for (const child of (current && current.children) || []) findText(child, className, found);
          return found;
        }}

        const chipUpdates = [];
        const gaugeUpdates = [];
        const quietRows = [];
        for (const row of rows) {{
          const root = node();
          const bottom = node();
          const concreteTypingRuntime = ctx.window.CodoxearTranscript.createTypingRowRuntime({{
            root,
            bottomSentinel: bottom,
            el: (_tag, attrs, children) => node(attrs, children),
            shouldAutoScroll: () => false,
            scheduleScrollToBottom: () => {{}},
          }});
          const typingRowRuntime = {{
            snapshot: concreteTypingRuntime.snapshot,
            updateTypingStats: concreteTypingRuntime.updateTypingStats,
            resetTypingStats: concreteTypingRuntime.resetTypingStats,
            updateSubagentGauge: (count) => {{
              gaugeUpdates.push(count);
              return concreteTypingRuntime.updateSubagentGauge(count);
            }},
          }};
          const noop = () => {{}};
          const flow = ctx.window.CodoxearMessageFlow.createMessageFlowController({{
            getSelected: () => "sid", getGeneration: () => 1, isAppDisposed: () => false,
            getTurnOpen: () => false, setTurnOpen: noop,
            getSessionInfo: () => ({{ agent_backend: row.agent_backend }}), patchSessionInfo: noop,
            sessionLaunchFailed: () => false, api: async () => ({{}}), resolveAppUrl: (path) => path,
            handleAppAuthLoss: noop, refreshSessions: async () => [], openSession: async () => null,
            clearSelectedSessionAfterRemoval: noop,
            activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "cursor", logPath: "/tmp/log" }}),
            updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
            renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
            resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop,
            appendEvent: noop, appendTailSnapshotEvents: noop, setStatus: noop, setContext: noop, setTyping: noop,
            setSubagentsRunning: (count) => chipUpdates.push(`Idle · ▸${{count}}`),
            updateSessionTitle: noop, initPageLimit: () => 60, typingRowRuntime,
            getSending: () => false, setSending: noop, getCurrentRunning: () => false, setCurrentRunning: noop,
            getStagedAttachments: () => [], normalizedStagedAttachments: () => [], setSelectedSessionPendingAttachment: noop,
            syncSendButtonState: noop, syncAttachButtonState: noop, syncQueueSubmitState: noop, syncRecoveryUiForSession: noop,
            confirmAction: async () => false, setToast: noop, isTranscriptRenewalCommand: () => false,
            nextLocalEchoId: () => 1, renderedAtLiveTail: () => true, clearTranscriptDom: noop,
            clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({{ epoch: 0 }}),
            addPendingUser: noop, deleteTailCache: noop, beginTranscriptRenewal: noop, clearLiveCursor: noop,
            invalidateOlderLoad: noop, dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
            visibilityState: () => "hidden", navigatorValue: () => ({{ onLine: true }}), EventSource: null,
            AbortController: null, setTimeout: () => 0, clearTimeout: noop, now: () => 0,
            consoleWarn: noop, consoleError: noop,
          }});
          flow.updateTypingStatsFromSession(row);
          concreteTypingRuntime.setSubagentVisible(true);
          quietRows.push(root.children[0].children[0].children[1].textContent);
        }}
        process.stdout.write(JSON.stringify({{
          sidebarMarkers: findText(wrap, "muted subagentMarker"),
          idleChips: chipUpdates,
          quietRows,
          gaugeUpdates,
        }}));
        """
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as file:
        file.write(script)
        file.flush()
        proc = subprocess.run(["node", file.name], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_subagent_indicator_reconciles_native_sources_to_every_surface(tmp_path: Path, monkeypatch) -> None:
    pi_root = tmp_path / "pi-runs"
    cc_root = tmp_path / "cc-runs"
    sessions_root = tmp_path / "codex" / "sessions"
    pi_log = tmp_path / "pi-parent.jsonl"
    codex_parent_log = sessions_root / "2026" / "08" / "05" / "rollout-parent.jsonl"
    codex_child_log = sessions_root / "2026" / "08" / "05" / "rollout-child.jsonl"
    cc_log = tmp_path / "cc-parent.jsonl"
    for path in (pi_log, codex_parent_log, cc_log):
        path.parent.mkdir(parents=True, exist_ok=True)

    pi_notice = {
        "type": "custom_message",
        "customType": "subagent_control_notice",
        "id": "pi-progress-1",
        "timestamp": "2026-08-05T00:00:00Z",
        "content": "Subagent progress update: executor\nRun: pi-live\nUPDATE: inspecting",
    }
    pi_log.write_text(json.dumps(pi_notice) + "\n", encoding="utf-8")
    pi_status = pi_root / "pi-live" / "status.json"
    pi_status.parent.mkdir(parents=True)
    pi_status.write_text(
        json.dumps({"runId": "pi-live", "sessionId": str(pi_log), "state": "running", "startedAt": 1, "pid": os.getpid(), "agent": "executor"}),
        encoding="utf-8",
    )

    codex_header = {
        "type": "session_meta",
        "timestamp": "2026-08-05T00:00:00Z",
        "payload": {"id": "codex-child", "source": {"subagent": {"thread_spawn": {"parent_thread_id": "codex-parent"}}}},
    }
    codex_parent_log.write_text(json.dumps({"type": "session_meta", "payload": {"id": "codex-parent"}}) + "\n", encoding="utf-8")
    codex_child_log.write_text(json.dumps(codex_header) + "\n", encoding="utf-8")
    cc_log.write_text(json.dumps({"type": "user", "sessionId": CC_SESSION_ID, "message": {"content": "hello"}}) + "\n", encoding="utf-8")

    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(pi_root))
    monkeypatch.setenv("CODEX_WEB_CC_SUBAGENT_RUNS_ROOT", str(cc_root))
    assert handle_hook_event(
        {"hook_event_name": "SubagentStart", "session_id": CC_SESSION_ID, "agent_id": "cc-child", "agent_type": "Explore"},
        environ={
            "CODEX_WEB_OWNER": "web",
            "CODEX_WEB_AGENT_BACKEND": "cc",
            "CODEX_WEB_CC_SUBAGENT_RUNS_ROOT": str(cc_root),
            "CODEX_WEB_CC_SUBAGENT_BROKER_PID": str(os.getpid()),
        },
    )

    pi_events, _meta, _flags, _diagnostics = _extract_chat_events(_read_jsonl(pi_log))
    codex_event = get_agent_backend("codex").chat_event_from_log_row(_read_jsonl(codex_child_log)[0])
    cc_runs = scan_active_cc_subagents(parent_broker_pids={CC_SESSION_ID: os.getpid()})
    assert [event["message_id"] for event in pi_events] == ["pi-subagent:pi-progress-1"]
    assert codex_event is not None and codex_event["message_id"] == "codex-subagent:codex-child"
    assert cc_runs[CC_SESSION_ID][0]["event"]["message_id"] == "cc-subagent:cc-child"

    sessions = [
        Session("pi", "pi-parent", os.getpid(), 1, "pi", True, 1.0, "/repo", pi_log, tmp_path / "pi.sock"),
        Session("codex", "codex-parent", os.getpid(), 1, "codex", True, 1.0, "/repo", codex_parent_log, tmp_path / "codex.sock"),
        Session("cc", CC_SESSION_ID, os.getpid(), 1, "cc", True, 1.0, "/repo", cc_log, tmp_path / "cc.sock"),
    ]
    snapshot = build_active_session_rows_snapshot(
        sessions=sessions,
        queues={},
        unattended={},
        aliases={},
        store=_store(tmp_path),
        now_ts=10.0,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, *, allow_zero=False: int(value),
        provider_choice_for_settings=lambda **_kwargs: "",
        resolve_session_cwd=lambda _cwd: Path("/repo"),
        priority_half_life_seconds=100.0,
        priority_bucket_seconds=10.0,
    )
    rows = sorted(snapshot.rows, key=lambda row: row["agent_backend"])
    assert [(row["agent_backend"], row["subagents_running"]) for row in rows] == [("cc", 1), ("codex", 1), ("pi", 1)]

    surfaces = _run_surface_projection([{key: value for key, value in row.items() if not key.startswith("_")} for row in rows])
    assert surfaces == {
        "sidebarMarkers": ["▸1", "▸1", "▸1"],
        "idleChips": ["Idle · ▸1", "Idle · ▸1", "Idle · ▸1"],
        "quietRows": ["▸1 subagent working", "▸1 subagent working", "▸1 subagent working"],
        "gaugeUpdates": [1, 1, 1],
    }
