"""Undelivered-send recovery in the message-flow controller.

A confirmed send echoes optimistically; if the server later publishes an
`undelivered` verdict for that exact send_id, the echo is retracted and the
text goes back to the composer through `restoreUndeliveredDraft`.
"""
from frontend_module_loader import module_path
import json
import re
import subprocess
import tempfile
import textwrap

SOURCES = [
    module_path(name).read_text(encoding="utf-8")
    for name in (
        "app_polling.js",
        "app_transcript.js",
        "app_message_flow.js",
        "app_session_catalog.js",
        "app_session_state.js",
    )
]

TOAST_RESTORED = "prompt did not reach the backend — restored to draft"
TOAST_RECALL = "prompt did not reach the backend — recall it with ↑ on an empty composer"
SEND_ID_SHAPE = re.compile(r"^s[0-9]+-[0-9a-z]+$")


def run_flow(body: str) -> dict:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console, Date, URL, encodeURIComponent }};
        vm.createContext(ctx);
        {''.join(f'vm.runInContext({json.dumps(source)}, ctx);' for source in SOURCES)}
        const noop = () => {{}};
        function harness(overrides = {{}}) {{
          const log = {{ posts: [], toasts: [], restores: [], dropped: [], removedRows: [], appended: [] }};
          const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
          sessionState.applyRuntime({{ selected: "sid", turnOpen: false }});
          const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
          sessionCatalog.set("latestSessions", [{{ session_id: "sid", agent_backend: "pi" }}]);
          // Every server snapshot replaces the row objects, as a real refresh does.
          const publish = (row) => sessionCatalog.applySnapshot({{ latestSessions: [{{ session_id: "sid", agent_backend: "pi", ...row }}] }});
          const typingRowRuntime = {{ snapshot: () => ({{ stats: {{}} }}), updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop, updateActivityPeek: noop }};
          let echoCounter = 0;
          const options = {{
            sessionState, sessionCatalog,
            currentGeneration: () => 1, isAppDisposed: () => false, sessionLaunchFailed: () => false,
            api: async (path, init) => {{ log.posts.push({{ path, body: init && init.body }}); return {{ ok: true }}; }},
            resolveAppUrl: (path) => path, handleAppAuthLoss: noop,
            refreshSessions: async () => [], openSession: async () => null, clearSelectedSessionAfterRemoval: noop,
            activeTranscriptSnapshot: () => ({{ state: "bound" }}),
            updateSessionTranscriptSlot: () => ({{ ignoredStaleBound: false, current: {{ state: "bound" }} }}),
            renderPendingTranscriptSlot: noop, renderSessionTail: noop, applySessionRuntimeFromTail: noop,
            resetChatRenderState: noop, setAttachCount: noop, setLiveCursor: noop,
            appendEvents: (events) => log.appended.push(...events), appendTailSnapshotEvents: noop,
            initPageLimit: () => 60, typingRowRuntime,
            getStagedAttachments: () => [], hasPendingUploads: () => false, normalizedStagedAttachments: () => [],
            setSelectedSessionPendingAttachment: noop, confirmAction: async () => false,
            setToast: (text) => log.toasts.push(text), isTranscriptRenewalCommand: () => false,
            nextLocalEchoId: () => `local-${{++echoCounter}}`, renderedAtLiveTail: () => true,
            getSessionTranscriptSlot: () => ({{ epoch: 1 }}), addPendingUser: noop, deleteTailCache: noop,
            beginTranscriptRenewal: noop, clearLiveCursor: noop, invalidateOlderLoad: noop,
            dropPendingUser: (sessionId, localId) => log.dropped.push([sessionId, localId]),
            removePendingUserRow: (localId) => log.removedRows.push(localId),
            hasPendingForSession: () => false,
            restoreUndeliveredDraft: (sessionId, text) => {{ log.restores.push([sessionId, text]); return true; }},
            visibilityState: () => "visible", navigatorValue: () => ({{ onLine: true }}),
            EventSource: null, AbortController: null, setTimeout: () => 0, clearTimeout: noop,
            now: () => 1761000000000, consoleWarn: noop, consoleError: noop, ...overrides,
          }};
          const controller = ctx.window.CodoxearMessageFlow.createMessageFlowController(options);
          return {{ log, options, controller, sessionState, sessionCatalog, publish }};
        }}
        const verdict = (sendId, kind = "undelivered") => ({{ send_id: sendId, verdict: kind, ts: 1761000001.5 }});
        const out = (value) => process.stdout.write(JSON.stringify(value));
        {body}
        """
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as script_file:
        script_file.write(script)
        script_file.flush()
        completed = subprocess.run(["node", script_file.name], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def test_idle_plain_send_posts_a_well_formed_send_id() -> None:
    result = run_flow(
        """
        const t = harness();
        // The optimistic echo leaves the session running; settle it so the
        // second send is idle-origin too.
        t.controller.sendText("first").then(() => { t.sessionState.applyRuntime({ running: false }); return t.controller.sendText("second"); }).then(() => out({
          bodies: t.log.posts.map((post) => post.body),
        }));
        """
    )
    first, second = result["bodies"]
    assert first["text"] == "first" and second["text"] == "second"
    for body in (first, second):
        assert SEND_ID_SHAPE.match(body["send_id"]) and len(body["send_id"]) <= 120
        assert re.fullmatch(r"[A-Za-z0-9._:-]+", body["send_id"])
    assert first["send_id"] != second["send_id"]


def test_undelivered_verdict_restores_text_once_and_retracts_the_echo() -> None:
    result = run_flow(
        """
        const t = harness();
        t.controller.sendText("a prompt the backend ate").then(() => {
          const sendId = t.log.posts[0].body.send_id;
          const runningAfterSend = t.sessionState.get("running");
          const turnOpenAfterSend = t.sessionState.get("turnOpen");
          t.publish({ send_verdict: verdict(sendId) });
          const afterFirst = {
            restores: t.log.restores.slice(), dropped: t.log.dropped.slice(), removedRows: t.log.removedRows.slice(),
            running: t.sessionState.get("running"), turnOpen: t.sessionState.get("turnOpen"),
            lastToast: t.log.toasts[t.log.toasts.length - 1],
          };
          const toastsBefore = t.log.toasts.length;
          t.publish({ send_verdict: verdict(sendId), busy: false });
          t.publish({ send_verdict: verdict(sendId) });
          out({
            sendId, runningAfterSend, turnOpenAfterSend, afterFirst,
            restoresAfterRepeats: t.log.restores.length, droppedAfterRepeats: t.log.dropped.length,
            extraToasts: t.log.toasts.length - toastsBefore,
          });
        });
        """
    )
    assert result["runningAfterSend"] is True and result["turnOpenAfterSend"] is True
    first = result["afterFirst"]
    assert first["restores"] == [["sid", "a prompt the backend ate"]]
    assert first["dropped"] == [["sid", "local-1"]]
    assert first["removedRows"] == ["local-1"]
    assert first["running"] is False and first["turnOpen"] is False
    assert first["lastToast"] == TOAST_RESTORED
    assert result["restoresAfterRepeats"] == 1
    assert result["droppedAfterRepeats"] == 1
    assert result["extraToasts"] == 0


def test_declined_restore_points_the_user_at_arrow_up_recall() -> None:
    result = run_flow(
        """
        const t = harness({ restoreUndeliveredDraft: (sessionId, text) => { t.log.restores.push([sessionId, text]); return false; } });
        t.controller.sendText("typed over").then(() => {
          t.publish({ send_verdict: verdict(t.log.posts[0].body.send_id) });
          out({ restores: t.log.restores, lastToast: t.log.toasts[t.log.toasts.length - 1] });
        });
        """
    )
    assert result["restores"] == [["sid", "typed over"]]
    assert result["lastToast"] == TOAST_RECALL


def test_verdict_leaves_busy_state_alone_while_other_echoes_are_pending() -> None:
    result = run_flow(
        """
        const t = harness({ hasPendingForSession: () => true });
        t.controller.sendText("one of two").then(() => {
          t.publish({ send_verdict: verdict(t.log.posts[0].body.send_id) });
          out({ running: t.sessionState.get("running"), restores: t.log.restores.length });
        });
        """
    )
    assert result == {"running": True, "restores": 1}


def test_unrelated_or_malformed_verdicts_never_restore() -> None:
    result = run_flow(
        """
        const t = harness();
        t.controller.sendText("still in flight").then(() => {
          const sendId = t.log.posts[0].body.send_id;
          const noises = [
            { send_verdict: verdict("s999-zzzzzz") },
            { send_verdict: verdict(sendId, "delivered") },
            { send_verdict: null },
            { send_verdict: "undelivered" },
            { send_verdict: [verdict(sendId)] },
            { send_verdict: 42 },
            { send_verdict: {} },
            {},
          ];
          for (const row of noises) t.publish(row);
          out({ restores: t.log.restores.length, dropped: t.log.dropped.length, running: t.sessionState.get("running") });
        });
        """
    )
    assert result == {"restores": 0, "dropped": 0, "running": True}


def test_busy_origin_send_and_control_command_never_register_for_restore() -> None:
    result = run_flow(
        """
        const t = harness();
        t.sessionState.applyRuntime({ running: true });
        t.controller.sendText("steering a live turn")
          .then(() => { t.sessionState.applyRuntime({ running: false }); return t.controller.sendText("/model"); })
          .then(() => {
            for (const post of t.log.posts) {
              t.publish({ send_verdict: verdict(post.body.send_id || "s1-none") });
            }
            out({ bodies: t.log.posts.map((post) => post.body), restores: t.log.restores.length, dropped: t.log.dropped.length });
          });
        """
    )
    assert [("send_id" in body) for body in result["bodies"]] == [False, False]
    assert result["restores"] == 0 and result["dropped"] == 0


def test_transport_error_send_leaves_no_outstanding_record() -> None:
    result = run_flow(
        """
        let sentId = null;
        const t = harness({ api: async (path, init) => { sentId = init.body.send_id; throw new Error("network down"); } });
        t.controller.sendText("composer keeps this").then((sent) => {
          t.publish({ send_verdict: verdict(sentId) });
          out({ sent, sentId, restores: t.log.restores.length, toast: t.log.toasts.includes("send error: network down") });
        });
        """
    )
    assert result["sent"] is False and SEND_ID_SHAPE.match(result["sentId"])
    assert result["restores"] == 0
    assert result["toast"] is True


def test_outstanding_records_are_capped_to_the_newest_thirty_two() -> None:
    result = run_flow(
        """
        const t = harness();
        const run = async () => {
          for (let i = 0; i < 40; i += 1) {
            t.sessionState.applyRuntime({ running: false });
            await t.controller.sendText(`prompt ${i}`);
          }
          const ids = t.log.posts.map((post) => post.body.send_id);
          // Oldest eight were evicted; newest survive.
          for (const id of ids) t.publish({ send_verdict: verdict(id) });
          out({ restored: t.log.restores.map((entry) => entry[1]) });
        };
        run();
        """
    )
    assert result["restored"] == [f"prompt {i}" for i in range(8, 40)]


def test_records_older_than_fifteen_minutes_are_pruned_on_next_send() -> None:
    result = run_flow(
        """
        let clock = 1761000000000;
        const t = harness({ now: () => clock });
        t.controller.sendText("stale").then(() => {
          const staleId = t.log.posts[0].body.send_id;
          clock += 16 * 60 * 1000;
          t.sessionState.applyRuntime({ running: false });
          return t.controller.sendText("fresh").then(() => {
            t.publish({ send_verdict: verdict(staleId) });
            out({ restored: t.log.restores.map((entry) => entry[1]) });
          });
        });
        """
    )
    assert result["restored"] == []


def test_stop_unsubscribes_from_the_catalog() -> None:
    result = run_flow(
        """
        const t = harness();
        t.controller.sendText("late verdict").then(() => {
          const sendId = t.log.posts[0].body.send_id;
          t.controller.stop();
          t.publish({ send_verdict: verdict(sendId) });
          out({ restores: t.log.restores.length });
        });
        """
    )
    assert result == {"restores": 0}
