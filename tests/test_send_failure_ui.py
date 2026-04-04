import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_send_failure_rollback(*, prior_running: bool, prior_turn_open: bool) -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("async function sendText(raw, { sid = null } = {}) {")
    end = source.index("form.onsubmit = async (e) => {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_sendText = sendText;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const pendingEl = {{
          style: {{ opacity: "0.72" }},
          attrs: {{}},
          removeAttribute(name) {{ delete this.attrs[name]; }},
          querySelector() {{ return null; }},
          closest() {{ return null; }},
        }};
        const ctx = {{
          selected: "sess-1",
          sending: false,
          localEchoSeq: 0,
          turnOpen: {json.dumps(prior_turn_open)},
          currentRunning: {json.dumps(prior_running)},
          pendingUser: [],
          pollFastUntilMs: 0,
          toastMessages: [],
          sendBtn: {{ disabled: false }},
          $: (selector) => selector === "#sendBtn" ? ctx.sendBtn : null,
          setToast: (msg) => {{ ctx.toastMessages.push(msg); }},
          pendingMatchKey: (value) => String(value || "").trim(),
          normalizeTextForPendingMatch: (value) => String(value || ""),
          appendEvent: (ev) => {{
            if (!ev || !ev.pending) return;
            pendingEl.attrs["data-pending"] = "1";
            pendingEl.attrs["data-local-id"] = String(ev.localId);
            ctx.lastPendingLocalId = ev.localId;
          }},
          chatInner: {{
            querySelector: (selector) => selector.includes(String(ctx.lastPendingLocalId || "")) ? pendingEl : null,
          }},
          api: async () => {{ throw new Error("prompt rejected"); }},
          setAttachCount: () => {{}},
          kickPoll: () => {{}},
          refreshSessions: async () => {{}},
          Date,
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        (async () => {{
          await ctx.__test_sendText("hello pi");
          process.stdout.write(JSON.stringify({{
            pendingUserLength: ctx.pendingUser.length,
            currentRunning: ctx.currentRunning,
            turnOpen: ctx.turnOpen,
            bubblePending: Object.prototype.hasOwnProperty.call(pendingEl.attrs, "data-pending"),
            bubbleLocalId: Object.prototype.hasOwnProperty.call(pendingEl.attrs, "data-local-id"),
            bubbleOpacity: pendingEl.style.opacity,
            bubbleBorderColor: pendingEl.style.borderColor || "",
            toast: ctx.toastMessages[ctx.toastMessages.length - 1] || "",
            sendDisabled: ctx.sendBtn.disabled,
            sending: ctx.sending,
          }}));
        }})().catch((err) => {{
          console.error(err && err.stack ? err.stack : String(err));
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


def eval_send_refresh_failure_no_rollback() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("async function sendText(raw, { sid = null } = {}) {")
    end = source.index("form.onsubmit = async (e) => {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_sendText = sendText;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const pendingEl = {{
          style: {{ opacity: "0.72" }},
          attrs: {{}},
          removeAttribute(name) {{ delete this.attrs[name]; }},
          querySelector() {{ return null; }},
          closest() {{ return null; }},
        }};
        const ctx = {{
          selected: "sess-1",
          sending: false,
          localEchoSeq: 0,
          turnOpen: false,
          currentRunning: false,
          pendingUser: [],
          pollFastUntilMs: 0,
          toastMessages: [],
          sendBtn: {{ disabled: false }},
          refreshCalls: 0,
          $: (selector) => selector === "#sendBtn" ? ctx.sendBtn : null,
          setToast: (msg) => {{ ctx.toastMessages.push(msg); }},
          pendingMatchKey: (value) => String(value || "").trim(),
          normalizeTextForPendingMatch: (value) => String(value || ""),
          appendEvent: (ev) => {{
            if (!ev || !ev.pending) return;
            pendingEl.attrs["data-pending"] = "1";
            pendingEl.attrs["data-local-id"] = String(ev.localId);
            ctx.lastPendingLocalId = ev.localId;
          }},
          chatInner: {{
            querySelector: (selector) => selector.includes(String(ctx.lastPendingLocalId || "")) ? pendingEl : null,
          }},
          api: async () => ({{ ok: true, queued: false }}),
          setAttachCount: (count) => {{ ctx.attachCount = count; }},
          kickPoll: () => {{ ctx.kickPollCalled = true; }},
          refreshSessions: async () => {{
            ctx.refreshCalls += 1;
            throw new Error("session refresh exploded");
          }},
          console: {{ error: (...args) => {{ ctx.consoleErrors = (ctx.consoleErrors || []).concat([args.map(String).join(" ")]); }} }},
          Date,
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        (async () => {{
          const sent = await ctx.__test_sendText("hello pi");
          process.stdout.write(JSON.stringify({{
            sent,
            pendingUserLength: ctx.pendingUser.length,
            currentRunning: ctx.currentRunning,
            turnOpen: ctx.turnOpen,
            bubblePending: Object.prototype.hasOwnProperty.call(pendingEl.attrs, "data-pending"),
            bubbleLocalId: Object.prototype.hasOwnProperty.call(pendingEl.attrs, "data-local-id"),
            bubbleBorderColor: pendingEl.style.borderColor || "",
            toast: ctx.toastMessages[ctx.toastMessages.length - 1] || "",
            kickPollCalled: !!ctx.kickPollCalled,
            attachCount: ctx.attachCount,
            refreshCalls: ctx.refreshCalls,
            sendDisabled: ctx.sendBtn.disabled,
            sending: ctx.sending,
          }}));
        }})().catch((err) => {{
          console.error(err && err.stack ? err.stack : String(err));
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


def eval_submit_restore_after_send_failure() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function clearComposer() {")
    end = source.index("\n\t        (async () => {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_sendText = sendText;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const msg = {{ value: "hello pi" }};
        const form = {{ requestSubmit() {{}} }};
        const ctx = {{
          selected: "sess-1",
          sending: false,
          localEchoSeq: 0,
          turnOpen: false,
          currentRunning: false,
          pendingUser: [],
          pollFastUntilMs: 0,
          toastMessages: [],
          sendBtn: {{ disabled: false }},
          msg,
          form,
          chatInner: {{ querySelector: () => null }},
          $: (selector) => {{
            if (selector === "#msg") return msg;
            if (selector === "#sendBtn") return ctx.sendBtn;
            return null;
          }},
          autoGrow: () => {{}},
          setToast: (msg) => {{ ctx.toastMessages.push(msg); }},
          pendingMatchKey: (value) => String(value || "").trim(),
          normalizeTextForPendingMatch: (value) => String(value || ""),
          appendEvent: () => {{}},
          api: async () => {{ throw new Error("prompt rejected"); }},
          setAttachCount: () => {{}},
          kickPoll: () => {{}},
          refreshSessions: async () => {{}},
          showSendChoice: () => {{ throw new Error("should not queue"); }},
          Date,
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        (async () => {{
          await ctx.form.onsubmit({{ preventDefault() {{}} }});
          process.stdout.write(JSON.stringify({{
            msgValue: ctx.msg.value,
            toast: ctx.toastMessages[ctx.toastMessages.length - 1] || "",
            sendDisabled: ctx.sendBtn.disabled,
            sending: ctx.sending,
          }}));
        }})().catch((err) => {{
          console.error(err && err.stack ? err.stack : String(err));
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


def eval_submit_no_restore_after_refresh_failure() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function clearComposer() {")
    end = source.index("\n\t        (async () => {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_sendText = sendText;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const msg = {{ value: "hello pi" }};
        const form = {{ requestSubmit() {{}} }};
        const ctx = {{
          selected: "sess-1",
          sending: false,
          localEchoSeq: 0,
          turnOpen: false,
          currentRunning: false,
          pendingUser: [],
          pollFastUntilMs: 0,
          toastMessages: [],
          sendBtn: {{ disabled: false }},
          msg,
          form,
          $: (selector) => {{
            if (selector === "#msg") return msg;
            if (selector === "#sendBtn") return ctx.sendBtn;
            return null;
          }},
          autoGrow: () => {{ ctx.autoGrowCalls = (ctx.autoGrowCalls || 0) + 1; }},
          setToast: (msg) => {{ ctx.toastMessages.push(msg); }},
          pendingMatchKey: (value) => String(value || "").trim(),
          normalizeTextForPendingMatch: (value) => String(value || ""),
          appendEvent: () => {{}},
          chatInner: {{ querySelector: () => null }},
          api: async () => ({{ ok: true, queued: false }}),
          setAttachCount: () => {{}},
          kickPoll: () => {{}},
          refreshSessions: async () => {{ throw new Error("session refresh exploded"); }},
          showSendChoice: () => {{ throw new Error("should not queue"); }},
          console: {{ error: () => {{}} }},
          Date,
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        (async () => {{
          await ctx.form.onsubmit({{ preventDefault() {{}} }});
          process.stdout.write(JSON.stringify({{
            msgValue: ctx.msg.value,
            autoGrowCalls: ctx.autoGrowCalls || 0,
            toast: ctx.toastMessages[ctx.toastMessages.length - 1] || "",
            sendDisabled: ctx.sendBtn.disabled,
            sending: ctx.sending,
          }}));
        }})().catch((err) => {{
          console.error(err && err.stack ? err.stack : String(err));
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


class TestSendFailureUi(unittest.TestCase):
    def test_send_failure_rolls_back_pending_state_for_idle_session(self) -> None:
        result = eval_send_failure_rollback(prior_running=False, prior_turn_open=False)

        self.assertEqual(result["pendingUserLength"], 0)
        self.assertFalse(result["currentRunning"])
        self.assertFalse(result["turnOpen"])
        self.assertFalse(result["bubblePending"])
        self.assertFalse(result["bubbleLocalId"])
        self.assertEqual(result["bubbleOpacity"], "1")
        self.assertIn("185, 28, 28", result["bubbleBorderColor"])
        self.assertEqual(result["toast"], "send error: prompt rejected")
        self.assertFalse(result["sendDisabled"])
        self.assertFalse(result["sending"])

    def test_send_failure_restores_prior_running_state_when_session_was_busy(self) -> None:
        result = eval_send_failure_rollback(prior_running=True, prior_turn_open=True)

        self.assertEqual(result["pendingUserLength"], 0)
        self.assertTrue(result["currentRunning"])
        self.assertTrue(result["turnOpen"])
        self.assertFalse(result["bubblePending"])
        self.assertFalse(result["bubbleLocalId"])

    def test_submit_restores_draft_when_send_fails_after_clear(self) -> None:
        result = eval_submit_restore_after_send_failure()

        self.assertEqual(result["msgValue"], "hello pi")
        self.assertEqual(result["toast"], "send error: prompt rejected")
        self.assertFalse(result["sendDisabled"])
        self.assertFalse(result["sending"])

    def test_refresh_failure_after_accepted_send_keeps_pending_state(self) -> None:
        result = eval_send_refresh_failure_no_rollback()

        self.assertTrue(result["sent"])
        self.assertEqual(result["pendingUserLength"], 1)
        self.assertTrue(result["currentRunning"])
        self.assertTrue(result["turnOpen"])
        self.assertTrue(result["bubblePending"])
        self.assertTrue(result["bubbleLocalId"])
        self.assertEqual(result["bubbleBorderColor"], "")
        self.assertEqual(result["toast"], "sent")
        self.assertTrue(result["kickPollCalled"])
        self.assertEqual(result["attachCount"], 0)
        self.assertEqual(result["refreshCalls"], 1)
        self.assertFalse(result["sendDisabled"])
        self.assertFalse(result["sending"])

    def test_submit_does_not_restore_draft_when_only_refresh_fails(self) -> None:
        result = eval_submit_no_restore_after_refresh_failure()

        self.assertEqual(result["msgValue"], "")
        self.assertEqual(result["toast"], "sent")
        self.assertEqual(result["autoGrowCalls"], 1)
        self.assertFalse(result["sendDisabled"])
        self.assertFalse(result["sending"])


if __name__ == "__main__":
    unittest.main()
