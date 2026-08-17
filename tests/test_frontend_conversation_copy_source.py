from frontend_module_loader import module_path
import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = module_path("app_application_composition.js")
APP_GUARD_JS = module_path("app_application.js")
APP_FILE_OPS_JS = module_path("app_file_ops.js")
APP_CONVERSATION_COPY_JS = module_path("app_conversation_copy.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_conversation_copy(events) -> dict:
    source = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearConversationCopy;
        const result = helpers.formatConversationForCopyResult({json.dumps(events)});
        process.stdout.write(JSON.stringify({{
          text: helpers.formatConversationForCopy({json.dumps(events)}),
          resultText: result.text,
          messageCount: result.messageCount,
          frozen: Object.isFrozen(helpers),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_conversation_copy_helpers(expression: str) -> dict:
    source = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearConversationCopy;
        process.stdout.write(JSON.stringify({expression}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_app_copy_failure_toasts() -> dict:
    source = APP_GUARD_JS.read_text(encoding="utf-8")
    start = source.index("function transcriptExportTooLargeCopyMessage(err)")
    end = source.index("function normalizeAgentBackendName", start)
    helper_source = source[start:end]
    setup = """
    window.CodoxearConversationCopy = {
      transcriptExportTooLargeCopyMessage(err) {
        if (err && err.status === 413 && err.obj && err.obj.max_bytes === 52428800) return "Conversation too large to copy (max 50 MiB). Use search or copy a smaller range.";
        return "";
      },
    };
    """
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(setup + "\n" + helper_source)}, ctx);
        const exportErr = Object.assign(new Error("transcript log is too large to export"), {{ status: 413, obj: {{ error: "transcript log is too large to export", max_bytes: 52428800 }} }});
        const genericErr = new Error("denied");
        process.stdout.write(JSON.stringify({{
          specific: ctx.copyConversationFailureToast(exportErr),
          generic: ctx.copyConversationFailureToast(genericErr),
          unknown: ctx.copyConversationFailureToast(null),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_app_clipboard_fallback() -> dict:
    source = APP_FILE_OPS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        let primaryCalls = 0;
        const ctx = {{
          navigator: {{ clipboard: {{ writeText() {{ primaryCalls += 1; throw new Error("permission denied"); }} }} }},
          window: {{ isSecureContext: true, navigator: null }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        ctx.window.CodoxearClipboard.copyToClipboard("conversation text")
          .then(() => process.stdout.write(JSON.stringify({{ primaryCalls, threw: false }})))
          .catch((err) => process.stdout.write(JSON.stringify({{ primaryCalls, threw: true, message: err.message || String(err) }})));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_app_copy_conversation_success(events) -> dict:
    app_source = APP_JS.read_text(encoding="utf-8")
    helper_source = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
    start = app_source.index("function formatConversationForCopy(events)")
    end = app_source.index("let fileOpsController = null", start)
    copy_source = app_source[start:end]
    runtime_source = "const codoxearConversationCopy = window.CodoxearConversationCopy;\n" + copy_source + "\nthis.__copyConversation = copyConversation;"
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          window: {{}},
          selected: "session-1",
          toastText: "",
          clipboardText: "",
        }};
        ctx.api = async (url) => {{
          if (url !== "/api/sessions/session-1/messages/export") throw new Error(`unexpected api url: ${{url}}`);
          return {{ events: {json.dumps(events)} }};
        }};
        ctx.copyToClipboard = async (text) => {{ ctx.clipboardText = text; }};
        ctx.setToast = (text) => {{ ctx.toastText = text; }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
        ctx.sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
        ctx.sessionState.set("selected", "session-1");
        vm.runInContext({json.dumps(runtime_source)}, ctx);
        vm.runInContext("__copyConversation()", ctx).then(() => {{
          process.stdout.write(JSON.stringify({{
            toast: ctx.toastText,
            clipboardText: ctx.clipboardText,
          }}));
        }}).catch((err) => {{
          console.error(err && err.stack || err);
          process.exit(1);
        }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendConversationCopySource(unittest.TestCase):
    def test_transcript_export_too_large_helper_recognizes_api_error_shape(self) -> None:
        result = eval_conversation_copy_helpers(
            """
            (() => {
              const known = Object.assign(new Error("transcript log is too large to export (60 bytes > 50 bytes)"), {
                status: 413,
                obj: { error: "transcript log is too large to export (60 bytes > 50 bytes)", max_bytes: 52428800 },
              });
              return {
                known: helpers.transcriptExportTooLargeCopyMessage(known),
                tagged: helpers.transcriptExportTooLargeCopyMessage({ status: 413, obj: { error: "transcript-export-too-large", max_bytes: 1024 } }),
                unrelated413: helpers.transcriptExportTooLargeCopyMessage({ status: 413, obj: { error: "file too large", max_bytes: 52428800 } }),
                missingLimit: helpers.transcriptExportTooLargeCopyMessage({ status: 413, obj: { error: "transcript log is too large to export" } }),
                network: helpers.transcriptExportTooLargeCopyMessage(new Error("network down")),
              };
            })()
            """
        )
        self.assertMatches(result["known"], r"Conversation.*too large.*copy")
        self.assertContains("50 MiB", result["known"])
        self.assertNotContains("copy failed", result["known"].lower())
        self.assertMatches(result["tagged"], r"Conversation.*too large.*copy")
        self.assertEqual(result["unrelated413"], "")
        self.assertEqual(result["missingLimit"], "")
        self.assertEqual(result["network"], "")

    def test_app_copy_uses_secure_clipboard_only(self) -> None:
        result = eval_app_clipboard_fallback()
        self.assertEqual(result["primaryCalls"], 1)
        self.assertTrue(result.get("threw"))

    def test_app_copy_conversation_success_toast_counts_copied_messages_not_raw_events(self) -> None:
        result = eval_app_copy_conversation_success(
            [
                {"role": "system", "text": "ignored system", "ts": 1},
                {"role": "user", "text": "first user", "ts": 2},
                {"role": "assistant", "text": "   ", "ts": 3},
                {"role": "tool", "text": "ignored tool", "ts": 4},
                {"role": "assistant", "text": "assistant answer", "ts": 5},
                {"role": "user", "text": "\n\t", "ts": 6},
            ]
        )
        self.assertEqual(result["toast"], "Copied 2 messages")
        self.assertEqual(result["clipboardText"].count("## User"), 1)
        self.assertEqual(result["clipboardText"].count("## Assistant"), 1)
        self.assertNotContains("ignored system", result["clipboardText"])
        self.assertNotContains("ignored tool", result["clipboardText"])

    def test_app_copy_conversation_success_toast_uses_singular_message_grammar(self) -> None:
        result = eval_app_copy_conversation_success([{"role": "assistant", "text": "one answer"}])
        self.assertEqual(result["toast"], "Copied 1 message")

    def test_app_copy_conversation_failure_toast_preserves_generic_failures(self) -> None:
        result = eval_app_copy_failure_toasts()
        self.assertMatches(result["specific"], r"Conversation.*too large.*copy")
        self.assertNotContains("copy failed", result["specific"].lower())
        self.assertEqual(result["generic"], "copy failed: denied")
        self.assertEqual(result["unknown"], "copy failed: unknown error")

    def test_format_conversation_for_copy_preserves_existing_contract(self) -> None:
        result = eval_conversation_copy(
            [
                {"role": "system", "text": "ignored", "ts": 1},
                {"role": "user", "text": "  hello user  \n\n", "ts": 0},
                {"role": "assistant", "text": "assistant answer\t \n", "ts": "not-a-number"},
                {"role": "assistant", "text": "   ", "ts": 2},
                {"role": "user", "text": None, "ts": 3},
                None,
            ]
        )
        self.assertTrue(result["frozen"])
        self.assertEqual(result["resultText"], result["text"])
        self.assertEqual(result["messageCount"], 2)
        self.assertMatches(
            result["text"],
            re.compile(r"^## User \(.+\)\n\n  hello user\n\n---\n\n## Assistant\n\nassistant answer$", re.DOTALL),
        )

    def test_format_conversation_for_copy_returns_empty_for_no_copyable_text(self) -> None:
        self.assertEqual(eval_conversation_copy({"events": []})["text"], "")
        self.assertEqual(
            eval_conversation_copy([
                {"role": "system", "text": "ignored"},
                {"role": "assistant", "text": "\n\t  "},
                {"role": "user", "text": None},
                {"role": "assistant", "text": 0},
                {"role": "user", "text": False},
            ])["text"],
            "",
        )
        self.assertEqual(
            eval_conversation_copy([
                {"role": "system", "text": "ignored"},
                {"role": "assistant", "text": "\n\t  "},
            ])["messageCount"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
