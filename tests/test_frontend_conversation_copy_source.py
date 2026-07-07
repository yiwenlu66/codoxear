import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CONVERSATION_COPY_JS = ROOT / "codoxear" / "static" / "app_conversation_copy.js"
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
        process.stdout.write(JSON.stringify({{
          text: helpers.formatConversationForCopy({json.dumps(events)}),
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


def run_app_conversation_copy_guard(setup_js: str = "") -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const codoxearConversationCopy = window.CodoxearConversationCopy;")
    end = source.index("function normalizeAgentBackendName", start)
    guard_source = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        try {{
          vm.runInContext({json.dumps(setup_js + "\n" + guard_source)}, ctx);
          process.stdout.write(JSON.stringify({{ ok: true, message: "" }}));
        }} catch (err) {{
          process.stdout.write(JSON.stringify({{ ok: false, message: String(err && err.message || err) }}));
        }}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_app_copy_failure_toasts() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const codoxearConversationCopy = window.CodoxearConversationCopy;")
    end = source.index("function normalizeAgentBackendName", start)
    guard_source = source[start:end]
    setup = """
    window.CodoxearConversationCopy = {
      formatConversationForCopy() {},
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
        vm.runInContext({json.dumps(setup + "\n" + guard_source)}, ctx);
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


class TestFrontendConversationCopySource(unittest.TestCase):
    def test_index_loads_conversation_copy_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_polling.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_conversation_copy_helper_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearConversationCopy = window.CodoxearConversationCopy;", source)
        self.assertIn('throw new Error("Codoxear conversation-copy helpers failed to load")', source)
        self.assertIn('typeof codoxearConversationCopy.formatConversationForCopy !== "function"', source)
        self.assertIn('typeof codoxearConversationCopy.transcriptExportTooLargeCopyMessage !== "function"', source)
        self.assertIn("function formatConversationForCopy(events)", source)
        self.assertIn("return codoxearConversationCopy.formatConversationForCopy(events);", source)
        self.assertIn("function copyConversationFailureToast(err)", source)
        self.assertIn("setToast(copyConversationFailureToast(err));", source)
        self.assertIn("window.CodoxearConversationCopy = Object.freeze({", helper_source)
        self.assertIn("transcriptExportTooLargeCopyMessage,", helper_source)
        self.assertIn('parts.push(`## ${role}${when}\\n\\n${text}`);', helper_source)
        self.assertNotIn('parts.push(`## ${role}${when}\\n\\n${text}`);', source)

    def test_app_conversation_copy_guard_throws_for_missing_or_partial_helper(self) -> None:
        missing = run_app_conversation_copy_guard()
        self.assertEqual(missing, {"ok": False, "message": "Codoxear conversation-copy helpers failed to load"})
        partial = run_app_conversation_copy_guard("window.CodoxearConversationCopy = {};")
        self.assertEqual(partial, {"ok": False, "message": "Codoxear conversation-copy helpers failed to load"})
        formatting_only = run_app_conversation_copy_guard("window.CodoxearConversationCopy = { formatConversationForCopy() {} };")
        self.assertEqual(formatting_only, {"ok": False, "message": "Codoxear conversation-copy helpers failed to load"})
        complete = run_app_conversation_copy_guard("window.CodoxearConversationCopy = { formatConversationForCopy() {}, transcriptExportTooLargeCopyMessage() {} };")
        self.assertEqual(complete, {"ok": True, "message": ""})

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
        self.assertRegex(result["known"], r"Conversation.*too large.*copy")
        self.assertIn("50 MiB", result["known"])
        self.assertNotIn("copy failed", result["known"].lower())
        self.assertRegex(result["tagged"], r"Conversation.*too large.*copy")
        self.assertEqual(result["unrelated413"], "")
        self.assertEqual(result["missingLimit"], "")
        self.assertEqual(result["network"], "")

    def test_app_copy_conversation_failure_toast_preserves_generic_failures(self) -> None:
        result = eval_app_copy_failure_toasts()
        self.assertRegex(result["specific"], r"Conversation.*too large.*copy")
        self.assertNotIn("copy failed", result["specific"].lower())
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
        self.assertRegex(
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


if __name__ == "__main__":
    unittest.main()
