from frontend_module_loader import module_path
import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE_OPS_JS = module_path("app_file_ops.js")
APP_CONVERSATION_COPY_JS = module_path("app_conversation_copy.js")


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


class TestFrontendConversationCopyBehavior(unittest.TestCase):
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
