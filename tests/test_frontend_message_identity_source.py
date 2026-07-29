import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_MESSAGE_IDENTITY_JS = ROOT / "codoxear" / "static" / "app_message_identity.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_message_identity() -> dict:
    source = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const identity = ctx.window.CodoxearMessageIdentity;
        process.stdout.write(JSON.stringify({{
          normalized: identity.normalizeTextForPendingMatch("a\\r\\nb\\rc"),
          pendingKey: identity.pendingMatchKey("hello   \\nworld\\t\\n\\n"),
          eventKey: identity.eventKey({{ role: "user", ts: 1.2345, text: "hello   \\n" }}),
          invalidEventKey: identity.eventKey({{ role: "system", ts: 1, text: "x" }}),
          assistantKey: identity.chatAssistantDedupeKey({{ role: "assistant", message_class: "final_response", text: " same\\nfinal   text " }}),
          nonAssistantKey: identity.chatAssistantDedupeKey({{ role: "user", text: "same" }}),
          frozen: Object.isFrozen(identity),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendMessageIdentitySource(unittest.TestCase):


    def test_message_identity_preserves_keys_and_normalization(self) -> None:
        result = eval_message_identity()
        self.assertEqual(result["normalized"], "a\nb\nc")
        self.assertEqual(result["pendingKey"], "hello\nworld")
        self.assertEqual(result["eventKey"], "user|1235|hello")
        self.assertEqual(result["invalidEventKey"], "")
        self.assertEqual(result["assistantKey"], "final_response|same final text")
        self.assertEqual(result["nonAssistantKey"], "")
        self.assertTrue(result["frozen"])


if __name__ == "__main__":
    unittest.main()
