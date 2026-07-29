import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def eval_attach_button_state() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    def function_block(name: str) -> str:
        start = source.index(f"function {name}")
        end = source.index("\n      }", start) + len("\n      }")
        return source[start:end]

    predicates = "\n".join(
        function_block(name)
        for name in [
            "sessionLaunchFailed(s)",
            "sessionHasUnknownSend(s)",
            "sessionIsOrphanRecovery(s)",
            "sessionHasOrphanQueueRecovery(s)",
        ]
    )
    blocker_start = source.index("function attachmentBlockerForSession(sessionId, sessionInfo = null) {")
    blocker_end = source.index("setAttachCount(0);", blocker_start)
    snippet = predicates + "\n" + source[blocker_start:blocker_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const button = {{ disabled: null, title: "", attrs: {{}}, setAttribute(name, value) {{ this.attrs[name] = value; }} }};
        const sessions = new Map();
        const ctx = {{
          selected: null,
          sessionIndex: sessions,
          currentRunning: false,
          sending: false,
          ATTACH_UPLOAD_MAX_BYTES: 1048576,
          fmtBytes: (n) => `${{n / 1048576}} MB`,
          $: (selector) => selector === "#attachBtn" ? button : null,
          codoxearSessionHelpers: {{
            sessionLaunchFailed: (s) => Boolean(s && s.launch_state === "failed"),
            sessionHasUnknownSend: (s) => Boolean(s && s.commit_unknown_send),
            sessionIsOrphanRecovery: (s) => Boolean(s && s.orphan_recovery),
            sessionHasOrphanQueueRecovery: (s) => Boolean(s && s.queue_recovery),
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__attach = { attachmentBlockerForSession, syncAttachButtonState };\n")}, ctx);
        const states = [];
        function record(label, info = null) {{
          if (info) sessions.set("sid", info); else sessions.delete("sid");
          ctx.selected = label === "none" ? null : "sid";
          ctx.__attach.syncAttachButtonState();
          states.push({{ label, disabled: button.disabled, title: button.title, aria: button.attrs["aria-label"] }});
        }}
        record("none");
        record("idle", {{ launch_state: "ready" }});
        record("busy", {{ launch_state: "ready", busy: true }});
        record("failed", {{ launch_state: "failed" }});
        record("unknown", {{ launch_state: "ready", commit_unknown_send: true }});
        record("orphan", {{ launch_state: "ready", orphan_recovery: true }});
        record("queue-recovery", {{ launch_state: "ready", queue_recovery: true }});
        process.stdout.write(JSON.stringify({{ states }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestAttachButtonSource(unittest.TestCase):
    def test_attach_button_reflects_session_selection(self) -> None:
        states = {row["label"]: row for row in eval_attach_button_state()["states"]}
        self.assertTrue(states["none"]["disabled"])
        self.assertEqual(states["none"]["title"], "Select a session to attach a file")
        self.assertFalse(states["idle"]["disabled"])
        self.assertEqual(states["idle"]["title"], "Attach file (max 1 MB)")
        for label, title in {
            "busy": "Wait for the current response to finish before attaching a file",
            "failed": "Failed launch cannot receive file attachments",
            "unknown": "Resolve the unknown send before attaching a file",
            "orphan": "Missing session can only be reviewed",
            "queue-recovery": "Review preserved queued recovery items before attaching a file",
        }.items():
            self.assertTrue(states[label]["disabled"])
            self.assertEqual(states[label]["title"], title)
            self.assertEqual(states[label]["aria"], title)

    def test_attach_button_blocks_client_send_in_progress(self) -> None:
        result = eval_attach_button_state()
        self.assertEqual(len(result["states"]), 7)
        self.assertTrue(all(row["aria"] == row["title"] for row in result["states"]))


if __name__ == "__main__":
    unittest.main()
