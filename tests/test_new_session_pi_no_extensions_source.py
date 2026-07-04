"""Regression tests for the Pi no-default-extensions (`-ne`) New Session option.

These tests execute the real launch request construction (`spawnSessionWithCwd`)
inside a Node VM and assert the resulting `/api/sessions` POST body, plus the
DOM wiring / backend-visibility / reset contracts in app.js source.

The server already accepts `args: [str]` (see codoxear/launch_config.py), so no
server change is required; these tests pin the frontend contract.
"""

import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def _run_node(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def _spawn_session_fn_source() -> str:
    """Extract the `spawnSessionWithCwd` function body from app.js source.

    Anchored on the function header and the next sibling statement
    (`$("#newBtn").onclick = async () => {`), which immediately follows it.
    """
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("async function spawnSessionWithCwd(")
    end = source.index('$("#newBtn").onclick = async () => {', start)
    return source[start:end].strip()


def _eval_spawn_body(cwd: str, agent_backend: str, pi_no_extensions: bool) -> dict:
    """Run the extracted spawnSessionWithCwd with mocked deps and return the
    captured POST body passed to `api("/api/sessions", ...)`."""
    fn_source = _spawn_session_fn_source()
    js = textwrap.dedent(
        f"""
        let capturedBody = null;
        let capturedUrl = null;
        function normalizeAgentBackendName(name) {{ return String(name || ""); }}
        function providerChoiceToSettings(provider, backend) {{
          return provider ? {{ model_provider: provider }} : {{}};
        }}
        function backendSupportsFast(backend) {{ return false; }}
        function setToast(_msg) {{}}
        async function refreshSessions() {{ return []; }}
        function selectSession(_sid) {{}}
        async function api(url, opts) {{
          capturedUrl = url;
          capturedBody = opts && opts.body ? JSON.parse(JSON.stringify(opts.body)) : null;
          // Return a pending launch so the function returns immediately after
          // building the body, without entering the broker-pid polling loop.
          return {{ pending: true, launch_id: "L1" }};
        }}
        {fn_source}
        (async () => {{
          await spawnSessionWithCwd(
            {json.dumps(cwd)},
            null,
            null,
            "",
            "openai-codex",
            "gpt-5.4-mini",
            "high",
            false,
            false,
            null,
            {json.dumps(agent_backend)},
            {json.dumps(pi_no_extensions)},
          );
          process.stdout.write(JSON.stringify({{ capturedUrl, capturedBody }}));
        }})();
        """
    )
    return _run_node(js)


class TestSpawnSessionWithCwdArgs(unittest.TestCase):
    def test_pi_no_extensions_sends_ne_args(self) -> None:
        result = _eval_spawn_body("/workspace", "pi", True)
        self.assertEqual(result["capturedUrl"], "/api/sessions")
        body = result["capturedBody"]
        self.assertEqual(body["agent_backend"], "pi")
        self.assertEqual(body["args"], ["-ne"])
        # Provider/model/reasoning/cwd are preserved alongside args.
        self.assertEqual(body["cwd"], "/workspace")
        self.assertEqual(body["model_provider"], "openai-codex")
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertEqual(body["reasoning_effort"], "high")

    def test_pi_extensions_enabled_omits_args(self) -> None:
        result = _eval_spawn_body("/workspace", "pi", False)
        body = result["capturedBody"]
        self.assertEqual(body["agent_backend"], "pi")
        self.assertNotIn("args", body)

    def test_codex_unaffected_even_if_flag_forced(self) -> None:
        # Backend switch must clear -ne: even if a stale true reached the
        # function for a non-pi backend, args must not be emitted.
        result = _eval_spawn_body("/repo", "codex", True)
        body = result["capturedBody"]
        self.assertEqual(body["agent_backend"], "codex")
        self.assertNotIn("args", body)

    def test_cc_unaffected(self) -> None:
        result = _eval_spawn_body("/repo", "cc", True)
        body = result["capturedBody"]
        self.assertEqual(body["agent_backend"], "cc")
        self.assertNotIn("args", body)


class TestPiNoExtensionsSourceWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.source = APP_JS.read_text(encoding="utf-8")

    def test_state_variable_declared(self) -> None:
        self.assertIn("let newSessionPiNoExtensions = false;", self.source)

    def test_dom_toggle_and_field_present(self) -> None:
        self.assertIn('id: "newSessionPiNoExtensionsToggle"', self.source)
        self.assertIn('id: "newSessionPiNoExtensionsField"', self.source)
        self.assertIn("Disable default Pi extensions (-ne)", self.source)
        # Mechanism is explained to the user.
        self.assertIn("Codoxear's active-session bridge still loads", self.source)
        self.assertIn("installed Pi extensions break startup", self.source)

    def test_field_mounted_in_form_body(self) -> None:
        start = self.source.index("newSessionLaunchRow,")
        end = self.source.index("newSessionWorktreeField", start)
        # The pi-no-extensions field is declared once and mounted in form body.
        self.assertGreaterEqual(self.source.count("newSessionPiNoExtensionsField"), 2)
        # Mounted after the worktree field inside the form body.
        form_start = self.source.index("newSessionLaunchRow,\n            newSessionWorktreeField,")
        self.assertIn("newSessionPiNoExtensionsField,", self.source[form_start:form_start + 120])

    def test_onchange_wires_to_setter(self) -> None:
        self.assertIn(
            "newSessionPiNoExtensionsToggle.onchange = () => setNewSessionPiNoExtensions(newSessionPiNoExtensionsToggle.checked);",
            self.source,
        )

    def test_setter_resets_when_backend_not_pi(self) -> None:
        block = self.source[self.source.index("function setNewSessionPiNoExtensions("):]
        head = block[: block.index("}") + 1]
        self.assertIn("newSessionPiNoExtensions = newSessionBackend === \"pi\" && !!value;", head)
        self.assertIn("newSessionPiNoExtensionsToggle.checked = newSessionPiNoExtensions;", head)

    def test_visibility_projection_pi_only(self) -> None:
        block = self.source[self.source.index("function syncNewSessionRunConfigUi("):]
        head = block[: block.index("function ", 10)]
        self.assertIn("const piNoExtensionsSupported = newSessionBackend === \"pi\";", head)
        self.assertIn('newSessionPiNoExtensionsField.style.display = piNoExtensionsSupported ? "" : "none";', head)
        self.assertIn("if (!piNoExtensionsSupported) setNewSessionPiNoExtensions(false);", head)

    def test_start_handler_passes_pi_only_flag(self) -> None:
        start = self.source.index("newSessionStartBtn.onclick = async () => {")
        end = self.source.index("const FILE_CANDIDATE_CACHE_TTL_MS", start)
        block = self.source[start:end]
        self.assertIn('const piNoExtensions = agentBackend === "pi" && !!newSessionPiNoExtensions;', block)
        self.assertIn("}, agentBackend, piNoExtensions);", block)

    def test_spawn_function_accepts_pi_no_extensions_param(self) -> None:
        self.assertIn(
            "agentBackend = \"codex\", piNoExtensions = false) {",
            self.source,
        )

    def test_spawn_function_emits_args_only_for_pi(self) -> None:
        self.assertIn(
            'if (backend === "pi" && piNoExtensions) body.args = ["-ne"];',
            self.source,
        )

    def test_dialog_open_resets_toggle(self) -> None:
        start = self.source.index("function openNewSessionDialog")
        end = self.source.index("editPriorityRange.oninput", start)
        block = self.source[start:end]
        self.assertIn("newSessionPiNoExtensions = false;", block)
        self.assertIn("newSessionPiNoExtensionsToggle.checked = false;", block)


if __name__ == "__main__":
    unittest.main()
