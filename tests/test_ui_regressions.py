import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def extract_function_block(source: str, signature: str) -> str:
    start = source.index(signature)
    brace_start = source.index("{", start)
    depth = 0
    for idx in range(brace_start, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start:idx + 1]
    raise ValueError(f"could not extract function block for {signature!r}")


def eval_delete_session_confirm_text(session: dict[str, object]) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function baseName(p) {")
    end = source.index("function sessionIsFast(s) {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_deleteSessionConfirmText = deleteSessionConfirmText;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_deleteSessionConfirmText({json.dumps(session)})));
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


def eval_dispose_file_editor_order() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_function_block(source, "function disposeFileEditor() {")
    injected = json.dumps(snippet + "\nglobalThis.__test_disposeFileEditor = disposeFileEditor;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const ctx = {{
          fileDiff: {{ innerHTML: "before" }},
          fileEditorKind: "diff",
          fileEditorModels: [
            {{ dispose() {{ calls.push("model:1"); }} }},
            {{ dispose() {{ calls.push("model:2"); }} }},
          ],
          fileEditor: {{ dispose() {{ calls.push("editor"); }} }},
          fileEditorChangeDisposable: null,
          fileEditorProgrammaticChange: false,
          fileTouchSelectMode: false,
          fileTouchSelectAnchor: null,
          fileTouchSelectHead: null,
          fileTouchSelectGoalColumn: null,
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_disposeFileEditor();
        process.stdout.write(JSON.stringify({{
          calls,
          fileDiffInnerHTML: ctx.fileDiff.innerHTML,
          fileEditor: ctx.fileEditor,
          fileEditorKind: ctx.fileEditorKind,
          fileEditorModelsLength: ctx.fileEditorModels.length,
        }}));
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


def eval_set_status(*, running: bool, queue_len: int, mobile: bool = False, selected: str | None = "sid") -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_function_block(source, "function setStatus({ running, queueLen }) {")
    injected = json.dumps(snippet + "\nglobalThis.__test_setStatus = setStatus;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const classSet = new Set();
        const ctx = {{
          currentRunning: false,
          currentQueueLen: 0,
          selected: {json.dumps(selected)},
          statusChip: {{
            style: {{ display: "initial" }},
            textContent: "",
            classList: {{
              add(name) {{ classSet.add(name); }},
              remove(name) {{ classSet.delete(name); }},
            }},
          }},
          interruptBtn: {{ style: {{ display: "initial" }}, disabled: null }},
          isMobile() {{ return {str(mobile).lower()}; }},
          updateQueueBadge() {{ calls.push("updateQueueBadge"); }},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_setStatus({{ running: {str(running).lower()}, queueLen: {queue_len} }});
        process.stdout.write(JSON.stringify({{
          statusDisplay: ctx.statusChip.style.display,
          statusText: ctx.statusChip.textContent,
          statusClasses: Array.from(classSet),
          interruptDisplay: ctx.interruptBtn.style.display,
          interruptDisabled: ctx.interruptBtn.disabled,
          currentRunning: ctx.currentRunning,
          currentQueueLen: ctx.currentQueueLen,
          calls,
        }}));
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


class TestLoginUi(unittest.TestCase):
    def test_login_uses_form_submit_path_for_enter(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function renderLogin(onAuthed) {")
        end = source.index("function renderApp() {", start)
        snippet = source[start:end]

        self.assertIn('el("form"', snippet)
        self.assertIn('id: "loginForm"', snippet)
        self.assertIn('type: "submit"', snippet)
        self.assertIn('$("#loginForm").onsubmit = async (e) => {', snippet)

    def test_sync_pi_view_toggle_function_exists(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn("function syncPiViewToggle() {", source)


class TestDeleteConfirmationUi(unittest.TestCase):
    def test_terminal_owned_delete_warning_mentions_terminal_shutdown(self) -> None:
        text = eval_delete_session_confirm_text({"owned": False})

        self.assertIn("terminal-owned session", text)
        self.assertIn("terminal session", text)
        self.assertIn("stop", text.lower())

    def test_web_owned_delete_warning_stays_scoped_to_web_session(self) -> None:
        text = eval_delete_session_confirm_text({"owned": True})

        self.assertIn("web-owned session", text)
        self.assertNotIn("terminal session", text)

    def test_delete_warning_includes_session_name_and_short_id(self) -> None:
        text = eval_delete_session_confirm_text(
            {
                "owned": True,
                "session_id": "f8f17ffa7dc7432787693273a3701eb2",
                "first_user_message": "Investigate delete bug",
                "cwd": "/tmp/demo",
            }
        )

        self.assertIn("Investigate delete bug", text)
        self.assertIn("f8f17ffa", text)


class TestFileViewerUi(unittest.TestCase):
    def test_monaco_loader_uses_min_base_and_vs_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const base = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min";', source)
        self.assertIn('window.require.config({ paths: { vs: base + "/vs" } });', source)
        self.assertIn('self.MonacoEnvironment={baseUrl:${JSON.stringify(base + "/")}};', source)

    def test_dispose_file_editor_disposes_editor_before_models(self) -> None:
        result = eval_dispose_file_editor_order()

        self.assertEqual(result["calls"], ["editor", "model:1", "model:2"])
        self.assertEqual(result["fileDiffInnerHTML"], "")
        self.assertIsNone(result["fileEditor"])
        self.assertEqual(result["fileEditorKind"], "")
        self.assertEqual(result["fileEditorModelsLength"], 0)


class TestHeaderBarUi(unittest.TestCase):
    def test_top_meta_keeps_status_chip_visible(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('const statusChip = el("span", { class: "status-chip", id: "statusChip", text: "Idle" });')
        end = source.index('const topbar = el("div", { class: "topbar" }, [', start)
        block = source[start:end]

        self.assertIn('const topMeta = el("div", { class: "topMeta" }, [statusChip, piViewToggle, ctxChip]);', block)

    def test_set_status_keeps_running_chip_visible(self) -> None:
        result = eval_set_status(running=True, queue_len=0)

        self.assertEqual(result["statusDisplay"], "inline-flex")
        self.assertEqual(result["statusText"], "Running")
        self.assertIn("running", result["statusClasses"])
        self.assertEqual(result["interruptDisplay"], "inline-flex")
        self.assertFalse(result["interruptDisabled"])
        self.assertTrue(result["currentRunning"])
        self.assertEqual(result["currentQueueLen"], 0)
        self.assertEqual(result["calls"], ["updateQueueBadge"])

    def test_set_status_shows_queue_text_when_idle(self) -> None:
        result = eval_set_status(running=False, queue_len=2)

        self.assertEqual(result["statusDisplay"], "inline-flex")
        self.assertEqual(result["statusText"], "Queue 2")
        self.assertNotIn("running", result["statusClasses"])
        self.assertEqual(result["interruptDisplay"], "none")
        self.assertTrue(result["interruptDisabled"])


if __name__ == "__main__":
    unittest.main()
