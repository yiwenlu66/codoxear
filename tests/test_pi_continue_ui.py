import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def extract_js_function(source: str, name: str, occurrence: int = 1) -> str:
    needle = f"function {name}("
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = source.index(needle, search_from)
        search_from = start + len(needle)
    paren_start = source.index("(", start)
    paren_depth = 0
    body_start = -1
    for idx in range(paren_start, len(source)):
        ch = source[idx]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
            if paren_depth == 0:
                body_start = source.index("{", idx)
                break
    if body_start < 0:
        raise ValueError(f"Unable to locate body for function: {name}")
    depth = 0
    for idx in range(body_start, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise ValueError(f"Unterminated function: {name}")


def eval_pi_continue_missing_selection_message(*, backend: str, mode: str, candidate_count: int, has_selection: bool) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "piContinueMissingSelectionMessage")
    selection_literal = "{ session_id: 'sess-0' }" if has_selection else "null"
    injected = json.dumps(snippet + "\nglobalThis.__test_piContinueMissingSelectionMessage = piContinueMissingSelectionMessage;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: {json.dumps(backend)},
          newSessionPiMode: {json.dumps(mode)},
          newSessionResumeLoadError: "",
          newSessionResumeCandidates: Array.from({{ length: {candidate_count} }}, (_, idx) => ({{ session_id: `sess-${{idx}}` }})),
          newSessionResumeSelection: {selection_literal},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_piContinueMissingSelectionMessage()));
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


def eval_backend_switch_clears_resume_state(*, backend: str, mode: str, candidate_count: int, has_selection: bool) -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "setNewSessionBackend")
    selection_literal = "{ session_id: 'sess-0' }" if has_selection else "null"
    injected = json.dumps(snippet + "\nglobalThis.__test_setNewSessionBackend = setNewSessionBackend;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: "codex",
          newSessionPiMode: {json.dumps(mode)},
          newSessionProvider: "chatgpt",
          newSessionReasoningEffort: "medium",
          newSessionFast: true,
          newSessionResumeCandidates: Array.from({{ length: {candidate_count} }}, (_, idx) => ({{ session_id: `sess-${{idx}}` }})),
          newSessionResumeSelection: {selection_literal},
          newSessionModelInput: {{ value: "stale-model" }},
          newSessionResumeLoadSeq: 7,
          newSessionResumeLoadTimer: 42,
          clearTimeout: (value) => {{ ctx.clearedTimer = value; }},
          normalizeAgentBackendName: (value) => String(value || "").trim().toLowerCase() === "pi" ? "pi" : "codex",
          rememberBackendChoice: (value) => {{ ctx.rememberedBackend = value; }},
          defaultsForAgentBackend: (value) => value === "pi"
            ? {{ provider_choice: "macaron", model: "pi-model", reasoning_effort: "high", service_tier: "standard" }}
            : {{ provider_choice: "chatgpt", model: "codex-model", reasoning_effort: "medium", service_tier: "fast" }},
          providerChoicesForBackend: (value) => value === "pi" ? ["macaron"] : ["chatgpt", "openai-api"],
          loadRememberedProviderChoice: (value) => value === "pi" ? "macaron" : "chatgpt",
          reasoningChoicesForBackend: (value) => value === "pi" ? ["high", "medium"] : ["medium", "high"],
          setNewSessionProvider: (value) => {{ ctx.newSessionProvider = value; }},
          setNewSessionReasoningEffort: (value) => {{ ctx.newSessionReasoningEffort = value; }},
          setNewSessionFast: (value) => {{ ctx.newSessionFast = !!value; }},
          renderNewSessionBackendTabs: () => {{ ctx.renderBackendTabsCalls = (ctx.renderBackendTabsCalls || 0) + 1; }},
          renderNewSessionProviderMenu: () => {{ ctx.renderProviderMenuCalls = (ctx.renderProviderMenuCalls || 0) + 1; }},
          renderNewSessionReasoningMenu: () => {{ ctx.renderReasoningMenuCalls = (ctx.renderReasoningMenuCalls || 0) + 1; }},
          renderNewSessionModelMenu: () => {{ ctx.renderModelMenuCalls = (ctx.renderModelMenuCalls || 0) + 1; }},
          syncNewSessionRunConfigUi: () => {{ ctx.syncRunConfigCalls = (ctx.syncRunConfigCalls || 0) + 1; }},
          syncNewSessionWorktreeUi: () => {{ ctx.syncWorktreeCalls = (ctx.syncWorktreeCalls || 0) + 1; }},
          syncNewSessionBackendUi: () => {{ ctx.syncBackendCalls = (ctx.syncBackendCalls || 0) + 1; }},
          resetNewSessionResumeState: () => {{
            if (ctx.newSessionResumeLoadTimer) ctx.clearedTimer = ctx.newSessionResumeLoadTimer;
            ctx.newSessionResumeCandidates = [];
            ctx.newSessionResumeSelection = null;
            ctx.newSessionResumeLoadSeq += 1;
            ctx.newSessionResumeLoadTimer = null;
            ctx.renderResumeMenuCalls = (ctx.renderResumeMenuCalls || 0) + 1;
          }},
          syncNewSessionPiContinueStatus: () => {{ ctx.syncPiContinueCalls = (ctx.syncPiContinueCalls || 0) + 1; }},
          renderNewSessionResumeMenu: () => {{ ctx.renderResumeMenuCalls = (ctx.renderResumeMenuCalls || 0) + 1; }},
          scheduleNewSessionResumeLoad: () => {{ ctx.resumeLoadScheduled = (ctx.resumeLoadScheduled || 0) + 1; }},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_setNewSessionBackend({json.dumps(backend)});
        process.stdout.write(JSON.stringify({{
          backend: ctx.newSessionBackend,
          rememberedBackend: ctx.rememberedBackend || "",
          provider: ctx.newSessionProvider,
          model: ctx.newSessionModelInput.value,
          reasoningEffort: ctx.newSessionReasoningEffort,
          fast: ctx.newSessionFast,
          selection: ctx.newSessionResumeSelection,
          candidates: ctx.newSessionResumeCandidates,
          loadSeq: ctx.newSessionResumeLoadSeq,
          timerCleared: ctx.clearedTimer,
          timerValue: ctx.newSessionResumeLoadTimer,
          renderBackendTabsCalls: ctx.renderBackendTabsCalls || 0,
          renderProviderMenuCalls: ctx.renderProviderMenuCalls || 0,
          renderReasoningMenuCalls: ctx.renderReasoningMenuCalls || 0,
          renderModelMenuCalls: ctx.renderModelMenuCalls || 0,
          syncRunConfigCalls: ctx.syncRunConfigCalls || 0,
          syncWorktreeCalls: ctx.syncWorktreeCalls || 0,
          syncBackendCalls: ctx.syncBackendCalls || 0,
          renderResumeMenuCalls: ctx.renderResumeMenuCalls || 0,
          resumeLoadScheduled: ctx.resumeLoadScheduled || 0,
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


def eval_sync_new_session_backend_ui(*, backend: str, mode: str, has_selection: bool) -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "syncNewSessionBackendUi")
    selection_literal = "{ session_id: 'sess-0' }" if has_selection else "null"
    injected = json.dumps(snippet + "\nglobalThis.__test_syncNewSessionBackendUi = syncNewSessionBackendUi;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: {json.dumps(backend)},
          newSessionPiMode: {json.dumps(mode)},
          newSessionTmuxToggle: {{ checked: true }},
          newSessionWorktreeToggle: {{ checked: true }},
          newSessionPiModeField: {{ style: {{ display: "initial" }} }},
          newSessionResumeField: {{ style: {{ display: "initial" }} }},
          newSessionResumeFieldLabel: {{ textContent: "" }},
          newSessionProviderTmuxRow: {{ style: {{ display: "initial" }} }},
          newSessionResumeSelection: {selection_literal},
          renderNewSessionResumeMenu: () => {{ ctx.renderResumeMenuCalls = (ctx.renderResumeMenuCalls || 0) + 1; }},
          setNewSessionResumeSelection: (value) => {{ ctx.newSessionResumeSelection = value; }},
          syncNewSessionWorktreeUi: () => {{ ctx.syncWorktreeCalls = (ctx.syncWorktreeCalls || 0) + 1; }},
          syncNewSessionPiContinueStatus: () => {{ ctx.syncPiContinueCalls = (ctx.syncPiContinueCalls || 0) + 1; }},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_syncNewSessionBackendUi();
        process.stdout.write(JSON.stringify({{
          tmuxChecked: ctx.newSessionTmuxToggle.checked,
          worktreeChecked: ctx.newSessionWorktreeToggle.checked,
          piModeDisplay: ctx.newSessionPiModeField.style.display,
          resumeDisplay: ctx.newSessionResumeField.style.display,
          resumeLabel: ctx.newSessionResumeFieldLabel.textContent,
          providerTmuxDisplay: ctx.newSessionProviderTmuxRow.style.display,
          selection: ctx.newSessionResumeSelection,
          renderResumeMenuCalls: ctx.renderResumeMenuCalls || 0,
          syncWorktreeCalls: ctx.syncWorktreeCalls || 0,
          syncPiContinueCalls: ctx.syncPiContinueCalls || 0,
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


def eval_sync_new_session_worktree_ui(*, backend: str, git_repo: bool, has_selection: bool, checked: bool, mode: str = "new") -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "syncNewSessionWorktreeUi")
    selection_literal = "{ session_id: 'sess-0' }" if has_selection else "null"
    injected = json.dumps(snippet + "\nglobalThis.__test_syncNewSessionWorktreeUi = syncNewSessionWorktreeUi;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: {json.dumps(backend)},
          newSessionPiMode: {json.dumps(mode)},
          newSessionCwdInfo: {{ git_repo: {json.dumps(git_repo)} }},
          newSessionResumeSelection: {selection_literal},
          newSessionWorktreeToggle: {{ checked: {json.dumps(checked)} }},
          newSessionWorktreeField: {{ style: {{ display: "initial" }} }},
          newSessionWorktreeInput: {{ disabled: false, style: {{ display: "initial" }}, value: "feature/demo" }},
          newSessionStartBtn: {{ textContent: "" }},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_syncNewSessionWorktreeUi();
        process.stdout.write(JSON.stringify({{
          checked: ctx.newSessionWorktreeToggle.checked,
          fieldDisplay: ctx.newSessionWorktreeField.style.display,
          inputDisabled: ctx.newSessionWorktreeInput.disabled,
          inputDisplay: ctx.newSessionWorktreeInput.style.display,
          startText: ctx.newSessionStartBtn.textContent,
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


def eval_render_new_session_reasoning_menu(*, backend: str) -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "renderNewSessionReasoningMenu")
    injected = json.dumps(snippet + "\nglobalThis.__test_renderNewSessionReasoningMenu = renderNewSessionReasoningMenu;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        function el(_tag, attrs = {{}}, children = []) {{
          const node = {{
            attrs,
            children: [],
            textContent: attrs.text || "",
            appendChild(child) {{ this.children.push(child); return child; }},
          }};
          for (const child of Array.isArray(children) ? children : [children]) {{
            if (child) node.children.push(child);
          }}
          return node;
        }}
        const ctx = {{
          el,
          newSessionBackend: {json.dumps(backend)},
          newSessionReasoningEffort: "high",
          newSessionReasoningMenuOpen: false,
          newSessionReasoningMenu: {{
            innerHTML: "",
            children: [],
            appendChild(child) {{ this.children.push(child); return child; }},
          }},
          reasoningChoicesForBackend: (value) => value === "pi" ? ["high", "medium"] : ["medium", "low"],
          setNewSessionReasoningEffort(value) {{ ctx.newSessionReasoningEffort = value; }},
          applyDialogMenus() {{}},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_renderNewSessionReasoningMenu();
        const labels = ctx.newSessionReasoningMenu.children.map((child) => {{
          const labelNode = (child.children || []).find((item) => item && item.attrs && item.attrs.class === "fileMenuPath");
          return labelNode ? labelNode.textContent : "";
        }});
        process.stdout.write(JSON.stringify({{ labels }}));
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


def eval_apply_dialog_menus_without_backend_picker() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "applyDialogMenus")
    injected = json.dumps(snippet + "\nglobalThis.__test_applyDialogMenus = applyDialogMenus;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const toggler = () => ({{ classList: {{ toggle() {{}} }} }});
        const ctx = {{
          editDependencyMenu: toggler(),
          newSessionCwdMenu: toggler(),
          newSessionModelMenu: toggler(),
          newSessionProviderMenu: toggler(),
          newSessionReasoningMenu: toggler(),
          newSessionResumeMenu: toggler(),
          editDependencyMenuOpen: false,
          newSessionCwdMenuOpen: false,
          newSessionCwdMenuFocus: -1,
          newSessionModelMenuOpen: false,
          newSessionModelMenuFocus: -1,
          newSessionProviderMenuOpen: false,
          newSessionReasoningMenuOpen: false,
          newSessionResumeMenuOpen: false,
          editDependencyBtn: {{ setAttribute() {{}} }},
          newSessionCwdInput: {{ setAttribute() {{}}, removeAttribute() {{}} }},
          newSessionModelInput: {{ setAttribute() {{}}, removeAttribute() {{}} }},
          newSessionProviderBtn: {{ setAttribute() {{}} }},
          newSessionReasoningBtn: {{ setAttribute() {{}} }},
          newSessionResumeBtn: {{ setAttribute() {{}} }},
          positionDialogMenu() {{}},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        try {{
          ctx.__test_applyDialogMenus();
          process.stdout.write(JSON.stringify({{ ok: true }}));
        }} catch (error) {{
          process.stdout.write(JSON.stringify({{ ok: false, error: String(error && error.message ? error.message : error) }}));
        }}
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


def eval_hide_new_session_dialog_without_backend_picker() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "hideNewSessionDialog")
    injected = json.dumps(snippet + "\nglobalThis.__test_hideNewSessionDialog = hideNewSessionDialog;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionStatus: {{ textContent: "busy" }},
          newSessionCwdMenuOpen: true,
          newSessionCwdMenuFocus: 3,
          newSessionModelMenuOpen: true,
          newSessionModelMenuFocus: 2,
          newSessionProviderMenuOpen: true,
          newSessionReasoningMenuOpen: true,
          newSessionResumeMenuOpen: true,
          applyDialogMenus() {{}},
          newSessionBackdrop: {{ style: {{ display: "block" }} }},
          newSessionViewer: {{ style: {{ display: "flex" }} }},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        try {{
          ctx.__test_hideNewSessionDialog();
          process.stdout.write(JSON.stringify({{
            ok: true,
            status: ctx.newSessionStatus.textContent,
            viewerDisplay: ctx.newSessionViewer.style.display,
            backdropDisplay: ctx.newSessionBackdrop.style.display,
          }}));
        }} catch (error) {{
          process.stdout.write(JSON.stringify({{ ok: false, error: String(error && error.message ? error.message : error) }}));
        }}
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


def eval_open_new_session_dialog_without_backend_picker() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = extract_js_function(source, "setNewSessionBackend") + "\n" + extract_js_function(source, "openNewSessionDialog")
    injected = json.dumps(snippet + "\nglobalThis.__test_openNewSessionDialog = openNewSessionDialog;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          selected: null,
          sessionIndex: {{ get() {{ return null; }} }},
          loadRememberedBackendChoice: () => "pi",
          normalizeAgentBackendName: (value) => String(value || "").trim().toLowerCase() === "pi" ? "pi" : "codex",
          rememberBackendChoice(value) {{ ctx.rememberedBackend = value; }},
          defaultsForAgentBackend: (value) => value === "pi"
            ? {{ provider_choice: "macaron", model: "pi-model", reasoning_effort: "high", service_tier: "standard" }}
            : {{ provider_choice: "chatgpt", model: "codex-model", reasoning_effort: "medium", service_tier: "fast" }},
          providerChoicesForBackend: (value) => value === "pi" ? ["macaron"] : ["chatgpt", "openai-api"],
          loadRememberedProviderChoice: (value) => value === "pi" ? "macaron" : "chatgpt",
          reasoningChoicesForBackend: (value) => value === "pi" ? ["high", "medium"] : ["medium", "high"],
          newSessionDefaults: {{ default_backend: "codex" }},
          newSessionStatus: {{ textContent: "" }},
          newSessionCwdInput: {{ value: "" }},
          newSessionNameInput: {{ value: "" }},
          newSessionModelInput: {{ value: "" }},
          newSessionProvider: "chatgpt",
          newSessionReasoningEffort: "medium",
          newSessionFast: true,
          newSessionBackend: "codex",
          syncNewSessionNamePlaceholder() {{}},
          newSessionResumeCandidates: ["stale"],
          setNewSessionPiMode(value) {{ ctx.piMode = value; }},
          setNewSessionResumeSelection(value) {{ ctx.resumeSelection = value; }},
          setNewSessionCwdError(value) {{ ctx.cwdError = value; }},
          clearNewSessionCwdInfo() {{ ctx.clearedCwdInfo = true; }},
          tmuxAvailable: true,
          newSessionTmuxToggle: {{ checked: false }},
          newSessionWorktreeToggle: {{ checked: true }},
          newSessionWorktreeInput: {{ value: "branch", disabled: false, style: {{ display: "" }} }},
          newSessionWorktreeField: {{ style: {{ display: "" }} }},
          newSessionCwdMenuOpen: true,
          newSessionCwdMenuFocus: 1,
          newSessionModelMenuOpen: true,
          newSessionModelMenuFocus: 1,
          newSessionProviderMenuOpen: true,
          newSessionReasoningMenuOpen: true,
          renderRecentCwdMenu() {{}},
          setNewSessionProvider(value) {{ ctx.newSessionProvider = value; }},
          setNewSessionReasoningEffort(value) {{ ctx.newSessionReasoningEffort = value; }},
          setNewSessionFast(value) {{ ctx.newSessionFast = !!value; }},
          renderNewSessionBackendTabs() {{ ctx.renderBackendTabsCalls = (ctx.renderBackendTabsCalls || 0) + 1; }},
          renderNewSessionProviderMenu() {{ ctx.renderProviderMenuCalls = (ctx.renderProviderMenuCalls || 0) + 1; }},
          renderNewSessionReasoningMenu() {{ ctx.renderReasoningMenuCalls = (ctx.renderReasoningMenuCalls || 0) + 1; }},
          renderNewSessionModelMenu() {{ ctx.renderModelMenuCalls = (ctx.renderModelMenuCalls || 0) + 1; }},
          syncNewSessionRunConfigUi() {{ ctx.syncRunConfigCalls = (ctx.syncRunConfigCalls || 0) + 1; }},
          resetNewSessionResumeState() {{
            ctx.newSessionResumeCandidates = [];
            ctx.resumeSelection = null;
            ctx.newSessionResumeLoadSeq = (ctx.newSessionResumeLoadSeq || 0) + 1;
          }},
          syncNewSessionBackendUi() {{ ctx.syncBackendCalls = (ctx.syncBackendCalls || 0) + 1; }},
          syncNewSessionPiContinueStatus() {{ ctx.syncPiContinueCalls = (ctx.syncPiContinueCalls || 0) + 1; }},
          renderNewSessionResumeMenu() {{}},
          newSessionBackdrop: {{ style: {{ display: "none" }} }},
          newSessionViewer: {{ style: {{ display: "none" }} }},
          scheduleNewSessionResumeLoad() {{ ctx.resumeLoadScheduled = true; }},
          syncNewSessionTmuxUi() {{ ctx.tmuxSynced = true; }},
          syncNewSessionWorktreeUi() {{ ctx.worktreeSynced = true; }},
          isMobile() {{ return true; }},
          requestAnimationFrame() {{}},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        try {{
          ctx.__test_openNewSessionDialog({{ cwd: "/tmp/demo", statusText: "Loading" }});
          process.stdout.write(JSON.stringify({{
            ok: true,
            status: ctx.newSessionStatus.textContent,
            backdropDisplay: ctx.newSessionBackdrop.style.display,
            viewerDisplay: ctx.newSessionViewer.style.display,
            backend: ctx.newSessionBackend,
            rememberedBackend: ctx.rememberedBackend || "",
            provider: ctx.newSessionProvider,
          }}));
        }} catch (error) {{
          process.stdout.write(JSON.stringify({{ ok: false, error: String(error && error.message ? error.message : error) }}));
        }}
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


class TestPiContinueUi(unittest.TestCase):
    def test_pi_continue_requires_selected_candidate_when_choices_exist(self) -> None:
        self.assertEqual(
            eval_pi_continue_missing_selection_message(backend="pi", mode="continue", candidate_count=2, has_selection=False),
            "Choose a Pi session to continue.",
        )

    def test_pi_continue_reports_when_no_resume_candidates_exist(self) -> None:
        self.assertEqual(
            eval_pi_continue_missing_selection_message(backend="pi", mode="continue", candidate_count=0, has_selection=False),
            "No Pi sessions available to continue for this directory.",
        )

    def test_pi_continue_message_clears_once_selection_exists(self) -> None:
        self.assertEqual(
            eval_pi_continue_missing_selection_message(backend="pi", mode="continue", candidate_count=2, has_selection=True),
            "",
        )

    def test_switching_backend_clears_stale_resume_state_before_reload(self) -> None:
        result = eval_backend_switch_clears_resume_state(backend="pi", mode="continue", candidate_count=2, has_selection=True)

        self.assertEqual(result["backend"], "pi")
        self.assertEqual(result["rememberedBackend"], "pi")
        self.assertEqual(result["provider"], "macaron")
        self.assertEqual(result["model"], "pi-model")
        self.assertEqual(result["reasoningEffort"], "high")
        self.assertFalse(result["fast"])
        self.assertIsNone(result["selection"])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["loadSeq"], 8)
        self.assertEqual(result["timerCleared"], 42)
        self.assertIsNone(result["timerValue"])
        self.assertEqual(result["renderBackendTabsCalls"], 1)
        self.assertEqual(result["renderProviderMenuCalls"], 1)
        self.assertEqual(result["renderReasoningMenuCalls"], 1)
        self.assertEqual(result["renderModelMenuCalls"], 1)
        self.assertEqual(result["syncRunConfigCalls"], 1)
        self.assertEqual(result["syncBackendCalls"], 1)
        self.assertEqual(result["resumeLoadScheduled"], 1)

    def test_new_session_dialog_uses_header_tabs_as_only_backend_control(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('const newSessionViewer = el("div", { class: "formViewer newSessionViewer"')
        end = source.index("root.appendChild(newSessionBackdrop);", start)
        block = source[start:end]

        self.assertIn("newSessionBackendTabs", block)
        self.assertNotIn("newSessionBackendRow", block)
        self.assertNotIn("newSessionBackendField", block)
        self.assertNotIn("newSessionBackendBtn", block)
        self.assertIn("newSessionResumeField,", block)
        self.assertIn("newSessionProviderTmuxRow,", block)

    def test_backend_tab_buttons_still_switch_backend_state(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function renderNewSessionBackendTabs() {")
        end = source.index("function renderNewSessionReasoningMenu() {", start)
        block = source[start:end]

        self.assertIn('for (const backend of ["codex", "pi"]) {', block)
        self.assertIn("const active = newSessionBackend === backend;", block)
        self.assertIn('class: `agentBackendTab${active ? " active" : ""}`', block)
        self.assertIn("if (active) return;", block)
        self.assertIn("setNewSessionBackend(backend, { resetSelections: true });", block)
        self.assertIn("newSessionBackendTabs.appendChild(btn);", block)

    def test_backend_selection_still_persists_remembered_choice(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        set_backend_start = source.index("function setNewSessionBackend(value, { resetSelections = false } = {}) {")
        set_backend_end = source.index("function setNewSessionProvider(value) {", set_backend_start)
        set_backend_block = source[set_backend_start:set_backend_end]
        self.assertIn("rememberBackendChoice(next);", set_backend_block)

        open_start = source.index("function openNewSessionDialog({ cwd = null, statusText = \"\" } = {}) {")
        open_end = source.index("editPriorityRange.oninput = syncEditPriorityLabel;", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("const rememberedBackend = loadRememberedBackendChoice();", open_block)
        self.assertIn("const initialBackend = rememberedBackend ||", open_block)

    def test_apply_dialog_menus_runs_without_backend_picker_objects(self) -> None:
        result = eval_apply_dialog_menus_without_backend_picker()
        self.assertTrue(result["ok"], result.get("error"))

    def test_document_click_does_not_reference_removed_backend_picker(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('document.addEventListener("click", (e) => {')
        end = source.index('document.addEventListener("keydown", (e) => {', start)
        block = source[start:end]

        self.assertNotIn("#newSessionBackendField", block)
        self.assertNotIn("#newSessionBackendMenu", block)
        self.assertNotIn("newSessionBackendMenuOpen", block)

    def test_backend_ui_sync_updates_backend_specific_fields(self) -> None:
        pi_result = eval_sync_new_session_backend_ui(backend="pi", mode="new", has_selection=True)
        self.assertFalse(pi_result["tmuxChecked"])
        self.assertFalse(pi_result["worktreeChecked"])
        self.assertEqual(pi_result["piModeDisplay"], "")
        self.assertEqual(pi_result["resumeDisplay"], "none")
        self.assertEqual(pi_result["resumeLabel"], "Continue session")
        self.assertEqual(pi_result["providerTmuxDisplay"], "none")
        self.assertIsNone(pi_result["selection"])
        self.assertEqual(pi_result["renderResumeMenuCalls"], 1)
        self.assertEqual(pi_result["syncWorktreeCalls"], 1)
        self.assertEqual(pi_result["syncPiContinueCalls"], 1)

        codex_result = eval_sync_new_session_backend_ui(backend="codex", mode="new", has_selection=False)
        self.assertTrue(codex_result["tmuxChecked"])
        self.assertTrue(codex_result["worktreeChecked"])
        self.assertEqual(codex_result["piModeDisplay"], "none")
        self.assertEqual(codex_result["resumeDisplay"], "")
        self.assertEqual(codex_result["resumeLabel"], "Resume conversation")
        self.assertEqual(codex_result["providerTmuxDisplay"], "")

    def test_worktree_ui_updates_for_backend_specific_availability(self) -> None:
        pi_result = eval_sync_new_session_worktree_ui(backend="pi", git_repo=True, has_selection=False, checked=True)
        self.assertEqual(pi_result["fieldDisplay"], "none")
        self.assertTrue(pi_result["inputDisabled"])
        self.assertEqual(pi_result["inputDisplay"], "none")
        self.assertEqual(pi_result["startText"], "Start session")

        codex_result = eval_sync_new_session_worktree_ui(backend="codex", git_repo=True, has_selection=False, checked=True)
        self.assertEqual(codex_result["fieldDisplay"], "")
        self.assertFalse(codex_result["inputDisabled"])
        self.assertEqual(codex_result["inputDisplay"], "")
        self.assertEqual(codex_result["startText"], "Create worktree session")

    def test_reasoning_menu_choices_change_with_backend(self) -> None:
        pi_result = eval_render_new_session_reasoning_menu(backend="pi")
        codex_result = eval_render_new_session_reasoning_menu(backend="codex")

        self.assertEqual(pi_result["labels"], ["high", "medium"])
        self.assertEqual(codex_result["labels"], ["medium", "low"])

    def test_new_session_dialog_close_runs_without_backend_picker_state(self) -> None:
        result = eval_hide_new_session_dialog_without_backend_picker()
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["status"], "")
        self.assertEqual(result["viewerDisplay"], "none")
        self.assertEqual(result["backdropDisplay"], "none")

    def test_new_session_dialog_open_runs_without_backend_picker_state(self) -> None:
        result = eval_open_new_session_dialog_without_backend_picker()
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["status"], "Loading")
        self.assertEqual(result["backdropDisplay"], "block")
        self.assertEqual(result["viewerDisplay"], "flex")
        self.assertEqual(result["backend"], "pi")
        self.assertEqual(result["rememberedBackend"], "pi")
        self.assertEqual(result["provider"], "macaron")

    def test_backend_picker_infrastructure_is_fully_removed(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertNotIn("const newSessionBackendBtn = el(", source)
        self.assertNotIn("const newSessionBackendField = el(", source)
        self.assertNotIn("const newSessionBackendRow = el(", source)
        self.assertNotIn("let newSessionBackendMenuOpen = false;", source)
        self.assertNotIn("function renderNewSessionBackendMenu() {", source)
        self.assertNotIn("newSessionViewer.appendChild(newSessionBackendMenu);", source)

    def test_start_handler_uses_pi_continue_blocker_message(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("newSessionStartBtn.onclick = async () => {")
        end = source.index('let fileViewMode = localStorage.getItem("codexweb.fileViewMode") || "diff";', start)
        block = source[start:end]
        self.assertIn("const piContinueBlocker = piContinueMissingSelectionMessage();", block)
        self.assertIn("if (piContinueBlocker) {", block)
        self.assertIn("newSessionStatus.textContent = piContinueBlocker;", block)
        self.assertIn("return;", block)
        self.assertIn("const brokerPid = await spawnSessionWithCwd({", block)
        self.assertIn("cwd,", block)
        self.assertIn("resumeSessionId,", block)
        self.assertIn("worktreeBranch,", block)
        self.assertIn("sessionName,", block)
        self.assertIn("providerChoice,", block)
        self.assertIn("model,", block)
        self.assertIn("reasoningEffort: newSessionReasoningEffort,", block)
        self.assertIn("fast: newSessionFast,", block)
        self.assertIn("createInTmux,", block)
        self.assertIn("backend,", block)
        self.assertIn("errorHandler: (e) => {", block)

    def test_duplicate_session_uses_spawn_options_object(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("dupBtn.onclick = async (e) => {")
        end = source.index("const delBtn = el(", start)
        block = source[start:end]

        self.assertIn("await spawnSessionWithCwd({", block)
        self.assertIn("cwd,", block)
        self.assertIn("providerChoice: sessionProviderChoice(s),", block)
        self.assertIn('model: s && s.model ? s.model : "default",', block)
        self.assertIn('reasoningEffort: s && s.reasoning_effort ? s.reasoning_effort : "high",', block)
        self.assertIn("fast: sessionIsFast(s),", block)
        self.assertIn('createInTmux: !!(s && s.transport === "tmux"),', block)
        self.assertIn("backend: sessionAgentBackend(s),", block)

    def test_spawn_session_helper_accepts_options_object(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function spawnSessionWithCwd(")
        end = source.index('$("#newBtn").onclick = async () => {', start)
        block = source[start:end]

        self.assertIn("async function spawnSessionWithCwd({", block)
        self.assertIn("cwd,", block)
        self.assertIn("resumeSessionId = null,", block)
        self.assertIn("worktreeBranch = null,", block)
        self.assertIn('sessionName = "",', block)
        self.assertIn('providerChoice = "chatgpt",', block)
        self.assertIn('model = "default",', block)
        self.assertIn('reasoningEffort = "high",', block)
        self.assertIn("fast = false,", block)
        self.assertIn("createInTmux = false,", block)
        self.assertIn('backend = "codex",', block)
        self.assertIn("errorHandler = null,", block)
        self.assertIn("} = {}) {", block)


if __name__ == "__main__":
    unittest.main()
