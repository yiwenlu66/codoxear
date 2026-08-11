from frontend_module_loader import module_path
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_ATTACHMENTS_JS = module_path("app_attachments.js")


def run_attachments(body: str) -> dict:
    source = APP_ATTACHMENTS_JS.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const listeners = new Map();
        function node(id) {{
          const nodeListeners = new Map();
          const classes = new Set();
          return {{
            id,
            style: {{}}, attrs: {{}}, children: [], value: "", files: [], disabled: false,
            textContent: "", selectionStart: 0, selectionEnd: 0,
            classList: {{ toggle(name, enabled) {{ if (enabled) classes.add(name); else classes.delete(name); }}, contains(name) {{ return classes.has(name); }} }},
            appendChild(child) {{ this.children.push(child); return child; }},
            setAttribute(name, value) {{ this.attrs[name] = String(value); }},
            addEventListener(type, handler) {{ nodeListeners.set(type, handler); listeners.set(`${{id}}:${{type}}`, handler); }},
            removeEventListener() {{}},
            dispatchEvent() {{}},
            click() {{ this.clicked = true; }},
            set innerHTML(value) {{ this.children = []; }},
            get innerHTML() {{ return ""; }},
          }};
        }}
        const attachBtn = node("attachBtn");
        const imgInput = node("imgInput");
        const composer = node("composer");
        const textarea = node("textarea");
        const tray = node("tray");
        const sessions = new Map([["sid", {{ session_id: "sid", launch_state: "ready" }}]]);
        let selected = "sid";
        let sending = false;
        const apiCalls = [];
        const toasts = [];
        const pollCalls = [];
        const ctx = {{
          window: {{ innerWidth: 1200, innerHeight: 800, addEventListener() {{}}, removeEventListener() {{}} }},
          document: {{ documentElement: {{}}, body: {{}} }},
          Event: function Event(type, options) {{ this.type = type; this.options = options; }},
          Uint8Array, console,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx, {{ filename: "app_attachments.js" }});
        function el(tag, attrs = {{}}) {{
          const out = node(tag);
          out.tag = tag;
          out.className = attrs.class || "";
          out.title = attrs.title || "";
          out.textContent = attrs.text || "";
          if (attrs["aria-label"]) out.attrs["aria-label"] = attrs["aria-label"];
          return out;
        }}
        const controller = ctx.window.CodoxearAttachments.createAttachmentsController({{
          attachBtn, imgInput, composer, textarea,
          getSelected: () => selected,
          getSessionInfo: (sid) => sessions.get(sid) || null,
          patchSessionInfo: (sid, patch) => sessions.set(sid, Object.assign({{}}, sessions.get(sid), patch)),
          getSending: () => sending,
          sessionLaunchFailed: (info) => !!(info && info.launch_state === "failed"),
          sessionHasUnknownSend: (info) => !!(info && info.commit_unknown_send),
          sessionIsOrphanRecovery: (info) => !!(info && info.orphan_recovery),
          sessionHasOrphanQueueRecovery: (info) => !!(info && info.queue_recovery),
          api: async (path, options) => {{
            apiCalls.push({{ path, body: options.body }});
            return {{ ok: true, attachments: [
              {{ id: "att-1", display_name: "report.txt", filename: "server-report.txt", size: "5", created_ts: "8" }},
              {{ id: "att-1", display_name: "duplicate.txt", size: 10 }},
              {{ id: "", display_name: "invalid.txt" }},
            ] }};
          }},
          setToast: (text) => toasts.push(text), handleAppAuthLoss: () => toasts.push("auth-loss"),
          refreshSessions: async () => pollCalls.push("refresh"),
          setPollFastUntilMs: (value) => pollCalls.push(["fast", value]), kickPoll: (value) => pollCalls.push(["kick", value]),
          resizeComposer: () => pollCalls.push("resize"), getTray: () => tray, el,
          fmtBytes: (bytes) => `${{bytes}} B`, safeAttachmentStem: (name) => name,
          isLikelyHeic: () => false, looksLikeImage: () => false,
          b64FromBytes: (bytes) => Buffer.from(bytes).toString("base64"),
          dataTransferHasFiles: () => false, extractFilesFromClipboardData: () => [], extractFilesFromDropData: () => [],
          addEventListener: (target, type, handler) => target.addEventListener(type, handler),
          uploadMaxBytes: 1024, now: () => 100,
        }});
        {body}
        """
    )
    # The module source is deliberately evaluated as production JavaScript.
    # Passing it as `node -e` makes every consumer of this shared harness
    # depend on the OS command-line size limit as the module grows.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8", delete=False) as script_file:
        script_file.write(script)
    try:
        completed = subprocess.run(["node", script_file.name], check=True, capture_output=True, text=True)
    finally:
        Path(script_file.name).unlink(missing_ok=True)
    return json.loads(completed.stdout)


def test_stage_files_normalizes_dedupes_projects_badge_and_uploads_base64() -> None:
    result = run_attachments(
        """
        const file = { name: "notes.txt", size: 5, type: "text/plain", arrayBuffer: async () => new Uint8Array([104, 101, 108, 108, 111]).buffer };
        controller.stageFiles([file], { sid: "sid", source: "picker" }).then((staged) => {
          const badge = attachBtn.children[0];
          process.stdout.write(JSON.stringify({
            staged,
            upload: apiCalls[0],
            attachments: controller.getStagedAttachments(),
            session: sessions.get("sid"),
            badge: { text: badge.textContent, display: badge.style.display },
            tray: { display: tray.style.display, chips: tray.children.length },
            toasts,
          }));
        });
        """
    )
    assert result["staged"] is True
    assert result["upload"] == {
        "path": "/api/sessions/sid/inject_file",
        "body": {"filename": "notes.txt", "data_b64": "aGVsbG8="},
    }
    assert result["attachments"] == [
        {"id": "att-1", "display_name": "report.txt", "filename": "server-report.txt", "size": 5, "created_ts": 8}
    ]
    assert result["session"]["pending_attachment"] is True
    assert result["badge"] == {"text": "1", "display": "inline-flex"}
    assert result["tray"] == {"display": "flex", "chips": 2}
    assert result["toasts"][-1] == "file staged"


def test_send_cleanup_clears_selected_attachment_projection() -> None:
    result = run_attachments(
        """
        controller.setStagedAttachments([{ id: "att-1", display_name: "report.txt", size: 5 }]);
        const cleared = controller.setSelectedSessionPendingAttachment("sid", false);
        const badge = attachBtn.children[0];
        process.stdout.write(JSON.stringify({
          cleared,
          attachments: controller.getStagedAttachments(),
          session: sessions.get("sid"),
          badge: { text: badge.textContent, display: badge.style.display },
          trayDisplay: tray.style.display,
        }));
        """
    )
    assert result == {
        "cleared": True,
        "attachments": [],
        "session": {"session_id": "sid", "launch_state": "ready", "pending_attachment": False, "staged_attachments": []},
        "badge": {"text": "", "display": "none"},
        "trayDisplay": "none",
    }
