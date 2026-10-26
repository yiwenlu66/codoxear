from frontend_module_loader import module_path
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_ATTACHMENTS_JS = module_path("app_attachments.js")
APP_SESSION_CATALOG_JS = module_path("app_session_catalog.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")


def run_attachments(body: str) -> dict:
    source = APP_ATTACHMENTS_JS.read_text(encoding="utf-8")
    state_source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
    catalog_source = APP_SESSION_CATALOG_JS.read_text(encoding="utf-8")
    refresh_source = module_path("app_session_refresh.js").read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const listeners = new Map();
        const scheduled = [];
        class FakeAbortController {{
          constructor() {{ this.signal = {{ aborted: false, listeners: [] }}; }}
          abort() {{ this.signal.aborted = true; this.signal.listeners.slice().forEach((fn) => fn()); }}
        }}
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
          setTimeout: (fn, ms) => {{ const timer = {{ fn, ms, cleared: false }}; scheduled.push(timer); return scheduled.length; }},
          clearTimeout: (id) => {{ const timer = scheduled[id - 1]; if (timer) timer.cleared = true; }},
          AbortController: FakeAbortController,
          URL: {{ createObjectURL: (file) => `blob:thumb-${{file && file.name ? file.name : "x"}}`, revokeObjectURL: () => {{}} }},
          Uint8Array, console,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(state_source)}, ctx, {{ filename: "app_session_state.js" }});
        vm.runInContext({json.dumps(catalog_source)}, ctx, {{ filename: "app_session_catalog.js" }});
        vm.runInContext({json.dumps(source)}, ctx, {{ filename: "app_attachments.js" }});
        vm.runInContext({json.dumps(refresh_source)}, ctx, {{ filename: "app_session_refresh.js" }});
        function el(tag, attrs = {{}}) {{
          const out = node(tag);
          out.tag = tag;
          for (const [key, value] of Object.entries(attrs)) {{
            if (key === "class") out.className = value;
            else if (key === "text") out.textContent = value;
            else {{
              out.attrs[key] = String(value);
              if (key === "title") out.title = value;
            }}
          }}
          return out;
        }}
        const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
        sessionState.set("selected", selected);
        const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
        sessionCatalog.set("latestSessions", Array.from(sessions.values()));
        let apiImpl = async () => ({{ ok: true, attachment: {{ id: "att-1" }}, attachments: [
          {{ id: "att-1", display_name: "report.txt", filename: "server-report.txt", size: "5", created_ts: "8" }},
          {{ id: "att-1", display_name: "duplicate.txt", size: 10 }},
          {{ id: "", display_name: "invalid.txt" }},
        ] }});
        const controller = ctx.window.CodoxearAttachments.createAttachmentsController({{
          attachBtn, imgInput, composer, textarea, sessionState, sessionCatalog,
          sessionLaunchFailed: (info) => !!(info && info.launch_state === "failed"),
          sessionHasUnknownSend: (info) => !!(info && info.commit_unknown_send),
          sessionIsOrphanRecovery: (info) => !!(info && info.orphan_recovery),
          sessionHasOrphanQueueRecovery: (info) => !!(info && info.queue_recovery),
          api: async (path, options) => {{
            apiCalls.push({{ path, body: options && options.body, rawName: options && options.rawBody && options.rawBody.name, headers: options && options.headers, cache: options && options.cache }});
            return apiImpl(path, options);
          }},
          setToast: (text) => toasts.push(text), handleAppAuthLoss: () => toasts.push("auth-loss"),
          refreshSessions: async () => pollCalls.push("refresh"),
          setPollFastUntilMs: (value) => pollCalls.push(["fast", value]), kickPoll: (value) => pollCalls.push(["kick", value]),
          resizeComposer: () => pollCalls.push("resize"), getTray: () => tray, el,
          fmtBytes: (bytes) => `${{bytes}} B`, safeAttachmentStem: (name) => name,
          isLikelyHeic: () => false, looksLikeImage: (file) => String(file && file.type || "").startsWith("image/"),
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
    except subprocess.CalledProcessError as exc:
        print("NODE STDERR:\n" + (exc.stderr or ""))
        print("NODE STDOUT:\n" + (exc.stdout or ""))
        raise
    finally:
        Path(script_file.name).unlink(missing_ok=True)
    return json.loads(completed.stdout)


def test_stage_files_normalizes_dedupes_projects_badge_and_uploads_raw_blob() -> None:
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
    assert result["upload"]["path"] == "/api/sessions/sid/inject_file"
    assert result["upload"]["rawName"] == "notes.txt"
    assert "body" not in result["upload"]
    assert result["upload"]["headers"]["Content-Type"] == "application/octet-stream"
    assert result["upload"]["headers"]["X-Codoxear-Filename"] == "notes.txt"
    assert result["upload"]["headers"]["X-Codoxear-Upload-Id"]
    assert result["attachments"] == [
        {"id": "att-1", "display_name": "report.txt", "filename": "server-report.txt", "size": 5, "created_ts": 8}
    ]
    assert result["session"]["pending_attachment"] is True
    assert result["badge"] == {"text": "1", "display": "inline-flex"}
    assert result["tray"] == {"display": "flex", "chips": 2}
    assert result["toasts"][-1] == "file staged"


def test_attachment_button_follows_selected_and_sending_store_updates() -> None:
    result = run_attachments(
        """
        sessions.set("other", { session_id: "other", launch_state: "ready" });
        sessionState.set("sending", true);
        const disabledWhileSending = attachBtn.disabled;
        sessionState.set("selected", "other");
        const disabledAfterSwitch = attachBtn.disabled;
        sessionState.set("sending", false);
        const enabledAfterSend = attachBtn.disabled;
        controller.dispose();
        sessionState.set("sending", true);
        process.stdout.write(JSON.stringify({ disabledWhileSending, disabledAfterSwitch, enabledAfterSend, disabledAfterDispose: attachBtn.disabled }));
        """
    )
    assert result == {
        "disabledWhileSending": True,
        "disabledAfterSwitch": True,
        "enabledAfterSend": False,
        "disabledAfterDispose": False,
    }


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


def test_pending_chip_shows_synchronously_and_swaps_to_staged_with_thumbnail() -> None:
    result = run_attachments(
        """
        apiImpl = (path, options) => new Promise((resolve) => {
          setTimeout(() => resolve({ ok: true, attachment: { id: "att-9" }, attachments: [
            { id: "att-9", display_name: decodeURIComponent(options.headers["X-Codoxear-Filename"]), filename: decodeURIComponent(options.headers["X-Codoxear-Filename"]), size: 3, created_ts: 1 },
          ] }), 0);
        });
        const file = { name: "shot.png", size: 3, type: "image/png", arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer };
        const stagedPromise = controller.stageFiles([file], { sid: "sid", source: "paste" });
        const pendingChip = tray.children[0];
        const syncState = {
          chips: tray.children.length,
          trayDisplay: tray.style.display,
          pending: pendingChip.attrs["data-pending"] || "",
          thumb: pendingChip.children[0] && pendingChip.children[0].attrs ? pendingChip.children[0].attrs.src || "" : "",
          phase: pendingChip.children.map((c) => c.className).join(","),
          badge: attachBtn.children[0].textContent,
          hasPending: controller.hasPendingUploads(),
          uploadCalls: apiCalls.length,
        };
        stagedPromise.then((staged) => {
          const stagedChip = tray.children[0];
          process.stdout.write(JSON.stringify({
            syncState,
            staged,
            finalChips: tray.children.length,
            fresh: stagedChip.attrs["data-fresh"] || "",
            finalPending: stagedChip.attrs["data-pending"] || "",
            finalThumb: stagedChip.children[0] && stagedChip.children[0].attrs ? stagedChip.children[0].attrs.src || "" : "",
            finalBadge: attachBtn.children[0].textContent,
            hasPendingAfter: controller.hasPendingUploads(),
            toasts,
          }));
        });
        """
    )
    assert result["syncState"] == {
        "chips": 1,
        "trayDisplay": "flex",
        "pending": "1",
        "thumb": "blob:thumb-shot.png",
        "phase": "stagedAttachmentThumb,stagedAttachmentName,stagedAttachmentPhase,stagedAttachmentRemove",
        "badge": "1",
        "hasPending": True,
        "uploadCalls": 0,
    }
    assert result["staged"] is True
    assert result["finalChips"] == 2  # staged chip + Clear button
    assert result["fresh"] == "1"
    assert result["finalPending"] == ""
    assert result["finalThumb"] == "blob:thumb-shot.png"
    assert result["finalBadge"] == "1"
    assert result["hasPendingAfter"] is False
    assert result["toasts"] == ["file staged"]


def test_unknown_upload_keeps_visible_reason_and_retry_cancel_without_expiry() -> None:
    result = run_attachments(
        """
        apiImpl = async () => { throw new Error("file too large (max 1.0 KB)"); };
        const file = { name: "big.png", size: 5, type: "image/png", arrayBuffer: async () => new Uint8Array([1, 2, 3, 4, 5]).buffer };
        controller.stageFiles([file], { sid: "sid", source: "paste" }).then((staged) => {
          const failedChip = tray.children.find((c) => c.attrs["data-failed"]);
          const failureTimers = scheduled.filter((t) => t.ms === 2000 && !t.cleared);
          failureTimers.forEach((t) => t.fn());
          process.stdout.write(JSON.stringify({
            staged,
            failedShown: Boolean(failedChip),
            failTitle: failedChip ? failedChip.title : "",
            visibleReason: failedChip && failedChip.children.find((c) => c.className === "stagedAttachmentPhase").textContent,
            retry: Boolean(failedChip && failedChip.children.find((c) => c.textContent === "Retry")),
            failureTimers: failureTimers.length,
            hasPendingWhileFailed: controller.hasPendingUploads(),
            chipsAfterTimer: tray.children.length,
            trayDisplayAfterTimer: tray.style.display,
            toasts,
          }));
        });
        """
    )
    assert result["staged"] is False
    assert result["failedShown"] is True
    assert result["failTitle"] == "file too large (max 1.0 KB)"
    assert "Upload outcome unknown: file too large" in result["visibleReason"]
    assert result["retry"] is True
    assert result["failureTimers"] == 0
    assert result["hasPendingWhileFailed"] is True
    assert result["chipsAfterTimer"] == 1
    assert result["trayDisplayAfterTimer"] == "flex"
    assert result["toasts"] == ["attach error: big.png: file too large (max 1.0 KB)"]


def test_cancel_pending_upload_aborts_and_clears_without_error_toast() -> None:
    result = run_attachments(
        """
        apiImpl = (path, options) => path.endsWith("upload-cancel")
          ? Promise.resolve({ ok: true, state: "cancelled", attachments: [], pending_attachment: false, attachments_revision: 2 })
          : new Promise((resolve, reject) => {
          options.signal.listeners.push(() => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
        });
        const file = { name: "shot.png", size: 3, type: "image/png", arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer };
        const stagedPromise = controller.stageFiles([file], { sid: "sid", source: "paste" });
        const pendingChip = tray.children[0];
        const cancelBtn = pendingChip.children.find((c) => c.className === "stagedAttachmentRemove");
        const hasBefore = controller.hasPendingUploads();
        const cancelled = cancelBtn.onclick();
        Promise.all([stagedPromise, cancelled]).then(([staged]) => {
          process.stdout.write(JSON.stringify({
            staged,
            hasBefore,
            hasAfter: controller.hasPendingUploads(),
            chips: tray.children.length,
            trayDisplay: tray.style.display,
            refreshed: pollCalls.filter((c) => c === "refresh").length,
            toasts,
          }));
        });
        """
    )
    assert result["staged"] is False
    assert result["hasBefore"] is True
    assert result["hasAfter"] is False
    assert result["chips"] == 0
    assert result["trayDisplay"] == "none"
    assert result["refreshed"] >= 1
    assert result["toasts"] == []


def test_session_switch_aborts_pending_upload_without_touching_old_session() -> None:
    result = run_attachments(
        """
        apiImpl = (path, options) => path.endsWith("upload-cancel")
          ? Promise.resolve({ ok: true, state: "cancelled", attachments: [], pending_attachment: false, attachments_revision: 2 })
          : new Promise((resolve, reject) => {
          options.signal.listeners.push(() => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
        });
        sessions.set("other", { session_id: "other", launch_state: "ready" });
        const file = { name: "shot.png", size: 3, type: "image/png", arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer };
        const stagedPromise = controller.stageFiles([file], { sid: "sid", source: "paste" });
        const chipsBeforeSwitch = tray.children.length;
        sessionState.set("selected", "other");
        stagedPromise.then((staged) => {
          process.stdout.write(JSON.stringify({
            staged,
            chipsBeforeSwitch,
            chips: tray.children.length,
            trayDisplay: tray.style.display,
            hasAfter: controller.hasPendingUploads(),
            oldSession: sessions.get("sid"),
            refreshed: pollCalls.filter((c) => c === "refresh").length,
            toasts,
          }));
        });
        """
    )
    assert result["staged"] is False
    assert result["chipsBeforeSwitch"] == 1
    assert result["chips"] == 0
    assert result["trayDisplay"] == "none"
    assert result["hasAfter"] is False
    assert result["oldSession"] == {"session_id": "sid", "launch_state": "ready"}
    assert result["refreshed"] == 0
    assert result["toasts"] == []
