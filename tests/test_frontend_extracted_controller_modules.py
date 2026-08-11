from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"


def run_controller_harness() -> dict:
    sources = {
        name: module_path(name).read_text(encoding="utf-8")
        for name in (
            "app_session_title.js",
            "app_file_unsaved.js",
            "app_dialog_menu.js",
            "app_ios_viewport.js",
        )
    }
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const sources = __SOURCES__;
        class ElementStub {}
        function node() {
          const listeners = {};
          return Object.assign(new ElementStub(), {
            style: {}, attributes: {}, tabIndex: 0,
            setAttribute(name, value) { this.attributes[name] = String(value); },
            removeAttribute(name) { delete this.attributes[name]; },
            addEventListener(name, callback) { listeners[name] = callback; },
            emit(name, event = {}) { return listeners[name](event); },
          });
        }
        const ctx = {
          window: {},
          HTMLElement: ElementStub,
          Date,
        };
        vm.createContext(ctx);
        for (const source of Object.values(sources)) vm.runInContext(source, ctx);

        let selected = null;
        const title = node();
        const opened = [];
        const titleController = ctx.window.CodoxearSessionTitle.createSessionTitleController({
          titleLabel: title,
          getSelected: () => selected,
          openEditSession: (sessionId) => opened.push(sessionId),
        });
        const disabledTitle = {
          cursor: title.style.cursor,
          tabIndex: title.tabIndex,
          disabled: title.attributes["aria-disabled"],
        };
        selected = "s-1";
        titleController.syncTitleEditState();
        let prevented = false;
        title.onkeydown({ key: "Enter", preventDefault() { prevented = true; } });

        const activeElement = node();
        const dialogCalls = [];
        const viewerCalls = [];
        const unsavedController = ctx.window.CodoxearFileUnsaved.createFileUnsavedController({
          documentTarget: { activeElement },
          ElementCtor: ElementStub,
          dialogRuntime: {
            promptChoice: (active, ctor) => { dialogCalls.push({ active: active === activeElement, ctor: ctor === ElementStub }); return Promise.resolve("discard"); },
            hide: (choice) => { dialogCalls.push({ hide: choice }); return true; },
          },
          getFileViewerController: () => ({
            maybeHandleUnsavedFileChanges: () => { viewerCalls.push("maybe"); return Promise.resolve(false); },
            handleFileUnsavedSaveChoice: () => { viewerCalls.push("save"); return true; },
            handleFileUnsavedDiscardChoice: () => { viewerCalls.push("discard"); return true; },
            handleFileUnsavedCancelChoice: () => { viewerCalls.push("cancel"); return true; },
          }),
        });

        const host = { getBoundingClientRect: () => ({ left: 0, top: 0, width: 500 }) };
        const menu = { parentElement: host, scrollHeight: 260, style: {} };
        const anchor = { getBoundingClientRect: () => ({ left: 20, top: 460, bottom: 490, width: 100 }) };
        const dialogMenuController = ctx.window.CodoxearDialogMenu.createDialogMenuController({
          windowTarget: { innerHeight: 600, visualViewport: null },
        });
        dialogMenuController.positionDialogMenu(menu, anchor);

        let now = 1000;
        const scheduled = [];
        const textarea = node();
        const alternateEntry = node();
        const visualViewport = node();
        const documentTarget = {
          activeElement: textarea,
          documentElement: { scrollTop: 21 },
          body: { scrollTop: 21 },
        };
        const viewportCalls = [];
        const iosController = ctx.window.CodoxearIOSViewport.createIOSViewportController({
          windowTarget: {
            innerHeight: 700,
            scrollY: 21,
            scrollTo: () => viewportCalls.push("scroll-top"),
            visualViewport,
          },
          documentTarget,
          navigatorTarget: { userAgent: "iPhone", platform: "iPhone", maxTouchPoints: 1 },
          textarea,
          isTextEntryElement: (element) => element === textarea || element === alternateEntry,
          updateAppHeightVar: () => viewportCalls.push("height"),
          transcriptScrollRuntime: {
            isNearBottom: () => true,
            enableAutoScroll: () => viewportCalls.push("enable-auto"),
            syncJumpButton: () => viewportCalls.push("sync-jump"),
            shouldAutoScrollOrNearBottom: () => true,
            scrollToBottom: () => viewportCalls.push("bottom"),
            scheduleScrollToBottom: () => viewportCalls.push("scheduled-bottom"),
          },
          addAppEvent: (target, name, callback) => target.addEventListener(name, callback),
          requestAnimationFrame: (callback) => callback(),
          setTimeout: (callback, delay) => { scheduled.push({ callback, delay }); return scheduled.length; },
          clearTimeout: () => viewportCalls.push("clear"),
          now: () => now,
        });
        textarea.emit("focus");
        const guardActiveAfterFocus = iosController.isIOSViewportGuardActive();
        documentTarget.activeElement = alternateEntry;
        visualViewport.emit("resize");
        const guardActiveAfterAlternateFocus = iosController.isIOSViewportGuardActive();

        Promise.all([
          unsavedController.promptFileUnsavedChoice(),
          unsavedController.maybeHandleUnsavedFileChanges(),
        ]).then(([choice]) => {
          unsavedController.hideFileUnsavedDialog("save");
          unsavedController.handleFileUnsavedSaveChoice();
          unsavedController.handleFileUnsavedDiscardChoice();
          unsavedController.handleFileUnsavedCancelChoice();
          process.stdout.write(JSON.stringify({
            title: { disabledTitle, enabled: { cursor: title.style.cursor, tabIndex: title.tabIndex, role: title.attributes.role }, opened, prevented },
            unsaved: { choice, dialogCalls, viewerCalls },
            menu: { top: menu.style.top, maxHeight: menu.style.maxHeight, left: menu.style.left },
            ios: { isIOS: iosController.isIOS(), guardActiveAfterFocus, guardActiveAfterAlternateFocus, viewportCalls, scheduledDelays: scheduled.map((item) => item.delay) },
          }));
        });
        """
    ).replace("__SOURCES__", json.dumps(sources))
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(completed.stdout)


def test_extracted_controller_behavior() -> None:
    result = run_controller_harness()

    assert result["title"]["disabledTitle"] == {"cursor": "default", "tabIndex": -1, "disabled": "true"}
    assert result["title"]["enabled"] == {"cursor": "pointer", "tabIndex": 0, "role": "button"}
    assert result["title"]["opened"] == ["s-1"]
    assert result["title"]["prevented"] is True

    assert result["unsaved"] == {
        "choice": "discard",
        "dialogCalls": [{"active": True, "ctor": True}, {"hide": "save"}],
        "viewerCalls": ["maybe", "save", "discard", "cancel"],
    }

    assert result["menu"] == {"top": "192px", "maxHeight": "440px", "left": "20px"}

    assert result["ios"]["isIOS"] is True
    assert result["ios"]["guardActiveAfterFocus"] is True
    assert result["ios"]["guardActiveAfterAlternateFocus"] is False
    assert result["ios"]["scheduledDelays"] == [50]
    assert result["ios"]["viewportCalls"] == [
        "enable-auto",
        "sync-jump",
        "height",
        "scroll-top",
        "bottom",
        "height",
        "clear",
    ]


def run_file_viewer_integration_harness() -> dict:
    source = (module_path("app_file_viewer_integration.js")).read_text(encoding="utf-8")
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const context = { window: {} };
        vm.createContext(context);
        vm.runInContext(__SOURCE__, context, { filename: "app_file_viewer_integration.js" });

        let launchFailed = true;
        const calls = [];
        const integration = context.window.CodoxearFileViewerIntegration.createFileViewerIntegration({
          selectedSessionLaunchFailed: () => launchFailed,
          setToast: (message) => calls.push(["toast", message]),
          lifecycleRuntime: {
            show: (options) => { calls.push(["show", options]); return Promise.resolve("shown"); },
            hide: () => { calls.push(["hide"]); return "hidden"; },
          },
          fileViewerController: {
            setFileEditMode: (mode) => { calls.push(["edit", mode]); return mode; },
            handleFileTouchSelectionKeydown: (event) => { calls.push(["touch", event.key]); return event.key; },
          },
          fileLoadResultRuntime: {
            apply: (rel, result, request, options) => {
              calls.push(["load", rel, result, request, options]);
              return Promise.resolve("loaded");
            },
          },
          confirmAction: (options) => { calls.push(["confirm", options]); return Promise.resolve(true); },
          fileUnsavedController: {
            promptFileUnsavedChoice: () => { calls.push(["unsaved"]); return Promise.resolve("discard"); },
          },
        });

        const blocked = integration.openFileViewer();
        launchFailed = false;
        Promise.all([
          integration.openFileViewer({ path: "notes.txt", mode: "preview", line: 4, pickerQuery: "notes" }),
          integration.applyFileLoadResult("notes.txt", { text: "x" }, "request-1", { viewMode: "diff" }),
          integration.confirmReload("Discard draft?"),
          integration.promptUnsavedFileChoice(),
        ]).then(([opened, loaded, confirmed, unsaved]) => {
          const closed = integration.closeFileViewer();
          const edit = integration.setFileEditMode(true);
          const touch = integration.handleFileTouchSelectionKeydown({ key: "ArrowDown" });
          process.stdout.write(JSON.stringify({ blocked, opened, loaded, confirmed, unsaved, closed, edit, touch, calls }));
        });
        """
    ).replace("__SOURCE__", json.dumps(source))
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(completed.stdout)


def test_file_viewer_integration_behavior() -> None:
    result = run_file_viewer_integration_harness()

    assert result == {
        "blocked": False,
        "opened": "shown",
        "loaded": "loaded",
        "confirmed": True,
        "unsaved": "discard",
        "closed": "hidden",
        "edit": True,
        "touch": "ArrowDown",
        "calls": [
            ["toast", "failed launch has no file browser"],
            ["show", {"path": "notes.txt", "mode": "preview", "line": 4, "pickerQuery": "notes"}],
            ["load", "notes.txt", {"text": "x"}, "request-1", {"viewMode": "diff"}],
            [
                "confirm",
                {
                    "title": "Reload file from disk?",
                    "message": "Discard draft?",
                    "confirmText": "Reload",
                    "cancelText": "Cancel",
                    "destructive": True,
                },
            ],
            ["unsaved"],
            ["hide"],
            ["edit", True],
            ["touch", "ArrowDown"],
        ],
    }
