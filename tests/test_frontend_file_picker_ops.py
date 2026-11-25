from frontend_module_loader import module_path
import json
import subprocess
import textwrap


FILE_PICKER_OPS_SOURCE = module_path("app_file_picker_ops.js").read_text(encoding="utf-8")


BIND_TEMPLATE = """
        const vm = require("node:vm");
        const ctx = { window: {}, console };
        vm.createContext(ctx);
        vm.runInContext(__SOURCE__, ctx);

        class Element { closest() { return null; } }
        const node = (key) => ({ __key: key, addEventListener() {} });
        const nodes = new Proxy({}, { get(store, key) { return store[key] || (store[key] = node(key)); } });
        const handlers = {};
        const noop = () => {};
        let documentClick = null;
        let closeCalls = 0;
        const calls = [];
        const fileDownloadRuntime = __DOWNLOAD_RUNTIME__;
        ctx.window.CodoxearFilePickerOps.bindFilePickerInteractions({
          eventBindings: { on(target, type, handler) { if (target && target.__key) handlers[target.__key + ":" + type] = handler; } },
          fileBtn: nodes.fileBtn,
          showFileViewer: noop,
          filePickerInput: nodes.filePickerInput,
          filePickerInputRuntime: { focus: noop, click: noop, input: noop, blur: noop, keydown: noop },
          fileModeDiffBtn: nodes.fileModeDiffBtn,
          fileModePreviewBtn: nodes.fileModePreviewBtn,
          fileEditBtn: nodes.fileEditBtn,
          handleFileDiffModeButtonPress: noop,
          handleFilePreviewModeButtonPress: noop,
          handleFileEditButtonPress: noop,
          fileMediaPreviewBtn: nodes.fileMediaPreviewBtn,
          fileMediaPreviewRuntime: { handleButtonPress: noop },
          fileDownloadBtn: nodes.fileDownloadBtn,
          fileDownloadRuntime,
          activeFileDownloadApiPath: () => "/api/sessions/s/file/download?path=src%2Fnotes.md",
          codoxearFileViewer: { bindFileTouchPress: noop, bindFileTouchClick: noop },
          fileTouchSelectBtn: nodes.fileTouchSelectBtn,
          fileTouchCopyBtn: nodes.fileTouchCopyBtn,
          fileTouchPasteBtn: nodes.fileTouchPasteBtn,
          fileTouchUpBtn: nodes.fileTouchUpBtn,
          fileTouchLeftBtn: nodes.fileTouchLeftBtn,
          fileTouchDownBtn: nodes.fileTouchDownBtn,
          fileTouchRightBtn: nodes.fileTouchRightBtn,
          toggleFileTouchSelectionMode: noop,
          copyActiveFileSelection: noop,
          pasteFromClipboardIntoActiveFile: noop,
          handleFileTouchMoveButtonPress: noop,
          fileCloseBtn: nodes.fileCloseBtn,
          fileBackdrop: nodes.fileBackdrop,
          requestHideFileViewer: noop,
          $: () => node("query"),
          fileUnsavedController: { handleFileUnsavedSaveChoice: noop, handleFileUnsavedDiscardChoice: noop, handleFileUnsavedCancelChoice: noop },
          fileUnsavedBackdrop: nodes.fileUnsavedBackdrop,
          filePasteInput: { value: "" },
          handleFilePasteInsert: noop,
          hideFilePasteDialog: noop,
          filePasteBackdrop: nodes.filePasteBackdrop,
          chatInner: node("chatInner"),
          codeBlockCopyRuntime: { handleClick: () => false },
          fileReferenceRuntime: { handleClick: noop },
          fileDiff: node("fileDiff"),
          addAppEvent(_target, type, handler) { if (type === "click") documentClick = handler; },
          document: {},
          Element,
          isFileViewerOpen: () => true,
          menuState: { isOpen: () => true },
          closeFilePickerMenu(options) { if (options && options.restoreInput === true) closeCalls += 1; },
        });
__TAIL__
"""


def _bind_script(download_runtime_js: str, tail_js: str) -> str:
    return (
        BIND_TEMPLATE.replace("__SOURCE__", json.dumps(FILE_PICKER_OPS_SOURCE))
        .replace("__DOWNLOAD_RUNTIME__", download_runtime_js)
        .replace("__TAIL__", textwrap.dedent(tail_js))
    )


def _run(script: str) -> dict:
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def test_document_click_closes_open_picker_menu_without_throwing() -> None:
    script = _bind_script(
        download_runtime_js="({ download: noop, downloadActive: noop })",
        tail_js="""
        const target = new Element();
        let error = null;
        try { documentClick({ target }); } catch (caught) { error = caught && caught.message ? caught.message : String(caught); }
        process.stdout.write(JSON.stringify({ closeCalls, error }));
        """,
    )
    assert _run(script) == {"closeCalls": 1, "error": None}


def test_download_button_asks_the_runtime_for_the_active_file() -> None:
    script = _bind_script(
        download_runtime_js="""({
          download: () => { calls.push({ kind: "download" }); return true; },
          downloadActive: (apiPath) => { calls.push({ kind: "active", apiPath }); return Promise.resolve({ ok: true }); },
        })""",
        tail_js="""
        const handler = handlers["fileDownloadBtn:click"];
        const prevented = [];
        handler({ preventDefault: () => prevented.push("preventDefault"), stopPropagation: () => prevented.push("stopPropagation") });
        process.stdout.write(JSON.stringify({ calls, prevented, hasHandler: typeof handler === "function" }));
        """,
    )
    result = _run(script)
    assert result["hasHandler"] is True
    assert result["prevented"] == ["preventDefault", "stopPropagation"]
    # The caller supplies the path (the viewer owns it); the runtime owns the
    # strategy and the filename, so the streaming entry point is not used directly.
    assert result["calls"] == [
        {"kind": "active", "apiPath": "/api/sessions/s/file/download?path=src%2Fnotes.md"}
    ]
