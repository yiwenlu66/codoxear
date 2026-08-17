from frontend_module_loader import module_path
import json
import subprocess
import textwrap


FILE_PICKER_OPS_SOURCE = module_path("app_file_picker_ops.js").read_text(encoding="utf-8")


def test_document_click_closes_open_picker_menu_without_throwing() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(FILE_PICKER_OPS_SOURCE)}, ctx);

        class Element {{
          closest() {{ return null; }}
        }}
        const target = new Element();
        const node = () => ({{ addEventListener() {{}} }});
        const nodes = new Proxy({{}}, {{ get(store, key) {{ return store[key] || (store[key] = node()); }} }});
        let documentClick = null;
        let closeCalls = 0;
        const noop = () => {{}};
        ctx.window.CodoxearFilePickerOps.bindFilePickerInteractions({{
          eventBindings: {{ on() {{}} }},
          fileBtn: nodes.fileBtn,
          showFileViewer: noop,
          filePickerInput: nodes.filePickerInput,
          filePickerInputRuntime: {{ focus: noop, click: noop, input: noop, blur: noop, keydown: noop }},
          fileModeDiffBtn: nodes.fileModeDiffBtn,
          fileModePreviewBtn: nodes.fileModePreviewBtn,
          fileEditBtn: nodes.fileEditBtn,
          handleFileDiffModeButtonPress: noop,
          handleFilePreviewModeButtonPress: noop,
          handleFileEditButtonPress: noop,
          fileVideoPreviewBtn: nodes.fileVideoPreviewBtn,
          fileVideoPreviewRuntime: {{ handleButtonPress: noop }},
          fileDownloadBtn: nodes.fileDownloadBtn,
          fileDownloadRuntime: {{ download: noop }},
          activeFileDownloadApiPath: () => "",
          codoxearFileViewer: {{ bindFileTouchPress: noop, bindFileTouchClick: noop }},
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
          $: () => node(),
          fileUnsavedController: {{ handleFileUnsavedSaveChoice: noop, handleFileUnsavedDiscardChoice: noop, handleFileUnsavedCancelChoice: noop }},
          fileUnsavedBackdrop: nodes.fileUnsavedBackdrop,
          filePasteInput: {{ value: "" }},
          handleFilePasteInsert: noop,
          hideFilePasteDialog: noop,
          filePasteBackdrop: nodes.filePasteBackdrop,
          chatInner: node(),
          codeBlockCopyRuntime: {{ handleClick: () => false }},
          fileReferenceRuntime: {{ handleClick: noop }},
          fileDiff: node(),
          addAppEvent(_target, type, handler) {{ if (type === "click") documentClick = handler; }},
          document: {{}},
          Element,
          isFileViewerOpen: () => true,
          menuState: {{ isOpen: () => true }},
          closeFilePickerMenu(options) {{
            if (options && options.restoreInput === true) closeCalls += 1;
          }},
        }});
        let error = null;
        try {{ documentClick({{ target }}); }} catch (caught) {{ error = caught && caught.message ? caught.message : String(caught); }}
        process.stdout.write(JSON.stringify({{ closeCalls, error }}));
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)

    assert result == {"closeCalls": 1, "error": None}
