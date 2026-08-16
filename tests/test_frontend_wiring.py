from frontend_module_loader import module_path
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"


def test_wiring_factories_preserve_explicit_controller_dependencies() -> None:
    sources = {
        name: module_path(name).read_text(encoding="utf-8")
        for name in ("app_wiring.js", "app_application_composition.js")
    }
    session_display_keys = [
        "getSelected", "setRunning", "getQueueLen", "setQueueLen",
        "getAttachmentsController", "updateQueueBadge", "setToast", "statusChip",
        "interruptBtn", "ctxChip", "eventBindings",
    ]
    file_ops_keys = [
        "getSelected", "getSessionIndex", "getSessionLifecycleController", "wiring", "document", "window", "HTMLElement", "requestAnimationFrame", "setTimeout", "$", "el", "iconSvg", "resolveAppUrl", "api", "setToast", "confirmApp", "addAppEvent", "sessionLaunchFailed", "normalizeLineNumber", "markdownPreviewHtml", "blockedFileMessage", "listFromFilesField", "listFromFileRecords", "baseName", "codoxearFilePicker", "codoxearFilePickerOps", "codoxearFileViewer", "codoxearFileEditor", "codoxearFileEditorOps", "codoxearFileEditMode", "codoxearFileTouch", "codoxearDialogMenus", "prepareModalOpen", "afterModalVisibilityChanged", "focusModalCloseButton", "restoreModalFocus", "isModalTargetOpen", "newSessionDialogController", "eventBindings", "codoxearFileHelpers", "copyToClipboard", "dialogMenuController", "duplicateFilePickerPaths", "editCloseBtn", "editDependencyBtn", "editDependencyMenu", "editNameInput", "editPriorityRange", "editPriorityResetBtn", "editPriorityValue", "editSaveBtn", "editSnoozeCustomDate", "editSnoozeCustomRow", "editSnoozeCustomTime", "editSnoozeModeButtons", "editStatus", "editViewer", "fileBtn", "filePickerIdentityHint", "filePickerSectionLabel", "filePickerTitle", "fmtBytes", "formatPriorityOffset", "handleAppAuthLoss", "isDiffableFileKind", "isMarkdownPreviewable", "isTextEntryElement", "isTextFileKind", "modalIsolationTargets", "normalizeDraftFilePath", "parseLocalFileRef", "rawByteDuplicatePaths", "refreshSessions", "selectedSessionLaunchFailed", "sessionDisplayName", "sessionTitleWithId", "setPickerButtonContent", "storageGetItem", "storageSetItem", "stripPathLocationSuffix", "titleLabel", "useTouchFileEditorControls", "filePickerField", "filePickerMenu", "filePickerInput", "fileStatus", "fileDiff", "fileImage", "fileVideo", "fileVideoPreviewBtn", "fileTouchToolbar", "fileTouchActions", "fileTouchDpad", "fileTouchCopyBtn", "fileTouchPasteBtn", "fileTouchSelectBtn", "fileTouchUpBtn", "fileTouchLeftBtn", "fileTouchDownBtn", "fileTouchRightBtn", "fileModeDiffBtn", "fileModePreviewBtn", "fileDownloadBtn", "fileBackdrop", "fileViewer", "fileCloseBtn", "fileUnsavedBackdrop", "fileUnsavedDialog", "filePasteBackdrop", "filePasteDialog", "filePasteInput", "fileEditBtn", "chatInner", "codeBlockCopyRuntime", "appConfirm", "appConfirmFocusableControls", "resolveAppConfirm", "sendChoice", "closeSendChoiceDialog", "queueViewer", "hideQueueViewer", "helpViewer", "hideHelpViewer", "diagViewer", "hideDiagViewer", "voiceController", "hideVoiceSettingsDialog",
    ]
    program = """
const vm = require("vm");
const sources = __SOURCES__;
const sessionDisplayKeys = __SESSION_DISPLAY_KEYS__;
const fileOpsKeys = __FILE_OPS_KEYS__;
const context = { window: {} };
vm.createContext(context);
for (const source of Object.values(sources)) vm.runInContext(source, context);
const wiring = context.window.CodoxearWiring.createWiring();
const selected = () => "s-1";
const flow = wiring.createMessageFlowOptions({
  getSelected: selected,
  api: () => null,
  accidental: "must not reach controller",
});
const lifecycle = wiring.createSessionLifecycleOptions({
  getSelected: selected,
  setSelected: () => null,
  clearDeletedSessionClientState: "not a lifecycle dependency",
});
const sessionDisplayInput = Object.fromEntries(sessionDisplayKeys.map((key) => [key, key]));
const fileOpsInput = Object.fromEntries(fileOpsKeys.map((key) => [key, key]));
const sessionDisplay = wiring.createSessionDisplayOptions({ ...sessionDisplayInput, accidental: "must not reach controller" });
const fileOps = wiring.createFileOpsOptions({ ...fileOpsInput, accidental: "must not reach controller" });
const listeners = [];
const target = {
  addEventListener(type, handler, options) { listeners.push(["add", type, handler, options]); },
  removeEventListener(type, handler, options) { listeners.push(["remove", type, handler, options]); },
};
const registered = [];
const events = context.window.CodoxearEventBindings.createEventBindings({
  addEvent(target, type, handler, options) {
    registered.push([target, type, handler, options]);
    target.addEventListener(type, handler, options);
    return handler;
  },
});
const handler = () => "clicked";
const bound = events.onClick(target, handler, { capture: true });
process.stdout.write(JSON.stringify({
  flowKeys: Object.keys(flow).sort(),
  flowSelected: flow.getSelected(),
  lifecycleKeys: Object.keys(lifecycle).sort(),
  sessionDisplayKeys: Object.keys(sessionDisplay).sort(),
  sessionDisplayValues: Object.values(sessionDisplay).sort(),
  fileOpsKeys: Object.keys(fileOps).sort(),
  fileOpsValues: Object.values(fileOps).sort(),
  event: { registered: registered.length, type: listeners[0][1], sameHandler: bound === handler, capture: listeners[0][3].capture },
}));
""".replace("__SOURCES__", json.dumps(sources)).replace("__SESSION_DISPLAY_KEYS__", json.dumps(session_display_keys)).replace("__FILE_OPS_KEYS__", json.dumps(file_ops_keys))
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    result = json.loads(completed.stdout)

    assert result["flowKeys"] == ["api", "getSelected"]
    assert result["flowSelected"] == "s-1"
    assert result["lifecycleKeys"] == ["getSelected", "setSelected"]
    assert result["sessionDisplayKeys"] == sorted(session_display_keys)
    assert result["sessionDisplayValues"] == sorted(session_display_keys)
    assert result["fileOpsKeys"] == sorted(file_ops_keys)
    assert result["fileOpsValues"] == sorted(file_ops_keys)
    assert result["event"] == {"registered": 1, "type": "click", "sameHandler": True, "capture": True}
