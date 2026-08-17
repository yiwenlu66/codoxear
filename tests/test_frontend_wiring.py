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
    topbar_keys = [
        "el", "iconSvg", "setToast", "onInterrupt", "sessionState",
        "topMeta", "topActions", "eventBindings",
    ]
    file_viewer_operations_keys = [
        "el", "fileStatus", "fileEditButton", "iconSvg", "currentSessionId", "currentFileSessionId", "normalizeSessionId", "normalizeFileApiPath", "isFileViewerOpen", "hideFileUnsavedDialog", "resetFileSearchState", "closeFilePickerMenu", "isTextFileKind", "isDiffableFileKind", "confirmReload", "promptUnsavedFileChoice", "restoreFileEditorText", "hideFileViewer", "setFilePath", "resetFileViewerPanel", "applyFileLoadResult", "normalizeDraftFilePath", "inspectSessionFilePath", "api", "focusEditor", "disposeOpenRender", "isMarkdownPreviewable", "updateFileTouchToolbar", "useTouchFileEditorControls", "hasActiveFileCodeEditor", "hasBlockingFileEditorModal", "isTextEntryTarget", "eventTargetElement", "normalizeFileEditorPosition", "applyFileEditorSelection", "isCollapsedFileSelection", "positionAfterInsertedText", "fileEditorEditSupportAvailable", "updateFileDiffEditorOptions", "showFilePasteDialog", "hideFilePasteDialog", "clipboardReadAvailable", "readClipboardText", "fileEditorDeleteCommandForKey", "isActiveFileEditorInput", "getActiveFileSelectionText", "copyToClipboard", "focusActiveFileCodeEditor", "nowMs", "setToast", "renderMonacoFile", "getFileEditorText", "fmtBytes", "applyFileMode", "rememberOpenedFile", "renderFilePickerMenu", "currentFileViewMode", "currentFileNonDiffMode", "setFileViewMode", "currentFileEditMode", "currentFileEditorKind", "setFileEditorKind", "setFileEditMode", "currentActiveFileKind", "currentActiveFileText", "currentActiveFileEditable", "currentActiveFileVersion", "currentActiveFileDraft", "applyActiveFileTextState", "applyActiveFileDiffState", "applyActiveFileNonTextState", "currentActiveFileIdentity", "currentActiveFileLine", "startFileOpenRequest", "isCurrentFileOpenRequest", "normalizeExplicitFileOpenMode", "resolveFileOpenMode", "isFileOpenAbortError", "activeFileEntry", "isGitFileCandidatePath", "currentFileCandidateGitStateFresh", "activeFileCanEnterEditMode", "activeFileEditorWritable", "activeFileEditorIdleTextWritable", "currentFileEditorState", "isUnavailable", "blockUnavailableFileAction", "fileEntryForPath", "resetActiveFileBufferState", "resolveFileOpenViewMode", "activeFileEditorIdleWritable", "isFileViewerSessionUnavailable", "rememberActiveFileSelection", "setActiveFileIdentity",
    ]
    file_picker_ops_keys = [
        "wiring", "codoxearFilePicker", "normalizeLineNumber", "filePickerField", "filePickerMenu", "filePickerInput", "api", "document", "el", "sessionState", "blockUnavailableFileAction", "currentFileViewerSessionId", "fileViewerController", "fileCandidateKey", "currentActiveFileDraft", "activeFilePathValue", "normalizeFileApiPath", "renderFilePickerMenu", "applyFileMenuState", "normalizeDraftFilePath", "filePickerSectionLabel", "duplicateFilePickerPaths", "rawByteDuplicatePaths", "filePickerIdentityHint", "filePickerTitle", "currentActiveFileIdentity", "openDraftFilePathWithGuard", "openFilePathWithResolvedMode", "filePickerSelectionLine", "ensureCurrentFileViewerSession", "resetFilePickerInput", "closeFilePickerMenu", "resetFileSearchState", "setFileStatus", "requestAnimationFrame",
    ]
    file_editor_ops_keys = [
        "addAppEvent", "wiring", "codoxearFileEditor", "resolveAppUrl", "fileDiff", "normalizeLineNumber", "requestAnimationFrame", "setTimeout", "isCurrentFileOpenRequest", "renderPlainTextFallback", "disposeFileEditor", "currentEditorKind", "setEditorKind", "currentFileEditMode", "currentActiveFileEditable", "isUnavailable", "isProgrammaticChange", "currentTouchSelectMode", "resetTouchSelectionState", "currentActiveFileText", "setDirty", "runProgrammaticChange", "syncReadOnly", "updateTouchToolbar",
    ]
    file_picker_delegate_keys = [
        "fileViewerController", "fileModeControlsRuntime", "filePickerDomRuntime", "filePickerMenuState", "filePickerInput", "filePickerInputRuntime", "activeFilePathValue", "openedFileRuntime", "fileReferenceRuntime", "filePickerSearchState", "filePickerRenderRuntime", "fileViewerPanelRuntime", "sessionState", "sessionCatalog", "stripPathLocationSuffix",
    ]
    file_picker_interaction_keys = [
        "eventBindings", "fileBtn", "showFileViewer", "filePickerInput", "filePickerInputRuntime", "fileModeDiffBtn", "fileModePreviewBtn", "fileEditBtn", "handleFileDiffModeButtonPress", "handleFilePreviewModeButtonPress", "handleFileEditButtonPress", "fileVideoPreviewBtn", "fileVideoPreviewRuntime", "fileDownloadBtn", "fileDownloadRuntime", "activeFileDownloadApiPath", "codoxearFileViewer", "fileTouchSelectBtn", "fileTouchCopyBtn", "fileTouchPasteBtn", "fileTouchUpBtn", "fileTouchLeftBtn", "fileTouchDownBtn", "fileTouchRightBtn", "toggleFileTouchSelectionMode", "copyActiveFileSelection", "pasteFromClipboardIntoActiveFile", "handleFileTouchMoveButtonPress", "fileCloseBtn", "fileBackdrop", "requestHideFileViewer", "$", "fileUnsavedController", "fileUnsavedBackdrop", "filePasteInput", "handleFilePasteInsert", "hideFilePasteDialog", "filePasteBackdrop", "chatInner", "codeBlockCopyRuntime", "fileReferenceRuntime", "fileDiff", "addAppEvent", "document", "Element", "isFileViewerOpen", "menuState", "closeFilePickerMenu",
    ]
    file_ops_keys = [
        "sessionState", "sessionCatalog", "getSessionLifecycleController", "wiring", "document", "window", "HTMLElement", "requestAnimationFrame", "setTimeout", "$", "el", "iconSvg", "resolveAppUrl", "api", "setToast", "confirmApp", "addAppEvent", "normalizeLineNumber", "markdownPreviewHtml", "blockedFileMessage", "listFromFilesField", "listFromFileRecords", "baseName", "codoxearFilePicker", "codoxearFilePickerOps", "codoxearFileViewer", "codoxearFileEditor", "codoxearFileEditorOps", "codoxearFileEditMode", "codoxearFileTouch", "codoxearDialogMenus", "prepareModalOpen", "afterModalVisibilityChanged", "focusModalCloseButton", "restoreModalFocus", "isModalTargetOpen", "newSessionDialogController", "eventBindings", "codoxearFileHelpers", "copyToClipboard", "dialogMenuController", "duplicateFilePickerPaths", "editCloseBtn", "editDependencyBtn", "editDependencyMenu", "editNameInput", "editPriorityRange", "editPriorityResetBtn", "editPriorityValue", "editSaveBtn", "editSnoozeCustomDate", "editSnoozeCustomRow", "editSnoozeCustomTime", "editSnoozeModeButtons", "editStatus", "editViewer", "fileBtn", "filePickerIdentityHint", "filePickerSectionLabel", "filePickerTitle", "fmtBytes", "formatPriorityOffset", "handleAppAuthLoss", "isDiffableFileKind", "isMarkdownPreviewable", "isTextEntryElement", "isTextFileKind", "modalIsolationTargets", "normalizeDraftFilePath", "parseLocalFileRef", "rawByteDuplicatePaths", "refreshSessions", "selectedSessionLaunchFailed", "sessionDisplayName", "sessionTitleWithId", "setPickerButtonContent", "storageGetItem", "storageSetItem", "stripPathLocationSuffix", "titleLabel", "useTouchFileEditorControls", "filePickerField", "filePickerMenu", "filePickerInput", "fileStatus", "fileDiff", "fileImage", "fileVideo", "fileVideoPreviewBtn", "fileTouchToolbar", "fileTouchActions", "fileTouchDpad", "fileTouchCopyBtn", "fileTouchPasteBtn", "fileTouchSelectBtn", "fileTouchUpBtn", "fileTouchLeftBtn", "fileTouchDownBtn", "fileTouchRightBtn", "fileModeDiffBtn", "fileModePreviewBtn", "fileDownloadBtn", "fileBackdrop", "fileViewer", "fileCloseBtn", "fileUnsavedBackdrop", "fileUnsavedDialog", "filePasteBackdrop", "filePasteDialog", "filePasteInput", "fileEditBtn", "chatInner", "codeBlockCopyRuntime", "appConfirm", "appConfirmFocusableControls", "resolveAppConfirm", "sendChoice", "closeSendChoiceDialog", "queueViewer", "hideQueueViewer", "helpViewer", "hideHelpViewer", "diagViewer", "hideDiagViewer", "voiceController", "hideVoiceSettingsDialog",
    ]
    program = """
const vm = require("vm");
const sources = __SOURCES__;
const topbarKeys = __TOPBAR_KEYS__;
const fileOpsKeys = __FILE_OPS_KEYS__;
const fileViewerOperationsKeys = __FILE_VIEWER_OPERATIONS_KEYS__;
const filePickerOpsKeys = __FILE_PICKER_OPS_KEYS__;
const fileEditorOpsKeys = __FILE_EDITOR_OPS_KEYS__;
const filePickerDelegateKeys = __FILE_PICKER_DELEGATE_KEYS__;
const filePickerInteractionKeys = __FILE_PICKER_INTERACTION_KEYS__;
const context = { window: {} };
vm.createContext(context);
for (const source of Object.values(sources)) vm.runInContext(source, context);
const wiring = context.window.CodoxearWiring.createWiring();
const sessionState = { get: () => "s-1", set: () => false, applyRuntime: () => [], subscribe: () => () => {} };
const flow = wiring.createMessageFlowOptions({
  sessionState,
  api: () => null,
  accidental: "must not reach controller",
});
const lifecycle = wiring.createSessionLifecycleOptions({
  sessionState,
  clearDeletedSessionClientState: "not a lifecycle dependency",
});
const topbarInput = Object.fromEntries(topbarKeys.map((key) => [key, key]));
const fileOpsInput = Object.fromEntries(fileOpsKeys.map((key) => [key, key]));
const topbar = wiring.createTopbarOptions({ ...topbarInput, accidental: "must not reach controller" });
const fileOps = wiring.createFileOpsOptions({ ...fileOpsInput, accidental: "must not reach controller" });
const optionContracts = [
  ["fileViewerOperations", "createFileViewerOperationsOptions", fileViewerOperationsKeys],
  ["filePickerOps", "createFilePickerOpsOptions", filePickerOpsKeys],
  ["fileEditorOps", "createFileEditorOpsOptions", fileEditorOpsKeys],
  ["filePickerDelegates", "createFilePickerOperationDelegatesOptions", filePickerDelegateKeys],
  ["filePickerInteractions", "createFilePickerInteractionsOptions", filePickerInteractionKeys],
].map(([name, factory, keys]) => {
  const projection = wiring[factory]({ ...Object.fromEntries(keys.map((key) => [key, key])), accidental: "must not reach controller" });
  return [name, Object.keys(projection).sort(), Object.values(projection).sort()];
});
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
  lifecycleKeys: Object.keys(lifecycle).sort(),
  topbarKeys: Object.keys(topbar).sort(),
  topbarValues: Object.values(topbar).sort(),
  fileOpsKeys: Object.keys(fileOps).sort(),
  fileOpsValues: Object.values(fileOps).sort(),
  optionContracts,
  event: { registered: registered.length, type: listeners[0][1], sameHandler: bound === handler, capture: listeners[0][3].capture },
}));
""".replace("__SOURCES__", json.dumps(sources)).replace("__TOPBAR_KEYS__", json.dumps(topbar_keys)).replace("__FILE_OPS_KEYS__", json.dumps(file_ops_keys)).replace("__FILE_VIEWER_OPERATIONS_KEYS__", json.dumps(file_viewer_operations_keys)).replace("__FILE_PICKER_OPS_KEYS__", json.dumps(file_picker_ops_keys)).replace("__FILE_EDITOR_OPS_KEYS__", json.dumps(file_editor_ops_keys)).replace("__FILE_PICKER_DELEGATE_KEYS__", json.dumps(file_picker_delegate_keys)).replace("__FILE_PICKER_INTERACTION_KEYS__", json.dumps(file_picker_interaction_keys))
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    result = json.loads(completed.stdout)

    assert result["flowKeys"] == ["api", "sessionState"]
    assert result["lifecycleKeys"] == ["sessionState"]
    assert result["topbarKeys"] == sorted(topbar_keys)
    assert result["topbarValues"] == sorted(topbar_keys)
    assert result["fileOpsKeys"] == sorted(file_ops_keys)
    assert result["fileOpsValues"] == sorted(file_ops_keys)
    expected_contracts = [
        ["fileViewerOperations", file_viewer_operations_keys],
        ["filePickerOps", file_picker_ops_keys],
        ["fileEditorOps", file_editor_ops_keys],
        ["filePickerDelegates", file_picker_delegate_keys],
        ["filePickerInteractions", file_picker_interaction_keys],
    ]
    assert result["optionContracts"] == [[name, sorted(keys), sorted(keys)] for name, keys in expected_contracts]
    assert result["event"] == {"registered": 1, "type": "click", "sameHandler": True, "capture": True}
