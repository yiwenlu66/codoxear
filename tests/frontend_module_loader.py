"""Compatibility loader for VM tests while frontend sources are ESM.

Production serves the bundled ESM entrypoint. These tests intentionally execute
small controller modules inside Node's classic ``vm`` context, so this helper
projects one ESM module at a time back into the legacy browser-global shape.
The projection is test-only; source modules remain import/export modules.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"

MODULE_NAMESPACES = {
    # generated from the pre-ESM global assignments; a module with several
    # namespaces intentionally projects the same namespace object to each,
    # preserving every public export used by old harness fixtures.
    "app_api.js": ["CodoxearApi", "CodoxearPerf"],
    "app_application.js": ["CodoxearApplication", "CodoxearDom", "CodoxearUrls"],
    "app_application_composition.js": ["CodoxearApplicationComposition", "CodoxearEventBindings", "CodoxearToast"],
    "app_attachments.js": ["CodoxearAttachments"],
    "app_chat_interaction.js": ["CodoxearChatInteraction"],
    "app_chat_navigation.js": ["CodoxearChatNavigation"],
    "app_chat_search.js": ["CodoxearChatSearch"],
    "app_code_copy.js": ["CodoxearCodeCopy"],
    "app_composer.js": ["CodoxearComposer"],
    "app_conversation_copy.js": ["CodoxearConversationCopy"],
    "app_diagnostics.js": ["CodoxearDiagnostics"],
    "app_dialog_menu.js": ["CodoxearDialogMenu", "CodoxearDialogMenus"],
    "app_display.js": ["CodoxearDisplay"],
    "app_file_candidate_state.js": ["CodoxearFileCandidateState"],
    "app_file_candidates.js": ["CodoxearFileCandidates"],
    "app_file_download.js": ["CodoxearFileDownload"],
    "app_file_editor.js": ["CodoxearFileEditor"],
    "app_file_editor_ops.js": ["CodoxearFileEditorOps"],
    "app_file_helpers.js": ["CodoxearFileHelpers"],
    "app_file_mode.js": ["CodoxearFileMode"],
    "app_file_ops.js": ["CodoxearClipboard", "CodoxearFileEditMode", "CodoxearFileOps", "CodoxearFileTouch"],
    "app_file_paste_dialog.js": ["CodoxearFilePasteDialog"],
    "app_file_pdf.js": ["CodoxearFilePdf"],
    "app_file_picker.js": ["CodoxearFilePicker"],
    "app_file_picker_ops.js": ["CodoxearFilePickerOps"],
    "app_file_render_surface.js": ["CodoxearFileRenderSurface"],
    "app_file_unsaved.js": ["CodoxearFileUnsaved"],
    "app_file_unsaved_dialog.js": ["CodoxearFileUnsavedDialog"],
    "app_file_video.js": ["CodoxearFileVideo"],
    "app_file_viewer.js": ["CodoxearFileViewer"],
    "app_file_viewer_controller.js": ["CodoxearFileViewerController"],
    "app_file_viewer_integration.js": ["CodoxearFileViewerIntegration"],
    "app_file_viewer_lifecycle.js": ["CodoxearFileViewerLifecycle"],
    "app_file_viewer_operations.js": ["CodoxearFileViewerOperations"],
    "app_file_viewer_panel.js": ["CodoxearFileViewerPanel"],
    "app_hint_mode.js": ["CodoxearHintMode"],
    "app_ios_viewport.js": ["CodoxearIOSViewport"],
    "app_launch.js": ["CodoxearLaunch"],
    "app_markdown.js": ["CodoxearMarkdown"],
    "app_message_flow.js": ["CodoxearMessageFlow"],
    "app_message_history.js": ["CodoxearMessageHistory"],
    "app_message_rows.js": ["CodoxearMessageRows"],
    "app_modal.js": ["CodoxearModal"],
    "app_network.js": ["CodoxearNetwork"],
    "app_new_session.js": ["CodoxearNewSession"],
    "app_polling.js": ["CodoxearPolling", "CodoxearSecondaryPoll"],
    "app_queue.js": ["CodoxearQueue"],
    "app_send_lifecycle.js": ["CodoxearSendLifecycle"],
    "app_session_display.js": ["CodoxearSessionDisplay"],
    "app_session_edit.js": ["CodoxearSessionEdit"],
    "app_session_helpers.js": ["CodoxearSessionHelpers"],
    "app_session_lifecycle.js": ["CodoxearInterrupt", "CodoxearSessionLifecycle"],
    "app_session_refresh.js": ["CodoxearSessionRefresh"],
    "app_session_title.js": ["CodoxearSessionTitle"],
    "app_sessions.js": ["CodoxearSessions"],
    "app_shell.js": ["CodoxearShell"],
    "app_sse.js": ["CodoxearSse"],
    "app_storage.js": ["CodoxearStorage"],
    "app_transcript.js": ["CodoxearMessageIdentity", "CodoxearPendingUser", "CodoxearTranscript"],
    "app_transcript_render.js": ["CodoxearNavigationPulse", "CodoxearTranscriptRender"],
    "app_transcript_view.js": ["CodoxearTranscriptView"],
    "app_unattended.js": ["CodoxearUnattended"],
    "app_viewport.js": ["CodoxearViewport"],
    "app_voice.js": ["CodoxearVoice"],
    "app_voice_helpers.js": ["CodoxearVoiceHelpers"],
    "app_wiring.js": ["CodoxearWiring"],
}

_CACHE = Path(tempfile.gettempdir()) / "codoxear-esm-vm"


def strip_esm(source: str, module_name: str) -> str:
    """Turn one ESM source into an isolated classic VM script.

    Namespace imports are redirected to the compatibility globals established by
    earlier harness loads. Named exports are projected as frozen globals after
    the module body, matching the old browser load contract.
    """
    imports: list[str] = []

    def remove_import(match: re.Match[str]) -> str:
        imports.append(match.group(1))
        return ""

    source = re.sub(
        r'^import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+["\']\./[^"\']+["\'];?\s*$',
        remove_import,
        source,
        flags=re.MULTILINE,
    )
    for namespace in imports:
        source = re.sub(rf'\b{re.escape(namespace)}\b', f'window.{namespace}', source)

    exports: list[str] = []
    for match in re.finditer(r'^export\s*\{(?P<body>[^}]*)\};?\s*$', source, re.MULTILINE):
        for item in match.group("body").split(","):
            item = item.strip()
            if not item:
                continue
            local, _, public = item.partition(" as ")
            exports.append(f"{public.strip() if public else local} : {local.strip()}")
    source = re.sub(r'^export\s*\{[^}]*\};?\s*$', '', source, flags=re.MULTILINE)

    namespaces = MODULE_NAMESPACES.get(module_name, [])
    projections: list[str] = []
    if module_name in {"app_queue.js", "app_unattended.js"}:
        controller = "createQueueController" if module_name == "app_queue.js" else "createUnattendedController"
        dom = "createQueueDom" if module_name == "app_queue.js" else "createUnattendedDom"
        object_literal = f"{controller}: {controller}"
        projection = f'window.{{namespace}} = Object.freeze({{{object_literal}}});'
        if dom in {name.split(":", 1)[0].strip() for name in exports}:
            projection = f'const api = {{ {object_literal} }}; Object.defineProperty(api, "{dom}", {{ value: {dom} }}); window.{{namespace}} = Object.freeze(api);'
        for namespace in namespaces:
            projections.append(projection.replace("{namespace}", namespace))
    elif exports and namespaces:
        object_literal = ", ".join(exports)
        projections.extend(f'window.{namespace} = Object.freeze({{{object_literal}}});' for namespace in namespaces)
    body = source.strip()
    return "(function () {\n" + body + "\n" + "\n".join(projections) + "\n})();\n"


def module_path(module_name: str) -> Path:
    """Return a cached classic VM fixture generated from the ESM module."""
    module_name = Path(module_name).name
    source_path = STATIC / module_name
    if module_name not in MODULE_NAMESPACES or not source_path.is_file():
        raise ValueError(f"unsupported frontend module: {module_name}")
    _CACHE.mkdir(parents=True, exist_ok=True)
    output = _CACHE / module_name
    source = source_path.read_text(encoding="utf-8")
    transformed = strip_esm(source, module_name)
    if not output.is_file() or output.read_text(encoding="utf-8") != transformed:
        output.write_text(transformed, encoding="utf-8")
    return output
