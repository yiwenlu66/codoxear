import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
APP_QUEUE_JS = ROOT / "codoxear" / "static" / "app_queue.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"


def test_app_confirm_dialog_is_accessible_dom_modal() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = APP_CSS.read_text(encoding="utf-8")

    assert 'id: "appConfirmBackdrop"' in source
    assert 'id: "appConfirm"' in source
    assert 'role: "dialog"' in source
    assert '"aria-modal": "true"' in source
    assert '"aria-labelledby": "appConfirmTitle"' in source
    assert '"aria-describedby": "appConfirmMessage"' in source
    assert 'class: "sendChoice appConfirm"' in source
    assert 'class: "muted appConfirmMessage"' in source
    assert '.appConfirmBackdrop {' in css
    assert '.appConfirm {' in css
    assert '.appConfirmMessage {' in css
    assert 'white-space: pre-wrap;' in css


def test_app_confirm_returns_promise_and_restores_focus_on_all_cancel_paths() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'function confirmApp(options = {}) {' in source
    assert 'return new Promise((resolve) => {' in source
    assert 'pending.resolve(Boolean(result));' in source
    assert 'appConfirmReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;' in source
    assert 'restoreModalFocus(target, () => appConfirm.style.display === "flex")' in source
    assert 'appConfirmConfirmBtn.onclick = () => resolveAppConfirm(true);' in source
    assert 'appConfirmCancelBtn.onclick = () => resolveAppConfirm(false);' in source
    assert 'appConfirmBackdrop.onclick = () => resolveAppConfirm(false);' in source
    assert 'if (appConfirm.style.display === "flex") {' in source
    assert 'resolveAppConfirm(false);' in source


def test_product_static_js_has_no_native_confirm_calls() -> None:
    for path in (ROOT / "codoxear" / "static").glob("*.js"):
        source = path.read_text(encoding="utf-8")
        assert re.search(r"\b(?:window\.)?confirm\s*\(", source) is None, path.name


def test_async_confirm_seams_are_wired_from_app() -> None:
    app_source = APP_JS.read_text(encoding="utf-8")
    queue_source = APP_QUEUE_JS.read_text(encoding="utf-8")
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")

    assert 'confirmReload: (message) => confirmApp({ title: "Reload file from disk?", message, confirmText: "Reload", cancelText: "Cancel" })' in app_source
    assert 'confirmAction: (options) => confirmApp(options)' in app_source
    assert 'const confirmed = await confirmAction({' in queue_source
    assert 'const ok = await confirmReload(`Reload ${savePath} from disk and discard your unsaved editor draft?`);' in viewer_source
