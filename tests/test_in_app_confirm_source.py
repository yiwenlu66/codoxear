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

    assert 'confirmReload: (message) => confirmApp({ title: "Reload file from disk?", message, confirmText: "Reload", cancelText: "Cancel", destructive: true })' in app_source
    assert 'confirmAction: (options) => confirmApp(options)' in app_source
    assert 'const confirmed = await confirmAction({' in queue_source
    assert 'destructive: true,' in queue_source
    assert 'const ok = await confirmReload(`Reload ${savePath} from disk and discard your unsaved editor draft?`);' in viewer_source


def test_destructive_confirm_dialog_focuses_cancel_and_traps_tab() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'destructive: Boolean(raw.destructive),' in source
    assert 'function focusAppConfirmInitial({ destructive = false } = {}) {' in source
    assert 'const preferred = destructive ? appConfirmCancelBtn : appConfirmConfirmBtn;' in source
    assert 'focusAppConfirmInitial(normalized);' in source
    assert 'if (e.key === "Tab" && appConfirm.style.display === "flex") {' in source
    assert 'const focusable = appConfirmFocusableControls();' in source
    assert 'e.preventDefault();' in source
    assert 'e.stopPropagation();' in source
    assert 'focusable[nextIndex].focus({ preventScroll: true });' in source


def test_destructive_confirm_call_sites_are_marked() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    queue_source = APP_QUEUE_JS.read_text(encoding="utf-8")

    destructive_titles = [
        "Clear unknown-send marker?",
        "Dismiss launch record?",
        "Reload file from disk?",
        "Clear pending attachment state?",
    ]
    for title in destructive_titles:
        pattern = re.compile(rf'title:\s*(?:launchRow \? )?"{re.escape(title)}"[\s\S]{{0,360}}destructive:\s*true')
        assert pattern.search(source), title

    assert re.search(r'title:\s*launchRow \? "Dismiss launch record\?" : "Delete session\?"[\s\S]{0,360}destructive:\s*true', source)
    assert re.search(r'title:\s*"Delete recovery item\?"[\s\S]{0,360}destructive:\s*true', queue_source)

    pending_attachment_block = re.search(r'title:\s*"Send pending attachment\?"[\s\S]{0,240}\}\);', source)
    assert pending_attachment_block is not None
    assert "destructive" not in pending_attachment_block.group(0)
