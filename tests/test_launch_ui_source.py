from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def test_launch_failure_sidebar_uses_single_visible_failure_marker() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert '`${cwdName} launch failed`' not in source
    assert '`${name} launch failed`' not in source
    assert 'const stateTxt = launchPending ? "starting" : fmtRelativeAge(ageS);' in source
    assert '(launchPending ? " pending" : s.snoozed || s.blocked ? " suppressed" : s.busy ? " busy" : " idle")' in source
    assert 'setToast(`launch failed:' not in source


def test_launch_attempt_rows_use_dismiss_language() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'const launchRow = launchFailed || launchPending;' in source
    assert 'confirm(launchRow ? "Dismiss this launch record?" : "Delete this session?")' in source
    assert 'title: launchRow ? "Dismiss launch record" : "Delete session"' in source
    assert 'if (launchRow && card && card.parentNode) card.remove();' in source


def test_failed_launch_rows_are_clickable_transcripts() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'return !!(s && !sessionLaunchPending(s));' in source
    assert 'raw === "bound" || raw === "pending_bind" || raw === "failed"' in source
    assert 'if (slotChange.current.state !== "failed") kickPoll(900);' in source
    assert 'if (activeTranscriptState === "failed") return;' in source
    assert 'failed session cannot receive messages' in source
