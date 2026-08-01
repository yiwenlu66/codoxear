from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def test_initial_session_and_transcript_load_precedes_auxiliary_voice_requests() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    startup = source[source.index("const sessions = await refreshSessions();", source.index("__codoxearMarkBootstrapped")):]

    assert startup.index("const sessions = await refreshSessions();") < startup.index("await Promise.all([loadVoiceSettings(), syncNotificationState()])")
    assert startup.index("if (pick) await selectSession(pick);") < startup.index("await Promise.all([loadVoiceSettings(), syncNotificationState()])")
