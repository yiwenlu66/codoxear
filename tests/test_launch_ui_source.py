import json
import subprocess
import textwrap
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_provider_choice_to_settings() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function normalizeAgentBackendName(value) {")
    end = source.index("function fmtIdleAge", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_providerChoiceToSettings = providerChoiceToSettings;\nglobalThis.__test_sessionProviderChoice = sessionProviderChoice;\n")}, ctx);
        const piAbsent = {{ agent_backend: "pi", model_provider: null, provider_choice: "openai-api", model: "claude" }};
        const piActual = {{ agent_backend: "pi", model_provider: "anthropic", provider_choice: "openai-api", model: "claude" }};
        process.stdout.write(JSON.stringify({{
          codexDefault: ctx.__test_providerChoiceToSettings("", "codex"),
          codexApi: ctx.__test_providerChoiceToSettings("openai-api", "codex"),
          codexCustom: ctx.__test_providerChoiceToSettings("crs", "codex"),
          piEmpty: ctx.__test_providerChoiceToSettings("", "pi"),
          piExplicit: ctx.__test_providerChoiceToSettings("macaron", "pi"),
          ccIgnored: ctx.__test_providerChoiceToSettings("macaron", "cc"),
          piAbsentSessionProvider: ctx.__test_sessionProviderChoice(piAbsent),
          piActualSessionProvider: ctx.__test_sessionProviderChoice(piActual),
          piAbsentProviderSettings: ctx.__test_providerChoiceToSettings(ctx.__test_sessionProviderChoice(piAbsent), "pi"),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


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


def test_provider_choice_mapping_is_backend_specific() -> None:
    result = eval_provider_choice_to_settings()
    assert result["codexDefault"] == {"model_provider": "openai", "preferred_auth_method": "chatgpt"}
    assert result["codexApi"] == {"model_provider": "openai", "preferred_auth_method": "apikey"}
    assert result["codexCustom"] == {"model_provider": "crs", "preferred_auth_method": "apikey"}
    assert result["piEmpty"] == {"model_provider": None, "preferred_auth_method": None}
    assert result["piExplicit"] == {"model_provider": "macaron", "preferred_auth_method": None}
    assert result["ccIgnored"] == {"model_provider": None, "preferred_auth_method": None}
    assert result["piAbsentSessionProvider"] == ""
    assert result["piActualSessionProvider"] == "anthropic"
    assert result["piAbsentProviderSettings"] == {"model_provider": None, "preferred_auth_method": None}
