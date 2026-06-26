import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_URL_JS = ROOT / "codoxear" / "static" / "app_url.js"
APP_STORAGE_JS = ROOT / "codoxear" / "static" / "app_storage.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"


def eval_provider_choice_to_settings() -> dict:
    source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          console,
          window: {{
            CodoxearUrls: {{ resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^[/]/, ""), "http://localhost/").toString() }},
            CodoxearStorage: {{
              getItem: () => null,
              setItem: () => true,
              removeItem: () => true,
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const launch = ctx.window.CodoxearLaunch;
        const piAbsent = {{ agent_backend: "pi", model_provider: null, provider_choice: "openai-api", model: "claude" }};
        const piActual = {{ agent_backend: "pi", model_provider: "anthropic", provider_choice: "openai-api", model: "claude" }};
        process.stdout.write(JSON.stringify({{
          codexDefault: launch.providerChoiceToSettings("", "codex"),
          codexApi: launch.providerChoiceToSettings("openai-api", "codex"),
          codexCustom: launch.providerChoiceToSettings("crs", "codex"),
          piEmpty: launch.providerChoiceToSettings("", "pi"),
          piExplicit: launch.providerChoiceToSettings("macaron", "pi"),
          ccIgnored: launch.providerChoiceToSettings("macaron", "cc"),
          piAbsentSessionProvider: launch.sessionProviderChoice(piAbsent),
          piActualSessionProvider: launch.sessionProviderChoice(piActual),
          piAbsentProviderSettings: launch.providerChoiceToSettings(launch.sessionProviderChoice(piAbsent), "pi"),
          modelMatchEmpty: launch.modelOptionMatches({{ searchText: "crs/gpt-5.4 gpt-5.4" }}, ""),
          modelMatchExact: launch.modelOptionMatches({{ searchText: "crs/gpt-5.4 gpt-5.4" }}, "crs/gpt-5.4 gpt-5.4"),
          modelMatchPrefix: launch.modelOptionMatches({{ searchText: "crs/gpt-5.4 gpt-5.4" }}, "crs/gpt"),
          modelMatchContains: launch.modelOptionMatches({{ searchText: "crs/gpt-5.4 gpt-5.4" }}, "gpt-5"),
          modelMatchFallbackModel: launch.modelOptionMatches({{ model: "o4-mini" }}, "o4"),
          modelMatchNoMatch: launch.modelOptionMatches({{ searchText: "crs/gpt-5.4 gpt-5.4" }}, "claude"),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def test_launch_module_loads_after_real_url_and_storage_helpers() -> None:
    scripts = [APP_URL_JS, APP_STORAGE_JS, APP_LAUNCH_JS]
    js = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const localStorage = {{
          data: new Map(),
          getItem(key) {{ return this.data.has(String(key)) ? this.data.get(String(key)) : null; }},
          setItem(key, value) {{ this.data.set(String(key), String(value)); }},
          removeItem(key) {{ this.data.delete(String(key)); }},
        }};
        const ctx = {{
          URL,
          console,
          window: {{
            location: {{ href: "http://localhost/codoxear/", origin: "http://localhost", pathname: "/codoxear/" }},
            localStorage,
          }},
        }};
        vm.createContext(ctx);
        for (const file of {json.dumps([str(path) for path in scripts])}) {{
          vm.runInContext(fs.readFileSync(file, "utf8"), ctx, {{ filename: file }});
        }}
        const launch = ctx.window.CodoxearLaunch;
        launch.rememberBackendChoice("claude-code");
        process.stdout.write(JSON.stringify({{
          hasLaunch: Boolean(launch),
          backend: launch.loadRememberedBackendChoice(),
          logo: launch.agentBackendLogoPath("cc"),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = json.loads(proc.stdout)
    assert result["hasLaunch"] is True
    assert result["backend"] == "cc"
    assert result["logo"] == "http://localhost/codoxear/static/logos/cc.svg"


def test_app_js_requires_launch_module_without_fallback() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")

    assert "const codoxearLaunch = window.CodoxearLaunch;" in source
    assert 'throw new Error("Codoxear launch helpers failed to load")' in source
    assert "function defaultsForAgentBackend(backend)" in source
    assert "return codoxearLaunch.defaultsForAgentBackend(backend, newSessionDefaults);" in source
    assert "function providerChoiceToSettings(choice, agentBackend = \"codex\")" in source
    assert "return codoxearLaunch.providerChoiceToSettings(choice, agentBackend);" in source
    assert 'typeof codoxearLaunch.modelOptionMatches !== "function"' in source
    assert "function modelOptionMatches(option, query)" in source
    assert "return codoxearLaunch.modelOptionMatches(option, query);" in source
    assert 'const text = String(option && option.searchText ? option.searchText : option && option.model ? option.model : "").toLowerCase();' not in source
    assert 'const text = String(option && option.searchText ? option.searchText : option && option.model ? option.model : "").toLowerCase();' in launch_source
    assert "const LAST_BACKEND_KEY" not in source
    assert "const LAST_BACKEND_KEY" in launch_source
    assert "window.CodoxearLaunch = Object.freeze({" in launch_source
    assert 'throw new Error("Codoxear URL helpers failed to load")' in launch_source
    assert 'throw new Error("Codoxear storage helpers failed to load")' in launch_source


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
    session_helper_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")

    assert 'function sessionSelectable(s) {' in source
    assert 'return codoxearSessionHelpers.sessionSelectable(s);' in source
    assert 'return !!(s && !sessionLaunchPending(s));' in session_helper_source
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
    assert result["modelMatchEmpty"] is True
    assert result["modelMatchExact"] is True
    assert result["modelMatchPrefix"] is True
    assert result["modelMatchContains"] is True
    assert result["modelMatchFallbackModel"] is True
    assert result["modelMatchNoMatch"] is False
