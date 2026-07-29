import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "codoxear" / "static" / "app_launch.js"
URLS = ROOT / "codoxear" / "static" / "app_url.js"
STORAGE = ROOT / "codoxear" / "static" / "app_storage.js"


def eval_launch_ui() -> dict:
    programs = [path.read_text(encoding="utf-8") for path in (URLS, STORAGE, LAUNCH)]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const localStorage = {{ values: new Map(), getItem(k) {{ return this.values.get(k) || null; }}, setItem(k,v) {{ this.values.set(k, String(v)); }}, removeItem(k) {{ this.values.delete(k); }} }};
        const ctx = {{ URL, window: {{ location: {{ href: "http://host/codoxear/", origin: "http://host", pathname: "/codoxear/" }}, localStorage, CODOXEAR_ASSET_VERSION: "v1" }} }};
        vm.createContext(ctx);
        for (const source of {json.dumps(programs)}) vm.runInContext(source, ctx);
        const launch = ctx.window.CodoxearLaunch;
        launch.rememberBackendChoice("claude-code");
        launch.rememberProviderModelChoice("pi", "anthropic", "claude-haiku-4-5");
        process.stdout.write(JSON.stringify({{
          rememberedBackend: launch.loadRememberedBackendChoice(),
          rememberedModel: launch.loadRememberedProviderModelChoice("pi"),
          pi: launch.providerChoiceToSettings("anthropic", "pi"),
          codex: launch.providerChoiceToSettings("openai-api", "codex"),
          cc: launch.providerChoiceToSettings("anthropic", "cc"),
          logo: launch.agentBackendLogoPath("cc"),
          modelLabel: launch.providerModelDisplay("gpt-5.4", "crs", {{hasProviderChoices:true}}),
          redacted: launch.redactedLaunchErrorText("API_TOKEN=secret Authorization: Bearer tokenvalue"),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def test_launch_ui_projects_backend_choices_and_safe_labels() -> None:
    result = eval_launch_ui()
    assert result["rememberedBackend"] == "cc"
    assert result["rememberedModel"] == "anthropic/claude-haiku-4-5"
    assert result["pi"] == {"model_provider": "anthropic", "preferred_auth_method": None}
    assert result["codex"] == {"model_provider": "openai", "preferred_auth_method": "apikey"}
    assert result["cc"] == {"model_provider": None, "preferred_auth_method": None}
    assert result["logo"] == "http://host/codoxear/static/logos/cc.svg?v=v1"
    assert result["modelLabel"] == "crs/gpt-5.4"
    assert result["redacted"] == "API_TOKEN=[redacted] Authorization: [redacted]"
