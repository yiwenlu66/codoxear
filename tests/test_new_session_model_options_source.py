from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = module_path("app_launch.js")
DISPLAY = module_path("app_display.js")
NEW_SESSION = module_path("app_new_session.js")
SESSION_CATALOG = module_path("app_session_catalog.js")


def eval_model_options(
    query: str,
    *,
    backend: str = "codex",
    provider_choices: list[str] | None = None,
    provider_models: dict[str, list[str]] | None = None,
    latest_sessions: list[dict] | None = None,
) -> dict:
    sources = [path.read_text(encoding="utf-8") for path in (LAUNCH, DISPLAY, SESSION_CATALOG, NEW_SESSION)]
    providers = provider_choices if provider_choices is not None else ["chatgpt", "openai-api", "crs"]
    model_map = provider_models if provider_models is not None else {}
    sessions = latest_sessions if latest_sessions is not None else [
        {"agent_backend": "codex", "model": "gpt-5.4", "model_provider": "openai", "preferred_auth_method": "chatgpt"},
        {"agent_backend": "codex", "model": "gpt-5.4", "model_provider": "crs", "preferred_auth_method": "apikey"},
        {"agent_backend": "pi", "model": "other", "model_provider": "anthropic"},
    ]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const storage = {{ data:new Map(), getItem(k) {{ return this.data.get(k)||null; }}, setItem(k,v) {{this.data.set(k,String(v));}}, removeItem(k) {{this.data.delete(k);}} }};
        const ctx = {{ URL, window: {{ CodoxearUrls: {{resolveAppUrl:(p)=>String(p)}}, CodoxearStorage:storage }} }};
        vm.createContext(ctx); for (const source of {json.dumps(sources)}) vm.runInContext(source,ctx);
        let currentBackend={json.dumps(backend)}, provider={json.dumps("" if backend == "pi" else "chatgpt")}, literal="", absent=false;
        const modelInput={{value:{json.dumps(query)},focus(){{}},setSelectionRange(){{}}}}, noop=()=>{{}}, cls={{toggle(){{}},remove(){{}}}}, node={{innerHTML:"",appendChild(){{}}}};
        const sessionCatalog=ctx.window.CodoxearSessionCatalog.createSessionCatalog({{consoleError:()=>{{}}}});
        sessionCatalog.set("newSessionDefaults",{{model:"gpt-5.4-mini",models:["gpt-5.4","o4-mini"],model_providers:{json.dumps(providers)},provider_choices:{json.dumps(providers)},provider_models:{json.dumps(model_map)},reasoning_efforts:["off","low","high"],reasoning_efforts_by_model:{{}}}});
        sessionCatalog.set("latestSessions",{json.dumps(sessions)}); sessionCatalog.set("tmuxAvailable",true);
        const c=ctx.window.CodoxearNewSession.createNewSessionController({{
          backend:()=>currentBackend, provider:()=>provider, reasoningEffort:()=>"high", literalModelInputValue:()=>literal, launchPresetProviderAbsent:()=>absent,
          sessionCatalog,
          assignProvider:v=>provider=v, assignReasoningEffort:noop, assignLiteralModelInputValue:v=>literal=v, assignLaunchPresetProviderAbsent:v=>absent=Boolean(v), modelInput, modelField:{{classList:cls}},status:{{textContent:""}},reasoningBtn:node,setPickerButtonContent:noop,renderReasoningMenu:noop,renderModelMenu:noop,setFast:noop,setBackend:noop,setTmuxChecked:noop,applyDialogMenus:noop,closeModelMenu:noop,
          cwdInput:{{value:""}},cwdMenu:{{innerHTML:""}},cwdField:{{classList:cls}},cwdHint:{{classList:cls}},nameInput:{{value:""}},cwdMenuFocus:()=>-1,assignCwdMenuFocus:noop,closeCwdMenu:noop,el:()=>({{appendChild:noop}}),resumeMenu:{{innerHTML:""}},resumeBtn:{{}},closeResumeMenu:noop,fetchResumeCandidates:async()=>({{sessions:[]}}),tmuxToggle:{{}},tmuxField:{{style:{{}}}},worktreeToggle:{{}},worktreeInput:{{value:""}},worktreeField:{{style:{{}}}},startBtn:{{}}
        }});
        const options=c.sessionModelOptions(); const filtered=c.filteredNewSessionModelOptions(); const parsed=c.parseNewSessionProviderModelInput(); if (filtered[0] || options[0]) c.selectNewSessionModel(filtered[0] || options[0]);
        const selectedInput = modelInput.value;
        const longDisplay = c.newSessionProviderModelDisplay("glm-5.2", "dexgem-completions");
        modelInput.value = longDisplay;
        const longParsed = c.parseNewSessionProviderModelInput();
        process.stdout.write(JSON.stringify({{options,filtered,parsed,input:selectedInput,provider,literal,absent,longDisplay,longParsed}}));
        """
    )
    proc = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


class TestNewSessionModelOptionsBehavior(unittest.TestCase):
    def test_options_merge_defaults_and_backend_matched_recent_models(self) -> None:
        result = eval_model_options("")
        pairs = {(item["providerChoice"], item["model"]) for item in result["options"]}
        self.assertContains(("chatgpt", "gpt-5.4"), pairs)
        self.assertContains(("crs", "gpt-5.4"), pairs)
        self.assertNotContains(("anthropic", "other"), pairs)
        self.assertEqual(result["options"][0]["displayText"], "gpt-5.4-mini · chatgpt")

    def test_provider_model_filter_and_selection_render_a_single_input_value(self) -> None:
        result = eval_model_options("crs/gpt")
        self.assertEqual([(x["providerChoice"], x["model"]) for x in result["filtered"]], [("crs", "gpt-5.4")])
        self.assertEqual(result["input"], "gpt-5.4 · crs")

    def test_pi_accepts_a_custom_provider_model_pair(self) -> None:
        result = eval_model_options("anthropic/claude-haiku-4-5", backend="pi", provider_choices=[])
        self.assertEqual(result["parsed"]["providerChoice"], "anthropic")
        self.assertEqual(result["parsed"]["model"], "claude-haiku-4-5")
        self.assertEqual(result["parsed"]["providerError"], "")
    def test_recent_models_must_match_a_configured_provider_model_pair(self) -> None:
        result = eval_model_options(
            "",
            provider_choices=["chatgpt"],
            latest_sessions=[
                {"agent_backend": "codex", "model": "gpt-5.4", "model_provider": "openai", "preferred_auth_method": "chatgpt"},
                {"agent_backend": "codex", "model": "ghost-model", "model_provider": "retired", "preferred_auth_method": "apikey"},
            ],
        )
        pairs = {(item["providerChoice"], item["model"]) for item in result["options"]}
        self.assertContains(("chatgpt", "gpt-5.4"), pairs)
        self.assertNotContains(("", "ghost-model"), pairs)

    def test_configured_provider_model_map_prevents_cross_product_options(self) -> None:
        result = eval_model_options(
            "",
            provider_choices=["chatgpt", "crs"],
            provider_models={"chatgpt": ["gpt-5.4"], "crs": ["kimi-k3"]},
            latest_sessions=[],
        )
        pairs = {(item["providerChoice"], item["model"]) for item in result["options"]}
        self.assertEqual(pairs, {("chatgpt", "gpt-5.4"), ("crs", "kimi-k3")})

    def test_long_provider_is_shown_without_abbreviation_and_preserves_identity(self) -> None:
        result = eval_model_options("", provider_choices=["chatgpt", "openai-api", "dexgem-completions"])
        # Providers are shown in full — lossy abbreviation is non-injective and
        # resolved to the wrong provider (dexgem-messages vs dexgem-responses).
        self.assertEqual(result["longDisplay"], "glm-5.2 · dexgem-completions")
        self.assertEqual(result["longParsed"]["model"], "glm-5.2")
        self.assertEqual(result["longParsed"]["providerChoice"], "dexgem-completions")


if __name__ == "__main__":
    unittest.main()
