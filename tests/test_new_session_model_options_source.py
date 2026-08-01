import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "codoxear" / "static" / "app_launch.js"
DISPLAY = ROOT / "codoxear" / "static" / "app_display.js"
NEW_SESSION = ROOT / "codoxear" / "static" / "app_new_session.js"
APP = ROOT / "codoxear" / "static" / "app.js"


def eval_model_options(query: str, *, backend: str = "codex", provider_choices: list[str] | None = None) -> dict:
    sources = [path.read_text(encoding="utf-8") for path in (LAUNCH, DISPLAY, NEW_SESSION)]
    providers = provider_choices if provider_choices is not None else ["chatgpt", "openai-api", "crs"]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const storage = {{ data:new Map(), getItem(k) {{ return this.data.get(k)||null; }}, setItem(k,v) {{this.data.set(k,String(v));}}, removeItem(k) {{this.data.delete(k);}} }};
        const ctx = {{ URL, window: {{ CodoxearUrls: {{resolveAppUrl:(p)=>String(p)}}, CodoxearStorage:storage }} }};
        vm.createContext(ctx); for (const source of {json.dumps(sources)}) vm.runInContext(source,ctx);
        let currentBackend={json.dumps(backend)}, provider={json.dumps("" if backend == "pi" else "chatgpt")}, literal="", absent=false;
        const modelInput={{value:{json.dumps(query)},focus(){{}},setSelectionRange(){{}}}}, noop=()=>{{}}, cls={{toggle(){{}},remove(){{}}}}, node={{innerHTML:"",appendChild(){{}}}};
        const c=ctx.window.CodoxearNewSession.createNewSessionController({{
          backend:()=>currentBackend, provider:()=>provider, reasoningEffort:()=>"high", literalModelInputValue:()=>literal, launchPresetProviderAbsent:()=>absent,
          defaultsSource:()=>({{model:"gpt-5.4-mini",models:["gpt-5.4","o4-mini"],model_providers:{json.dumps(providers)},provider_choices:{json.dumps(providers)},reasoning_efforts:["off","low","high"],reasoning_efforts_by_model:{{}}}}),
          latestSessions:()=>[{{agent_backend:"codex",model:"gpt-5.4",model_provider:"openai",preferred_auth_method:"chatgpt"}},{{agent_backend:"codex",model:"gpt-5.4",model_provider:"crs",preferred_auth_method:"apikey"}},{{agent_backend:"pi",model:"other",model_provider:"anthropic"}}], tmuxAvailable:()=>true,
          assignProvider:v=>provider=v, assignReasoningEffort:noop, assignLiteralModelInputValue:v=>literal=v, assignLaunchPresetProviderAbsent:v=>absent=Boolean(v), modelInput, modelField:{{classList:cls}},status:{{textContent:""}},reasoningBtn:node,setPickerButtonContent:noop,renderReasoningMenu:noop,renderModelMenu:noop,setFast:noop,setBackend:noop,setTmuxChecked:noop,applyDialogMenus:noop,closeModelMenu:noop,
          cwdInput:{{value:""}},cwdMenu:{{innerHTML:""}},cwdField:{{classList:cls}},cwdHint:{{classList:cls}},nameInput:{{value:""}},recentCwds:()=>[],cwdMenuFocus:()=>-1,assignCwdMenuFocus:noop,closeCwdMenu:noop,el:()=>({{appendChild:noop}}),resumeMenu:{{innerHTML:""}},resumeBtn:{{}},closeResumeMenu:noop,fetchResumeCandidates:async()=>({{sessions:[]}}),tmuxToggle:{{}},tmuxField:{{style:{{}}}},worktreeToggle:{{}},worktreeInput:{{value:""}},worktreeField:{{style:{{}}}},startBtn:{{}}
        }});
        const options=c.sessionModelOptions(); const filtered=c.filteredNewSessionModelOptions(); const parsed=c.parseNewSessionProviderModelInput(); if (filtered[0] || options[0]) c.selectNewSessionModel(filtered[0] || options[0]);
        process.stdout.write(JSON.stringify({{options,filtered,parsed,input:modelInput.value,provider,literal,absent}}));
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
        self.assertEqual(result["options"][0]["displayText"], "chatgpt/gpt-5.4-mini")

    def test_provider_model_filter_and_selection_render_a_single_input_value(self) -> None:
        result = eval_model_options("crs/gpt")
        self.assertEqual([(x["providerChoice"], x["model"]) for x in result["filtered"]], [("crs", "gpt-5.4")])
        self.assertEqual(result["input"], "crs/gpt-5.4")

    def test_pi_accepts_a_custom_provider_model_pair(self) -> None:
        result = eval_model_options("anthropic/claude-haiku-4-5", backend="pi", provider_choices=[])
        self.assertEqual(result["parsed"]["providerChoice"], "anthropic")
        self.assertEqual(result["parsed"]["model"], "claude-haiku-4-5")
        self.assertEqual(result["parsed"]["providerError"], "")

    def test_new_session_dropdown_visuals_prioritize_model_but_name_canonical_pair(self) -> None:
        source = APP.read_text(encoding="utf-8")
        self.assertIn('"aria-label": title,', source)
        self.assertIn('class: "fileMenuPath", text: model', source)
        self.assertIn('class: "fileMenuHint", text: item.providerChoice', source)


if __name__ == "__main__":
    unittest.main()
