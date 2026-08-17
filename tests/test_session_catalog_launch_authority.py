from frontend_module_loader import module_path
import json
import subprocess
import textwrap


CATALOG = module_path("app_session_catalog.js")
LIFECYCLE = module_path("app_session_lifecycle.js")
SESSION_STATE = module_path("app_session_state.js")


def test_launch_fast_capability_reads_refreshed_catalog_defaults() -> None:
    sources = [path.read_text(encoding="utf-8") for path in (CATALOG, SESSION_STATE, LIFECYCLE)]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        for (const source of {json.dumps(sources)}) vm.runInContext(source, ctx);
        const noop = () => {{}};
        const requests = [];
        const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
        const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
        const required = () => noop;
        const options = new Proxy({{
          sessionCatalog,
          sessionState,
          asyncEpoch: {{ currentGeneration: () => 0, nextGeneration: () => 0, incrementGeneration: noop }},
          backendSupportsFastForDefaults: (backend, defaults) => Boolean(defaults.backends[backend].supports_fast),
          normalizeAgentBackendName: (value) => value,
          providerChoiceToSettings: () => ({{}}),
          api: async (path, options = {{}}) => {{
            if (path === "/api/sessions") requests.push(options.body);
            return {{ broker_pid: requests.length }};
          }},
          refreshSessions: async () => [],
          sleep: async () => {{}},
          setToast: noop,
          consoleError: noop,
        }}, {{ get: (target, field) => field in target ? target[field] : required(field) }});
        const lifecycle = ctx.window.CodoxearSessionLifecycle.createSessionLifecycleController(options);
        (async () => {{
          sessionCatalog.set("newSessionDefaults", {{ default_backend: "pi", backends: {{ codex: {{ supports_fast: false }}, pi: {{}}, cc: {{}} }} }});
          await lifecycle.spawnSessionWithCwd("/tmp/project", null, null, "", "chatgpt", "gpt", "high", true, false, null, "codex");
          sessionCatalog.set("newSessionDefaults", {{ default_backend: "pi", backends: {{ codex: {{ supports_fast: true }}, pi: {{}}, cc: {{}} }} }});
          await lifecycle.spawnSessionWithCwd("/tmp/project", null, null, "", "chatgpt", "gpt", "high", true, false, null, "codex");
          process.stdout.write(JSON.stringify(requests));
        }})().catch((error) => {{ console.error(error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    requests = json.loads(completed.stdout)
    assert "service_tier" not in requests[0]
    assert requests[1]["service_tier"] == "fast"
