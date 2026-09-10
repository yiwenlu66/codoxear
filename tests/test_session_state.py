from frontend_module_loader import module_path
import json
import subprocess
import textwrap


APP_SESSION_STATE_JS = module_path("app_session_state.js")


def evaluate(script: str) -> dict:
    source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const createSessionState = ctx.window.CodoxearSessionState.createSessionState;
        {script}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def test_session_state_initial_values_and_set_get_roundtrip() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        const initial = ["selected", "running", "queueLen", "subagentsRunning", "subagentDetails", "turnOpen", "sending", "token"]
          .reduce((out, field) => ({ ...out, [field]: state.get(field) }), {});
        const updates = {
          selected: { id: "session-a" },
          running: true,
          queueLen: 2,
          subagentsRunning: 1,
          subagentDetails: [{ role: "reviewer", tools: 2 }],
          turnOpen: true,
          sending: true,
          token: { context_window: 128000 },
        };
        const changes = Object.fromEntries(Object.entries(updates).map(([field, value]) => [field, state.set(field, value)]));
        const current = Object.keys(updates).reduce((out, field) => ({ ...out, [field]: state.get(field) }), {});
        process.stdout.write(JSON.stringify({ initial, changes, current }));
        """
    )
    assert result["initial"] == {
        "selected": None,
        "running": False,
        "queueLen": 0,
        "subagentsRunning": 0,
        "subagentDetails": [],
        "turnOpen": False,
        "sending": False,
        "token": None,
    }
    assert result["changes"] == {
        "selected": True,
        "running": True,
        "queueLen": True,
        "subagentsRunning": True,
        "subagentDetails": True,
        "turnOpen": True,
        "sending": True,
        "token": True,
    }
    assert result["current"] == {
        "selected": {"id": "session-a"},
        "running": True,
        "queueLen": 2,
        "subagentsRunning": 1,
        "subagentDetails": [{"role": "reviewer", "tools": 2}],
        "turnOpen": True,
        "sending": True,
        "token": {"context_window": 128000},
    }


def test_session_state_rejects_unknown_fields_in_every_field_api() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        function errorFor(action) {
          try { action(); return null; } catch (error) { return { name: error.name, message: error.message }; }
        }
        process.stdout.write(JSON.stringify({
          get: errorFor(() => state.get("selectedSession")),
          set: errorFor(() => state.set("selectedSession", "session-a")),
          subscribe: errorFor(() => state.subscribe("selectedSession", () => {})),
          applyRuntime: errorFor(() => state.applyRuntime({ selectedSession: "session-a" })),
        }));
        """
    )
    for error in result.values():
        assert error["name"] == "TypeError"
        assert "selectedSession" in error["message"]


def test_session_state_subscribers_fire_only_for_changes() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        const calls = [];
        state.subscribe("running", (value, field) => calls.push([value, field]));
        const first = state.set("running", true);
        const same = state.set("running", true);
        const second = state.set("running", false);
        process.stdout.write(JSON.stringify({ first, same, second, calls }));
        """
    )
    assert result == {
        "first": True,
        "same": False,
        "second": True,
        "calls": [[True, "running"], [False, "running"]],
    }


def test_session_state_apply_runtime_batches_per_changed_field() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        const calls = [];
        for (const field of ["selected", "running", "queueLen", "sending"]) {
          state.subscribe(field, (value, deliveredField) => calls.push([deliveredField, value]));
        }
        state.set("running", true);
        calls.length = 0;
        const changed = state.applyRuntime({
          selected: { id: "session-a" },
          running: true,
          queueLen: 3,
          sending: true,
        });
        process.stdout.write(JSON.stringify({ changed, calls, values: {
          selected: state.get("selected"),
          running: state.get("running"),
          queueLen: state.get("queueLen"),
          sending: state.get("sending"),
        }}));
        """
    )
    assert result == {
        "changed": ["selected", "queueLen", "sending"],
        "calls": [["selected", {"id": "session-a"}], ["queueLen", 3], ["sending", True]],
        "values": {"selected": {"id": "session-a"}, "running": True, "queueLen": 3, "sending": True},
    }


def test_session_state_apply_runtime_notifies_shared_subscriber_once() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        const calls = [];
        const shared = (value, field) => calls.push([field, value, state.get("running"), state.get("queueLen")]);
        state.subscribe("running", shared);
        state.subscribe("queueLen", shared);
        const changed = state.applyRuntime({ running: true, queueLen: 2 });
        process.stdout.write(JSON.stringify({ changed, calls }));
        """
    )
    assert result == {
        "changed": ["running", "queueLen"],
        "calls": [["running", True, True, 2]],
    }


def test_session_state_subagent_snapshot_is_atomic_and_detail_refreshes_at_same_count() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        const calls = [];
        const shared = (_value, field) => calls.push({
          field,
          count: state.get("subagentsRunning"),
          tools: state.get("subagentDetails").map((detail) => detail.tools),
          tokens: state.get("subagentDetails").map((detail) => detail.tokens),
        });
        state.subscribe("subagentsRunning", shared);
        state.subscribe("subagentDetails", shared);
        const first = state.applyRuntime({
          subagentsRunning: 2,
          subagentDetails: [{ role: "reviewer", tools: 2, tokens: 1200 }, { role: "worker", tools: 4, tokens: 2400 }],
        });
        const second = state.applyRuntime({
          subagentsRunning: 2,
          subagentDetails: [{ role: "reviewer", tools: 3, tokens: 1300 }, { role: "worker", tools: 5, tokens: 2500 }],
        });
        process.stdout.write(JSON.stringify({ first, second, calls }));
        """
    )
    assert result == {
        "first": ["subagentsRunning", "subagentDetails"],
        "second": ["subagentDetails"],
        "calls": [
            {"field": "subagentsRunning", "count": 2, "tools": [2, 4], "tokens": [1200, 2400]},
            {"field": "subagentDetails", "count": 2, "tools": [3, 5], "tokens": [1300, 2500]},
        ],
    }


def test_session_state_unsubscribe_stops_delivery() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        const calls = [];
        const unsubscribe = state.subscribe("queueLen", (value) => calls.push(value));
        state.set("queueLen", 1);
        const first = unsubscribe();
        const second = unsubscribe();
        state.set("queueLen", 2);
        process.stdout.write(JSON.stringify({ calls, first, second }));
        """
    )
    assert result == {"calls": [1], "first": True, "second": False}


def test_session_state_isolates_subscriber_errors_and_reports_them() -> None:
    result = evaluate(
        """
        const errors = [];
        const state = createSessionState({ consoleError: (error) => errors.push(error.message) });
        const calls = [];
        state.subscribe("token", () => { throw new Error("subscriber failed"); });
        state.subscribe("token", (value, field) => calls.push([value, field]));
        state.set("token", { context_window: 128000 });
        process.stdout.write(JSON.stringify({ errors, calls }));
        """
    )
    assert result == {
        "errors": ["subscriber failed"],
        "calls": [[{"context_window": 128000}, "token"]],
    }


def test_session_state_inventory_has_no_selected_session_alias() -> None:
    result = evaluate(
        """
        const state = createSessionState();
        let error = null;
        try { state.get("selectedSession"); } catch (caught) { error = { name: caught.name, message: caught.message }; }
        process.stdout.write(JSON.stringify(error));
        """
    )
    assert result == {"name": "TypeError", "message": "unknown session state field: selectedSession"}
