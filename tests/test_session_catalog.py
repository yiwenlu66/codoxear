from frontend_module_loader import module_path
import json
import subprocess
import textwrap


APP_SESSION_CATALOG_JS = module_path("app_session_catalog.js")


def evaluate(script: str) -> dict:
    source = APP_SESSION_CATALOG_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const createSessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog;
        {script}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def test_session_catalog_initial_field_inventory() -> None:
    result = evaluate(
        """
        const catalog = createSessionCatalog({ consoleError: () => {} });
        process.stdout.write(JSON.stringify({
          latestSessions: catalog.get("latestSessions"),
          sessionIndexSize: catalog.get("sessionIndex").size,
          recentCwds: catalog.get("recentCwds"),
          newSessionDefaults: catalog.get("newSessionDefaults"),
          tmuxAvailable: catalog.get("tmuxAvailable"),
        }));
        """
    )
    assert result == {
        "latestSessions": [],
        "sessionIndexSize": 0,
        "recentCwds": [],
        "newSessionDefaults": {
            "default_backend": "pi",
            "backends": {"codex": None, "pi": None, "cc": None},
        },
        "tmuxAvailable": False,
    }


def test_latest_sessions_atomically_derives_session_index() -> None:
    result = evaluate(
        """
        const catalog = createSessionCatalog({ consoleError: () => {} });
        const calls = [];
        const shared = (_value, field) => calls.push([field, catalog.get("latestSessions").length, catalog.get("sessionIndex").size]);
        catalog.subscribe("latestSessions", shared);
        catalog.subscribe("sessionIndex", shared);
        const source = [{ session_id: "a", model: "one" }, { session_id: "b", model: "two" }];
        const changed = catalog.set("latestSessions", source);
        source.push({ session_id: "outside" });
        process.stdout.write(JSON.stringify({
          changed,
          ids: Array.from(catalog.get("sessionIndex").keys()),
          model: catalog.get("sessionIndex").get("b").model,
          latestLength: catalog.get("latestSessions").length,
          calls,
        }));
        """
    )
    assert result == {
        "changed": True,
        "ids": ["a", "b"],
        "model": "two",
        "latestLength": 2,
        "calls": [["latestSessions", 2, 2]],
    }


def test_patch_session_preserves_shared_record_and_notifies_both_catalog_views_once() -> None:
    result = evaluate(
        """
        const catalog = createSessionCatalog({ consoleError: () => {} });
        const session = { session_id: "a", model: "one" };
        catalog.set("latestSessions", [session]);
        const calls = [];
        const shared = (value, field) => calls.push({
          field,
          sameRecord: catalog.get("latestSessions")[0] === catalog.get("sessionIndex").get("a"),
          model: catalog.get("sessionIndex").get("a").model,
          deliveredRecord: field === "sessionIndex" ? value.get("a").model : value[0].model,
        });
        catalog.subscribe("sessionIndex", shared);
        catalog.subscribe("latestSessions", shared);
        const patched = catalog.patchSession("a", { model: "two", commit_unknown_send: true });
        const missing = catalog.patchSession("missing", { model: "none" });
        process.stdout.write(JSON.stringify({
          patchedIsSource: patched === session,
          missing,
          source: session,
          calls,
        }));
        """
    )
    assert result == {
        "patchedIsSource": True,
        "missing": None,
        "source": {"session_id": "a", "model": "two", "commit_unknown_send": True},
        "calls": [{"field": "sessionIndex", "sameRecord": True, "model": "two", "deliveredRecord": "two"}],
    }


def test_apply_snapshot_installs_all_fields_before_deduplicated_notification() -> None:
    result = evaluate(
        """
        const catalog = createSessionCatalog({ consoleError: () => {} });
        const calls = [];
        const shared = (_value, field) => calls.push({
          field,
          ids: Array.from(catalog.get("sessionIndex").keys()),
          defaults: catalog.get("newSessionDefaults").generation,
          tmux: catalog.get("tmuxAvailable"),
          cwds: catalog.get("recentCwds").slice(),
        });
        for (const field of ["latestSessions", "sessionIndex", "newSessionDefaults", "tmuxAvailable", "recentCwds"]) {
          catalog.subscribe(field, shared);
        }
        const affected = catalog.applySnapshot({
          latestSessions: [{ session_id: "next" }],
          newSessionDefaults: { generation: 2 },
          tmuxAvailable: true,
          recentCwds: ["/next"],
        });
        process.stdout.write(JSON.stringify({ affected, calls }));
        """
    )
    assert result == {
        "affected": ["latestSessions", "newSessionDefaults", "tmuxAvailable", "recentCwds", "sessionIndex"],
        "calls": [{"field": "latestSessions", "ids": ["next"], "defaults": 2, "tmux": True, "cwds": ["/next"]}],
    }


def test_session_index_rejects_independent_writes_and_unknown_fields() -> None:
    result = evaluate(
        """
        const catalog = createSessionCatalog({ consoleError: () => {} });
        function errorFor(action) {
          try { action(); return null; } catch (error) { return { name: error.name, message: error.message }; }
        }
        process.stdout.write(JSON.stringify({
          derived: errorFor(() => catalog.set("sessionIndex", new Map())),
          get: errorFor(() => catalog.get("sessions")),
          set: errorFor(() => catalog.set("sessions", [])),
          subscribe: errorFor(() => catalog.subscribe("sessions", () => {})),
        }));
        """
    )
    assert result["derived"] == {"name": "TypeError", "message": "session catalog field is derived: sessionIndex"}
    for key in ("get", "set", "subscribe"):
        assert result[key] == {"name": "TypeError", "message": "unknown session catalog field: sessions"}


def test_catalog_subscriptions_change_only_and_isolate_errors() -> None:
    result = evaluate(
        """
        const errors = [];
        const catalog = createSessionCatalog({ consoleError: (error) => errors.push(error.message) });
        const calls = [];
        catalog.subscribe("tmuxAvailable", () => { throw new Error("failed subscriber"); });
        const unsubscribe = catalog.subscribe("tmuxAvailable", (value, field) => calls.push([value, field]));
        const first = catalog.set("tmuxAvailable", true);
        const same = catalog.set("tmuxAvailable", 1);
        unsubscribe();
        const second = catalog.set("tmuxAvailable", false);
        process.stdout.write(JSON.stringify({ first, same, second, calls, errors }));
        """
    )
    assert result == {
        "first": True,
        "same": False,
        "second": True,
        "calls": [[True, "tmuxAvailable"]],
        "errors": ["failed subscriber", "failed subscriber"],
    }
