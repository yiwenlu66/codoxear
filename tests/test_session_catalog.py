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


def test_attachment_revision_rejects_old_and_legacy_reads_but_accepts_newer_delete() -> None:
    result = evaluate(
        """
        const catalog = createSessionCatalog({ storage: null });
        const initial = { session_id: "a", start_ts: 10, busy: true, queue_len: 4,
          attachments_revision: 0, staged_attachments: [], pending_attachment: false };
        catalog.set("latestSessions", [initial]);
        catalog.patchSession("a", { attachments_revision: 2, staged_attachments: [{ id: "file" }], pending_attachment: true });
        catalog.set("latestSessions", [{ ...initial, busy: false, queue_len: 0, attachments_revision: 1, staged_attachments: [], pending_attachment: false }]);
        const stale = JSON.parse(JSON.stringify(catalog.get("sessionIndex").get("a")));
        catalog.patchSession("a", { pending_attachment: false, staged_attachments: [], model: "new" });
        const legacy = JSON.parse(JSON.stringify(catalog.get("sessionIndex").get("a")));
        catalog.set("latestSessions", [{ session_id: "a", start_ts: 10, attachments_revision: 3, staged_attachments: [], pending_attachment: false }]);
        const deleted = catalog.get("sessionIndex").get("a");
        process.stdout.write(JSON.stringify({ stale, legacy, deleted }));
        """
    )
    assert result["stale"]["staged_attachments"] == [{"id": "file"}]
    assert result["stale"]["attachments_revision"] == 2
    assert result["stale"]["pending_attachment"] is True
    assert result["stale"]["busy"] is False
    assert result["stale"]["queue_len"] == 0
    assert result["legacy"]["pending_attachment"] is True
    assert result["legacy"]["model"] == "new"
    assert result["deleted"]["attachments_revision"] == 3
    assert result["deleted"]["staged_attachments"] == []
    assert result["deleted"]["pending_attachment"] is False


def test_reload_checkpoint_preserves_ack_and_deletion_against_old_http_cache() -> None:
    result = evaluate(
        """
        const saved = new Map();
        const storage = { getItem: (key) => saved.get(key), setItem: (key, value) => saved.set(key, value) };
        const cached = { session_id: "a", start_ts: 11, attachments_revision: 0, staged_attachments: [], pending_attachment: false };
        let catalog = createSessionCatalog({ storage });
        catalog.set("latestSessions", [{ ...cached }]);
        catalog.patchSession("a", { attachments_revision: 1, staged_attachments: [{ id: "f", display_name: "report", private_path: "/secret", data: "bytes" }], pending_attachment: true });
        const stored = JSON.parse(Array.from(saved.values())[0]);
        catalog = createSessionCatalog({ storage });
        catalog.set("latestSessions", [{ ...cached, busy: false }]);
        const reloaded = JSON.parse(JSON.stringify(catalog.get("sessionIndex").get("a")));
        catalog.patchSession("a", { attachments_revision: 2, staged_attachments: [], pending_attachment: false });
        catalog = createSessionCatalog({ storage });
        catalog.set("latestSessions", [{ ...cached, attachments_revision: 1, staged_attachments: [{ id: "f" }], pending_attachment: true }]);
        const deleted = JSON.parse(JSON.stringify(catalog.get("sessionIndex").get("a")));
        catalog.set("latestSessions", [{ ...cached, start_ts: 12 }]);
        const reusedPid = catalog.get("sessionIndex").get("a");
        process.stdout.write(JSON.stringify({ stored, reloaded, deleted, reusedPid }));
        """
    )
    assert result["stored"][0]["staged_attachments"] == [{"id": "f", "display_name": "report"}]
    assert result["reloaded"]["staged_attachments"] == [{"id": "f", "display_name": "report"}]
    assert result["reloaded"]["pending_attachment"] is True
    assert result["reloaded"]["busy"] is False
    assert result["deleted"]["attachments_revision"] == 2
    assert result["deleted"]["staged_attachments"] == []
    assert result["reusedPid"]["attachments_revision"] == 0
    assert result["reusedPid"]["staged_attachments"] == []


def test_checkpoint_storage_denial_and_corruption_leave_live_revision_reducer_usable() -> None:
    result = evaluate(
        """
        const results = [];
        for (const storage of [
          { getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("denied"); } },
          { getItem: () => "corrupt", setItem: () => { throw new Error("quota"); } },
        ]) {
          const catalog = createSessionCatalog({ storage });
          catalog.set("latestSessions", [{ session_id: "a", start_ts: 11, attachments_revision: 1, staged_attachments: [{ id: "f" }] }]);
          catalog.set("latestSessions", [{ session_id: "a", start_ts: 11, attachments_revision: 0, staged_attachments: [] }]);
          results.push(catalog.get("sessionIndex").get("a").staged_attachments);
        }
        process.stdout.write(JSON.stringify({ results }));
        """
    )
    assert result["results"] == [[{"id": "f"}], [{"id": "f"}]]


def test_checkpoint_is_bounded_and_only_reuses_matching_incarnation() -> None:
    result = evaluate(
        """
        let text = "[]";
        const storage = { getItem: () => text, setItem: (key, value) => { text = value; } };
        const catalog = createSessionCatalog({ storage });
        catalog.set("latestSessions", Array.from({ length: 110 }, (_, i) => ({ session_id: `s${i}`, start_ts: i, attachments_revision: 2, staged_attachments: [{ id: "f" }] })));
        const saved = JSON.parse(text);
        const reloaded = createSessionCatalog({ storage });
        reloaded.set("latestSessions", [{ session_id: "s109", attachments_revision: 0, staged_attachments: [] }]);
        process.stdout.write(JSON.stringify({ count: saved.length, first: saved[0].session_id, size: text.length, unscoped: reloaded.get("latestSessions")[0] }));
        """
    )
    assert result["count"] == 100
    assert result["first"] == "s10"
    assert result["size"] <= 262144
    assert result["unscoped"]["staged_attachments"] == []
