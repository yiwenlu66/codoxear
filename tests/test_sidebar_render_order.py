"""The browser renders the server's sidebar order verbatim.

Order has exactly one writer (`session_listing.sort_session_rows`, which applies
the rendered group rank before priority/recency). These tests pin both halves of
the client contract: the refresh controller must not re-sort, and the sidebar's
group partition must preserve the input order inside each block.
"""

from __future__ import annotations

import json
import subprocess
import textwrap

from frontend_module_loader import module_path

HELPERS_JS = module_path("app_session_helpers.js")
REFRESH_JS = module_path("app_session_refresh.js")
CATALOG_JS = module_path("app_session_catalog.js")


def run_node(driver: str) -> object:
    script = textwrap.dedent(
        """
        const vm = require("vm");
        const sandbox = { window: {}, console };
        vm.createContext(sandbox);
        const sources = [process.env.CATALOG_JS, process.env.HELPERS_JS, process.env.REFRESH_JS];
        for (const file of sources) {
          vm.runInContext(require("fs").readFileSync(file, "utf8"), sandbox, { filename: file });
        }
        // The driver runs outside the context; expose the same window object.
        globalThis.window = sandbox.window;
        """
    ) + driver + "\n"
    proc = subprocess.run(
        ["node", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            "CATALOG_JS": str(CATALOG_JS),
            "HELPERS_JS": str(HELPERS_JS),
            "REFRESH_JS": str(REFRESH_JS),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_refresh_renders_the_server_order_without_resorting() -> None:
    rendered = run_node(
        textwrap.dedent(
            """
            (async () => {
              const serverSessions = [
                // Deliberately not sorted by priority: the server already applied
                // the rendered order and the client must not second-guess it.
                { session_id: "third", final_priority: 0.05, updated_ts: 1 },
                { session_id: "first", final_priority: 1.0, updated_ts: 3 },
                { session_id: "second", final_priority: 0.6, updated_ts: 2 },
              ];
              const catalog = window.CodoxearSessionCatalog.createSessionCatalog({ consoleError: () => {} });
              const sessionState = { get: () => null };
              let received = null;
              const controller = window.CodoxearSessionRefresh.createSessionRefreshController({
                api: async () => ({ sessions: serverSessions, new_session_defaults: {}, tmux_available: true, recent_cwds: [] }),
                isDisposed: () => false,
                apiResponseNotModified: () => false,
                sessionCatalog: catalog,
                emptyDefaults: () => ({ default_backend: "pi", backends: { codex: null, pi: null, cc: null } }),
                clearFileDiscoveryCaches: () => {},
                useDesktopSessionActions: () => true,
                sessionState,
                clearSelectedSessionAfterRemoval: () => {},
                applySessionListTranscriptIdentity: () => {},
                syncAttachments: () => {},
                clearAttachments: () => {},
                renderSessions: (sessions) => {
                  received = sessions.map((session) => session.session_id);
                  return true;
                },
                hasDeferredRefresh: () => false,
                updateTypingStats: () => {},
                maybeSelectPendingHashSession: () => {},
              });
              await controller.refreshSessions();
              console.log(JSON.stringify(received));
            })().catch((error) => {
              console.error(error && error.stack ? error.stack : error);
              process.exit(1);
            });
            """
        )
    )

    assert rendered == ["third", "first", "second"]


def test_sidebar_entries_emit_group_blocks_in_rank_order_and_keep_input_order() -> None:
    entries = run_node(
        textwrap.dedent(
            """
            const helpers = window.CodoxearSessionHelpers;
            const sessions = [
              { session_id: "n1" },
              { session_id: "w1", blocked: true },
              { session_id: "n2" },
              { session_id: "l1", snoozed: true },
              { session_id: "w2", blocked: true },
              { session_id: "both", blocked: true, snoozed: true },
            ];
            const projected = helpers.sidebarSessionEntries(sessions).map((entry) =>
              entry.type === "header" ? `H:${entry.key}:${entry.count}` : entry.session.session_id
            );
            console.log(JSON.stringify(projected));
            """
        )
    )

    # Group blocks in rendering order; rows keep the order the server handed us,
    # and a blocked+snoozed session waits (blocked wins), matching
    # `session_listing.sidebar_group_key`.
    assert entries == ["H:now:2", "n1", "n2", "H:waiting:3", "w1", "w2", "both", "H:later:1", "l1"]


def test_server_group_rank_matches_the_client_group_labels() -> None:
    from codoxear.session_listing import SIDEBAR_GROUP_RANKS, sidebar_group_key

    labels = run_node(
        textwrap.dedent(
            """
            const groups = window.CodoxearSessionHelpers.sidebarSessionEntries([
              { session_id: "n" },
              { session_id: "w", blocked: true },
              { session_id: "l", snoozed: true },
            ]).filter((entry) => entry.type === "header").map((entry) => `${entry.key}:${entry.label}`);
            console.log(JSON.stringify(groups));
            """
        )
    )

    assert labels == ["now:Now", "waiting:Waiting", "later:Later"]
    assert list(SIDEBAR_GROUP_RANKS) == ["now", "waiting", "later"]
    assert [SIDEBAR_GROUP_RANKS[key] for key in SIDEBAR_GROUP_RANKS] == [0, 1, 2]
    assert sidebar_group_key({"blocked": True, "snoozed": True}) == "waiting"
    assert sidebar_group_key({"snoozed": True}) == "later"
    assert sidebar_group_key({}) == "now"
