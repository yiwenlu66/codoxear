"""Behavioral regression for sequential unattended-mode edits.

The browser controller owns the debounce/in-flight boundary.  This test holds
config A's POST open, sends config B through the same UI handlers, then releases
A and verifies that the endpoint observes A followed by B's merged latest
configuration.  A stale completion must not repaint B away.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_FRONTEND_TEST = Path(__file__).with_name("test_frontend_unattended_module_source.py")
_SPEC = importlib.util.spec_from_file_location("frontend_unattended_harness", _FRONTEND_TEST)
assert _SPEC is not None and _SPEC.loader is not None
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)


def test_in_flight_config_a_finishes_before_pending_config_b_applies() -> None:
    result = _HARNESS.run_node_json(
        _HARNESS.harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", {
              launch_state: "ready",
              unattended_enabled: false,
              unattended_cooldown_minutes: 5,
              unattended_remaining_injections: 10,
            });
            h.select("sid-1");

            // Config A: change the cooldown. Keep its POST in flight.
            h.dom.cooldownEl.value = "7";
            h.dom.cooldownEl.oninput({ target: h.dom.cooldownEl });
            let resolveA;
            const configA = new Promise((resolve) => { resolveA = resolve; });
            h.setApiResponses([
              { __promise: configA },
              { enabled: false, request: "Config B", cooldown_minutes: 7, remaining_injections: 10 },
            ]);
            h.runNextTimer();
            const postsWhileAInFlight = h.calls.filter((c) => c[0] === "api" && c[2]).map((c) => c[2]);

            // Config B arrives before A resolves. Its debounce timer cannot
            // start another request while A owns the in-flight boundary.
            h.dom.requestEl.value = "Config B";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            h.runPendingTimers();
            const postsBeforeACompletes = h.calls.filter((c) => c[0] === "api" && c[2]).map((c) => c[2]);

            // A commits first. The pending merged B snapshot must then drain
            // and become the final visible configuration.
            resolveA({ enabled: false, request: "", cooldown_minutes: 7, remaining_injections: 10 });
            await new Promise((resolve) => setTimeout(resolve, 0));
            const postsAfterACompletes = h.calls.filter((c) => c[0] === "api" && c[2]).map((c) => c[2]);
            globalThis.__result = {
              postsWhileAInFlight,
              postsBeforeACompletes,
              postsAfterACompletes,
              cooldown: h.dom.cooldownEl.value,
              request: h.dom.requestEl.value,
            };
            """
        )
    )

    assert result["postsWhileAInFlight"] == [{"cooldown_minutes": 7}]
    assert result["postsBeforeACompletes"] == [{"cooldown_minutes": 7}]
    assert result["postsAfterACompletes"] == [
        {"cooldown_minutes": 7},
        {"cooldown_minutes": 7, "request": "Config B"},
    ]
    assert result["cooldown"] == "7"
    assert result["request"] == "Config B"
