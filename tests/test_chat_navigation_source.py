import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_CHAT_NAVIGATION_JS = ROOT / "codoxear" / "static" / "app_chat_navigation.js"


def eval_navigation() -> dict:
    source = APP_CHAT_NAVIGATION_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const events = {{}};
        const calls = [];
        const makeButton = () => ({{ style: {{}}, disabled: null, onclick: null }});
        const prev = makeButton(), next = makeButton();
        const userRows = [{{ id: "u1", scrollIntoView: (opts) => calls.push(["scroll", "u1", opts]) }}];
        const copyRows = [{{ id: "m1", scrollIntoView: (opts) => calls.push(["scroll", "m1", opts]) }}];
        let selected = "sid";
        let sidebarOpen = false;
        const ctx = {{ window: {{}}, document: {{ body: {{ classList: {{ contains: () => false }} }} }} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const controller = ctx.window.CodoxearChatNavigation.createChatNavigationController({{
          prevUserBtn: prev, nextUserBtn: next, getSelected: () => selected,
          loadedUserMessageRows: () => userRows, loadedCopyMessageRows: () => copyRows,
          loadedUserJumpTarget: (_rows, direction) => direction < 0 ? {{ target: null, reason: "first" }} : {{ target: userRows[0] }},
          loadedCopyJumpTarget: (_rows, direction) => direction < 0 ? {{ target: null, reason: "first" }} : {{ target: copyRows[0] }},
          getScrollTop: () => 100, prefersReducedMotion: () => false,
          pulseNavigatedRow: (row) => calls.push(["pulse", row.id]), setToast: (text) => calls.push(["toast", text]),
          openChatSearch: () => calls.push(["search"]), isTextEntryElement: (target) => target === "input",
          modalIsolationTargets: [{{ open: false }}], isModalTargetOpen: (target) => target.open,
          isSidebarOpen: () => sidebarOpen,
          addAppEvent: (_target, type, handler) => {{ events[type] = handler; }}, documentTarget: {{}},
        }});
        controller.syncButtons();
        const enabled = {{ prev: prev.disabled, next: next.disabled }};
        prev.onclick({{ preventDefault: () => calls.push(["prevent"]), stopPropagation: () => calls.push(["stop"]) }});
        next.onclick({{ preventDefault: () => calls.push(["prevent-next"]), stopPropagation: () => calls.push(["stop-next"]) }});
        events.keydown({{ key: "/", target: "body", defaultPrevented: false, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, preventDefault: () => calls.push(["key-prevent"]) }});
        events.keydown({{ key: "ArrowDown", target: "body", defaultPrevented: false, ctrlKey: false, metaKey: false, altKey: true, shiftKey: false, preventDefault: () => calls.push(["key-prevent-user"]) }});
        events.keydown({{ key: "ArrowUp", target: "input", defaultPrevented: false, ctrlKey: false, metaKey: false, altKey: true, shiftKey: false, preventDefault: () => calls.push(["blocked-prevent"]) }});
        selected = null; controller.syncButtons();
        process.stdout.write(JSON.stringify({{ enabled, disabledWithoutSession: {{ prev: prev.disabled, next: next.disabled }}, calls }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestChatNavigationSource(unittest.TestCase):
    def test_loaded_user_message_jump_buttons_live_in_chat_nav_rail(self) -> None:
        result = eval_navigation()
        self.assertEqual(result["enabled"], {"prev": False, "next": False})
        self.assertEqual(result["disabledWithoutSession"], {"prev": True, "next": True})

    def test_navigation_jumps_are_delegated_to_controller(self) -> None:
        calls = eval_navigation()["calls"]
        self.assertContains(["scroll", "u1", {"block": "start", "behavior": "auto"}], calls)
        self.assertContains(["pulse", "u1"], calls)
        self.assertContains(["search"], calls)
        self.assertNotContains(["key-prevent-user"], calls)
        self.assertNotContains(["blocked-prevent"], calls)

    def test_navigation_boundary_reports_loaded_message_limit(self) -> None:
        calls = eval_navigation()["calls"]
        self.assertContains(["prevent"], calls)
        self.assertContains(["stop"], calls)
        self.assertContains(["toast", "At first loaded user message"], calls)


if __name__ == "__main__":
    unittest.main()
