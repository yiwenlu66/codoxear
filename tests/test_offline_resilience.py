from frontend_module_loader import NETWORK_STATUS_STUB_JS, module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NETWORK_JS = module_path("app_network.js")


def render_network_status(on_line: bool, failures: int = 0) -> dict:
    source = NETWORK_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        {NETWORK_STATUS_STUB_JS}
        const ctx = {{ window: {{}}, navigator: {{ onLine: {str(on_line).lower()} }} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const net = ctx.window.CodoxearNetwork.createNetworkStatusController({{
          status,
          statusText,
          navigatorLike: ctx.navigator,
        }});
        const snapshots = [netSnap(net.sync())];
        for (let i = 0; i < {int(failures)}; i += 1) snapshots.push(netSnap(net.reportFailure()));
        process.stdout.write(JSON.stringify({{
          initial: snapshots[0],
          snapshots,
          offline: net.browserOffline(ctx.navigator),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestOfflineResilience(unittest.TestCase):
    def test_offline_navigator_shows_connection_icon_with_spoken_reason(self) -> None:
        rendered = render_network_status(on_line=False)

        self.assertTrue(rendered["offline"])
        initial = rendered["initial"]
        self.assertEqual(initial["state"], "offline")
        self.assertFalse(initial["hidden"])
        self.assertEqual(initial["title"], "Offline — waiting for a network connection. Updates retry automatically.")
        # The glyph carries no text of its own; the sentence survives for
        # assistive tech through the sr-only span inside the live region.
        self.assertEqual(initial["spoken"], initial["title"])

    def test_online_navigator_keeps_connection_icon_hidden(self) -> None:
        rendered = render_network_status(on_line=True)

        self.assertFalse(rendered["offline"])
        initial = rendered["initial"]
        self.assertEqual(initial["state"], "")
        self.assertTrue(initial["hidden"])
        self.assertIsNone(initial["title"])
        self.assertEqual(initial["spoken"], "")


class TestTransportFailureThreshold(unittest.TestCase):
    def test_single_failure_does_not_show_the_icon(self) -> None:
        rendered = render_network_status(on_line=True, failures=1)

        (initial, after_failure) = rendered["snapshots"]
        self.assertEqual(initial["state"], "")
        # One flaky poll must not flash the icon.
        self.assertEqual(after_failure["state"], "")
        self.assertTrue(after_failure["hidden"])

    def test_two_consecutive_failures_show_the_icon(self) -> None:
        rendered = render_network_status(on_line=True, failures=2)

        after_second = rendered["snapshots"][2]
        self.assertEqual(after_second["state"], "failing")
        self.assertFalse(after_second["hidden"])
        self.assertEqual(after_second["title"], "Connection unavailable — retrying automatically.")
        self.assertEqual(after_second["spoken"], after_second["title"])


if __name__ == "__main__":
    unittest.main()
