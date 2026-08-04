import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NETWORK_JS = ROOT / "codoxear" / "static" / "app_network.js"


def render_network_status(on_line: bool) -> dict:
    source = NETWORK_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const banner = {{
          textContent: "",
          hidden: true,
          attributes: {{}},
          setAttribute(name, value) {{ this.attributes[name] = String(value); }},
        }};
        const ctx = {{ window: {{}}, navigator: {{ onLine: {str(on_line).lower()} }} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const status = ctx.window.CodoxearNetwork.createNetworkStatusController({{
          banner,
          navigatorLike: ctx.navigator,
        }});
        process.stdout.write(JSON.stringify({{
          text: banner.textContent,
          hidden: banner.hidden,
          ariaHidden: banner.attributes["aria-hidden"],
          offline: status.browserOffline(ctx.navigator),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestOfflineResilience(unittest.TestCase):
    def test_offline_navigator_renders_persistent_network_banner(self) -> None:
        rendered = render_network_status(on_line=False)

        self.assertTrue(rendered["offline"])
        self.assertFalse(rendered["hidden"])
        self.assertEqual(rendered["ariaHidden"], "false")
        self.assertEqual(rendered["text"], "Offline — waiting for a network connection. Updates retry automatically.")

    def test_online_navigator_keeps_network_banner_hidden(self) -> None:
        rendered = render_network_status(on_line=True)

        self.assertFalse(rendered["offline"])
        self.assertTrue(rendered["hidden"])
        self.assertEqual(rendered["ariaHidden"], "true")
        self.assertEqual(rendered["text"], "")


if __name__ == "__main__":
    unittest.main()
