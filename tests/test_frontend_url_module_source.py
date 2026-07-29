import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_URL_JS = ROOT / "codoxear" / "static" / "app_url.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_app_url_cases() -> dict:
    source = APP_URL_JS.read_text(encoding="utf-8")
    cases = [
        "http://example.test/codoxear/static/index.html",
        "http://example.test/codoxear/static/",
        "http://example.test/codoxear/",
    ]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
        const cases = {json.dumps(cases)};
        const out = {{}};
        for (const href of cases) {{
          const ctx = {{ URL, URLSearchParams, window: {{ location: {{ href }} }} }};
          vm.createContext(ctx);
          vm.runInContext(source, ctx);
          const hashLocation = {{ pathname: "/codoxear/", search: "?x=1", hash: "#session=%20abc%20&view=chat" }};
          let replaced = null;
          const cleared = ctx.window.CodoxearUrls.setSessionHash("", {{ locationLike: hashLocation, historyLike: {{ replaceState: (_state, _title, target) => {{ replaced = target; }} }} }});
          out[href] = {{
            base: ctx.window.CodoxearUrls.appBaseHref,
            api: ctx.window.CodoxearUrls.resolveAppUrl("/api/sessions"),
            logo: ctx.window.CodoxearUrls.resolveAppUrl("static/logos/pi.svg"),
            sid: ctx.window.CodoxearUrls.sessionIdFromHash(hashLocation),
            cleared,
            replaced,
          }};
        }}
        process.stdout.write(JSON.stringify(out));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendUrlModuleSource(unittest.TestCase):
    def test_url_module_resolves_prefixed_app_paths(self) -> None:
        result = eval_app_url_cases()
        for href, values in result.items():
            with self.subTest(href=href):
                self.assertEqual(values["base"], "http://example.test/codoxear/")
                self.assertEqual(values["api"], "http://example.test/codoxear/api/sessions")
                self.assertEqual(values["logo"], "http://example.test/codoxear/static/logos/pi.svg")
                self.assertEqual(values["sid"], "abc")
                self.assertEqual(values["cleared"], "/codoxear/?x=1#view=chat")
                self.assertEqual(values["replaced"], "/codoxear/?x=1#view=chat")


if __name__ == "__main__":
    unittest.main()
