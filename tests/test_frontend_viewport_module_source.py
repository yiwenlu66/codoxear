import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_VIEWPORT_JS = ROOT / "codoxear" / "static" / "app_viewport.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_viewport(query_matches: dict[str, bool], with_match_media: bool = True) -> dict:
    source = APP_VIEWPORT_JS.read_text(encoding="utf-8")
    match_media = "matchMedia: (query) => ({ matches: Boolean(queryMatches[query]) })," if with_match_media else ""
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const queryMatches = {json.dumps(query_matches)};
        const ctx = {{ window: {{ {match_media} }} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearViewport;
        const mobile = helpers.isMobile();
        process.stdout.write(JSON.stringify({{
          mobile: mobile === undefined ? null : mobile,
          reduced: helpers.prefersReducedMotion(),
          desktopActions: helpers.useDesktopSessionActions(),
          touchControls: helpers.useTouchFileEditorControls(),
          frozen: Object.isFrozen(helpers),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_app_viewport_guard(setup_js: str = "") -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const codoxearViewport = window.CodoxearViewport;")
    end = source.index("const codoxearPolling = window.CodoxearPolling;", start)
    guard_source = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        try {{
          vm.runInContext({json.dumps(setup_js + "\n" + guard_source)}, ctx);
          process.stdout.write(JSON.stringify({{ ok: true, message: "" }}));
        }} catch (err) {{
          process.stdout.write(JSON.stringify({{ ok: false, message: String(err && err.message || err) }}));
        }}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendViewportModuleSource(unittest.TestCase):
    def test_index_loads_viewport_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_viewport.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_viewport.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_viewport.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_viewport_helpers_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_VIEWPORT_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearViewport = window.CodoxearViewport;", source)
        self.assertIn('throw new Error("Codoxear viewport helpers failed to load")', source)
        for helper in ["isMobile", "prefersReducedMotion", "useDesktopSessionActions", "useTouchFileEditorControls"]:
            self.assertIn(f"typeof codoxearViewport.{helper} !== \"function\"", source)
            self.assertIn(f"function {helper}", source)
        self.assertIn("window.CodoxearViewport = Object.freeze({", helper_source)
        self.assertIn('window.matchMedia && window.matchMedia("(max-width: 880px)").matches', helper_source)
        self.assertIn('mediaQueryMatches("(prefers-reduced-motion: reduce)")', helper_source)
        self.assertIn('mediaQueryMatches("(hover: hover) and (pointer: fine) and (min-width: 881px)")', helper_source)
        self.assertIn('mediaQueryMatches("(pointer: coarse)") || mediaQueryMatches("(hover: none)")', helper_source)
        self.assertNotIn('window.matchMedia && window.matchMedia("(max-width: 880px)").matches', source)
        self.assertNotIn('window.matchMedia("(pointer: coarse)").matches || window.matchMedia("(hover: none)").matches', source)

    def test_app_viewport_guard_throws_for_missing_or_partial_helper(self) -> None:
        missing = run_app_viewport_guard()
        self.assertEqual(missing, {"ok": False, "message": "Codoxear viewport helpers failed to load"})
        partial = run_app_viewport_guard(
            "window.CodoxearViewport = { isMobile() {}, prefersReducedMotion() {}, useDesktopSessionActions() {} };"
        )
        self.assertEqual(partial, {"ok": False, "message": "Codoxear viewport helpers failed to load"})
        complete = run_app_viewport_guard(
            "window.CodoxearViewport = { isMobile() {}, prefersReducedMotion() {}, useDesktopSessionActions() {}, useTouchFileEditorControls() {} };"
        )
        self.assertEqual(complete, {"ok": True, "message": ""})

    def test_viewport_queries_preserve_media_contracts(self) -> None:
        result = eval_viewport(
            {
                "(max-width: 880px)": True,
                "(prefers-reduced-motion: reduce)": True,
                "(hover: hover) and (pointer: fine) and (min-width: 881px)": True,
                "(pointer: coarse)": False,
                "(hover: none)": True,
            }
        )
        self.assertTrue(result["mobile"])
        self.assertTrue(result["reduced"])
        self.assertTrue(result["desktopActions"])
        self.assertTrue(result["touchControls"])
        self.assertTrue(result["frozen"])

    def test_missing_match_media_preserves_existing_falsy_contracts(self) -> None:
        result = eval_viewport({}, with_match_media=False)
        self.assertEqual(
            result,
            {"mobile": None, "reduced": False, "desktopActions": False, "touchControls": False, "frozen": True},
        )

    def test_touch_controls_accept_either_coarse_pointer_or_no_hover(self) -> None:
        self.assertTrue(eval_viewport({"(pointer: coarse)": True, "(hover: none)": False})["touchControls"])
        self.assertTrue(eval_viewport({"(pointer: coarse)": False, "(hover: none)": True})["touchControls"])
        self.assertFalse(eval_viewport({"(pointer: coarse)": False, "(hover: none)": False})["touchControls"])


if __name__ == "__main__":
    unittest.main()
