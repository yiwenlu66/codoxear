import json
import os
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


def eval_display_module() -> dict:
    source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const display = ctx.window.CodoxearDisplay;
        process.stdout.write(JSON.stringify({{
          tooltipAria: display.defaultButtonTooltip({{ "aria-label": " Send " }}),
          tooltipNode: display.defaultButtonTooltip({{}}, {{ textContent: " Queue " }}),
          fmtEpoch: display.fmtTs(0),
          fmtKnown: display.fmtTs(1710000000),
          bytesSmall: display.fmtBytes(12),
          bytes1023: display.fmtBytes(1023),
          bytes1024: display.fmtBytes(1024),
          bytesLarge: display.fmtBytes(1536),
          bytesMb: display.fmtBytes(1048576),
          bytesNan: display.fmtBytes(Number.NaN),
          base: display.baseName("/tmp/project/file.txt"),
          baseRoot: display.baseName("/"),
          shortUuid: display.shortSessionId("12345678abcdefabcdefabcdefabcdefabcd-42"),
          shortPlain: display.shortSessionId("plain-session"),
          aliasName: display.sessionDisplayName({{ alias: "  Alias  ", cwd: "/tmp/ignored" }}),
          cwdName: display.sessionDisplayName({{ cwd: "/tmp/project" }}),
          timeName: display.sessionDisplayName({{ updated_ts: 1710000000 }}),
          idleNegative: display.fmtIdleAge(-1),
          idle59: display.fmtIdleAge(59),
          idle60: display.fmtIdleAge(60),
          idle3599: display.fmtIdleAge(3599),
          idle3600: display.fmtIdleAge(3600),
          idle86400: display.fmtIdleAge(86400),
          relativeJustNow: display.fmtRelativeAge(5),
          relativeMinutes: display.fmtRelativeAge(125),
          relativeHour: display.fmtRelativeAge(3600),
          titleEmpty: display.sessionTitleWithId(null),
          recoveryPreviewWhitespace: display.recoveryPromptPreview("  hello\\n\\tworld  "),
          recoveryPreviewTruncated: display.recoveryPromptPreview("abcdef", 3),
          recoveryPreviewExact: display.recoveryPromptPreview("abc", 3),
          recoveryPreviewFalsy: display.recoveryPromptPreview(0),
          recoveryPreviewDefaultLimit: display.recoveryPromptPreview("x".repeat(321)).length,
          cwdScoreNoQuery: display.fuzzyRecentCwdScore("/tmp/project", "   "),
          cwdScoreExact: display.fuzzyRecentCwdScore("/tmp/project", " /TMP/PROJECT "),
          cwdScoreBaseExact: display.fuzzyRecentCwdScore("/tmp/project", "project"),
          cwdScoreBoundaryToken: display.fuzzyRecentCwdScore("/tmp/project-alpha", "project"),
          cwdScoreMultiToken: display.fuzzyRecentCwdScore("/work/foo-bar", "foo bar"),
          cwdScoreSubsequence: display.fuzzyRecentCwdScore("abc", "ac"),
          cwdScoreNoMatch: display.fuzzyRecentCwdScore("abc", "az"),
          iconKnown: display.iconSvg("send").includes("<svg"),
          iconUnknown: display.iconSvg("missing"),
          frozen: Object.isFrozen(display),
        }}));
        """
    )
    return run_node_json(js)


def eval_icon_outputs(icon_names: set[str]) -> dict:
    source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const display = ctx.window.CodoxearDisplay;
        const names = {json.dumps(sorted(icon_names))};
        const outputs = Object.fromEntries(names.map((name) => [name, display.iconSvg(name)]));
        process.stdout.write(JSON.stringify(outputs));
        """
    )
    return run_node_json(js)


class TestFrontendDisplayModuleSource(unittest.TestCase):
    def test_index_loads_display_module_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_display.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_launch.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_display.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_display.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_display_module_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearDisplay = window.CodoxearDisplay;", source)
        self.assertIn('throw new Error("Codoxear display helpers failed to load")', source)
        self.assertIn("function fmtBytes(n) {", source)
        self.assertIn("return codoxearDisplay.fmtBytes(n);", source)
        self.assertIn("function iconSvg(name) {", source)
        self.assertIn("return codoxearDisplay.iconSvg(name);", source)
        self.assertIn('typeof codoxearDisplay.recoveryPromptPreview !== "function"', source)
        self.assertIn("function recoveryPromptPreview(text, maxLen = 320)", source)
        self.assertIn("return codoxearDisplay.recoveryPromptPreview(text, maxLen);", source)
        self.assertIn('typeof codoxearDisplay.fuzzyRecentCwdScore !== "function"', source)
        self.assertIn("function fuzzyRecentCwdScore(candidate, query)", source)
        self.assertIn("return codoxearDisplay.fuzzyRecentCwdScore(candidate, query);", source)
        self.assertIn("window.CodoxearDisplay = Object.freeze({", display_source)
        recent_cwd_start = source.index("function renderRecentCwdOptions()")
        recent_cwd_end = source.index("function filteredRecentCwdOptions()", recent_cwd_start)
        recent_cwd_block = source[recent_cwd_start:recent_cwd_end]
        self.assertNotIn("function fmtBytes(n) {\n        const v = Number(n);", source)
        self.assertNotIn('const raw = String(text || "").replace(/\\s+/g, " ").trim();', source)
        self.assertNotIn("function fuzzyRecentCwdScore(candidate, query)", recent_cwd_block)
        self.assertNotIn('const raw = String(query || "").trim().toLowerCase();', recent_cwd_block)
        self.assertNotIn("function iconSvg(name) {\n    if (name ===", source)

    def test_display_module_preserves_presentation_helpers(self) -> None:
        result = eval_display_module()
        self.assertEqual(result["tooltipAria"], "Send")
        self.assertEqual(result["tooltipNode"], "Queue")
        self.assertEqual(result["fmtEpoch"], "1970-01-01 00:00")
        self.assertEqual(result["fmtKnown"], "2024-03-09 16:00")
        self.assertEqual(result["bytesSmall"], "12 B")
        self.assertEqual(result["bytes1023"], "1023 B")
        self.assertEqual(result["bytes1024"], "1.00 KB")
        self.assertEqual(result["bytesLarge"], "1.50 KB")
        self.assertEqual(result["bytesMb"], "1.00 MB")
        self.assertEqual(result["bytesNan"], "NaN")
        self.assertEqual(result["base"], "file.txt")
        self.assertEqual(result["baseRoot"], "/")
        self.assertEqual(result["shortUuid"], "12345678-42")
        self.assertEqual(result["shortPlain"], "plain-se")
        self.assertEqual(result["aliasName"], "Alias")
        self.assertEqual(result["cwdName"], "project")
        self.assertEqual(result["timeName"], "Session 2024-03-09 16:00")
        self.assertEqual(result["idleNegative"], "")
        self.assertEqual(result["idle59"], "just now")
        self.assertEqual(result["idle60"], "1m")
        self.assertEqual(result["idle3599"], "59m")
        self.assertEqual(result["idle3600"], "1h")
        self.assertEqual(result["idle86400"], "1d")
        self.assertEqual(result["relativeJustNow"], "just now")
        self.assertEqual(result["relativeMinutes"], "2m ago")
        self.assertEqual(result["relativeHour"], "1h ago")
        self.assertEqual(result["titleEmpty"], "No session selected")
        self.assertEqual(result["recoveryPreviewWhitespace"], "hello world")
        self.assertEqual(result["recoveryPreviewTruncated"], "abc…")
        self.assertEqual(result["recoveryPreviewExact"], "abc")
        self.assertEqual(result["recoveryPreviewFalsy"], "")
        self.assertEqual(result["recoveryPreviewDefaultLimit"], 321)
        self.assertEqual(result["cwdScoreNoQuery"], 0)
        self.assertEqual(result["cwdScoreExact"], 10000)
        self.assertEqual(result["cwdScoreBaseExact"], 9000)
        self.assertEqual(result["cwdScoreBoundaryToken"], 314)
        self.assertEqual(result["cwdScoreMultiToken"], 612)
        self.assertEqual(result["cwdScoreSubsequence"], 124)
        self.assertEqual(result["cwdScoreNoMatch"], -1)
        self.assertTrue(result["iconKnown"])
        self.assertEqual(result["iconUnknown"], "")
        self.assertTrue(result["frozen"])

    def test_icon_svg_covers_all_literal_and_dynamic_app_uses(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
        literal_names = set(re.findall(r'iconSvg\("([A-Za-z0-9]+)"\)', app_source))
        dynamic_session_launch_names = {"info", "tmux", "web", "terminal"}
        required_names = literal_names | dynamic_session_launch_names
        defined_names = set(re.findall(r'if \(name === "([^"]+)"\)', display_source))
        self.assertFalse(required_names - defined_names)
        outputs = eval_icon_outputs(required_names)
        for name in sorted(required_names):
            self.assertIn("<svg", outputs[name], name)
            self.assertIn("</svg>", outputs[name], name)


if __name__ == "__main__":
    unittest.main()
