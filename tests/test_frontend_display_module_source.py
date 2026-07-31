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
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_NEW_SESSION_JS = ROOT / "codoxear" / "static" / "app_new_session.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
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
        const now = new Date();
        const todayDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 12, 34);
        const yesterdayDate = new Date(todayDate.getTime() - 86400000);
        const oldDate = new Date(todayDate.getTime() - 3 * 86400000);
        process.stdout.write(JSON.stringify({{
          tooltipAria: display.defaultButtonTooltip({{ "aria-label": " Send " }}),
          tooltipNode: display.defaultButtonTooltip({{}}, {{ textContent: " Queue " }}),
          fmtEpoch: display.fmtTs(0),
          fmtKnown: display.fmtTs(1710000000),
          ymdKnown: display.ymd(new Date(1710000000 * 1000)),
          time24Known: display.time24(new Date(1710000000 * 1000)),
          dayLabelToday: display.dayLabel(todayDate),
          dayLabelYesterday: display.dayLabel(yesterdayDate),
          dayLabelOld: display.dayLabel(oldDate),
          todayYmd: display.ymd(todayDate),
          yesterdayYmd: display.ymd(yesterdayDate),
          oldYmd: display.ymd(oldDate),
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
          chatSnippetWhitespace: display.compactChatSearchSnippet("  hello\\n\\tworld  ", ""),
          chatSnippetEmpty: display.compactChatSearchSnippet("   ", "needle"),
          chatSnippetShort: display.compactChatSearchSnippet("short text", "text"),
          chatSnippetMinLimit: display.compactChatSearchSnippet("abcdefghijklmnopqrstuvwxyz0123456789", "", 5),
          chatSnippetCentered: display.compactChatSearchSnippet("a".repeat(40) + "needle" + "b".repeat(80), "needle", 40),
          chatSnippetNoNeedle: display.compactChatSearchSnippet("x".repeat(80), "missing", 30),
          chatSnippetDefaultLimitLength: display.compactChatSearchSnippet("x".repeat(110), "", Number.NaN).length,
          chatHintNull: display.chatSearchTranscriptHint(null, "x"),
          chatHintUser: display.chatSearchTranscriptHint({{ role: "user", text: "  hi  " }}, "hi"),
          chatHintAssistant: display.chatSearchTranscriptHint({{ role: "assistant", text: "answer" }}, "ans"),
          chatHintOther: display.chatSearchTranscriptHint({{ role: "system", text: "note" }}, "note"),
          chatHintBlank: display.chatSearchTranscriptHint({{ role: "assistant", text: "   " }}, "x"),
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


def eval_new_session_cwd_filter(query: str, recent_cwds: list[str]) -> dict:
    """Exercise the controller-owned recent-cwd filtering end-to-end."""
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    new_session_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          URL,
          console,
          window: {{
            CodoxearUrls: {{ resolveAppUrl: (path) => String(path || "") }},
            CodoxearStorage: {{
              getItem: () => null,
              setItem: () => true,
              removeItem: () => true,
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(launch_source)}, ctx, {{ filename: "app_launch.js" }});
        vm.runInContext({json.dumps(display_source)}, ctx, {{ filename: "app_display.js" }});
        vm.runInContext({json.dumps(new_session_source)}, ctx, {{ filename: "app_new_session.js" }});
        const cwdInput = {{ value: {json.dumps(query)} }};
        const controller = ctx.window.CodoxearNewSession.createNewSessionController({{
          backend: () => "codex",
          provider: () => "",
          reasoningEffort: () => "high",
          literalModelInputValue: () => "",
          launchPresetProviderAbsent: () => false,
          defaultsSource: () => ({{}}),
          latestSessions: () => [],
          tmuxAvailable: () => true,
          assignProvider: () => {{}},
          assignReasoningEffort: () => {{}},
          assignLiteralModelInputValue: () => {{}},
          assignLaunchPresetProviderAbsent: () => {{}},
          modelInput: {{ value: "" }},
          modelField: {{ classList: {{ toggle() {{}}, remove() {{}} }} }},
          status: {{ textContent: "" }},
          reasoningBtn: {{}},
          setPickerButtonContent: () => {{}},
          renderReasoningMenu: () => {{}},
          renderModelMenu: () => {{}},
          setFast: () => {{}},
          setBackend: () => {{}},
          setTmuxChecked: () => {{}},
          applyDialogMenus: () => {{}},
          closeModelMenu: () => {{}},
          cwdInput,
          cwdMenu: {{ innerHTML: "" }},
          cwdField: {{ classList: {{ toggle() {{}}, remove() {{}} }} }},
          cwdHint: {{ classList: {{ toggle() {{}} }} }},
          nameInput: {{ value: "" }},
          recentCwds: () => {json.dumps(recent_cwds)},
          cwdMenuFocus: () => -1,
          assignCwdMenuFocus: () => {{}},
          closeCwdMenu: () => {{}},
          el: () => ({{ appendChild() {{}} }}),
          resumeMenu: {{ innerHTML: "" }},
          resumeBtn: {{}},
          closeResumeMenu: () => {{}},
          fetchResumeCandidates: async () => ({{ sessions: [] }}),
          tmuxToggle: {{}},
          tmuxField: {{ style: {{}} }},
          worktreeToggle: {{}},
          worktreeInput: {{ value: "" }},
          worktreeField: {{ style: {{}} }},
          startBtn: {{}},
        }});
        process.stdout.write(JSON.stringify({{
          options: controller.renderRecentCwdOptions(),
          filtered: controller.filteredRecentCwdOptions(),
        }}));
        """
    )
    return run_node_json(js)


class TestFrontendDisplayModuleSource(unittest.TestCase):
    def test_display_module_preserves_presentation_helpers(self) -> None:
        result = eval_display_module()
        self.assertEqual(result["tooltipAria"], "Send")
        self.assertEqual(result["tooltipNode"], "Queue")
        self.assertEqual(result["fmtEpoch"], "1970-01-01 00:00")
        self.assertEqual(result["fmtKnown"], "2024-03-09 16:00")
        self.assertEqual(result["ymdKnown"], "2024-03-09")
        self.assertEqual(result["time24Known"], "16:00")
        self.assertEqual(result["dayLabelToday"], f"Today ({result['todayYmd']})")
        self.assertEqual(result["dayLabelYesterday"], f"Yesterday ({result['yesterdayYmd']})")
        self.assertEqual(result["dayLabelOld"], result["oldYmd"])
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
        self.assertEqual(result["chatSnippetWhitespace"], "hello world")
        self.assertEqual(result["chatSnippetEmpty"], "")
        self.assertEqual(result["chatSnippetShort"], "short text")
        self.assertEqual(result["chatSnippetMinLimit"], "abcdefghijklmnopqrstuvwx…")
        self.assertEqual(result["chatSnippetCentered"], "…aaaaaaaaaaaaaaaaaaaaaaaaneedlebbbbbbbbb…")
        self.assertEqual(result["chatSnippetNoNeedle"], "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx…")
        self.assertEqual(result["chatSnippetDefaultLimitLength"], 97)
        self.assertEqual(result["chatHintNull"], "")
        self.assertEqual(result["chatHintUser"], "user: hi")
        self.assertEqual(result["chatHintAssistant"], "assistant: answer")
        self.assertEqual(result["chatHintOther"], "match: note")
        self.assertEqual(result["chatHintBlank"], "")
        self.assertTrue(result["iconKnown"])
        self.assertEqual(result["iconUnknown"], "")
        self.assertTrue(result["frozen"])

    def test_recent_cwd_filter_lives_in_new_session_controller(self) -> None:
        # No-query path preserves recent-cwd ordering and caps at 10.
        blank = eval_new_session_cwd_filter("", ["/a", "/b", "/c"])
        self.assertEqual([item["cwd"] for item in blank["filtered"]], ["/a", "/b", "/c"])
        self.assertEqual(blank["options"], ["/a", "/b", "/c"])
        # Subsequence fuzzy match against the basename.
        matched = eval_new_session_cwd_filter("proj", ["/tmp/project", "/tmp/other"])
        self.assertEqual([item["cwd"] for item in matched["filtered"]], ["/tmp/project"])
        # Non-matching query yields an empty filtered list while options remain.
        missed = eval_new_session_cwd_filter("zzz", ["/tmp/project"])
        self.assertEqual(missed["filtered"], [])
        self.assertEqual(missed["options"], ["/tmp/project"])


if __name__ == "__main__":
    unittest.main()
