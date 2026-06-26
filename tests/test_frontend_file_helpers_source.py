import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_file_helpers_real_order() -> dict:
    scripts = [APP_DISPLAY_JS, APP_FILE_HELPERS_JS]
    js = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        for (const file of {json.dumps([str(path) for path in scripts])}) {{
          vm.runInContext(fs.readFileSync(file, "utf8"), ctx, {{ filename: file }});
        }}
        const helpers = ctx.window.CodoxearFileHelpers;
        process.stdout.write(JSON.stringify({{
          listed: helpers.listFromFilesField(["/repo/trail.md ", "/repo/new\\n.md", "", "/repo/trail.md ", 7, null]),
          listedNonArray: helpers.listFromFilesField("/repo/a"),
          hashSuffix: helpers.stripPathLocationSuffix("/repo/file.py#L12-20"),
          colonSuffix: helpers.stripPathLocationSuffix("/repo/file.py:12:3"),
          noSuffixPreservesRaw: helpers.stripPathLocationSuffix("/repo/trail.md "),
          noSuffixPreservesNewline: helpers.stripPathLocationSuffix("/repo/new\\n.md"),
          textKind: helpers.isTextFileKind("text"),
          markdownKind: helpers.isTextFileKind("markdown"),
          imageKind: helpers.isTextFileKind("image"),
          diffMarkdown: helpers.isDiffableFileKind("markdown"),
          diffBinary: helpers.isDiffableFileKind("binary"),
          tooLarge: helpers.blockedFileMessage("big.txt", "too_large", 1024, 1536),
          tooLargeZeroLimit: helpers.blockedFileMessage("big.txt", "too_large", 0, 1536),
          unsupported: helpers.blockedFileMessage("image.bin", "unsupported", 1024, 1536),
          priorityPositive: helpers.formatPriorityOffset(1.234),
          priorityNegative: helpers.formatPriorityOffset(-0.5),
          priorityInvalid: helpers.formatPriorityOffset(Number.NaN),
          videoErrorMessage: helpers.fileVideoPreviewErrorText(new Error(" bad codec ")),
          videoErrorString: helpers.fileVideoPreviewErrorText(" transcode failed \\n"),
          videoErrorBlank: helpers.fileVideoPreviewErrorText(new Error("   ")),
          videoErrorNull: helpers.fileVideoPreviewErrorText(null),
          fileScoreNoQuery: helpers.fileSearchScore("/tmp/project", "   "),
          fileScoreExact: helpers.fileSearchScore("/tmp/project", " /TMP/PROJECT "),
          fileScoreBaseExact: helpers.fileSearchScore("/tmp/project", "project"),
          fileScoreBoundaryToken: helpers.fileSearchScore("/tmp/project-alpha", "project"),
          fileScoreMultiToken: helpers.fileSearchScore("/work/foo-bar", "foo bar"),
          fileScoreSubsequence: helpers.fileSearchScore("abc", "ac"),
          fileScoreNoMatch: helpers.fileSearchScore("abc", "az"),
          normalizedDotSlash: helpers.normalizeDraftFilePath("./foo.py"),
          normalizedBackslash: helpers.normalizeDraftFilePath("foo" + "\\\\" + "bar.py"),
          normalizedParent: helpers.normalizeDraftFilePath("../foo.py"),
          normalizedAbs: helpers.normalizeDraftFilePath("/tmp/foo.py"),
          normalizedTrailing: helpers.normalizeDraftFilePath("foo/"),
          normalizedNul: helpers.normalizeDraftFilePath("foo" + String.fromCharCode(0) + "bar"),
          rangeExact: helpers.filePickerMatchRangesForQuery("src/foo_bar.py", "foo"),
          rangeFuzzy: helpers.filePickerMatchRangesForQuery("src/foo_bar.py", "fb"),
          rangeNormalized: helpers.filePickerMatchRangesForQuery("foo.py", "./foo.py"),
          rangeNone: helpers.filePickerMatchRangesForQuery("src/foo.py", "zz"),
          rangeMerged: helpers.filePickerMatchRangesForQuery("src/foo.py", "src foo"),
          turkishSlices: helpers.filePickerMatchRangesForQuery("İfoo.py", "foo").map(([start, end]) => "İfoo.py".slice(start, end)),
          emojiSlices: helpers.filePickerMatchRangesForQuery("a😀-b.txt", "😀b").map(([start, end]) => "a😀-b.txt".slice(start, end)),
          candidateNormalizedScore: helpers.filePickerCandidateScore("foo.py", "./foo.py"),
          compareScore: helpers.compareFilePickerEntries({{ path: "b", score: 1 }}, {{ path: "a", score: 2 }}),
          comparePath: helpers.compareFilePickerEntries({{ path: "a", score: 1 }}, {{ path: "b", score: 1 }}),
          compareGitPath: helpers.compareFilePickerEntries({{ path: "a", gitPath: false }}, {{ path: "a", gitPath: true }}),
          compareChanged: helpers.compareFilePickerEntries({{ path: "a", changed: true }}, {{ path: "a", changed: false }}),
          sourceChangedTrimmed: helpers.normalizeFileCandidateSource(" changed "),
          sourceMentioned: helpers.normalizeFileCandidateSource("mentioned"),
          sourceRecent: helpers.normalizeFileCandidateSource("recent"),
          sourceUnknown: helpers.normalizeFileCandidateSource("other"),
          sourceBlank: helpers.normalizeFileCandidateSource("   "),
          sectionChanged: helpers.filePickerSectionLabel("changed"),
          sectionMentioned: helpers.filePickerSectionLabel("mentioned"),
          sectionRecent: helpers.filePickerSectionLabel("recent"),
          sectionUnknown: helpers.filePickerSectionLabel("other"),
          frozen: Object.isFrozen(helpers),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFileHelpersSource(unittest.TestCase):
    def test_index_loads_file_helpers_after_display_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_file_helpers.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_display.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_file_helpers.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_file_helpers.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_file_helpers_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearFileHelpers = window.CodoxearFileHelpers;", source)
        self.assertIn('throw new Error("Codoxear file helpers failed to load")', source)
        for helper in [
            "listFromFilesField",
            "stripPathLocationSuffix",
            "isTextFileKind",
            "isDiffableFileKind",
            "blockedFileMessage",
            "formatPriorityOffset",
            "fileVideoPreviewErrorText",
            "fileSearchScore",
            "normalizeDraftFilePath",
            "filePickerFoldedSearchText",
            "filePickerOriginalRangeForFolded",
            "filePickerMatchRanges",
            "filePickerMatchRangesForQuery",
            "filePickerCandidateScore",
            "compareFilePickerEntries",
            "normalizeFileCandidateSource",
            "filePickerSectionLabel",
        ]:
            self.assertIn(f"typeof codoxearFileHelpers.{helper} !== \"function\"", source)
            self.assertIn(f"function {helper}", source)
        self.assertIn("window.CodoxearFileHelpers = Object.freeze({", helper_source)
        self.assertIn('throw new Error("Codoxear display helpers failed to load")', helper_source)
        self.assertIn('typeof codoxearDisplay.baseName !== "function"', helper_source)
        self.assertIn("return codoxearFileHelpers.fileVideoPreviewErrorText(err);", source)
        self.assertIn("return codoxearFileHelpers.filePickerMatchRangesForQuery(text, query);", source)
        self.assertIn("return codoxearFileHelpers.filePickerCandidateScore(path, query);", source)
        self.assertIn("return codoxearFileHelpers.normalizeFileCandidateSource(source);", source)
        self.assertIn("return codoxearFileHelpers.filePickerSectionLabel(source);", source)
        self.assertIn('return ["changed", "mentioned", "recent"].includes(value) ? value : "";', helper_source)
        self.assertIn('if (source === "changed") return "Changed files";', helper_source)
        file_search_region_start = source.index("function collectMessageFileRefs()")
        file_search_region_end = source.index("function appendHighlightedFileMenuPath(parent, text, query)", file_search_region_start)
        file_search_region = source[file_search_region_start:file_search_region_end]
        self.assertNotIn("const raw = String(rawPath ?? \"\");", source)
        self.assertNotIn("const out = [];\n        for (const v of val)", source)
        self.assertNotIn("const raw = err && err.message ? String(err.message)", source)
        self.assertNotIn('const raw = String(query || "").trim().toLowerCase();', file_search_region)
        self.assertNotIn("function filePickerFoldedSearchText(text)", file_search_region)
        self.assertNotIn("function filePickerCandidateScore(path, query)", file_search_region)
        self.assertNotIn('return ["changed", "mentioned", "recent"].includes(value) ? value : "";', source)
        self.assertNotIn('if (source === "changed") return "Changed files";', source)

    def test_file_helpers_preserve_literal_and_formatting_contracts(self) -> None:
        result = eval_file_helpers_real_order()
        self.assertEqual(result["listed"], ["/repo/trail.md ", "/repo/new\n.md"])
        self.assertEqual(result["listedNonArray"], [])
        self.assertEqual(result["hashSuffix"], "/repo/file.py")
        self.assertEqual(result["colonSuffix"], "/repo/file.py:12")
        self.assertEqual(result["noSuffixPreservesRaw"], "/repo/trail.md ")
        self.assertEqual(result["noSuffixPreservesNewline"], "/repo/new\n.md")
        self.assertTrue(result["textKind"])
        self.assertTrue(result["markdownKind"])
        self.assertFalse(result["imageKind"])
        self.assertTrue(result["diffMarkdown"])
        self.assertFalse(result["diffBinary"])
        self.assertEqual(result["tooLarge"], "big.txt is 1.50 KB. The viewer refuses to render text beyond 1.00 KB. Use Download instead.")
        self.assertEqual(result["tooLargeZeroLimit"], "big.txt is 1.50 KB. The viewer refuses to render text beyond the viewer limit. Use Download instead.")
        self.assertEqual(result["unsupported"], "image.bin is not renderable as text, markdown, image, or PDF. Use Download instead.")
        self.assertEqual(result["priorityPositive"], "+1.23")
        self.assertEqual(result["priorityNegative"], "-0.50")
        self.assertEqual(result["priorityInvalid"], "0.00")
        self.assertEqual(result["videoErrorMessage"], "bad codec")
        self.assertEqual(result["videoErrorString"], "transcode failed")
        self.assertEqual(result["videoErrorBlank"], "compatible video preview failed")
        self.assertEqual(result["videoErrorNull"], "compatible video preview failed")
        self.assertEqual(result["fileScoreNoQuery"], 0)
        self.assertEqual(result["fileScoreExact"], 12000)
        self.assertEqual(result["fileScoreBaseExact"], 10000)
        self.assertEqual(result["fileScoreBoundaryToken"], 298)
        self.assertEqual(result["fileScoreMultiToken"], 580)
        self.assertEqual(result["fileScoreSubsequence"], 124)
        self.assertEqual(result["fileScoreNoMatch"], -1)
        self.assertEqual(result["normalizedDotSlash"], "foo.py")
        self.assertEqual(result["normalizedBackslash"], "foo/bar.py")
        self.assertEqual(result["normalizedParent"], "")
        self.assertEqual(result["normalizedAbs"], "")
        self.assertEqual(result["normalizedTrailing"], "")
        self.assertEqual(result["normalizedNul"], "")
        self.assertEqual(result["rangeExact"], [[4, 7]])
        self.assertEqual(result["rangeFuzzy"], [[4, 5], [8, 9]])
        self.assertEqual(result["rangeNormalized"], [[0, 6]])
        self.assertEqual(result["rangeNone"], [])
        self.assertEqual(result["rangeMerged"], [[0, 3], [4, 7]])
        self.assertEqual(result["turkishSlices"], ["foo"])
        self.assertEqual(result["emojiSlices"], ["😀", "b"])
        self.assertEqual(result["candidateNormalizedScore"], 12000)
        self.assertGreater(result["compareScore"], 0)
        self.assertLess(result["comparePath"], 0)
        self.assertLess(result["compareGitPath"], 0)
        self.assertLess(result["compareChanged"], 0)
        self.assertEqual(result["sourceChangedTrimmed"], "changed")
        self.assertEqual(result["sourceMentioned"], "mentioned")
        self.assertEqual(result["sourceRecent"], "recent")
        self.assertEqual(result["sourceUnknown"], "")
        self.assertEqual(result["sourceBlank"], "")
        self.assertEqual(result["sectionChanged"], "Changed files")
        self.assertEqual(result["sectionMentioned"], "Mentioned in chat")
        self.assertEqual(result["sectionRecent"], "Recently opened")
        self.assertEqual(result["sectionUnknown"], "")
        self.assertTrue(result["frozen"])


if __name__ == "__main__":
    unittest.main()
