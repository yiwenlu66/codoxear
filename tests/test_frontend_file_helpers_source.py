import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"
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
          duplicatePaths: Array.from(helpers.duplicateFilePickerPaths([
            {{ path: "src/a.py", gitPath: true }},
            {{ path: "src/a.py", gitPath: false }},
            {{ path: "src/b.py", createNew: true }},
            {{ path: "src/b.py" }},
            {{ path: "" }},
            null,
          ])).sort(),
          duplicatePathsNonArray: Array.from(helpers.duplicateFilePickerPaths(null)),
          identityPending: helpers.filePickerIdentityHint({{ path: "src/a.py", pendingSessionPath: true }}, new Set(), {{ showSourceSections: true }}),
          identityGitDuplicateChanged: helpers.filePickerIdentityHint({{ path: "src/a.py", gitPath: true, changed: true }}, new Set(["src/a.py"]), {{ showSourceSections: true }}),
          identityGitNoSection: helpers.filePickerIdentityHint({{ path: "src/a.py", gitPath: true, changed: false }}, new Set(), {{ showSourceSections: false }}),
          identitySessionDuplicate: helpers.filePickerIdentityHint({{ path: "src/a.py", gitPath: false }}, new Set(["src/a.py"]), {{ showSourceSections: true }}),
          identityBlank: helpers.filePickerIdentityHint({{ path: "src/a.py", gitPath: false }}, new Set(), {{ showSourceSections: true }}),
          identityCreate: helpers.filePickerIdentityHint({{ path: "src/a.py", createNew: true }}, new Set(["src/a.py"]), {{ showSourceSections: false }}),
          // Raw-byte / literal collision hints: a tokenized entry (apiPath set)
          // and a literal entry (no apiPath) share a display path. Both are
          // duplicated; the tokenized side must read "non-UTF bytes" and the
          // literal side "literal name" so users/automation can tell them apart.
          rawByteCollisionPaths: Array.from(helpers.rawByteDuplicatePaths([
            {{ path: "dup-name.txt", apiPath: "tok-raw" }},
            {{ path: "dup-name.txt", apiPath: "" }},
            {{ path: "plain.txt", apiPath: "" }},
            {{ path: "plain.txt", apiPath: "" }},
          ])).sort(),
          rawByteCollisionNonArray: Array.from(helpers.rawByteDuplicatePaths(null)),
          identitySessionDuplicateTokenized: helpers.filePickerIdentityHint({{ path: "dup-name.txt", gitPath: false, apiPath: "tok-raw" }}, new Set(["dup-name.txt"]), {{ showSourceSections: true, tokenizedDuplicatePaths: new Set(["dup-name.txt"]) }}),
          identitySessionDuplicateLiteral: helpers.filePickerIdentityHint({{ path: "dup-name.txt", gitPath: false, apiPath: "" }}, new Set(["dup-name.txt"]), {{ showSourceSections: true, tokenizedDuplicatePaths: new Set(["dup-name.txt"]) }}),
          identityGitDuplicateTokenized: helpers.filePickerIdentityHint({{ path: "dup-name.txt", gitPath: true, changed: true, apiPath: "tok-raw" }}, new Set(["dup-name.txt"]), {{ showSourceSections: true, tokenizedDuplicatePaths: new Set(["dup-name.txt"]) }}),
          identityGitDuplicateLiteral: helpers.filePickerIdentityHint({{ path: "dup-name.txt", gitPath: true, changed: false, apiPath: "" }}, new Set(["dup-name.txt"]), {{ showSourceSections: false, tokenizedDuplicatePaths: new Set(["dup-name.txt"]) }}),
          // No tokenized sibling -> ordinary duplicate, no byte qualifier noise.
          identitySessionDuplicateNoByte: helpers.filePickerIdentityHint({{ path: "plain.txt", gitPath: false, apiPath: "" }}, new Set(["plain.txt"]), {{ showSourceSections: true, tokenizedDuplicatePaths: new Set() }}),
          identitySessionDuplicateMissingOption: helpers.filePickerIdentityHint({{ path: "plain.txt", gitPath: false, apiPath: "" }}, new Set(["plain.txt"]), {{ showSourceSections: true }}),
          titlePlain: helpers.filePickerTitle({{ path: "src/a.py" }}, ""),
          titleHint: helpers.filePickerTitle({{ path: "src/a.py" }}, "git root"),
          titleNull: helpers.filePickerTitle(null, "hint"),
          positionEmpty: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, ""),
          positionNull: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, null),
          positionSingleLine: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, "abc"),
          positionLf: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, "a\\nbc"),
          positionCrLf: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, "a\\r\\nbc"),
          positionCr: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, "a\\rb"),
          positionTrailingNewline: helpers.positionAfterInsertedText({{ lineNumber: 2, column: 5 }}, "abc\\n"),
          deleteBackspace: helpers.fileEditorDeleteCommandForKey("backspace"),
          deleteDelete: helpers.fileEditorDeleteCommandForKey("delete"),
          deleteBackspaceUpper: helpers.fileEditorDeleteCommandForKey("Backspace"),
          deleteUnknown: helpers.fileEditorDeleteCommandForKey("x"),
          deleteBlank: helpers.fileEditorDeleteCommandForKey(""),
          attachmentStemPath: helpers.attachmentSafeStem("/tmp/hello world.HEIC"),
          attachmentStemNoExt: helpers.attachmentSafeStem("no ext??"),
          attachmentStemFallback: helpers.attachmentSafeStem("$$$"),
          attachmentExtUpper: helpers.attachmentExtensionLower("Photo.HEIC"),
          attachmentExtNone: helpers.attachmentExtensionLower("README"),
          attachmentHeicType: helpers.attachmentIsLikelyHeic({{ type: "image/heif", name: "x.bin" }}),
          attachmentHeicExt: helpers.attachmentIsLikelyHeic({{ type: "", name: "x.HEIC" }}),
          attachmentHeicNo: helpers.attachmentIsLikelyHeic({{ type: "image/jpeg", name: "x.jpg" }}),
          attachmentImageType: helpers.attachmentLooksLikeImage({{ type: "image/svg+xml", name: "x.txt" }}),
          attachmentImageExt: helpers.attachmentLooksLikeImage({{ type: "", name: "x.webp" }}),
          attachmentImageNo: helpers.attachmentLooksLikeImage({{ type: "application/pdf", name: "x.pdf" }}),
          attachmentBytesB64: helpers.bytesToBase64(new Uint8Array([104, 101, 108, 108, 111]), (bin) => Buffer.from(bin, "binary").toString("base64")),
          clipboardItemFiles: helpers.extractFilesFromClipboardData({{ items: [{{ kind: "string", getAsFile: () => ({{ name: "bad.txt" }}) }}, {{ kind: "file", getAsFile: () => ({{ name: "clip.png" }}) }}], files: [{{ name: "fallback.bin" }}] }}).map((f) => f.name),
          clipboardFileFallback: helpers.extractFilesFromClipboardData({{ items: [{{ kind: "file", getAsFile: () => null }}], files: [{{ name: "clip-fallback.txt" }}] }}).map((f) => f.name),
          clipboardTextOnly: helpers.extractFilesFromClipboardData({{ items: [{{ kind: "string", getAsFile: () => null }}], files: [] }}).length,
          dropPrefersFiles: helpers.extractFilesFromDropData({{ files: [{{ name: "drop-a.txt" }}], items: [{{ kind: "file", getAsFile: () => ({{ name: "drop-b.txt" }}) }}] }}).map((f) => f.name),
          dropItemFallback: helpers.extractFilesFromDropData({{ files: [], items: [{{ kind: "file", getAsFile: () => ({{ name: "drop-item.txt" }}) }}] }}).map((f) => f.name),
          hasFileList: helpers.dataTransferHasFiles({{ files: [{{ name: "x" }}] }}),
          hasFileItem: helpers.dataTransferHasFiles({{ items: [{{ kind: "file" }}] }}),
          hasFileTypesArray: helpers.dataTransferHasFiles({{ types: ["text/plain", "Files"] }}),
          hasFileTypesContains: helpers.dataTransferHasFiles({{ types: {{ length: 0, contains: (value) => value === "Files" }} }}),
          hasNoFiles: helpers.dataTransferHasFiles({{ items: [{{ kind: "string" }}], types: ["text/plain"], files: [] }}),
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
            "duplicateFilePickerPaths",
            "rawByteDuplicatePaths",
            "filePickerIdentityHint",
            "filePickerTitle",
        ]:
            self.assertIn(f"typeof codoxearFileHelpers.{helper} !== \"function\"", source)
            self.assertIn(f"function {helper}", source)
        self.assertIn('typeof codoxearFileHelpers.positionAfterInsertedText !== "function"', source)
        self.assertIn('typeof codoxearFileHelpers.fileEditorDeleteCommandForKey !== "function"', source)
        self.assertIn('typeof codoxearFileHelpers.fileVideoPreviewErrorText !== "function"', source)
        self.assertNotIn("function positionAfterInsertedText", source)
        self.assertNotIn("function fileEditorDeleteCommandForKey", source)
        self.assertNotIn("function fileVideoPreviewErrorText", source)
        for helper in [
            "dataTransferHasFiles",
            "extractFilesFromClipboardData",
            "extractFilesFromDropData",
            "attachmentSafeStem",
            "attachmentExtensionLower",
            "attachmentIsLikelyHeic",
            "attachmentLooksLikeImage",
            "bytesToBase64",
        ]:
            self.assertIn(f"typeof codoxearFileHelpers.{helper} !== \"function\"", source)
        self.assertIn("window.CodoxearFileHelpers = Object.freeze({", helper_source)
        self.assertIn('throw new Error("Codoxear display helpers failed to load")', helper_source)
        self.assertIn('typeof codoxearDisplay.baseName !== "function"', helper_source)
        self.assertIn("errorText: (error) => codoxearFileHelpers.fileVideoPreviewErrorText(error)", source)
        self.assertIn("return codoxearFileHelpers.filePickerMatchRangesForQuery(text, query);", source)
        self.assertIn("return codoxearFileHelpers.filePickerCandidateScore(path, query);", source)
        self.assertIn("return codoxearFileHelpers.normalizeFileCandidateSource(source);", source)
        self.assertIn("return codoxearFileHelpers.filePickerSectionLabel(source);", source)
        self.assertIn("return codoxearFileHelpers.duplicateFilePickerPaths(entries);", source)
        self.assertIn("return codoxearFileHelpers.rawByteDuplicatePaths(entries);", source)
        self.assertIn("return codoxearFileHelpers.filePickerIdentityHint(entry, duplicatePaths, options);", source)
        self.assertIn("return codoxearFileHelpers.filePickerTitle(entry, hint);", source)
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        self.assertNotIn("return codoxearFileHelpers.positionAfterInsertedText(start, text);", source)
        self.assertNotIn("return codoxearFileHelpers.fileEditorDeleteCommandForKey(key);", source)
        self.assertIn("CodoxearFileHelpers.positionAfterInsertedText", viewer_source)
        self.assertIn("CodoxearFileHelpers.fileEditorDeleteCommandForKey", viewer_source)
        self.assertIn("return codoxearFileHelpers.attachmentSafeStem(name);", source)
        self.assertIn("return codoxearFileHelpers.attachmentIsLikelyHeic(file);", source)
        self.assertIn("return codoxearFileHelpers.attachmentLooksLikeImage(file);", source)
        self.assertIn("return codoxearFileHelpers.bytesToBase64(bytes, btoa);", source)
        self.assertIn('return ["changed", "mentioned", "recent"].includes(value) ? value : "";', helper_source)
        self.assertIn('if (source === "changed") return "Changed files";', helper_source)
        self.assertIn('function duplicateFilePickerPaths(entries) {', helper_source)
        self.assertIn('function rawByteDuplicatePaths(entries) {', helper_source)
        self.assertIn('function filePickerIdentityHint(entry, duplicatePaths, options) {', helper_source)
        self.assertIn('function filePickerTitle(entry, hint = "") {', helper_source)
        self.assertIn('const parts = value.replace(/\\r\\n?/g, "\\n").split("\\n");', helper_source)
        self.assertIn('if (key === "backspace") return "deleteLeft";', helper_source)
        self.assertIn('if (key === "delete") return "deleteRight";', helper_source)
        self.assertIn('function attachmentSafeStem(name) {', helper_source)
        self.assertIn('function attachmentExtensionLower(name) {', helper_source)
        self.assertIn('function attachmentIsLikelyHeic(file) {', helper_source)
        self.assertIn('function attachmentLooksLikeImage(file) {', helper_source)
        self.assertIn('function dataTransferHasFiles(data) {', helper_source)
        self.assertIn('function extractFilesFromClipboardData(data) {', helper_source)
        self.assertIn('function extractFilesFromDropData(data) {', helper_source)
        self.assertIn('function bytesToBase64(bytes, btoaFunc) {', helper_source)
        file_search_region_start = source.index("function collectMessageFileRefs()")
        file_search_region_end = source.index("function resetFileSearchState()", file_search_region_start)
        file_search_region = source[file_search_region_start:file_search_region_end]
        attachment_upload_start = source.index('async function stageFiles(files, { sid = selected, source = "picker" } = {}) {')
        attachment_upload_end = source.index("function clearComposer()", attachment_upload_start)
        attachment_upload_block = source[attachment_upload_start:attachment_upload_end]
        self.assertNotIn("const raw = String(rawPath ?? \"\");", source)
        self.assertNotIn("const out = [];\n        for (const v of val)", source)
        self.assertNotIn("const raw = err && err.message ? String(err.message)", source)
        self.assertNotIn('const raw = String(query || "").trim().toLowerCase();', file_search_region)
        self.assertNotIn("function filePickerFoldedSearchText(text)", file_search_region)
        self.assertNotIn("function filePickerCandidateScore(path, query)", file_search_region)
        self.assertNotIn('return ["changed", "mentioned", "recent"].includes(value) ? value : "";', source)
        self.assertNotIn('if (source === "changed") return "Changed files";', source)
        self.assertNotIn('function duplicateFilePickerPaths(entries) {\n          const counts = new Map();', source)
        self.assertNotIn('function filePickerIdentityHint(entry, duplicatePaths, options) {\n          const showSourceSections = Boolean(options && options.showSourceSections);', source)
        self.assertNotIn('function filePickerTitle(entry, hint = "") {\n          const path = String(entry && entry.path || "");', source)
        self.assertNotIn('const parts = value.replace(/\\r\\n?/g, "\\n").split("\\n");', source)
        self.assertNotIn('if (key === "backspace") return "deleteLeft";', source)
        self.assertNotIn('if (key === "delete") return "deleteRight";', source)
        self.assertNotIn("function safeStem(name)", attachment_upload_block)
        self.assertNotIn("function extLower(name)", attachment_upload_block)
        self.assertNotIn("function isLikelyHeic(file)", attachment_upload_block)
        self.assertNotIn("function looksLikeImage(file)", attachment_upload_block)
        self.assertNotIn("String.fromCharCode.apply(null, bytes.subarray", attachment_upload_block)
        self.assertIn("const stem = safeAttachmentStem(uploadName);", attachment_upload_block)
        self.assertIn("const b64 = b64FromBytes(new Uint8Array(ab));", attachment_upload_block)

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
        self.assertEqual(result["duplicatePaths"], ["src/a.py"])
        self.assertEqual(result["duplicatePathsNonArray"], [])
        self.assertEqual(result["identityPending"], "current folder")
        self.assertEqual(result["identityGitDuplicateChanged"], "git root · changed")
        self.assertEqual(result["identityGitNoSection"], "git root")
        self.assertEqual(result["identitySessionDuplicate"], "current folder")
        self.assertEqual(result["identityBlank"], "")
        self.assertEqual(result["identityCreate"], "")
        self.assertEqual(result["rawByteCollisionPaths"], ["dup-name.txt"])
        self.assertEqual(result["rawByteCollisionNonArray"], [])
        self.assertEqual(result["identitySessionDuplicateTokenized"], "current folder · non-UTF bytes")
        self.assertEqual(result["identitySessionDuplicateLiteral"], "current folder · literal name")
        self.assertEqual(result["identityGitDuplicateTokenized"], "git root · changed · non-UTF bytes")
        self.assertEqual(result["identityGitDuplicateLiteral"], "git root · literal name")
        self.assertEqual(result["identitySessionDuplicateNoByte"], "current folder")
        self.assertEqual(result["identitySessionDuplicateMissingOption"], "current folder")
        self.assertEqual(result["titlePlain"], "src/a.py")
        self.assertEqual(result["titleHint"], "src/a.py — git root")
        self.assertEqual(result["titleNull"], " — hint")
        self.assertEqual(result["positionEmpty"], {"lineNumber": 2, "column": 5})
        self.assertEqual(result["positionNull"], {"lineNumber": 2, "column": 5})
        self.assertEqual(result["positionSingleLine"], {"lineNumber": 2, "column": 8})
        self.assertEqual(result["positionLf"], {"lineNumber": 3, "column": 3})
        self.assertEqual(result["positionCrLf"], {"lineNumber": 3, "column": 3})
        self.assertEqual(result["positionCr"], {"lineNumber": 3, "column": 2})
        self.assertEqual(result["positionTrailingNewline"], {"lineNumber": 3, "column": 1})
        self.assertEqual(result["deleteBackspace"], "deleteLeft")
        self.assertEqual(result["deleteDelete"], "deleteRight")
        self.assertEqual(result["deleteBackspaceUpper"], "")
        self.assertEqual(result["deleteUnknown"], "")
        self.assertEqual(result["deleteBlank"], "")
        self.assertEqual(result["attachmentStemPath"], "hello_world")
        self.assertEqual(result["attachmentStemNoExt"], "no_ext_")
        self.assertEqual(result["attachmentStemFallback"], "_")
        self.assertEqual(result["attachmentExtUpper"], "heic")
        self.assertEqual(result["attachmentExtNone"], "")
        self.assertTrue(result["attachmentHeicType"])
        self.assertTrue(result["attachmentHeicExt"])
        self.assertFalse(result["attachmentHeicNo"])
        self.assertTrue(result["attachmentImageType"])
        self.assertTrue(result["attachmentImageExt"])
        self.assertFalse(result["attachmentImageNo"])
        self.assertEqual(result["attachmentBytesB64"], "aGVsbG8=")
        self.assertEqual(result["clipboardItemFiles"], ["clip.png"])
        self.assertEqual(result["clipboardFileFallback"], ["clip-fallback.txt"])
        self.assertEqual(result["clipboardTextOnly"], 0)
        self.assertEqual(result["dropPrefersFiles"], ["drop-a.txt"])
        self.assertEqual(result["dropItemFallback"], ["drop-item.txt"])
        self.assertTrue(result["hasFileList"])
        self.assertTrue(result["hasFileItem"])
        self.assertTrue(result["hasFileTypesArray"])
        self.assertTrue(result["hasFileTypesContains"])
        self.assertFalse(result["hasNoFiles"])
        self.assertTrue(result["frozen"])


if __name__ == "__main__":
    unittest.main()
