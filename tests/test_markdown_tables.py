import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


def css_block(source: str, selector: str) -> str:
    start = source.index(selector)
    end = source.index("}\n", start) + 1
    return source[start:end]


def markdown_vm_prelude(source: str) -> str:
    return textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          URL,
          console,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{
            CodoxearUrls: {{
              resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^\\//, ""), "http://localhost/").toString(),
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        """
    )


def render_markdown(markdown: str) -> str:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    js = markdown_vm_prelude(source) + textwrap.dedent(
        f"""
        process.stdout.write(ctx.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout


def render_markdown_preview(markdown: str, file_path: str, session_id: str) -> str:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    js = markdown_vm_prelude(source) + textwrap.dedent(
        f"""
        process.stdout.write(ctx.window.CodoxearMarkdown.markdownPreviewHtml({json.dumps(markdown)}, {{ filePath: {json.dumps(file_path)}, sessionId: {json.dumps(session_id)} }}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout


def render_chat_markdown_sequence(markdown: str, session_ids: list[str | None]) -> list[str]:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    js = markdown_vm_prelude(source) + textwrap.dedent(
        f"""
        const outputs = {json.dumps(session_ids)}.map((sid) => sid === null ? ctx.window.CodoxearMarkdown.chatMarkdownHtmlCached({json.dumps(markdown)}) : ctx.window.CodoxearMarkdown.chatMarkdownHtmlCached({json.dumps(markdown)}, sid));
        process.stdout.write(JSON.stringify(outputs));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestMarkdownTables(unittest.TestCase):
    def test_local_markdown_file_link_shows_line_on_link_face(self) -> None:
        html = render_markdown("[server.py](/home/yiwen/codex-web/codoxear/server.py#L123)")
        self.assertIn('data-candidate-file-path="/home/yiwen/codex-web/codoxear/server.py"', html)
        self.assertIn('data-candidate-file-line="123"', html)
        self.assertIn(">server.py#L123</span>", html)

    def test_descriptive_markdown_link_text_stays_unchanged(self) -> None:
        html = render_markdown("[server implementation](/home/yiwen/codex-web/codoxear/server.py#L123)")
        self.assertIn(">server implementation</span>", html)

    def test_pipe_table_renders_as_html_table(self) -> None:
        html = render_markdown(
            textwrap.dedent(
                """\
                | ID | Urgency | Started | Project | Task |
                |---|---:|---|---|---|
                | 14 | 16.96 | yes | `offline` | throw away old clothes |
                | 22 | 16.96 | yes | `offline` | set up object inventory |
                """
            )
        )
        self.assertIn('<div class="md-table-wrap"><table>', html)
        self.assertIn("<thead><tr>", html)
        self.assertIn('<th style="text-align:right">Urgency</th>', html)
        self.assertIn("<code>offline</code>", html)
        self.assertIn('<td style="text-align:right">16.96</td>', html)

    def test_non_table_pipe_text_stays_paragraph(self) -> None:
        html = render_markdown("Top 10 | pending | tasks")
        self.assertEqual(html, "<p>Top 10 | pending | tasks</p>")

    def test_markdown_css_keeps_code_light_and_tables_contained(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        pre = css_block(css, ".md pre {")
        table_wrap = css_block(css, ".md-table-wrap {")
        table = css_block(css, ".md table {")
        cells = css_block(css, ".md th,\n      .md td {")

        self.assertIn("background: rgba(248, 250, 252, 0.96);", pre)
        self.assertIn("color: var(--text);", pre)
        self.assertIn("border: 1px solid rgba(15, 23, 42, 0.1);", pre)
        self.assertNotIn("background: #0b1220;", pre)
        self.assertIn("max-width: 100%;", table_wrap)
        self.assertIn("overflow-x: auto;", table_wrap)
        self.assertIn("width: 100%;", table)
        self.assertIn("max-width: 100%;", table)
        self.assertIn("table-layout: auto;", table)
        self.assertNotIn("table-layout: fixed;", table)
        self.assertNotIn("width: max-content;", table)
        self.assertIn("overflow-wrap: anywhere;", cells)
        self.assertIn("word-break: break-word;", cells)

    def test_markdown_preview_resolves_relative_image_against_file_path(self) -> None:
        html = render_markdown_preview("![diagram](../images/flow.png)", "docs/guides/intro.md", "sess-123")
        self.assertIn('src="http://localhost/api/sessions/sess-123/file/blob?path=docs%2Fimages%2Fflow.png"', html)
        self.assertIn('alt="diagram"', html)

    def test_chat_markdown_resolves_absolute_local_image_through_session_blob(self) -> None:
        html = render_chat_markdown_sequence("![loss](/home/yiwen/glm5/loss.png)", ["sess-123"])[0]

        self.assertIn('src="http://localhost/api/sessions/sess-123/file/blob?path=%2Fhome%2Fyiwen%2Fglm5%2Floss.png"', html)
        self.assertIn('alt="loss"', html)

    def test_chat_markdown_resolves_relative_local_image_through_session_blob(self) -> None:
        html = render_chat_markdown_sequence("![plot](plots/loss.png)", ["sess-123"])[0]

        self.assertIn('src="http://localhost/api/sessions/sess-123/file/blob?path=plots%2Floss.png"', html)

    def test_chat_markdown_cache_is_scoped_by_session(self) -> None:
        first, second = render_chat_markdown_sequence("![plot](plots/loss.png)", ["sess-a", "sess-b"])

        self.assertIn("/api/sessions/sess-a/file/blob", first)
        self.assertIn("/api/sessions/sess-b/file/blob", second)
        self.assertNotIn("/api/sessions/sess-a/file/blob", second)

    def test_chat_markdown_without_session_does_not_throw(self) -> None:
        html = render_chat_markdown_sequence("**bold**", [None])[0]

        self.assertEqual(html, "<p><strong>bold</strong></p>")

    def test_oai_mem_citation_block_renders_memory_links(self) -> None:
        html = render_markdown(
            textwrap.dedent(
                """\
                <oai-mem-citation>
                <citation_entries>
                MEMORY.md:4773-4779|note=[used corrected memex memory-estimation guidance and the bs64 OOM interpretation]
                rollout_summaries/2026-02-17T21-23-02-example.md:10-12|note=[weekly report format]
                </citation_entries>
                <rollout_ids>
                </rollout_ids>
                </oai-mem-citation>
                """
            )
        )
        self.assertIn("<hr />", html)
        self.assertIn("<p>Memory citations:</p>", html)
        self.assertIn('<ol class="md-literal-ol">', html)
        self.assertIn('<span class="md-list-marker">1.</span>', html)
        self.assertIn('<span class="md-list-marker">2.</span>', html)
        self.assertIn('data-candidate-file-path="~/.codex/memories/MEMORY.md"', html)
        self.assertIn('data-candidate-file-line="4773"', html)
        self.assertIn(">used corrected memex memory-estimation guidance and the bs64 OOM interpretation</span>", html)
        self.assertIn('data-candidate-file-path="~/.codex/memories/rollout_summaries/2026-02-17T21-23-02-example.md"', html)
        self.assertIn('data-candidate-file-line="10"', html)
        self.assertIn(">weekly report format</span>", html)
        self.assertNotIn("oai-mem-citation", html)

    def test_candidate_file_link_source_prefers_resolved_path(self) -> None:
        viewer_source = (APP_JS.parent / "app_file_viewer.js").read_text(encoding="utf-8")
        self.assertIn('const resolvedPath = String(result.resolvedPath || result.inspectPath || path);', viewer_source)
        self.assertIn('"data-file-path": resolvedPath,', viewer_source)

    def test_ambiguous_candidate_file_refs_open_identity_picker(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        picker_source = (APP_JS.parent / "app_file_picker.js").read_text(encoding="utf-8")
        viewer_source = (APP_JS.parent / "app_file_viewer.js").read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn("function replaceAmbiguousNode(node, path, line = null)", viewer_source)
        self.assertIn('class: "inlineFileLink inlineFileAmbiguousRef"', viewer_source)
        self.assertIn('"data-file-picker-query": query,', viewer_source)
        self.assertIn('class: "inlineFileChoiceHint", text: "choose"', viewer_source)
        self.assertIn("if (result && result.ambiguous) {", viewer_source)
        self.assertIn("replaceAmbiguousNode(node, result.path || path, line);", viewer_source)
        self.assertIn("async function openAmbiguousChoice(query, line = null)", viewer_source)
        self.assertIn("await requireFunction(options.showFileViewer, \"showFileViewer\")({ pickerQuery: rawQuery, line });", viewer_source)
        self.assertIn('const choice = source.closest("a[data-file-picker-query]");', viewer_source)
        self.assertIn('fileDiff.addEventListener("click", (e) => {', source)
        self.assertIn("function filePickerSelectionLine()", source)
        self.assertIn("filePickerMenuState = codoxearFilePicker.createMenuState", source)
        self.assertNotIn("let filePickerPreserveSearchOnFocus = false;", source)
        self.assertIn("if (takePreservedSearchOnFocus()) {", picker_source)
        self.assertIn("return filePickerInputRuntime.openSearchQuery(query, { line, suppressDraft });", source)
        self.assertIn("ui.setPreserveSearchOnFocus(true);", (APP_JS.parent / "app_file_viewer.js").read_text(encoding="utf-8"))
        self.assertNotIn("function filePickerDraftSuppressed()", source)
        self.assertNotIn("function filePickerAmbiguousChoiceActive()", source)
        self.assertIn("draftSuppressed: () => Boolean(menuState && typeof menuState.draftSuppressed === \"function\" && menuState.draftSuppressed(inputValue()))", picker_source)
        self.assertIn("if (ambiguousChoiceActive(input.value)) {", picker_source)
        self.assertNotIn("const showDraft = draftPath && !filePickerDraftSuppressed();", source)
        self.assertIn("const showDraft = draftPath && !draftSuppressed();", picker_source)
        self.assertIn("await openFilePathWithResolvedMode(selectedEntry.path, { line: filePickerSelectionLine(),", source)
        self.assertIn("void openFilePathWithResolvedMode(active.path, { line: selectionLine(input.value),", picker_source)
        self.assertIn(".inlineFileAmbiguousRef", css)
        self.assertIn(".inlineFileChoiceHint", css)


if __name__ == "__main__":
    unittest.main()
