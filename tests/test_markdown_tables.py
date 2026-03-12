import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def render_markdown(markdown: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function escapeHtml")
    end = source.index("const mdCache = new Map();")
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          console,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^\\//, ""), "http://localhost/").toString(),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_mdToHtml = mdToHtml;\n")}, ctx);
        process.stdout.write(ctx.__test_mdToHtml({json.dumps(markdown)}));
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
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function escapeHtml")
    end = source.index("function iconSvg")
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          console,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^\\//, ""), "http://localhost/").toString(),
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_markdownPreviewHtml = markdownPreviewHtml;\n")}, ctx);
        process.stdout.write(ctx.__test_markdownPreviewHtml({json.dumps(markdown)}, {{ filePath: {json.dumps(file_path)}, sessionId: {json.dumps(session_id)} }}));
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

    def test_markdown_preview_resolves_relative_image_against_file_path(self) -> None:
        html = render_markdown_preview("![diagram](../images/flow.png)", "docs/guides/intro.md", "sess-123")
        self.assertIn('src="http://localhost/api/sessions/sess-123/file/blob?path=docs%2Fimages%2Fflow.png"', html)
        self.assertIn('alt="diagram"', html)

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
        self.assertIn("<ol>", html)
        self.assertIn('data-candidate-file-path="~/.codex/memories/MEMORY.md"', html)
        self.assertIn('data-candidate-file-line="4773"', html)
        self.assertIn(">used corrected memex memory-estimation guidance and the bs64 OOM interpretation</span>", html)
        self.assertIn('data-candidate-file-path="~/.codex/memories/rollout_summaries/2026-02-17T21-23-02-example.md"', html)
        self.assertIn('data-candidate-file-line="10"', html)
        self.assertIn(">weekly report format</span>", html)
        self.assertNotIn("oai-mem-citation", html)

    def test_candidate_file_link_source_prefers_resolved_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('const resolvedPath = String(result.resolvedPath || result.inspectPath || path).trim();', source)
        self.assertIn('"data-file-path": resolvedPath,', source)


if __name__ == "__main__":
    unittest.main()
