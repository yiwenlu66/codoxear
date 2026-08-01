import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def run_renderer_with_marked_stub(markdown: str) -> tuple[str, str]:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        let parsed = "";
        const ctx = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          console,
          window: {{
            marked: {{ parse(value) {{ parsed = value; return `<p>${{value}}</p>`; }} }},
            CodoxearUrls: {{ resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^\\//, ""), "http://localhost/").toString() }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const html = ctx.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)});
        process.stdout.write(JSON.stringify([parsed, html]));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return tuple(json.loads(proc.stdout))


class TestMarkdownRendererSource(unittest.TestCase):
    def test_marked_is_loaded_before_codoxear_renderer(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")
        marked_tag = '<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js" defer></script>'
        self.assertIn(marked_tag, html)
        self.assertLess(html.index(marked_tag), html.index('src="app_markdown.js'))

    def test_renderer_delegates_standard_markdown_to_marked(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn("window.marked.parse(prepared)", source)
        self.assertNotIn("function renderInlineMd", source)
        self.assertNotIn("function parseList", source)
        self.assertNotIn("function parseTable", source)
        parsed, html = run_renderer_with_marked_stub("*italic* and **bold**")
        self.assertEqual(parsed, "*italic* and **bold**")
        self.assertEqual(html, "<p>*italic* and **bold**</p>")

    def test_math_is_extracted_before_marked_including_single_dollars(self) -> None:
        parsed, html = run_renderer_with_marked_stub("Inline $x^2$; display $$y$$; and \\(z\\).")
        self.assertNotIn("$x^2$", parsed)
        self.assertNotIn("$$y$$", parsed)
        self.assertNotIn("\\(z\\)", parsed)
        self.assertIn("@@MATH0@@", parsed)
        self.assertIn("md-math-fallback md-math-inline", html)
        self.assertIn("md-math-fallback md-math-display", html)

    def test_math_does_not_rewrite_code_spans_or_fenced_code(self) -> None:
        parsed, _html = run_renderer_with_marked_stub("`$code$`\n\n```text\n$also_code$\n```")
        self.assertIn("`$code$`", parsed)
        self.assertIn("$also_code$", parsed)

    def test_citation_rewrite_happens_before_marked(self) -> None:
        parsed, _html = run_renderer_with_marked_stub(
            "<oai-mem-citation><citation_entries>notes/plan.md:7-9|note=[Plan]</citation_entries><rollout_ids>r1</rollout_ids></oai-mem-citation>"
        )
        self.assertIn("Memory citations:", parsed)
        self.assertIn("[Plan](~/.codex/memories/notes/plan.md#L7-9)", parsed)


if __name__ == "__main__":
    unittest.main()
