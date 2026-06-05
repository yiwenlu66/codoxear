import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def render_markdown(markdown: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function escapeHtml(s) {")
    end = source.index("function isMarkdownPreviewable(path) {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          console,
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


class TestMarkdownRendererSource(unittest.TestCase):
    def test_fenced_code_block_nested_under_list_item_renders_as_code(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "**Implementation Plan**",
                    "",
                    "1. Update SFT training submission to pipeline steps:",
                    "   - Build the next N SFT batches ahead of time, e.g. window size 10.",
                    "   - Submit requests in exact semantic order:",
                    "     ```text",
                    "     fwdbwd(1), optim(1), fwdbwd(2), optim(2), ...",
                    "     ```",
                    "   - Do not submit all `fwdbwd` first.",
                ]
            )
        )

        self.assertIn('<pre><code data-lang="text">fwdbwd(1), optim(1), fwdbwd(2), optim(2), ...</code></pre>', html)
        self.assertIn("Submit requests in exact semantic order:<pre>", html)
        self.assertIn("Do not submit all <code>fwdbwd</code> first.", html)
        self.assertNotIn("```", html)

    def test_nested_fenced_code_block_preserves_blank_lines(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "- intro:",
                    "  ```text",
                    "  one",
                    "",
                    "  two",
                    "  ```",
                    "- after",
                ]
            )
        )

        self.assertIn('<pre><code data-lang="text">one\n\ntwo</code></pre>', html)
        self.assertIn("<li>after</li>", html)
        self.assertNotIn("```", html)

    def test_nested_fenced_code_block_after_blank_line_in_list_item(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "- OpenAI native web search:",
                    "  - Command:",
                    "",
                    "    ```bash",
                    "    pi --no-session --print 'Use native provider web_search ...'",
                    "    ```",
                    "",
                    "  - Output:",
                    "",
                    "    ```text",
                    "    https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses",
                    "    ```",
                ]
            )
        )

        self.assertIn(
            '<pre><code data-lang="bash">pi --no-session --print &#39;Use native provider web_search ...&#39;</code></pre>',
            html,
        )
        self.assertIn(
            '<pre><code data-lang="text">https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses</code></pre>',
            html,
        )
        self.assertIn("<li>OpenAI native web search:<ul><li>Command:<pre>", html)
        self.assertIn("</pre></li><li>Output:<pre>", html)
        self.assertNotIn("```", html)

    def test_ordered_list_markers_are_literal_when_blank_separated(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "1. one",
                    "",
                    "3. three",
                    "",
                    "```text",
                    "ignored",
                    "```",
                    "",
                    "5. five",
                ]
            )
        )

        markers = re.findall(r'<span class="md-list-marker">([^<]+)</span>', html)
        self.assertEqual(markers, ["1.", "3.", "5."])

    def test_ordered_list_markers_are_literal_when_contiguous(self) -> None:
        html = render_markdown("1. one\n3. three\n5. five")

        markers = re.findall(r'<span class="md-list-marker">([^<]+)</span>', html)
        self.assertEqual(markers, ["1.", "3.", "5."])


if __name__ == "__main__":
    unittest.main()
