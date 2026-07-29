import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"


def render_markdown(markdown: str, katex_expr: str | None = None) -> str:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    katex_stmt = f"katex = {katex_expr};" if katex_expr else ""
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          console,
          window: {{
            CodoxearUrls: {{
              resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^\\//, ""), "http://localhost/").toString(),
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        {f"vm.runInContext({json.dumps(katex_stmt)}, ctx);" if katex_stmt else ""}
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


class TestMarkdownRendererSource(unittest.TestCase):
    def test_markdown_renderer_renders_tables_inline_formatting_citations_and_file_paths(self) -> None:
        html = render_markdown(
            "| Name | Value |\n| :--- | ---: |\n| **bold** | `code` |\n\n"
            "<oai-mem-citation><citation_entries>notes/plan.md:7-9|note=[Plan]</citation_entries>"
            "<rollout_ids>r1</rollout_ids></oai-mem-citation>\n\n"
            "See src/app.py:12 and [settings](./config/settings.json#L3)."
        )
        self.assertIn('<div class="md-table-wrap"><table>', html)
        self.assertIn('<th style="text-align:left">Name</th>', html)
        self.assertIn('<td style="text-align:right"><code>code</code></td>', html)
        self.assertIn('<strong>bold</strong>', html)
        self.assertIn('Memory citations:', html)
        self.assertIn('data-candidate-file-path="~/.codex/memories/notes/plan.md" data-candidate-file-line="7"', html)
        self.assertIn('data-candidate-file-path="src/app.py" data-candidate-file-line="12"', html)
        self.assertIn('data-candidate-file-path="./config/settings.json" data-candidate-file-line="3"', html)

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

        self.assertIn('<pre><button class="code-copy-btn" type="button" aria-label="Copy code" title="Copy code"></button><code data-lang="text">fwdbwd(1), optim(1), fwdbwd(2), optim(2), ...</code></pre>', html)
        self.assertIn("Submit requests in exact semantic order:<pre><button", html)
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

        self.assertIn('<pre><button class="code-copy-btn" type="button" aria-label="Copy code" title="Copy code"></button><code data-lang="text">one\n\ntwo</code></pre>', html)
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
            '<pre><button class="code-copy-btn" type="button" aria-label="Copy code" title="Copy code"></button><code data-lang="bash">pi --no-session --print &#39;Use native provider web_search ...&#39;</code></pre>',
            html,
        )
        self.assertIn(
            '<pre><button class="code-copy-btn" type="button" aria-label="Copy code" title="Copy code"></button><code data-lang="text">https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses</code></pre>',
            html,
        )
        self.assertIn("<li>OpenAI native web search:<ul><li>Command:<pre><button", html)
        self.assertIn("</pre></li><li>Output:<pre><button", html)
        self.assertNotIn("```", html)

    def test_blockquote_renders_as_blockquote(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "So the best current mechanism is:",
                    "",
                    "> OCC does not recognize Pi/my locally-cloaked request as native Claude Code traffic and reads `12368`.",
                    "",
                    "This also explains why body-level fixes did not work.",
                ]
            )
        )

        self.assertIn("<blockquote><p>OCC does not recognize", html)
        self.assertIn("<code>12368</code>", html)
        self.assertIn("</p></blockquote>", html)
        self.assertNotIn("&gt; OCC", html)

    def test_blockquote_allows_lazy_continuation(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "> first quoted line",
                    "continued quoted line with **bold** text",
                ]
            )
        )

        self.assertEqual(
            html,
            "<blockquote><p>first quoted line<br />continued quoted line with <strong>bold</strong> text</p></blockquote>",
        )

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

    def test_inline_paren_math_renders_as_inline_math(self) -> None:
        html = render_markdown("For a residual stream vector \\(h_{\\ell,t}\\) at layer \\(\\ell\\), the J-lens asks.")
        # No KaTeX in the Node VM: the fallback retains escaped source delimiters.
        self.assertIn('<span class="md-math-fallback md-math-inline">\\(h_{\\ell,t}\\)</span>', html)
        self.assertIn('<span class="md-math-fallback md-math-inline">\\(\\ell\\)</span>', html)
        self.assertNotIn("@@MATH", html)
        self.assertIn("the J-lens asks.", html)

    def test_display_bracket_math_renders_as_block(self) -> None:
        markdown = "It computes an average Jacobian:\n\n\\[\nJ_\\ell\n=\n\\mathbb{E}\\left[\\frac{\\partial h}{\\partial h_\\ell}\\right]\n\\]\n\nThen it decodes."
        html = render_markdown(markdown)
        self.assertIn('<span class="md-math-fallback md-math-display">\\[', html)
        self.assertIn("\\mathbb{E}", html)
        self.assertIn("Then it decodes.", html)
        self.assertNotIn("@@MATH", html)
        self.assertNotIn("<p>@@MATH", html)

    def test_dollar_dollar_display_math(self) -> None:
        html = render_markdown("The value of $$x^2 + y^2$$ is large.")
        self.assertIn('<span class="md-math-fallback md-math-display">\\[x^2 + y^2\\]</span>', html)
        self.assertNotIn("@@MATH", html)
        self.assertNotIn("$$", html)

    def test_single_dollar_inline_is_not_treated_as_math(self) -> None:
        html = render_markdown("The single-`$` rule is guarded, so `$VAR` is safe.")
        self.assertNotIn("md-math-fallback", html)
        self.assertIn("$", html)
        self.assertIn("rule is guarded", html)
        self.assertIn("is safe", html)

    def test_currency_dollars_not_treated_as_math(self) -> None:
        html = render_markdown("That costs $5 and $10 more.")
        self.assertNotIn("md-math-fallback", html)
        self.assertIn("$5", html)
        self.assertIn("$10", html)

    def test_math_inside_fenced_code_block_is_not_rendered(self) -> None:
        html = render_markdown(
            "\n".join(
                [
                    "\\text{outside}",
                    "",
                    "```text",
                    "\\(not math\\)",
                    "```",
                    "",
                    "After.",
                ]
            )
        )
        self.assertIn("outside", html)
        self.assertIn('<code data-lang="text">\\(not math\\)</code>', html)
        self.assertNotIn("md-math-fallback", html)
        self.assertNotIn("@@MATH", html)

    def test_katex_render_path_is_invoked_with_display_mode(self) -> None:
        katex_expr = "{ renderToString: function(src, opts){ return '<kmx>' + src + ':' + (opts.displayMode ? 'D' : 'I') + '</kmx>'; } }"
        html = render_markdown("Inline \\(x\\) and block:\n\n\\[y\\]", katex_expr=katex_expr)
        self.assertIn("<kmx>x:I</kmx>", html)
        self.assertIn("<kmx>y:D</kmx>", html)
        self.assertNotIn("md-math-fallback", html)
        self.assertNotIn("@@MATH", html)


if __name__ == "__main__":
    unittest.main()
