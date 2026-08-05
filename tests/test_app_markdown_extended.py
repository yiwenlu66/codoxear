"""Behavioral Markdown regression coverage using the vendored Marked runtime.

The VM intentionally omits ``document`` so this focuses on the renderer's HTML
contract before browser-only post-processing (image hydration and table wrapping).
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"
MARKED_JS = ROOT / "codoxear" / "static" / "vendor" / "marked.min.js"


def render_markdown(markdown: str) -> str:
    app_source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    marked_source = MARKED_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const context = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{ CodoxearUrls: {{ resolveAppUrl: (path) => path }} }},
        }};
        vm.createContext(context);
        vm.runInContext({json.dumps(marked_source)}, context);
        context.window.marked = context.marked;
        vm.runInContext({json.dumps(app_source)}, context);
        process.stdout.write(context.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)}));
        """
    )
    return subprocess.run(
        ["node", "-e", program],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def test_markdown_edge_cases_render_as_structured_html_and_preserve_fenced_source() -> None:
    long_code_line = "x" * 4096
    markdown = "\n".join(
        [
            "````markdown",
            "outer starts",
            "```shell",
            "$ must_stay_literal",
            "$must_stay_literal$",
            "# nested comment",
            "```",
            "outer ends",
            "````",
            "",
            "```sh",
            '$ printf "hello"',
            "# shell comment",
            long_code_line,
            "```",
            "",
            "![wide chart](https://example.test/chart.png)",
            "",
            "| C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |",
            "| -- | -- | -- | -- | -- | -- | -- | -- |",
            "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |",
            "",
            "first line",
            "second line",
            "",
            "- first item",
            "",
            "  continuation paragraph",
            "",
            "- second item",
        ]
    )

    html = render_markdown(markdown)

    assert '<pre><code class="language-markdown">' in html
    assert "```shell\n$ must_stay_literal\n$must_stay_literal$\n# nested comment\n```" in html
    assert "$must_stay_literal$" in html
    assert "md-math-fallback" not in html
    assert '<pre><code class="language-sh">$ printf &quot;hello&quot;\n# shell comment' in html
    assert long_code_line in html
    assert '<img src="https://example.test/chart.png" alt="wide chart">' in html
    assert html.count("<th>") == 8
    assert html.count("<td>") == 8
    assert "<p>first line<br>second line</p>" in html


def test_nested_tilde_fences_in_lists_and_aligned_tables_preserve_their_distinct_syntax() -> None:
    markdown = "\n".join(
        [
            "- example with a nested fence:",
            "",
            "  ~~~~md",
            "  ```javascript",
            '  const price = "$5";',
            '  const formula = "$x^2$";',
            "  ```",
            "  ~~~~",
            "",
            "| Alignment | Literal pipe | Formula |",
            "| :-------- | :----------: | -------: |",
            r"| left | `a\|b` | $x_1$ |",
        ]
    )

    html = render_markdown(markdown)

    assert '<pre><code class="language-md">```javascript' in html
    assert 'const formula = &quot;$x^2$&quot;;' in html
    assert "md-math-fallback" not in html.split("</code></pre>", 1)[0]
    assert '<th align="left">Alignment</th>' in html
    assert '<th align="center">Literal pipe</th>' in html
    assert '<th align="right">Formula</th>' in html
    assert '<td align="center"><code>a|b</code></td>' in html
    assert '<td align="right"><span class="md-math-fallback md-math-inline">\\(x_1\\)</span></td>' in html
