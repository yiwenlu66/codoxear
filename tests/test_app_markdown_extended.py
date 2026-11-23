"""Behavioral Markdown regression coverage using the vendored Marked runtime.

The VM intentionally omits ``document`` so this focuses on the renderer's HTML
contract before browser-only post-processing (image hydration and table wrapping).
"""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = module_path("app_markdown.js")
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


def run_markdown_expr(expr: str) -> str:
    """Evaluate an expression against the markdown module namespace in the VM."""
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
        process.stdout.write(JSON.stringify({expr}));
        """
    )
    return subprocess.run(
        ["node", "-e", program],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def test_attachment_prefix_projection_rules() -> None:
    ns = "context.window.CodoxearMarkdown.attachmentDisplayMarkdown"
    out = json.loads(
        run_markdown_expr(
            f"""
            [{ns}("Attachment 1: /up/shot.png\\nAttachment 2: /up/notes.txt\\nplease review"),
             {ns}("Attachment 1: /up/shot.PNG\\nbody"),
             {ns}("see this\\nAttachment 1: /up/shot.png"),
             {ns}("Attachment 1: /up/my file.png\\nbody"),
             {ns}("Attachment 1: /up/a)b.png\\nbody"),
             {ns}("Attachment one: /up/shot.png\\nbody"),
             {ns}("plain message")]
            """
        )
    )
    # Image lines become markdown image syntax keyed on the basename.
    assert out[0] == "![shot.png](/up/shot.png)\nAttachment 2: /up/notes.txt\nplease review"
    # Extension matching is case-insensitive.
    assert out[1] == "![shot.PNG](/up/shot.PNG)\nbody"
    # Only the leading generated block is projected; later lookalikes are text.
    assert out[2] == "see this\nAttachment 1: /up/shot.png"
    # Whitespace/parentheses in the path would break markdown destinations:
    # those lines stay plain file-ref text.
    assert out[3] == "Attachment 1: /up/my file.png\nbody"
    assert out[4] == "Attachment 1: /up/a)b.png\nbody"
    # Non-numeric labels and non-attachment text pass through untouched.
    assert out[5] == "Attachment one: /up/shot.png\nbody"
    assert out[6] == "plain message"


def test_chat_markdown_renders_image_attachments_inline() -> None:
    html = json.loads(
        run_markdown_expr(
            'context.window.CodoxearMarkdown.chatMarkdownHtmlCached('
            '"Attachment 1: /home/u/.local/share/codoxear/uploads/broker-1/1_shot.png\\n'
            'Attachment 2: /home/u/.local/share/codoxear/uploads/broker-1/2_notes.txt\\n'
            'what do you see?", "sid-1")'
        )
    )
    assert '<img src="/home/u/.local/share/codoxear/uploads/broker-1/1_shot.png" alt="1_shot.png">' in html
    # Non-image attachments keep their plain line; the path stays visible text.
    assert "Attachment 2: /home/u/.local/share/codoxear/uploads/broker-1/2_notes.txt" in html
    assert "what do you see?" in html


def test_chat_markdown_leaves_messages_without_prefix_untouched() -> None:
    html = json.loads(run_markdown_expr('context.window.CodoxearMarkdown.chatMarkdownHtmlCached("hello world", "sid-1")'))
    assert "<p>hello world</p>" in html


def _run_markdown_program(body: str) -> dict:
    app_source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    marked_source = MARKED_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const context = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{ CodoxearUrls: {{ resolveAppUrl: (path) => path }} }},
          console,
        }};
        vm.createContext(context);
        vm.runInContext({json.dumps(marked_source)}, context);
        context.window.marked = context.marked;
        vm.runInContext({json.dumps(app_source)}, context);
        const image = () => ({{
          style: {{}},
          dataset: {{}},
          attributes: {{}},
          setAttribute(name, value) {{ this.attributes[name] = String(value); }},
          getAttribute(name) {{ return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null; }},
        }});
        {body}
        """
    )
    proc = subprocess.run(
        ["node", "-e", program],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def test_an_image_box_is_reserved_before_its_ratio_is_known() -> None:
    """A source-less pending image must still occupy a box.

    The transcript's height has to be stable while the image's header
    dimensions are in flight: a collapsed box grows by a whole screen when the
    ratio lands and moves whatever the reader is looking at (which is the
    reported session-switch symptom).
    """
    out = _run_markdown_program(
        """
        const unknown = image();
        context.window.CodoxearMarkdown.prepareImageForDisplay(unknown, "/api/files/blob?path=shot.png");
        const known = image();
        context.window.localStorage = {
          _value: JSON.stringify({ "/api/files/blob?path=tall.png": { width: 900, height: 1600, usedAt: 1 } }),
          getItem() { return this._value; },
          setItem(_key, value) { this._value = value; },
        };
        context.window.CodoxearMarkdown.prepareImageForDisplay(known, "/api/files/blob?path=tall.png");
        process.stdout.write(JSON.stringify({ unknown: {
          ratio: unknown.style.aspectRatio, width: unknown.style.width, height: unknown.style.height,
          attrWidth: unknown.getAttribute("width"), attrHeight: unknown.getAttribute("height"),
        }, known: {
          ratio: known.style.aspectRatio, attrWidth: known.getAttribute("width"), attrHeight: known.getAttribute("height"),
        } }));
        """
    )
    # Unknown ratio: reserve the default 16:9 box instead of collapsing.
    assert out["unknown"]["ratio"] == "16 / 9"
    assert out["unknown"]["width"] == "100%"
    assert out["unknown"]["height"] == "auto"
    assert (out["unknown"]["attrWidth"], out["unknown"]["attrHeight"]) == ("16", "9")
    # Known ratio: the cached dimensions are the box.
    assert out["known"]["ratio"] == "900 / 1600"
    assert (out["known"]["attrWidth"], out["known"]["attrHeight"]) == ("900", "1600")
