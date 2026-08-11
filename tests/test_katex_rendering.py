"""Behavioral integration coverage for the vendored KaTeX runtime."""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = module_path("app_markdown.js")
KATEX_JS = ROOT / "codoxear" / "static" / "vendor" / "katex.min.js"
MARKED_JS = ROOT / "codoxear" / "static" / "vendor" / "marked.min.js"


def render_markdown_with_vendored_katex(markdown: str) -> str:
    program = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const context = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{ CodoxearUrls: {{ resolveAppUrl: (path) => path }} }},
        }};
        vm.createContext(context);
        vm.runInContext({json.dumps(MARKED_JS.read_text(encoding="utf-8"))}, context);
        context.window.marked = context.marked;
        vm.runInContext({json.dumps(KATEX_JS.read_text(encoding="utf-8"))}, context);
        vm.runInContext({json.dumps(APP_MARKDOWN_JS.read_text(encoding="utf-8"))}, context);
        process.stdout.write(context.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)}));
        """
    )
    return subprocess.run(
        ["node"],
        input=program,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_vendored_katex_renders_inline_and_display_math_through_markdown_pipeline() -> None:
    html = render_markdown_with_vendored_katex(r"Inline $x^2$ and display $$\frac{a}{b}$$.")

    assert 'class="katex"' in html
    assert 'class="katex-html" aria-hidden="true"' in html
    assert '<annotation encoding="application/x-tex">x^2</annotation>' in html
    assert '<annotation encoding="application/x-tex">\\frac{a}{b}</annotation>' in html
    assert 'class="katex-display"' in html
    assert "md-math-fallback" not in html
