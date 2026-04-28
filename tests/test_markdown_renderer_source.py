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
