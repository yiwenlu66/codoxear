"""Behavioral coverage for `@`-mention file references.

The compose-side `@` picker inserts an `@`-prefixed path, so the renderer must
resolve the path behind the marker while keeping the marker in the label. These
tests execute the real module in a Node VM: `parseLocalFileRef` directly, and
`rewriteTextFileRefs` / `rewriteInlineCodeFileRefs` against a minimal fake DOM
(the same pattern as `test_markdown_video.py`).
"""
from __future__ import annotations

from frontend_module_loader import module_path

import json
import subprocess
import textwrap


APP_MARKDOWN_JS = module_path("app_markdown.js")


def run_markdown_vm(body: str) -> object:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
        const ctx = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{ CodoxearUrls: {{ resolveAppUrl: (path) => path }} }},
        }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);
        const api = ctx.window.CodoxearMarkdown;

        function makeRoot(tagName) {{
          return {{ tagName, parentElement: null }};
        }}
        function textNode(text, parent) {{
          return {{
            nodeValue: text,
            parentElement: parent,
            replaced: null,
            replaceWith(fragment) {{ this.replaced = fragment; }},
          }};
        }}
        function makeDoc(nodes) {{
          const pending = nodes.slice();
          return {{
            createTreeWalker() {{ return {{ nextNode() {{ return pending.length ? pending.shift() : null; }} }}; }},
            createDocumentFragment() {{
              return {{ children: [], append(...items) {{ this.children.push(...items); }} }};
            }},
            createElement(tag) {{
              return {{ tagName: String(tag).toUpperCase(), dataset: {{}}, textContent: "", children: [], appendChild(child) {{ this.children.push(child); }} }};
            }},
          }};
        }}
        function spanSummary(span) {{
          const line = span.dataset.candidateFileLine;
          return {{ path: span.dataset.candidateFilePath, line: line == null ? null : String(line), label: span.textContent }};
        }}
        function fragmentSummary(fragment) {{
          return fragment.children
            .filter((child) => typeof child !== "string" || child !== "")
            .map((child) => (typeof child === "string" ? {{ text: child }} : spanSummary(child)));
        }}
        function codeNode(text) {{
          return {{
            tagName: "CODE",
            parentElement: null,
            textContent: text,
            replaced: null,
            replaceChildren(...children) {{ this.replaced = children; }},
          }};
        }}
        let result = null;
        """
    ) + textwrap.dedent(body) + "\nprocess.stdout.write(JSON.stringify(result));\n"
    completed = subprocess.run(["node", "-e", script], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_mention_path_resolution_matrix() -> None:
    cases = [
        "@src/a.js",
        "@src/a.js:12",
        "@src/a.js#L7",
        "@README.md",
        "@/etc/hosts",
        "@scope/pkg",
        "@~/notes.md",
        "@foo",
        "@",
        "@.gitignore",
        "src/@scope/x.js",
        "foo@bar.com",
    ]
    result = run_markdown_vm(
        "const refs = " + json.dumps(cases) + ".map((raw) => [raw, api.parseLocalFileRef(raw)]);\n"
        "result = { refs };"
    )
    assert dict(result["refs"]) == {
        "@src/a.js": {"path": "src/a.js", "line": None},
        "@src/a.js:12": {"path": "src/a.js", "line": 12},
        "@src/a.js#L7": {"path": "src/a.js", "line": 7},
        "@README.md": {"path": "README.md", "line": None},
        "@/etc/hosts": {"path": "/etc/hosts", "line": None},
        "@scope/pkg": {"path": "scope/pkg", "line": None},
        "@~/notes.md": {"path": "~/notes.md", "line": None},
        "@foo": None,
        "@": None,
        "@.gitignore": None,
        "src/@scope/x.js": {"path": "src/@scope/x.js", "line": None},
        "foo@bar.com": None,
    }


def test_text_refs_link_the_mention_and_keep_its_label() -> None:
    result = run_markdown_vm(
        """
        const cases = [
          "see @src/a.js end",
          "看看@src/a.js",
          "@src/a.js",
          "@src/a.js:12",
          "@scope/pkg and more",
          "mail foo@bar.com now",
          "@@MATH1@@",
          "100~300",
          "see src/@scope/x.js",
        ];
        const rows = cases.map((text) => {
          const root = makeRoot("DIV");
          const node = textNode(text, root);
          const doc = makeDoc([node]);
          api.rewriteTextFileRefs(root, doc, {});
          return { text, parts: node.replaced ? fragmentSummary(node.replaced) : null };
        });
        result = { rows };
        """
    )
    rows = {row["text"]: row["parts"] for row in result["rows"]}
    assert rows["see @src/a.js end"] == [
        {"text": "see "},
        {"path": "src/a.js", "line": None, "label": "@src/a.js"},
        {"text": " end"},
    ]
    assert rows["看看@src/a.js"] == [
        {"text": "看看"},
        {"path": "src/a.js", "line": None, "label": "@src/a.js"},
    ]
    assert rows["@src/a.js"] == [{"path": "src/a.js", "line": None, "label": "@src/a.js"}]
    assert rows["@src/a.js:12"] == [{"path": "src/a.js", "line": "12", "label": "@src/a.js:12"}]
    assert rows["@scope/pkg and more"] == [
        {"path": "scope/pkg", "line": None, "label": "@scope/pkg"},
        {"text": " and more"},
    ]
    # An email-shaped token is not a file reference, so nothing is rewritten and
    # the domain can never be mistaken for a mention target.
    assert rows["mail foo@bar.com now"] is None
    assert rows["@@MATH1@@"] is None
    assert rows["100~300"] is None
    assert rows["see src/@scope/x.js"] == [
        {"text": "see "},
        {"path": "src/@scope/x.js", "line": None, "label": "src/@scope/x.js"},
    ]


def test_inline_code_mention_is_linked() -> None:
    result = run_markdown_vm(
        """
        const code = codeNode("@src/a.js");
        const root = { tagName: "DIV", parentElement: null, querySelectorAll: (sel) => (sel === "code" ? [code] : []) };
        api.rewriteInlineCodeFileRefs(root, makeDoc([]), {});
        result = { spans: code.replaced ? code.replaced.map(spanSummary) : null };
        """
    )
    assert result["spans"] == [{"path": "src/a.js", "line": None, "label": "@src/a.js"}]
