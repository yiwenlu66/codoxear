import json
import re
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def eval_render_diag_todo_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function renderDiagTodoSnapshot(snapshot) {")
    end = source.index("async function showDiagViewer() {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_renderDiagTodoSnapshot = renderDiagTodoSnapshot;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        function makeNode(tag, attrs = {{}}, children = []) {{
          const node = {{
            tagName: String(tag || "div").toUpperCase(),
            className: attrs.class || "",
            textContent: attrs.text || "",
            children: [],
            appendChild(child) {{ this.children.push(child); return child; }},
          }};
          for (const child of children) node.appendChild(child);
          return node;
        }}
        function serialize(node) {{
          if (!node) return null;
          return {{
            tagName: String(node.tagName || ""),
            className: String(node.className || ""),
            textContent: String(node.textContent || ""),
            children: Array.isArray(node.children) ? node.children.map(serialize) : [],
          }};
        }}
        const ctx = {{ el: makeNode }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        const root = ctx.__test_renderDiagTodoSnapshot({json.dumps(snapshot)});
        process.stdout.write(JSON.stringify(serialize(root)));
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


def flatten_rendered_text(node: dict[str, object] | None) -> str:
    if not node:
        return ""
    text = str(node.get("textContent") or "")
    children = node.get("children") or []
    child_text = " ".join(flatten_rendered_text(child) for child in children)
    return (text + " " + child_text).strip()


def collect_class_names(node: dict[str, object] | None) -> set[str]:
    if not node:
        return set()
    class_names = {token for token in str(node.get("className") or "").split() if token}
    for child in node.get("children") or []:
        class_names.update(collect_class_names(child))
    return class_names


def count_class_name(node: dict[str, object] | None, target: str) -> int:
    if not node:
        return 0
    own = 1 if target in str(node.get("className") or "").split() else 0
    return own + sum(count_class_name(child, target) for child in node.get("children") or [])


class TestPiDetailsTodoUi(unittest.TestCase):
    def test_details_todo_css_and_rendered_class_contract(self) -> None:
        source = APP_CSS.read_text(encoding="utf-8")
        tree = eval_render_diag_todo_snapshot(
            {
                "available": True,
                "error": False,
                "progress_text": "2/3 completed",
                "items": [
                    {
                        "id": 1,
                        "title": "Explore project context",
                        "description": "inspect files first",
                        "status": "completed",
                    },
                    {
                        "id": 2,
                        "title": "Ask clarifying questions",
                        "description": "confirm scope",
                        "status": "in-progress",
                    },
                ],
            }
        )
        class_names = collect_class_names(tree)
        required_selectors = {
            ".detailsSection": "detailsSection",
            ".todoSnapshotSection": "todoSnapshotSection",
            ".detailsSectionTitle": "detailsSectionTitle",
            ".todoSnapshotSummary": "todoSnapshotSummary",
            ".todoSnapshotList": "todoSnapshotList",
            ".todoSnapshotItem": "todoSnapshotItem",
            ".todoSnapshotHead": "todoSnapshotHead",
            ".todoSnapshotTitle": "todoSnapshotTitle",
            ".todoStatusChip": "todoStatusChip",
            ".todoSnapshotDescription": "todoSnapshotDescription",
        }

        for selector, class_name in required_selectors.items():
            self.assertIn(selector, source)
            self.assertIn(class_name, class_names)

        self.assertEqual(2, count_class_name(tree, "todoSnapshotItem"))
        self.assertEqual(1, count_class_name(tree, "todoSnapshotList"))

    def test_todo_section_border_override_avoids_double_divider(self) -> None:
        source = APP_CSS.read_text(encoding="utf-8")

        self.assertRegex(
            source,
            re.compile(r"\.detailsRow\s*\+\s*\.detailsSection\s*\{[^}]*border-top:\s*0\s*;", re.DOTALL),
        )

    def test_render_diag_todo_snapshot_shows_summary_and_items(self) -> None:
        text = flatten_rendered_text(
            eval_render_diag_todo_snapshot(
                {
                    "available": True,
                    "error": False,
                    "progress_text": "2/3 completed",
                    "items": [
                        {
                            "id": 1,
                            "title": "Explore project context",
                            "description": "inspect files first",
                            "status": "completed",
                        },
                        {
                            "id": 2,
                            "title": "Ask clarifying questions",
                            "description": "confirm scope",
                            "status": "in-progress",
                        },
                    ],
                }
            )
        )

        self.assertIn("Todo list", text)
        self.assertIn("2/3 completed", text)
        self.assertIn("Explore project context", text)
        self.assertIn("Ask clarifying questions", text)
        self.assertIn("completed", text)
        self.assertIn("in-progress", text)

    def test_render_diag_todo_snapshot_distinguishes_empty_and_error_states(self) -> None:
        self.assertIn(
            "No todo list yet",
            flatten_rendered_text(eval_render_diag_todo_snapshot({"available": False, "error": False, "items": []})),
        )
        self.assertIn(
            "Todo list unavailable",
            flatten_rendered_text(eval_render_diag_todo_snapshot({"available": False, "error": True, "items": []})),
        )

        empty_tree = eval_render_diag_todo_snapshot({"available": False, "error": False, "items": []})
        self.assertIn("detailsSectionEmpty", collect_class_names(empty_tree))
        self.assertNotIn("todoSnapshotList", collect_class_names(empty_tree))

    def test_render_diag_todo_snapshot_keeps_available_empty_snapshot(self) -> None:
        tree = eval_render_diag_todo_snapshot(
            {"available": True, "error": False, "progress_text": "0/0 completed", "items": []}
        )
        text = flatten_rendered_text(tree)

        self.assertIn("0/0 completed", text)
        self.assertNotIn("No todo list yet", text)
        self.assertIn("todoSnapshotList", collect_class_names(tree))

    def test_show_diag_viewer_calls_render_diag_todo_snapshot(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block_start = source.index("async function showDiagViewer() {")
        block_end = source.index("function hideDiagViewer() {", block_start)
        block = source[block_start:block_end]

        self.assertIn("const todoSection = renderDiagTodoSnapshot", block)
        self.assertIn("diagContent.appendChild(todoSection)", block)

    def test_show_diag_viewer_uses_pi_specific_session_file_label(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block_start = source.index("async function showDiagViewer() {")
        block_end = source.index("function hideDiagViewer() {", block_start)
        block = source[block_start:block_end]

        self.assertIn('addRow("Log", d && d.log_path ? d.log_path : "-", { mono: true });', block)
        self.assertIn('addRow("Session file", d && d.session_file_path ? d.session_file_path : "-", { mono: true });', block)


if __name__ == "__main__":
    unittest.main()
