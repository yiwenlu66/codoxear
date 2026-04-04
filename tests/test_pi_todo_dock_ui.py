import json
import subprocess
import textwrap
import unittest
from pathlib import Path
from typing import cast


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def eval_resolve_todo_dock_snapshot(events: list[dict[str, object]], diagnostics_snapshot: dict[str, object] | None) -> object:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function normalizeTodoDockSnapshot(snapshot) {")
    end = source.index("function updateTodoDockJumpOffset() {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_resolveTodoDockSnapshot = resolveTodoDockSnapshot;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        const result = ctx.__test_resolveTodoDockSnapshot({json.dumps(events)}, {json.dumps(diagnostics_snapshot)});
        process.stdout.write(JSON.stringify(result ?? null));
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


class TestPiTodoDockUi(unittest.TestCase):
    def test_main_layout_places_todo_dock_between_chat_and_composer(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const todoDockHost = el("div", { class: "todoDockHost", id: "todoDockHost" });', source)
        self.assertLess(source.index("main.appendChild(chatWrap);"), source.index("main.appendChild(todoDockHost);"))
        self.assertLess(source.index("main.appendChild(todoDockHost);"), source.index("main.appendChild(composer);"))

    def test_resolve_todo_dock_snapshot_prefers_latest_event_snapshot(self) -> None:
        result = eval_resolve_todo_dock_snapshot(
            [
                {"type": "todo_snapshot", "ts": 10, "progress_text": "1/3", "items": [{"title": "old"}], "counts": {"total": 3, "completed": 1}},
                {"type": "assistant", "text": "ignore me", "ts": 11},
                {"type": "todo_snapshot", "ts": 12, "progress_text": "2/3", "items": [{"title": "new"}], "counts": {"total": 3, "completed": 2}},
            ],
            {"available": True, "error": False, "progress_text": "diag", "items": [{"title": "diag"}], "counts": {"total": 1, "completed": 0}},
        )

        self.assertIsInstance(result, dict)
        result_dict = cast(dict[str, object], result)
        self.assertEqual("2/3", result_dict["progress_text"])
        self.assertEqual("new", cast(list[dict[str, object]], result_dict["items"])[0]["title"])
        self.assertEqual(12, result_dict["ts"])

    def test_resolve_todo_dock_snapshot_falls_back_to_diagnostics(self) -> None:
        result = eval_resolve_todo_dock_snapshot(
            [{"type": "assistant", "text": "hello", "ts": 7}],
            {"available": True, "error": False, "progress_text": "4/5", "items": [{"title": "diag"}], "counts": {"total": 5, "completed": 4}},
        )

        self.assertIsInstance(result, dict)
        result_dict = cast(dict[str, object], result)
        self.assertEqual("4/5", result_dict["progress_text"])
        self.assertEqual("diag", cast(list[dict[str, object]], result_dict["items"])[0]["title"])
        self.assertNotIn("ts", result_dict)

    def test_resolve_todo_dock_snapshot_hides_when_no_snapshot_exists(self) -> None:
        result = eval_resolve_todo_dock_snapshot(
            [{"type": "assistant", "text": "hello", "ts": 7}],
            {"available": False, "error": False, "items": []},
        )

        self.assertIsNone(result)

    def test_todo_dock_css_contract_exists(self) -> None:
        source = APP_CSS.read_text(encoding="utf-8")

        for selector in [
            ".todoDockHost",
            ".todoDockPanel",
            ".todoDockSummary",
            ".todoDockBody",
            ".todoDockPanel.expanded .todoDockBody",
        ]:
            self.assertIn(selector, source)

    def test_jump_button_stays_decoupled_from_todo_dock_height(self) -> None:
        css_source = APP_CSS.read_text(encoding="utf-8")
        js_source = APP_JS.read_text(encoding="utf-8")

        self.assertNotIn(".chatWrap.todoDockExpanded .jumpBtn", css_source)
        self.assertNotIn("--todoDockJumpOffset", css_source)
        self.assertNotIn("todoDockBody.offsetHeight", js_source)
        self.assertNotIn('style.setProperty("--todoDockJumpOffset"', js_source)


if __name__ == "__main__":
    unittest.main()
