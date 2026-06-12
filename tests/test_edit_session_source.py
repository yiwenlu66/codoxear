import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestEditSessionSource(unittest.TestCase):
    def test_edit_save_is_bound_to_original_session(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("editSaveBtn.onclick = async () => {")
        end = source.index("newSessionCloseBtn.onclick", start)
        block = source[start:end]

        self.assertIn("const sid = editSessionId;", block)
        self.assertIn("if (!sid || editSaveBtn.disabled) return;", block)
        self.assertIn("editSaveBtn.disabled = true;", block)
        self.assertIn("await api(`/api/sessions/${sid}/edit`", block)
        self.assertIn("await refreshSessions();\n            if (editSessionId !== sid) return;", block)
        self.assertIn("if (editSessionId !== sid) return;\n            editStatus.textContent", block)
        self.assertIn("if (editSessionId === sid) editSaveBtn.disabled = false;", block)
        self.assertIn("if (selected === sid)", block)
        self.assertIn("const s2 = sessionIndex.get(sid);", block)
        self.assertNotIn("/api/sessions/${editSessionId}/edit", block)
        self.assertNotIn("selected === editSessionId", block)

    def test_edit_modal_open_and_hide_reset_save_disabled_state(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        hide_start = source.index("function hideEditSession()")
        hide_end = source.index("function syncEditPriorityLabel()", hide_start)
        hide_block = source[hide_start:hide_end]
        open_start = source.index("function openEditSession(sid)")
        open_end = source.index("function hideNewSessionDialog()", open_start)
        open_block = source[open_start:open_end]

        self.assertIn("editSaveBtn.disabled = false;", hide_block)
        self.assertIn("editSessionId = sid;\n          editStatus.textContent = \"\";\n          editSaveBtn.disabled = false;", open_block)


if __name__ == "__main__":
    unittest.main()
