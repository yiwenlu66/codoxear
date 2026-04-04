import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def _extract_named_function(name: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    anchor = f"function {name}("
    start = source.index(anchor)
    func_start = source.index("{", start)
    depth = 0
    end = func_start
    for index in range(func_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    else:
        raise AssertionError(f"failed to extract function {name}")
    return source[start:end]


def eval_textarea_keydown(
    *,
    enter_to_send_enabled: bool,
    key: str = "Enter",
    is_composing: bool = False,
    shift_key: bool = False,
    ctrl_key: bool = False,
    meta_key: bool = False,
) -> dict[str, object]:
    handler = _extract_named_function("handleComposerKeydown")
    injected = json.dumps(handler + "\nglobalThis.__test_textarea_keydown = handleComposerKeydown;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          submitted: 0,
          prevented: 0,
          enterToSendEnabled: () => {json.dumps(enter_to_send_enabled)},
          form: {{
            requestSubmit() {{
              ctx.submitted += 1;
            }},
          }},
        }};
        const event = {{
          key: {json.dumps(key)},
          isComposing: {json.dumps(is_composing)},
          shiftKey: {json.dumps(shift_key)},
          ctrlKey: {json.dumps(ctrl_key)},
          metaKey: {json.dumps(meta_key)},
          preventDefault() {{
            ctx.prevented += 1;
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        ctx.__test_textarea_keydown(event);
        process.stdout.write(JSON.stringify({{
          submitted: ctx.submitted,
          prevented: ctx.prevented,
        }}));
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


class TestEnterToSendUi(unittest.TestCase):
    def test_settings_source_includes_enter_to_send_toggle(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('localStorage.getItem("codoxear.enterToSend") === "1"', source)
        self.assertIn('const enterToSendSettingToggle = el("input", { id: "enterToSendSettingToggle", type: "checkbox" });', source)
        self.assertIn('el("span", { text: "Press Enter to send" })', source)
        self.assertIn('function setEnterToSendEnabled(enabled) {', source)
        self.assertIn('enterToSendSettingToggle.checked = !!enterToSendEnabled();', source)

    def test_plain_enter_submits_when_enter_to_send_enabled(self) -> None:
        result = eval_textarea_keydown(enter_to_send_enabled=True)

        self.assertEqual(result["submitted"], 1)
        self.assertEqual(result["prevented"], 1)

    def test_shift_enter_does_not_submit_when_enter_to_send_enabled(self) -> None:
        result = eval_textarea_keydown(enter_to_send_enabled=True, shift_key=True)

        self.assertEqual(result["submitted"], 0)
        self.assertEqual(result["prevented"], 0)

    def test_plain_enter_does_not_submit_when_enter_to_send_disabled(self) -> None:
        result = eval_textarea_keydown(enter_to_send_enabled=False)

        self.assertEqual(result["submitted"], 0)
        self.assertEqual(result["prevented"], 0)

    def test_ctrl_or_cmd_enter_submit_when_enter_to_send_disabled(self) -> None:
        for key_name, modifiers in (
            ("Ctrl+Enter", {"ctrl_key": True}),
            ("Cmd+Enter", {"meta_key": True}),
        ):
            with self.subTest(key_name=key_name):
                result = eval_textarea_keydown(
                    enter_to_send_enabled=False,
                    **modifiers,
                )

                self.assertEqual(result["submitted"], 1)
                self.assertEqual(result["prevented"], 1)

    def test_composing_enter_never_submits(self) -> None:
        for enabled in (False, True):
            with self.subTest(enter_to_send_enabled=enabled):
                result = eval_textarea_keydown(
                    enter_to_send_enabled=enabled,
                    is_composing=True,
                )

                self.assertEqual(result["submitted"], 0)
                self.assertEqual(result["prevented"], 0)


if __name__ == "__main__":
    unittest.main()
