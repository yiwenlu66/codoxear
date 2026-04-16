# Enter-to-Send Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser-local `Press Enter to send` setting so the composer can switch between the current `Ctrl/Cmd+Enter` behavior and chat-style `Enter` send with `Shift+Enter` newline.

**Architecture:** Keep the feature entirely in the web client by extending the existing Settings dialog in `codoxear/static/app.js` and reusing the current `form.requestSubmit()` send path. Persist the preference in `localStorage`, expose it through a dedicated helper, and branch only inside the textarea `keydown` handler so queue/send/attachment flows remain unchanged.

**Tech Stack:** Vanilla browser JS in `codoxear/static/app.js`, Python `unittest` source/runtime tests, Node `vm` harness for evaluating extracted browser snippets.

---

Precondition: execute this plan in a clean dedicated worktree. The current main worktree already contains unrelated edits, so implementation should not start there.

## File Map

- Modify: `codoxear/static/app.js`
  - Add browser-local `enter-to-send` state near the existing local notification/announcement toggles.
  - Extend the existing Settings dialog with a new checkbox row.
  - Sync the checkbox from local state and persist changes back to `localStorage`.
  - Update the textarea `keydown` handler so the shortcut behavior changes without altering `form.onsubmit`.
- Create: `tests/test_enter_to_send_ui.py`
  - Source assertions for the new setting key, UI control, and label.
  - Runtime tests that execute the extracted `keydown` handler under different modifier-key combinations.
- Verify only: `tests/test_send_failure_ui.py`
  - Protect the existing `form.requestSubmit()` / submit recovery behavior while adding the new keybinding mode.
- Verify only: `tests/test_voice_playback_source.py`
  - Ensure changes in the shared Settings dialog do not accidentally remove existing settings controls.

### Task 1: Add the Settings Toggle and Local Persistence

**Files:**
- Modify: `codoxear/static/app.js`
- Create: `tests/test_enter_to_send_ui.py`
- Verify: `tests/test_voice_playback_source.py`

- [ ] **Step 1: Write the failing Settings source test**

```python
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestEnterToSendSettingsSource(unittest.TestCase):
    def test_settings_dialog_exposes_enter_to_send_toggle(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('localStorage.getItem("codoxear.enterToSend") === "1"', source)
        self.assertIn(
            'const enterToSendSettingToggle = el("input", { id: "enterToSendSettingToggle", type: "checkbox" });',
            source,
        )
        self.assertIn('el("span", { text: "Press Enter to send" })', source)
        self.assertIn('function setEnterToSendEnabled(enabled) {', source)
        self.assertIn('enterToSendSettingToggle.checked = !!enterToSendEnabled();', source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_enter_to_send_ui -v`
Expected: FAIL with an `AssertionError` because `codoxear/static/app.js` does not yet contain the `codoxear.enterToSend` state or Settings toggle.

- [ ] **Step 3: Add the browser-local state and Settings checkbox in `codoxear/static/app.js`**

```javascript
let localAnnouncementEnabled = localStorage.getItem("codoxear.announcementEnabled") === "1";
let localNotificationEnabled = localStorage.getItem("codoxear.notificationEnabled") === "1";
let localEnterToSend = localStorage.getItem("codoxear.enterToSend") === "1";

function enterToSendEnabled() {
  return !!localEnterToSend;
}

function setEnterToSendEnabled(enabled) {
  localEnterToSend = !!enabled;
  if (localEnterToSend) localStorage.setItem("codoxear.enterToSend", "1");
  else localStorage.removeItem("codoxear.enterToSend");
  if (enterToSendSettingToggle) enterToSendSettingToggle.checked = !!enterToSendEnabled();
}

const narrationSettingToggle = el("input", { id: "narrationSettingToggle", type: "checkbox" });
const enterToSendSettingToggle = el("input", { id: "enterToSendSettingToggle", type: "checkbox" });
const voiceSettingsViewer = el("dialog", { class: "formViewer formDialog", id: "voiceSettingsViewer", "aria-label": "Settings" }, [
  el("div", { class: "queueHeader" }, [
    el("div", { class: "title", text: "Settings" }),
    el("div", { class: "actions" }, [voiceSettingsCloseBtn]),
  ]),
  voiceSettingsStatus,
  el("div", { class: "formBody" }, [
    el("label", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "OpenAI-compatible API base URL" }),
      voiceBaseUrlInput,
      el("span", { class: "fieldHint", text: "Used for both summarization and speech." }),
    ]),
    el("label", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "OpenAI-compatible API key" }),
      voiceApiKeyInput,
    ]),
    el("div", { class: "field" }, [
      el("label", { class: "voiceToggleRow" }, [
        narrationSettingToggle,
        el("span", { text: "Announce narration messages" }),
      ]),
    ]),
    el("div", { class: "field" }, [
      el("label", { class: "voiceToggleRow" }, [
        enterToSendSettingToggle,
        el("span", { text: "Press Enter to send" }),
      ]),
    ]),
  ]),
  el("div", { class: "formActions" }, [
    el("button", { id: "voiceSettingsCancelBtn", type: "button", text: "Cancel" }),
    el("button", { id: "voiceSettingsSaveBtn", class: "primary", type: "button", text: "Save" }),
  ]),
]);

function updateVoiceUi() {
  announceBtn.classList.toggle("active", voiceAnnouncementsEnabled());
  announceBtn.title = voiceAnnouncementsEnabled() ? "Announcements on" : "Announcements off";
  announceBtn.setAttribute("aria-label", announceBtn.title);
  notificationBtn.classList.toggle("active", notificationsEnabledLocally());
  const transport = activeNotificationTransport();
  notificationBtn.title = notificationsEnabledLocally()
    ? transport === "push"
      ? "Notifications on (push)"
      : transport === "desktop"
        ? "Notifications on"
        : "Notifications pending"
    : "Notifications off";
  notificationBtn.setAttribute("aria-label", notificationBtn.title);
  if (voiceBaseUrlInput) voiceBaseUrlInput.value = String(voiceSettings.tts_base_url || "");
  if (voiceApiKeyInput && !voiceApiKeyInput.matches(":focus")) voiceApiKeyInput.value = String(voiceSettings.tts_api_key || "");
  if (narrationSettingToggle) narrationSettingToggle.checked = !!voiceSettings.tts_enabled_for_narration;
  if (enterToSendSettingToggle) enterToSendSettingToggle.checked = !!enterToSendEnabled();
  notificationState.permission = typeof Notification === "undefined" ? "unsupported" : Notification.permission;
}

enterToSendSettingToggle.onchange = (e) => {
  setEnterToSendEnabled(Boolean(e.target.checked));
};
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_enter_to_send_ui tests.test_voice_playback_source -v`
Expected: PASS. The new source test should succeed, and the existing Settings-related playback test should still find the narration toggle and related strings.

- [ ] **Step 5: Commit the Settings-only slice**

```bash
git add codoxear/static/app.js tests/test_enter_to_send_ui.py
git commit -m "feat: add enter-to-send preference toggle"
```

### Task 2: Add Keyboard Shortcut Behavior Behind the Toggle

**Files:**
- Modify: `codoxear/static/app.js`
- Modify: `tests/test_enter_to_send_ui.py`
- Verify: `tests/test_send_failure_ui.py`

- [ ] **Step 1: Extend the test file with failing keyboard-behavior coverage**

```python
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_enter_key_handler(*, enter_to_send: bool, shift: bool = False, ctrl: bool = False, meta: bool = False, composing: bool = False) -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index('textarea.addEventListener("keydown", (e) => {')
    end = source.index('window.addEventListener("resize", () => {', start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_keyHandler = __capturedKeyHandler;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        let __capturedKeyHandler = null;
        const textarea = {{
          addEventListener(type, handler) {{
            if (type === "keydown") __capturedKeyHandler = handler;
          }},
        }};
        const form = {{
          requestSubmit() {{
            ctx.submitCalls += 1;
          }},
        }};
        const ctx = {{
          textarea,
          form,
          submitCalls: 0,
          enterToSendEnabled: () => {json.dumps(enter_to_send)},
        }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        const event = {{
          key: "Enter",
          shiftKey: {json.dumps(shift)},
          ctrlKey: {json.dumps(ctrl)},
          metaKey: {json.dumps(meta)},
          isComposing: {json.dumps(composing)},
          defaultPrevented: false,
          preventDefault() {{
            this.defaultPrevented = true;
          }},
        }};
        ctx.__test_keyHandler(event);
        process.stdout.write(JSON.stringify({{
          submitCalls: ctx.submitCalls,
          defaultPrevented: event.defaultPrevented,
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


class TestEnterToSendSettingsSource(unittest.TestCase):
    def test_settings_dialog_exposes_enter_to_send_toggle(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('localStorage.getItem("codoxear.enterToSend") === "1"', source)
        self.assertIn(
            'const enterToSendSettingToggle = el("input", { id: "enterToSendSettingToggle", type: "checkbox" });',
            source,
        )
        self.assertIn('el("span", { text: "Press Enter to send" })', source)
        self.assertIn('function setEnterToSendEnabled(enabled) {', source)
        self.assertIn('enterToSendSettingToggle.checked = !!enterToSendEnabled();', source)


class TestEnterToSendKeyHandling(unittest.TestCase):
    def test_plain_enter_submits_when_toggle_enabled(self) -> None:
        result = eval_enter_key_handler(enter_to_send=True)

        self.assertEqual(result["submitCalls"], 1)
        self.assertTrue(result["defaultPrevented"])

    def test_shift_enter_keeps_newline_when_toggle_enabled(self) -> None:
        result = eval_enter_key_handler(enter_to_send=True, shift=True)

        self.assertEqual(result["submitCalls"], 0)
        self.assertFalse(result["defaultPrevented"])

    def test_ctrl_or_cmd_enter_still_submits_when_toggle_disabled(self) -> None:
        ctrl_result = eval_enter_key_handler(enter_to_send=False, ctrl=True)
        meta_result = eval_enter_key_handler(enter_to_send=False, meta=True)

        self.assertEqual(ctrl_result["submitCalls"], 1)
        self.assertTrue(ctrl_result["defaultPrevented"])
        self.assertEqual(meta_result["submitCalls"], 1)
        self.assertTrue(meta_result["defaultPrevented"])

    def test_composing_enter_never_submits(self) -> None:
        result = eval_enter_key_handler(enter_to_send=True, composing=True)

        self.assertEqual(result["submitCalls"], 0)
        self.assertFalse(result["defaultPrevented"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails on plain `Enter` send**

Run: `python3 -m unittest tests.test_enter_to_send_ui -v`
Expected: FAIL in `test_plain_enter_submits_when_toggle_enabled` because the current handler still requires `Ctrl` or `Cmd`.

- [ ] **Step 3: Update the textarea `keydown` handler in `codoxear/static/app.js`**

```javascript
textarea.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (e.isComposing) return;

  if (enterToSendEnabled()) {
    if (e.shiftKey) return;
    e.preventDefault();
    form.requestSubmit();
    return;
  }

  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  form.requestSubmit();
});
```

- [ ] **Step 4: Run the targeted regression suite**

Run: `python3 -m unittest tests.test_enter_to_send_ui tests.test_send_failure_ui tests.test_voice_playback_source -v`
Expected: PASS. The new keyboard tests should all pass, the existing submit-recovery tests should keep passing because `form.onsubmit` is untouched, and the Settings-source regression should still pass.

- [ ] **Step 5: Commit the shortcut slice**

```bash
git add codoxear/static/app.js tests/test_enter_to_send_ui.py
git commit -m "feat: add enter-to-send composer shortcut"
```

### Task 3: Final Verification Before Merge or PR

**Files:**
- Verify: `codoxear/static/app.js`
- Verify: `tests/test_enter_to_send_ui.py`
- Verify: `tests/test_send_failure_ui.py`
- Verify: `tests/test_voice_playback_source.py`

- [ ] **Step 1: Re-run the focused feature suite from a clean shell**

Run: `python3 -m unittest tests.test_enter_to_send_ui tests.test_send_failure_ui tests.test_voice_playback_source -v`
Expected: PASS with all enter-to-send, submit-recovery, and shared Settings checks green.

- [ ] **Step 2: Manually inspect the final diff**

Run: `git diff -- codoxear/static/app.js tests/test_enter_to_send_ui.py`
Expected: The diff is limited to the new local setting, the Settings checkbox row, the textarea `keydown` branch, and the dedicated regression test file.

- [ ] **Step 3: Smoke-check the implementation in a browser-backed session**

Run: `codoxear-server`
Expected: The app starts normally. In the browser, verify that:
- Settings shows `Press Enter to send`
- unchecked mode keeps `Ctrl/Cmd+Enter` send behavior
- checked mode sends on `Enter`
- checked mode preserves newline on `Shift+Enter`
- reloading the page keeps the preference on the same browser

- [ ] **Step 4: Commit the verification checkpoint if anything changed during smoke testing**

```bash
git add codoxear/static/app.js tests/test_enter_to_send_ui.py
git commit -m "test: verify enter-to-send composer behavior"
```

If the smoke test does not require code changes, skip this commit.
