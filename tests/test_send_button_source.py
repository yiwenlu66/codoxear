import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


class TestSendButtonSource(unittest.TestCase):
    def test_send_button_reflects_session_and_sending_state(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('function syncSendButtonState() {', source)
        self.assertIn('const sendControl = $("#sendBtn");', source)
        self.assertIn('const unknownSend = selectedSessionHasUnknownSend();', source)
        self.assertIn('const orphanRecovery = selectedSessionIsOrphanRecovery();', source)
        self.assertIn('const recoveryQueue = selectedSessionHasOrphanQueueRecovery();', source)
        self.assertIn('const launchFailed = selectedSessionLaunchFailed();', source)
        self.assertIn('sendControl.disabled = !!sending || !selected || launchFailed || unknownSend || orphanRecovery || recoveryQueue;', source)
        self.assertIn('Failed launch cannot receive messages', source)
        self.assertIn('Resolve the unknown send before sending', source)
        self.assertIn('Missing session can only be reviewed', source)
        self.assertIn('Review preserved queued recovery items before sending', source)
        self.assertIn('sendControl.setAttribute("aria-label", sendLabel);', source)
        self.assertIn('syncSendButtonState();\n          diagBtn.disabled = !selected;', source)
        self.assertIn('sending = true;\n          syncSendButtonState();', source)
        self.assertIn('sending = false;\n            syncSendButtonState();', source)
        self.assertIn('setToast("select a session first");', source)

    def test_mobile_composer_stop_reuses_interrupt_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('id: "composerStopBtn"', source)
        self.assertIn('title: "Stop current response"', source)
        self.assertIn('"aria-label": "Stop current response"', source)
        self.assertIn('html: iconSvg("stop")', source)
        self.assertIn('const canInterrupt = Boolean(running && selected);', source)
        self.assertIn('composerStopControl.classList.toggle("is-visible", canInterrupt);', source)
        self.assertIn('composerStopControl.disabled = !canInterrupt;', source)
        self.assertIn('async function interruptSelectedSession()', source)
        self.assertIn('interruptBtn.onclick = (e) => {', source)
        self.assertIn('composerStopBtn.onclick = (e) => {', source)
        self.assertIn('void interruptSelectedSession();', source)
        stop_start = css.index('.composerStopBtn {')
        media_start = css.index('@media (max-width: 700px), (pointer: coarse)', stop_start)
        self.assertIn('display: none;', css[stop_start:media_start])
        self.assertIn('.composerStopBtn.is-visible', css)
        self.assertIn('@media (max-width: 700px), (pointer: coarse)', css)

    def test_busy_send_choice_is_keyboard_focus_owned(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "sendChoice", role: "dialog", "aria-modal": "true", "aria-label": "Send options"', source)
        self.assertIn("let sendChoiceReturnFocusEl = null;", source)
        self.assertIn("function focusSendChoiceInitial()", source)
        self.assertIn('const target = laterBtn && !laterBtn.disabled ? laterBtn : nowBtn && !nowBtn.disabled ? nowBtn : cancelBtn;', source)
        self.assertIn("function showSendChoice(raw, { opener = null } = {})", source)
        self.assertIn("sendChoiceReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;", source)
        self.assertIn("focusSendChoiceInitial();", source)
        self.assertIn("function hideSendChoice({ restoreFocus = false } = {})", source)
        self.assertIn("if (restoreFocus) restoreModalFocus(target, () => sendChoice.style.display === \"flex\");", source)
        self.assertIn("hideSendChoice({ restoreFocus: true });", source)
        self.assertIn("const ok = await sendText(raw, { sid });", source)
        self.assertIn("const ok = await enqueueComposerText(raw, { sid });", source)
        self.assertIn("showSendChoice(raw, { opener: document.activeElement instanceof HTMLElement ? document.activeElement : textarea });", source)


if __name__ == "__main__":
    unittest.main()
