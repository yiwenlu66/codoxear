"""The composer's own writers reconcile startup with ordinary sendability."""
from test_composer_height_mode import run_scenario


def test_quick_start_enables_typing_but_refuses_send_until_selected():
    result = run_scenario('''
      let starting = false, listener;
      const quickSession = {isStarting: () => starting, subscribe(fn) {listener = fn; return () => {};}};
      const h = harness({quickSession});
      h.sessionState.set("selected", null);
      const idle = [h.textarea.disabled, h.sendBtn.disabled];
      starting = true; listener();
      const boot = [h.textarea.disabled, h.sendBtn.disabled, h.sendBtn.title];
      h.textarea.value = "keep typed text";
      const refused = await h.controller.sendText(h.textarea.value);
      await h.form.onsubmit({preventDefault() {}});
      const during = h.calls.slice();
      h.sessionState.set("selected", "new"); starting = false; listener();
      const ready = [h.textarea.disabled, h.sendBtn.disabled];
      const sent = await h.controller.sendText(h.textarea.value);
      process.stdout.write(JSON.stringify({idle, boot, refused, during, ready, sent, calls: h.calls, text: h.textarea.value}));
    ''')
    assert result == {
        "idle": [True, True], "boot": [False, True, "Session starting…"],
        "refused": False, "during": [], "ready": [False, False], "sent": True,
        "calls": [["send", "keep typed text"]], "text": "keep typed text",
    }
