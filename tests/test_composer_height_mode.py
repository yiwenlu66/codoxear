"""Behavioral coverage for the composer height modes (normal / tall / full).

Each scenario runs the shipped composer controller in a Node VM against
browser-node mocks and asserts the observable outcomes: the expand button's
visibility, ``composer.dataset.height``, inline textarea styles, the tall-height
custom property, and the localStorage writes.
"""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
from typing import Any


COMPOSER_SOURCE = module_path("app_composer.js").read_text(encoding="utf-8")
MODAL_SOURCE = module_path("app_modal.js").read_text(encoding="utf-8")
SESSION_HELPERS_SOURCE = module_path("app_session_helpers.js").read_text(encoding="utf-8")
SESSION_CATALOG_SOURCE = module_path("app_session_catalog.js").read_text(encoding="utf-8")
SESSION_STATE_SOURCE = module_path("app_session_state.js").read_text(encoding="utf-8")

PRELUDE = textwrap.dedent(
    f"""
    const vm = require("vm");
    const ctx = {{
      HTMLElement: function HTMLElement() {{}},
      Event: function Event(type, options) {{ this.type = type; this.options = options; }},
      Uint8Array,
      console,
      document: {{ activeElement: null, querySelector: () => null, documentElement: {{}}, body: {{}} }},
      window: {{ innerWidth: 1200, innerHeight: 800, addEventListener() {{}}, removeEventListener() {{}} }},
    }};
    vm.createContext(ctx);
    for (const [name, source] of [
      ["app_modal.js", {json.dumps(MODAL_SOURCE)}],
      ["app_session_helpers.js", {json.dumps(SESSION_HELPERS_SOURCE)}],
      ["app_session_state.js", {json.dumps(SESSION_STATE_SOURCE)}],
      ["app_session_catalog.js", {json.dumps(SESSION_CATALOG_SOURCE)}],
      ["app_composer.js", {json.dumps(COMPOSER_SOURCE)}],
    ]) vm.runInContext(source, ctx, {{ filename: name }});

    const noop = () => {{}};
    function node(extra = {{}}) {{
      const listeners = new Map();
      const classes = new Set();
      let html = "";
      return {{
        style: {{
          props: {{}},
          setProperty(name, value) {{ this.props[name] = value; }},
          removeProperty(name) {{ delete this.props[name]; }},
        }},
        dataset: {{}}, value: "", textContent: "", title: "", disabled: false,
        scrollHeight: 32, selectionStart: 0, selectionEnd: 0, children: [], attrs: {{}}, files: [],
        classList: {{
          toggle(name, enabled) {{ if (enabled) classes.add(name); else classes.delete(name); }},
          add(name) {{ classes.add(name); }}, remove(name) {{ classes.delete(name); }},
          contains(name) {{ return classes.has(name); }},
        }},
        addEventListener(type, handler) {{ listeners.set(type, handler); }},
        removeEventListener(type) {{ listeners.delete(type); }},
        emit(type, event = {{}}) {{ return listeners.get(type)?.(event); }},
        setAttribute(name, value) {{ this.attrs[name] = String(value); }},
        getAttribute(name) {{ return this.attrs[name]; }},
        removeAttribute(name) {{ delete this.attrs[name]; }},
        appendChild(child) {{ this.children.push(child); return child; }},
        focus() {{}}, blur() {{}},
        set innerHTML(value) {{ html = String(value); this.children = []; }}, get innerHTML() {{ return html; }},
        ...extra,
      }};
    }}

    // One composer wired to an in-memory storage. Returns the nodes and probes.
    function harness({{ stored = {{}}, running = false, textareaHeight = 100, appHeight = 1000, staged = [], sendOk = true, queueOk = true }} = {{}}) {{
      const storage = new Map(Object.entries(stored));
      const writes = [];
      const captures = [];
      const calls = [];
      const hookCalls = {{ autoGrow: 0 }};
      const form = node({{ requestSubmit: noop }});
      const textarea = node();
      const composer = node();
      const clearBtn = node();
      const draftUndoBtn = node();
      const composerStrip = node();
      const edits = [];
      textarea.focus = () => calls.push(["focus"]);
      textarea.setSelectionRange = (start, end) => {{ textarea.selectionStart = start; textarea.selectionEnd = end; }};
      const expandBtn = node({{
        setPointerCapture(id) {{ captures.push(["set", id]); }},
        releasePointerCapture(id) {{ captures.push(["release", id]); }},
      }});
      const sendChoiceNowBtn = node();
      const sendChoiceLaterBtn = node();
      const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
      sessionState.applyRuntime({{ selected: "sid", running }});
      const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
      sessionCatalog.set("latestSessions", [{{ session_id: "sid", launch_state: "ready" }}, {{ session_id: "other", launch_state: "ready" }}]);
      const controller = ctx.window.CodoxearComposer.createComposerController({{
        form, textarea, msgPh: node(), sendBtn: node(), sendChoice: node(), sendChoiceBackdrop: node(),
        sendChoiceNowBtn, sendChoiceLaterBtn, sendChoiceCancelBtn: node(),
        composer, expandBtn, clearBtn, draftUndoBtn, composerStrip,
        onDraftEdited: (text) => edits.push(text),
        measureTextareaHeight: () => textareaHeight,
        getAppHeight: () => appHeight,
        sessionCatalog, sessionLaunchFailed: () => false, sessionState,
        getStagedAttachments: () => staged, api: async () => ({{}}), setToast: noop,
        setPollFastUntilMs: noop, kickPoll: noop,
        sendText: async (text) => {{ calls.push(["send", text]); return sendOk; }},
        enqueueComposerText: async (text) => {{ calls.push(["queue", text]); return queueOk; }},
        prepareModalOpen: noop, afterModalVisibilityChanged: noop, restoreModalFocus: noop,
        storageGetItem: (key) => (storage.has(key) ? storage.get(key) : null),
        storageSetItem: (key, value) => {{ storage.set(key, value); writes.push(["set", key, value]); }},
        storageRemoveItem: (key) => {{ storage.delete(key); writes.push(["remove", key]); }},
        onAutoGrow: () => {{ hookCalls.autoGrow += 1; }},
        requestFrame: (callback) => callback(),
        getComputedStyle: () => ({{ minHeight: "32px", maxHeight: "180px" }}),
        activeElement: () => null, isHTMLElement: () => false,
      }});
      const snapshot = () => ({{
        mode: composer.dataset.height,
        display: expandBtn.style.display,
        pressed: expandBtn.attrs["aria-pressed"],
        label: expandBtn.attrs["aria-label"],
        expanded: expandBtn.classList.contains("expanded"),
        inlineHeight: textarea.style.height,
        inlineOverflow: textarea.style.overflowY,
        tallVar: composer.style.props["--composer-tall-h"] ?? null,
        stored: storage.get("codexweb.composerHeight.sid") ?? null,
      }});
      return {{ clearBtn, draftUndoBtn, composerStrip, edits, storage, writes, captures, calls, hookCalls, form, textarea, composer, expandBtn, sendChoiceNowBtn, sendChoiceLaterBtn, sessionState, controller, snapshot }};
    }}
    const pointer = (clientY, extra = {{}}) => ({{ pointerId: 7, pointerType: "touch", button: 0, clientY, preventDefault: noop, ...extra }});
    """
)


def run_scenario(body: str) -> Any:
    script = (
        "(async () => {\n"
        + PRELUDE
        + textwrap.dedent(body)
        + "\n})().catch((error) => { console.error(error); process.exit(1); });\n"
    )
    completed = subprocess.run(["node", "-"], check=True, input=script, capture_output=True, text=True)
    return json.loads(completed.stdout)


def test_expand_button_is_hidden_until_the_draft_overflows_the_normal_cap() -> None:
    result = run_scenario(
        """
        const h = harness();
        const initial = h.snapshot();
        h.textarea.value = "short";
        h.textarea.emit("input");
        const short = h.snapshot();
        h.textarea.value = "long\\n".repeat(40);
        h.textarea.scrollHeight = 400;
        h.textarea.emit("input");
        const overflowing = h.snapshot();
        h.textarea.value = "short";
        h.textarea.scrollHeight = 32;
        h.textarea.emit("input");
        const shrunk = h.snapshot();
        process.stdout.write(JSON.stringify({ initial, short, overflowing, shrunk }));
        """
    )
    assert result["initial"]["display"] == "none"
    assert result["short"]["display"] == "none"
    assert result["overflowing"]["display"] == ""
    assert result["overflowing"]["inlineOverflow"] == "auto"
    assert result["overflowing"]["inlineHeight"] == "180px"
    assert result["shrunk"]["display"] == "none"


def test_click_cycles_normal_tall_full_normal_and_persists_each_step() -> None:
    result = run_scenario(
        """
        const h = harness();
        h.textarea.scrollHeight = 400;
        h.textarea.value = "long";
        h.textarea.emit("input");
        const normal = h.snapshot();
        const hooksBefore = h.hookCalls.autoGrow;
        h.expandBtn.emit("click");
        const tall = h.snapshot();
        // Typing in a fixed mode must not reintroduce inline sizing.
        h.textarea.value = "longer";
        h.textarea.emit("input");
        const tallTyped = h.snapshot();
        h.expandBtn.emit("click");
        const full = h.snapshot();
        h.expandBtn.emit("click");
        const back = h.snapshot();
        process.stdout.write(JSON.stringify({ normal, tall, tallTyped, full, back, hooksFired: h.hookCalls.autoGrow > hooksBefore, writes: h.writes }));
        """
    )
    assert result["normal"]["mode"] == "normal" and result["normal"]["inlineHeight"] == "180px"
    assert result["normal"]["pressed"] == "false" and result["normal"]["expanded"] is False
    assert result["normal"]["label"] == "Expand draft"

    tall = result["tall"]
    assert tall["mode"] == "tall" and tall["inlineHeight"] == "" and tall["inlineOverflow"] == ""
    assert tall["stored"] == '{"mode":"tall"}' and tall["tallVar"] is None
    assert tall["pressed"] == "true" and tall["expanded"] is True
    assert tall["label"] == "Expand draft fully" and tall["display"] == ""
    assert result["tallTyped"]["inlineHeight"] == "" and result["tallTyped"]["inlineOverflow"] == ""

    full = result["full"]
    assert full["mode"] == "full" and full["stored"] == '{"mode":"full"}'
    assert full["label"] == "Restore draft height" and full["expanded"] is True
    assert full["inlineHeight"] == "" and full["inlineOverflow"] == ""

    back = result["back"]
    assert back["mode"] == "normal" and back["stored"] is None
    assert back["expanded"] is False and back["pressed"] == "false"
    # Normal mode re-derives the auto-grow styles and keeps the button for the overflowing draft.
    assert back["inlineHeight"] == "180px" and back["display"] == ""
    assert result["hooksFired"] is True
    assert [w[0] for w in result["writes"] if w[1].endswith("composerHeight.sid")] == ["set", "set", "remove"]


def test_stored_mode_is_applied_at_construction_and_when_the_session_is_selected() -> None:
    result = run_scenario(
        """
        const h = harness({ stored: {
          "codexweb.composerHeight.sid": '{"mode":"tall","height":432}',
          "codexweb.composerHeight.other": '{"mode":"full"}',
        } });
        const atConstruction = h.snapshot();
        h.sessionState.applyRuntime({ selected: "other" });
        const other = h.snapshot();
        h.sessionState.applyRuntime({ selected: "sid" });
        const backToSid = h.snapshot();
        process.stdout.write(JSON.stringify({ atConstruction, other, backToSid, writes: h.writes }));
        """
    )
    assert result["atConstruction"]["mode"] == "tall" and result["atConstruction"]["tallVar"] == "432px"
    assert result["other"]["mode"] == "full" and result["other"]["expanded"] is True
    assert result["backToSid"]["mode"] == "tall" and result["backToSid"]["tallVar"] == "432px"
    # Loading is a read: it never rewrites storage.
    assert result["writes"] == []


def test_successful_send_and_enqueue_reset_height_mode_but_clear_composer_does_not() -> None:
    result = run_scenario(
        """
        const key = "codexweb.composerHeight.sid";
        const stored = { [key]: '{"mode":"full"}' };

        const direct = harness({ stored: { ...stored } });
        direct.textarea.value = "hello";
        await direct.form.onsubmit({ preventDefault: noop });
        const afterSend = direct.snapshot();

        const picker = harness({ stored: { ...stored } });
        picker.textarea.value = "/model x";
        picker.controller.clearComposer();
        const afterClear = picker.snapshot();

        const now = harness({ stored: { ...stored }, running: true });
        now.textarea.value = "steer";
        await now.form.onsubmit({ preventDefault: noop });
        const choiceOpen = now.snapshot();
        await now.sendChoiceNowBtn.onclick();
        const afterNow = now.snapshot();

        const later = harness({ stored: { ...stored }, running: true });
        later.textarea.value = "queue me";
        await later.form.onsubmit({ preventDefault: noop });
        await later.sendChoiceLaterBtn.onclick();
        const afterLater = later.snapshot();

        process.stdout.write(JSON.stringify({
          afterSend, sendCalls: direct.calls, afterClear, choiceOpen, afterNow, afterLater,
          laterCalls: later.calls, nowCalls: now.calls,
        }));
        """
    )
    assert result["sendCalls"] == [["send", "hello"]]
    assert result["afterSend"]["mode"] == "normal" and result["afterSend"]["stored"] is None
    assert result["afterClear"]["mode"] == "full" and result["afterClear"]["stored"] == '{"mode":"full"}'
    assert result["choiceOpen"]["mode"] == "full"
    assert result["nowCalls"] == [["send", "steer"]]
    assert result["afterNow"]["mode"] == "normal" and result["afterNow"]["stored"] is None
    assert result["laterCalls"] == [["queue", "queue me"]]
    assert result["afterLater"]["mode"] == "normal" and result["afterLater"]["stored"] is None


def test_drag_resizes_continuously_with_snap_gravity_and_suppresses_the_click() -> None:
    result = run_scenario(
        """
        // start height 100px, app height 1000px, normal cap 180px:
        // snap-to-normal below 234px, snap-to-full above 900px.
        const drag = (h, moves, finish = "pointerup") => {
          h.expandBtn.emit("pointerdown", pointer(500));
          const frames = [];
          for (const y of moves) { h.expandBtn.emit("pointermove", pointer(y)); frames.push(h.snapshot()); }
          h.expandBtn.emit(finish, pointer(moves[moves.length - 1]));
          return frames;
        };

        const tallRun = harness();
        const tallFrames = drag(tallRun, [495, 300]);
        const tallAfterDrag = tallRun.snapshot();
        tallRun.expandBtn.emit("click");               // the click that follows a real drag
        const tallAfterSuppressedClick = tallRun.snapshot();
        tallRun.expandBtn.emit("click");               // a genuine click cycles again
        const tallNextClick = tallRun.snapshot();

        const normalRun = harness({ stored: { "codexweb.composerHeight.sid": '{"mode":"tall","height":500}' } });
        const normalFrames = drag(normalRun, [450]);
        const normalAfter = normalRun.snapshot();

        const fullRun = harness();
        const fullFrames = drag(fullRun, [-450]);
        const fullAfter = fullRun.snapshot();

        const shortClick = harness();
        shortClick.expandBtn.emit("pointerdown", pointer(500));
        shortClick.expandBtn.emit("pointermove", pointer(494));
        shortClick.expandBtn.emit("pointerup", pointer(494));
        shortClick.expandBtn.emit("click");
        const shortClickAfter = shortClick.snapshot();

        const cancelRun = harness();
        drag(cancelRun, [300], "pointercancel");
        const cancelAfter = cancelRun.snapshot();
        cancelRun.expandBtn.emit("click");
        const cancelNextClick = cancelRun.snapshot();

        process.stdout.write(JSON.stringify({
          tallFrames, tallAfterDrag, tallAfterSuppressedClick, tallNextClick, captures: tallRun.captures,
          normalFrames, normalAfter, normalWrites: normalRun.writes,
          fullFrames, fullAfter,
          shortClickAfter, cancelAfter, cancelNextClick,
        }));
        """
    )
    # Below the 8px threshold nothing moves; past it the height follows the finger.
    assert result["tallFrames"][0]["mode"] == "normal"
    assert result["tallFrames"][1]["mode"] == "tall" and result["tallFrames"][1]["tallVar"] == "300px"
    assert result["tallFrames"][1]["display"] == ""
    assert result["tallAfterDrag"]["mode"] == "tall" and result["tallAfterDrag"]["tallVar"] == "300px"
    assert result["tallAfterDrag"]["stored"] == '{"mode":"tall","height":300}'
    assert result["captures"] == [["set", 7], ["release", 7]]
    assert result["tallAfterSuppressedClick"]["mode"] == "tall"
    assert result["tallNextClick"]["mode"] == "full"

    # Dragging down to 150px snaps to normal mid-gesture, keeps the handle visible, then hides it.
    assert result["normalFrames"][0]["mode"] == "normal" and result["normalFrames"][0]["display"] == ""
    assert result["normalAfter"]["mode"] == "normal" and result["normalAfter"]["stored"] is None
    assert result["normalAfter"]["display"] == "none"
    assert ["remove", "codexweb.composerHeight.sid"] in result["normalWrites"]

    # 100 + 950 clamps to the 1000px app height, which is above 0.9 * 1000.
    assert result["fullFrames"][0]["mode"] == "full"
    assert result["fullAfter"]["mode"] == "full" and result["fullAfter"]["stored"] == '{"mode":"full"}'

    # A wiggle under the threshold is still a click.
    assert result["shortClickAfter"]["mode"] == "tall"

    # A cancelled drag persists its final mode and does not swallow the next click.
    assert result["cancelAfter"]["stored"] == '{"mode":"tall","height":300}'
    assert result["cancelNextClick"]["mode"] == "full"


def test_drag_previews_do_not_pollute_the_committed_tall_height() -> None:
    result = run_scenario(
        """
        // Stored committed preference: tall at 500px. The textarea currently
        // renders at 500px (textareaHeight), app height 1000px.
        const h = harness({
          stored: { "codexweb.composerHeight.sid": '{"mode":"tall","height":500}' },
          textareaHeight: 500,
        });
        const before = h.snapshot();
        // Drag up: preview passes through tall heights (700px live var) without
        // committing storage or the preference.
        h.expandBtn.emit("pointerdown", pointer(500));
        h.expandBtn.emit("pointermove", pointer(300));
        const previewTall = h.snapshot();
        // Drag back down past the normal snap threshold and release: aborted.
        h.expandBtn.emit("pointermove", pointer(900));
        const previewNormal = h.snapshot();
        h.expandBtn.emit("pointerup", pointer(900));
        const afterAbort = h.snapshot();
        h.expandBtn.emit("click");  // post-drag click is suppressed
        h.expandBtn.emit("click");  // genuine click cycles normal -> tall
        const cycleTall = h.snapshot();
        process.stdout.write(JSON.stringify({ before, previewTall, previewNormal, afterAbort, cycleTall }));
        """
    )
    assert result["before"]["mode"] == "tall" and result["before"]["tallVar"] == "500px"
    # Preview renders the live height but writes nothing to storage.
    assert result["previewTall"]["mode"] == "tall" and result["previewTall"]["tallVar"] == "700px"
    assert result["previewTall"]["stored"] == '{"mode":"tall","height":500}'
    assert result["previewNormal"]["mode"] == "normal"
    # The aborted gesture ends in normal and clears the stored mode.
    assert result["afterAbort"]["mode"] == "normal" and result["afterAbort"]["stored"] is None
    # The next cycle's tall stop reuses the committed 500px preference, not a
    # preview value left over from the aborted drag.
    assert result["cycleTall"]["mode"] == "tall" and result["cycleTall"]["tallVar"] == "500px"
    assert result["cycleTall"]["stored"] == '{"mode":"tall","height":500}'


def test_invalid_stored_values_fall_back_to_normal() -> None:
    result = run_scenario(
        """
        const key = "codexweb.composerHeight.sid";
        const cases = {};
        for (const [name, raw] of Object.entries({
          garbage: "{not json",
          unknownMode: '{"mode":"huge"}',
          jsonNull: "null",
          jsonString: '"full"',
        })) {
          cases[name] = harness({ stored: { [key]: raw } }).snapshot();
        }
        const missing = harness().snapshot();
        const badHeight = harness({ stored: { [key]: '{"mode":"tall","height":"tall"}' } }).snapshot();
        process.stdout.write(JSON.stringify({ cases, missing, badHeight }));
        """
    )
    for name, snap in result["cases"].items():
        assert snap["mode"] == "normal", name
        assert snap["tallVar"] is None, name
    assert result["missing"]["mode"] == "normal"
    # A tall record without a usable height still restores tall at the CSS default.
    assert result["badHeight"]["mode"] == "tall" and result["badHeight"]["tallVar"] is None
