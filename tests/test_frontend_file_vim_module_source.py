"""File viewer vim controller: behavioral VM suite.

Loads app_file_vim.js into a VM with a fake DOM + fake Monaco editor that
records trigger/executeEdits/updateOptions/setPosition calls, then drives the
registered capture keydown listener to pin the sub-mode state machine, motions,
scroll fallback, guards, and hint entry.
"""

from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_FILE_VIM_JS = module_path("app_file_vim.js")


def run_node(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


HARNESS = r'''
const vm = require("vm");
const fs = require("fs");
const source = fs.readFileSync(__FILE__, "utf-8");
const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(source, ctx, { filename: "app_file_vim.js" });

function makeEnv(overrides = {}) {
  const state = {
    viewerOpen: true,
    blockingModal: false,
    hints: false,
    touchSelect: false,
    editMode: false,
    dirty: false,
    editorKind: "file",
    insertWritable: true,
    shortcutBlocked: false,
  };
  Object.assign(state, overrides);
  const editor = {
    triggers: [],
    edits: [],
    optionUpdates: [],
    positions: [],
    reveals: [],
    readOnly: true,
    focused: 0,
    trigger(source, command, args) { this.triggers.push([command, args]); return true; },
    executeEdits(source, ops) { this.edits.push([source, ops]); return true; },
    updateOptions(options) { this.optionUpdates.push({ ...options }); if ("readOnly" in options) this.readOnly = options.readOnly; return true; },
    pushUndoStop() { this.triggers.push(["pushUndoStop", null]); return true; },
    setPosition(position) { this.positions.push(position); return true; },
    revealPositionInCenter(position) { this.reveals.push(position); return true; },
    getPosition() { return { lineNumber: 2, column: 3 }; },
    getModel() {
      return {
        getLineCount() { return 10; },
        getLineMaxColumn(line) { return line === 2 ? 14 : 20; },
      };
    },
    focus() { this.focused += 1; },
  };
  const chip = { hidden: true, textContent: "", style: {} };
  const markdownPreview = { scrollTop: 0, scrollHeight: 2400, clientHeight: 400, className: "fileMarkdownPreview" };
  const fileDiff = {
    scrollTop: 0,
    scrollHeight: 1200,
    clientHeight: 300,
    querySelector(selector) { return selector === ".fileMarkdownPreview" ? markdownPreview : null; },
  };
  const keydowns = [];
  const calls = { setFileEditMode: [], toasts: [], dirty: [], hintEnters: 0, readOnlySyncs: 0 };
  const controller = ctx.window.CodoxearFileVim.createFileVimController({
    addAppEvent: (target, type, handler, capture) => {
      if (type === "keydown") keydowns.push({ handler, capture: Boolean(capture), target });
      return handler;
    },
    document: { addEventListener() {} },
    fileVimModeChip: chip,
    fileDiff,
    isFileViewerOpen: () => state.viewerOpen,
    hasBlockingFileEditorModal: () => state.blockingModal,
    hintModeActive: () => state.hints,
    touchSelectActive: () => state.touchSelect,
    fileEditorShortcutBlocked: (target) => Boolean(state.shortcutBlocked || (target && target.isTextEntry)),
    currentFileEditMode: () => state.editMode,
    setFileEditMode: (mode) => { calls.setFileEditMode.push(mode); state.editMode = Boolean(mode); if (!mode) { chip.hidden = true; chip.textContent = ""; } },
    currentFileDirty: () => state.dirty,
    currentFileEditorKind: () => state.editorKind,
    activeFileEditor: () => (state.editorKind === "file" || state.editorKind === "diff") && state.editorKind !== "none" ? editor : null,
    focusActiveFileEditor: () => { editor.focused += 1; return editor; },
    activeFileEditorInsertWritable: () => state.insertWritable,
    syncFileEditorReadOnly: () => { calls.readOnlySyncs += 1; },
    getFileEditorText: () => "buffer-text",
    currentActiveFileText: () => "buffer-text",
    setFileDirty: (dirty) => { calls.dirty.push(dirty); state.dirty = dirty; },
    setToast: (message) => calls.toasts.push(message),
    enterHintMode: () => { calls.hintEnters += 1; return true; },
  });
  function press(key, { target = null, ctrl = false, shift = false, alt = false, meta = false } = {}) {
    const event = {
      key,
      target,
      ctrlKey: ctrl,
      shiftKey: shift,
      altKey: alt,
      metaKey: meta,
      defaultPrevented: false,
      stoppedImmediate: false,
      preventDefault() { this.defaultPrevented = true; },
      stopImmediatePropagation() { this.stoppedImmediate = true; },
      stopPropagation() {},
    };
    for (const entry of keydowns) entry.handler(event);
    return event;
  }
  return { state, editor, chip, markdownPreview, fileDiff, keydowns, calls, controller, press };
}
'''.replace("__FILE__", json.dumps(str(APP_FILE_VIM_JS)))


def run_vim(js: str) -> dict:
    return run_node(HARNESS + js)


class TestFileVimSubModes(unittest.TestCase):
    def test_edit_entry_defaults_to_insert_and_escape_cycles_sub_modes(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editMode = true;
env.controller.syncEditMode();
const chipAtEntry = { hidden: env.chip.hidden, text: env.chip.textContent };
const esc1 = env.press("Escape");
const chipNormal = { hidden: env.chip.hidden, text: env.chip.textContent, readOnlySyncs: env.calls.readOnlySyncs };
const escDirty = env.press("Escape");  // normal + clean buffer exits edit mode
const esc2Mode = env.calls.setFileEditMode.slice();
// Edit again: entry resets to insert.
env.state.editMode = true;
env.controller.syncEditMode();
const iKey = env.press("i");
const chipInsert = { hidden: env.chip.hidden, text: env.chip.textContent };
process.stdout.write(JSON.stringify({ chipAtEntry, esc1: { consumed: esc1.defaultPrevented, stopped: esc1.stoppedImmediate }, chipNormal, escDirty: { consumed: escDirty.defaultPrevented, exit: esc2Mode }, chipInsert }));
''')
        self.assertEqual(result["chipAtEntry"], {"hidden": False, "text": "INSERT"})
        self.assertTrue(result["esc1"]["consumed"])
        self.assertTrue(result["esc1"]["stopped"])
        self.assertEqual(result["chipNormal"], {"hidden": False, "text": "NORMAL", "readOnlySyncs": 1})
        # Clean buffer: Esc in normal exits edit mode (view mode).
        self.assertEqual(result["escDirty"]["exit"], [False])
        self.assertTrue(result["escDirty"]["consumed"])
        # Re-entering edit mode resets to insert; i in insert is not consumed.
        self.assertEqual(result["chipInsert"], {"hidden": False, "text": "INSERT"})
        # i in INSERT mode must fall through (not consumed) so Monaco types it.
        result2 = run_vim(r'''
const env = makeEnv();
env.state.editMode = true;
env.controller.syncEditMode();
const iKey = env.press("i");
process.stdout.write(JSON.stringify({ consumed: iKey.defaultPrevented, triggers: env.editor.triggers.length }));
''')
        self.assertFalse(result2["consumed"])
        self.assertEqual(result2["triggers"], 0)

    def test_escape_in_normal_with_dirty_buffer_is_noop_with_status_hint(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editMode = true;
env.controller.syncEditMode();
env.press("Escape");           // insert -> normal
env.state.dirty = true;
const esc = env.press("Escape");
process.stdout.write(JSON.stringify({ consumed: esc.defaultPrevented, exitCalls: env.calls.setFileEditMode, toasts: env.calls.toasts, chip: env.chip.textContent }));
''')
        self.assertTrue(result["consumed"])
        self.assertEqual(result["exitCalls"], [])
        self.assertEqual(result["toasts"], ["unsaved changes"])
        self.assertEqual(result["chip"], "NORMAL")

    def test_escape_in_view_mode_does_nothing(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
const esc = env.press("Escape");
process.stdout.write(JSON.stringify({ consumed: esc.defaultPrevented, exitCalls: env.calls.setFileEditMode }));
''')
        self.assertFalse(result["consumed"])
        self.assertEqual(result["exitCalls"], [])

    def test_insert_entries_and_verbs(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editMode = true;
env.controller.syncEditMode();
env.press("Escape");           // -> normal
const a = env.press("a");
const normalChip = env.chip.textContent;
const subModeAfterA = env.controller.currentSubMode();
const aTriggers = env.editor.triggers.slice();
// back to normal for A
env.press("Escape");
const capitalA = env.press("A", { shift: true });
const aTriggersAfter = env.editor.triggers.slice();
env.press("Escape");
const o = env.press("o");
const chipAfterO = env.chip.textContent;
const oEdits = env.editor.edits.slice();
env.press("Escape");
const capitalO = env.press("O", { shift: true });
const oEditsAfter = env.editor.edits.slice();
env.press("Escape");
const triggersBeforeX = env.editor.triggers.length;
const x = env.press("x");
const xTriggers = env.editor.triggers.slice(triggersBeforeX);
// Still in normal mode: verbs chain without leaving it (Esc here would exit
// edit mode entirely on a clean buffer).
const editsBeforeDd = env.editor.edits.length;
const dd1 = env.press("d");
const dd2 = env.press("d");
const ddEdits = env.editor.edits.slice(editsBeforeDd);
const triggersBeforeU = env.editor.triggers.length;
const u = env.press("u");
const undoTriggers = env.editor.triggers.slice(triggersBeforeU);
const triggersBeforeRedo = env.editor.triggers.length;
const redo = env.press("r", { ctrl: true });
const redoTriggers = env.editor.triggers.slice(triggersBeforeRedo);
process.stdout.write(JSON.stringify({
  a: { chip: normalChip, triggers: aTriggers, subMode: subModeAfterA },
  capitalA: { triggers: aTriggersAfter.slice(aTriggers.length), chipAtPress: "INSERT" },
  o: { edits: oEdits, chip: chipAfterO },
  capitalO: { edits: oEditsAfter.slice(oEdits.length) },
  x: { triggers: xTriggers },
  dd: { edits: ddEdits },
  u: { triggers: undoTriggers },
  redo: { triggers: redoTriggers },
  readOnly: env.editor.readOnly,
  optionUpdates: env.editor.optionUpdates,
  dirtyCalls: env.calls.dirty,
}));
''')
        # a: cursorRight then insert
        self.assertEqual(result["a"]["triggers"], [["cursorRight", None]])
        self.assertEqual(result["a"]["subMode"], "insert")
        # A: cursorEnd then insert
        self.assertEqual(result["capitalA"]["triggers"], [["cursorEnd", None]])
        # o: newline below, then insert
        self.assertEqual(result["o"]["edits"], [["file-vim", [{
            "range": {"startLineNumber": 2, "startColumn": 14, "endLineNumber": 2, "endColumn": 14},
            "text": "\n",
            "forceMoveMarkers": True,
        }]]])
        self.assertEqual(result["o"]["chip"], "INSERT")
        # O: newline above (insertLine = lineNumber - 1)
        self.assertEqual(result["capitalO"]["edits"], [["file-vim", [{
            "range": {"startLineNumber": 1, "startColumn": 1, "endLineNumber": 1, "endColumn": 1},
            "text": "\n",
            "forceMoveMarkers": True,
        }]]])
        # x: deleteRight
        self.assertEqual(result["x"]["triggers"], [["deleteRight", None]])
        # dd: delete full line 2 (position line 2, not last of 10)
        self.assertEqual(result["dd"]["edits"], [["file-vim", [{
            "range": {"startLineNumber": 2, "startColumn": 1, "endLineNumber": 3, "endColumn": 1},
            "text": "",
            "forceMoveMarkers": True,
        }]]])
        # u: undo, Ctrl-r: redo
        self.assertEqual(result["u"]["triggers"], [["undo", None]])
        self.assertEqual(result["redo"]["triggers"], [["redo", None]])
        # Every verb ends with readOnly restored to true (normal mode holds).
        self.assertTrue(result["readOnly"])
        # The writable lift is visible in optionUpdates (false then true per verb).
        self.assertIn({"readOnly": False}, result["optionUpdates"])
        self.assertIn({"readOnly": True}, result["optionUpdates"])
        # Dirty is recomputed after verbs (equal buffers -> false).
        self.assertTrue(all(call is False for call in result["dirtyCalls"]))
        self.assertTrue(len(result["dirtyCalls"]) >= 5)

    def test_normal_mode_printable_keys_never_reach_the_editor(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editMode = true;
env.controller.syncEditMode();
env.press("Escape");
const z = env.press("z");
const q = env.press("q");
process.stdout.write(JSON.stringify({ z: { consumed: z.defaultPrevented, stopped: z.stoppedImmediate }, q: q.defaultPrevented, triggers: env.editor.triggers.length, edits: env.editor.edits.length }));
''')
        self.assertTrue(result["z"]["consumed"])
        self.assertTrue(result["z"]["stopped"])
        self.assertTrue(result["q"])
        self.assertEqual(result["triggers"], 0)
        self.assertEqual(result["edits"], 0)


class TestFileVimMotions(unittest.TestCase):
    def test_view_mode_motions_dispatch_monaco_cursor_commands(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
const pressed = {};
for (const key of ["h", "l", "j", "k", "w", "b", "e", "0", "$"]) {
  const before = env.editor.triggers.length;
  const event = env.press(key);
  pressed[key] = { consumed: event.defaultPrevented, command: env.editor.triggers.slice(before) };
}
const capitalG = env.press("G", { shift: true });
const ctrlD = env.press("d", { ctrl: true });
const ctrlU = env.press("u", { ctrl: true });
process.stdout.write(JSON.stringify({
  pressed,
  capitalG: { consumed: capitalG.defaultPrevented, position: env.editor.positions, reveal: env.editor.reveals, focused: env.editor.focused > 0 },
  ctrlD: env.editor.triggers.slice(-2),
}));
''')
        expected_commands = {
            "h": "cursorLeft", "l": "cursorRight", "j": "cursorDown", "k": "cursorUp",
            "w": "cursorWordStartRight", "b": "cursorWordStartLeft", "e": "cursorWordEndRight",
            "0": "cursorHome", "$": "cursorEnd",
        }
        for key, command in expected_commands.items():
            self.assertTrue(result["pressed"][key]["consumed"], key)
            self.assertEqual(result["pressed"][key]["command"], [[command, None]], key)
        # G goes to the last line (model has 10 lines) and reveals it.
        self.assertTrue(result["capitalG"]["consumed"])
        self.assertEqual(result["capitalG"]["position"], [{"lineNumber": 10, "column": 1}])
        self.assertEqual(result["capitalG"]["reveal"], [{"lineNumber": 10, "column": 1}])
        self.assertTrue(result["capitalG"]["focused"])
        # Ctrl-d then Ctrl-u each move half a page with the cursor.
        self.assertEqual(result["ctrlD"], [["cursorMove", {"to": "down", "by": "halfPage", "value": 1, "select": False}], ["cursorMove", {"to": "up", "by": "halfPage", "value": 1, "select": False}]])

    def test_gg_sequence_and_prefix_reset(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
const g1 = env.press("g");
const jAfterG = env.press("j");
const triggersAfterReset = env.editor.triggers.length;
const g2 = env.press("g");
const g3 = env.press("g");
const positions = env.editor.positions.slice();
process.stdout.write(JSON.stringify({
  g1: g1.defaultPrevented,
  jAfterG: { consumed: jAfterG.defaultPrevented, moved: triggersAfterReset },
  gg: { consumed: g3.defaultPrevented, positions },
}));
''')
        self.assertTrue(result["g1"])
        # g then j: the pending prefix is cleared and the j is dropped, not run.
        self.assertTrue(result["jAfterG"]["consumed"])
        self.assertEqual(result["jAfterG"]["moved"], 0)
        # A fresh gg moves to line 1.
        self.assertTrue(result["gg"]["consumed"])
        self.assertEqual(result["gg"]["positions"], [{"lineNumber": 1, "column": 1}])

    def test_motions_work_in_edit_normal_mode_and_letters_are_swallowed_in_view_mode(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editMode = true;
env.controller.syncEditMode();
env.press("Escape");
const before = env.editor.triggers.length;
const j = env.press("j");
const normalJ = env.editor.triggers.slice(before);
env.state.editMode = false;
env.controller.syncEditMode();
const afterNormal = env.editor.triggers.length;
const q = env.press("q");
const triggersAfterQ = env.editor.triggers.length - afterNormal;
const e2 = env.press("e");
process.stdout.write(JSON.stringify({
  normalJ,
  viewQ: { consumed: q.defaultPrevented, triggers: triggersAfterQ },
  viewE: e2.defaultPrevented,
}));
''')
        self.assertEqual(result["normalJ"], [["cursorDown", None]])
        self.assertTrue(result["viewQ"]["consumed"])
        self.assertEqual(result["viewQ"]["triggers"], 0)
        self.assertTrue(result["viewE"])

    def test_guards_defer_to_other_key_owners(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
const cases = [];
// Nested blocking dialog open.
env.state.blockingModal = true;
cases.push(["blocking", env.press("j").defaultPrevented]);
env.state.blockingModal = false;
// Hint mode active (its bubble listener owns the keys).
env.state.hints = true;
cases.push(["hints", env.press("j").defaultPrevented]);
env.state.hints = false;
// Touch-selection mode active.
env.state.touchSelect = true;
cases.push(["touch", env.press("j").defaultPrevented]);
env.state.touchSelect = false;
// Text-entry target (picker input) -> pass through untouched.
const input = { isTextEntry: true, closest: (selector) => (selector === "#fileViewer" ? {} : null) };
cases.push(["textEntryPassThrough", env.press("j", { target: input }).defaultPrevented]);
// Shortcut blocked (viewer closed).
env.state.shortcutBlocked = true;
cases.push(["blocked", env.press("j").defaultPrevented]);
env.state.shortcutBlocked = false;
// Viewer closed entirely.
env.state.viewerOpen = false;
cases.push(["closed", env.press("j").defaultPrevented]);
process.stdout.write(JSON.stringify({ cases, triggers: env.editor.triggers.length }));
''')
        guards = dict(result["cases"])
        for case, consumed in guards.items():
            self.assertFalse(consumed, case)
        self.assertEqual(result["triggers"], 0)


class TestFileVimScrollFallback(unittest.TestCase):
    def test_scroll_surface_receives_j_k_halfpage_and_edges(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editorKind = "none";   // markdown preview / PDF / fallback surface
const j = env.press("j");
const k = env.press("k");
const afterScroll = env.markdownPreview.scrollTop;
const ctrlD = env.press("d", { ctrl: true });
const ctrlU = env.press("u", { ctrl: true });
const afterPages = env.markdownPreview.scrollTop;
const G = env.press("G", { shift: true });
const topAfterG = env.markdownPreview.scrollTop;
const g1 = env.press("g");
const g2 = env.press("g");
const topAfterGg = env.markdownPreview.scrollTop;
const w = env.press("w");
const zero = env.press("0");
process.stdout.write(JSON.stringify({
  j: { consumed: j.defaultPrevented, afterScroll },
  k: { consumed: k.defaultPrevented },
  ctrlD: ctrlD.defaultPrevented,
  ctrlU: ctrlU.defaultPrevented,
  afterPages,
  G: { consumed: G.defaultPrevented, top: topAfterG },
  gg: { consumed: g2.defaultPrevented, top: topAfterGg },
  wordKeys: { w: w.defaultPrevented, zero: zero.defaultPrevented },
  hostScrollTop: env.fileDiff.scrollTop,
}));
''')
        # j/k scroll the markdown preview by 24px each: 24 - 0 after j, back to 0 after k.
        self.assertTrue(result["j"]["consumed"])
        self.assertEqual(result["j"]["afterScroll"], 0)
        self.assertTrue(result["k"]["consumed"])
        # Ctrl-d then Ctrl-u: +200 then -200 (clientHeight 400 / 2).
        self.assertTrue(result["ctrlD"])
        self.assertTrue(result["ctrlU"])
        self.assertEqual(result["afterPages"], 0)
        # G jumps to the bottom (scrollHeight), gg back to the top.
        self.assertTrue(result["G"]["consumed"])
        self.assertEqual(result["G"]["top"], 2400)
        self.assertTrue(result["gg"]["consumed"])
        self.assertEqual(result["gg"]["top"], 0)
        # Word and line-edge motions are consumed no-ops on scroll surfaces.
        self.assertTrue(result["wordKeys"]["w"])
        self.assertTrue(result["wordKeys"]["zero"])
        self.assertEqual(result["hostScrollTop"], 0)

    def test_plain_host_is_the_scroll_surface_without_markdown_preview(self) -> None:
        result = run_vim(r'''
const env = makeEnv();
env.state.editorKind = "none";
env.fileDiff.querySelector = () => null;   // no preview element: host scrolls
const j = env.press("j");
const G = env.press("G", { shift: true });
process.stdout.write(JSON.stringify({ consumedJ: j.defaultPrevented, hostTop: env.fileDiff.scrollTop }));
''')
        self.assertTrue(result["consumedJ"])
        self.assertEqual(result["hostTop"], 1200)


if __name__ == "__main__":
    unittest.main()
