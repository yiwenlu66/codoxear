"""Execute the path runtime and composer's DOM interactions in a Node VM."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from frontend_module_loader import module_path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [module_path(name).read_text(encoding="utf-8") for name in (
    "app_file_helpers.js", "app_path_completion.js", "app_session_state.js",
    "app_session_catalog.js", "app_composer.js",
)]

HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const timers = new Map();
let nextTimer = 0;
const schedule = (callback, delay) => { timers.set(++nextTimer, { callback, delay }); return nextTimer; };
const cancel = (id) => timers.delete(id);
const ctx = { window: {}, console, AbortController, setTimeout: schedule, clearTimeout: cancel };
vm.createContext(ctx);
for (const source of SOURCES) vm.runInContext(source, ctx);
const plain = (value) => JSON.parse(JSON.stringify(value));
async function flush() {
  const scheduled = [...timers.values()]; timers.clear();
  for (const { callback, delay } of scheduled) { assert.equal(delay, 120); await callback(); }
}
function node() {
  const listeners = new Map();
  const attributes = new Map();
  const classes = new Set();
  let text = '';
  return {
    style: {}, value: '', title: '', disabled: false, scrollHeight: 32,
    selectionStart: 0, selectionEnd: 0, children: [], className: '',
    classList: { toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
    addEventListener(type, callback) { listeners.set(type, callback); },
    removeEventListener(type) { listeners.delete(type); },
    emit(type, event = {}) { return listeners.get(type)?.(event); },
    listenerCount() { return listeners.size; },
    setAttribute(key, value) { attributes.set(key, String(value)); },
    getAttribute(key) { return attributes.get(key); },
    removeAttribute(key) { attributes.delete(key); },
    set innerHTML(value) { text = ''; this.children = []; },
    get textContent() { return text + this.children.map((child) => child.textContent).join(''); },
    set textContent(value) { text = value; this.children = []; },
    appendChild(child) { this.children.push(child); return child; },
    setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; },
    focus() { ctx.document.activeElement = this; },
    blur() { ctx.document.activeElement = null; },
    scrollIntoView() {},
  };
}
ctx.document = {
  activeElement: null,
  createElement: () => node(),
  createTextNode: (text) => ({ textContent: text }),
};
const noop = () => {};
const listing = (entries, dir = '/work') => ({ ok: true, entries, dir, truncated: false });
const directory = { name: 'src space', completion: './src space/', is_dir: true };
const file = { name: 'read me.md', completion: './src space/read me.md', is_dir: false };
function harness(respond = () => listing([directory])) {
  const textarea = node(), modelPicker = node(), form = node();
  const storage = new Map(), edited = [], requests = [], sent = [];
  const sessionState = ctx.window.CodoxearSessionState.createSessionState({ consoleError: noop });
  sessionState.set('selected', 'sid');
  const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({ consoleError: noop });
  sessionCatalog.set('latestSessions', [{ session_id: 'sid', agent_backend: 'pi', slash_commands: [{ name: 'help', description: 'Help' }] }]);
  const controller = ctx.window.CodoxearComposer.createComposerController({
    form, textarea, modelPicker, msgPh: node(), sendBtn: node(), sendChoice: node(),
    sendChoiceBackdrop: node(), sendChoiceNowBtn: node(), sendChoiceLaterBtn: node(), sendChoiceCancelBtn: node(),
    sessionState, sessionCatalog, sessionLaunchFailed: () => false, getStagedAttachments: () => [],
    api: async (url, options) => { requests.push({ url, options }); return respond(url); },
    setToast: noop, setPollFastUntilMs: noop, kickPoll: noop,
    sendText: async (text) => { sent.push(text); return true; }, enqueueComposerText: async () => true,
    prepareModalOpen: noop, afterModalVisibilityChanged: noop, restoreModalFocus: noop,
    storageGetItem: (key) => storage.get(key), storageSetItem: (key, value) => storage.set(key, value),
    storageRemoveItem: (key) => storage.delete(key), onDraftEdited: (value) => edited.push(value),
    getComputedStyle: () => ({ minHeight: '32px' }), requestFrame: (callback) => callback(),
  });
  const type = async (value, caret = value.length) => {
    textarea.value = value; textarea.setSelectionRange(caret, caret); textarea.focus();
    textarea.emit('input'); await flush();
  };
  const key = (key, extra = {}) => {
    const event = { key, prevented: false, preventDefault() { this.prevented = true; }, stopPropagation() {}, ...extra };
    textarea.emit('keydown', event); textarea.emit('keyup', event); return event;
  };
  const options = () => modelPicker.children.filter((child) => child.getAttribute('role') === 'option');
  const open = () => modelPicker.style.display === 'block';
  return { textarea, modelPicker, controller, sessionState, storage, edited, requests, sent, type, key, options, open };
}
"""


def run_js(body: str) -> None:
    script = "const SOURCES = " + json.dumps(SOURCES) + ";\n" + HARNESS
    script += "\n(async () => {\n" + body + "\n})().catch((error) => { console.error(error); process.exit(1); });"
    # The complete VM program can exceed Linux's per-argument limit even when
    # total argv fits ARG_MAX. Keep every source and assertion; stream via stdin.
    result = subprocess.run(["node", "-"], input=script, cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("value", "caret", "expected"), [
    ("./", 2, {"start": 0, "end": 2, "token": "./"}),
    ("/", 1, None),
    ("/tmp", 4, None),
    ("read /tmp", 9, {"start": 5, "end": 9, "token": "/tmp"}),
    ("https://x", 9, None),
    ("a/b", 3, None),
    ("see ./file.txt end", 8, {"start": 4, "end": 14, "token": "./file.txt"}),
    ("../x", 4, {"start": 0, "end": 4, "token": "../x"}),
    ("~/x", 3, {"start": 0, "end": 3, "token": "~/x"}),
    ("a\t./b\nc", 5, {"start": 2, "end": 5, "token": "./b"}),
    ("a ./b ", 6, None),
    ("a\n/tmp", 6, {"start": 2, "end": 6, "token": "/tmp"}),
    # A mention anchors at its own `@`, including in CJK prose with no spaces.
    ("@", 1, {"start": 0, "end": 1, "token": "@"}),
    ("@", 0, {"start": 0, "end": 1, "token": "@"}),
    ("see @sr end", 7, {"start": 4, "end": 7, "token": "@sr"}),
    ("看看@sr", 5, {"start": 2, "end": 5, "token": "@sr"}),
    ("看看@sr 后", 5, {"start": 2, "end": 5, "token": "@sr"}),
    ("@~/x", 5, {"start": 0, "end": 5, "token": "@~/x"}),
    ("@src/x", 6, {"start": 0, "end": 6, "token": "@src/x"}),
    # An email local part before the `@` keeps the token inert.
    ("foo@bar", 7, None),
    ("a@b/c.txt", 9, None),
])
def test_path_token_trigger_matrix(value: str, caret: int, expected: dict | None) -> None:
    run_js(f"assert.deepEqual(plain(ctx.window.CodoxearPathCompletion.pathTokenAt({json.dumps(value)}, {caret})), {json.dumps(expected)});")


def test_runtime_anchors_spaces_debounces_aborts_and_drops_stale_responses() -> None:
    run_js(r"""
    const calls = [], changes = [];
    const runtime = ctx.window.CodoxearPathCompletion.createPathCompletionRuntime({
      api: (url, options) => new Promise((resolve) => calls.push({ url, options, resolve })),
      onChange: (state) => changes.push(plain(state)),
    });
    runtime.sync('read ./s', 8, 'sid');
    runtime.sync('read ./sr', 9, 'sid');
    assert.equal(timers.size, 1);
    assert.equal(runtime.snapshot().loading, true);
    let task = [...timers.values()][0].callback(); timers.clear();
    assert.match(calls[0].url, /path=.%2Fsr&limit=50$/);
    runtime.sync('read ./src space/', 17, 'sid');
    assert.equal(calls[0].options.signal.aborted, true);
    calls[0].resolve(listing([{ name: 'stale' }])); await task;
    assert.equal(changes.length, 0);
    task = [...timers.values()][0].callback(); timers.clear();
    calls[1].resolve({ ...listing([file], '/work/src space'), truncated: true }); await task;
    assert.equal(runtime.snapshot().entries[0].name, 'read me.md');
    assert.equal(runtime.snapshot().truncated, true);
    assert.equal(runtime.snapshot().start, 5);
    assert.equal(runtime.snapshot().token, './src space/');
    runtime.sync('read ./src space/', 17, 'sid');
    assert.equal(timers.size, 0);
    assert.equal(runtime.sync('read ./src space/', 17, 'other'), null);
    assert.equal(runtime.snapshot().open, false);
    runtime.sync('./x', 3, 'sid');
    assert.equal(runtime.sync('./x\nnext', 8, 'sid'), null);
    runtime.sync('read ./x', 8, 'sid');
    assert.equal(runtime.sync('read ./x', 4, 'sid'), null);
    runtime.sync('./x', 3, 'sid'); runtime.close();
    assert.equal(timers.size, 0);
    """)


def test_mention_completion_keeps_the_at_prefix_and_drills(tmp_path: Path) -> None:
    run_js(r"""
    const mentionDir = { name: 'src space', completion: 'src space/', is_dir: true };
    const mentionFile = { name: 'read me.md', completion: 'src space/read me.md', is_dir: false };
    const h = harness((url) => url.includes('src%20space%2F') ? listing([mentionFile], '/work/src space') : listing([mentionDir], '/work'));
    await h.type('@');
    assert.equal(h.open(), true);
    assert.match(h.requests[0].url, /file\/complete\?path=&limit=50$/);
    assert.equal(h.modelPicker.children[0].textContent, '/work');
    assert.equal(h.options()[0].textContent, 'src space/');

    await h.type('@sr');
    assert.match(h.requests[1].url, /file\/complete\?path=sr&limit=50$/);
    const marks = h.options()[0].children.filter((child) => child.className === 'fileMenuMatch');
    assert.equal(marks.map((child) => child.textContent).join(''), 'sr');

    h.key('Enter');
    assert.equal(h.textarea.value, '@src space/');
    await flush();
    assert.match(h.requests[2].url, /path=src%20space%2F&limit=50$/);
    assert.equal(h.options()[0].textContent, 'read me.md');

    h.key('Enter');
    assert.equal(h.textarea.value, '@src space/read me.md');
    assert.equal(h.textarea.selectionStart, 21);
    assert.equal(h.open(), false);
    assert.equal(h.storage.get('codexweb.draft.sid'), h.textarea.value);
    assert.equal(h.edited.at(-1), h.textarea.value);
    h.controller.dispose();
    """)


def test_mention_in_cjk_prose_and_email_nontrigger() -> None:
    run_js(r"""
    const h = harness((url) => listing([{ name: 'README.md', completion: 'README.md', is_dir: false }]));
    await h.type('看看@re');
    assert.equal(h.open(), true);
    assert.match(h.requests[0].url, /path=re&limit=50$/);
    h.key('Enter');
    assert.equal(h.textarea.value, '看看@README.md');
    assert.equal(h.open(), false);

    await h.type('foo@bar');
    assert.equal(h.open(), false);
    assert.equal(h.requests.length, 1);
    await h.type('mail@bar.com');
    assert.equal(h.open(), false);
    assert.equal(h.requests.length, 1);
    h.controller.dispose();
    """)


def test_no_match_mention_does_not_swallow_enter() -> None:
    run_js(r"""
    const h = harness(() => listing([]));
    await h.type('mail @zzz');
    assert.equal(h.open(), true);
    assert.equal(h.modelPicker.children.map((child) => child.textContent).join('|'), '/work|No matches');
    assert.equal(h.key('Enter').prevented, false);
    assert.equal(h.key('Tab').prevented, false);
    assert.equal(h.textarea.value, 'mail @zzz');
    h.key('Escape');
    assert.equal(h.open(), false);
    h.controller.dispose();
    """)


def test_typed_whitespace_ends_anchored_path_session() -> None:
    run_js(r"""
    const h = harness(() => listing([{ name: 'src', completion: './src/', is_dir: true }]));
    await h.type('./sr');
    assert.equal(h.open(), true);
    await h.type('./sr ');
    assert.equal(h.open(), false);
    assert.equal(h.requests.length, 1);
    await h.type('./sr more words');
    assert.equal(h.open(), false);
    assert.equal(h.requests.length, 1);
    await h.type('./sr more words ./re');
    assert.equal(h.open(), true);
    assert.match(h.requests[1].url, /path=.%2Fre&limit=50$/);
    h.controller.dispose();
    """)


def test_directory_then_file_acceptance_preserves_focus_saves_and_highlights() -> None:
    run_js(r"""
    const h = harness((url) => listing(url.includes('space%2F') ? [file] : [directory], '/work'));
    await h.type('read ./sr suffix', 9);
    assert.equal(h.open(), true);
    assert.equal(h.modelPicker.children[0].textContent, '/work');
    const option = h.options()[0];
    assert.equal(option.textContent, 'src space/');
    assert.equal(option.id, 'path-picker-option-0');
    assert.equal(h.textarea.getAttribute('aria-activedescendant'), option.id);
    assert.equal(option.children.filter((child) => child.className === 'fileMenuMatch').map((child) => child.textContent).join(''), 'sr');
    const pointer = { prevented: false, preventDefault() { this.prevented = true; } };
    option.onpointerdown(pointer); assert.equal(pointer.prevented, true);
    option.onclick();
    assert.equal(h.textarea.value, 'read ./src space/ suffix');
    assert.equal(h.textarea.selectionStart, 17);
    assert.equal(h.textarea.selectionEnd, 17);
    assert.equal(h.open(), true);
    assert.equal(ctx.document.activeElement, h.textarea);
    await flush();
    assert.match(h.requests[1].url, /path=.%2Fsrc%20space%2F&limit=50$/);
    h.key('Enter');
    assert.equal(h.textarea.value, 'read ./src space/read me.md suffix');
    assert.equal(h.textarea.selectionStart, 27);
    assert.equal(h.open(), false);
    assert.equal(h.storage.get('codexweb.draft.sid'), h.textarea.value);
    assert.equal(h.edited.at(-1), h.textarea.value);
    assert.equal(ctx.document.activeElement, h.textarea);
    assert.equal(h.textarea.getAttribute('role'), undefined);
    assert.deepEqual(h.sent, []);
    h.controller.dispose();
    assert.equal(h.textarea.listenerCount(), 0);
    """)


def test_tab_escape_caret_movement_and_programmatic_draft_dismissal() -> None:
    run_js(r"""
    const h = harness(() => listing([{ name: 'file', completion: './file', is_dir: false }]));
    await h.type('./fi');
    assert.equal(h.key('Tab', { isComposing: true }).prevented, false);
    assert.equal(h.open(), true);
    assert.equal(h.key('Tab').prevented, true);
    assert.equal(h.textarea.value, './file');
    assert.equal(h.open(), false);
    await h.type('read ./fi');
    h.key('Escape'); assert.equal(h.open(), false);
    await h.type('read ./fi');
    h.textarea.setSelectionRange(3, 3); h.textarea.emit('keyup', { key: 'ArrowLeft' });
    assert.equal(h.open(), false);
    await h.type('read ./fi');
    h.textarea.setSelectionRange(3, 3); h.textarea.emit('click');
    assert.equal(h.open(), false);
    await h.type('./fi');
    h.storage.set('codexweb.draft.other', './other'); h.controller.loadSessionDraft('other');
    assert.equal(h.textarea.value, './other'); assert.equal(h.open(), false);
    await h.type('./fi'); h.controller.setDraftFromServer('sid', 'server draft');
    assert.equal(h.open(), false);
    await h.type('./fi'); h.controller.clearComposer({ blur: false });
    assert.equal(h.open(), false);
    await h.type('./fi'); h.sessionState.set('selected', 'other');
    assert.equal(h.open(), false);
    h.controller.dispose();
    """)


def test_empty_failure_and_slash_command_exclusion() -> None:
    run_js(r"""
    const h = harness(() => { throw new Error('network failure'); });
    await h.type('./missing');
    assert.equal(h.open(), true);
    assert.equal(h.modelPicker.children[0].textContent, 'No matches');
    assert.equal(h.options().length, 0);
    h.key('ArrowDown'); h.key('Enter'); h.key('Tab');
    assert.equal(h.textarea.value, './missing');
    await h.type('/help');
    assert.equal(h.requests.length, 1);
    assert.equal(h.options()[0].id, 'command-picker-option-0');
    assert.equal(h.key('Tab').prevented, false);
    await h.type('https://x'); assert.equal(h.open(), false);
    await h.type('a/b'); assert.equal(h.open(), false);
    h.sessionState.set('selected', null);
    await h.type('./fi'); assert.equal(h.open(), false);
    h.controller.dispose();
    """)


def test_keyboard_navigation_skips_directory_header_and_replaces_whole_token() -> None:
    run_js(r"""
    const h = harness(() => listing([
      { name: 'one', completion: './one', is_dir: false },
      { name: 'two', completion: './two', is_dir: false },
    ]));
    await h.type('see ./original end', 8);
    h.key('ArrowDown');
    assert.equal(h.textarea.getAttribute('aria-activedescendant'), 'path-picker-option-1');
    assert.equal(h.modelPicker.children[0].getAttribute('aria-selected'), undefined);
    h.key('Enter', { isComposing: true }); assert.equal(h.open(), true);
    h.key('Enter');
    assert.equal(h.textarea.value, 'see ./two end');
    assert.equal(h.textarea.selectionStart, 9);
    assert.equal(h.open(), false);
    h.controller.dispose();
    """)
