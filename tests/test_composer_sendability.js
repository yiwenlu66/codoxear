"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const source = (name) => fs.readFileSync(path.join(root, "codoxear", "static", name), "utf8");

function node(extra = {}) {
  const listeners = new Map();
  return {
    style: {},
    value: "",
    textContent: "",
    title: "",
    disabled: false,
    scrollHeight: 32,
    children: [],
    classList: { toggle() {} },
    attributes: new Map(),
    addEventListener(type, handler) { listeners.set(type, handler); },
    removeEventListener(type) { listeners.delete(type); },
    emit(type, event = {}) { listeners.get(type)?.(event); },
    setAttribute(name, value) { this.attributes.set(name, String(value)); },
    getAttribute(name) { return this.attributes.get(name); },
    removeAttribute(name) { this.attributes.delete(name); },
    appendChild(child) { this.children.push(child); return child; },
    focus() {},
    blur() {},
    ...extra,
  };
}

function noop() {}

function composerHarness() {
  const ctx = { window: {}, console };
  vm.createContext(ctx);
  vm.runInContext(source("app_composer.js"), ctx);

  const form = node({ requestSubmit: noop });
  const textarea = node();
  const sendBtn = node();
  const selected = "session-1";
  ctx.window.CodoxearComposer.createComposerController({
    form,
    textarea,
    msgPh: node(),
    sendBtn,
    sendChoice: node(),
    sendChoiceBackdrop: node(),
    sendChoiceNowBtn: node(),
    sendChoiceLaterBtn: node(),
    sendChoiceCancelBtn: node(),
    getSelected: () => selected,
    getSessionInfo: () => ({ session_id: selected, launch_state: "ready" }),
    sessionLaunchFailed: () => false,
    getSending: () => false,
    getCurrentRunning: () => false,
    getStagedAttachments: () => [],
    api: async () => ({}),
    setToast: noop,
    setPollFastUntilMs: noop,
    kickPoll: noop,
    sendText: async () => true,
    enqueueComposerText: async () => true,
    prepareModalOpen: noop,
    afterModalVisibilityChanged: noop,
    restoreModalFocus: noop,
    storageGetItem: () => null,
    storageSetItem: noop,
    storageRemoveItem: noop,
    requestFrame: (callback) => callback(),
    getComputedStyle: () => ({ minHeight: "32px" }),
    activeElement: () => null,
    isHTMLElement: () => false,
  });
  return { textarea, sendBtn };
}

function queueHarness() {
  const ctx = {
    HTMLElement: function HTMLElement() {},
    document: { activeElement: null, querySelector: () => null },
    window: {},
    console,
  };
  vm.createContext(ctx);
  vm.runInContext(source("app_modal.js"), ctx);
  vm.runInContext(source("app_session_helpers.js"), ctx);
  vm.runInContext(source("app_queue.js"), ctx);

  const queueBtn = node();
  const selected = "session-1";
  const session = { session_id: selected, launch_state: "ready", queue_len: 2 };
  const controller = ctx.window.CodoxearQueue.createQueueController({
    queueBackdrop: node(),
    queueCloseBtn: node(),
    queueList: node(),
    queueEmpty: node(),
    queueViewer: node(),
    queueBtn,
    getSelected: () => selected,
    getSessionInfo: () => session,
    isAppDisposed: () => false,
    api: async () => ({}),
    setToast: noop,
    clearCommitUnknownSend: async () => {},
    refreshSessions: async () => {},
    updateQueueBadge: noop,
    syncRecoveryUiForSession: noop,
    kickPoll: noop,
    setPollFastUntilMs: noop,
    handleAppAuthLoss: noop,
    prepareModalOpen: noop,
    afterModalVisibilityChanged: noop,
    el: (tag, attrs = {}) => node({ tag, ...attrs }),
    iconSvg: () => "",
    recoveryPanelFocusFallback: () => null,
    confirmAction: async () => true,
    requestFrame: (callback) => callback(),
    setTimeout: () => 0,
    clearTimeout: noop,
  });
  return { controller, queueBtn };
}

test("composer send remains enabled for text in a ready selected session", () => {
  const { textarea, sendBtn } = composerHarness();
  textarea.value = "send this";
  textarea.emit("input");
  assert.equal(sendBtn.disabled, false);
});

test("queue control remains enabled for a selected session with queued items", () => {
  const { controller, queueBtn } = queueHarness();
  controller.syncQueueSubmitState();
  assert.equal(queueBtn.disabled, false);
});
