import * as CodoxearApi from "./app_api.js";
import * as CodoxearApplicationComposition from "./app_application_composition.js";
import * as CodoxearAttachments from "./app_attachments.js";
import * as CodoxearComposer from "./app_composer.js";
import * as CodoxearConversationCopy from "./app_conversation_copy.js";
import * as CodoxearDialogMenus from "./app_dialog_menu.js";
import * as CodoxearDisplay from "./app_display.js";
import * as CodoxearFileEditMode from "./app_file_ops.js";
import * as CodoxearFileEditor from "./app_file_editor.js";
import * as CodoxearFileHelpers from "./app_file_helpers.js";
import * as CodoxearFilePicker from "./app_file_picker.js";
import * as CodoxearFileTouch from "./app_file_ops.js";
import * as CodoxearFileViewer from "./app_file_viewer.js";
import * as CodoxearInterrupt from "./app_session_lifecycle.js";
import * as CodoxearLaunch from "./app_launch.js";
import * as CodoxearMarkdown from "./app_markdown.js";
import * as CodoxearMessageFlow from "./app_message_flow.js";
import * as CodoxearNavigationPulse from "./app_transcript_render.js";
import * as CodoxearNetwork from "./app_network.js";
import * as CodoxearNewSession from "./app_new_session.js";
import * as CodoxearPendingUser from "./app_transcript.js";
import * as CodoxearPerf from "./app_api.js";
import * as CodoxearPolling from "./app_polling.js";
import * as CodoxearSessionHelpers from "./app_session_helpers.js";
import * as CodoxearSessions from "./app_sessions.js";
import * as CodoxearShell from "./app_shell.js";
import * as CodoxearStorage from "./app_storage.js";
import * as CodoxearTopbar from "./app_topbar.js";
import * as CodoxearViewport from "./app_viewport.js";
import * as CodoxearVoice from "./app_voice.js";
import * as CodoxearVoiceHelpers from "./app_voice_helpers.js";

const global = window;

/* Application composition runtime. This owns the former app.js shell,
 * lifecycle, and controller wiring; app.js intentionally remains only bootstrap. */

  function createElement(tag, attrs = {}, children = [], defaultButtonTooltip = null) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else n.setAttribute(k, v);
    }
    if (tag === "button" && !n.getAttribute("title") && typeof defaultButtonTooltip === "function") {
      const tooltip = defaultButtonTooltip(attrs, n);
      if (tooltip) n.setAttribute("title", tooltip);
    }
    for (const c of children) n.appendChild(c);
    return n;
  }

  function computeAppBaseUrl(locationLike) {
    const here = new URL(locationLike.href);
    const p0 = String(here.pathname || "/");
    if (p0.endsWith("/static/index.html")) {
      return new URL(p0.slice(0, -"/static/index.html".length) + "/", here.origin);
    }
    if (p0.endsWith("/static/")) {
      return new URL(p0.slice(0, -"/static/".length) + "/", here.origin);
    }
    return new URL(".", here);
  }

  const appBaseUrl = computeAppBaseUrl(global.location);

  function resolveAppUrl(path) {
    const s = String(path ?? "");
    const rel = s.startsWith("/") ? s.slice(1) : s;
    return new URL(rel, appBaseUrl).toString();
  }

  function sessionIdFromHash(locationLike = global.location) {
    const raw = String(locationLike.hash || "").startsWith("#") ? String(locationLike.hash || "").slice(1) : String(locationLike.hash || "");
    const params = new URLSearchParams(raw);
    const sid = params.get("session");
    return sid && sid.trim() ? sid.trim() : "";
  }

  function setSessionHash(sessionId, { locationLike = global.location, historyLike = global.history } = {}) {
    const raw = String(locationLike.hash || "").startsWith("#") ? String(locationLike.hash || "").slice(1) : String(locationLike.hash || "");
    const params = new URLSearchParams(raw);
    if (sessionId) params.set("session", sessionId);
    else params.delete("session");
    const next = params.toString();
    const target = `${locationLike.pathname}${locationLike.search}${next ? `#${next}` : ""}`;
    historyLike.replaceState(null, "", target);
    return target;
  }

  function createApplicationController(deps = {}) {
    const window = deps.windowTarget || global;
    const document = deps.documentTarget || window.document;
    if (!window || !document) throw new Error("Codoxear application requires a browser window and document");
    const navigator = deps.navigatorTarget || window.navigator;
    const HTMLElement = deps.HTMLElement || window.HTMLElement;
    const EventSource = deps.EventSource || window.EventSource;
    const AbortController = deps.AbortController || window.AbortController;
    const getComputedStyle = deps.getComputedStyle || window.getComputedStyle.bind(window);
    const requestAnimationFrame = deps.requestAnimationFrame || window.requestAnimationFrame.bind(window);
    const setTimeout = deps.setTimeout || window.setTimeout.bind(window);
    const clearTimeout = deps.clearTimeout || window.clearTimeout.bind(window);

	      const $ = (q) => document.querySelector(q);
	      const UI_VERSION = String(window.CODOXEAR_ASSET_VERSION || "dev");
	      const ATTACH_UPLOAD_MAX_BYTES = (() => {
	        const raw = Number(window.CODOXEAR_ATTACH_MAX_BYTES);
	        if (!Number.isFinite(raw) || raw <= 0) return 16 * 1024 * 1024;
	        return Math.max(1, Math.floor(raw));
	      })();
      function isTextEntryElement(target) {
        return CodoxearViewport.isTextEntryElement(target);
      }
      function updateAppHeightVar() {
        return CodoxearViewport.updateAppHeightVar();
      }
      updateAppHeightVar();
      window.addEventListener("resize", updateAppHeightVar);
      function defaultButtonTooltip(attrs = {}, node = null) {
        return CodoxearDisplay.defaultButtonTooltip(attrs, node);
      }

      // Voice helpers and the voice/settings/notification/announcement
      // controller live in codoxear/static/app_voice.js. ESM surfaces a missing
      // module during loading; controllers validate their injected options.

      const el = (tag, attrs = {}, children = []) => CodoxearDom.createElement(tag, attrs, children, defaultButtonTooltip);

      function pushPerfSample(name, valueMs) {
        return CodoxearPerf.pushSample(name, valueMs);
      }
      function summarizePerf() {
        return CodoxearPerf.summarize();
      }

      window.codoxearPerf = summarizePerf;

      function resolveAppUrl(path) {
        return CodoxearUrls.resolveAppUrl(path);
      }
      function versionedShellAssetPath(path) {
        const version = String(window.CODOXEAR_ASSET_VERSION || "").trim();
        if (!version) return path;
        return `${path}?v=${encodeURIComponent(version)}`;
      }

      function optionalLocalStorage() {
        return typeof CodoxearStorage.optionalLocalStorage === "function" ? CodoxearStorage.optionalLocalStorage() : null;
      }
      function storageGetItem(key) {
        return CodoxearStorage.getItem(key);
      }
      function storageSetItem(key, value) {
        return CodoxearStorage.setItem(key, value);
      }
      function storageRemoveItem(key) {
        return CodoxearStorage.removeItem(key);
      }

      function lastProviderKey(backend) {
        return CodoxearLaunch.lastProviderKey(backend);
      }
      function lastProviderModelKey(backend) {
        return CodoxearLaunch.lastProviderModelKey(backend);
      }
      function loadRememberedBackendChoice() {
        return CodoxearLaunch.loadRememberedBackendChoice();
      }
      function rememberBackendChoice(backend) {
        return CodoxearLaunch.rememberBackendChoice(backend);
      }
      function loadRememberedProviderChoice(backend) {
        return CodoxearLaunch.loadRememberedProviderChoice(backend);
      }
      function rememberProviderChoice(backend, provider) {
        return CodoxearLaunch.rememberProviderChoice(backend, provider);
      }
      function loadRememberedProviderModelChoice(backend) {
        return CodoxearLaunch.loadRememberedProviderModelChoice(backend);
      }
      function rememberedProviderModelAbsentChoice(value) {
        return CodoxearLaunch.rememberedProviderModelAbsentChoice(value);
      }
      function rememberProviderModelChoice(backend, provider, model, options = {}) {
        return CodoxearLaunch.rememberProviderModelChoice(backend, provider, model, options);
      }

      function apiResponseNotModified(obj) {
        return CodoxearApi.apiResponseNotModified(obj);
      }
      function clearApiCache() {
        return CodoxearApi.clearApiCache();
      }
      async function api(path, options = {}) {
        return CodoxearApi.api(path, options);
      }

      function fmtTs(ts) {
        return CodoxearDisplay.fmtTs(ts);
      }

      function fmtBytes(n) {
        return CodoxearDisplay.fmtBytes(n);
      }

      function listFromFilesField(val) {
        return CodoxearFileHelpers.listFromFilesField(val);
      }

      function listFromFileRecords(val) {
        return CodoxearFileHelpers.listFromFileRecords(val);
      }

      function baseName(p) {
        return CodoxearDisplay.baseName(p);
      }

      function fuzzyRecentCwdScore(candidate, query) {
        return CodoxearDisplay.fuzzyRecentCwdScore(candidate, query);
      }

      function shortSessionId(sid) {
        return CodoxearDisplay.shortSessionId(sid);
      }

      function sessionDisplayName(s) {
        return CodoxearDisplay.sessionDisplayName(s);
      }

      const SIDEBAR_REASONING_EFFORT_CODES = Object.freeze({
        off: "off",
        minimal: "min",
        low: "low",
        medium: "med",
        high: "hi",
        xhigh: "xh",
        max: "max",
      });

      function sidebarEffortCode(effort, agentBackend) {
        const normalized = typeof effort === "string" ? effort.trim().toLowerCase() : "";
        const code = SIDEBAR_REASONING_EFFORT_CODES[normalized] || "";
        // Claude Code logs carry no effort evidence: the value is launch-time
        // metadata, not verified current state, so mark it instead of
        // presenting it as live.
        if (agentBackend === "cc" && code) return `${code}*`;
        return code;
      }

      function sidebarModelText(s) {
        const model = s && typeof s.model === "string" ? s.model.trim() : "";
        if (!model || model.toLowerCase() === "default") return "";
        if (model.length <= 16) return model;
        const slash = model.indexOf("/");
        if (slash > 0 && slash < model.length - 1) {
          const provider = model.slice(0, slash);
          const modelName = model.slice(slash + 1);
          const suffixBudget = 8;
          if (modelName.length <= suffixBudget) {
            const providerBudget = Math.max(1, 16 - modelName.length - 2);
            return `${provider.slice(0, providerBudget)}…/${modelName}`;
          }
        }
        return `${model.slice(0, 6)}…${model.slice(-8)}`;
      }

      function sessionIdFromHash() {
        return CodoxearUrls.sessionIdFromHash();
      }

      function setSessionHash(sessionId) {
        CodoxearUrls.setSessionHash(sessionId);
      }

      const SESSION_SIDEBAR_GROUPS = CodoxearSessionHelpers.SESSION_SIDEBAR_GROUPS;

      function sessionLaunchKind(s) {
        return CodoxearSessionHelpers.sessionLaunchKind(s);
      }

      function sessionLaunchIcon(s) {
        return CodoxearSessionHelpers.sessionLaunchIcon(s);
      }

      function sessionLaunchFailed(s) {
        return CodoxearSessionHelpers.sessionLaunchFailed(s);
      }

      function sessionLaunchPending(s) {
        return CodoxearSessionHelpers.sessionLaunchPending(s);
      }

      function sessionHasUnknownSend(s) {
        return CodoxearSessionHelpers.sessionHasUnknownSend(s);
      }

      function sessionIsOrphanRecovery(s) {
        return CodoxearSessionHelpers.sessionIsOrphanRecovery(s);
      }

      function sessionHasOrphanQueueRecovery(s) {
        return CodoxearSessionHelpers.sessionHasOrphanQueueRecovery(s);
      }

      function sessionSidebarGroupKey(s) {
        return CodoxearSessionHelpers.sessionSidebarGroupKey(s);
      }

      function sidebarSessionEntries(sessions) {
        return CodoxearSessionHelpers.sidebarSessionEntries(sessions);
      }

      function sidebarRenderSignature(entries, { selectedId = "", swipeActions = false } = {}) {
        return CodoxearSessionHelpers.sidebarRenderSignature(entries, { selectedId, swipeActions });
      }

      function sessionSelectable(s) {
        return CodoxearSessionHelpers.sessionSelectable(s);
      }

      function diagnosticsProviderDisplay(d) {
        return CodoxearSessionHelpers.diagnosticsProviderDisplay(d, sessionAgentBackend(d));
      }

      function diagnosticsCopyText(sessionId, rows) {
        return CodoxearSessionHelpers.diagnosticsCopyText(sessionId, rows);
      }

      function normalizeQueueItems(data) {
        return CodoxearSessionHelpers.normalizeQueueItems(data);
      }

      function transcriptExportTooLargeCopyMessage(err) {
        return CodoxearConversationCopy.transcriptExportTooLargeCopyMessage(err);
      }

      function copyConversationFailureToast(err) {
        return transcriptExportTooLargeCopyMessage(err) || `copy failed: ${err && err.message ? err.message : "unknown error"}`;
      }

      function normalizeAgentBackendName(value) {
        return CodoxearLaunch.normalizeAgentBackendName(value);
      }
      function agentBackendDisplayName(value) {
        return CodoxearLaunch.agentBackendDisplayName(value);
      }
      function agentBackendLogoPath(value) {
        return CodoxearLaunch.agentBackendLogoPath(value);
      }
      function sessionAgentBackend(session) {
        return CodoxearLaunch.sessionAgentBackend(session);
      }
      function legacyCodexLaunchDefaults(seed = {}) {
        return CodoxearLaunch.legacyCodexLaunchDefaults(seed);
      }
      function emptyPiLaunchDefaults(seed = {}) {
        return CodoxearLaunch.emptyPiLaunchDefaults(seed);
      }
      function emptyCcLaunchDefaults(seed = {}) {
        return CodoxearLaunch.emptyCcLaunchDefaults(seed);
      }
      function redactedLaunchErrorText(value) {
        return CodoxearLaunch.redactedLaunchErrorText(value);
      }

      function sessionLaunchLabel(s) {
        const kind = sessionLaunchKind(s);
        if (kind === "failed") return redactedLaunchErrorText(s && s.launch_error) || "session launch failed";
        if (kind === "web_tmux") return "web-owned tmux session";
        return kind === "web" ? "web-owned session" : "terminal-owned session";
      }

      function sessionIsFast(s) {
        return CodoxearSessionHelpers.sessionIsFast(s);
      }

      function providerChoiceToSettings(choice, agentBackend = "codex") {
        return CodoxearLaunch.providerChoiceToSettings(choice, agentBackend);
      }
      function sessionProviderChoice(session) {
        return CodoxearLaunch.sessionProviderChoice(session);
      }
      function modelOptionMatches(option, query) {
        return CodoxearLaunch.modelOptionMatches(option, query);
      }
      function providerModelDisplay(model, providerChoice = "", options = {}) {
        return CodoxearLaunch.providerModelDisplay(model, providerChoice, options);
      }

	      function fmtIdleAge(seconds) {
        return CodoxearDisplay.fmtIdleAge(seconds);
      }

	      function fmtRelativeAge(seconds) {
        return CodoxearDisplay.fmtRelativeAge(seconds);
      }

      function sessionTitleWithId(s) {
        return CodoxearDisplay.sessionTitleWithId(s);
      }

      function stripPathLocationSuffix(rawPath) {
        return CodoxearFileHelpers.stripPathLocationSuffix(rawPath);
      }

      function isTextFileKind(kind) {
        return CodoxearFileHelpers.isTextFileKind(kind);
      }

      function isDiffableFileKind(kind) {
        return CodoxearFileHelpers.isDiffableFileKind(kind);
      }

      function blockedFileMessage(rel, reason, viewerMaxBytes, size) {
        return CodoxearFileHelpers.blockedFileMessage(rel, reason, viewerMaxBytes, size);
      }

      function formatPriorityOffset(value) {
        return CodoxearFileHelpers.formatPriorityOffset(value);
      }

      function fileSearchScore(candidate, query) {
        return CodoxearFileHelpers.fileSearchScore(candidate, query);
      }

      function normalizeDraftFilePath(raw) {
        return CodoxearFileHelpers.normalizeDraftFilePath(raw);
      }

      function filePickerFoldedSearchText(text) {
        return CodoxearFileHelpers.filePickerFoldedSearchText(text);
      }

      function filePickerOriginalRangeForFolded(mapped, start, end) {
        return CodoxearFileHelpers.filePickerOriginalRangeForFolded(mapped, start, end);
      }

      function filePickerMatchRanges(text, query) {
        return CodoxearFileHelpers.filePickerMatchRanges(text, query);
      }

      function filePickerMatchRangesForQuery(text, query) {
        return CodoxearFileHelpers.filePickerMatchRangesForQuery(text, query);
      }

      function filePickerCandidateScore(path, query) {
        return CodoxearFileHelpers.filePickerCandidateScore(path, query);
      }

      function compareFilePickerEntries(a, b) {
        return CodoxearFileHelpers.compareFilePickerEntries(a, b);
      }

      function normalizeFileCandidateSource(source) {
        return CodoxearFileHelpers.normalizeFileCandidateSource(source);
      }

      function filePickerSectionLabel(source) {
        return CodoxearFileHelpers.filePickerSectionLabel(source);
      }

      function duplicateFilePickerPaths(entries) {
        return CodoxearFileHelpers.duplicateFilePickerPaths(entries);
      }

      function rawByteDuplicatePaths(entries) {
        return CodoxearFileHelpers.rawByteDuplicatePaths(entries);
      }

      function filePickerIdentityHint(entry, duplicatePaths, options) {
        return CodoxearFileHelpers.filePickerIdentityHint(entry, duplicatePaths, options);
      }

      function filePickerTitle(entry, hint = "") {
        return CodoxearFileHelpers.filePickerTitle(entry, hint);
      }

      function dataTransferHasFiles(data) {
        return CodoxearFileHelpers.dataTransferHasFiles(data);
      }

      function extractFilesFromClipboardData(data) {
        return CodoxearFileHelpers.extractFilesFromClipboardData(data);
      }

      function extractFilesFromDropData(data) {
        return CodoxearFileHelpers.extractFilesFromDropData(data);
      }

      function safeAttachmentStem(name) {
        return CodoxearFileHelpers.attachmentSafeStem(name);
      }

      function isLikelyHeic(file) {
        return CodoxearFileHelpers.attachmentIsLikelyHeic(file);
      }

      function looksLikeImage(file) {
        return CodoxearFileHelpers.attachmentLooksLikeImage(file);
      }

      function b64FromBytes(bytes) {
        return CodoxearFileHelpers.bytesToBase64(bytes, btoa);
      }




      function normalizeLineNumber(value) {
        return CodoxearMarkdown.normalizeLineNumber(value);
      }
      function parseLocalFileRef(rawValue) {
        return CodoxearMarkdown.parseLocalFileRef(rawValue);
      }
      function isMarkdownPreviewable(path) {
        return CodoxearMarkdown.isMarkdownPreviewable(path);
      }
      function markdownPreviewHtml(src, options = {}) {
        return CodoxearMarkdown.markdownPreviewHtml(src, options);
      }
      function chatMarkdownHtmlCached(src, sessionId) {
        return CodoxearMarkdown.chatMarkdownHtmlCached(src, sessionId);
      }

      function iconSvg(name) {
        return CodoxearDisplay.iconSvg(name);
      }

      let activeAppCleanup = null;
      function cleanupActiveApp() {
        if (typeof activeAppCleanup !== "function") return;
        const cleanup = activeAppCleanup;
        activeAppCleanup = null;
        cleanup();
      }

      function renderLogin(onAuthed) {
        cleanupActiveApp();
        const root = $("#root");
        root.innerHTML = "";
        const err = el("div", { class: "err", id: "loginError", role: "alert" });
        const pwInput = el("input", {
          type: "password",
          id: "pw",
          name: "password",
          placeholder: "Password",
          "aria-label": "Password",
          autocomplete: "current-password",
          "aria-describedby": "loginError",
        });
        const loginBtn = el("button", { class: "primary", id: "loginBtn", type: "submit", text: "Login" });
        const wrap = el("div", { class: "loginWrap" });
        const form = el("form", { class: "login", id: "loginForm" }, [
          el("h1", { text: "Codoxear login" }),
          el("label", { class: "sr-only", for: "pw", text: "Password" }),
          el("div", { class: "row2" }, [
            pwInput,
            loginBtn,
            err,
          ]),
        ]);
        wrap.appendChild(form);
        root.appendChild(wrap);
        form.onsubmit = async (e) => {
          e.preventDefault();
          err.textContent = "";
          const pw = pwInput.value;
          try {
            await api("/api/login", { method: "POST", body: { password: pw } });
          onAuthed();
          } catch (e2) {
          err.textContent = e2.obj?.error || e2.message;
          }
        };
        pwInput.focus();
        if (typeof window.__codoxearMarkBootstrapped === "function") window.__codoxearMarkBootstrapped();
      }

      const applicationComposition = CodoxearApplicationComposition.createApplicationComposition({
        window, document, navigator, HTMLElement, EventSource, AbortController, getComputedStyle,
        requestAnimationFrame, setTimeout, clearTimeout, $, UI_VERSION, ATTACH_UPLOAD_MAX_BYTES,
        isTextEntryElement, updateAppHeightVar,
        codoxearViewport: CodoxearViewport, codoxearDisplay: CodoxearDisplay, defaultButtonTooltip, codoxearVoiceHelpers: CodoxearVoiceHelpers, codoxearVoice: CodoxearVoice,
        codoxearDom: CodoxearDom, el, codoxearShell: CodoxearShell, codoxearSessions: CodoxearSessions, codoxearComposer: CodoxearComposer, codoxearAttachments: CodoxearAttachments, codoxearTopbar: CodoxearTopbar,
        codoxearMessageFlow: CodoxearMessageFlow, codoxearInterrupt: CodoxearInterrupt, codoxearDialogMenus: CodoxearDialogMenus,
        codoxearFileEditMode: CodoxearFileEditMode, codoxearPendingUser: CodoxearPendingUser, codoxearNavigationPulse: CodoxearNavigationPulse,
        codoxearFileTouch: CodoxearFileTouch, codoxearPerfHelpers: CodoxearPerf, pushPerfSample, summarizePerf, codoxearUrls: CodoxearUrls,
        resolveAppUrl, versionedShellAssetPath, codoxearStorage: CodoxearStorage, optionalLocalStorage, storageGetItem,
        storageSetItem, storageRemoveItem, codoxearLaunch: CodoxearLaunch, codoxearNewSession: CodoxearNewSession, lastProviderKey,
        lastProviderModelKey, loadRememberedBackendChoice, rememberBackendChoice,
        loadRememberedProviderChoice, rememberProviderChoice, loadRememberedProviderModelChoice,
        rememberedProviderModelAbsentChoice, rememberProviderModelChoice, codoxearApi: CodoxearApi,
        apiResponseNotModified, clearApiCache, api, fmtTs, fmtBytes, codoxearFileHelpers: CodoxearFileHelpers,
        listFromFilesField, listFromFileRecords, baseName, fuzzyRecentCwdScore, shortSessionId,
        sessionDisplayName, sidebarEffortCode, sidebarModelText, sessionIdFromHash, setSessionHash,
        codoxearSessionHelpers: CodoxearSessionHelpers, sessionLaunchKind, sessionLaunchIcon, sessionLaunchFailed,
        sessionLaunchPending, sessionHasUnknownSend, sessionIsOrphanRecovery,
        sessionHasOrphanQueueRecovery, sessionSidebarGroupKey, sidebarSessionEntries,
        sidebarRenderSignature, sessionSelectable, diagnosticsProviderDisplay, diagnosticsCopyText,
        normalizeQueueItems, codoxearPolling: CodoxearPolling, codoxearNetwork: CodoxearNetwork,
        transcriptExportTooLargeCopyMessage, copyConversationFailureToast, normalizeAgentBackendName,
        agentBackendDisplayName, agentBackendLogoPath, sessionAgentBackend, legacyCodexLaunchDefaults,
        emptyPiLaunchDefaults, emptyCcLaunchDefaults, redactedLaunchErrorText, sessionLaunchLabel,
        sessionIsFast, providerChoiceToSettings, sessionProviderChoice, modelOptionMatches,
        providerModelDisplay, fmtIdleAge, fmtRelativeAge, sessionTitleWithId, stripPathLocationSuffix,
        isTextFileKind, isDiffableFileKind, blockedFileMessage, formatPriorityOffset, fileSearchScore,
        normalizeDraftFilePath, filePickerFoldedSearchText, filePickerOriginalRangeForFolded,
        filePickerMatchRanges, filePickerMatchRangesForQuery, filePickerCandidateScore,
        compareFilePickerEntries, normalizeFileCandidateSource, filePickerSectionLabel,
        duplicateFilePickerPaths, rawByteDuplicatePaths, filePickerIdentityHint, filePickerTitle,
        dataTransferHasFiles, extractFilesFromClipboardData, extractFilesFromDropData, safeAttachmentStem,
        isLikelyHeic, looksLikeImage, b64FromBytes, codoxearFilePicker: CodoxearFilePicker, codoxearFileViewer: CodoxearFileViewer,
        codoxearFileEditor: CodoxearFileEditor, codoxearMarkdown: CodoxearMarkdown, normalizeLineNumber, parseLocalFileRef,
        isMarkdownPreviewable, markdownPreviewHtml, chatMarkdownHtmlCached, iconSvg,
        cleanupActiveApp, renderLogin,
        setActiveAppCleanup: (cleanup) => { activeAppCleanup = cleanup; },
        clearActiveAppCleanup: (cleanup) => { if (activeAppCleanup === cleanup) activeAppCleanup = null; },
      });
      function renderApp() {
        return applicationComposition.renderApp();
      }


    return Object.freeze({ api, cleanupActiveApp, renderApp, renderLogin });
  }

const appBaseHref = appBaseUrl.toString();
const CodoxearDom = { createElement };
const CodoxearUrls = { appBaseHref, resolveAppUrl, sessionIdFromHash, setSessionHash };

export { createElement, appBaseHref, resolveAppUrl, sessionIdFromHash, setSessionHash, createApplicationController };
