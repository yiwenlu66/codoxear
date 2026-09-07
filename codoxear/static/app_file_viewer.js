import * as CodoxearFileCandidates from "./app_file_candidates.js";
import * as CodoxearFileDownload from "./app_file_download.js";
import * as CodoxearFileMode from "./app_file_mode.js";
import * as CodoxearFilePasteDialog from "./app_file_paste_dialog.js";
import * as CodoxearFilePdf from "./app_file_pdf.js";
import * as CodoxearFileRenderSurface from "./app_file_render_surface.js";
import * as CodoxearFileUnsavedDialog from "./app_file_unsaved_dialog.js";
import * as CodoxearFileVideo from "./app_file_video.js";
import * as CodoxearFileViewerController from "./app_file_viewer_controller.js";
import * as CodoxearFileViewerLifecycle from "./app_file_viewer_lifecycle.js";
import * as CodoxearFileViewerPanel from "./app_file_viewer_panel.js";


function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireStatusNode(value) {
    if (!value || typeof value.replaceChildren !== "function") throw new TypeError("file viewer dependency missing: fileStatus");
    return value;
  }

  function requireEditButtonNode(value) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function" || typeof value.setAttribute !== "function") {
      throw new TypeError("file viewer dependency missing: fileEditButton");
    }
    return value;
  }

  const BROWSER_SAFE_VIDEO_TYPES = new Set(["video/mp4", "video/webm", "video/ogg"]);
  const FILE_EDITOR_UNAVAILABLE_MESSAGE = "Editing is unavailable because the code editor failed to load. Read-only preview remains available.";

  function bindFileTouchPress(button, handler, options = {}) {
    if (!button || typeof button.addEventListener !== "function" || typeof handler !== "function") return false;
    const nowMs = typeof options.nowMs === "function" ? options.nowMs : () => Date.now();
    let suppressClickUntil = 0;
    let sawPointerTouchAt = 0;
    const run = (event) => {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      suppressClickUntil = nowMs() + 700;
      handler();
    };
    button.addEventListener("pointerdown", (event) => {
      if (event && event.pointerType === "touch") sawPointerTouchAt = nowMs();
      run(event);
    });
    button.addEventListener(
      "touchstart",
      (event) => {
        if (nowMs() - sawPointerTouchAt < 700) {
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        run(event);
      },
      { passive: false }
    );
    button.addEventListener("click", (event) => {
      if (nowMs() < suppressClickUntil) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      run(event);
    });
    return true;
  }

  function bindFileTouchClick(button, handler) {
    if (!button || typeof button.addEventListener !== "function" || typeof handler !== "function") return false;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      handler();
    });
    return true;
  }

  function requireStyledNode(value, name) {
    if (!value || !value.style) throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireRenderHostNode(value, name) {
    if (!value || !("innerHTML" in value) || typeof value.appendChild !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requirePasteInput(value) {
    if (!value || !("value" in value)) throw new TypeError("file viewer dependency missing: filePasteInput");
    return value;
  }

  function requireVideoNode(value) {
    if (!value || !value.style || typeof value.removeAttribute !== "function" || typeof value.load !== "function") {
      throw new TypeError("file viewer dependency missing: fileVideo");
    }
    return value;
  }

  function requireImageNode(value) {
    if (!value || !value.style || typeof value.removeAttribute !== "function") {
      throw new TypeError("file viewer dependency missing: fileImage");
    }
    return value;
  }

  function requireToggleClassNode(value, name) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireModeControlButton(value, name) {
    if (
      !value ||
      !value.classList ||
      typeof value.classList.toggle !== "function" ||
      !value.style ||
      typeof value.setAttribute !== "function" ||
      !("disabled" in value)
    ) {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireModalHostNode(value, name) {
    if (!value || typeof value.setAttribute !== "function" || typeof value.removeAttribute !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireTextNode(value, name) {
    if (!value || !("textContent" in value)) throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireUnsavedButtonNode(value, name) {
    if (!value || !("hidden" in value) || !("disabled" in value) || !("textContent" in value)) {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  const { createFileCandidateRefreshRuntime } = CodoxearFileCandidates;
  const { createFileDownloadRuntime } = CodoxearFileDownload;
  const { createFileViewerLifecycleRuntime } = CodoxearFileViewerLifecycle;
  const { createFileViewerPanelRuntime } = CodoxearFileViewerPanel;
  const { createFileUnsavedDialogRuntime } = CodoxearFileUnsavedDialog;
  const { createFilePasteDialogRuntime } = CodoxearFilePasteDialog;
  const { createFilePdfRenderRuntime } = CodoxearFilePdf;
  const { createFileVideoPreviewRuntime } = CodoxearFileVideo;
  const { createFileModeControlsRuntime } = CodoxearFileMode;
  const { createFileRenderSurfaceRuntime } = CodoxearFileRenderSurface;
  const { createFileViewerController } = CodoxearFileViewerController;

  function createFileViewerModalRuntime(options = {}) {
    const backdrop = requireStyledNode(options.backdrop, "fileBackdrop");
    const viewer = requireStyledNode(options.viewer, "fileViewer");
    const pickerInput = options.pickerInput || null;
    const closeButton = options.closeButton || null;
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const focusModalSurface = requireFunction(options.focusModalSurface, "focusModalSurface");
    const restoreModalFocus = requireFunction(options.restoreModalFocus, "restoreModalFocus");
    const isModalTargetOpen = requireFunction(options.isModalTargetOpen, "isModalTargetOpen");
    const setReturnFocusElement = requireFunction(options.setReturnFocusElement, "setReturnFocusElement");
    const takeReturnFocusElement = requireFunction(options.takeReturnFocusElement, "takeReturnFocusElement");

    function focusPickerInput() {
      if (!pickerInput || typeof pickerInput.focus !== "function") return false;
      try {
        pickerInput.focus({ preventScroll: true });
      } catch (_) {
        pickerInput.focus();
      }
      return true;
    }

    function isOpen() {
      return viewer.style.display === "flex";
    }

    function show({ wasOpen = false, queryOpen = false, activeElement = null, ElementCtor = null } = {}) {
      if (!wasOpen) setReturnFocusElement(activeElement, ElementCtor);
      prepareModalOpen();
      backdrop.style.display = "block";
      viewer.style.display = "flex";
      afterModalVisibilityChanged();
      if (!wasOpen && queryOpen) focusPickerInput();
      else if (!wasOpen) focusModalSurface(viewer);
      return true;
    }

    function beginHide() {
      const wasOpen = isModalTargetOpen(viewer);
      const focusTarget = takeReturnFocusElement();
      return Object.freeze({ wasOpen, focusTarget });
    }

    function hideDisplay() {
      backdrop.style.display = "none";
      viewer.style.display = "none";
      return true;
    }

    function finishHide(state = {}) {
      afterModalVisibilityChanged();
      if (state.wasOpen) restoreModalFocus(state.focusTarget, () => isModalTargetOpen(viewer));
      return true;
    }

    return Object.freeze({ beginHide, finishHide, hideDisplay, isOpen, show });
  }

  function createFileTouchToolbarRuntime(options = {}) {
    const toolbar = requireStyledNode(options.toolbar, "fileTouchToolbar");
    const actions = requireStyledNode(options.actions, "fileTouchActions");
    const dpad = requireStyledNode(options.dpad, "fileTouchDpad");
    const copyButton = requireStyledNode(options.copyButton, "fileTouchCopyButton");
    const pasteButton = requireStyledNode(options.pasteButton, "fileTouchPasteButton");
    const selectButton = requireToggleClassNode(options.selectButton, "fileTouchSelectButton");

    function update(state = {}) {
      const toolbarState = state || {};
      if (!toolbarState.visible) {
        toolbar.style.display = "none";
        dpad.style.display = "none";
        copyButton.style.display = "none";
        pasteButton.style.display = "none";
        return Object.freeze({ visible: false });
      }
      selectButton.classList.toggle("active", Boolean(toolbarState.selectActive));
      dpad.style.display = toolbarState.dpadVisible ? "grid" : "none";
      copyButton.style.display = toolbarState.copyVisible ? "" : "none";
      pasteButton.style.display = toolbarState.pasteVisible ? "" : "none";
      actions.style.display = "flex";
      toolbar.style.display = "flex";
      return Object.freeze({ visible: true });
    }

    return Object.freeze({ update });
  }

  function createFileFallbackRuntime(options = {}) {
    const host = requireRenderHostNode(options.host, "fileFallbackHost");
    const el = requireFunction(options.el, "el");
    const normalizeLineNumber = requireFunction(options.normalizeLineNumber, "normalizeLineNumber");
    const requestFrame = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");

    function renderPlainText(rel, text, lineNumber = null, reason = "Rich file viewer unavailable") {
      host.innerHTML = "";
      const targetLine = normalizeLineNumber(lineNumber) || 1;
      const notice = el("div", { class: "fileFallbackNotice" }, [
        el("div", { class: "title", text: "Plain text fallback" }),
        el("p", { text: `${reason}. Showing a read-only plain-text view.` }),
      ]);
      const pre = el("pre", { class: "filePlainFallbackText", text: String(text || "") });
      host.appendChild(el("div", { class: "filePlainFallback", "data-path": rel }, [notice, pre]));
      if (targetLine > 1) {
        requestFrame(() => {
          host.scrollTop = Math.max(0, (targetLine - 1) * 18);
        });
      }
      return Object.freeze({ targetLine });
    }

    function renderDownload(rel, url, reason = "Preview unavailable") {
      host.innerHTML = "";
      const link = el("a", { href: url, target: "_blank", rel: "noopener", text: "Open or download file" });
      const body = el("div", { class: "fileBlockedNotice fileDownloadFallback" }, [
        el("div", { class: "title", text: "Preview unavailable" }),
        el("p", { text: `${reason}. You can still open or download ${rel}.` }),
        el("div", { class: "fileFallbackActions" }, [link]),
      ]);
      host.appendChild(body);
      return true;
    }

    function renderBlocked(message) {
      host.innerHTML = "";
      const body = el("div", { class: "fileBlockedNotice" }, [
        el("div", { class: "title", text: "Preview unavailable" }),
        el("p", { text: String(message || "") }),
      ]);
      host.appendChild(body);
      return true;
    }

    function renderMarkdown(rel, text, sessionId, markdownPreviewHtml, upgradeCandidateFileRefs) {
      const renderHtml = requireFunction(markdownPreviewHtml, "markdownPreviewHtml");
      const upgradeRefs = requireFunction(upgradeCandidateFileRefs, "upgradeCandidateFileRefs");
      host.innerHTML = "";
      const preview = el("div", {
        class: "md fileMarkdownPreview",
        html: renderHtml(String(text || ""), { filePath: rel, sessionId: String(sessionId || "") }),
      });
      host.appendChild(preview);
      void upgradeRefs(preview);
      return preview;
    }

    function applicationDeps() {
      return {
        disposeFileEditor: requireFunction(options.disposeFileEditor, "disposeFileEditor"),
        disposePdfRender: requireFunction(options.disposePdfRender, "disposePdfRender"),
        clearFileVideo: requireFunction(options.clearFileVideo, "clearFileVideo"),
        setFileRenderSurface: requireFunction(options.setFileRenderSurface, "setFileRenderSurface"),
        setFileEditorKind: requireFunction(options.setFileEditorKind, "setFileEditorKind"),
        applyPlainTextFallbackState: requireFunction(options.applyPlainTextFallbackState, "applyPlainTextFallbackState"),
        updateFileTouchToolbar: requireFunction(options.updateFileTouchToolbar, "updateFileTouchToolbar"),
        currentSessionId: requireFunction(options.currentSessionId, "currentSessionId"),
        markdownPreviewHtml: requireFunction(options.markdownPreviewHtml, "markdownPreviewHtml"),
        upgradeCandidateFileRefs: requireFunction(options.upgradeCandidateFileRefs, "upgradeCandidateFileRefs"),
        blockedFileMessage: requireFunction(options.blockedFileMessage, "blockedFileMessage"),
      };
    }

    function prepareFallbackSurface(deps) {
      deps.disposeFileEditor();
      deps.clearFileVideo();
      deps.setFileRenderSurface("diff");
      return true;
    }

    function applyPlainText(rel, text, lineNumber = null, reason = "Rich file viewer unavailable") {
      const deps = applicationDeps();
      prepareFallbackSurface(deps);
      deps.setFileEditorKind("plain-fallback");
      deps.applyPlainTextFallbackState();
      return renderPlainText(rel, text, lineNumber, reason);
    }

    function applyDownload(rel, url, reason = "Preview unavailable") {
      const deps = applicationDeps();
      deps.disposePdfRender();
      prepareFallbackSurface(deps);
      const result = renderDownload(rel, url, reason);
      deps.updateFileTouchToolbar();
      return result;
    }

    function applyMarkdown(rel, text) {
      const deps = applicationDeps();
      prepareFallbackSurface(deps);
      const result = renderMarkdown(rel, text, deps.currentSessionId(), deps.markdownPreviewHtml, deps.upgradeCandidateFileRefs);
      deps.updateFileTouchToolbar();
      return result;
    }

    function applyBlocked(rel, reason, viewerMaxBytes, size) {
      const deps = applicationDeps();
      prepareFallbackSurface(deps);
      const result = renderBlocked(deps.blockedFileMessage(rel, reason, viewerMaxBytes, size));
      deps.updateFileTouchToolbar();
      return result;
    }

    return Object.freeze({ applyBlocked, applyDownload, applyMarkdown, applyPlainText, renderBlocked, renderDownload, renderMarkdown, renderPlainText });
  }

  function createFileLoadResultRuntime(options = {}) {
    const controller = options.controller || null;
    if (!controller || typeof controller !== "object") throw new TypeError("file viewer dependency missing: controller");
    const prepareFileLoadResult = requireFunction(controller.prepareFileLoadResult, "controller.prepareFileLoadResult").bind(controller);
    const isCurrentFileOpenRequest = requireFunction(controller.isCurrentFileOpenRequest, "controller.isCurrentFileOpenRequest").bind(controller);
    const handleActiveVideoLoadError = requireFunction(controller.handleActiveVideoLoadError, "controller.handleActiveVideoLoadError").bind(controller);
    const handleActiveVideoLoadedMetadata = requireFunction(controller.handleActiveVideoLoadedMetadata, "controller.handleActiveVideoLoadedMetadata").bind(controller);
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const setStatus = requireFunction(options.setStatus, "setStatus");
    const disposeFileEditor = requireFunction(options.disposeFileEditor, "disposeFileEditor");
    const renderMonacoDiff = requireFunction(options.renderMonacoDiff, "renderMonacoDiff");
    const renderMonacoFile = requireFunction(options.renderMonacoFile, "renderMonacoFile");
    const renderMarkdownPreview = requireFunction(options.renderMarkdownPreview, "renderMarkdownPreview");
    const renderBlockedFileNotice = requireFunction(options.renderBlockedFileNotice, "renderBlockedFileNotice");
    const renderPdfFile = requireFunction(options.renderPdfFile, "renderPdfFile");
    const showImage = requireFunction(options.showImage, "showImage");
    const showVideo = requireFunction(options.showVideo, "showVideo");
    const loadCompatibleVideoPreview = requireFunction(options.loadCompatibleVideoPreview, "loadCompatibleVideoPreview");

    async function apply(rel, result, request, { viewMode = "file" } = {}) {
      const loadPlan = prepareFileLoadResult(rel, result, request, { viewMode });
      if (!loadPlan) return false;
      if (loadPlan.kind === "diff") {
        if (loadPlan.noDiff) {
          disposeFileEditor();
          setStatus(loadPlan.status);
          return true;
        }
        const rendered = await renderMonacoDiff(rel, loadPlan.baseText, loadPlan.currentText, request.line, request);
        if (!rendered || !isCurrentFileOpenRequest(request)) return false;
        setStatus(rendered.monacoUnavailable && rendered.status ? rendered.status : loadPlan.status);
        return true;
      }
      if (loadPlan.kind === "image") {
        showImage(resolveAppUrl(loadPlan.imageUrl), loadPlan.alt);
        setStatus(loadPlan.status);
        return true;
      }
      if (loadPlan.kind === "pdf") {
        const rendered = await renderPdfFile(rel, resolveAppUrl(loadPlan.pdfUrl), request);
        if (!rendered || !isCurrentFileOpenRequest(request)) return false;
        setStatus(loadPlan.status);
        return true;
      }
      if (loadPlan.kind === "video") {
        showVideo(loadPlan, {
          resolveAppUrl,
          setStatus,
          loadPreview: (nextToken, options) => loadCompatibleVideoPreview(nextToken, options),
          handleError: (plan, helpers) => handleActiveVideoLoadError(plan.token, {
            rel: plan.rel,
            previewUrl: plan.previewUrl,
            clearVideoHandlers: helpers.clearVideoHandlers,
            loadPreview: helpers.loadPreview,
          }),
          handleLoadedMetadata: (plan) => handleActiveVideoLoadedMetadata(plan.token),
        });
        return true;
      }
      if (loadPlan.kind === "download_only") {
        renderBlockedFileNotice(rel, loadPlan.reason, loadPlan.viewerMaxBytes, loadPlan.size);
        setStatus(loadPlan.status);
        return true;
      }
      if (loadPlan.kind === "text") {
        if (loadPlan.renderPreview) {
          renderMarkdownPreview(rel, loadPlan.text);
        } else {
          const rendered = await renderMonacoFile(rel, loadPlan.text, request.line, "", request);
          if (!rendered || !isCurrentFileOpenRequest(request)) return false;
          if (rendered.monacoUnavailable && rendered.status) {
            setStatus(rendered.status);
            return true;
          }
        }
        setStatus(loadPlan.status);
        return true;
      }
      throw new Error("invalid file load plan");
    }

    return Object.freeze({ apply });
  }

  function createFileInspectRuntime(options = {}) {
    const currentSessionId = requireFunction(options.currentSessionId, "currentSessionId");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("file viewer dependency missing: sessionState");
    const normalizeFileApiPath = requireFunction(options.normalizeFileApiPath, "normalizeFileApiPath");
    const api = requireFunction(options.api, "api");

    async function inspectSessionFilePath(path, { gitPath = false, apiPath = "" } = {}) {
      const sid = currentSessionId() || sessionState.get("selected") || "";
      if (!sid) throw new Error("select a session first");
      try {
        const body = { session_id: sid, path };
        if (gitPath) {
          body.git_path = true;
          const token = normalizeFileApiPath(apiPath);
          if (token) body.path_token = token;
        } else {
          const token = normalizeFileApiPath(apiPath);
          if (token) body.path_token = token;
        }
        const res = await api("/api/files/inspect", {
          method: "POST",
          body,
        });
        return { exists: true, ...res };
      } catch (error) {
        if (error && error.status === 404) return { exists: false };
        throw error;
      }
    }

    return Object.freeze({ inspectSessionFilePath });
  }

  function createFileReferenceRuntime(options = {}) {
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("file viewer dependency missing: sessionState");
    const sessionById = requireFunction(options.sessionById, "sessionById");
    const chatRoot = options.chatRoot;
    const ElementCtor = options.ElementCtor || null;
    const sessionRelativePath = requireFunction(options.sessionRelativePath, "sessionRelativePath");
    const listFromFilesField = requireFunction(options.listFromFilesField, "listFromFilesField");
    const listFromFileRecords = requireFunction(options.listFromFileRecords, "listFromFileRecords");
    const normalizeFileApiPath = requireFunction(options.normalizeFileApiPath, "normalizeFileApiPath");
    const normalizeLineNumber = requireFunction(options.normalizeLineNumber, "normalizeLineNumber");
    const api = requireFunction(options.api, "api");
    const el = requireFunction(options.el, "el");
    const validationCache = new Map();
    const validationPending = new Map();
    const candidateCache = new Map();
    const searchCache = new Map();

    function clearDiscoveryCaches() {
      candidateCache.clear();
      searchCache.clear();
      return true;
    }

    function collectMessageFileRefs() {
      const selected = sessionState.get("selected");
      if (!selected) return [];
      const out = [];
      const seen = new Set();
      const nodes = chatRoot && typeof chatRoot.querySelectorAll === "function" ? Array.from(chatRoot.querySelectorAll("[data-file-path]")) : [];
      for (const node of nodes) {
        if (ElementCtor && !(node instanceof ElementCtor)) continue;
        const kind = String(node.getAttribute("data-file-kind") || "").trim();
        if (kind === "directory") continue;
        const raw = String(node.getAttribute("data-file-path") ?? "");
        if (raw === "") continue;
        const rel = raw.startsWith("/") ? sessionRelativePath(raw) || "" : raw.replace(/^\.?\//, "");
        if (!rel || rel === "." || seen.has(rel)) continue;
        seen.add(rel);
        out.push(rel);
      }
      return out;
    }

    function normalizeCandidate(candidate) {
      if (typeof candidate === "string") return candidate ? { path: candidate, gitPath: false, apiPath: "" } : null;
      if (!candidate || typeof candidate !== "object") return null;
      const path = typeof candidate.path === "string" ? candidate.path : "";
      if (!path) return null;
      return { path, gitPath: Boolean(candidate.gitPath), apiPath: normalizeFileApiPath(candidate.apiPath || candidate.api_path || "") };
    }

    function exactBareMatches(paths, rawPath) {
      const target = String(rawPath ?? "");
      const out = [];
      const seen = new Set();
      for (const candidate of Array.isArray(paths) ? paths : []) {
        const entry = normalizeCandidate(candidate);
        if (!entry) continue;
        const key = `${entry.gitPath ? "git" : "session"}\u0000${entry.path}\u0000${entry.apiPath || ""}`;
        if (seen.has(key)) continue;
        const tail = entry.path.split("/").pop() || "";
        if (tail !== target) continue;
        seen.add(key);
        out.push(entry);
      }
      return out;
    }

    function entriesMayReferToSamePath(entries) {
      const normalized = (Array.isArray(entries) ? entries : []).map(normalizeCandidate).filter(Boolean);
      for (let i = 0; i < normalized.length; i += 1) {
        for (let j = i + 1; j < normalized.length; j += 1) {
          const a = normalized[i];
          const b = normalized[j];
          if (a.gitPath === b.gitPath) continue;
          if (a.path === b.path || a.path.endsWith(`/${b.path}`) || b.path.endsWith(`/${a.path}`)) return true;
        }
      }
      return false;
    }

    async function getKnownCandidates() {
      const sid = sessionState.get("selected");
      if (!sid) return [];
      const hit = candidateCache.get(sid);
      if (hit) return hit;
      const task = (async () => {
        const out = new Set();
        const session = sessionById(sid);
        const addCandidate = (path, gitPath = false, apiPath = "") => {
          const rel = String(path || "");
          if (!rel || rel === ".") return;
          const token = normalizeFileApiPath(apiPath);
          out.add(JSON.stringify({ path: rel, gitPath: Boolean(gitPath), apiPath: token }));
        };
        for (const record of listFromFileRecords(session && session.files)) {
          const rel = sessionRelativePath(record.path);
          if (typeof rel === "string") addCandidate(rel, false, record.apiPath || "");
        }
        for (const rel of collectMessageFileRefs()) addCandidate(rel, false);
        try {
          const res = await api(`/api/sessions/${sid}/git/changed_files`);
          const entries = Array.isArray(res.entries) ? res.entries : [];
          for (const entry of entries) {
            if (!entry || typeof entry.path !== "string") continue;
            if (entry.path !== "") addCandidate(entry.path, true, entry.api_path || entry.apiPath || "");
          }
        } catch {}
        return [...out].map((raw) => {
          try {
            return JSON.parse(raw);
          } catch {
            return null;
          }
        }).filter(Boolean);
      })();
      candidateCache.set(sid, task);
      const resolved = await task;
      candidateCache.set(sid, resolved);
      return resolved;
    }

    function validationKey(path, gitPath = false) {
      return `${sessionState.get("selected") || ""}|${gitPath ? "git" : "session"}|${String(path ?? "")}`;
    }

    async function searchBareCandidates(rawPath) {
      const sid = sessionState.get("selected") || "";
      const query = String(rawPath ?? "");
      if (!sid || query === "" || query.includes("/")) return { matches: [], truncated: false };
      const key = `${sid}|${query}`;
      if (searchCache.has(key)) return searchCache.get(key);
      const task = (async () => {
        try {
          const res = await api(`/api/sessions/${sid}/file/search?q=${encodeURIComponent(query)}&limit=80`);
          const matchObjects = Array.isArray(res && res.matches)
            ? res.matches.filter((item) => item && typeof item.path === "string")
            : [];
          return { matches: exactBareMatches(matchObjects, query), truncated: Boolean(res && res.truncated), failed: false };
        } catch {
          return { matches: [], truncated: true, failed: true };
        }
      })();
      searchCache.set(key, task);
      const resolved = await task;
      if (resolved && resolved.failed) searchCache.delete(key);
      else searchCache.set(key, resolved);
      return resolved;
    }

    async function inspectCandidate(entry, rawPath) {
      const candidate = normalizeCandidate(entry) || { path: String(rawPath ?? ""), gitPath: false, apiPath: "" };
      const inspectPath = candidate.path;
      const key = validationKey(inspectPath, candidate.gitPath);
      if (validationCache.has(key)) return { ...validationCache.get(key), path: rawPath };
      const pending = validationPending.get(key);
      if (pending) {
        const pendingResult = await pending;
        return pendingResult && typeof pendingResult === "object" ? { ...pendingResult, path: rawPath } : pendingResult;
      }
      const task = (async () => {
        try {
          const body = { path: inspectPath };
          const sid = sessionState.get("selected");
          if (sid) body.session_id = sid;
          if (candidate.gitPath) body.git_path = true;
          const inspectToken = normalizeFileApiPath(candidate.apiPath);
          if (inspectToken) body.path_token = inspectToken;
          const res = await api("/api/files/inspect", { method: "POST", body });
          return { ok: true, path: rawPath, inspectPath, gitPath: candidate.gitPath, kind: res.kind, resolvedPath: res.path };
        } catch {
          return { ok: false, path: rawPath, inspectPath, gitPath: candidate.gitPath };
        }
      })();
      validationPending.set(key, task);
      const result = await task;
      validationPending.delete(key);
      if (result && result.ok) validationCache.set(key, result);
      return result;
    }

    async function inspectPlainCandidates(paths) {
      const sid = sessionState.get("selected") || "";
      const results = new Map();
      if (!sid) return results;
      const uniquePaths = [...new Set((Array.isArray(paths) ? paths : []).filter((path) => typeof path === "string" && path !== ""))];
      const uncachedPaths = [];
      for (const path of uniquePaths) {
        const cached = validationCache.get(validationKey(path, false));
        if (cached) {
          results.set(path, { ...cached, path });
          continue;
        }
        uncachedPaths.push(path);
      }
      for (let offset = 0; offset < uncachedPaths.length; offset += 50) {
        const chunk = uncachedPaths.slice(offset, offset + 50);
        try {
          const res = await api("/api/files/inspect-batch", { method: "POST", body: { session_id: sid, paths: chunk } });
          const inspected = Array.isArray(res && res.results) ? res.results : [];
          for (let index = 0; index < chunk.length; index += 1) {
            const inspectPath = chunk[index];
            const inspect = inspected[index];
            const result = inspect && inspect.exists
              ? { ok: true, path: inspectPath, inspectPath, gitPath: false, kind: inspect.kind, resolvedPath: inspect.resolved_path }
              : { ok: false, path: inspectPath, inspectPath, gitPath: false };
            if (result.ok) validationCache.set(validationKey(inspectPath, false), result);
            results.set(inspectPath, result);
          }
        } catch {
          for (const inspectPath of chunk) results.set(inspectPath, { ok: false, path: inspectPath, inspectPath, gitPath: false });
        }
      }
      return results;
    }

    async function equivalentInspection(entries, rawPath) {
      if (!entriesMayReferToSamePath(entries)) return null;
      const inspected = [];
      for (const entry of entries) {
        const result = await inspectCandidate(entry, rawPath);
        if (!result || !result.ok || !result.resolvedPath) return null;
        inspected.push(result);
      }
      const resolved = new Set(inspected.map((result) => String(result.resolvedPath || "")));
      if (resolved.size !== 1) return null;
      return inspected.find((result) => !result.gitPath) || inspected[0] || null;
    }

    async function inspectionPlan(path) {
      const rawPath = String(path ?? "");
      if (rawPath === "") return { invalid: true };
      let entry = { path: rawPath, gitPath: false, apiPath: "" };
      if (!rawPath.includes("/") && sessionState.get("selected")) {
        const candidates = await getKnownCandidates();
        const matches = exactBareMatches(candidates, rawPath);
        const searched = matches.length > 1 && !entriesMayReferToSamePath(matches) ? { matches: [], truncated: false } : await searchBareCandidates(rawPath);
        const merged = exactBareMatches([...matches, ...searched.matches], rawPath);
        if (merged.length === 1 && !searched.truncated) entry = merged[0];
        else if (merged.length > 1 && !searched.truncated) return { rawPath, equivalentEntries: merged };
        else if (searched.truncated) return { rawPath, ambiguous: true };
      }
      return { rawPath, entry };
    }

    async function inspectPath(path) {
      const plan = await inspectionPlan(path);
      if (plan.invalid) return { ok: false };
      if (plan.ambiguous) return { ok: false, ambiguous: true, path: plan.rawPath };
      if (plan.equivalentEntries) {
        const equivalent = await equivalentInspection(plan.equivalentEntries, plan.rawPath);
        if (equivalent) return equivalent;
        return { ok: false, ambiguous: true, path: plan.rawPath };
      }
      return await inspectCandidate(plan.entry, plan.rawPath);
    }

    function replaceAmbiguousNode(node, path, line = null) {
      const query = String(path ?? "");
      if (!node || query === "") return false;
      const link = el("a", {
        href: "#",
        class: "inlineFileLink inlineFileAmbiguousRef",
        "data-file-picker-query": query,
        title: `Choose which ${query} to open`,
      });
      if (line) link.setAttribute("data-file-line", String(line));
      link.appendChild(el("span", { text: node.textContent || query }));
      link.appendChild(el("span", { class: "inlineFileChoiceHint", text: "choose" }));
      node.replaceWith(link);
      return true;
    }

    async function upgradeCandidateRefs(root) {
      if (!root) return false;
      const nodes = Array.from(root.querySelectorAll("[data-candidate-file-path]"));
      const plans = await Promise.all(
        nodes.map(async (node) => {
          const path = String(node.getAttribute("data-candidate-file-path") ?? "");
          return { node, line: normalizeLineNumber(node.getAttribute("data-candidate-file-line")), plan: await inspectionPlan(path) };
        })
      );
      const batchPaths = plans
        .map(({ plan }) => plan.entry && normalizeCandidate(plan.entry))
        .filter((entry) => entry && !entry.gitPath && !entry.apiPath)
        .map((entry) => entry.path);
      const directInspections = await inspectPlainCandidates(batchPaths);
      for (const { node, line, plan } of plans) {
        if (plan.invalid) continue;
        if (plan.ambiguous) {
          replaceAmbiguousNode(node, plan.rawPath, line);
          continue;
        }
        let result;
        if (plan.equivalentEntries) result = await equivalentInspection(plan.equivalentEntries, plan.rawPath);
        else {
          const entry = normalizeCandidate(plan.entry);
          const direct = entry && !entry.gitPath && !entry.apiPath ? directInspections.get(entry.path) : null;
          result = direct ? { ...direct, path: plan.rawPath } : await inspectCandidate(entry, plan.rawPath);
        }
        if (!result || !result.ok) {
          if (plan.equivalentEntries) replaceAmbiguousNode(node, plan.rawPath, line);
          continue;
        }
        const resolvedPath = String(result.resolvedPath || result.inspectPath || plan.rawPath);
        const link = el("a", {
          href: "#",
          class: "inlineFileLink",
          "data-file-path": resolvedPath,
          "data-file-kind": result.kind || "text",
        });
        if (line && result.kind !== "directory") link.setAttribute("data-file-line", String(line));
        link.textContent = node.textContent || plan.rawPath;
        node.replaceWith(link);
      }
      return true;
    }

    async function openAmbiguousChoice(query, line = null) {
      const rawQuery = String(query ?? "");
      if (rawQuery === "") return false;
      if (!sessionState.get("selected")) {
        requireFunction(options.setToast, "setToast")("select a session first");
        return false;
      }
      await requireFunction(options.showFileViewer, "showFileViewer")({ pickerQuery: rawQuery, line });
      return true;
    }

    async function openReference(ref) {
      if (!ref || typeof ref.path !== "string") return false;
      const rawPath = String(ref.path ?? "");
      const line = normalizeLineNumber(ref.line);
      if (rawPath === "") return false;
      const parsed = ref.literal ? { path: rawPath, line } : requireFunction(options.parseLocalFileRef, "parseLocalFileRef")(rawPath);
      const setToast = requireFunction(options.setToast, "setToast");
      if (!parsed) {
        setToast("unsupported file reference");
        return false;
      }
      const showFileViewer = requireFunction(options.showFileViewer, "showFileViewer");
      if (!parsed.path.startsWith("/")) {
        if (!sessionState.get("selected")) {
          setToast("select a session first");
          return false;
        }
        await showFileViewer({ path: parsed.path, mode: "file", manual: false, line });
        return true;
      }
      if (sessionState.get("selected")) {
        await showFileViewer({ path: parsed.path, mode: "file", manual: false, line });
        return true;
      }
      const currentRel = sessionRelativePath(parsed.path);
      if (currentRel) {
        await showFileViewer({ path: currentRel, mode: "file", manual: false, line });
        return true;
      }
      const sessions = requireFunction(options.sessions, "sessions")();
      const match = (Array.isArray(sessions) ? sessions : []).find((session) => {
        const cwd = String(session && session.cwd ? session.cwd : "").replace(/\/+$/, "");
        return cwd && (parsed.path === cwd || parsed.path.startsWith(`${cwd}/`));
      });
      if (!match) {
        setToast("file is outside the known session roots");
        return false;
      }
      await requireFunction(options.selectSession, "selectSession")(match.session_id);
      const matchRoot = String(match.cwd || "").replace(/\/+$/, "");
      const rel = parsed.path === matchRoot ? "." : parsed.path.slice(matchRoot.length + 1);
      await showFileViewer({ path: rel, mode: "file", manual: false, line });
      return true;
    }

    async function openDirectoryReference(rawPath) {
      const cwd = String(rawPath || "").trim();
      if (!cwd) return false;
      requireFunction(options.openDirectorySession, "openDirectorySession")({
        cwd,
        statusText: "Review resume or worktree options, then start the session.",
      });
      return true;
    }

    async function handleClick(event) {
      const source = event && (!ElementCtor || event.target instanceof ElementCtor) ? event.target : null;
      if (!source || typeof source.closest !== "function") return false;
      const choice = source.closest("a[data-file-picker-query]");
      if (choice) {
        if (typeof event.preventDefault === "function") event.preventDefault();
        const query = String(choice.getAttribute("data-file-picker-query") ?? "");
        const line = normalizeLineNumber(choice.getAttribute("data-file-line"));
        await openAmbiguousChoice(query, line);
        return true;
      }
      const target = source.closest("a[data-file-path]");
      if (!target) return false;
      if (typeof event.preventDefault === "function") event.preventDefault();
      const path = String(target.getAttribute("data-file-path") ?? "");
      const kind = String(target.getAttribute("data-file-kind") || "").trim();
      const line = normalizeLineNumber(target.getAttribute("data-file-line"));
      if (kind === "directory") {
        await openDirectoryReference(path);
        return true;
      }
      await openReference({ path, line, literal: true });
      return true;
    }

    return Object.freeze({
      clearDiscoveryCaches,
      collectMessageFileRefs,
      exactBareMatches,
      getKnownCandidates,
      handleClick,
      inspectPath,
      openAmbiguousChoice,
      openDirectoryReference,
      openReference,
      replaceAmbiguousNode,
      upgradeCandidateRefs,
    });
  }

  function createOpenedFileRuntime(options = {}) {
    const currentSessionId = requireFunction(options.currentSessionId, "currentSessionId");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("file viewer dependency missing: sessionState");
    const sessionRelativePath = requireFunction(options.sessionRelativePath, "sessionRelativePath");
    const activeIdentity = requireFunction(options.activeIdentity, "activeIdentity");
    const fileEntryForPath = requireFunction(options.fileEntryForPath, "fileEntryForPath");
    const upsertFileEntry = requireFunction(options.upsertFileEntry, "upsertFileEntry");
    const sessionById = requireFunction(options.sessionById, "sessionById");
    const listFromFilesField = requireFunction(options.listFromFilesField, "listFromFilesField");
    const listFromFileRecords = requireFunction(options.listFromFileRecords, "listFromFileRecords");
    const deleteCandidateCache = requireFunction(options.deleteCandidateCache, "deleteCandidateCache");

    function historySelection(sessionId) {
      const sid = String(sessionId || "").trim();
      if (!sid) return { path: "", line: null, gitPath: false, apiPath: "" };
      const session = sessionById(sid);
      if (!session) return { path: "", line: null, gitPath: false, apiPath: "" };
      // Rehydrate the most recent recorded file WITH its reversible token so a
      // raw-byte (non-UTF) filename is reopened through the token channel
      // instead of being reduced to an un-openable display string.
      for (const record of listFromFileRecords(session.files)) {
        const rel = sessionRelativePath(record.path, sid);
        if (typeof rel === "string" && rel && rel !== ".") {
          return { path: rel, line: null, gitPath: false, apiPath: record.apiPath || "" };
        }
      }
      return { path: "", line: null, gitPath: false, apiPath: "" };
    }

    function remember(relPath, absPath = null) {
      const raw = String(relPath ?? "");
      const sid = currentSessionId() || sessionState.get("selected") || "";
      const rel = sessionRelativePath(raw, sid) || raw;
      if (!rel) return false;
      const identity = activeIdentity();
      const gitPath = Boolean(identity && identity.gitPath);
      const apiPath = identity && identity.apiPath ? identity.apiPath : "";
      const current = fileEntryForPath(rel, gitPath, apiPath);
      upsertFileEntry({
        path: rel,
        apiPath,
        gitPath,
        additions: current && current.changed ? current.additions : null,
        deletions: current && current.changed ? current.deletions : null,
        changed: Boolean(current && current.changed),
        source: current && current.changed ? "changed" : "recent",
      });
      const session = sid ? sessionById(sid) : null;
      if (!session) return false;
      // Preserve the reversible token when remembering a raw-byte file: write a
      // structured { path, apiPath } record so historySelection can reopen it.
      // Token-less entries keep the legacy plain-string shape. MRU dedup is by
      // display path (matching the prior string-comparison behavior): opening a
      // file evicts any prior entry with the same display path, regardless of
      // token, so the most recent open wins.
      const records = listFromFileRecords(session.files);
      const abs = typeof absPath === "string" && absPath !== ""
        ? absPath
        : session.cwd && rel !== "."
          ? `${String(session.cwd).replace(/\/+$/, "")}/${rel.replace(/^\.?\//, "")}`
          : "";
      if (!abs) return false;
      const nextFiles = [
        apiPath ? { path: abs, apiPath } : abs,
        ...records.filter((record) => record.path !== abs)
          .map((record) => (record.apiPath ? { path: record.path, apiPath: record.apiPath } : record.path)),
      ];
      session.files = nextFiles;
      deleteCandidateCache(sid);
      return true;
    }

    return Object.freeze({ historySelection, remember });
  }

  function timeoutPromise(promise, timeoutMs, message) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      promise.then(
        (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        (error) => {
          clearTimeout(timer);
          reject(error);
        }
      );
    });
  }

  function createPdfLoader(options = {}) {
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const globalObject = options.globalObject || window;
    const timeoutMs = Math.max(1, Number(options.timeoutMs || 6000));
    const importModule = typeof options.importModule === "function" ? options.importModule : (url) => import(url);
    let readyPromise = null;

    // pdf.mjs / pdf.worker.mjs carry no content hash of their own; append the
    // deployed asset version so they use the immutable static cache instead of
    // revalidating (and, before ETag support, re-downloading) on every load.
    const assetVersion = typeof globalObject.CODOXEAR_ASSET_VERSION === "string" ? globalObject.CODOXEAR_ASSET_VERSION.trim() : "";
    const versionedUrl = (path) => {
      const url = resolveAppUrl(path);
      if (!assetVersion) return url;
      return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(assetVersion)}`;
    };

    function ensure() {
      if (readyPromise) return readyPromise;
      if (globalObject.pdfjsLib && typeof globalObject.pdfjsLib.getDocument === "function") {
        readyPromise = Promise.resolve(globalObject.pdfjsLib);
      } else {
        readyPromise = timeoutPromise(importModule(versionedUrl("pdf.mjs")), timeoutMs, "PDF renderer timed out");
      }
      readyPromise = readyPromise.then((pdfjs) => {
        if (pdfjs && pdfjs.GlobalWorkerOptions) pdfjs.GlobalWorkerOptions.workerSrc = versionedUrl("pdf.worker.mjs");
        return pdfjs;
      });
      readyPromise.catch(() => {
        readyPromise = null;
      });
      return readyPromise;
    }

    return Object.freeze({ ensure });
  }

export { bindFileTouchClick, bindFileTouchPress, createFileDownloadRuntime, createFileFallbackRuntime, createFileInspectRuntime, createFileLoadResultRuntime, createFileCandidateRefreshRuntime, createFileVideoPreviewRuntime, createFileViewerPanelRuntime, createFileViewerLifecycleRuntime, createFileModeControlsRuntime, createFilePasteDialogRuntime, createFilePdfRenderRuntime, createFileViewerModalRuntime, createFileReferenceRuntime, createFileRenderSurfaceRuntime, createOpenedFileRuntime, createFileTouchToolbarRuntime, createFileUnsavedDialogRuntime, createFileViewerController, createPdfLoader };
