

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


  function createFileCandidateRefreshRuntime(options = {}) {
    const controller = options.controller || null;
    if (!controller) throw new TypeError("file viewer dependency missing: controller");
    const beginRefresh = requireFunction(controller.beginFileCandidateRefresh, "controller.beginFileCandidateRefresh").bind(controller);
    const isCurrentRefresh = requireFunction(controller.isCurrentFileCandidateRefresh, "controller.isCurrentFileCandidateRefresh").bind(controller);
    const clearRefreshEntries = requireFunction(controller.clearFileCandidateRefreshEntries, "controller.clearFileCandidateRefreshEntries").bind(controller);
    const applyRefreshEntries = requireFunction(controller.applyFileCandidateRefreshEntries, "controller.applyFileCandidateRefreshEntries").bind(controller);
    const setGitStateMessage = requireFunction(controller.setFileCandidateGitStateMessage, "controller.setFileCandidateGitStateMessage").bind(controller);
    const applyFreshCache = requireFunction(controller.applyFreshFileCandidateCache, "controller.applyFreshFileCandidateCache").bind(controller);
    const fileCandidateKeyForEntry = requireFunction(controller.fileCandidateKeyForEntry, "controller.fileCandidateKeyForEntry").bind(controller);
    const rememberCandidateCache = requireFunction(controller.rememberFileCandidateCache, "controller.rememberFileCandidateCache").bind(controller);
    const currentSessionId = requireFunction(options.currentSessionId, "currentSessionId");
    const selectedSessionId = requireFunction(options.selectedSessionId, "selectedSessionId");
    const blockUnavailableFileAction = requireFunction(options.blockUnavailableFileAction, "blockUnavailableFileAction");
    const isSessionCurrent = requireFunction(options.isSessionCurrent, "isSessionCurrent");
    const collectMessageFileRefs = requireFunction(options.collectMessageFileRefs, "collectMessageFileRefs");
    const sessionFiles = requireFunction(options.sessionFiles, "sessionFiles");
    const sessionFileRecords = requireFunction(options.sessionFileRecords, "sessionFileRecords");
    const sessionRelativePath = requireFunction(options.sessionRelativePath, "sessionRelativePath");
    const api = requireFunction(options.api, "api");
    const normalizeFileApiPath = requireFunction(options.normalizeFileApiPath, "normalizeFileApiPath");
    const renderMenu = requireFunction(options.renderMenu, "renderMenu");
    const nowMs = typeof options.nowMs === "function" ? options.nowMs : () => Date.now();
    const ttlMs = Number(options.ttlMs || 0);

    function mentionedEntry(path) {
      return { path, additions: null, deletions: null, changed: false, gitPath: false, source: "mentioned" };
    }

    function recentEntry(path, apiPath = "") {
      return { path, additions: null, deletions: null, changed: false, gitPath: false, apiPath: normalizeFileApiPath(apiPath), source: "recent" };
    }

    function normalizeChangedEntry(entry) {
      if (!entry || typeof entry.path !== "string" || entry.path === "") return null;
      const untracked = Boolean(entry.untracked || entry.state === "untracked");
      const oldPath = typeof entry.old_path === "string" && entry.old_path !== "" ? entry.old_path : "";
      const rename = Boolean(entry.rename || oldPath);
      return {
        path: entry.path,
        apiPath: normalizeFileApiPath(entry.api_path || entry.apiPath),
        additions: typeof entry.additions === "number" && Number.isFinite(entry.additions) ? entry.additions : null,
        deletions: typeof entry.deletions === "number" && Number.isFinite(entry.deletions) ? entry.deletions : null,
        changed: untracked ? false : true,
        untracked,
        rename,
        oldPath,
        gitPath: true,
        source: "changed",
      };
    }

    function mergeCandidateEntries(baseEntries, messageEntries, manualEntries) {
      const merged = [];
      const seen = new Set();
      for (const entry of [...baseEntries, ...messageEntries, ...manualEntries]) {
        if (!entry || entry.path === "") continue;
        const key = fileCandidateKeyForEntry(entry);
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(entry);
      }
      return merged;
    }

    function manualEntriesForSession(sid) {
      // sessionFileRecords preserves the reversible api_path token for raw-byte
      // (non-UTF) recent files so they can be reopened from the picker instead
      // of being reduced to an un-openable JSON-safe display string.
      return sessionFileRecords(sid)
        .map((record) => ({ rel: sessionRelativePath(record.path, sid), apiPath: record.apiPath || "" }))
        .filter((item) => typeof item.rel === "string" && item.rel && item.rel !== ".")
        .map((item) => recentEntry(item.rel, item.apiPath));
    }

    function candidateCacheKey(sid) {
      const filesKey = JSON.stringify(sessionFiles(sid));
      const refsKey = JSON.stringify(collectMessageFileRefs());
      return `${sid || ""}\u0000${filesKey}\u0000${refsKey}`;
    }

    async function refresh({ force = false, sessionId = null, syncToken = null } = {}) {
      const explicitSession = sessionId !== null && sessionId !== undefined && String(sessionId || "").trim() !== "";
      if (!explicitSession && blockUnavailableFileAction()) return false;
      const sid = String(sessionId || currentSessionId() || selectedSessionId() || "").trim();
      const requestSeq = beginRefresh();
      const current = () => isCurrentRefresh(requestSeq) && (!explicitSession || isSessionCurrent(sid, syncToken));
      if (!sid) {
        if (!current()) return false;
        clearRefreshEntries();
        return true;
      }
      const cacheKey = candidateCacheKey(sid);
      if (!force && current() && applyFreshCache(sid, cacheKey, { now: nowMs(), ttl: ttlMs })) {
        renderMenu();
        return true;
      }
      if (!current()) return false;
      clearRefreshEntries();
      const messageEntries = collectMessageFileRefs().map(mentionedEntry);
      const manualEntries = manualEntriesForSession(sid);
      const fallbackEntries = mergeCandidateEntries([], messageEntries, manualEntries);
      let renderedFallback = false;
      if (fallbackEntries.length) {
        if (!current()) return false;
        applyRefreshEntries(fallbackEntries, { gitStateFresh: false });
        renderMenu();
        renderedFallback = true;
      }
      let changedEntries = [];
      let changedEntriesFresh = false;
      let gitStateMessage = "";
      try {
        const res = await api(`/api/sessions/${sid}/git/changed_files`);
        const entriesIn = Array.isArray(res.entries) ? res.entries : [];
        changedEntries = entriesIn.map(normalizeChangedEntry).filter(Boolean);
        changedEntriesFresh = true;
      } catch (error) {
        // A non-repo / git error must be surfaced explicitly instead of leaving
        // the user with a silently empty changed-files list. Other failures
        // (transient network, auth) still fall back silently to mentioned/recent
        // entries below, matching the historical behaviour.
        const message = String((error && error.message) || "");
        // Precisely match git's non-repo fatal so a transient 409 (e.g. "git
        // changed during refresh") is not misreported as a non-repo cwd.
        const isNonRepo = /not a git repository/i.test(message);
        if (isNonRepo) gitStateMessage = "Not a git repository \u2014 no changed files";
      }
      if (!changedEntriesFresh && renderedFallback) {
        if (gitStateMessage) setGitStateMessage(gitStateMessage);
        return true;
      }
      const merged = mergeCandidateEntries(changedEntries, messageEntries, manualEntries);
      if (!current()) return false;
      applyRefreshEntries(merged, { gitStateFresh: changedEntriesFresh, gitStateMessage });
      if (changedEntriesFresh) rememberCandidateCache(sid, cacheKey, nowMs());
      if (!current()) return false;
      renderMenu();
      return true;
    }

    return Object.freeze({ refresh });
  }

export { createFileCandidateRefreshRuntime };
