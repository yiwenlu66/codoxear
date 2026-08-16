
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function createFileCandidateStateRuntime(options = {}) {
    const normalizeFileApiPath = requireFunction(options.normalizeFileApiPath, "normalizeFileApiPath");
    const normalizeLineNumber = requireFunction(options.normalizeLineNumber, "normalizeLineNumber");
    const normalizeSessionId = requireFunction(options.normalizeSessionId, "normalizeSessionId");
    const preferredFileSelectionForSession = requireFunction(options.preferredFileSelectionForSession, "preferredFileSelectionForSession");
    const currentActiveFileIdentity = requireFunction(options.currentActiveFileIdentity, "currentActiveFileIdentity");
    const applyFileMode = requireFunction(options.applyFileMode, "applyFileMode");
    let fileCandidateRequestSeq = 0;
    let fileCandidateList = [];
    let fileEntryMap = new Map();
    let fileCandidateGitStateFresh = false;
    let fileCandidateGitStateMessage = "";
    let fileCandidateCache = new Map();
    let fileViewerSessionId = "";
    let fileViewerSessionSyncToken = 0;

    function fileCandidateKey(path, gitPath = false, apiPath = "") {
      const token = normalizeFileApiPath(apiPath);
      const identity = token || String(path ?? "");
      return `${gitPath ? "git" : "session"}\u0000${identity}`;
    }

    function normalizeFileCandidateSource(source) {
      const value = String(source || "").trim();
      if (value === "changed" || value === "mentioned" || value === "recent") return value;
      return "";
    }

    function cloneFileCandidateEntry(entry) {
      if (!entry || typeof entry.path !== "string" || entry.path === "") return null;
      const source = normalizeFileCandidateSource(entry.source);
      const gitPath = entry.gitPath === undefined ? Boolean(entry.changed && source === "changed") : Boolean(entry.gitPath);
      const apiPath = normalizeFileApiPath(entry.apiPath || entry.api_path);
      const untracked = Boolean(entry.untracked);
      const oldPath = typeof entry.oldPath === "string" ? entry.oldPath : "";
      const rename = Boolean(entry.rename || oldPath);
      return {
        path: entry.path,
        apiPath,
        gitPath,
        key: fileCandidateKey(entry.path, gitPath, apiPath),
        additions: entry.additions ?? null,
        deletions: entry.deletions ?? null,
        changed: untracked ? false : Boolean(entry.changed),
        untracked,
        rename,
        oldPath,
        source,
      };
    }

    function fileCandidateKeyForEntry(entry) {
      return fileCandidateKey(entry && entry.path, Boolean(entry && entry.gitPath), normalizeFileApiPath(entry && entry.apiPath));
    }

    function applyFileCandidateEntries(entries) {
      const nextList = [];
      const nextMap = new Map();
      for (const raw of Array.isArray(entries) ? entries : []) {
        const entry = cloneFileCandidateEntry(raw);
        if (!entry || nextMap.has(entry.key)) continue;
        nextList.push(entry.key);
        nextMap.set(entry.key, entry);
      }
      fileCandidateList = nextList;
      fileEntryMap = nextMap;
    }

    function currentFileCandidateKeys() {
      return fileCandidateList.slice();
    }

    function currentFileCandidateEntries() {
      return fileCandidateList.map((key) => cloneFileCandidateEntry(fileEntryMap.get(key))).filter(Boolean);
    }

    function fileEntryForKey(key) {
      return cloneFileCandidateEntry(fileEntryMap.get(String(key || "")));
    }

    function fileEntryForPath(path, gitPath = false, apiPath = "") {
      const token = normalizeFileApiPath(apiPath);
      const preferred = fileEntryMap.get(fileCandidateKey(path, gitPath, token));
      if (preferred) return cloneFileCandidateEntry(preferred);
      const fallback = fileEntryMap.get(fileCandidateKey(path, gitPath));
      if (fallback && (!token || normalizeFileApiPath(fallback.apiPath) === token)) return cloneFileCandidateEntry(fallback);
      for (const key of fileCandidateList) {
        const entry = fileEntryMap.get(key);
        if (!entry || entry.path !== path || Boolean(entry.gitPath) !== Boolean(gitPath)) continue;
        if (!token || normalizeFileApiPath(entry.apiPath) === token) return cloneFileCandidateEntry(entry);
      }
      return null;
    }

    function fileApiPathForPath(path, apiPath = "") {
      const existing = normalizeFileApiPath(apiPath);
      if (existing) return existing;
      const entry = fileEntryForPath(path, true) || fileEntryForPath(path, false);
      return normalizeFileApiPath(entry && entry.apiPath);
    }

    function activeFileEntry() {
      const identity = currentActiveFileIdentity();
      if (!identity.path) return null;
      return fileEntryForPath(identity.path, identity.gitPath, identity.apiPath);
    }

    function isGitFileCandidatePath(path, changed = null, gitPath = null, apiPath = "") {
      if (gitPath !== null && gitPath !== undefined) return Boolean(gitPath);
      if (changed !== null && changed !== undefined) return Boolean(changed);
      const gitEntry = fileEntryForPath(path, true, normalizeFileApiPath(apiPath));
      if (gitEntry) return true;
      const sessionEntry = fileEntryForPath(path, false);
      return Boolean(sessionEntry && sessionEntry.gitPath);
    }

    function currentFileCandidateGitStateFresh() { return fileCandidateGitStateFresh; }
    function setFileCandidateGitStateFresh(fresh) { fileCandidateGitStateFresh = Boolean(fresh); return fileCandidateGitStateFresh; }
    function currentFileCandidateGitStateMessage() { return fileCandidateGitStateMessage; }
    function setFileCandidateGitStateMessage(message) { fileCandidateGitStateMessage = String(message || ""); return fileCandidateGitStateMessage; }
    function clearFileCandidateGitStateMessage() { fileCandidateGitStateMessage = ""; return fileCandidateGitStateMessage; }

    function rememberFileCandidateCache(sessionId, key, now = Date.now()) {
      const sid = String(sessionId || "").trim();
      if (!sid || !key) return false;
      fileCandidateCache.set(sid, { key, ts: Number(now || 0), entries: currentFileCandidateEntries() });
      return true;
    }

    function fileCandidateCacheEntry(sessionId) {
      const sid = String(sessionId || "").trim();
      const cached = sid ? fileCandidateCache.get(sid) : null;
      if (!cached || typeof cached !== "object") return null;
      return Object.freeze({ key: String(cached.key || ""), ts: Number(cached.ts || 0), entries: Array.isArray(cached.entries) ? cached.entries.map(cloneFileCandidateEntry).filter(Boolean) : [] });
    }

    function deleteFileCandidateCache(sessionId) { const sid = String(sessionId || "").trim(); return sid ? fileCandidateCache.delete(sid) : false; }
    function fileCandidateCacheSize() { return fileCandidateCache.size; }

    function applyFileCandidateRefreshEntries(entries, { gitStateFresh = false, gitStateMessage = "" } = {}) {
      applyFileCandidateEntries(entries);
      setFileCandidateGitStateFresh(gitStateFresh);
      fileCandidateGitStateMessage = gitStateFresh ? "" : String(gitStateMessage || "");
      applyFileMode();
      return true;
    }

    function clearFileCandidateRefreshEntries() { return applyFileCandidateRefreshEntries([], { gitStateFresh: false }); }

    function applyFreshFileCandidateCache(sid, key, { now = Date.now(), ttl = 0 } = {}) {
      const cached = fileCandidateCacheEntry(sid);
      if (!cached || cached.key !== key) return false;
      const age = Number(now || 0) - Number(cached.ts || 0);
      if (!(age >= 0 && age < Number(ttl || 0))) return false;
      return applyFileCandidateRefreshEntries(cached.entries, { gitStateFresh: false });
    }

    function upsertFileEntry(entry) {
      const merged = cloneFileCandidateEntry(entry);
      if (!merged) return false;
      const current = fileEntryMap.get(merged.key);
      const next = current && !merged.source ? { ...merged, source: normalizeFileCandidateSource(current.source) } : merged;
      if (!fileEntryMap.has(next.key)) fileCandidateList.push(next.key);
      fileEntryMap.set(next.key, Object.freeze({ ...next }));
      return true;
    }

    function pickerEntryForKey(key, { score = 0 } = {}) { const entry = fileEntryForKey(key); return entry ? { ...entry, added: true, score } : null; }

    function pickerEntryForPath(path, { score = 0, gitPath = false, apiPath = "" } = {}) {
      const token = normalizeFileApiPath(apiPath);
      const key = fileCandidateKey(path, gitPath, token);
      const existing = fileEntryMap.get(key);
      const baseEntry = existing || { path, gitPath, additions: null, deletions: null, changed: false, source: "" };
      const entry = cloneFileCandidateEntry({ ...baseEntry, apiPath: token });
      return entry ? { ...entry, added: Boolean(existing), score } : null;
    }

    function resolveFileViewerOpenTarget({ sessionId = "", explicitPath = "", explicitLine = null } = {}) {
      const sid = String(sessionId || "").trim();
      if (!sid) return Object.freeze({ kind: "none" });
      const requestedPath = String(explicitPath ?? "");
      if (requestedPath) return Object.freeze({ kind: "path", source: "explicit", path: requestedPath, line: normalizeLineNumber(explicitLine), changed: null, gitPath: false, apiPath: "" });
      const preferred = preferredFileSelectionForSession(sid);
      if (preferred.path) return Object.freeze({ kind: "path", source: "preferred", path: preferred.path, line: preferred.line, changed: null, gitPath: Boolean(preferred.gitPath), apiPath: normalizeFileApiPath(preferred.apiPath) });
      const firstKey = fileCandidateList.length ? fileCandidateList[0] : "";
      const first = firstKey ? fileEntryForKey(firstKey) : null;
      return first ? Object.freeze({ kind: "path", source: "first", path: first.path, line: null, changed: Boolean(first.changed), gitPath: Boolean(first.gitPath), apiPath: normalizeFileApiPath(first.apiPath) }) : Object.freeze({ kind: "none" });
    }

    function currentFileViewerSessionId() { return normalizeSessionId(fileViewerSessionId); }
    function setFileViewerSessionId(sessionId) { fileViewerSessionId = normalizeSessionId(sessionId); return fileViewerSessionId; }
    function clearFileViewerSessionId() { fileViewerSessionId = ""; }
    function beginFileViewerSessionSync() { fileViewerSessionSyncToken += 1; return fileViewerSessionSyncToken; }
    function invalidateFileViewerSessionSync() { fileViewerSessionSyncToken += 1; return fileViewerSessionSyncToken; }
    function isCurrentFileViewerSessionSync(token) { return token === fileViewerSessionSyncToken; }
    function beginFileCandidateRefresh() { fileCandidateRequestSeq += 1; return fileCandidateRequestSeq; }
    function isCurrentFileCandidateRefresh(requestSeq) { return requestSeq === fileCandidateRequestSeq; }

    return Object.freeze({
      fileCandidateKey, fileCandidateKeyForEntry, cloneFileCandidateEntry, applyFileCandidateEntries,
      currentFileCandidateKeys, currentFileCandidateEntries, fileEntryForKey, fileEntryForPath,
      fileApiPathForPath, activeFileEntry, isGitFileCandidatePath, currentFileCandidateGitStateFresh,
      setFileCandidateGitStateFresh, currentFileCandidateGitStateMessage, setFileCandidateGitStateMessage,
      clearFileCandidateGitStateMessage, rememberFileCandidateCache, fileCandidateCacheEntry,
      deleteFileCandidateCache, fileCandidateCacheSize, applyFileCandidateRefreshEntries,
      clearFileCandidateRefreshEntries, applyFreshFileCandidateCache, upsertFileEntry, pickerEntryForKey,
      pickerEntryForPath, resolveFileViewerOpenTarget, currentFileViewerSessionId, setFileViewerSessionId,
      clearFileViewerSessionId, beginFileViewerSessionSync, invalidateFileViewerSessionSync,
      isCurrentFileViewerSessionSync, beginFileCandidateRefresh, isCurrentFileCandidateRefresh,
    });
  }

export { createFileCandidateStateRuntime };
