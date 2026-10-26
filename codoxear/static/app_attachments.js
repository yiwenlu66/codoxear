
  // Attachment staging authority. Owns the selected session's local staged-list
  // projection, upload producers (picker/paste/drop), image compression, and
  // all attachment-specific DOM/event state. The server's staged list remains
  // authoritative; direct mutation responses update this projection until the
  // next revision-aware catalog update replaces it.
  //
  // Pending uploads are local-only optimistic tray entries created
  // synchronously when files are staged, so pasting/dropping/picking gives
  // immediate visual feedback (thumbnail + phase) before the server
  // round-trip completes. They never merge into stagedAttachments.
  // Reconciliation: when a file's inject_file response arrives, its pending
  // entry is removed and the response's authoritative list commits in the
  // same tick, so the attach badge (staged count + active pending count)
  // never double-counts. A successful handoff moves the entry's object-URL
  // thumbnail into localThumbs keyed by the server attachment id (the server
  // has no attachment-bytes endpoint, so thumbnails are page-lifetime local);
  // renderStagedAttachments prunes any localThumbs id absent from the current
  // staged list, covering remove, clear, and session switch through one
  // funnel. Pending uploads are selected-session scoped: a selected change
  // aborts and clears them.

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`attachments controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value.addEventListener !== "function")
      throw new TypeError(`attachments controller dependency missing: ${name}`);
    return value;
  }

  function createAttachmentsController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("attachments controller dependency missing: options");
    const attachBtn = requireNode(options.attachBtn, "attachBtn");
    const imgInput = requireNode(options.imgInput, "imgInput");
    const composer = requireNode(options.composer, "composer");
    const textarea = requireNode(options.textarea, "textarea");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") {
      throw new TypeError("attachments controller dependency missing: sessionState");
    }
    const sessionCatalog = options.sessionCatalog;
    if (
      !sessionCatalog ||
      typeof sessionCatalog.get !== "function" ||
      typeof sessionCatalog.patchSession !== "function" ||
      typeof sessionCatalog.subscribe !== "function"
    ) throw new TypeError("attachments controller dependency missing: sessionCatalog");
    const getSessionInfo = (sessionId) => sessionCatalog.get("sessionIndex").get(sessionId) || null;

    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const sessionHasUnknownSend = requireFunction(options.sessionHasUnknownSend, "sessionHasUnknownSend");
    const sessionIsOrphanRecovery = requireFunction(options.sessionIsOrphanRecovery, "sessionIsOrphanRecovery");
    const sessionHasOrphanQueueRecovery = requireFunction(options.sessionHasOrphanQueueRecovery, "sessionHasOrphanQueueRecovery");
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const setPollFastUntilMs = requireFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const kickPoll = requireFunction(options.kickPoll, "kickPoll");
    const resizeComposer = requireFunction(options.resizeComposer, "resizeComposer");
    const getTray = requireFunction(options.getTray, "getTray");
    const el = requireFunction(options.el, "el");
    const fmtBytes = requireFunction(options.fmtBytes, "fmtBytes");
    const safeAttachmentStem = requireFunction(options.safeAttachmentStem, "safeAttachmentStem");
    const isLikelyHeic = requireFunction(options.isLikelyHeic, "isLikelyHeic");
    const looksLikeImage = requireFunction(options.looksLikeImage, "looksLikeImage");
    requireFunction(options.b64FromBytes, "b64FromBytes"); // Legacy DI contract; binary uploads need no base64 copy.
    const dataTransferHasFiles = requireFunction(options.dataTransferHasFiles, "dataTransferHasFiles");
    const extractFilesFromClipboardData = requireFunction(options.extractFilesFromClipboardData, "extractFilesFromClipboardData");
    const extractFilesFromDropData = requireFunction(options.extractFilesFromDropData, "extractFilesFromDropData");
    const addEventListener = requireFunction(options.addEventListener, "addEventListener");
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const uploadMaxBytes = Number(options.uploadMaxBytes);
    if (!Number.isFinite(uploadMaxBytes) || uploadMaxBytes <= 0)
      throw new TypeError("attachments controller dependency missing: uploadMaxBytes");

    let stagedAttachments = [];
    let pendingUploads = [];
    const localThumbs = new Map();
    const freshIds = new Set();
    let composerDragDepth = 0;
    const globalSetTimeout = typeof setTimeout === "function" ? setTimeout : null;
    const AbortCtor = typeof AbortController === "function" ? AbortController : null;
    const objectUrlApi =
      typeof URL !== "undefined" && URL && typeof URL.createObjectURL === "function" ? URL : null;
    const attachBadgeEl = el("span", { class: "attachBadge", id: "attachBadge" });
    attachBtn.appendChild(attachBadgeEl);

    function normalizedStagedAttachments(list) {
      if (!Array.isArray(list)) return [];
      const seen = new Set();
      return list
        .filter((item) => {
          if (!item || typeof item !== "object" || typeof item.id !== "string" || !item.id || seen.has(item.id)) return false;
          seen.add(item.id);
          return true;
        })
        .map((item) => ({
          id: String(item.id),
          display_name: String(item.display_name || item.filename || "file"),
          filename: String(item.filename || item.display_name || "file"),
          size: Number.isFinite(Number(item.size)) ? Number(item.size) : 0,
          created_ts: Number.isFinite(Number(item.created_ts)) ? Number(item.created_ts) : 0,
        }));
    }

    function attachmentIdentityText(item) {
      const name = item && (item.display_name || item.filename) ? String(item.display_name || item.filename) : "staged attachment";
      const id = item && item.id ? String(item.id).slice(0, 8) : "";
      const size = item && Number.isFinite(Number(item.size)) ? fmtBytes(Number(item.size)) : "0 B";
      return id ? `${name} · ${size} · attachment ${id}` : `${name} · ${size}`;
    }

    function middleEllipsis(text, limit = 44) {
      const value = String(text || "");
      if (value.length <= limit) return value;
      const left = Math.ceil((limit - 1) / 2);
      const right = Math.floor((limit - 1) / 2);
      return `${value.slice(0, left)}…${value.slice(-right)}`;
    }

    function createThumbUrl(file) {
      if (!objectUrlApi) return "";
      try {
        return objectUrlApi.createObjectURL(file);
      } catch (_) {
        return "";
      }
    }

    function revokeThumbUrl(url) {
      if (!url || !objectUrlApi || typeof objectUrlApi.revokeObjectURL !== "function") return;
      try {
        objectUrlApi.revokeObjectURL(url);
      } catch (_) {
        // Revocation is best-effort cleanup; a failure only leaks until page unload.
      }
    }

    function getPendingUpload(key) {
      return pendingUploads.find((entry) => entry.key === key) || null;
    }

    function addPendingUploads(entries) {
      if (!entries.length) return;
      pendingUploads = pendingUploads.concat(entries);
      renderStagedAttachments();
      resizeComposer();
      projectSelectedAttachmentIndicator();
    }

    function removePendingUpload(key) {
      const entry = getPendingUpload(key);
      if (!entry) return;
      revokeThumbUrl(entry.thumbUrl);
      pendingUploads = pendingUploads.filter((item) => item.key !== key);
      renderStagedAttachments();
      resizeComposer();
      projectSelectedAttachmentIndicator();
    }

    function clearPendingUploads() {
      if (!pendingUploads.length) return;
      for (const entry of pendingUploads) {
        entry.canceled = true;
        if (entry.abortController) {
          try {
            entry.abortController.abort();
          } catch (_) {
            // Abort is best-effort; the response guard drops late completions.
          }
        }
        revokeThumbUrl(entry.thumbUrl);
      }
      pendingUploads = [];
      renderStagedAttachments();
      resizeComposer();
      projectSelectedAttachmentIndicator();
    }

    function isActivePendingUpload(entry) {
      return !entry.done && !entry.canceled && (!entry.failed || entry.unknown);
    }

    function hasPendingUploads() {
      return pendingUploads.some(isActivePendingUpload);
    }

    async function cancelPendingUpload(key) {
      const entry = getPendingUpload(key);
      if (!entry || entry.canceling) return;
      entry.cancelRequested = true;
      if (entry.abortController) entry.abortController.abort();
      entry.canceling = true;
      entry.failed = false;
      entry.phase = "cancelling";
      renderStagedAttachments();
      try {
        // A local abort cannot retract a server commit. The identity tombstone
        // makes cancel win even when the POST is still being consumed.
        const response = await api(`/api/sessions/${entry.sessionId}/attachments/upload-cancel`, {
          method: "POST", body: { upload_id: entry.key }, cache: "no-store", timeoutMs: 15000,
        });
        if (!entryIsCurrent(entry)) return;
        applyAttachmentResource(entry.sessionId, response);
        if (response.state !== "cancelled" && !(response.state === "committed" && response.active === false)) {
          throw new Error("cancellation not confirmed");
        }
        entry.canceled = true;
        removePendingUpload(key);
        refreshAfterAttachmentMutation();
      } catch (error) {
        if (!entryIsCurrent(entry)) return;
        if (error && error.status === 401) handleAppAuthLoss();
        failEntry(entry, error, true, "Cancellation outcome unknown");
      } finally {
        entry.canceling = false;
      }
    }

    function markFreshStaged(attachmentId) {
      freshIds.add(attachmentId);
      if (globalSetTimeout) globalSetTimeout(() => freshIds.delete(attachmentId), 1200);
    }

    function isAbortError(error) {
      return Boolean(error && (error.name === "AbortError" || error.code === 20));
    }

    function getStagedAttachments() {
      return stagedAttachments.slice();
    }

    function setStagedAttachments(list) {
      stagedAttachments = normalizedStagedAttachments(list);
      renderStagedAttachments();
      resizeComposer();
      projectSelectedAttachmentIndicator();
    }

    function setAttachCount(count) {
      const info = getSessionInfo(sessionState.get("selected"));
      if (Math.max(0, Number(count) || 0) === 0 && stagedAttachments.length && !Number.isSafeInteger(info && info.attachments_revision)) stagedAttachments = [];
      renderStagedAttachments();
      projectSelectedAttachmentIndicator();
    }

    function applyAttachmentResource(sessionId, response) {
      if (!response || !Array.isArray(response.attachments)) throw new Error("invalid attachment resource response");
      const list = normalizedStagedAttachments(response.attachments);
      const patch = {
        staged_attachments: list,
        pending_attachment: typeof response.pending_attachment === "boolean" ? response.pending_attachment : list.length > 0,
      };
      if (Number.isSafeInteger(response.attachments_revision) && response.attachments_revision >= 0) {
        patch.attachments_revision = response.attachments_revision;
      }
      sessionCatalog.patchSession(sessionId, patch);
      if (sessionState.get("selected") === sessionId) syncStagedAttachmentsFromSelectedSession();
    }

    function syncStagedAttachmentsFromSelectedSession() {
      const sessionId = sessionState.get("selected");
      const info = sessionId ? getSessionInfo(sessionId) : null;
      setStagedAttachments(info && Array.isArray(info.staged_attachments) ? info.staged_attachments : []);
    }

    function refreshAfterAttachmentMutation() {
      void refreshSessions().catch((error) => {
        if (error && error.status === 401) handleAppAuthLoss();
        else console.error("refreshSessions failed", error);
      });
    }

    function renderPendingChip(entry) {
      const phaseText = entry.failed ? `${entry.phase}: ${entry.failReason}` : entry.phase;
      const chip = el("div", {
        class: "stagedAttachmentChip",
        "data-pending": "1",
        title: entry.failed ? entry.failReason || "upload failed" : `${entry.name} · ${phaseText}`,
      });
      if (entry.failed) chip.setAttribute("data-failed", "1");
      if (entry.thumbUrl) chip.appendChild(el("img", { class: "stagedAttachmentThumb", src: entry.thumbUrl, alt: "" }));
      chip.appendChild(el("span", { class: "stagedAttachmentName", text: middleEllipsis(entry.name) }));
      chip.appendChild(el("span", { class: "stagedAttachmentPhase", text: phaseText }));
      if (entry.failed) {
        const retryBtn = el("button", { class: "stagedAttachmentRetry", type: "button", text: "Retry", "aria-label": `Retry ${entry.name}` });
        retryBtn.onclick = () => entry.cancelRequested ? cancelPendingUpload(entry.key) : runUpload(entry, true);
        chip.appendChild(retryBtn);
      }
      const cancelBtn = el("button", { class: "stagedAttachmentRemove", type: "button", text: "×", title: `Cancel ${entry.name}`, "aria-label": `Cancel ${entry.name}` });
      cancelBtn.disabled = entry.canceling;
      cancelBtn.onclick = () => cancelPendingUpload(entry.key);
      chip.appendChild(cancelBtn);
      return chip;
    }

    function renderStagedAttachments() {
      const liveIds = new Set(stagedAttachments.map((item) => item.id));
      for (const id of Array.from(localThumbs.keys())) {
        if (!liveIds.has(id)) {
          revokeThumbUrl(localThumbs.get(id));
          localThumbs.delete(id);
        }
      }
      const tray = getTray();
      if (!tray) return;
      tray.innerHTML = "";
      if (!stagedAttachments.length && !pendingUploads.length) {
        tray.style.display = "none";
        return;
      }
      tray.style.display = "flex";
      for (const entry of pendingUploads) {
        tray.appendChild(renderPendingChip(entry));
      }
      for (const item of stagedAttachments) {
        const chip = el("div", { class: "stagedAttachmentChip", title: attachmentIdentityText(item) });
        if (freshIds.has(item.id)) chip.setAttribute("data-fresh", "1");
        const thumbUrl = localThumbs.get(item.id) || "";
        if (thumbUrl) chip.appendChild(el("img", { class: "stagedAttachmentThumb", src: thumbUrl, alt: "" }));
        const name = item.display_name || item.filename || "file";
        chip.appendChild(el("span", { class: "stagedAttachmentName", text: middleEllipsis(name) }));
        chip.appendChild(el("span", { class: "stagedAttachmentMeta", text: fmtBytes(item.size || 0) }));
        const removeBtn = el("button", { class: "stagedAttachmentRemove", type: "button", text: "×", title: `Remove ${item.display_name || "attachment"}`, "aria-label": `Remove ${item.display_name || "attachment"}` });
        removeBtn.onclick = async () => {
          const sessionId = sessionState.get("selected");
          if (!sessionId) return;
          try {
            const response = await api(`/api/sessions/${sessionId}/attachments/delete`, { method: "POST", body: { id: item.id } });
            applyAttachmentResource(sessionId, response);
            if (sessionState.get("selected") === sessionId) {
              setToast("attachment removed");
              refreshAfterAttachmentMutation();
            }
          } catch (error) {
            if (error && error.status === 401) {
              handleAppAuthLoss();
              return;
            }
            if (sessionState.get("selected") === sessionId) setToast(`remove attachment error: ${error && error.message ? error.message : "unknown error"}`);
          }
        };
        chip.appendChild(removeBtn);
        tray.appendChild(chip);
      }
      if (!stagedAttachments.length) return;
      const clearBtn = el("button", { class: "stagedAttachmentsClear", type: "button", text: "Clear", title: "Clear staged attachments", "aria-label": "Clear staged attachments" });
      clearBtn.onclick = async () => {
        const sessionId = sessionState.get("selected");
        if (!sessionId) return;
        try {
          const response = await api(`/api/sessions/${sessionId}/attachments/clear`, { method: "POST", body: {} });
          applyAttachmentResource(sessionId, response);
          if (sessionState.get("selected") === sessionId) {
            setToast("attachments cleared");
            refreshAfterAttachmentMutation();
          }
        } catch (error) {
          if (error && error.status === 401) {
            handleAppAuthLoss();
            return;
          }
          if (sessionState.get("selected") === sessionId) setToast(`clear attachments error: ${error && error.message ? error.message : "unknown error"}`);
        }
      };
      tray.appendChild(clearBtn);
    }

    function projectSelectedAttachmentIndicator() {
      const sessionId = sessionState.get("selected");
      const sessionInfo = sessionId ? getSessionInfo(sessionId) : null;
      const serverListCount = sessionInfo && Array.isArray(sessionInfo.staged_attachments) ? normalizedStagedAttachments(sessionInfo.staged_attachments).length : 0;
      const serverPending = Boolean(sessionInfo && sessionInfo.pending_attachment);
      const pendingActive = pendingUploads.filter(isActivePendingUpload).length;
      const visible = Math.max(stagedAttachments.length, serverListCount, serverPending ? 1 : 0) + pendingActive;
      if (visible > 0) {
        attachBadgeEl.textContent = String(visible);
        attachBadgeEl.style.display = "inline-flex";
      } else {
        attachBadgeEl.textContent = "";
        attachBadgeEl.style.display = "none";
      }
    }

    function setSelectedSessionPendingAttachment(sessionId, value) {
      if (!sessionId || sessionState.get("selected") !== sessionId) return false;
      const info = getSessionInfo(sessionId);
      if (!info) return false;
      if (Number.isSafeInteger(info.attachments_revision)) {
        // Legacy send callbacks carry no revision. Never invent a newer one or
        // let that unversioned write defeat the resource ordering boundary.
        void api(`/api/sessions/${sessionId}/attachments`, { cache: "no-store", timeoutMs: 15000 })
          .then((response) => applyAttachmentResource(sessionId, response))
          .catch((error) => {
            if (error && error.status === 401) handleAppAuthLoss();
            else if (sessionState.get("selected") === sessionId) setToast("Could not verify attachment state; refresh to retry.");
          });
      } else {
        sessionCatalog.patchSession(sessionId, {
          pending_attachment: Boolean(value), ...(value ? {} : { staged_attachments: [] }),
        });
        syncStagedAttachmentsFromSelectedSession();
      }
      return true;
    }

    function attachmentBlockerForSession(sessionId, sessionInfo = null) {
      if (!sessionId) return "Select a session to attach a file";
      const info = sessionInfo || getSessionInfo(sessionId) || null;
      if (info && sessionLaunchFailed(info)) return "Failed launch cannot receive file attachments";
      if (info && sessionHasUnknownSend(info)) return "Resolve the unknown send before attaching a file";
      if (info && sessionIsOrphanRecovery(info)) return "Missing session can only be reviewed";
      if (info && sessionHasOrphanQueueRecovery(info)) return "Review preserved queued recovery items before attaching a file";
      if (sessionState.get("sending")) return "Wait for the current send to finish before attaching a file";
      return "";
    }

    function latestAttachmentBlockerForSession(sessionId) {
      return attachmentBlockerForSession(sessionId, sessionId ? getSessionInfo(sessionId) || null : null);
    }

    function syncAttachButtonState() {
      const sessionId = sessionState.get("selected");
      const attachBlocker = attachmentBlockerForSession(sessionId, sessionId ? getSessionInfo(sessionId) || null : null);
      const attachLabel = attachBlocker || `Attach file (max ${fmtBytes(uploadMaxBytes)})`;
      attachBtn.disabled = Boolean(attachBlocker);
      attachBtn.title = attachLabel;
      attachBtn.setAttribute("aria-label", attachLabel);
    }

    async function toJpegBlob(file, { maxDim = 2048, quality = 0.86 } = {}) {
      const url = URL.createObjectURL(file);
      try {
        const img = new Image();
        img.decoding = "async";
        img.src = url;
        if (img.decode) await img.decode();
        else await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = () => reject(new Error("decode failed"));
        });
        const w0 = img.naturalWidth || img.width || 0;
        const h0 = img.naturalHeight || img.height || 0;
        if (!w0 || !h0) throw new Error("invalid image dimensions");
        const scale = Math.min(1, maxDim / Math.max(w0, h0));
        const width = Math.max(1, Math.round(w0 * scale));
        const height = Math.max(1, Math.round(h0 * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d", { alpha: false });
        if (!context) throw new Error("no canvas");
        context.drawImage(img, 0, 0, width, height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
        if (!blob) throw new Error("jpeg encode failed");
        return blob;
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    function imageExtensionFromMimeType(type, fallback = "") {
      const normalized = String(type || "").toLowerCase();
      if (normalized === "image/jpeg" || normalized === "image/jpg") return "jpg";
      if (normalized === "image/png") return "png";
      if (normalized === "image/gif") return "gif";
      if (normalized === "image/webp") return "webp";
      if (normalized === "image/heic") return "heic";
      if (normalized === "image/heif") return "heif";
      if (normalized === "image/avif") return "avif";
      return normalized.startsWith("image/") ? fallback : "";
    }

    function pastedFileName(file, index, seed) {
      const suffix = index > 0 ? `-${index + 1}` : "";
      const base = `pasted-${seed}${suffix}`;
      const ext = imageExtensionFromMimeType(file && file.type, "png");
      return ext ? `${base}.${ext}` : base;
    }

    function entryIsCurrent(entry) {
      return !entry.canceled && pendingUploads.includes(entry) && sessionState.get("selected") === entry.sessionId;
    }

    function failEntry(entry, error, unknown, phase = entry.phase) {
      entry.failed = true;
      entry.unknown = unknown;
      entry.phase = unknown ? (phase === "Cancellation outcome unknown" ? phase : "Upload outcome unknown") : phase;
      entry.failReason = error && error.message ? String(error.message) : "unknown error";
      renderStagedAttachments();
      projectSelectedAttachmentIndicator();
    }

    function finishUpload(entry, response) {
      applyAttachmentResource(entry.sessionId, response);
      // A committed-but-retired identity is terminal: never recreate the file.
      const matched = response.active === false ? null : response.attachment;
      const staged = Boolean(matched && stagedAttachments.some((item) => item.id === matched.id));
      if (staged && entry.thumbUrl) {
        localThumbs.set(matched.id, entry.thumbUrl);
        entry.thumbUrl = "";
        markFreshStaged(matched.id);
      }
      entry.done = true;
      removePendingUpload(entry.key);
      setToast(staged ? "file staged" : "upload already removed or sent");
      return staged;
    }

    function waitForStatus(entry, ms) {
      return new Promise((resolve) => {
        if (!globalSetTimeout || !entryIsCurrent(entry)) return resolve();
        const signal = entry.abortController && entry.abortController.signal;
        if (signal && signal.aborted) return resolve();
        const done = () => {
          if (signal && typeof signal.removeEventListener === "function") signal.removeEventListener("abort", done);
          resolve();
        };
        globalSetTimeout(done, ms);
        if (signal && typeof signal.addEventListener === "function") signal.addEventListener("abort", done, { once: true });
      });
    }

    async function lookupUpload(entry) {
      entry.phase = "checking upload";
      renderStagedAttachments();
      for (let poll = 0; poll < 3; poll += 1) {
        if (!entryIsCurrent(entry) || entry.cancelRequested) return null;
        const response = await api(`/api/sessions/${entry.sessionId}/attachments/upload-status?upload_id=${encodeURIComponent(entry.key)}`, {
          cache: "no-store", timeoutMs: 15000,
          ...(entry.abortController ? { signal: entry.abortController.signal } : {}),
        });
        if (!entryIsCurrent(entry) || entry.cancelRequested) return null;
        applyAttachmentResource(entry.sessionId, response);
        if (!["absent", "pending", "committed", "failed", "cancelled"].includes(response.state)) throw new Error("invalid upload status");
        if (response.state !== "pending") return response;
        if (poll < 2) await waitForStatus(entry, 500 * (poll + 1));
      }
      throw new Error("server is still processing; retry to check again");
    }

    async function prepareUpload(entry) {
      if (entry.blob) return;
      let blob = entry.file;
      if (looksLikeImage(blob) && (blob.size > uploadMaxBytes || isLikelyHeic(blob))) {
        const name = `${safeAttachmentStem(entry.name)}.jpg`;
        for (const attempt of [
          { maxDim: 2048, quality: 0.86 }, { maxDim: 1600, quality: 0.82 },
          { maxDim: 1600, quality: 0.72 }, { maxDim: 1280, quality: 0.68 }, { maxDim: 1280, quality: 0.58 },
        ]) {
          blob = await toJpegBlob(entry.file, attempt);
          if (blob.size <= uploadMaxBytes) break;
        }
        entry.name = name;
      }
      if (!blob || blob.size > uploadMaxBytes) throw new Error(`file too large (max ${fmtBytes(uploadMaxBytes)})`);
      entry.blob = blob;
      entry.size = blob.size;
    }

    async function runUpload(entry, reconcileFirst = false) {
      if (!entryIsCurrent(entry) || entry.working || entry.cancelRequested) return false;
      const blocker = latestAttachmentBlockerForSession(entry.sessionId);
      if (blocker) { failEntry(entry, new Error(blocker), entry.unknown); return false; }
      entry.working = true;
      // Retry is intent, not evidence of a possible server commit. Preserve
      // prior uncertainty, but a file that never reached submission can fail
      // preparation repeatedly without becoming an unknown server outcome.
      let outcomeUnknown = entry.unknown;
      let reconcile = reconcileFirst && entry.submitted;
      entry.failed = false;
      entry.unknown = false;
      entry.abortController = AbortCtor ? new AbortCtor() : null;
      try {
        await prepareUpload(entry);
        for (let attempt = 0; attempt <= 2; attempt += 1) {
          if (!entryIsCurrent(entry) || entry.cancelRequested) return false;
          if (outcomeUnknown || reconcile) {
            outcomeUnknown = true;
            const status = await lookupUpload(entry);
            reconcile = false;
            if (!status) return false;
            if (status.state === "committed") return finishUpload(entry, status);
            if (status.state === "cancelled") { removePendingUpload(entry.key); return false; }
            if (status.state === "failed" && status.retryable === false) {
              outcomeUnknown = false;
              throw new Error(status.error || "upload rejected");
            }
            outcomeUnknown = false;
          }
          if (attempt === 2) throw new Error("upload not committed after two attempts; retry when ready");
          entry.phase = "uploading";
          renderStagedAttachments();
          const headers = { "Content-Type": "application/octet-stream", "X-Codoxear-Upload-Id": entry.key, "X-Codoxear-Filename": encodeURIComponent(entry.name) };
          try {
            entry.submitted = true;
            const response = await api(`/api/sessions/${entry.sessionId}/inject_file`, {
              method: "POST", rawBody: entry.blob, headers,
              ...(entry.abortController ? { signal: entry.abortController.signal } : {}),
            });
            if (!entryIsCurrent(entry) || entry.cancelRequested) return false;
            if (!response || !response.ok) throw new Error("upload response did not confirm commit");
            if (response.state === "cancelled") {
              applyAttachmentResource(entry.sessionId, response);
              removePendingUpload(entry.key);
              return false;
            }
            if (response.state && response.state !== "committed") throw new Error("upload has not committed yet");
            return finishUpload(entry, response);
          } catch (error) {
            if (!entryIsCurrent(entry) || entry.cancelRequested || isAbortError(error)) return false;
            if (error && error.status >= 400 && error.status < 500 && error.status !== 408 && error.status !== 409) throw error;
            outcomeUnknown = true;
          }
        }
      } catch (error) {
        if (!entryIsCurrent(entry) || entry.cancelRequested || isAbortError(error)) return false;
        if (error && error.status === 401) handleAppAuthLoss();
        failEntry(entry, error, outcomeUnknown);
        setToast(`attach error: ${entry.name}: ${entry.failReason}`);
        return false;
      } finally {
        entry.working = false;
        if (entryIsCurrent(entry) && !entry.cancelRequested) {
          setPollFastUntilMs(now() + 4000);
          kickPoll(0);
          refreshAfterAttachmentMutation();
        }
      }
    }

    async function stageFiles(files, { sid = sessionState.get("selected"), source = "picker" } = {}) {
      const sessionId = sid || sessionState.get("selected");
      const uploadFiles = Array.from(files || []).filter(Boolean);
      if (!uploadFiles.length || sessionId !== sessionState.get("selected")) return false;
      const seed = now();
      const entries = uploadFiles.map((file, index) => ({
        key: typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID() : `upload-${seed}-${Math.random().toString(36).slice(2)}-${index}`,
        sessionId, file, blob: null,
        name: file.name || (source === "paste" ? pastedFileName(file, index, seed) : "file"),
        size: Number(file.size) || 0,
        phase: looksLikeImage(file) && (file.size > uploadMaxBytes || isLikelyHeic(file)) ? "compressing" : "uploading",
        done: false, failed: false, unknown: false, submitted: false, failReason: "", canceled: false,
        cancelRequested: false, canceling: false, working: false,
        thumbUrl: looksLikeImage(file) ? createThumbUrl(file) : "", abortController: null,
      }));
      addPendingUploads(entries);
      let successes = 0;
      for (const entry of entries) if (await runUpload(entry)) successes += 1;
      if (sessionState.get("selected") === sessionId) {
        setPollFastUntilMs(now() + 4000);
        kickPoll(0);
        refreshAfterAttachmentMutation();
      }
      return successes > 0;
    }

    function clipboardPlainText(data) {
      if (!data || typeof data.getData !== "function") return "";
      try {
        return data.getData("text/plain") || data.getData("text") || "";
      } catch (_) {
        return "";
      }
    }

    function insertComposerPastedText(text) {
      const value = String(text || "");
      if (!value) return false;
      const start = Number.isFinite(textarea.selectionStart) ? textarea.selectionStart : textarea.value.length;
      const end = Number.isFinite(textarea.selectionEnd) ? textarea.selectionEnd : start;
      if (typeof textarea.setRangeText === "function") textarea.setRangeText(value, start, end, "end");
      else {
        textarea.value = `${textarea.value.slice(0, start)}${value}${textarea.value.slice(end)}`;
        textarea.selectionStart = start + value.length;
        textarea.selectionEnd = start + value.length;
      }
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      return true;
    }

    function setComposerDropActive(active) {
      composer.classList.toggle("drop-active", Boolean(active));
    }

    function clearComposerDropActive() {
      composerDragDepth = 0;
      setComposerDropActive(false);
    }

    attachBtn.onclick = () => {
      const sessionId = sessionState.get("selected");
      const attachBlocker = attachmentBlockerForSession(sessionId, sessionId ? getSessionInfo(sessionId) || null : null);
      if (attachBlocker) {
        setToast(attachBlocker);
        return;
      }
      imgInput.value = "";
      imgInput.click();
    };
    addEventListener(imgInput, "change", async () => {
      const sessionId = sessionState.get("selected");
      if (!sessionId) return;
      const files = Array.from(imgInput.files || []);
      imgInput.value = "";
      await stageFiles(files, { sid: sessionId, source: "picker" });
    });
    addEventListener(textarea, "paste", (event) => {
      const files = extractFilesFromClipboardData(event.clipboardData);
      if (!files.length) return;
      const pastedText = clipboardPlainText(event.clipboardData);
      event.preventDefault();
      if (pastedText) insertComposerPastedText(pastedText);
      void stageFiles(files, { sid: sessionState.get("selected"), source: "paste" });
    });
    addEventListener(composer, "dragenter", (event) => {
      if (!dataTransferHasFiles(event.dataTransfer)) return;
      event.preventDefault();
      composerDragDepth += 1;
      setComposerDropActive(true);
    }, { passive: false });
    addEventListener(composer, "dragover", (event) => {
      if (!dataTransferHasFiles(event.dataTransfer)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      setComposerDropActive(true);
    }, { passive: false });
    addEventListener(composer, "dragleave", (event) => {
      if (!dataTransferHasFiles(event.dataTransfer)) return;
      composerDragDepth = Math.max(0, composerDragDepth - 1);
      if (composerDragDepth === 0) setComposerDropActive(false);
    }, { passive: false });
    addEventListener(composer, "drop", (event) => {
      if (!dataTransferHasFiles(event.dataTransfer)) return;
      event.preventDefault();
      clearComposerDropActive();
      const files = extractFilesFromDropData(event.dataTransfer);
      if (files.length) void stageFiles(files, { sid: sessionState.get("selected"), source: "drop" });
    }, { passive: false });
    addEventListener(window, "dragover", (event) => {
      if (dataTransferHasFiles(event.dataTransfer)) event.preventDefault();
    }, { passive: false });
    addEventListener(window, "dragleave", (event) => {
      const outsideWindow =
        event.clientX <= 0 ||
        event.clientY <= 0 ||
        event.clientX >= window.innerWidth ||
        event.clientY >= window.innerHeight ||
        (!event.relatedTarget && (event.target === document || event.target === document.documentElement || event.target === document.body));
      if (outsideWindow) clearComposerDropActive();
    }, { passive: false });
    addEventListener(window, "dragend", clearComposerDropActive, { passive: false });
    addEventListener(window, "drop", (event) => {
      if (dataTransferHasFiles(event.dataTransfer)) event.preventDefault();
      clearComposerDropActive();
    }, { passive: false });

    const unsubscribeSessionState = [
      sessionState.subscribe("selected", syncAttachButtonState),
      sessionState.subscribe("selected", clearPendingUploads),
      sessionState.subscribe("sending", syncAttachButtonState),
      sessionCatalog.subscribe("sessionIndex", syncAttachButtonState),
    ];
    setAttachCount(0);
    syncAttachButtonState();

    return Object.freeze({
      normalizedStagedAttachments,
      getStagedAttachments,
      setStagedAttachments,
      setAttachCount,
      syncStagedAttachmentsFromSelectedSession,
      setSelectedSessionPendingAttachment,
      attachmentBlockerForSession,
      syncAttachButtonState,
      stageFiles,
      hasPendingUploads,
      projectSelectedAttachmentIndicator,
      dispose() {
        clearPendingUploads();
        while (unsubscribeSessionState.length) unsubscribeSessionState.pop()();
      },
    });
  }

export { createAttachmentsController };
