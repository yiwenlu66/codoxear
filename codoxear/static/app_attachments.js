
  // Attachment staging authority. Owns the selected session's local staged-list
  // projection, upload producers (picker/paste/drop), image compression, and
  // all attachment-specific DOM/event state. The server's staged list remains
  // authoritative; direct mutation responses update this projection until the
  // next session refresh replaces it.

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
    if (!sessionCatalog || typeof sessionCatalog.get !== "function") throw new TypeError("attachments controller dependency missing: sessionCatalog");
    const getSessionInfo = (sessionId) => sessionCatalog.get("sessionIndex").get(sessionId) || null;
    const patchSessionInfo = (sessionId, patch) => {
      const current = getSessionInfo(sessionId);
      if (current) Object.assign(current, patch || {});
    };

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
    const b64FromBytes = requireFunction(options.b64FromBytes, "b64FromBytes");
    const dataTransferHasFiles = requireFunction(options.dataTransferHasFiles, "dataTransferHasFiles");
    const extractFilesFromClipboardData = requireFunction(options.extractFilesFromClipboardData, "extractFilesFromClipboardData");
    const extractFilesFromDropData = requireFunction(options.extractFilesFromDropData, "extractFilesFromDropData");
    const addEventListener = requireFunction(options.addEventListener, "addEventListener");
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const uploadMaxBytes = Number(options.uploadMaxBytes);
    if (!Number.isFinite(uploadMaxBytes) || uploadMaxBytes <= 0)
      throw new TypeError("attachments controller dependency missing: uploadMaxBytes");

    let stagedAttachments = [];
    let composerDragDepth = 0;
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
      if (Math.max(0, Number(count) || 0) === 0 && stagedAttachments.length) stagedAttachments = [];
      renderStagedAttachments();
      projectSelectedAttachmentIndicator();
    }

    function setSelectedSessionStagedAttachments(list) {
      const sessionId = sessionState.get("selected");
      if (sessionId) {
        const normalized = normalizedStagedAttachments(list);
        patchSessionInfo(sessionId, {
          staged_attachments: normalized,
          pending_attachment: normalized.length > 0,
        });
      }
      setStagedAttachments(list);
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

    function renderStagedAttachments() {
      const tray = getTray();
      if (!tray) return;
      tray.innerHTML = "";
      if (!stagedAttachments.length) {
        tray.style.display = "none";
        return;
      }
      tray.style.display = "flex";
      for (const item of stagedAttachments) {
        const chip = el("div", { class: "stagedAttachmentChip", title: attachmentIdentityText(item) });
        const name = item.display_name || item.filename || "file";
        chip.appendChild(el("span", { class: "stagedAttachmentName", text: middleEllipsis(name) }));
        chip.appendChild(el("span", { class: "stagedAttachmentMeta", text: fmtBytes(item.size || 0) }));
        const removeBtn = el("button", { class: "stagedAttachmentRemove", type: "button", text: "×", title: `Remove ${item.display_name || "attachment"}`, "aria-label": `Remove ${item.display_name || "attachment"}` });
        removeBtn.onclick = async () => {
          const sessionId = sessionState.get("selected");
          if (!sessionId) return;
          try {
            const response = await api(`/api/sessions/${sessionId}/attachments/delete`, { method: "POST", body: { id: item.id } });
            if (sessionState.get("selected") === sessionId) {
              setSelectedSessionStagedAttachments(response && Array.isArray(response.attachments) ? response.attachments : []);
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
      const clearBtn = el("button", { class: "stagedAttachmentsClear", type: "button", text: "Clear", title: "Clear staged attachments", "aria-label": "Clear staged attachments" });
      clearBtn.onclick = async () => {
        const sessionId = sessionState.get("selected");
        if (!sessionId) return;
        try {
          const response = await api(`/api/sessions/${sessionId}/attachments/clear`, { method: "POST", body: {} });
          if (sessionState.get("selected") === sessionId) {
            setSelectedSessionStagedAttachments(response && Array.isArray(response.attachments) ? response.attachments : []);
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
      const visible = Math.max(stagedAttachments.length, serverListCount, serverPending ? 1 : 0);
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
      patchSessionInfo(sessionId, {
        pending_attachment: Boolean(value),
        ...(value ? {} : { staged_attachments: [] }),
      });
      if (!value) setStagedAttachments([]);
      else projectSelectedAttachmentIndicator();
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

    async function stageFiles(files, { sid = sessionState.get("selected"), source = "picker" } = {}) {
      const sessionId = sid || sessionState.get("selected");
      const uploadFiles = Array.from(files || []).filter(Boolean);
      if (!uploadFiles.length) return false;

      const producer = String(source || "picker");
      const progressVerb = producer === "paste" ? "pasting" : producer === "drop" ? "dropping" : "uploading";
      const producerNameSeed = now();
      let successes = 0;
      let stoppedByBlocker = "";
      const failures = [];
      for (let fileIndex = 0; fileIndex < uploadFiles.length; fileIndex += 1) {
        const file = uploadFiles[fileIndex];
        try {
          if (sessionState.get("selected") !== sessionId) break;
          const attachBlocker = latestAttachmentBlockerForSession(sessionId);
          if (attachBlocker) {
            stoppedByBlocker = attachBlocker;
            break;
          }
          setToast(uploadFiles.length > 1 ? `${progressVerb} ${fileIndex + 1}/${uploadFiles.length}...` : "uploading file...");
          let uploadBlob = file;
          let uploadName = file.name || (producer === "paste" ? pastedFileName(file, fileIndex, producerNameSeed) : "file");
          if (looksLikeImage(file) && (file.size > uploadMaxBytes || isLikelyHeic(file))) {
            setToast("compressing image...");
            const stem = safeAttachmentStem(uploadName);
            uploadName = `${stem}.jpg`;
            const tries = [
              { maxDim: 2048, quality: 0.86 },
              { maxDim: 1600, quality: 0.82 },
              { maxDim: 1600, quality: 0.72 },
              { maxDim: 1280, quality: 0.68 },
              { maxDim: 1280, quality: 0.58 },
            ];
            let blob = null;
            for (const attempt of tries) {
              blob = await toJpegBlob(file, attempt);
              if (blob.size <= uploadMaxBytes) break;
            }
            if (!blob || blob.size > uploadMaxBytes) throw new Error(`image too large (max ${fmtBytes(uploadMaxBytes)})`);
            uploadBlob = blob;
          }
          const bytes = await uploadBlob.arrayBuffer();
          if (bytes.byteLength > uploadMaxBytes) throw new Error(`file too large (max ${fmtBytes(uploadMaxBytes)})`);
          const response = await api(`/api/sessions/${sessionId}/inject_file`, {
            method: "POST",
            body: { filename: uploadName, data_b64: b64FromBytes(new Uint8Array(bytes)) },
          });
          if (sessionState.get("selected") === sessionId && response && response.ok) {
            successes += 1;
            setSelectedSessionStagedAttachments(Array.isArray(response.attachments) ? response.attachments : []);
          }
        } catch (error) {
          if (error && error.status === 401) {
            handleAppAuthLoss();
            return false;
          }
          failures.push(`${file && file.name ? file.name : "file"}: ${error && error.message ? error.message : "unknown error"}`);
        }
      }
      if (sessionState.get("selected") === sessionId) {
        if (successes && failures.length) setToast(`attached ${successes}; ${failures.length} failed: ${failures[0]}`);
        else if (successes && stoppedByBlocker) setToast(`attached ${successes}; stopped: ${stoppedByBlocker}`);
        else if (successes) setToast(successes === 1 ? "file staged" : `${successes} files staged`);
        else if (failures.length) setToast(`attach error: ${failures[0]}`);
        else if (stoppedByBlocker) setToast(stoppedByBlocker);
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
      sessionState.subscribe("sending", syncAttachButtonState),
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
      projectSelectedAttachmentIndicator,
      dispose() {
        while (unsubscribeSessionState.length) unsubscribeSessionState.pop()();
      },
    });
  }

export { createAttachmentsController };
