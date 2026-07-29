(function () {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`composer controller dependency missing: ${name}`);
    return value;
  }
  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.addEventListener !== "function")
      throw new TypeError(`composer controller dependency missing: ${name}`);
    return value;
  }

  // Composer authority is deliberately separate from session selection and
  // transcript rendering. Those app-wide concerns are injected through a small
  // explicit boundary, while staging, drafts, resize, sendability projection,
  // and submit wiring remain together so they cannot drift apart.
  function createComposerController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("composer controller dependency missing: options");
    const form = requireNode(options.form, "form");
    const composer = requireNode(options.composer, "composer");
    const textarea = requireNode(options.textarea, "textarea");
    const msgPh = requireNode(options.msgPh, "msgPh");
    const attachBtn = requireNode(options.attachBtn, "attachBtn");
    const imgInput = requireNode(options.imgInput, "imgInput");
    const sendBtn = requireNode(options.sendBtn, "sendBtn");
    const stagedTray = requireNode(options.stagedTray, "stagedTray");
    const el = requireFunction(options.el, "el");
    const fmtBytes = requireFunction(options.fmtBytes, "fmtBytes");
    const api = requireFunction(options.api, "api");
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const setSessionInfo = requireFunction(options.setSessionInfo, "setSessionInfo");
    const getSending = requireFunction(options.getSending, "getSending");
    const setSending = requireFunction(options.setSending, "setSending");
    const getCurrentRunning = requireFunction(options.getCurrentRunning, "getCurrentRunning");
    const setToast = requireFunction(options.setToast, "setToast");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const kickPoll = requireFunction(options.kickPoll, "kickPoll");
    const setPollFastUntilMs = requireFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
    const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
    const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const sessionHasUnknownSend = requireFunction(options.sessionHasUnknownSend, "sessionHasUnknownSend");
    const sessionIsOrphanRecovery = requireFunction(options.sessionIsOrphanRecovery, "sessionIsOrphanRecovery");
    const sessionHasOrphanQueueRecovery = requireFunction(options.sessionHasOrphanQueueRecovery, "sessionHasOrphanQueueRecovery");
    const confirmAction = requireFunction(options.confirmAction, "confirmAction");
    const submit = requireFunction(options.submit, "submit");
    const attachmentUploadMaxBytes = Number(options.attachmentUploadMaxBytes);
    if (!Number.isFinite(attachmentUploadMaxBytes) || attachmentUploadMaxBytes <= 0)
      throw new TypeError("composer controller dependency missing: attachmentUploadMaxBytes");
    const dataTransferHasFiles = requireFunction(options.dataTransferHasFiles, "dataTransferHasFiles");
    const extractFilesFromClipboardData = requireFunction(options.extractFilesFromClipboardData, "extractFilesFromClipboardData");
    const extractFilesFromDropData = requireFunction(options.extractFilesFromDropData, "extractFilesFromDropData");
    const b64FromBytes = requireFunction(options.b64FromBytes, "b64FromBytes");
    const looksLikeImage = requireFunction(options.looksLikeImage, "looksLikeImage");
    const isLikelyHeic = requireFunction(options.isLikelyHeic, "isLikelyHeic");
    const safeAttachmentStem = requireFunction(options.safeAttachmentStem, "safeAttachmentStem");
    const toJpegBlob = requireFunction(options.toJpegBlob, "toJpegBlob");
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const onAutoGrow = typeof options.onAutoGrow === "function" ? options.onAutoGrow : () => {};

    let stagedAttachments = [];
    let dragDepth = 0;
    const cleanups = [];
    const listen = (target, type, handler, eventOptions) => {
      target.addEventListener(type, handler, eventOptions);
      cleanups.push(() => target.removeEventListener(type, handler, eventOptions));
    };
    const normalizedAttachments = (list) => Array.isArray(list) ? list.filter((item) => item && item.id).map((item) => ({
      id: String(item.id), display_name: String(item.display_name || item.filename || "file"), filename: String(item.filename || item.display_name || "file"), size: Number(item.size) || 0, created_ts: Number(item.created_ts) || 0,
    })) : [];
    const sessionDraftKey = (sid) => `codexweb.draft.${sid}`;
    const selectedInfo = () => {
      const sid = getSelected();
      return sid ? getSessionInfo(sid) || null : null;
    };
    function attachmentBlockerForSession(sessionId, info = null) {
      if (!sessionId) return "Select a session to attach a file";
      const session = info || getSessionInfo(sessionId) || null;
      if (session && sessionLaunchFailed(session)) return "Failed launch cannot receive file attachments";
      if (session && sessionHasUnknownSend(session)) return "Resolve the unknown send before attaching a file";
      if (session && sessionIsOrphanRecovery(session)) return "Missing session can only be reviewed";
      if (session && sessionHasOrphanQueueRecovery(session)) return "Review preserved queued recovery items before attaching a file";
      if (session && session.busy) return "Wait for the current response to finish before attaching a file";
      if (getCurrentRunning() || getSending()) return "Wait for the current response to finish before attaching a file";
      return "";
    }
    function syncSendButtonState() {
      const session = selectedInfo();
      const blocked = !session || sessionLaunchFailed(session) || sessionHasUnknownSend(session) || sessionIsOrphanRecovery(session) || sessionHasOrphanQueueRecovery(session);
      const label = !session ? "Select a session to send" : sessionLaunchFailed(session) ? "Failed launch cannot receive messages" : sessionHasUnknownSend(session) ? "Resolve the unknown send before sending" : sessionIsOrphanRecovery(session) ? "Missing session can only be reviewed" : sessionHasOrphanQueueRecovery(session) ? "Review preserved queued recovery items before sending" : "Send";
      sendBtn.disabled = Boolean(getSending() || blocked);
      sendBtn.title = label;
      sendBtn.setAttribute("aria-label", label);
      syncComposerState();
    }
    function syncComposerState() {
      const session = selectedInfo();
      const blocked = !session || sessionLaunchFailed(session) || sessionHasUnknownSend(session) || sessionIsOrphanRecovery(session) || sessionHasOrphanQueueRecovery(session);
      const label = !session ? "Select a session to send" : sessionLaunchFailed(session) ? "Failed launch cannot receive messages" : sessionHasUnknownSend(session) ? "Resolve the unknown send before sending" : sessionIsOrphanRecovery(session) ? "Missing session can only be reviewed" : sessionHasOrphanQueueRecovery(session) ? "Review preserved queued recovery items before sending" : "Enter your instructions here";
      textarea.disabled = blocked;
      textarea.setAttribute("aria-label", label);
      textarea.title = blocked ? label : "";
      msgPh.textContent = label;
    }
    function syncAttachButtonState() {
      const blocker = attachmentBlockerForSession(getSelected(), selectedInfo());
      const label = blocker || `Attach file (max ${fmtBytes(attachmentUploadMaxBytes)})`;
      attachBtn.disabled = Boolean(blocker);
      attachBtn.title = label;
      attachBtn.setAttribute("aria-label", label);
    }
    function autoGrow() {
      const basePx = parseFloat(getComputedStyle(textarea).minHeight || "0") || 32;
      const maxPx = 180;
      msgPh.style.display = textarea.value || stagedAttachments.length ? "none" : "flex";
      textarea.style.height = `${basePx}px`;
      let height = textarea.scrollHeight;
      const multiline = textarea.value.includes("\n") || height > basePx + 1;
      form.classList.toggle("multiline", multiline);
      textarea.style.height = multiline ? "auto" : `${basePx}px`;
      height = textarea.scrollHeight;
      textarea.style.height = `${multiline ? Math.min(height, maxPx) : basePx}px`;
      textarea.style.overflowY = height > maxPx ? "auto" : "hidden";
      onAutoGrow();
    }
    function renderStagedAttachments() {
      stagedTray.innerHTML = "";
      stagedTray.style.display = stagedAttachments.length ? "flex" : "none";
      for (const item of stagedAttachments) {
        const chip = el("div", { class: "stagedAttachmentChip", title: `${item.display_name} · ${fmtBytes(item.size)}` });
        chip.append(el("span", { class: "stagedAttachmentName", text: item.display_name }), el("span", { class: "stagedAttachmentMeta", text: fmtBytes(item.size) }));
        const remove = el("button", { class: "stagedAttachmentRemove", type: "button", text: "×", title: `Remove ${item.display_name}`, "aria-label": `Remove ${item.display_name}` });
        remove.onclick = () => void mutateAttachments(`/attachments/delete`, { id: item.id }, "attachment removed");
        chip.appendChild(remove);
        stagedTray.appendChild(chip);
      }
      if (stagedAttachments.length) {
        const clear = el("button", { class: "stagedAttachmentsClear", type: "button", text: "Clear", title: "Clear staged attachments", "aria-label": "Clear staged attachments" });
        clear.onclick = () => void mutateAttachments(`/attachments/clear`, {}, "attachments cleared");
        stagedTray.appendChild(clear);
      }
    }
    function setStagedAttachments(list) { stagedAttachments = normalizedAttachments(list); renderStagedAttachments(); autoGrow(); }
    function syncStagedAttachmentsFromSelectedSession() { const session = selectedInfo(); setStagedAttachments(session && session.staged_attachments); }
    async function mutateAttachments(path, body, successToast) {
      const sid = getSelected();
      if (!sid) return;
      try {
        const response = await api(`/api/sessions/${sid}${path}`, { method: "POST", body });
        if (getSelected() === sid) { setStagedAttachments(response && response.attachments); setToast(successToast); void refreshSessions(); }
      } catch (error) {
        if (error && error.status === 401) return handleAppAuthLoss();
        if (getSelected() === sid) setToast(`${successToast.replace("ed", "")} error: ${error && error.message ? error.message : "unknown error"}`);
      }
    }
    async function stageFiles(files, { sid = getSelected(), source = "picker" } = {}) {
      const uploadFiles = Array.from(files || []).filter(Boolean);
      if (!sid || !uploadFiles.length) return false;
      let successes = 0;
      let blocker = "";
      const failures = [];
      for (let index = 0; index < uploadFiles.length; index += 1) {
        const file = uploadFiles[index];
        try {
          if (getSelected() !== sid || (blocker = attachmentBlockerForSession(sid))) break;
          setToast(uploadFiles.length > 1 ? `uploading ${index + 1}/${uploadFiles.length}...` : "uploading file...");
          let blob = file;
          let filename = file.name || (source === "paste" ? `pasted-${now()}.png` : "file");
          if (looksLikeImage(file) && (file.size > attachmentUploadMaxBytes || isLikelyHeic(file))) {
            filename = `${safeAttachmentStem(filename)}.jpg`;
            for (const settings of [{ maxDim: 2048, quality: .86 }, { maxDim: 1600, quality: .72 }, { maxDim: 1280, quality: .58 }]) {
              blob = await toJpegBlob(file, settings);
              if (blob.size <= attachmentUploadMaxBytes) break;
            }
          }
          const bytes = await blob.arrayBuffer();
          if (bytes.byteLength > attachmentUploadMaxBytes) throw new Error(`file too large (max ${fmtBytes(attachmentUploadMaxBytes)})`);
          const response = await api(`/api/sessions/${sid}/inject_file`, { method: "POST", body: { filename, data_b64: b64FromBytes(new Uint8Array(bytes)) } });
          if (getSelected() === sid && response && response.ok) { successes += 1; setStagedAttachments(response.attachments); }
        } catch (error) {
          if (error && error.status === 401) { handleAppAuthLoss(); return false; }
          failures.push(`${file.name || "file"}: ${error && error.message ? error.message : "unknown error"}`);
        }
      }
      if (getSelected() === sid) {
        if (successes) setToast(successes === 1 ? "file staged" : `${successes} files staged`);
        else setToast(blocker || `attach error: ${failures[0] || "unknown error"}`);
        setPollFastUntilMs(now() + 4000); kickPoll(0); void refreshSessions();
      }
      return successes > 0;
    }
    function clearComposer() { textarea.value = ""; const sid = getSelected(); if (sid) storageRemoveItem(sessionDraftKey(sid)); autoGrow(); }
    function saveSessionDraft(sid) { if (!sid) return; const value = String(textarea.value || ""); if (value) storageSetItem(sessionDraftKey(sid), value); else storageRemoveItem(sessionDraftKey(sid)); }
    function loadSessionDraft(sid) { textarea.value = sid ? storageGetItem(sessionDraftKey(sid)) || "" : ""; autoGrow(); }
    async function sendText(raw, { sid = null } = {}) { return submit(raw, { sid: sid || getSelected(), attachments: stagedAttachments.slice() }); }

    attachBtn.onclick = () => { const blocker = attachmentBlockerForSession(getSelected()); if (blocker) return setToast(blocker); imgInput.value = ""; imgInput.click(); };
    listen(imgInput, "change", () => { const sid = getSelected(); const files = Array.from(imgInput.files || []); imgInput.value = ""; void stageFiles(files, { sid }); });
    listen(textarea, "input", () => { autoGrow(); saveSessionDraft(getSelected()); });
    listen(textarea, "keydown", (event) => { if (event.key === "Enter" && !event.isComposing && (event.ctrlKey || event.metaKey)) { event.preventDefault(); form.requestSubmit(); } });
    listen(textarea, "paste", (event) => { const files = extractFilesFromClipboardData(event.clipboardData); if (!files.length) return; event.preventDefault(); void stageFiles(files, { sid: getSelected(), source: "paste" }); });
    listen(composer, "dragenter", (event) => { if (!dataTransferHasFiles(event.dataTransfer)) return; event.preventDefault(); dragDepth += 1; composer.classList.add("drop-active"); }, { passive: false });
    listen(composer, "dragover", (event) => { if (!dataTransferHasFiles(event.dataTransfer)) return; event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; }, { passive: false });
    listen(composer, "dragleave", (event) => { if (dataTransferHasFiles(event.dataTransfer) && --dragDepth <= 0) { dragDepth = 0; composer.classList.remove("drop-active"); } }, { passive: false });
    listen(composer, "drop", (event) => { if (!dataTransferHasFiles(event.dataTransfer)) return; event.preventDefault(); dragDepth = 0; composer.classList.remove("drop-active"); void stageFiles(extractFilesFromDropData(event.dataTransfer), { sid: getSelected(), source: "drop" }); }, { passive: false });
    form.onsubmit = (event) => { event.preventDefault(); const raw = textarea.value; if (raw && raw.trim()) void sendText(raw).then((ok) => { if (ok && textarea.value === raw) clearComposer(); }); };
    syncSendButtonState(); syncAttachButtonState(); autoGrow();
    return Object.freeze({ sendText, clearComposer, syncState() { syncSendButtonState(); syncAttachButtonState(); }, syncStagedAttachmentsFromSelectedSession, saveSessionDraft, loadSessionDraft, stageFiles, getStagedAttachments: () => stagedAttachments.slice(), dispose() { form.onsubmit = null; while (cleanups.length) cleanups.pop()(); } });
  }

  window.CodoxearComposer = Object.freeze({ createComposerController });
})();
