// Browser proof driver for staged-upload producer polish on http://127.0.0.1:19377/#session=upload-proof.
(async () => {
  const sid = "upload-proof";
  const prompt = "process polished upload producers";
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  async function waitFor(predicate, label, timeoutMs = 10000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const value = await predicate();
      if (value) return value;
      await sleep(100);
    }
    throw new Error(`timed out waiting for ${label}`);
  }
  function eventWithData(type, dataTransfer, init = {}) {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", { value: dataTransfer });
    for (const [key, value] of Object.entries(init)) Object.defineProperty(event, key, { value });
    return event;
  }
  await waitFor(() => document.querySelector("#msg") && document.querySelector(".composer"), "composer");
  const textarea = document.querySelector("#msg");
  const composer = document.querySelector(".composer");

  const injectRequests = [];
  const injectResponses = [];
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    const rawUrl = args[0] && typeof args[0] === "object" && "url" in args[0] ? args[0].url : args[0];
    const url = String(rawUrl || "");
    if (url.includes(`/api/sessions/${sid}/inject_file`)) {
      const opts = args[1] || {};
      try { injectRequests.push(JSON.parse(String(opts.body || "{}"))); }
      catch (err) { injectRequests.push({ parse_error: String(err && err.message ? err.message : err), raw: String(opts.body || "") }); }
      try { injectResponses.push(await response.clone().json()); }
      catch (err) { injectResponses.push({ capture_error: String(err && err.message ? err.message : err) }); }
    }
    return response;
  };

  textarea.value = "prefix-suffix";
  textarea.selectionStart = "prefix-".length;
  textarea.selectionEnd = "prefix-".length;
  textarea.dispatchEvent(new Event("input", { bubbles: true }));

  const jpegPasteFile = new File(["jpeg-bytes"], "", { type: "image/jpeg", lastModified: Date.now() });
  const webpPasteFile = new File(["webp-bytes"], "", { type: "image/webp", lastModified: Date.now() });
  const pasteData = {
    items: [
      { kind: "file", getAsFile: () => jpegPasteFile },
      { kind: "file", getAsFile: () => webpPasteFile },
    ],
    files: [jpegPasteFile, webpPasteFile],
    getData: (kind) => (kind === "text/plain" || kind === "text" ? "MIXED-TEXT" : ""),
  };
  const pasteEvent = new Event("paste", { bubbles: true, cancelable: true });
  Object.defineProperty(pasteEvent, "clipboardData", { value: pasteData });
  const pasteDispatchReturned = textarea.dispatchEvent(pasteEvent);
  await waitFor(() => injectResponses.length === 2 && document.querySelectorAll(".stagedAttachmentChip").length === 2, "mixed paste staging");
  const textareaAfterMixedPaste = textarea.value;
  const attachmentsAfterPaste = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
  const chipsAfterPaste = [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({ text: el.textContent || "", title: el.getAttribute("title") || "" }));

  const dragData = new DataTransfer();
  dragData.items.add(new File(["drag"], "drag-polish.txt", { type: "text/plain", lastModified: Date.now() }));
  const dragEnter = eventWithData("dragenter", dragData);
  const dragEnterReturned = composer.dispatchEvent(dragEnter);
  const activeAfterDragEnter = composer.classList.contains("drop-active");
  const windowLeave = eventWithData("dragleave", dragData, { clientX: -1, clientY: 10, relatedTarget: null });
  const windowLeaveReturned = window.dispatchEvent(windowLeave);
  const activeAfterWindowLeave = composer.classList.contains("drop-active");

  const dragEnterAgain = eventWithData("dragenter", dragData);
  composer.dispatchEvent(dragEnterAgain);
  const activeBeforeOffComposerDrop = composer.classList.contains("drop-active");
  const offComposerDrop = eventWithData("drop", dragData);
  const offComposerDropReturned = window.dispatchEvent(offComposerDrop);
  await sleep(250);
  const activeAfterOffComposerDrop = composer.classList.contains("drop-active");
  const attachmentsAfterOffComposerDrop = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());

  textarea.value = prompt;
  textarea.selectionStart = prompt.length;
  textarea.selectionEnd = prompt.length;
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
  document.querySelector("#sendBtn").click();
  await waitFor(() => document.querySelectorAll(".stagedAttachmentChip").length === 0, "chips cleared after send");
  const attachmentsAfterSend = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());

  return {
    pasteDefaultPrevented: pasteEvent.defaultPrevented,
    pasteDispatchReturned,
    textareaAfterMixedPaste,
    mixedPasteTextPreservedBeforeSend: textareaAfterMixedPaste === "prefix-MIXED-TEXTsuffix",
    injectFilenames: injectRequests.map((request) => request.filename),
    injectResponses,
    attachmentsAfterPaste,
    chipsAfterPaste,
    pastedNamesHaveMimeExtensions: injectRequests.some((request) => String(request.filename || "").endsWith(".jpg")) && injectRequests.some((request) => String(request.filename || "").endsWith(".webp")),
    publicPastePayloadContainsPathKey: (attachmentsAfterPaste.attachments || []).some((entry) => Object.prototype.hasOwnProperty.call(entry || {}, "path")),
    chipTitlesContainSlash: chipsAfterPaste.some((chip) => chip.title.includes("/")),
    dragEnterDefaultPrevented: dragEnter.defaultPrevented,
    dragEnterDispatchReturned: dragEnterReturned,
    activeAfterDragEnter,
    windowLeaveDefaultPrevented: windowLeave.defaultPrevented,
    windowLeaveDispatchReturned: windowLeaveReturned,
    activeAfterWindowLeave,
    activeBeforeOffComposerDrop,
    offComposerDropDefaultPrevented: offComposerDrop.defaultPrevented,
    offComposerDropDispatchReturned: offComposerDropReturned,
    activeAfterOffComposerDrop,
    attachmentsAfterOffComposerDrop,
    offComposerDropDidNotStage: (attachmentsAfterOffComposerDrop.attachments || []).length === (attachmentsAfterPaste.attachments || []).length,
    attachmentsAfterSend,
    toast: document.querySelector("#toast")?.textContent || "",
  };
})();
