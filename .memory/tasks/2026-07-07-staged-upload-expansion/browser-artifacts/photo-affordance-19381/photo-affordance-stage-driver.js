// Exercises the real #captureInput change listener and delays /inject_file so
// the transient user-facing progress toast can be observed before completion.
(async () => {
  const sid = "photo-proof";
  const button = document.querySelector("#captureBtn");
  const input = document.querySelector("#captureInput");
  const msg = document.querySelector("#msg");
  const before = {
    buttonTitle: button && button.title,
    buttonAria: button && button.getAttribute("aria-label"),
    inputAccept: input && input.getAttribute("accept"),
    inputCapture: input && input.getAttribute("capture"),
    chipCount: document.querySelectorAll(".stagedAttachmentChip").length,
    text: msg && msg.value,
  };
  const originalFetch = window.fetch.bind(window);
  const timeline = [];
  let releaseFirstInject;
  const injectGate = new Promise((resolve) => { releaseFirstInject = resolve; });
  window.fetch = async (inputArg, init) => {
    const url = typeof inputArg === "string" ? inputArg : String(inputArg && inputArg.url || "");
    if (url.includes(`/api/sessions/${sid}/inject_file`)) {
      timeline.push({ event: "inject-start", toast: document.querySelector("#toast")?.textContent || "", ts: Date.now() });
      const response = await originalFetch(inputArg, init);
      timeline.push({ event: "inject-response", toast: document.querySelector("#toast")?.textContent || "", ts: Date.now() });
      await injectGate;
      timeline.push({ event: "inject-release", toast: document.querySelector("#toast")?.textContent || "", ts: Date.now() });
      return response;
    }
    return originalFetch(inputArg, init);
  };
  const file = new File(
    [new Uint8Array([255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 255, 217])],
    "",
    { type: "image/jpeg", lastModified: Date.now() }
  );
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
  const start = Date.now();
  while (Date.now() - start < 6000) {
    if (timeline.some((entry) => entry.event === "inject-response")) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  const mid = {
    toast: document.querySelector("#toast")?.textContent || "",
    timeline: timeline.slice(),
  };
  releaseFirstInject();
  const stageStart = Date.now();
  while (Date.now() - stageStart < 6000) {
    const chips = [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({ text: el.textContent, title: el.title }));
    if (chips.length) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  const api = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
  const sessions = await fetch("/api/sessions").then((response) => response.json());
  return {
    before,
    mid,
    after: {
      buttonTitle: button && button.title,
      buttonAria: button && button.getAttribute("aria-label"),
      toast: document.querySelector("#toast")?.textContent || "",
      text: msg && msg.value,
      chips: [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({ text: el.textContent, title: el.title })),
      badge: document.querySelector("#attachBadge")?.textContent || "",
      api,
      sessionsRow: (sessions.sessions || []).find((row) => row.session_id === sid) || null,
      publicPayloadContainsPath: JSON.stringify({ api, sessions }).includes("/home/tester/.local/share/codoxear/uploads"),
      stagedEntriesHavePathKey: (api.attachments || []).some((item) => Object.prototype.hasOwnProperty.call(item, "path")),
    },
  };
})();
