// Browser proof driver for staged-attachment path redaction on http://127.0.0.1:19371/#session=redaction-proof.
// It uses the real #imgInput change listener so UI chips and API responses are produced by the product path.
(async () => {
  const sid = "redaction-proof";
  const uploadRoot = "/home/tester/.local/share/codoxear/uploads";
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  async function waitFor(predicate, label, timeoutMs = 8000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const value = await predicate();
      if (value) return value;
      await sleep(100);
    }
    throw new Error(`timed out waiting for ${label}`);
  }
  await waitFor(() => document.querySelector("#imgInput") && document.querySelector("#msg"), "composer");

  const capturedInjectResponses = [];
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    const rawUrl = args[0] && typeof args[0] === "object" && "url" in args[0] ? args[0].url : args[0];
    const url = String(rawUrl || "");
    if (url.includes(`/api/sessions/${sid}/inject_file`)) {
      try {
        capturedInjectResponses.push(await response.clone().json());
      } catch (err) {
        capturedInjectResponses.push({ capture_error: String(err && err.message ? err.message : err) });
      }
    }
    return response;
  };

  const input = document.querySelector("#imgInput");
  const dt = new DataTransfer();
  dt.items.add(new File(["alpha bytes"], "alpha-secret.txt", { type: "text/plain", lastModified: Date.now() }));
  dt.items.add(new File(["beta bytes"], "beta-secret.txt", { type: "text/plain", lastModified: Date.now() }));
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));

  await waitFor(() => document.querySelectorAll(".stagedAttachmentChip").length === 2 && capturedInjectResponses.length === 2, "two staged chips and two inject responses");
  const chipsBeforeSend = [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({
    text: el.textContent || "",
    title: el.getAttribute("title") || "",
  }));
  const attachmentsApi = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
  const sessionsApi = await fetch("/api/sessions").then((response) => response.json());
  const rows = Array.isArray(sessionsApi) ? sessionsApi : Array.isArray(sessionsApi.sessions) ? sessionsApi.sessions : [];
  const sessionRow = rows.find((row) => row && row.session_id === sid) || null;
  const preSendPublicPayload = { capturedInjectResponses, attachmentsApi, sessionRow, chipsBeforeSend };
  const preSendJson = JSON.stringify(preSendPublicPayload);
  const stagedEntries = []
    .concat(capturedInjectResponses.flatMap((r) => (Array.isArray(r.attachments) ? r.attachments : [])))
    .concat(Array.isArray(attachmentsApi.attachments) ? attachmentsApi.attachments : [])
    .concat(sessionRow && Array.isArray(sessionRow.staged_attachments) ? sessionRow.staged_attachments : []);

  const msg = document.querySelector("#msg");
  msg.value = "process redacted staged attachments";
  msg.dispatchEvent(new Event("input", { bubbles: true }));
  document.querySelector("#sendBtn").click();
  await waitFor(() => document.querySelectorAll(".stagedAttachmentChip").length === 0, "staged chips cleared after send");
  const attachmentsAfterSend = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());

  return {
    uploadRoot,
    chipsBeforeSend,
    capturedInjectResponses,
    attachmentsApi,
    sessionRowStagedAttachments: sessionRow ? sessionRow.staged_attachments : null,
    preSendContainsUploadRoot: preSendJson.includes(uploadRoot),
    preSendStagedEntriesHavePathKey: stagedEntries.some((entry) => entry && Object.prototype.hasOwnProperty.call(entry, "path")),
    chipTitlesContainSlash: chipsBeforeSend.some((chip) => chip.title.includes("/")),
    chipTextsContainSlash: chipsBeforeSend.some((chip) => chip.text.includes("/")),
    attachmentsAfterSend,
    chipsAfterSend: [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => el.textContent || ""),
    toast: document.querySelector("#toast")?.textContent || "",
  };
})();
