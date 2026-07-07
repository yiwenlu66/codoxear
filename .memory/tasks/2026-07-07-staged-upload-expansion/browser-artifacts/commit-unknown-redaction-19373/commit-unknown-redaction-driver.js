// Browser proof driver for commit-unknown staged-attachment preview redaction.
(async () => {
  const sid = "commit-unknown-redaction";
  const uploadRoot = "/home/tester/.local/share/codoxear/uploads";
  const prompt = "commit unknown redaction prompt";
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
  await waitFor(() => document.querySelector("#imgInput") && document.querySelector("#msg"), "composer");

  const capturedInjectResponses = [];
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    const rawUrl = args[0] && typeof args[0] === "object" && "url" in args[0] ? args[0].url : args[0];
    const url = String(rawUrl || "");
    if (url.includes(`/api/sessions/${sid}/inject_file`)) {
      try { capturedInjectResponses.push(await response.clone().json()); }
      catch (err) { capturedInjectResponses.push({ capture_error: String(err && err.message ? err.message : err) }); }
    }
    return response;
  };

  const input = document.querySelector("#imgInput");
  const dt = new DataTransfer();
  dt.items.add(new File(["secret bytes"], "commit-secret.txt", { type: "text/plain", lastModified: Date.now() }));
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
  await waitFor(() => document.querySelectorAll(".stagedAttachmentChip").length === 1 && capturedInjectResponses.length === 1, "one staged chip");

  const chipsBeforeSend = [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({ text: el.textContent || "", title: el.getAttribute("title") || "" }));
  const attachmentsBeforeSend = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());

  const msg = document.querySelector("#msg");
  msg.value = prompt;
  msg.dispatchEvent(new Event("input", { bubbles: true }));
  document.querySelector("#sendBtn").click();

  const row = await waitFor(async () => {
    const sessionsApi = await fetch("/api/sessions").then((response) => response.json());
    const rows = Array.isArray(sessionsApi) ? sessionsApi : Array.isArray(sessionsApi.sessions) ? sessionsApi.sessions : [];
    return rows.find((item) => item && item.session_id === sid && item.commit_unknown_send) || null;
  }, "commit unknown session row");
  const attachmentsAfterUnknown = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
  const bodyText = document.body.innerText || "";
  const publicPayload = { row, attachmentsAfterUnknown, bodyText, chipsBeforeSend, attachmentsBeforeSend, capturedInjectResponses };
  const publicJson = JSON.stringify(publicPayload);
  return {
    uploadRoot,
    rowCommitUnknownText: row.commit_unknown_send_text,
    rowContainsUploadRoot: JSON.stringify(row).includes(uploadRoot),
    publicPayloadContainsUploadRoot: publicJson.includes(uploadRoot),
    rowCommitUnknownTextContainsAttachmentLine: String(row.commit_unknown_send_text || "").includes("Attachment 1:"),
    rowCommitUnknownTextEqualsPrompt: row.commit_unknown_send_text === prompt,
    attachmentsAfterUnknown,
    stagedEntriesHavePathKey: (attachmentsAfterUnknown.attachments || []).some((entry) => entry && Object.prototype.hasOwnProperty.call(entry, "path")),
    chipsBeforeSend,
    chipTitlesContainSlash: chipsBeforeSend.some((chip) => chip.title.includes("/")),
    bodyContainsUploadRoot: bodyText.includes(uploadRoot),
    toast: document.querySelector("#toast")?.textContent || "",
  };
})();
