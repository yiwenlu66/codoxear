// Browser proof driver for per-file attachment blocker recheck.
(async () => {
  const sid = "batch-blocker-proof";
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  async function waitFor(predicate, label, timeoutMs = 12000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const value = await predicate();
      if (value) return value;
      await sleep(100);
    }
    throw new Error(`timed out waiting for ${label}`);
  }
  await waitFor(() => document.querySelector("#imgInput") && document.querySelector(".stagedAttachments"), "attachment UI");
  const input = document.querySelector("#imgInput");

  const injectRequests = [];
  const injectResponses = [];
  const fetchTimeline = [];
  const markerResponses = [];
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const rawUrl = args[0] && typeof args[0] === "object" && "url" in args[0] ? args[0].url : args[0];
    const url = String(rawUrl || "");
    const isInject = url.includes(`/api/sessions/${sid}/inject_file`);
    let requestIndex = 0;
    if (isInject) {
      requestIndex = injectRequests.length + 1;
      const opts = args[1] || {};
      try { injectRequests.push(JSON.parse(String(opts.body || "{}"))); }
      catch (err) { injectRequests.push({ parse_error: String(err && err.message ? err.message : err), raw: String(opts.body || "") }); }
      fetchTimeline.push({ event: "inject-start", requestIndex, ts: Date.now() });
    }
    const response = await originalFetch(...args);
    if (isInject) {
      try { injectResponses.push(await response.clone().json()); }
      catch (err) { injectResponses.push({ capture_error: String(err && err.message ? err.message : err) }); }
      fetchTimeline.push({ event: "inject-response", requestIndex, ts: Date.now() });
      if (requestIndex === 1) {
        const markerResponse = await originalFetch(`/api/sessions/${sid}/send`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: "FORCE_COMMIT_UNKNOWN batch blocker marker", allow_pending_attachment: true }),
        });
        try { markerResponses.push({ status: markerResponse.status, body: await markerResponse.clone().json() }); }
        catch (err) { markerResponses.push({ status: markerResponse.status, capture_error: String(err && err.message ? err.message : err) }); }
        fetchTimeline.push({ event: "commit-unknown-marker", requestIndex, status: markerResponse.status, ts: Date.now() });
        await sleep(6500);
        fetchTimeline.push({ event: "inject-release", requestIndex, ts: Date.now() });
      }
    }
    return response;
  };

  const dt = new DataTransfer();
  dt.items.add(new File(["first"], "first.txt", { type: "text/plain", lastModified: Date.now() }));
  dt.items.add(new File(["second"], "second.txt", { type: "text/plain", lastModified: Date.now() }));
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));

  await waitFor(() => injectRequests.length >= 1 && injectResponses.length >= 1, "first upload response captured");
  await sleep(7500);
  const attachmentsAfterBatch = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
  const sessionsApi = await fetch("/api/sessions").then((response) => response.json());
  const rows = Array.isArray(sessionsApi) ? sessionsApi : Array.isArray(sessionsApi.sessions) ? sessionsApi.sessions : [];
  const row = rows.find((item) => item && item.session_id === sid) || null;
  const chips = [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({ text: el.textContent || "", title: el.getAttribute("title") || "" }));
  return {
    injectRequestCount: injectRequests.length,
    injectFilenames: injectRequests.map((request) => request.filename),
    injectResponses,
    fetchTimeline,
    markerResponses,
    attachmentsAfterBatch,
    stagedCountAfterBatch: (attachmentsAfterBatch.attachments || []).length,
    stagedEntriesHavePathKey: (attachmentsAfterBatch.attachments || []).some((entry) => entry && Object.prototype.hasOwnProperty.call(entry, "path")),
    rowStateBusy: row ? row.state_busy : null,
    rowStagedCount: row && Array.isArray(row.staged_attachments) ? row.staged_attachments.length : null,
    chips,
    chipTitlesContainSlash: chips.some((chip) => chip.title.includes("/")),
    toast: document.querySelector("#toast")?.textContent || "",
    stoppedToast: /attached 1; stopped:/.test(document.querySelector("#toast")?.textContent || ""),
  };
})();
