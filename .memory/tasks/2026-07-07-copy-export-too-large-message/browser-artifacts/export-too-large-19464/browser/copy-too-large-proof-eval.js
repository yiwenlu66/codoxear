(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  window.__clipboardWriteCount = 0;
  try {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async () => {
          window.__clipboardWriteCount += 1;
          throw new Error("clipboard should not be reached for oversized export");
        },
      },
    });
  } catch (_err) {}

  const sid = "export-too-large-session";
  if (!location.hash.includes(sid)) location.hash = `#session=${sid}`;
  await sleep(800);
  const row = [...document.querySelectorAll(".session")].find((node) => (node.innerText || "").includes("oversized-export-proof"));
  if (row) row.click();
  await sleep(500);
  const btn = document.querySelector("#copyConversationBtn");
  if (!btn) return { ok: false, reason: "missing copyConversationBtn" };
  const before = { disabled: btn.disabled, toast: (document.querySelector("#toast") || {}).innerText || "", selectedHash: location.hash };
  btn.click();
  await sleep(1200);
  const toast = (document.querySelector("#toast") || {}).innerText || "";
  const exportResponse = await fetch(`/api/sessions/${sid}/messages/export`);
  const exportBody = await exportResponse.json();
  return {
    ok: true,
    before,
    after: {
      toast,
      copyButtonDisabled: btn.disabled,
      clipboardWriteCount: window.__clipboardWriteCount,
      selectedHash: location.hash,
      hasGenericCopyFailed: /copy failed/i.test(toast),
      hasSpecificConversationTooLarge: /conversation too large to copy/i.test(toast),
      hasLimit: /1 KiB|1024/.test(toast),
      exportStatus: exportResponse.status,
      exportBody,
      bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth || document.body.scrollWidth > document.body.clientWidth,
    },
  };
})()
