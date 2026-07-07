(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const sid = "copy-count-session";
  window.__copiedTexts = [];
  try {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text) => {
          window.__copiedTexts.push(String(text));
        },
      },
    });
  } catch (_err) {}
  if (!location.hash.includes(sid)) location.hash = `#session=${sid}`;
  await sleep(900);
  const row = [...document.querySelectorAll(".session")].find((node) => (node.innerText || "").includes("copy-count-proof"));
  if (row) row.click();
  await sleep(900);
  const btn = document.querySelector("#copyConversationBtn");
  if (!btn) return { ok: false, reason: "missing copyConversationBtn" };
  const exportResponse = await fetch(`/api/sessions/${sid}/messages/export`);
  const exportBody = await exportResponse.json();
  btn.click();
  await sleep(700);
  const toast = (document.querySelector("#toast") || {}).innerText || "";
  const copied = window.__copiedTexts.slice();
  const text = copied[0] || "";
  return {
    ok: true,
    selectedHash: location.hash,
    exportStatus: exportResponse.status,
    exportEventCount: Array.isArray(exportBody.events) ? exportBody.events.length : null,
    exportRoles: Array.isArray(exportBody.events) ? exportBody.events.map((ev) => ({ role: ev.role, text: ev.text, textLength: String(ev.text || "").length })) : null,
    toast,
    copied,
    copiedSectionCount: (text.match(/^## /gm) || []).length,
    hasExpectedTwoMessageToast: /Copied 2 messages/.test(toast),
    hasRawEventOvercountToast: /Copied 4 messages/.test(toast),
    clipboardHasBlankSections: /^## .*\n\n\s*$/m.test(text),
    clipboardHasUser: text.includes("first copied user turn"),
    clipboardHasAssistant: text.includes("assistant copied answer"),
    clipboardExcludesBlankOnlyRows: !/^## .*\n\n\s*---/m.test(text),
    bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth || document.body.scrollWidth > document.body.clientWidth,
  };
})()
