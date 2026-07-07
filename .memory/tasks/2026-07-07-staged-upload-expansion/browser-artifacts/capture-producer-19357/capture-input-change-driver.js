// Browser proof driver used on http://127.0.0.1:19357/#session=capture-proof.
// It exercises the real #captureInput change listener: stageFiles is closure-local
// and not callable from window, so the observed captured-* staged filename proves
// the listener path, not a direct helper call.
(async () => {
  const input = document.querySelector("#captureInput");
  const beforeText = document.querySelector("#msg")?.value || "";
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
    const chips = [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => el.textContent);
    if (chips.length) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  const api = await fetch("/api/sessions/capture-proof/attachments").then((response) => response.json());
  return {
    beforeText,
    afterText: document.querySelector("#msg")?.value || "",
    chipTexts: [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => el.textContent),
    badge: document.querySelector("#attachBadge")?.textContent || "",
    toast: document.querySelector("#toast")?.textContent || "",
    api,
  };
})();
