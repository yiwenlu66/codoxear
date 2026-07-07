(async () => {
  const sid = "photo-proof";
  const msg = document.querySelector("#msg");
  msg.value = "please process added photo";
  msg.dispatchEvent(new Event("input", { bubbles: true }));
  document.querySelector("#sendBtn").click();
  const start = Date.now();
  while (Date.now() - start < 6000) {
    const api = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
    if (!api.pending_attachment && (!api.attachments || api.attachments.length === 0)) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  const api = await fetch(`/api/sessions/${sid}/attachments`).then((response) => response.json());
  return {
    toast: document.querySelector("#toast")?.textContent || "",
    chips: [...document.querySelectorAll(".stagedAttachmentChip")].map((el) => ({ text: el.textContent, title: el.title })),
    badge: document.querySelector("#attachBadge")?.textContent || "",
    api,
  };
})();
