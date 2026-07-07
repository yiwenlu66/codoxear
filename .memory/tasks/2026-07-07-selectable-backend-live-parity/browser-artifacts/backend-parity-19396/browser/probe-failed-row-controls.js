(() => {
  const selected = localStorage.getItem("codexweb.selected");
  const row = document.querySelector(".session.active") || document.querySelector(".session");
  const panel = document.querySelector(".recovery-panel");
  const msg = document.querySelector("#msg");
  const send = document.querySelector("#sendBtn");
  const queue = document.querySelector("#queueBtn");
  const attach = document.querySelector("#attachBtn");
  const capture = document.querySelector("#captureBtn");
  const file = document.querySelector("#fileBtn");
  const details = document.querySelector("#diagBtn");
  return {
    selected,
    title: document.querySelector("#titleLabel")?.innerText || "",
    rowText: row?.innerText || "",
    rowHtml: row?.outerHTML.slice(0, 3000) || "",
    panelText: panel?.innerText || "",
    panelHtml: panel?.outerHTML.slice(0, 3000) || "",
    controls: {
      msg: { disabled: msg?.disabled, aria: msg?.getAttribute("aria-label"), title: msg?.title, placeholder: document.querySelector("#msgPh")?.innerText },
      send: { disabled: send?.disabled, aria: send?.getAttribute("aria-label"), title: send?.title },
      queue: { disabled: queue?.disabled, aria: queue?.getAttribute("aria-label"), title: queue?.title },
      attach: { disabled: attach?.disabled, aria: attach?.getAttribute("aria-label"), title: attach?.title },
      capture: { disabled: capture?.disabled, aria: capture?.getAttribute("aria-label"), title: capture?.title },
      file: { disabled: file?.disabled, aria: file?.getAttribute("aria-label"), title: file?.title },
      details: { disabled: details?.disabled, aria: details?.getAttribute("aria-label"), title: details?.title },
    },
    buttons: Array.from(document.querySelectorAll(".recovery-panel button")).map((b) => ({ text: b.innerText, title: b.title, disabled: b.disabled })),
  };
})()