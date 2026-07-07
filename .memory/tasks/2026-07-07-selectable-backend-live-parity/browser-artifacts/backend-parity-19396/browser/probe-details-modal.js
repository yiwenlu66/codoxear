(() => ({
  bodyText: document.body.innerText,
  diag: {
    display: getComputedStyle(document.querySelector("#diagViewer")).display,
    text: document.querySelector("#diagViewer")?.innerText || "",
    buttons: Array.from(document.querySelectorAll("#diagViewer button")).map((b) => ({
      id: b.id,
      text: b.innerText,
      title: b.title,
      aria: b.getAttribute("aria-label"),
      disabled: b.disabled,
    })),
  },
}))()