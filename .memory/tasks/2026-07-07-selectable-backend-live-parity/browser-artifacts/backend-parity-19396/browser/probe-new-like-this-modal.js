(() => ({
  bodyText: document.body.innerText,
  inputs: Array.from(document.querySelectorAll("input,textarea")).map((x, i) => ({
    i,
    type: x.type,
    value: x.value,
    aria: x.getAttribute("aria-label"),
    checked: x.checked,
    disabled: x.disabled,
  })),
  buttons: Array.from(document.querySelectorAll("button")).map((b, i) => ({
    i,
    text: b.innerText,
    aria: b.getAttribute("aria-label"),
    title: b.title,
    disabled: b.disabled,
    expanded: b.getAttribute("aria-expanded"),
  })),
}))()