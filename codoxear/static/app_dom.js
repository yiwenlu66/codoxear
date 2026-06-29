(function () {
  "use strict";

  function createElement(tag, attrs = {}, children = [], defaultButtonTooltip = null) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else n.setAttribute(k, v);
    }
    if (tag === "button" && !n.getAttribute("title") && typeof defaultButtonTooltip === "function") {
      const tooltip = defaultButtonTooltip(attrs, n);
      if (tooltip) n.setAttribute("title", tooltip);
    }
    for (const c of children) n.appendChild(c);
    return n;
  }

  window.CodoxearDom = Object.freeze({
    createElement,
  });
})();
