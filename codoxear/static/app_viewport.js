(function () {
  "use strict";

  function mediaQueryMatches(query) {
    return Boolean(window.matchMedia && window.matchMedia(query).matches);
  }

  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 880px)").matches;
  }

  function prefersReducedMotion() {
    return mediaQueryMatches("(prefers-reduced-motion: reduce)");
  }

  function useDesktopSessionActions() {
    return mediaQueryMatches("(hover: hover) and (pointer: fine) and (min-width: 881px)");
  }

  function useTouchFileEditorControls() {
    return mediaQueryMatches("(pointer: coarse)") || mediaQueryMatches("(hover: none)");
  }

  function isTextEntryElement(target) {
    const el = target instanceof Element ? target.closest("textarea, input, [contenteditable], [contenteditable=''], [contenteditable='true']") : null;
    if (!(el instanceof HTMLElement)) return false;
    if (el.tagName !== "INPUT") return true;
    const type = String(el.getAttribute("type") || "text").toLowerCase();
    return !["button", "checkbox", "color", "file", "hidden", "image", "radio", "range", "reset", "submit"].includes(type);
  }

  function updateAppHeightVar() {
    const vv = window.visualViewport;
    const layoutH = Math.round(window.innerHeight);
    const visualH = Math.round(vv ? vv.height : window.innerHeight);
    const visualTop = Math.max(0, Math.round(vv ? vv.offsetTop : 0));
    const visualBottom = Math.max(0, layoutH - visualH - visualTop);
    if (updateAppHeightVar._h === visualH && updateAppHeightVar._l === layoutH && updateAppHeightVar._t === visualTop && updateAppHeightVar._b === visualBottom) return;
    updateAppHeightVar._h = visualH;
    updateAppHeightVar._l = layoutH;
    updateAppHeightVar._t = visualTop;
    updateAppHeightVar._b = visualBottom;
    document.documentElement.style.setProperty("--appH", `${visualH}px`);
    document.documentElement.style.setProperty("--layoutH", `${layoutH}px`);
    document.documentElement.style.setProperty("--vvTop", `${visualTop}px`);
    document.documentElement.style.setProperty("--vvBottom", `${visualBottom}px`);
  }

  window.CodoxearViewport = Object.freeze({
    isMobile,
    prefersReducedMotion,
    useDesktopSessionActions,
    useTouchFileEditorControls,
    isTextEntryElement,
    updateAppHeightVar,
  });
})();
