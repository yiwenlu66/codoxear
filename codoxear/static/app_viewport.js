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

  window.CodoxearViewport = Object.freeze({
    isMobile,
    prefersReducedMotion,
    useDesktopSessionActions,
    useTouchFileEditorControls,
  });
})();
