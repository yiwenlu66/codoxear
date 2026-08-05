(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`iOS viewport controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`iOS viewport controller dependency missing: ${name}`);
    return value;
  }

  function createIOSViewportController(options = {}) {
    const windowTarget = requireNode(options.windowTarget || global, "windowTarget");
    const documentTarget = requireNode(options.documentTarget || global.document, "documentTarget");
    const textarea = requireNode(options.textarea, "textarea");
    const isTextEntryElement = requireFunction(options.isTextEntryElement, "isTextEntryElement");
    const updateAppHeightVar = requireFunction(options.updateAppHeightVar, "updateAppHeightVar");
    const transcriptScrollRuntime = requireNode(options.transcriptScrollRuntime, "transcriptScrollRuntime");
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const requestFrame = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");
    const setTimeoutFn = requireFunction(options.setTimeout, "setTimeout");
    const clearTimeoutFn = requireFunction(options.clearTimeout, "clearTimeout");
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const navigatorTarget = options.navigatorTarget || global.navigator || {};
    const isIOS = /iP(hone|od|ad)/.test(navigatorTarget.userAgent || "") ||
      (navigatorTarget.platform === "MacIntel" && navigatorTarget.maxTouchPoints && navigatorTarget.maxTouchPoints > 1);
    let guardTimer = null;
    let guardUntil = 0;

    function activeTextEntryElement() {
      const active = documentTarget.activeElement;
      return isTextEntryElement(active) ? active : null;
    }

    function normalizePageScroll() {
      if (!isIOS) return;
      const activeEntry = activeTextEntryElement();
      if (activeEntry && activeEntry !== textarea) return;
      const y = windowTarget.scrollY || documentTarget.documentElement.scrollTop || documentTarget.body.scrollTop || 0;
      if (y <= 0) return;
      windowTarget.scrollTo(0, 0);
      documentTarget.documentElement.scrollTop = 0;
      documentTarget.body.scrollTop = 0;
    }

    function stopIOSViewportGuard() {
      if (guardTimer) clearTimeoutFn(guardTimer);
      guardTimer = null;
      guardUntil = 0;
    }

    function isIOSViewportGuardActive() {
      return isIOS && now() < guardUntil;
    }

    function runIOSViewportGuard({ preserveChatBottom, durationMs = 1400 } = {}) {
      if (!isIOS) return;
      stopIOSViewportGuard();
      guardUntil = now() + Math.max(0, Number(durationMs) || 0);
      const tick = () => {
        const activeEntry = activeTextEntryElement();
        if (activeEntry && activeEntry !== textarea) {
          stopIOSViewportGuard();
          return;
        }
        updateAppHeightVar();
        normalizePageScroll();
        if (preserveChatBottom && transcriptScrollRuntime.shouldAutoScrollOrNearBottom()) transcriptScrollRuntime.scrollToBottom();
        if (!isIOSViewportGuardActive()) {
          guardTimer = null;
          return;
        }
        guardTimer = setTimeoutFn(tick, 50);
      };
      tick();
    }

    function onViewportShift() {
      updateAppHeightVar();
      if (!isIOS) return;
      const activeEntry = activeTextEntryElement();
      if (activeEntry && activeEntry !== textarea) {
        stopIOSViewportGuard();
        return;
      }
      if (documentTarget.activeElement === textarea || isIOSViewportGuardActive()) {
        normalizePageScroll();
        if (transcriptScrollRuntime.shouldAutoScrollOrNearBottom()) transcriptScrollRuntime.scheduleScrollToBottom();
      }
    }

    function handleComposerFocus() {
      const wasNear = transcriptScrollRuntime.isNearBottom();
      if (wasNear) {
        transcriptScrollRuntime.enableAutoScroll();
        transcriptScrollRuntime.syncJumpButton();
      }
      if (isIOS) {
        runIOSViewportGuard({ preserveChatBottom: wasNear, durationMs: 1800 });
        return;
      }
      const tick = () => {
        updateAppHeightVar();
        if (wasNear) transcriptScrollRuntime.scrollToBottom();
      };
      requestFrame(tick);
      setTimeoutFn(tick, 120);
    }

    function handleComposerBlur() {
      setTimeoutFn(() => {
        if (isIOS) {
          const activeEntry = activeTextEntryElement();
          if (activeEntry && activeEntry !== textarea) {
            stopIOSViewportGuard();
            updateAppHeightVar();
            return;
          }
          runIOSViewportGuard({ preserveChatBottom: false, durationMs: 900 });
          return;
        }
        updateAppHeightVar();
      }, 0);
    }

    if (windowTarget.visualViewport) {
      addAppEvent(windowTarget.visualViewport, "resize", onViewportShift);
      addAppEvent(windowTarget.visualViewport, "scroll", onViewportShift);
    }
    textarea.addEventListener("focus", handleComposerFocus, { passive: true });
    textarea.addEventListener("blur", handleComposerBlur, { passive: true });

    return Object.freeze({
      isIOS: () => isIOS,
      isIOSViewportGuardActive,
      runIOSViewportGuard,
      stopIOSViewportGuard,
      dispose: stopIOSViewportGuard,
    });
  }

  global.CodoxearIOSViewport = { createIOSViewportController };
})(window);
