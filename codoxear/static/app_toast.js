(function () {
  "use strict";

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !("textContent" in value))
      throw new TypeError(`toast dependency missing: ${name}`);
    return value;
  }

  function createToastController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("toast dependency missing: options");
    const toast = requireNode(options.toast, "toast");
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : setTimeout;
    const dismissAfterMs = Number.isFinite(options.dismissAfterMs) ? Math.max(0, options.dismissAfterMs) : 2200;

    function show(value) {
      const text = value ? String(value) : "";
      toast.textContent = text;
      if (!text) return "";
      setTimeoutFn(() => {
        if (toast.textContent === text) toast.textContent = "";
      }, dismissAfterMs);
      return text;
    }

    return Object.freeze({ show });
  }

  window.CodoxearToast = Object.freeze({ createToastController });
})();
