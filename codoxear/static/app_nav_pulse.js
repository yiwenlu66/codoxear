(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`navigation pulse controller dependency missing: ${name}`);
    return value;
  }

  function createNavigationPulseController(options = {}) {
    const setActiveRow = requireFunction(options.setActiveRow, "setActiveRow");
    const activeElementIsCopyButton = requireFunction(options.activeElementIsCopyButton, "activeElementIsCopyButton");
    const setTimeout = requireFunction(options.setTimeout, "setTimeout");

    function pulseNavigatedRow(row) {
      if (!row) return;
      setActiveRow(row, { focusCopy: activeElementIsCopyButton() });
      row.classList.remove("nav-pulse");
      void row.offsetWidth;
      row.classList.add("nav-pulse");
      setTimeout(() => row.classList.remove("nav-pulse"), 1400);
    }

    return Object.freeze({ pulseNavigatedRow });
  }

  global.CodoxearNavigationPulse = Object.freeze({ createNavigationPulseController });
})(window);
