/* Application-owned event registration with one cleanup authority. */
(function installCodoxearEventBindings(global) {
  "use strict";

  function createEventBindings(options = {}) {
    const addEvent = options.addEvent;
    if (typeof addEvent !== "function") throw new TypeError("event bindings dependency missing: addEvent");

    function on(target, type, handler, listenerOptions) {
      if (!target || typeof handler !== "function") throw new TypeError(`event binding requires ${type} target and handler`);
      return addEvent(target, type, handler, listenerOptions);
    }

    function onClick(target, handler, listenerOptions) {
      return on(target, "click", handler, listenerOptions);
    }

    return Object.freeze({ on, onClick });
  }

  global.CodoxearEventBindings = Object.freeze({ createEventBindings });
})(window);
