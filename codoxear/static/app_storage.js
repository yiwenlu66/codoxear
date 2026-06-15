(function () {
  "use strict";

  function optionalLocalStorage() {
    try {
      const store = window.localStorage;
      return store && typeof store.getItem === "function" ? store : null;
    } catch (_) {
      return null;
    }
  }

  function getItem(key) {
    const store = optionalLocalStorage();
    if (!store) return null;
    try {
      return store.getItem(String(key));
    } catch (_) {
      return null;
    }
  }

  function setItem(key, value) {
    const store = optionalLocalStorage();
    if (!store) return false;
    try {
      store.setItem(String(key), String(value));
      return true;
    } catch (_) {
      return false;
    }
  }

  function removeItem(key) {
    const store = optionalLocalStorage();
    if (!store) return false;
    try {
      store.removeItem(String(key));
      return true;
    } catch (_) {
      return false;
    }
  }

  window.CodoxearStorage = Object.freeze({
    optionalLocalStorage,
    getItem,
    setItem,
    removeItem,
  });
})();
