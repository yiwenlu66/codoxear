// Theme engine. This module is the sole authoritative writer of the UI theme
// surface: <html data-theme>, <html data-mode>, the theme stylesheet <link>,
// the custom-CSS <style>, and <meta name="theme-color">. The inline boot
// script in index.html renders the same persisted state once before first
// paint; from construction on, every change to those nodes goes through
// applyTheme()/setCustomCss() here.
//
// State: { family: paper|clay|slate, mode: system|light|dark, customCss }.
// Cascade: app.css (base = paper light) -> themes/<family>.css -> custom
// <style>, so the theme link is always inserted before the custom style.

const FAMILIES = Object.freeze(["paper", "clay", "slate"]);
const MODES = Object.freeze(["system", "light", "dark"]);
const DEFAULT_FAMILY = "paper";
const DEFAULT_MODE = "system";
const STORAGE_KEYS = Object.freeze({
  family: "codoxear.ui.theme.family",
  mode: "codoxear.ui.theme.mode",
  customCss: "codoxear.ui.customCss",
});
const THEME_LINK_ID = "codoxearThemeLink";
const CUSTOM_STYLE_ID = "codoxearCustomCss";
const DARK_SCHEME_QUERY = "(prefers-color-scheme: dark)";
// Browser chrome color per family x resolved mode: each family's --panel
// (the topbar/sidebar surface that meets the browser UI).
const THEME_COLORS = Object.freeze({
  paper: Object.freeze({ light: "#ffffff", dark: "#1c1a16" }),
  clay: Object.freeze({ light: "#faf7f0", dark: "#161310" }),
  slate: Object.freeze({ light: "#f9f9f9", dark: "#171717" }),
});

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`theme dependency missing: ${name}`);
  return value;
}

function normalizeFamily(value) {
  const text = String(value ?? "").trim().toLowerCase();
  return FAMILIES.includes(text) ? text : DEFAULT_FAMILY;
}

function normalizeMode(value) {
  const text = String(value ?? "").trim().toLowerCase();
  return MODES.includes(text) ? text : DEFAULT_MODE;
}

function themeStylesheetPath(family) {
  return `themes/${normalizeFamily(family)}.css`;
}

function createThemeController(options = {}) {
  if (!options || typeof options !== "object") throw new TypeError("theme dependency missing: options");
  const documentTarget = options.documentTarget;
  if (!documentTarget || typeof documentTarget !== "object" || !documentTarget.documentElement || !documentTarget.head) {
    throw new TypeError("theme dependency missing: documentTarget");
  }
  const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
  const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
  const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
  const matchMedia = requireFunction(options.matchMedia, "matchMedia");
  const versionedAssetPath = requireFunction(options.versionedAssetPath, "versionedAssetPath");

  const html = documentTarget.documentElement;
  const head = documentTarget.head;
  const darkQuery = matchMedia(DARK_SCHEME_QUERY) || null;
  const subscribers = new Set();
  const state = {
    family: normalizeFamily(storageGetItem(STORAGE_KEYS.family)),
    mode: normalizeMode(storageGetItem(STORAGE_KEYS.mode)),
    customCss: String(storageGetItem(STORAGE_KEYS.customCss) ?? ""),
  };

  function ensureMeta() {
    let meta = documentTarget.querySelector('meta[name="theme-color"]');
    if (!meta) {
      meta = documentTarget.createElement("meta");
      meta.setAttribute("name", "theme-color");
      head.appendChild(meta);
    }
    return meta;
  }

  function ensureCustomStyle() {
    let style = documentTarget.getElementById(CUSTOM_STYLE_ID);
    if (!style) {
      style = documentTarget.createElement("style");
      style.id = CUSTOM_STYLE_ID;
      head.appendChild(style);
    }
    return style;
  }

  const meta = ensureMeta();
  const customStyle = ensureCustomStyle();
  let themeLink = documentTarget.getElementById(THEME_LINK_ID) || null;

  function resolvedMode() {
    if (state.mode !== "system") return state.mode;
    return darkQuery && darkQuery.matches ? "dark" : "light";
  }

  function snapshot() {
    return Object.freeze({
      family: state.family,
      mode: state.mode,
      resolvedMode: resolvedMode(),
      customCss: state.customCss,
    });
  }

  function notify() {
    const current = snapshot();
    for (const subscriber of subscribers) subscriber(current);
  }

  // Swap-on-load: the previous family stylesheet stays applied until the new
  // one has loaded, so a family change never paints an unthemed frame. The
  // link is inserted before the custom style to keep the cascade order.
  function renderThemeLink() {
    const href = versionedAssetPath(themeStylesheetPath(state.family));
    if (themeLink && themeLink.getAttribute("href") === href) return;
    const previous = themeLink;
    const next = documentTarget.createElement("link");
    next.setAttribute("rel", "stylesheet");
    next.setAttribute("href", href);
    next.setAttribute("data-theme-family", state.family);
    const retirePrevious = () => {
      if (previous && previous.parentNode) previous.parentNode.removeChild(previous);
      next.id = THEME_LINK_ID;
    };
    if (previous) {
      previous.removeAttribute("id");
      if (typeof next.addEventListener === "function") {
        next.addEventListener("load", retirePrevious, { once: true });
        next.addEventListener("error", retirePrevious, { once: true });
      } else {
        retirePrevious();
      }
    } else {
      next.id = THEME_LINK_ID;
    }
    head.insertBefore(next, customStyle);
    themeLink = next;
  }

  function render() {
    const mode = resolvedMode();
    html.setAttribute("data-theme", state.family);
    html.setAttribute("data-mode", mode);
    renderThemeLink();
    if (customStyle.textContent !== state.customCss) customStyle.textContent = state.customCss;
    meta.setAttribute("content", THEME_COLORS[state.family][mode]);
  }

  function persist() {
    if (state.family === DEFAULT_FAMILY) storageRemoveItem(STORAGE_KEYS.family);
    else storageSetItem(STORAGE_KEYS.family, state.family);
    if (state.mode === DEFAULT_MODE) storageRemoveItem(STORAGE_KEYS.mode);
    else storageSetItem(STORAGE_KEYS.mode, state.mode);
    if (state.customCss) storageSetItem(STORAGE_KEYS.customCss, state.customCss);
    else storageRemoveItem(STORAGE_KEYS.customCss);
  }

  function applyTheme({ family = state.family, mode = state.mode } = {}) {
    state.family = normalizeFamily(family);
    state.mode = normalizeMode(mode);
    persist();
    render();
    notify();
    return snapshot();
  }

  function setCustomCss(css) {
    state.customCss = String(css ?? "");
    persist();
    render();
    notify();
    return snapshot();
  }

  function reset() {
    state.customCss = "";
    return applyTheme({ family: DEFAULT_FAMILY, mode: DEFAULT_MODE });
  }

  function subscribe(subscriber) {
    requireFunction(subscriber, "subscriber");
    subscribers.add(subscriber);
    subscriber(snapshot());
    return () => subscribers.delete(subscriber);
  }

  function onSchemeChange() {
    if (state.mode !== "system") return;
    render();
    notify();
  }

  if (darkQuery) {
    if (typeof darkQuery.addEventListener === "function") darkQuery.addEventListener("change", onSchemeChange);
    else if (typeof darkQuery.addListener === "function") darkQuery.addListener(onSchemeChange);
  }

  function dispose() {
    subscribers.clear();
    if (!darkQuery) return;
    if (typeof darkQuery.removeEventListener === "function") darkQuery.removeEventListener("change", onSchemeChange);
    else if (typeof darkQuery.removeListener === "function") darkQuery.removeListener(onSchemeChange);
  }

  render();

  return Object.freeze({
    applyTheme,
    dispose,
    get: snapshot,
    reset,
    setCustomCss,
    subscribe,
    families: FAMILIES,
    modes: MODES,
  });
}

export {
  DEFAULT_FAMILY,
  DEFAULT_MODE,
  FAMILIES,
  MODES,
  STORAGE_KEYS,
  THEME_COLORS,
  createThemeController,
  normalizeFamily,
  normalizeMode,
  themeStylesheetPath,
};
