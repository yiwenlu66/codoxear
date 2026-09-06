import * as CodoxearModal from "./app_modal.js";

// Settings dialog. Owns its DOM, its rendering, and its subscription to the
// theme controller end-to-end (widget rule). Appearance choices apply live
// through the theme controller, which remains the only writer of the theme
// surface; this dialog only renders that controller's state and forwards
// user intent to it.
//
// The "Voice & notifications" section is built and rendered by app_voice.js;
// this dialog mounts that section below Appearance and reports its own
// visibility to the voice controller (activate on show, deactivate on hide)
// so the voice form is seeded while visible and its draft is dropped on close.

const FAMILY_LABELS = Object.freeze({ paper: "Paper", clay: "Clay", slate: "Slate" });
const MODE_LABELS = Object.freeze({ system: "System", light: "Light", dark: "Dark" });
const CUSTOM_CSS_DEBOUNCE_MS = 250;

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`settings dependency missing: ${name}`);
  return value;
}

function requireNode(value, name) {
  if (!value || typeof value !== "object" || !value.style) throw new TypeError(`settings dependency missing: ${name}`);
  return value;
}

function createSettingsDialogController(options = {}) {
  if (!options || typeof options !== "object") throw new TypeError("settings dependency missing: options");
  const root = options.root;
  if (!root || typeof root.appendChild !== "function") throw new TypeError("settings dependency missing: root");
  const el = requireFunction(options.el, "el");
  const iconSvg = requireFunction(options.iconSvg, "iconSvg");
  const themeController = options.themeController;
  if (!themeController || typeof themeController.subscribe !== "function") throw new TypeError("settings dependency missing: themeController");
  const openButton = requireNode(options.openButton, "openButton");
  const voiceSection = requireNode(options.voiceSection, "voiceSection");
  const activateVoiceSection = requireFunction(options.activateVoiceSection, "activateVoiceSection");
  const deactivateVoiceSection = requireFunction(options.deactivateVoiceSection, "deactivateVoiceSection");
  const documentTarget = options.documentTarget;
  const ElementCtor = options.ElementCtor;
  if (!documentTarget || typeof documentTarget !== "object") throw new TypeError("settings dependency missing: documentTarget");
  if (typeof ElementCtor !== "function") throw new TypeError("settings dependency missing: ElementCtor");
  const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
  const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
  const addEvent = requireFunction(options.addEvent, "addEvent");
  const setTimeoutFn = requireFunction(options.setTimeout, "setTimeout");
  const clearTimeoutFn = requireFunction(options.clearTimeout, "clearTimeout");
  const focusModalCloseButton = options.focusModalCloseButton || CodoxearModal.focusModalCloseButton;
  const isModalTargetOpen = options.isModalTargetOpen || CodoxearModal.isModalTargetOpen;
  const restoreModalFocus = options.restoreModalFocus || CodoxearModal.restoreModalFocus;

  const backdrop = el("div", { class: "modalBackdrop", id: "settingsBackdrop" });
  const closeButton = el("button", {
    id: "settingsCloseBtn",
    class: "icon-btn",
    title: "Close",
    "aria-label": "Close",
    type: "button",
    html: iconSvg("x"),
  });

  // Family swatches: a pure-CSS miniature (sidebar strip, two message
  // bubbles, a primary button) rendered from the swatch palette in app.css.
  const familyButtons = new Map();
  const swatches = el("div", { class: "themeSwatches", role: "radiogroup", "aria-label": "Theme" });
  for (const family of themeController.families) {
    const preview = el("span", { class: "themeSwatchPreview", "data-swatch-family": family, "aria-hidden": "true" }, [
      el("span", { class: "themeSwatchSidebar" }, [el("span", { class: "themeSwatchRow" }), el("span", { class: "themeSwatchRow" })]),
      el("span", { class: "themeSwatchChat" }, [
        el("span", { class: "themeSwatchBubble user" }),
        el("span", { class: "themeSwatchBubble assistant" }),
        el("span", { class: "themeSwatchPrimary" }),
      ]),
    ]);
    const button = el("button", {
      type: "button",
      class: "themeSwatch",
      role: "radio",
      "aria-checked": "false",
      "data-theme-family": family,
    }, [preview, el("span", { class: "themeSwatchName", text: FAMILY_LABELS[family] || family })]);
    familyButtons.set(family, button);
    swatches.appendChild(button);
  }

  const modeButtons = new Map();
  const modeChips = el("div", { class: "choiceChips", id: "settingsModeChips", role: "radiogroup", "aria-label": "Mode" });
  for (const mode of themeController.modes) {
    const button = el("button", {
      type: "button",
      class: "choiceChip",
      role: "radio",
      "aria-checked": "false",
      "data-theme-mode": mode,
      text: MODE_LABELS[mode] || mode,
    });
    modeButtons.set(mode, button);
    modeChips.appendChild(button);
  }
  const modeHint = el("span", { class: "fieldHint", id: "settingsModeHint", text: "" });

  const customCssInput = el("textarea", {
    id: "settingsCustomCss",
    class: "customCssInput",
    rows: "6",
    spellcheck: "false",
    autocomplete: "off",
    autocapitalize: "off",
    placeholder: ":root { --accent: #c96442; }",
    "aria-describedby": "settingsCustomCssHint",
  });
  const resetButton = el("button", { id: "settingsResetAppearanceBtn", class: "text-btn", type: "button", text: "Reset appearance" });

  const viewer = el("dialog", { class: "formViewer formDialog", id: "settingsViewer", "aria-label": "Settings" }, [
    el("div", { class: "queueHeader" }, [
      el("div", { class: "title", text: "Settings" }),
      el("div", { class: "actions" }, [closeButton]),
    ]),
    el("div", { class: "formBody" }, [
      el("section", { class: "settingsSection", id: "appearanceSettingsSection", "aria-labelledby": "appearanceSettingsHeading" }, [
        el("h3", { class: "settingsSectionTitle", id: "appearanceSettingsHeading", text: "Appearance" }),
        el("div", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Theme" }),
          swatches,
        ]),
        el("div", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Mode" }),
          modeChips,
          modeHint,
        ]),
        el("label", { class: "field" }, [
          el("span", { class: "fieldLabel", text: "Custom CSS" }),
          customCssInput,
          el("span", { class: "fieldHint", id: "settingsCustomCssHint", text: "Applied live on this device after the theme stylesheet. Stored in this browser only." }),
        ]),
        el("div", { class: "field settingsResetRow" }, [resetButton]),
      ]),
      voiceSection,
    ]),
  ]);
  root.appendChild(backdrop);
  root.appendChild(viewer);

  let returnFocusElement = null;
  let customCssTimer = null;

  function render(theme) {
    for (const [family, button] of familyButtons) {
      const active = family === theme.family;
      button.classList.toggle("active", active);
      button.setAttribute("aria-checked", active ? "true" : "false");
    }
    swatches.setAttribute("data-swatch-mode", theme.resolvedMode);
    for (const [mode, button] of modeButtons) {
      const active = mode === theme.mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-checked", active ? "true" : "false");
    }
    modeHint.textContent = theme.mode === "system"
      ? `Follows the system setting (currently ${theme.resolvedMode}).`
      : `Always ${theme.mode}.`;
    // The textarea is the user's live draft while a debounce is pending; only
    // controller state that did not originate here (reset, initial render)
    // replaces its value.
    if (customCssTimer === null && customCssInput.value !== theme.customCss) customCssInput.value = theme.customCss;
  }

  const unsubscribe = themeController.subscribe(render);

  function flushCustomCss() {
    if (customCssTimer !== null) {
      clearTimeoutFn(customCssTimer);
      customCssTimer = null;
    }
    if (customCssInput.value !== themeController.get().customCss) themeController.setCustomCss(customCssInput.value);
  }

  function scheduleCustomCss() {
    if (customCssTimer !== null) clearTimeoutFn(customCssTimer);
    customCssTimer = setTimeoutFn(() => {
      customCssTimer = null;
      themeController.setCustomCss(customCssInput.value);
    }, CUSTOM_CSS_DEBOUNCE_MS);
  }

  function show({ opener = null } = {}) {
    returnFocusElement = opener instanceof ElementCtor
      ? opener
      : documentTarget.activeElement instanceof ElementCtor
        ? documentTarget.activeElement
        : null;
    prepareModalOpen();
    backdrop.style.display = "block";
    viewer.style.display = "flex";
    if (typeof viewer.showModal === "function" && !viewer.open) viewer.showModal();
    afterModalVisibilityChanged();
    focusModalCloseButton(viewer, closeButton);
    activateVoiceSection();
  }

  function hide() {
    const wasOpen = isModalTargetOpen(viewer);
    const focusTarget = returnFocusElement;
    returnFocusElement = null;
    flushCustomCss();
    deactivateVoiceSection();
    backdrop.style.display = "none";
    viewer.style.display = "none";
    if (typeof viewer.close === "function" && viewer.open) viewer.close();
    afterModalVisibilityChanged();
    if (wasOpen) restoreModalFocus(focusTarget, () => isModalTargetOpen(viewer));
  }

  addEvent(openButton, "click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    show({ opener: event.currentTarget });
  });
  addEvent(closeButton, "click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    hide();
  });
  addEvent(backdrop, "click", hide);
  addEvent(viewer, "click", (event) => {
    if (event.target === viewer) hide();
  });
  // Escape never closes a dialog: swallow the native <dialog> cancel.
  addEvent(viewer, "cancel", (event) => event.preventDefault());
  for (const [family, button] of familyButtons) {
    addEvent(button, "click", () => themeController.applyTheme({ family }));
  }
  for (const [mode, button] of modeButtons) {
    addEvent(button, "click", () => themeController.applyTheme({ mode }));
  }
  addEvent(customCssInput, "input", scheduleCustomCss);
  addEvent(customCssInput, "change", flushCustomCss);
  addEvent(resetButton, "click", () => {
    if (customCssTimer !== null) clearTimeoutFn(customCssTimer);
    customCssTimer = null;
    themeController.reset();
  });

  function dispose() {
    if (customCssTimer !== null) clearTimeoutFn(customCssTimer);
    customCssTimer = null;
    unsubscribe();
  }

  return Object.freeze({ dispose, hide, isOpen: () => isModalTargetOpen(viewer), show, viewer });
}

export { createSettingsDialogController };
