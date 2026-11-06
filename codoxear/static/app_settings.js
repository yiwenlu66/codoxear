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
  const focusModalSurface = options.focusModalSurface || CodoxearModal.focusModalSurface;
  const isModalTargetOpen = options.isModalTargetOpen || CodoxearModal.isModalTargetOpen;
  const restoreModalFocus = options.restoreModalFocus || CodoxearModal.restoreModalFocus;
  // Federation section deps are optional: without them (tests, stripped-down
  // embeds) the section hides itself instead of breaking the dialog.
  const fetchFederation = typeof options.fetchFederation === "function" ? options.fetchFederation : null;
  const saveFederation = typeof options.saveFederation === "function" ? options.saveFederation : null;
  const rotateFederationToken = typeof options.rotateFederationToken === "function" ? options.rotateFederationToken : null;
  const fetchFederationNodes = typeof options.fetchNodes === "function" ? options.fetchNodes : null;
  const copyText = typeof options.copyText === "function" ? options.copyText : null;
  const setToast = typeof options.setToast === "function" ? options.setToast : () => {};

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
    rows: "4",
    spellcheck: "false",
    autocorrect: "off",
    autocapitalize: "off",
    autocomplete: "off",
    placeholder: ":root { --accent: #c96442; }",
    "aria-describedby": "settingsCustomCssHint",
  });

  // Federation: hub card (listener + token + attached nodes) and node card
  // (upstream hub address, token, display name, live status). One Save posts
  // both cards; the server applies the change without a restart.
  const textInputAttrs = { type: "text", autocomplete: "off", spellcheck: "false", autocorrect: "off", autocapitalize: "off" };
  const hubEnabledToggle = el("input", { id: "federationHubEnabled", type: "checkbox" });
  const hubListenInput = el("input", { id: "federationHubListen", placeholder: "0.0.0.0:8744", ...textInputAttrs });
  const hubTokenInput = el("input", { id: "federationHubToken", class: "federationTokenInput", readonly: "readonly", ...textInputAttrs });
  const hubTokenCopyBtn = el("button", { id: "federationHubTokenCopyBtn", class: "text-btn", type: "button", text: "Copy" });
  const hubTokenRotateBtn = el("button", { id: "federationHubTokenRotateBtn", class: "text-btn", type: "button", text: "Regenerate" });
  const hubNodesList = el("div", { class: "federationNodesList", id: "federationHubNodesList" });
  const hubStatusLine = el("span", { class: "fieldHint", id: "federationHubStatus", text: "" });
  const nodeEnabledToggle = el("input", { id: "federationNodeEnabled", type: "checkbox" });
  const nodeHubInput = el("input", { id: "federationNodeHub", placeholder: "hub-host:8744", ...textInputAttrs });
  const nodeTokenInput = el("input", { id: "federationNodeToken", ...textInputAttrs });
  const nodeNameInput = el("input", { id: "federationNodeName", placeholder: "this-machine", ...textInputAttrs });
  const nodeConnectBtn = el("button", { id: "federationNodeConnectBtn", class: "text-btn", type: "button", text: "Connect" });
  const federationSaveBtn = el("button", { id: "federationSaveBtn", class: "primary", type: "button", text: "Save federation" });
  const nodeStatusLine = el("span", { class: "fieldHint", id: "federationNodeStatus", text: "" });
  const federationSection = el("section", { class: "settingsSection", id: "federationSettingsSection", "aria-labelledby": "federationSettingsHeading" }, [
    el("h3", { class: "settingsSectionTitle", id: "federationSettingsHeading", text: "Federation" }),
    el("div", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "Hub" }),
      el("label", { class: "checkField" }, [hubEnabledToggle, el("span", { text: "Accept nodes on this machine" })]),
    ]),
    el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Listen address" }), hubListenInput]),
    el("div", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "Hub token" }),
      el("div", { class: "federationTokenRow" }, [hubTokenInput, hubTokenCopyBtn, hubTokenRotateBtn]),
      el("span", { class: "fieldHint", text: "Copy this token into each node's settings." }),
    ]),
    el("div", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "Attached nodes" }),
      hubNodesList,
      hubStatusLine,
    ]),
    el("div", { class: "field" }, [
      el("span", { class: "fieldLabel", text: "Node" }),
      el("label", { class: "checkField" }, [nodeEnabledToggle, el("span", { text: "Publish this machine's sessions to a hub" })]),
    ]),
    el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Hub address" }), nodeHubInput]),
    el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Hub token" }), nodeTokenInput]),
    el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Display name" }), nodeNameInput]),
    el("div", { class: "field" }, [
      el("div", { class: "federationActions" }, [nodeConnectBtn, federationSaveBtn]),
      nodeStatusLine,
    ]),
  ]);
  if (!fetchFederation || !saveFederation) federationSection.style.display = "none";

  const viewer = el("dialog", { class: "formViewer formDialog", id: "settingsViewer", "aria-label": "Settings", tabindex: "-1" }, [
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
      ]),
      federationSection,
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
    // controller state that did not originate here (initial render) replaces
    // its value.
    if (customCssTimer === null && customCssInput.value !== theme.customCss) customCssInput.value = theme.customCss;
  }

  const unsubscribe = themeController.subscribe(render);

  let federationBusy = false;

  function renderFederation(config, nodes) {
    const hub = config && config.hub ? config.hub : {};
    const node = config && config.node ? config.node : {};
    const status = config && config.status ? config.status : {};
    hubEnabledToggle.checked = !!hub.enabled;
    hubListenInput.value = String(hub.listen || "");
    hubTokenInput.value = String(hub.token || "");
    nodeEnabledToggle.checked = !!node.enabled;
    nodeHubInput.value = String(node.hub || "");
    nodeTokenInput.value = String(node.token || "");
    nodeNameInput.value = String(node.name || "");
    const listening = status.hub && status.hub.listening;
    hubStatusLine.textContent = hub.enabled ? (listening ? `listening on ${listening}` : "not listening") : "";
    const nodeStatus = status.node || {};
    nodeStatusLine.textContent = node.enabled
      ? `node ${nodeStatus.state || "off"}${nodeStatus.error ? ` — ${nodeStatus.error}` : ""}`
      : "";
    nodeConnectBtn.textContent = nodeEnabledToggle.checked ? "Disconnect" : "Connect";
    hubNodesList.innerHTML = "";
    if (!nodes.length) {
      hubNodesList.appendChild(el("span", { class: "fieldHint", text: "No nodes attached." }));
    }
    for (const entry of nodes) {
      const count = Number(entry.session_count || 0);
      hubNodesList.appendChild(el("div", { class: "federationNodeRow" }, [
        el("span", { class: "federationNodeName", text: String(entry.name || "") }),
        el("span", { class: "fieldHint", text: `${count} session${count === 1 ? "" : "s"}` }),
      ]));
    }
  }

  async function refreshFederation() {
    if (!fetchFederation) return;
    try {
      const [config, nodesRes] = await Promise.all([
        fetchFederation(),
        fetchFederationNodes ? fetchFederationNodes() : Promise.resolve({ nodes: [] }),
      ]);
      renderFederation(config, Array.isArray(nodesRes && nodesRes.nodes) ? nodesRes.nodes : []);
    } catch (error) {
      setToast(`federation load error: ${error && error.message ? error.message : "unknown error"}`);
    }
  }

  function federationErrorText(error) {
    if (error && error.obj && typeof error.obj.error === "string") return error.obj.error;
    return error && error.message ? error.message : "unknown error";
  }

  async function saveFederationSettings() {
    if (!saveFederation || federationBusy) return;
    federationBusy = true;
    federationSaveBtn.disabled = true;
    try {
      const res = await saveFederation({
        hub: {
          enabled: hubEnabledToggle.checked,
          listen: String(hubListenInput.value || "").trim(),
          token: String(hubTokenInput.value || "").trim(),
        },
        node: {
          enabled: nodeEnabledToggle.checked,
          hub: String(nodeHubInput.value || "").trim(),
          token: String(nodeTokenInput.value || "").trim(),
          name: String(nodeNameInput.value || "").trim(),
        },
      });
      setToast(res && res.ok ? "federation saved" : `federation error: ${res && res.error ? res.error : "unknown error"}`);
    } catch (error) {
      setToast(`federation error: ${federationErrorText(error)}`);
    } finally {
      federationBusy = false;
      federationSaveBtn.disabled = false;
    }
    void refreshFederation();
  }

  async function regenerateHubToken() {
    if (!rotateFederationToken || federationBusy) return;
    federationBusy = true;
    hubTokenRotateBtn.disabled = true;
    try {
      const res = await rotateFederationToken();
      if (res && res.token) {
        hubTokenInput.value = String(res.token);
        setToast("new hub token generated");
      }
    } catch (error) {
      setToast(`token rotation error: ${federationErrorText(error)}`);
    } finally {
      federationBusy = false;
      hubTokenRotateBtn.disabled = false;
    }
    void refreshFederation();
  }

  function copyHubToken() {
    const value = String(hubTokenInput.value || "");
    if (!value || !copyText) return;
    copyText(value);
    setToast("hub token copied");
  }

  addEvent(hubTokenCopyBtn, "click", (event) => { event.preventDefault(); copyHubToken(); });
  addEvent(hubTokenRotateBtn, "click", (event) => { event.preventDefault(); void regenerateHubToken(); });
  addEvent(federationSaveBtn, "click", (event) => { event.preventDefault(); void saveFederationSettings(); });
  addEvent(nodeConnectBtn, "click", (event) => {
    event.preventDefault();
    nodeEnabledToggle.checked = !nodeEnabledToggle.checked;
    void saveFederationSettings();
  });
  addEvent(nodeEnabledToggle, "change", () => {
    nodeConnectBtn.textContent = nodeEnabledToggle.checked ? "Disconnect" : "Connect";
  });

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
    focusModalSurface(viewer);
    activateVoiceSection();
    void refreshFederation();
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

  function dispose() {
    if (customCssTimer !== null) clearTimeoutFn(customCssTimer);
    customCssTimer = null;
    unsubscribe();
  }

  return Object.freeze({ dispose, hide, isOpen: () => isModalTargetOpen(viewer), show, viewer });
}

export { createSettingsDialogController };
