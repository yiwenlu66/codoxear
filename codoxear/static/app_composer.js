import * as CodoxearPathCompletion from "./app_path_completion.js";
import * as CodoxearFileHelpers from "./app_file_helpers.js";
import * as CodoxearPickerGeometry from "./app_picker_geometry.js";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`composer controller dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || typeof value.addEventListener !== "function")
      throw new TypeError(`composer controller dependency missing: ${name}`);
    return value;
  }

  function createComposerController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("composer controller dependency missing: options");
    const form = requireNode(options.form, "form");
    const textarea = requireNode(options.textarea, "textarea");
    const msgPh = requireNode(options.msgPh, "msgPh");
    const sendBtn = requireNode(options.sendBtn, "sendBtn");
    const sendChoice = requireNode(options.sendChoice, "sendChoice");
    const sendChoiceBackdrop = requireNode(options.sendChoiceBackdrop, "sendChoiceBackdrop");
    const sendChoiceNowBtn = requireNode(options.sendChoiceNowBtn, "sendChoiceNowBtn");
    const sendChoiceLaterBtn = requireNode(options.sendChoiceLaterBtn, "sendChoiceLaterBtn");
    const sendChoiceCancelBtn = requireNode(options.sendChoiceCancelBtn, "sendChoiceCancelBtn");

    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function" || typeof sessionCatalog.subscribe !== "function") throw new TypeError("composer dependency missing: sessionCatalog");
    const getSessionInfo = (sessionId) => sessionCatalog.get("sessionIndex").get(sessionId) || null;
    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") throw new TypeError("composer dependency missing: sessionState");
    const getStagedAttachments = requireFunction(options.getStagedAttachments, "getStagedAttachments");
    const isModalOpen = typeof options.isModalOpen === "function" ? options.isModalOpen : () => false;
    const api = requireFunction(options.api, "api");
    const setToast = requireFunction(options.setToast, "setToast");
    const setPollFastUntilMs = requireFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const kickPoll = requireFunction(options.kickPoll, "kickPoll");
    const sendText = requireFunction(options.sendText, "sendText");
    const enqueueComposerText = requireFunction(options.enqueueComposerText, "enqueueComposerText");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const restoreModalFocus = requireFunction(options.restoreModalFocus, "restoreModalFocus");
    const storageGetItem = requireFunction(options.storageGetItem, "storageGetItem");
    const storageSetItem = requireFunction(options.storageSetItem, "storageSetItem");
    const storageRemoveItem = requireFunction(options.storageRemoveItem, "storageRemoveItem");
    const getNewSessionDefaults = () => sessionCatalog.get("newSessionDefaults");
    const modelPicker = options.modelPicker && typeof options.modelPicker === "object" ? options.modelPicker : null;
    // Height-mode controls are optional like modelPicker: without both nodes the
    // composer stays permanently in normal auto-grow mode.
    const composerEl = options.composer && typeof options.composer === "object" && options.composer.dataset ? options.composer : null;
    const expandBtn = options.expandBtn && typeof options.expandBtn.addEventListener === "function" ? options.expandBtn : null;
    // Browser-dependent measurement seams (real-browser defaults).
    const measureTextareaHeight = typeof options.measureTextareaHeight === "function" ? options.measureTextareaHeight : () => textarea.getBoundingClientRect().height;
    const getAppHeight = typeof options.getAppHeight === "function" ? options.getAppHeight : () => {
      const fromVar = parseFloat(getComputedStyleFn(document.documentElement).getPropertyValue("--appH"));
      return fromVar > 0 ? fromVar : window.innerHeight;
    };
    const onAutoGrow = typeof options.onAutoGrow === "function" ? options.onAutoGrow : () => {};
    const requestFrame = typeof options.requestFrame === "function" ? options.requestFrame : (callback) => requestAnimationFrame(callback);
    const getComputedStyleFn = typeof options.getComputedStyle === "function" ? options.getComputedStyle : (node) => getComputedStyle(node);
    const activeElement = typeof options.activeElement === "function" ? options.activeElement : () => document.activeElement;
    const isHTMLElement = typeof options.isHTMLElement === "function" ? options.isHTMLElement : (value) => typeof HTMLElement === "function" && value instanceof HTMLElement;
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const consoleError = typeof options.consoleError === "function" ? options.consoleError : () => {};
    const windowTarget = options.windowTarget && typeof options.windowTarget.addEventListener === "function" ? options.windowTarget : null;
    // Draft-sync notification seam: invoked from this module's own input
    // listener so no second listener attaches to the composer's textarea.
    const onDraftEdited = typeof options.onDraftEdited === "function" ? options.onDraftEdited : () => {};
    const historyController = options.historyController && typeof options.historyController.handleKeydown === "function" && typeof options.historyController.reset === "function"
      ? options.historyController
      : null;

    const cleanups = [];
    const listen = (target, type, handler, eventOptions) => {
      target.addEventListener(type, handler, eventOptions);
      cleanups.push(() => target.removeEventListener(type, handler, eventOptions));
    };
    const sessionDraftKey = (sessionId) => `codexweb.draft.${sessionId}`;
    const heightModeKey = (sessionId) => `codexweb.composerHeight.${sessionId}`;
    const DRAG_THRESHOLD_PX = 8;
    const NORMAL_SNAP_FACTOR = 1.3;
    const FULL_SNAP_FACTOR = 0.9;
    const DEFAULT_NORMAL_CAP_PX = 180;
    // Height mode: "normal" auto-grows to the CSS max-height cap, "tall" is a
    // fixed height, "full" fills the .main column. applyHeightMode is the sole
    // writer of composer[data-height] and of the fixed-height custom property.
    let heightMode = "normal";
    let tallHeightPx = null;
    let normalOverflow = false;
    let heightDragActive = false;
    let sendChoicePending = null;
    let sendChoiceReturnFocusEl = null;
    let modelPickerOpen = false;
    let modelPickerKind = null;
    let modelPickerOptions = [];
    let modelPickerFocus = -1;
    const pathCompletion = CodoxearPathCompletion.createPathCompletionRuntime({ api, onChange: renderPathPicker });

    const PI_THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
    const CC_EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max", "auto"];
    // Each entry describes controls that target the running shared session.
    // Pi and Claude use native text commands. Codex advertises these entries
    // only when its broker owns a reachable app-server settings transport.
    const BACKEND_COMMAND_SPECS = Object.freeze({
      pi: Object.freeze({
        model: Object.freeze({ command: "/model", aliases: Object.freeze(["model"]) }),
        effort: Object.freeze({ command: "/effort", aliases: Object.freeze(["effort", "thinking"]), requiresPiThinkingCapability: true }),
      }),
      cc: Object.freeze({
        model: Object.freeze({ command: "/model", aliases: Object.freeze(["model"]) }),
        effort: Object.freeze({ command: "/effort", aliases: Object.freeze(["effort"]) }),
      }),
      codex: Object.freeze({
        model: Object.freeze({ command: "/model", aliases: Object.freeze(["model"]), requiresAdvertisedCommand: true, typedSettings: true }),
        effort: Object.freeze({ command: "/effort", aliases: Object.freeze(["effort"]), requiresAdvertisedCommand: true, typedSettings: true }),
      }),
    });

    function selectedSession() {
      const sessionId = sessionState.get("selected");
      return sessionId ? getSessionInfo(sessionId) || null : null;
    }

    function sessionBackend(session) {
      return String(session && session.agent_backend || "").trim().toLowerCase();
    }

    function commandSpec(session, kind) {
      const backend = sessionBackend(session);
      const spec = BACKEND_COMMAND_SPECS[backend] && BACKEND_COMMAND_SPECS[backend][kind];
      if (!spec) return null;
      if (spec.requiresPiThinkingCapability && session.pi_thinking_command !== true) return null;
      if (spec.requiresAdvertisedCommand) {
        const advertised = Array.isArray(session.slash_commands)
          && session.slash_commands.some((entry) => String(entry && entry.name || "").replace(/^\//, "").toLowerCase() === kind);
        if (!advertised) return null;
      }
      return spec;
    }

    function codexLaunchDefaults() {
      const defaults = getNewSessionDefaults();
      return defaults && defaults.backends && defaults.backends.codex && typeof defaults.backends.codex === "object" ? defaults.backends.codex : {};
    }

    function piLaunchDefaults() {
      const defaults = getNewSessionDefaults();
      return defaults && defaults.backends && defaults.backends.pi && typeof defaults.backends.pi === "object" ? defaults.backends.pi : {};
    }

    function ccLaunchDefaults() {
      const defaults = getNewSessionDefaults();
      return defaults && defaults.backends && defaults.backends.cc && typeof defaults.backends.cc === "object" ? defaults.backends.cc : {};
    }

    function piModelIds() {
      const pi = piLaunchDefaults();
      const providerModels = pi.provider_models && typeof pi.provider_models === "object" ? pi.provider_models : null;
      const out = [];
      const seen = new Set();
      if (providerModels) {
        for (const [provider, models] of Object.entries(providerModels)) {
          if (!Array.isArray(models)) continue;
          for (const model of models) {
            const id = `${String(provider).trim()}/${String(model || "").trim()}`;
            if (!id.includes("/") || id.endsWith("/") || seen.has(id)) continue;
            seen.add(id);
            out.push(id);
          }
        }
      }
      if (!out.length && Array.isArray(pi.models)) {
        for (const model of pi.models) {
          const id = String(model || "").trim();
          if (id && !seen.has(id)) { seen.add(id); out.push(id); }
        }
      }
      return out;
    }

    function codexProviderChoice(session) {
      const explicit = String(session && session.provider_choice || "").trim();
      if (explicit) return explicit;
      const provider = String(session && session.model_provider || "").trim();
      if (provider !== "openai") return provider;
      return String(session && session.preferred_auth_method || "").trim() === "chatgpt" ? "chatgpt" : "openai-api";
    }

    function codexProviderGroups(session) {
      const codex = codexLaunchDefaults();
      const providerModels = codex.provider_models && typeof codex.provider_models === "object" ? codex.provider_models : {};
      const configuredProviders = Array.isArray(codex.provider_choices)
        ? codex.provider_choices
        : Array.isArray(codex.model_providers) ? codex.model_providers : [];
      const activeProvider = codexProviderChoice(session);
      const providers = configuredProviders.length ? configuredProviders : activeProvider ? [activeProvider] : [];
      const fallbackModels = Array.isArray(codex.models) ? codex.models : [];
      return providers.map((rawProvider) => {
        const provider = String(rawProvider || "").trim();
        if (!provider) return null;
        const configured = providerModels[provider];
        const rawModels = Array.isArray(configured)
          ? configured
          : provider === activeProvider ? fallbackModels : [];
        const models = [];
        for (const rawModel of rawModels) {
          const model = String(rawModel || "").trim();
          if (model && !models.includes(model)) models.push(model);
        }
        return { provider, models, active: provider === activeProvider };
      }).filter(Boolean);
    }

    function codexModelIds(session) {
      const current = String(session && session.model || "").trim();
      const group = codexProviderGroups(session).find((candidate) => candidate.active);
      const models = group ? group.models.slice() : [];
      if (current && !models.includes(current)) models.unshift(current);
      return models.map((model) => ({ model, provider: group ? group.provider : codexProviderChoice(session), current: model === current }));
    }

    function ccModelIds() {
      const models = ccLaunchDefaults().models;
      const out = [];
      for (const model of Array.isArray(models) ? models : []) {
        const id = String(model || "").trim();
        if (id && !out.includes(id)) out.push(id);
      }
      return out;
    }

    function piThinkingLevels(session) {
      const pi = piLaunchDefaults();
      const byModel = pi.reasoning_efforts_by_model && typeof pi.reasoning_efforts_by_model === "object" ? pi.reasoning_efforts_by_model : {};
      const provider = String(session.model_provider || "").trim();
      const model = String(session.model || "").trim();
      const scoped = byModel[provider && model ? `${provider}/${model}` : ""] || byModel[model];
      const configured = Array.isArray(scoped) ? scoped : Array.isArray(pi.reasoning_efforts) ? pi.reasoning_efforts : PI_THINKING_LEVELS;
      const seen = new Set();
      return configured
        .map((level) => String(level || "").trim().toLowerCase())
        .filter((level) => PI_THINKING_LEVELS.includes(level) && !seen.has(level) && seen.add(level));
    }

    function codexEffortLevels(session) {
      const codex = codexLaunchDefaults();
      const byModel = codex.reasoning_efforts_by_model && typeof codex.reasoning_efforts_by_model === "object" ? codex.reasoning_efforts_by_model : {};
      const model = String(session.model || "").trim();
      const scoped = model && Array.isArray(byModel[model]) ? byModel[model] : null;
      const configured = scoped || (Array.isArray(codex.reasoning_efforts) ? codex.reasoning_efforts : PI_THINKING_LEVELS.filter((level) => level !== "off"));
      const seen = new Set();
      return configured
        .map((level) => String(level || "").trim().toLowerCase())
        .filter((level) => level !== "off" && PI_THINKING_LEVELS.includes(level) && !seen.has(level) && seen.add(level));
    }

    function ccEffortLevels(session) {
      const cc = ccLaunchDefaults();
      const byModel = cc.reasoning_efforts_by_model && typeof cc.reasoning_efforts_by_model === "object" ? cc.reasoning_efforts_by_model : {};
      const model = String(session.model || "").trim();
      const scoped = model && Array.isArray(byModel[model]) ? byModel[model] : null;
      const configured = scoped || (Array.isArray(cc.reasoning_efforts) ? cc.reasoning_efforts : CC_EFFORT_LEVELS);
      const seen = new Set();
      return configured
        .map((level) => String(level || "").trim().toLowerCase())
        .filter((level) => CC_EFFORT_LEVELS.includes(level) && !seen.has(level) && seen.add(level));
    }

    function commandPickerMatches(kind) {
      const session = selectedSession();
      const spec = commandSpec(session, kind);
      if (!session || !spec) return null;
      const aliases = spec.aliases.map((alias) => alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
      const match = String(textarea.value || "").match(new RegExp(`^/(?:${aliases})(?:\\s+(.*))?$`, "i"));
      if (!match) return null;
      const query = String(match[1] || "").trim().toLowerCase();
      const backend = sessionBackend(session);
      const choices = kind === "model"
        ? backend === "pi" ? piModelIds() : backend === "codex" ? codexModelIds(session) : ccModelIds()
        : backend === "pi" ? piThinkingLevels(session) : backend === "codex" ? codexEffortLevels(session) : ccEffortLevels(session);
      const matches = choices.filter((choice) => {
        const name = typeof choice === "object" && choice !== null ? String(choice.model || "") : String(choice || "");
        return !query || name.toLowerCase().startsWith(query) || name.toLowerCase().includes(query);
      });
      if (kind !== "effort") return matches;
      const current = String(session.reasoning_effort || "").trim().toLowerCase();
      return current && matches.includes(current) ? [current, ...matches.filter((choice) => choice !== current)] : matches;
    }

    function modelPickerMatches() {
      return commandPickerMatches("model");
    }

    function effortPickerMatches() {
      return commandPickerMatches("effort");
    }

    function slashCommandMatches() {
      const session = selectedSession();
      if (!session || String(textarea.value || "")[0] !== "/") return null;
      const raw = String(textarea.value || "").slice(1);
      if (/^(?:model|effort|thinking)(?:\s|$)/i.test(raw)) return null;
      const query = raw.toLowerCase();
      const entries = Array.isArray(session.slash_commands) ? session.slash_commands : [];
      const merged = entries.map((entry) => ({ name: String(entry && entry.name || "").replace(/^\//, ""), description: String(entry && entry.description || "") }));
      for (const kind of ["model", "effort"]) {
        if (commandSpec(session, kind) && !merged.some((entry) => entry.name.toLowerCase() === kind)) merged.push({ name: kind, description: kind === "model" ? "Select model" : "Set reasoning effort", pickerKind: kind });
      }
      return merged.filter((entry) => entry.name && (!query || entry.name.toLowerCase().includes(query) || entry.description.toLowerCase().includes(query)));
    }
    function unsupportedPiThinkingCommand(raw, session) {
      return Boolean(
        sessionBackend(session) === "pi"
        && session.pi_thinking_command !== true
        && /^\/(?:effort|thinking)(?:\s|$)/i.test(String(raw || "")),
      );
    }

    function clearModelPickerGeometry() {
      if (!modelPicker) return;
      modelPicker.style.top = "";
      modelPicker.style.left = "";
      modelPicker.style.maxHeight = "";
      modelPicker.style.maxWidth = "";
      modelPicker.style.visibility = "";
      if (modelPicker.dataset) delete modelPicker.dataset.pickerDir;
    }

    function pickerAnchorIndex() {
      if (modelPickerKind !== "path") return 0;
      const state = pathCompletion.snapshot();
      return Number.isFinite(state.start) ? state.start : 0;
    }

    // This is the sole inline-style writer for the shared completion menu.
    // The geometry helper works in viewport coordinates; the picker remains
    // absolutely positioned inside form, so its result is translated once.
    function positionModelPicker() {
      if (!modelPicker) return;
      if (heightMode === "normal") {
        clearModelPickerGeometry();
        return;
      }
      if (!modelPickerOpen || modelPicker.style.display !== "block" || typeof textarea.getBoundingClientRect !== "function" || typeof modelPicker.getBoundingClientRect !== "function") return;

      const textareaRect = textarea.getBoundingClientRect();
      const wrap = textarea.parentElement && typeof textarea.parentElement.getBoundingClientRect === "function" ? textarea.parentElement : textarea;
      const wrapRect = wrap.getBoundingClientRect();
      const topbar = typeof document !== "undefined" && typeof document.querySelector === "function" ? document.querySelector(".topbar") : null;
      const topbarBottom = topbar && typeof topbar.getBoundingClientRect === "function" ? topbar.getBoundingClientRect().bottom : 0;
      const gutter = Math.max(0, parseFloat(getComputedStyleFn(composerEl).getPropertyValue("--composer-picker-inset")) || 0);
      const viewport = windowTarget && windowTarget.visualViewport
        ? windowTarget.visualViewport
        : typeof window !== "undefined" ? window.visualViewport : null;
      const viewportHeight = viewport && Number.isFinite(viewport.height)
        ? viewport.height
        : typeof window !== "undefined" && Number.isFinite(window.innerHeight) ? window.innerHeight : 0;
      const minTop = topbarBottom + gutter;
      const maxBottom = Math.max(minTop, viewportHeight - gutter);
      const wrapLeft = wrapRect.left + gutter;
      const wrapRight = Math.max(wrapLeft, wrapRect.right - gutter);

      // Clear only dimension constraints before measuring the newly-rendered
      // content. A narrow input wrap then gets a second measurement after its
      // width cap is applied, keeping the picker within that wrap.
      modelPicker.style.top = "";
      modelPicker.style.left = "";
      modelPicker.style.maxHeight = "";
      modelPicker.style.maxWidth = "";
      let menuRect = modelPicker.getBoundingClientRect();
      const wrapWidth = Math.max(0, wrapRight - wrapLeft);
      if (menuRect.width > wrapWidth) {
        modelPicker.style.maxWidth = `${wrapWidth}px`;
        menuRect = modelPicker.getBoundingClientRect();
      }
      const caretRect = CodoxearPickerGeometry.measureTextareaCaret(textarea, pickerAnchorIndex());
      if (!caretRect) return;
      const geometry = CodoxearPickerGeometry.decidePickerGeometry({
        caretTop: caretRect.top,
        caretBottom: caretRect.bottom,
        tokenLeft: caretRect.left,
        menuNaturalHeight: menuRect.height,
        menuWidth: menuRect.width,
        wrapLeft,
        wrapRight,
        minTop,
        maxBottom,
        textareaTop: textareaRect.top,
        textareaBottom: textareaRect.bottom,
        gap: 6,
        minHeight: 96,
      });
      const formRect = form.getBoundingClientRect();
      modelPicker.style.top = `${geometry.top - formRect.top}px`;
      modelPicker.style.left = `${geometry.left - formRect.left}px`;
      modelPicker.style.maxHeight = `${geometry.maxHeight}px`;
      modelPicker.style.visibility = geometry.visible ? "" : "hidden";
      if (modelPicker.dataset) modelPicker.dataset.pickerDir = geometry.dir;
    }

    function hideModelPicker() {
      pathCompletion.close();
      modelPickerOptions = [];
      modelPickerOpen = false;
      modelPickerKind = null;
      modelPickerFocus = -1;
      if (!modelPicker) return;
      modelPicker.style.display = "none";
      modelPicker.innerHTML = "";
      modelPicker.removeAttribute("aria-activedescendant");
      textarea.removeAttribute("role");
      textarea.removeAttribute("aria-autocomplete");
      textarea.removeAttribute("aria-controls");
      textarea.removeAttribute("aria-expanded");
      textarea.removeAttribute("aria-activedescendant");
    }

    function selectSlashCommand(entry) {
      const name = String(entry && entry.name || "").trim();
      if (!name) return;
      const session = selectedSession();
      const pickerKind = entry && entry.pickerKind || (name.toLowerCase() === "model" || name.toLowerCase() === "effort" ? name.toLowerCase() : null);
      if (pickerKind && commandSpec(session, pickerKind)) {
        textarea.value = `${commandSpec(session, pickerKind).command} `;
        modelPickerKind = pickerKind;
        modelPickerFocus = -1;
        syncModelPicker();
        autoGrow();
        textarea.focus();
        return;
      }
      textarea.value = `/${name} `;
      hideModelPicker();
      autoGrow();
      textarea.focus();
      saveSessionDraft(sessionState.get("selected"));
    }

    function renderCommandPicker(entries) {
      if (!modelPicker) return;
      modelPicker.innerHTML = "";
      modelPicker.setAttribute("role", "listbox");
      modelPicker.setAttribute("aria-label", "Available commands");
      entries.forEach((entry, index) => {
        const option = document.createElement("button");
        option.type = "button";
        option.tabIndex = -1;
        option.className = "modelPickerOption commandPickerOption";
        option.setAttribute("role", "option");
        option.id = `command-picker-option-${index}`;
        option.textContent = `/${entry.name}${entry.description ? ` — ${entry.description}` : ""}`;
        option.onpointerdown = (event) => event.preventDefault();
        option.onclick = () => selectSlashCommand(entry);
        modelPicker.appendChild(option);
      });
      modelPickerOptions = entries;
      modelPicker.style.display = entries.length ? "block" : "none";
      modelPickerOpen = entries.length > 0;
      if (modelPickerOpen) {
        textarea.setAttribute("role", "combobox");
        textarea.setAttribute("aria-autocomplete", "list");
        textarea.setAttribute("aria-controls", modelPicker.id || "modelPicker");
        textarea.setAttribute("aria-expanded", "true");
      }
      syncModelPickerSelection();
      positionModelPicker();
    }
    async function applyCodexSetting(kind, choice, session) {
      const sessionId = sessionState.get("selected");
      if (!sessionId || sessionBackend(session) !== "codex") return;
      try {
        await api(`/api/sessions/${sessionId}/settings`, { method: "POST", body: { [kind]: choice } });
        setToast(`${kind === "model" ? "model" : "reasoning effort"} accepted for the next Codex turn`);
        setPollFastUntilMs(now() + 5000);
        kickPoll();
      } catch (error) {
        setToast(`Codex ${kind} change failed: ${error && error.message ? error.message : error}`);
      }
    }

    function selectPath(entry) {
      const span = pathCompletion.snapshot();
      if (!entry || !span.open || span.sessionId !== sessionState.get("selected")) return;
      textarea.value = textarea.value.slice(0, span.start) + entry.completion + textarea.value.slice(span.end);
      const caret = span.start + entry.completion.length;
      textarea.setSelectionRange(caret, caret);
      saveSessionDraft(sessionState.get("selected"));
      onDraftEdited(textarea.value);
      autoGrow();
      modelPickerFocus = -1;
      if (entry.is_dir) syncModelPicker();
      else hideModelPicker();
      textarea.focus();
    }

    function renderPathPicker(state) {
      if (!modelPicker || !state.open || state.sessionId !== sessionState.get("selected")) return;
      modelPickerKind = "path";
      modelPickerOptions = state.entries;
      modelPickerFocus = Math.min(Math.max(modelPickerFocus, 0), state.entries.length - 1);
      modelPicker.innerHTML = "";
      modelPicker.setAttribute("role", "listbox");
      modelPicker.setAttribute("aria-label", "Available paths");
      if (state.dir) {
        const label = document.createElement("div");
        label.className = "modelPickerProvider";
        label.textContent = state.dir;
        modelPicker.appendChild(label);
      }
      const fragment = state.token.slice(state.token.lastIndexOf("/") + 1);
      state.entries.forEach((entry, index) => {
        const option = document.createElement("button");
        option.type = "button";
        option.tabIndex = -1;
        option.className = "modelPickerOption";
        option.setAttribute("role", "option");
        option.className += " pathPickerOption";
        option.id = `path-picker-option-${index}`;
        const name = entry.name + (entry.is_dir ? "/" : "");
        let offset = 0;
        for (const [start, end] of CodoxearFileHelpers.filePickerMatchRangesForQuery(entry.name, fragment)) {
          option.appendChild(document.createTextNode(name.slice(offset, start)));
          const mark = document.createElement("mark");
          mark.className = "fileMenuMatch";
          mark.textContent = name.slice(start, end);
          option.appendChild(mark);
          offset = end;
        }
        option.appendChild(document.createTextNode(name.slice(offset)));
        option.onpointerdown = (event) => event.preventDefault();
        option.onclick = () => selectPath(entry);
        modelPicker.appendChild(option);
      });
      if (!state.entries.length) {
        const empty = document.createElement("div");
        empty.className = "pickerEmpty";
        empty.textContent = state.loading ? "…" : "No matches";
        modelPicker.appendChild(empty);
      }
      modelPicker.style.display = "block";
      modelPickerOpen = true;
      textarea.setAttribute("role", "combobox");
      textarea.setAttribute("aria-autocomplete", "list");
      textarea.setAttribute("aria-controls", modelPicker.id || "modelPicker");
      textarea.setAttribute("aria-expanded", "true");
      syncModelPickerSelection();
      positionModelPicker();
    }

    function selectPickerOption(option) {
      if (modelPickerKind === "path") { selectPath(option); return; }
      if (modelPickerKind === "command") { selectSlashCommand(option); return; }
      const choice = typeof option === "object" && option !== null ? String(option.model || "").trim() : String(option || "").trim();
      const session = selectedSession();
      const kind = modelPickerKind;
      const spec = commandSpec(session, kind);
      if (!choice || !spec) return;
      hideModelPicker();
      clearComposer();
      // Delivery acknowledgement only: the session row changes later from
      // backend-log evidence, never from this picker selection.
      if (spec.typedSettings) {
        void applyCodexSetting(kind, choice, session);
      } else {
        void sendText(`${spec.command} ${choice}`);
      }
    }

    function syncModelPickerSelection({ scroll = false } = {}) {
      if (!modelPicker) return;
      const options = Array.from(modelPicker.children || []).filter((option) => String(option.className || "").split(/\s+/).includes("modelPickerOption"));
      options.forEach((option, index) => {
        const active = index === modelPickerFocus;
        option.classList.toggle("active", active);
        option.setAttribute("aria-selected", active ? "true" : "false");
      });
      const activeOption = modelPickerFocus >= 0 ? options[modelPickerFocus] : null;
      if (activeOption) {
        textarea.setAttribute("aria-activedescendant", activeOption.id);
        if (scroll && typeof activeOption.scrollIntoView === "function") {
          activeOption.scrollIntoView({ block: "nearest" });
        }
      } else {
        textarea.removeAttribute("aria-activedescendant");
      }
    }

    function renderModelPicker() {
      if (!modelPicker) return;
      modelPicker.innerHTML = "";
      modelPicker.setAttribute("role", "listbox");
      const session = selectedSession();
      const backend = sessionBackend(session);
      const isPiThinking = backend === "pi" && modelPickerKind === "effort";
      const kindLabel = modelPickerKind === "model" ? "models" : isPiThinking ? "thinking levels" : "effort levels";
      const backendLabel = backend === "cc" ? "Claude" : backend === "codex" ? "Codex" : "Pi";
      modelPicker.setAttribute("aria-label", `Available ${backendLabel} ${kindLabel}`);
      if (backend === "codex" && modelPickerKind === "model") {
        const groups = codexProviderGroups(session);
        const activeProvider = codexProviderChoice(session);
        for (const group of groups) {
          const label = document.createElement("div");
          label.className = "modelPickerProvider";
          label.textContent = group.active
            ? `${group.provider} — active provider`
            : group.models.length
              ? `${group.provider} — unavailable on current provider`
              : `${group.provider} — no configured models`;
          label.setAttribute("data-provider", group.provider);
          if (group.active) label.setAttribute("data-active", "true");
          if (!group.active && group.models.length) label.title = "Model provider changes require starting a new Codex session.";
          modelPicker.appendChild(label);
        }
        if (!groups.some((group) => group.active)) {
          const label = document.createElement("div");
          label.className = "modelPickerProvider";
          label.textContent = `${activeProvider || "current provider"} — active provider`;
          label.setAttribute("data-active", "true");
          modelPicker.appendChild(label);
        }
      }
      modelPickerOptions.forEach((entry, index) => {
        const id = typeof entry === "object" && entry !== null ? String(entry.model || "") : String(entry || "");
        const current = Boolean(entry && typeof entry === "object" && entry.current);
        const option = document.createElement("button");
        option.type = "button";
        option.tabIndex = -1;
        option.className = "modelPickerOption";
        option.setAttribute("role", "option");
        option.id = `${isPiThinking ? "thinking" : modelPickerKind}-picker-option-${index}`;
        option.textContent = current ? `${id} — current model` : id;
        if (current) {
          option.className += " currentModel";
          option.setAttribute("aria-current", "true");
        }
        option.onpointerdown = (event) => event.preventDefault();
        option.onclick = () => selectPickerOption(entry);
        modelPicker.appendChild(option);
      });
      modelPicker.style.display = modelPickerOptions.length ? "block" : "none";
      modelPickerOpen = modelPickerOptions.length > 0;
      if (modelPickerOpen) {
        textarea.setAttribute("role", "combobox");
        textarea.setAttribute("aria-autocomplete", "list");
        textarea.setAttribute("aria-controls", modelPicker.id || "modelPicker");
        textarea.setAttribute("aria-expanded", "true");
      }
      syncModelPickerSelection();
      positionModelPicker();
    }

    function syncModelPicker() {
      if (modelPicker) {
        const path = pathCompletion.sync(String(textarea.value || ""), textarea.selectionStart || 0, sessionState.get("selected"));
        if (path) { renderPathPicker(path); return; }
        if (modelPickerKind === "path") hideModelPicker();
      }
      const commands = slashCommandMatches();
      if (commands) {
        modelPickerKind = "command";
        modelPickerFocus = Math.min(Math.max(modelPickerFocus, 0), commands.length - 1);
        renderCommandPicker(commands);
        return;
      }
      const models = modelPickerMatches();
      const effortLevels = models ? null : effortPickerMatches();
      const matches = models || effortLevels;
      if (!matches || !matches.length) { hideModelPicker(); return; }
      modelPickerKind = models ? "model" : "effort";
      modelPickerOptions = matches;
      modelPickerFocus = Math.min(Math.max(modelPickerFocus, 0), matches.length - 1);
      renderModelPicker();
    }

    function selectedSessionLaunchFailed() {
      const sessionId = sessionState.get("selected");
      return sessionLaunchFailed(sessionId ? getSessionInfo(sessionId) : null);
    }

    function syncComposerState() {
      const sessionId = sessionState.get("selected");
      const launchFailed = selectedSessionLaunchFailed();
      const blocked = !sessionId || launchFailed;
      const label = !sessionId ? "Select a session to send" : launchFailed ? "Failed launch cannot receive messages" : "Message";
      textarea.disabled = blocked;
      textarea.setAttribute("aria-label", label);
      textarea.title = blocked ? label : "";
      msgPh.textContent = label;
    }

    function syncSendButtonState() {
      const sessionId = sessionState.get("selected");
      const launchFailed = selectedSessionLaunchFailed();
      const label = !sessionId ? "Select a session to send" : launchFailed ? "Failed launch cannot receive messages" : "Send";
      sendBtn.disabled = Boolean(sessionState.get("sending") || !sessionId || launchFailed);
      sendBtn.title = label;
      sendBtn.setAttribute("aria-label", label);
      syncComposerState();
    }

    // The CSS max-height on `.composer textarea` is the single source of truth
    // for the normal-mode cap. Fixed modes restyle max-height, so the cap is
    // only re-read while in normal mode and the last value is reused otherwise;
    // an unparseable value falls back to the CSS default.
    let normalCapPx = DEFAULT_NORMAL_CAP_PX;
    function normalCap() {
      if (heightMode === "normal") {
        const parsed = parseFloat(getComputedStyleFn(textarea).maxHeight);
        normalCapPx = Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_NORMAL_CAP_PX;
      }
      return normalCapPx;
    }

    function syncExpandBtn() {
      if (!expandBtn) return;
      const expanded = heightMode !== "normal";
      expandBtn.style.display = expanded || normalOverflow || heightDragActive ? "" : "none";
      expandBtn.classList.toggle("expanded", expanded);
      const label = heightMode === "normal" ? "Expand composer" : heightMode === "tall" ? "Expand composer fully" : "Restore composer height";
      expandBtn.title = label;
      expandBtn.setAttribute("aria-label", label);
      expandBtn.setAttribute("aria-pressed", expanded ? "true" : "false");
    }

    function autoGrow() {
      const basePx = parseFloat(getComputedStyleFn(textarea).minHeight || "0") || 32;
      const stagedCount = getStagedAttachments().length;
      msgPh.style.display = textarea.value || stagedCount ? "none" : "flex";
      if (heightMode !== "normal") {
        // Fixed modes: CSS owns the height; only the multiline class is derived.
        form.classList.toggle("multiline", textarea.value.includes("\n") || textarea.scrollHeight > basePx + 1);
        syncExpandBtn();
        onAutoGrow();
        return;
      }
      const maxPx = normalCap();
      textarea.style.height = `${basePx}px`;
      let height = textarea.scrollHeight;
      const multiline = textarea.value.includes("\n") || height > basePx + 1;
      form.classList.toggle("multiline", multiline);
      textarea.style.height = multiline ? "auto" : `${basePx}px`;
      height = textarea.scrollHeight;
      textarea.style.height = `${multiline ? Math.min(height, maxPx) : basePx}px`;
      normalOverflow = height > maxPx;
      textarea.style.overflowY = normalOverflow ? "auto" : "hidden";
      syncExpandBtn();
      onAutoGrow();
    }

    function persistHeightMode() {
      const sessionId = sessionState.get("selected");
      if (!sessionId) return;
      if (heightMode === "normal") storageRemoveItem(heightModeKey(sessionId));
      else storageSetItem(heightModeKey(sessionId), JSON.stringify(heightMode === "tall" && tallHeightPx != null ? { mode: "tall", height: tallHeightPx } : { mode: heightMode }));
    }

    function applyHeightMode(mode, { persist = true, height, preview = false } = {}) {
      if (composerEl) {
        heightMode = mode;
        composerEl.dataset.height = mode;
        if (preview) {
          // Drag previews write the live height to the CSS var only; the
          // committed tall preference changes when a gesture ends in tall, so
          // an aborted drag cannot pollute the height a later click reuses.
          if (mode === "tall" && Number.isFinite(height)) composerEl.style.setProperty("--composer-tall-h", `${Math.round(height)}px`);
        } else {
          if (mode === "tall" && Number.isFinite(height)) tallHeightPx = Math.round(height);
          if (tallHeightPx != null) composerEl.style.setProperty("--composer-tall-h", `${tallHeightPx}px`);
          else composerEl.style.removeProperty("--composer-tall-h");
          if (persist) persistHeightMode();
        }
        if (mode !== "normal") {
          textarea.style.height = "";
          textarea.style.overflowY = "";
        }
      }
      autoGrow();
      positionModelPicker();
    }

    function loadHeightMode(sessionId) {
      let stored = null;
      if (sessionId) {
        try { stored = JSON.parse(storageGetItem(heightModeKey(sessionId))); } catch (_) { stored = null; }
      }
      const mode = stored && (stored.mode === "tall" || stored.mode === "full") ? stored.mode : "normal";
      tallHeightPx = mode === "tall" && Number.isFinite(stored.height) && stored.height > 0 ? Math.round(stored.height) : null;
      applyHeightMode(mode, { persist: false });
    }

    // A successful send/enqueue consumed the draft: the next message starts compact.
    function resetHeightMode() {
      applyHeightMode("normal");
    }

    function previewHeightDrag(targetPx) {
      const appHeight = getAppHeight();
      const basePx = parseFloat(getComputedStyleFn(textarea).minHeight || "0") || 32;
      const target = Math.min(Math.max(targetPx, basePx), appHeight);
      const mode = target < NORMAL_SNAP_FACTOR * normalCap() ? "normal" : target > FULL_SNAP_FACTOR * appHeight ? "full" : "tall";
      applyHeightMode(mode, { persist: false, height: target, preview: true });
      return target;
    }

    function bindExpandButton() {
      if (!expandBtn || !composerEl) return;
      let gesture = null;
      let suppressClick = false;
      listen(expandBtn, "pointerdown", (event) => {
        if (event.pointerType === "mouse" && event.button !== 0) return;
        suppressClick = false;
        gesture = { startY: event.clientY, startHeight: measureTextareaHeight(), dragging: false };
        try { expandBtn.setPointerCapture(event.pointerId); } catch (_) {}
      });
      listen(expandBtn, "pointermove", (event) => {
        if (!gesture) return;
        const dy = event.clientY - gesture.startY;
        if (!gesture.dragging) {
          if (Math.abs(dy) <= DRAG_THRESHOLD_PX) return;
          gesture.dragging = true;
          // Keep the handle rendered even if the preview snaps to normal.
          heightDragActive = true;
        }
        if (typeof event.preventDefault === "function") event.preventDefault();
        gesture.lastTargetPx = previewHeightDrag(gesture.startHeight - dy);
      });
      // A browser fires `click` after a pointerup but never after a
      // pointercancel, so only pointerup arms the click suppression.
      const finishGesture = (event, { suppress }) => {
        if (!gesture) return;
        try { if (event && event.pointerId != null) expandBtn.releasePointerCapture(event.pointerId); } catch (_) {}
        const wasDrag = gesture.dragging;
        const lastTargetPx = gesture.lastTargetPx;
        gesture = null;
        if (!wasDrag) return;
        heightDragActive = false;
        suppressClick = suppress;
        // A gesture ending in tall commits the previewed height as the
        // session's tall preference; any other landing mode keeps the
        // previously committed preference untouched.
        if (heightMode === "tall" && Number.isFinite(lastTargetPx)) applyHeightMode("tall", { height: lastTargetPx });
        else applyHeightMode(heightMode);
      };
      listen(expandBtn, "pointerup", (event) => finishGesture(event, { suppress: true }));
      listen(expandBtn, "pointercancel", (event) => finishGesture(event, { suppress: false }));
      listen(expandBtn, "click", () => {
        if (suppressClick) { suppressClick = false; return; }
        applyHeightMode(heightMode === "normal" ? "tall" : heightMode === "tall" ? "full" : "normal");
      });
    }

    function saveSessionDraft(sessionId) {
      if (!sessionId) return;
      const value = String(textarea.value || "");
      if (value) storageSetItem(sessionDraftKey(sessionId), value);
      else storageRemoveItem(sessionDraftKey(sessionId));
    }

    function loadSessionDraft(sessionId) {
      hideModelPicker();
      textarea.value = sessionId ? storageGetItem(sessionDraftKey(sessionId)) || "" : "";
      autoGrow();
    }

    // Server-authoritative draft replacement (draft sync). Sets the textarea
    // and the localStorage cache directly — programmatic writes fire no input
    // event, so this never triggers a re-upload.
    function setDraftFromServer(sessionId, text) {
      if (!sessionId) return;
      hideModelPicker();
      textarea.value = String(text || "");
      saveSessionDraft(sessionId);
      autoGrow();
    }

    function clearSessionDraft(sessionId) {
      if (sessionId) storageRemoveItem(sessionDraftKey(sessionId));
    }

    function blurComposer() {
      if (typeof textarea.blur !== "function") return;
      try { textarea.blur(); } catch (_) {}
    }

    function clearComposer({ blur = true } = {}) {
      hideModelPicker();
      textarea.value = "";
      clearSessionDraft(sessionState.get("selected"));
      autoGrow();
      if (blur) blurComposer();
    }

    function syncSendChoiceAttachmentPolicy() {
      const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);
      const label = hasAttachments ? "Attachments cannot be queued; send now or wait until idle" : "Send after current";
      sendChoiceLaterBtn.disabled = hasAttachments;
      sendChoiceLaterBtn.title = label;
      sendChoiceLaterBtn.setAttribute("aria-label", label);
    }

    function focusSendChoiceInitial() {
      requestFrame(() => {
        if (sendChoice.style.display !== "flex") return;
        const target = !sendChoiceNowBtn.disabled ? sendChoiceNowBtn : !sendChoiceLaterBtn.disabled ? sendChoiceLaterBtn : sendChoiceCancelBtn;
        if (!target || typeof target.focus !== "function") return;
        try { target.focus({ preventScroll: true }); } catch (_) {}
      });
    }

    function showSendChoice(raw, { opener = null } = {}) {
      prepareModalOpen();
      const focused = activeElement();
      sendChoiceReturnFocusEl = isHTMLElement(opener) ? opener : isHTMLElement(focused) ? focused : null;
      sendChoicePending = { sid: sessionState.get("selected"), text: raw, attachmentCount: getStagedAttachments().length };
      syncSendChoiceAttachmentPolicy();
      sendChoiceBackdrop.style.display = "block";
      sendChoice.style.display = "flex";
      afterModalVisibilityChanged();
      focusSendChoiceInitial();
    }

    function hideSendChoice({ restoreFocus = false } = {}) {
      const target = sendChoiceReturnFocusEl;
      sendChoiceReturnFocusEl = null;
      sendChoicePending = null;
      syncSendChoiceAttachmentPolicy();
      sendChoiceBackdrop.style.display = "none";
      sendChoice.style.display = "none";
      afterModalVisibilityChanged();
      if (restoreFocus) restoreModalFocus(target, () => sendChoice.style.display === "flex");
    }

    listen(textarea, "input", () => {
      autoGrow();
      saveSessionDraft(sessionState.get("selected"));
      onDraftEdited(textarea.value);
      modelPickerFocus = -1;
      syncModelPicker();
    });
    listen(textarea, "click", syncModelPicker);
    listen(textarea, "select", () => {
      if (modelPickerOpen) positionModelPicker();
    });
    listen(textarea, "scroll", () => {
      if (modelPickerOpen) positionModelPicker();
    });
    listen(textarea, "keyup", (event) => {
      // Acceptance and dismissal must survive their corresponding keyup.
      if (!event.isComposing && !["Enter", "Tab", "Escape"].includes(event.key)) syncModelPicker();
    });
    listen(textarea, "keydown", (event) => {
      if (modelPickerOpen) {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const delta = event.key === "ArrowDown" ? 1 : -1;
          if (modelPickerOptions.length) modelPickerFocus = (modelPickerFocus + delta + modelPickerOptions.length) % modelPickerOptions.length;
          syncModelPickerSelection({ scroll: true });
          return;
        }
        // Only unmodified keys commit the picker: Ctrl/Cmd+Enter must fall through
        // to send (popup must never block sending), Shift+Enter must insert a newline.
        const isPickerAccept = (event.key === "Enter" && !event.ctrlKey && !event.metaKey && !event.shiftKey)
          || (event.key === "Tab" && modelPickerKind === "path");
        if (isPickerAccept && !event.isComposing) {
          event.preventDefault();
          selectPickerOption(modelPickerOptions[modelPickerFocus >= 0 ? modelPickerFocus : 0]);
          return;
        }
      }
      if (!modelPickerOpen && historyController && historyController.handleKeydown(event)) return;
      if (event.key === "Escape") {
        if (isModalOpen()) return;
        if (modelPickerOpen) {
          event.preventDefault();
          hideModelPicker();
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        blurComposer();
        return;
      }
      if (event.key !== "Enter" || event.isComposing || !(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      form.requestSubmit();
    });
    if (windowTarget) {
      listen(windowTarget, "resize", () => { onAutoGrow(); positionModelPicker(); });
      const visualViewport = windowTarget.visualViewport;
      if (visualViewport && typeof visualViewport.addEventListener === "function") listen(visualViewport, "resize", positionModelPicker);
    }
    bindExpandButton();

    form.onsubmit = async (event) => {
      event.preventDefault();
      const sessionId = sessionState.get("selected");
      if (!sessionId) { setToast("select a session first"); return; }
      if (sessionLaunchFailed(getSessionInfo(sessionId))) { setToast("failed session cannot receive messages"); return; }
      const raw = textarea.value;
      if (!raw || !raw.trim() || sessionState.get("sending")) return;
      const sessionInfo = getSessionInfo(sessionId);
      if (unsupportedPiThinkingCommand(raw, sessionInfo)) {
        setToast("this session runs an older bridge — send /reload to enable /effort");
        return;
      }
      if (sessionState.get("running")) {
        const focused = activeElement();
        showSendChoice(raw, { opener: isHTMLElement(focused) ? focused : textarea });
        return;
      }
      const ok = await sendText(raw);
      if (ok && historyController) historyController.reset();
      if (ok) blurComposer();
      if (ok && textarea.value === raw) {
        clearComposer({ blur: false });
        resetHeightMode();
      }
    };

    sendChoiceNowBtn.onclick = async () => {
      const raw = sendChoicePending && sendChoicePending.text;
      const sessionId = sendChoicePending && sendChoicePending.sid;
      hideSendChoice({ restoreFocus: true });
      if (!raw || !sessionId) return;
      const ok = await sendText(raw, { sid: sessionId });
      if (ok && historyController) historyController.reset();
      if (ok) blurComposer();
      if (ok && sessionId === sessionState.get("selected") && textarea.value === raw) {
        clearComposer({ blur: false });
        resetHeightMode();
      }
    };
    sendChoiceLaterBtn.onclick = async () => {
      const raw = sendChoicePending && sendChoicePending.text;
      const sessionId = sendChoicePending && sendChoicePending.sid;
      const hasAttachments = Boolean(sendChoicePending && sendChoicePending.attachmentCount > 0);
      if (hasAttachments) { setToast("attachments can only be sent now; wait until idle to queue text with files"); return; }
      hideSendChoice({ restoreFocus: true });
      if (!raw || !sessionId) return;
      const ok = await enqueueComposerText(raw, { sid: sessionId });
      if (ok && sessionId === sessionState.get("selected") && textarea.value === raw) {
        clearComposer();
        resetHeightMode();
      }
    };
    sendChoiceCancelBtn.onclick = () => hideSendChoice({ restoreFocus: true });
    sendChoiceBackdrop.onclick = () => hideSendChoice({ restoreFocus: true });

    const unsubscribeSessionState = [
      sessionState.subscribe("selected", () => { hideModelPicker(); loadHeightMode(sessionState.get("selected")); syncSendButtonState(); }),
      sessionState.subscribe("sending", syncSendButtonState),
      sessionCatalog.subscribe("sessionIndex", syncSendButtonState),
    ];
    syncSendButtonState();
    normalCap();
    loadHeightMode(sessionState.get("selected"));

    return Object.freeze({
      autoGrow,
      clearComposer,
      clearSessionDraft,
      loadSessionDraft,
      saveSessionDraft,
      setDraftFromServer,
      sendText,
      showSendChoice,
      hideSendChoice,
      isSendChoiceOpen: () => sendChoice.style.display === "flex",
      syncComposerState,
      syncSendButtonState,
      dispose() {
        form.onsubmit = null;
        sendChoiceNowBtn.onclick = null;
        sendChoiceLaterBtn.onclick = null;
        sendChoiceCancelBtn.onclick = null;
        sendChoiceBackdrop.onclick = null;
        hideModelPicker();
        while (cleanups.length) cleanups.pop()();
        while (unsubscribeSessionState.length) unsubscribeSessionState.pop()();
      },
    });
  }

export { createComposerController };
