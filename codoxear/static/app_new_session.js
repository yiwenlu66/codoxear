(function () {
  "use strict";

  const codoxearLaunch = window.CodoxearLaunch;
  if (
    !codoxearLaunch ||
    typeof codoxearLaunch.normalizeAgentBackendName !== "function" ||
    typeof codoxearLaunch.agentBackendDisplayName !== "function" ||
    typeof codoxearLaunch.sessionAgentBackend !== "function" ||
    typeof codoxearLaunch.sessionProviderChoice !== "function" ||
    typeof codoxearLaunch.providerChoicesForBackend !== "function" ||
    typeof codoxearLaunch.defaultsForAgentBackend !== "function" ||
    typeof codoxearLaunch.reasoningChoicesForBackend !== "function" ||
    typeof codoxearLaunch.providerModelDisplay !== "function" ||
    typeof codoxearLaunch.modelOptionMatches !== "function" ||
    typeof codoxearLaunch.loadRememberedProviderChoice !== "function" ||
    typeof codoxearLaunch.rememberProviderChoice !== "function" ||
    typeof codoxearLaunch.loadRememberedProviderModelChoice !== "function" ||
    typeof codoxearLaunch.rememberedProviderModelAbsentChoice !== "function" ||
    typeof codoxearLaunch.rememberProviderModelChoice !== "function"
  )
    throw new Error("Codoxear launch helpers failed to load");

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`new session controller dependency missing: ${name}`);
    return value;
  }

  function requireInputNode(value, name) {
    if (!value || !("value" in value)) throw new TypeError(`new session controller dependency missing: ${name}`);
    return value;
  }

  function requireClassListNode(value, name) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function" || typeof value.classList.remove !== "function") throw new TypeError(`new session controller dependency missing: ${name}`);
    return value;
  }

  function requireTextNode(value, name) {
    if (!value || !("textContent" in value)) throw new TypeError(`new session controller dependency missing: ${name}`);
    return value;
  }

  function requirePresentNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`new session controller dependency missing: ${name}`);
    return value;
  }

  // Provider/model + reasoning decision logic for the New Session dialog.
  // Selection state (backend/provider/reasoningEffort/literalModel/launchPresetProviderAbsent)
  // is owned by app.js and exposed to this controller through accessor + mutator
  // closures; DOM nodes the controller writes to are injected directly. Pure
  // launch helpers come from window.CodoxearLaunch.
  function createNewSessionController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("new session controller dependency missing: options");

    const backend = requireFunction(options.backend, "backend");
    const provider = requireFunction(options.provider, "provider");
    const reasoningEffort = requireFunction(options.reasoningEffort, "reasoningEffort");
    const literalModelInputValue = requireFunction(options.literalModelInputValue, "literalModelInputValue");
    const launchPresetProviderAbsent = requireFunction(options.launchPresetProviderAbsent, "launchPresetProviderAbsent");
    const defaultsSource = requireFunction(options.defaultsSource, "defaultsSource");
    const latestSessions = requireFunction(options.latestSessions, "latestSessions");
    const tmuxAvailable = requireFunction(options.tmuxAvailable, "tmuxAvailable");

    const assignProvider = requireFunction(options.assignProvider, "assignProvider");
    const assignReasoningEffort = requireFunction(options.assignReasoningEffort, "assignReasoningEffort");
    const assignLiteralModelInputValue = requireFunction(options.assignLiteralModelInputValue, "assignLiteralModelInputValue");
    const assignLaunchPresetProviderAbsent = requireFunction(options.assignLaunchPresetProviderAbsent, "assignLaunchPresetProviderAbsent");

    const modelInput = requireInputNode(options.modelInput, "modelInput");
    const modelField = requireClassListNode(options.modelField, "modelField");
    const status = requireTextNode(options.status, "status");
    const reasoningBtn = requirePresentNode(options.reasoningBtn, "reasoningBtn");

    const setPickerButtonContent = requireFunction(options.setPickerButtonContent, "setPickerButtonContent");
    const renderReasoningMenu = requireFunction(options.renderReasoningMenu, "renderReasoningMenu");
    const renderModelMenu = requireFunction(options.renderModelMenu, "renderModelMenu");
    const setFast = requireFunction(options.setFast, "setFast");
    const setBackend = requireFunction(options.setBackend, "setBackend");
    const setTmuxChecked = requireFunction(options.setTmuxChecked, "setTmuxChecked");
    const applyDialogMenus = requireFunction(options.applyDialogMenus, "applyDialogMenus");
    const closeModelMenu = requireFunction(options.closeModelMenu, "closeModelMenu");

    function newSessionProviderChoices() {
      return codoxearLaunch.providerChoicesForBackend(backend(), defaultsSource());
    }

    function newSessionHasProviderChoices() {
      return newSessionProviderChoices().length > 0;
    }

    function defaultNewSessionProviderChoice() {
      const choices = newSessionProviderChoices();
      if (!choices.length) return "";
      const defaults = codoxearLaunch.defaultsForAgentBackend(backend(), defaultsSource());
      const configured = typeof defaults.provider_choice === "string" ? defaults.provider_choice.trim() : "";
      const remembered = codoxearLaunch.loadRememberedProviderChoice(backend());
      if (remembered && choices.includes(remembered)) return remembered;
      if (configured && choices.includes(configured)) return configured;
      const currentProvider = provider();
      if (currentProvider && choices.includes(currentProvider)) return currentProvider;
      return choices[0] || "";
    }

    function newSessionProviderModelDisplay(model, providerChoice = "") {
      return codoxearLaunch.providerModelDisplay(model, providerChoice, {
        hasProviderChoices: newSessionHasProviderChoices(),
        allowCustomProvider: newSessionAllowsCustomProvider(),
      });
    }

    function newSessionAllowsCustomProvider() {
      return backend() === "pi";
    }

    function parseNewSessionProviderModelInput(value = modelInput.value) {
      const raw = String(value || "").trim();
      const choices = newSessionProviderChoices();
      const allowCustomProvider = newSessionAllowsCustomProvider();
      const hasProviders = choices.length > 0 || allowCustomProvider;
      const defaults = codoxearLaunch.defaultsForAgentBackend(backend(), defaultsSource());
      const fallbackModel = typeof defaults.model === "string" && defaults.model.trim() ? defaults.model.trim() : "default";
      let providerChoice = hasProviders ? defaultNewSessionProviderChoice() : "";
      let model = raw || fallbackModel;
      let providerError = "";
      const providerAbsent = Boolean(launchPresetProviderAbsent() && raw && raw === literalModelInputValue());
      if (providerAbsent) providerChoice = "";
      if (hasProviders && raw.includes("/") && raw !== literalModelInputValue()) {
        const slash = raw.indexOf("/");
        const typedProvider = raw.slice(0, slash).trim();
        const typedModel = raw.slice(slash + 1).trim();
        if (typedProvider && (choices.includes(typedProvider) || allowCustomProvider)) {
          providerChoice = typedProvider;
        } else if (typedProvider) {
          providerError = `Provider must be one of ${choices.join(", ")}.`;
        }
        model = typedModel || fallbackModel;
      }
      return { providerChoice, model: model || "default", providerError, providerAbsent };
    }

    function rememberedNewSessionProviderModelChoice() {
      const remembered = codoxearLaunch.loadRememberedProviderModelChoice(backend());
      if (!remembered) return null;
      const absent = codoxearLaunch.rememberedProviderModelAbsentChoice(remembered);
      if (absent) return absent;
      const parsed = parseNewSessionProviderModelInput(remembered);
      if (parsed.providerError) return null;
      const choices = newSessionProviderChoices();
      if (choices.length && parsed.providerChoice && !choices.includes(parsed.providerChoice) && !newSessionAllowsCustomProvider()) return null;
      return parsed;
    }

    function newSessionDefaultsWarningText() {
      const defaults = defaultsSource();
      const warnings = defaults && typeof defaults === "object" && defaults.warnings && typeof defaults.warnings === "object" ? defaults.warnings : null;
      if (!warnings) return "";
      const names = Object.keys(warnings).map(codoxearLaunch.agentBackendDisplayName).filter(Boolean);
      if (!names.length) return "";
      return `Launch defaults degraded for ${names.join(", ")}; using safe defaults.`;
    }

    function clearNewSessionProviderModelError() {
      modelField.classList.remove("error");
      if (String(status.textContent || "").startsWith("Provider must be one of ")) {
        status.textContent = newSessionDefaultsWarningText();
      }
    }

    function syncNewSessionProviderFromModelInput() {
      const parsed = parseNewSessionProviderModelInput();
      modelField.classList.toggle("error", Boolean(parsed.providerError));
      if (!parsed.providerError) clearNewSessionProviderModelError();
      if (parsed.providerChoice && !parsed.providerError && parsed.providerChoice !== provider()) {
        setNewSessionProvider(parsed.providerChoice);
      }
      return parsed;
    }

    function currentNewSessionModelForCapabilities() {
      const parsed = parseNewSessionProviderModelInput();
      const model = parsed.model;
      return model && model.toLowerCase() !== "default" ? model : null;
    }

    function currentReasoningChoices() {
      const parsed = parseNewSessionProviderModelInput();
      return codoxearLaunch.reasoningChoicesForBackend(backend(), defaultsSource(), {
        provider: parsed.providerAbsent ? "" : parsed.providerChoice || provider(),
        model: currentNewSessionModelForCapabilities(),
      });
    }

    function newSessionModelOption(model, { providerChoice = "", recent = false, configured = false, providerAbsent = false } = {}) {
      const cleanModel = String(model || "").trim() || "default";
      const cleanProvider = providerAbsent ? "" : String(providerChoice || "").trim();
      const displayText = newSessionProviderModelDisplay(cleanModel, cleanProvider);
      return {
        model: cleanModel,
        providerChoice: cleanProvider,
        providerAbsent: !!providerAbsent,
        recent: !!recent,
        configured: !!configured,
        displayText,
        searchText: cleanProvider ? `${cleanProvider}/${cleanModel} ${cleanModel}` : cleanModel,
      };
    }

    function addNewSessionModelOption(out, seen, model, opts = {}) {
      const cleanModel = String(model || "").trim();
      if (!cleanModel) return;
      const cleanProvider = String(opts.providerChoice || "").trim();
      const key = `${cleanProvider}|${cleanModel}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push(newSessionModelOption(cleanModel, opts));
    }

    function sessionModelOptions() {
      const seen = new Set();
      const out = [];
      const currentBackend = backend();
      const defaults = codoxearLaunch.defaultsForAgentBackend(currentBackend, defaultsSource());
      const providerChoices = newSessionProviderChoices();
      const configuredDefault = typeof defaults.model === "string" ? defaults.model.trim() : "";
      const activeProvider = providerChoices.length ? defaultNewSessionProviderChoice() : "";
      if (configuredDefault) addNewSessionModelOption(out, seen, configuredDefault, { providerChoice: activeProvider, configured: true });
      for (const item of latestSessions()) {
        if (codoxearLaunch.sessionAgentBackend(item) !== currentBackend) continue;
        const model = typeof item.model === "string" ? item.model.trim() : "";
        if (!model) continue;
        const prov = codoxearLaunch.sessionProviderChoice(item);
        const providerChoice = providerChoices.includes(prov) || (prov && newSessionAllowsCustomProvider()) ? prov : "";
        const providerAbsent = currentBackend === "pi" && !providerChoice && !(typeof item.model_provider === "string" && item.model_provider.trim());
        addNewSessionModelOption(out, seen, model, { providerChoice, providerAbsent, recent: true });
      }
      const configuredModels = Array.isArray(defaults.models) ? defaults.models : [];
      if (providerChoices.length) {
        for (const providerChoice of providerChoices) {
          for (const value of configuredModels) addNewSessionModelOption(out, seen, value, { providerChoice, configured: true });
        }
      } else {
        for (const value of configuredModels) addNewSessionModelOption(out, seen, value, { configured: true });
      }
      if (!out.length) addNewSessionModelOption(out, seen, "default", { providerChoice: activeProvider, configured: true });
      return out;
    }

    function filteredNewSessionModelOptions() {
      const query = String(modelInput.value || "").trim().toLowerCase();
      const options = sessionModelOptions();
      if (!query) return options.slice(0, 12);
      const exact = options.filter((item) => String(item.model || "").toLowerCase() === query || String(item.searchText || "").toLowerCase() === query);
      const prefix = options.filter((item) => !exact.includes(item) && String(item.searchText || item.model || "").toLowerCase().startsWith(query));
      const contains = options.filter((item) => !exact.includes(item) && !prefix.includes(item) && codoxearLaunch.modelOptionMatches(item, query));
      return exact.concat(prefix, contains).slice(0, 12);
    }

    function setNewSessionReasoningEffort(value) {
      const choices = currentReasoningChoices();
      const next = String(value || "").trim().toLowerCase();
      const fallback = String(codoxearLaunch.defaultsForAgentBackend(backend(), defaultsSource()).reasoning_effort || "").trim().toLowerCase();
      const resolved = choices.includes(next) ? next : choices.includes(fallback) ? fallback : choices[0] || "high";
      assignReasoningEffort(resolved);
      setPickerButtonContent(reasoningBtn, resolved);
    }

    function setNewSessionProvider(value) {
      const currentBackend = backend();
      const options = codoxearLaunch.providerChoicesForBackend(currentBackend);
      const fallback = String(codoxearLaunch.defaultsForAgentBackend(currentBackend, defaultsSource()).provider_choice || "").trim();
      const next = String(value || "").trim();
      const resolved = options.includes(next) || (next && newSessionAllowsCustomProvider()) ? next : (fallback && options.includes(fallback) ? fallback : options[0] || "");
      assignProvider(resolved);
      codoxearLaunch.rememberProviderChoice(currentBackend, resolved);
      setNewSessionReasoningEffort(reasoningEffort());
      renderReasoningMenu();
    }

    function selectNewSessionModel(option) {
      assignLiteralModelInputValue("");
      assignLaunchPresetProviderAbsent(false);
      const item = option && typeof option === "object" ? option : newSessionModelOption(option || "default");
      const selectedProvider = item.providerAbsent ? "" : item.providerChoice || provider();
      if (item.providerChoice && !item.providerAbsent && newSessionProviderChoices().includes(item.providerChoice)) {
        setNewSessionProvider(item.providerChoice);
      }
      modelInput.value = newSessionProviderModelDisplay(item.model || "default", selectedProvider);
      if (item.providerAbsent) {
        assignLiteralModelInputValue(modelInput.value);
        assignLaunchPresetProviderAbsent(true);
      }
      codoxearLaunch.rememberProviderModelChoice(backend(), selectedProvider, item.model || "default", { providerAbsent: Boolean(item.providerAbsent) });
      modelField.classList.remove("error");
      closeModelMenu();
      setNewSessionReasoningEffort(reasoningEffort());
      renderReasoningMenu();
      applyDialogMenus();
      modelInput.focus();
      const end = modelInput.value.length;
      try {
        modelInput.setSelectionRange(end, end);
      } catch (_) {}
    }

    function launchPresetProviderChoice(s) {
      if (!s || typeof s !== "object") return "";
      const backendValue = codoxearLaunch.sessionAgentBackend(s);
      const prov = typeof s.model_provider === "string" ? s.model_provider.trim() : "";
      if (backendValue === "pi") return prov;
      if (backendValue === "cc") return "";
      const explicit = typeof s.provider_choice === "string" ? s.provider_choice.trim() : "";
      if (explicit) return explicit;
      if (!prov) return "";
      if (backendValue === "codex" && prov === "openai") {
        const auth = typeof s.preferred_auth_method === "string" ? s.preferred_auth_method.trim() : "";
        return auth === "chatgpt" ? "chatgpt" : "openai-api";
      }
      return prov;
    }

    function applyNewSessionLaunchPreset(sessionInfo) {
      const s = sessionInfo && typeof sessionInfo === "object" ? sessionInfo : null;
      if (!s) return false;
      const backendValue = codoxearLaunch.sessionAgentBackend(s);
      if (backendValue !== backend()) setBackend(backendValue, { resetSelections: true });
      const prov = launchPresetProviderChoice(s);
      const providerChoices = newSessionProviderChoices();
      const acceptsProvider = Boolean(prov && (providerChoices.includes(prov) || newSessionAllowsCustomProvider()));
      if (acceptsProvider) setNewSessionProvider(prov);
      const model = typeof s.model === "string" && s.model.trim() ? s.model.trim() : "";
      const providerAbsent = backendValue === "pi" && !prov;
      assignLiteralModelInputValue("");
      assignLaunchPresetProviderAbsent(false);
      if (model || providerAbsent || acceptsProvider) {
        modelInput.value = newSessionProviderModelDisplay(model || "default", acceptsProvider ? prov : "");
        if (!acceptsProvider) {
          assignLiteralModelInputValue(modelInput.value);
          assignLaunchPresetProviderAbsent(providerAbsent);
        }
      }
      clearNewSessionProviderModelError();
      const reasoning = typeof s.reasoning_effort === "string" ? s.reasoning_effort.trim().toLowerCase() : "";
      if (reasoning) setNewSessionReasoningEffort(reasoning);
      const defaults = codoxearLaunch.defaultsForAgentBackend(backend(), defaultsSource());
      if (defaults && defaults.supports_fast) setFast(String(s.service_tier || "").trim().toLowerCase() === "fast");
      if (tmuxAvailable()) setTmuxChecked(Boolean(s.transport === "tmux" || s.tmux_session || s.tmux_window));
      renderReasoningMenu();
      renderModelMenu();
      return true;
    }

    return Object.freeze({
      newSessionProviderChoices,
      newSessionHasProviderChoices,
      defaultNewSessionProviderChoice,
      newSessionProviderModelDisplay,
      newSessionAllowsCustomProvider,
      parseNewSessionProviderModelInput,
      rememberedNewSessionProviderModelChoice,
      newSessionDefaultsWarningText,
      clearNewSessionProviderModelError,
      syncNewSessionProviderFromModelInput,
      currentNewSessionModelForCapabilities,
      currentReasoningChoices,
      newSessionModelOption,
      sessionModelOptions,
      filteredNewSessionModelOptions,
      setNewSessionReasoningEffort,
      setNewSessionProvider,
      selectNewSessionModel,
      launchPresetProviderChoice,
      applyNewSessionLaunchPreset,
    });
  }

  window.CodoxearNewSession = Object.freeze({ createNewSessionController });
})();
