(function () {
  "use strict";

  const codoxearLaunch = window.CodoxearLaunch;
  if (
    !codoxearLaunch ||
    typeof codoxearLaunch.normalizeAgentBackendName !== "function" ||
    typeof codoxearLaunch.agentBackendDisplayName !== "function" ||
    typeof codoxearLaunch.agentBackendLogoPath !== "function" ||
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

  const codoxearDisplay = window.CodoxearDisplay;
  if (
    !codoxearDisplay ||
    typeof codoxearDisplay.baseName !== "function" ||
    typeof codoxearDisplay.shortSessionId !== "function" ||
    typeof codoxearDisplay.fmtRelativeAge !== "function" ||
    typeof codoxearDisplay.fuzzyRecentCwdScore !== "function"
  )
    throw new Error("Codoxear display helpers failed to load");

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

    // Cwd menu + recent-cwd suggestion dependencies.
    const cwdInput = requireInputNode(options.cwdInput, "cwdInput");
    const cwdMenu = requirePresentNode(options.cwdMenu, "cwdMenu");
    const cwdField = requireClassListNode(options.cwdField, "cwdField");
    const cwdHint = requirePresentNode(options.cwdHint, "cwdHint");
    const nameInput = requireInputNode(options.nameInput, "nameInput");
    const recentCwds = requireFunction(options.recentCwds, "recentCwds");
    const cwdMenuFocus = requireFunction(options.cwdMenuFocus, "cwdMenuFocus");
    const assignCwdMenuFocus = requireFunction(options.assignCwdMenuFocus, "assignCwdMenuFocus");
    const closeCwdMenu = requireFunction(options.closeCwdMenu, "closeCwdMenu");
    const el = requireFunction(options.el, "el");

    // Resume-conversation menu dependencies.
    const resumeMenu = requirePresentNode(options.resumeMenu, "resumeMenu");
    const resumeBtn = requirePresentNode(options.resumeBtn, "resumeBtn");
    const closeResumeMenu = requireFunction(options.closeResumeMenu, "closeResumeMenu");
    const fetchResumeCandidates = requireFunction(options.fetchResumeCandidates, "fetchResumeCandidates");

    // Worktree / tmux UI dependencies.
    const tmuxToggle = requirePresentNode(options.tmuxToggle, "tmuxToggle");
    const tmuxField = requirePresentNode(options.tmuxField, "tmuxField");
    const worktreeToggle = requirePresentNode(options.worktreeToggle, "worktreeToggle");
    const worktreeInput = requireInputNode(options.worktreeInput, "worktreeInput");
    const worktreeField = requirePresentNode(options.worktreeField, "worktreeField");
    const startBtn = requirePresentNode(options.startBtn, "startBtn");

    // Controller-owned state for cwd validation, resume selection, and the
    // debounced resume-candidate load. App.js reads these only through the
    // exposed accessors / methods.
    let cwdError = "";
    let cwdInfo = { exists: false, will_create: false, git_repo: false, git_root: "", git_branch: "" };
    let resumeSelection = null;
    let resumeCandidates = [];
    let resumeLoadSeq = 0;
    let resumeLoadTimer = null;
    let filesystemCwdOptions = [];
    let cwdSuggestInput = null;
    let cwdSuggestLoadSeq = 0;
    let cwdSuggestTimer = null;

    const MODEL_PROVIDER_SEPARATOR = " · ";

    function abbreviatedNewSessionProvider(providerChoice) {
      // Do not abbreviate: a lossy provider abbreviation is non-injective
      // (dexgem-messages and dexgem-responses both became "de…es"), which made
      // the display ambiguous and the parse resolve to the wrong provider.
      return String(providerChoice || "").trim();
    }

    function formatNewSessionProviderModelDisplay(model, providerChoice = "") {
      const cleanModel = String(model || "").trim() || "default";
      const cleanProvider = String(providerChoice || "").trim();
      return cleanProvider ? `${cleanModel}${MODEL_PROVIDER_SEPARATOR}${cleanProvider}` : cleanModel;
    }

    function displayedNewSessionProviderModel(value, { choices, allowCustomProvider }) {
      const raw = String(value || "").trim();
      const separator = raw.lastIndexOf(MODEL_PROVIDER_SEPARATOR);
      if (separator < 1) return null;
      const model = raw.slice(0, separator).trim();
      const providerDisplay = raw.slice(separator + MODEL_PROVIDER_SEPARATOR.length).trim();
      const candidates = choices.concat(provider()).filter((candidate, index, values) => candidate && values.indexOf(candidate) === index);
      const providerChoice = candidates.find((candidate) => candidate === providerDisplay || abbreviatedNewSessionProvider(candidate) === providerDisplay) || "";
      if (!model || !providerChoice || (!choices.includes(providerChoice) && !allowCustomProvider)) return null;
      return { model, providerChoice };
    }

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
      return formatNewSessionProviderModelDisplay(model, abbreviatedNewSessionProvider(providerChoice));
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
      const display = displayedNewSessionProviderModel(raw, { choices, allowCustomProvider });
      if (display) {
        providerChoice = display.providerChoice;
        model = display.model;
      } else if (hasProviders && raw.includes("/") && raw !== literalModelInputValue()) {
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
      const rawProviderModelMap = defaults.provider_models && typeof defaults.provider_models === "object" ? defaults.provider_models : null;
      const providerModelMap = rawProviderModelMap && Object.values(rawProviderModelMap).some((models) => Array.isArray(models) && models.length)
        ? rawProviderModelMap
        : null;
      const configuredModels = Array.isArray(defaults.models) ? defaults.models : [];
      const configuredDefaultBelongsToProvider = !providerModelMap
        || !activeProvider
        || (Array.isArray(providerModelMap[activeProvider]) && providerModelMap[activeProvider].includes(configuredDefault));
      const isConfiguredProviderModel = (providerChoice, model) => {
        if (providerChoices.length && !providerChoices.includes(providerChoice)) return false;
        if (providerModelMap) return Array.isArray(providerModelMap[providerChoice]) && providerModelMap[providerChoice].includes(model);
        return model === configuredDefault || configuredModels.includes(model);
      };
      if (configuredDefault && configuredDefaultBelongsToProvider) addNewSessionModelOption(out, seen, configuredDefault, { providerChoice: activeProvider, configured: true });
      for (const item of latestSessions()) {
        if (codoxearLaunch.sessionAgentBackend(item) !== currentBackend) continue;
        const model = typeof item.model === "string" ? item.model.trim() : "";
        if (!model) continue;
        const prov = codoxearLaunch.sessionProviderChoice(item);
        const providerChoice = providerChoices.includes(prov) ? prov : "";
        if (!isConfiguredProviderModel(providerChoice, model)) continue;
        addNewSessionModelOption(out, seen, model, { providerChoice, recent: true });
      }
      if (providerChoices.length) {
        if (providerModelMap) {
          // Pi: each model belongs to a specific provider — no cross-product.
          for (const providerChoice of providerChoices) {
            const models = providerModelMap[providerChoice];
            if (Array.isArray(models)) {
              for (const value of models) addNewSessionModelOption(out, seen, value, { providerChoice, configured: true });
            }
          }
        } else {
          for (const providerChoice of providerChoices) {
            for (const value of configuredModels) addNewSessionModelOption(out, seen, value, { providerChoice, configured: true });
          }
        }
      } else {
        for (const value of configuredModels) addNewSessionModelOption(out, seen, value, { configured: true });
      }
      if (!out.length) addNewSessionModelOption(out, seen, "default", { providerChoice: activeProvider, configured: true });
      return out;
    }

    function filteredNewSessionModelOptions() {
      const rawQuery = String(modelInput.value || "").trim();
      const display = displayedNewSessionProviderModel(rawQuery, {
        choices: newSessionProviderChoices(),
        allowCustomProvider: newSessionAllowsCustomProvider(),
      });
      const query = display ? `${display.providerChoice}/${display.model}`.toLowerCase() : rawQuery.toLowerCase();
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

    // ---- Cwd input + recent-cwd suggestion menu ----

    function renderRecentCwdOptions() {
      const out = [];
      const seen = new Set();
      for (const raw of recentCwds()) {
        const cwd = typeof raw === "string" ? raw.trim() : "";
        if (!cwd || seen.has(cwd)) continue;
        seen.add(cwd);
        out.push(cwd);
      }
      return out;
    }

    function filteredRecentCwdOptions() {
      const items = renderRecentCwdOptions();
      const query = String(cwdInput.value || "").trim();
      if (!query) return items.slice(0, 10).map((cwd, idx) => ({ cwd, idx, score: 1000 - idx }));
      return items
        .map((cwd, idx) => ({ cwd, idx, score: codoxearDisplay.fuzzyRecentCwdScore(cwd, query) }))
        .filter((item) => item.score >= 0)
        .sort((a, b) => b.score - a.score || a.idx - b.idx || a.cwd.localeCompare(b.cwd))
        .slice(0, 10);
    }

    function syncNewSessionNamePlaceholder() {
      const fallback = codoxearDisplay.baseName(String(cwdInput.value || "").trim());
      nameInput.placeholder = fallback || "session-name";
    }

    function syncNewSessionCwdHint() {
      const errorText = String(cwdError || "").trim();
      const hintText = !errorText && cwdInfo && cwdInfo.will_create ? "Directory will be created when you start the session." : "";
      const text = errorText || hintText;
      cwdField.classList.toggle("error", !!errorText);
      cwdHint.classList.toggle("danger", !!errorText);
      cwdHint.textContent = text;
    }

    function setNewSessionCwdError(message) {
      cwdError = String(message || "").trim();
      syncNewSessionCwdHint();
    }

    function clearNewSessionCwdInfo() {
      cwdInfo = { exists: false, will_create: false, git_repo: false, git_root: "", git_branch: "" };
      syncNewSessionCwdHint();
    }

    function applyNewSessionCwdSuggestion(cwd) {
      cwdInput.value = String(cwd || "");
      setNewSessionCwdError("");
      syncNewSessionNamePlaceholder();
      closeCwdMenu();
      applyDialogMenus();
      scheduleNewSessionResumeLoad();
      cwdInput.focus();
      const end = cwdInput.value.length;
      try {
        cwdInput.setSelectionRange(end, end);
      } catch (_) {}
    }

    function cwdSuggestionQuery(raw) {
      const value = String(raw || "").trim();
      const slash = value.lastIndexOf("/");
      if (slash < 0) return { path: "/", prefix: value };
      if (slash === 0) return { path: "/", prefix: value.slice(1) };
      return { path: value.slice(0, slash) || "/", prefix: value.slice(slash + 1) };
    }

    async function loadFilesystemCwdOptions(raw, seq) {
      const query = cwdSuggestionQuery(raw);
      try {
        const response = await fetch(`/api/cwd-suggest?path=${encodeURIComponent(query.path)}&prefix=${encodeURIComponent(query.prefix)}`, {
          credentials: "same-origin",
        });
        if (!response.ok) throw new Error(`cwd suggestions failed: ${response.status}`);
        const result = await response.json();
        if (seq !== cwdSuggestLoadSeq) return;
        filesystemCwdOptions = Array.isArray(result && result.directories)
          ? result.directories.filter((item) => item && typeof item.name === "string" && typeof item.path === "string")
          : [];
        renderRecentCwdMenu();
        applyDialogMenus();
      } catch (_) {
        if (seq !== cwdSuggestLoadSeq) return;
        filesystemCwdOptions = [];
        renderRecentCwdMenu();
        applyDialogMenus();
      }
    }

    function scheduleFilesystemCwdLoad() {
      const raw = String(cwdInput.value || "").trim();
      if (raw === cwdSuggestInput) return;
      cwdSuggestInput = raw;
      filesystemCwdOptions = [];
      const seq = ++cwdSuggestLoadSeq;
      if (cwdSuggestTimer) clearTimeout(cwdSuggestTimer);
      cwdSuggestTimer = setTimeout(() => {
        cwdSuggestTimer = null;
        void loadFilesystemCwdOptions(raw, seq);
      }, 250);
    }

    function cwdMenuOptions(raw) {
      const recent = filteredRecentCwdOptions().map((item) => ({ cwd: item.cwd, source: "recent" }));
      const seen = new Set(recent.map((item) => item.cwd));
      const prefix = cwdSuggestionQuery(raw).prefix.toLocaleLowerCase();
      for (const item of filesystemCwdOptions) {
        const cwd = String(item.path || "").trim();
        const name = String(item.name || "").trim();
        if (!cwd || !name || seen.has(cwd) || (prefix && !name.toLocaleLowerCase().startsWith(prefix))) continue;
        seen.add(cwd);
        recent.push({ cwd, source: "filesystem" });
      }
      return recent;
    }

    function renderRecentCwdMenu() {
      cwdMenu.innerHTML = "";
      const raw = String(cwdInput.value || "").trim();
      scheduleFilesystemCwdLoad();
      const items = cwdMenuOptions(raw);
      let focus = cwdMenuFocus();
      if (focus >= items.length) {
        focus = items.length ? items.length - 1 : -1;
        assignCwdMenuFocus(focus);
      }
      if (!items.length) {
        const emptyText = raw ? "No matching directories. Start still uses the typed path." : "No recent directories";
        cwdMenu.appendChild(el("div", { class: "pickerEmpty", text: emptyText }));
        cwdInput.removeAttribute("aria-activedescendant");
        return items;
      }
      for (const [idx, item] of items.entries()) {
        const cwd = item.cwd;
        const active = focus === idx || (focus < 0 && raw === cwd);
        const btn = el("button", {
          id: `newSessionCwdOption-${idx}`,
          class: "fileMenuItem" + (active ? " active" : ""),
          type: "button",
          role: "option",
          "aria-selected": active ? "true" : "false",
          title: cwd,
        });
        btn.appendChild(el("span", { class: "fileMenuPath", text: cwd }));
        if (item.source === "recent") btn.appendChild(el("span", { class: "cwdSuggestionSource", text: "Recent" }));
        btn.onmousedown = (e) => e.preventDefault();
        btn.onclick = () => applyNewSessionCwdSuggestion(cwd);
        cwdMenu.appendChild(btn);
      }
      if (focus >= 0) cwdInput.setAttribute("aria-activedescendant", `newSessionCwdOption-${focus}`);
      else cwdInput.removeAttribute("aria-activedescendant");
      return items;
    }

    // ---- Resume-conversation menu ----

    function newSessionResumeLabel(item) {
      if (!item || typeof item !== "object") return "Start fresh";
      const alias = typeof item.alias === "string" ? item.alias.trim() : "";
      const firstUser = typeof item.first_user_message === "string" ? item.first_user_message.trim() : "";
      const primary = alias || firstUser || codoxearDisplay.shortSessionId(item.session_id);
      const ts = Number(item.updated_ts || 0);
      const age = ts > 0 ? codoxearDisplay.fmtRelativeAge(Math.max(0, Date.now() / 1000 - ts)) : "";
      return `${age ? `${age} | ` : ""}${primary}`;
    }

    function setNewSessionResumeSelection(item) {
      resumeSelection = item && typeof item === "object" ? item : null;
      setPickerButtonContent(
        resumeBtn,
        resumeSelection ? newSessionResumeLabel(resumeSelection) : "Start fresh",
        "",
        !resumeSelection
      );
      syncNewSessionWorktreeUi();
    }

    function renderNewSessionResumeMenu() {
      resumeMenu.innerHTML = "";
      const freshBtn = el("button", {
        class: "fileMenuItem" + (!resumeSelection ? " active" : ""),
        type: "button",
        title: "Start a new conversation",
      });
      freshBtn.appendChild(el("span", { class: "fileMenuPath", text: "Start fresh" }));
      freshBtn.onclick = () => {
        setNewSessionResumeSelection(null);
        closeResumeMenu();
        applyDialogMenus();
      };
      resumeMenu.appendChild(freshBtn);
      if (!resumeCandidates.length) {
        resumeMenu.appendChild(el("div", { class: "pickerEmpty", text: "No matching sessions" }));
        return;
      }
      for (const item of resumeCandidates) {
        const btn = el("button", {
          class: "fileMenuItem" + (resumeSelection && resumeSelection.session_id === item.session_id ? " active" : ""),
          type: "button",
          title: newSessionResumeLabel(item),
        });
        btn.appendChild(el("span", { class: "fileMenuPath", text: newSessionResumeLabel(item) }));
        btn.onclick = () => {
          setNewSessionResumeSelection(item);
          closeResumeMenu();
          applyDialogMenus();
        };
        resumeMenu.appendChild(btn);
      }
    }

    async function loadNewSessionResumeCandidates(cwd) {
      const raw = String(cwd || "").trim();
      const seq = ++resumeLoadSeq;
      const backendValue = backend();
      if (!raw) {
        setNewSessionCwdError("");
        resumeCandidates = [];
        setNewSessionResumeSelection(null);
        clearNewSessionCwdInfo();
        renderNewSessionResumeMenu();
        syncNewSessionWorktreeUi();
        return;
      }
      try {
        const res = await fetchResumeCandidates(raw, backendValue);
        if (seq !== resumeLoadSeq) return;
        cwdInfo = {
          exists: !!(res && res.exists),
          will_create: !!(res && res.will_create),
          git_repo: !!(res && res.git_repo),
          git_root: res && typeof res.git_root === "string" ? res.git_root : "",
          git_branch: res && typeof res.git_branch === "string" ? res.git_branch : "",
        };
        setNewSessionCwdError("");
        const items = Array.isArray(res && res.sessions) ? res.sessions.filter((item) => item && typeof item === "object" && typeof item.session_id === "string") : [];
        resumeCandidates = items;
        const currentId = resumeSelection && typeof resumeSelection.session_id === "string" ? resumeSelection.session_id : "";
        const next = currentId ? items.find((item) => item.session_id === currentId) || null : null;
        setNewSessionResumeSelection(next);
        renderNewSessionResumeMenu();
        syncNewSessionWorktreeUi();
      } catch (e) {
        if (seq !== resumeLoadSeq) return;
        resumeCandidates = [];
        setNewSessionResumeSelection(null);
        clearNewSessionCwdInfo();
        if (e && e.obj && e.obj.field === "cwd") setNewSessionCwdError(e.message);
        renderNewSessionResumeMenu();
        syncNewSessionWorktreeUi();
      }
    }

    function scheduleNewSessionResumeLoad() {
      if (resumeLoadTimer) clearTimeout(resumeLoadTimer);
      const cwd = String(cwdInput.value || "").trim();
      resumeLoadTimer = setTimeout(() => {
        resumeLoadTimer = null;
        void loadNewSessionResumeCandidates(cwd);
      }, 180);
    }

    function clearNewSessionResumeCandidates() {
      resumeCandidates = [];
    }

    function currentResumeSelection() {
      return resumeSelection;
    }

    function disposeResumeLoadTimer() {
      if (resumeLoadTimer) clearTimeout(resumeLoadTimer);
      resumeLoadTimer = null;
      if (cwdSuggestTimer) clearTimeout(cwdSuggestTimer);
      cwdSuggestTimer = null;
      cwdSuggestInput = null;
      filesystemCwdOptions = [];
      cwdSuggestLoadSeq += 1;
    }

    // ---- Worktree / tmux UI sync ----

    function syncNewSessionTmuxUi() {
      if (!tmuxAvailable()) tmuxToggle.checked = false;
      tmuxToggle.disabled = !tmuxAvailable();
      tmuxField.style.opacity = tmuxAvailable() ? "1" : "0.58";
    }

    function syncNewSessionWorktreeUi() {
      const canOffer = !!(cwdInfo && cwdInfo.git_repo) && !resumeSelection;
      if (!canOffer) worktreeToggle.checked = false;
      const enabled = canOffer && !!worktreeToggle.checked;
      worktreeField.style.display = canOffer ? "" : "none";
      worktreeInput.disabled = !enabled;
      worktreeInput.style.display = enabled ? "" : "none";
      if (resumeSelection) startBtn.textContent = "Resume session";
      else if (enabled) startBtn.textContent = "Create worktree session";
      else startBtn.textContent = "Start session";
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
      renderRecentCwdOptions,
      filteredRecentCwdOptions,
      syncNewSessionNamePlaceholder,
      syncNewSessionCwdHint,
      setNewSessionCwdError,
      clearNewSessionCwdInfo,
      applyNewSessionCwdSuggestion,
      renderRecentCwdMenu,
      newSessionResumeLabel,
      setNewSessionResumeSelection,
      renderNewSessionResumeMenu,
      scheduleNewSessionResumeLoad,
      clearNewSessionResumeCandidates,
      currentResumeSelection,
      disposeResumeLoadTimer,
      syncNewSessionTmuxUi,
      syncNewSessionWorktreeUi,
    });
  }

  function createNewSessionDialogController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("new session dialog dependency missing: options");
    const root = requirePresentNode(options.root, "root");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");
    const doc = requirePresentNode(options.document, "document");
    const win = requirePresentNode(options.window, "window");
    const addEvent = requireFunction(options.addEvent, "addEvent");
    const defaultsSource = requireFunction(options.defaultsSource, "defaultsSource");
    const latestSessions = requireFunction(options.latestSessions, "latestSessions");
    const recentCwds = requireFunction(options.recentCwds, "recentCwds");
    const tmuxAvailable = requireFunction(options.tmuxAvailable, "tmuxAvailable");
    const selectedSession = requireFunction(options.selectedSession, "selectedSession");
    const sessionForId = requireFunction(options.sessionForId, "sessionForId");
    const isMobile = requireFunction(options.isMobile, "isMobile");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const isModalTargetOpen = requireFunction(options.isModalTargetOpen, "isModalTargetOpen");
    const applyDialogMenus = requireFunction(options.applyDialogMenus, "applyDialogMenus");
    const positionDialogMenu = requireFunction(options.positionDialogMenu, "positionDialogMenu");
    const setPickerButtonContent = requireFunction(options.setPickerButtonContent, "setPickerButtonContent");
    const fetchResumeCandidates = requireFunction(options.fetchResumeCandidates, "fetchResumeCandidates");
    const spawnSession = requireFunction(options.spawnSession, "spawnSession");

    let backend = "pi";
    let provider = "chatgpt";
    let reasoningEffort = "high";
    let fast = false;
    let literalModelInputValue = "";
    let launchPresetProviderAbsent = false;
    let cwdMenuOpen = false;
    let cwdMenuFocus = -1;
    let modelMenuOpen = false;
    let modelMenuFocus = -1;
    let reasoningMenuOpen = false;
    let resumeMenuOpen = false;
    let startBusy = false;
    let returnFocusEl = null;

    const backdrop = el("div", { class: "modalBackdrop", id: "newSessionBackdrop" });
    const closeBtn = el("button", { id: "newSessionCloseBtn", class: "icon-btn", title: "Close", "aria-label": "Close", type: "button", html: iconSvg("x") });
    const status = el("div", { class: "muted", id: "newSessionStatus", text: "" });
    const cwdInput = el("input", { id: "newSessionCwdInput", type: "text", placeholder: "/path/to/project", autocomplete: "off", spellcheck: "false", role: "combobox", "aria-autocomplete": "list", "aria-controls": "newSessionCwdMenu", "aria-expanded": "false" });
    const cwdMenu = el("div", { id: "newSessionCwdMenu", class: "filePickerMenu dialogPickerMenu cwdSuggestionMenu", role: "listbox" });
    const cwdField = el("div", { class: "pickerField cwdPickerField cwdComboboxField", id: "newSessionCwdField" }, [el("span", { class: "cwdComboboxIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }), cwdInput, cwdMenu]);
    const cwdHint = el("div", { class: "fieldHint", id: "newSessionCwdHint", text: "" });
    const nameInput = el("input", { id: "newSessionNameInput", type: "text", placeholder: "session-name", autocomplete: "off", spellcheck: "false" });
    const modelInput = el("input", { id: "newSessionModelInput", type: "text", placeholder: "Provider/model", autocomplete: "off", spellcheck: "false", role: "combobox", "aria-autocomplete": "list", "aria-controls": "newSessionModelMenu", "aria-expanded": "false" });
    const modelMenu = el("div", { id: "newSessionModelMenu", class: "filePickerMenu dialogPickerMenu cwdSuggestionMenu", role: "listbox" });
    const modelField = el("div", { class: "pickerField comboboxField cwdComboboxField", id: "newSessionModelField" }, [el("span", { class: "cwdComboboxIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }), modelInput, modelMenu]);
    const modelLabel = el("span", { class: "fieldLabel", text: "Provider / model" });
    const backendTabs = el("div", { class: "agentBackendTabs", id: "newSessionBackendTabs" });
    const backendName = el("span", { class: "agentBackendTabName", id: "newSessionBackendName" });
    const reasoningBtn = el("button", { id: "newSessionReasoningBtn", class: "filePickerBtn dialogPickerBtn sidePickerBtn", type: "button", "aria-label": "Choose reasoning effort", "aria-haspopup": "menu", "aria-expanded": "false" });
    const reasoningMenu = el("div", { id: "newSessionReasoningMenu", class: "filePickerMenu dialogPickerMenu" });
    const reasoningField = el("div", { class: "pickerField comboboxField pickerButtonField", id: "newSessionReasoningField" }, [el("span", { class: "cwdComboboxIcon", html: iconSvg("chevronDown"), "aria-hidden": "true" }), reasoningBtn]);
    const resumeBtn = el("button", { id: "newSessionResumeBtn", class: "filePickerBtn dialogPickerBtn sidePickerBtn", type: "button", "aria-label": "Choose a conversation to resume", "aria-haspopup": "menu", "aria-expanded": "false" });
    const resumeMenu = el("div", { id: "newSessionResumeMenu", class: "filePickerMenu dialogPickerMenu" });
    const tmuxToggle = el("input", { id: "newSessionTmuxToggle", type: "checkbox" });
    const tmuxField = el("div", { class: "field", id: "newSessionTmuxField" }, [el("span", { class: "fieldLabel", text: "Launch mode" }), el("label", { class: "checkField" }, [tmuxToggle, el("span", { text: "Create in tmux" })])]);
    const fastToggle = el("input", { id: "newSessionFastToggle", type: "checkbox" });
    const fastField = el("div", { class: "field compactToggleField", id: "newSessionFastField" }, [el("span", { class: "fieldLabel", text: "Speed" }), el("label", { class: "checkField" }, [fastToggle, el("span", { text: "Fast" })])]);
    const worktreeToggle = el("input", { id: "newSessionWorktreeToggle", type: "checkbox" });
    const worktreeInput = el("input", { id: "newSessionWorktreeBranchInput", type: "text", placeholder: "feature/my-branch", autocomplete: "off", spellcheck: "false", disabled: true });
    const worktreeField = el("div", { class: "field", id: "newSessionWorktreeField" }, [el("span", { class: "fieldLabel", text: "Git worktree branch" }), el("label", { class: "checkField" }, [worktreeToggle, el("span", { text: "Create a new worktree for this session" })]), worktreeInput]);
    const startBtn = el("button", { class: "primary", id: "newSessionStartBtn", type: "button", text: "Start session" });
    const cancelBtn = el("button", { id: "newSessionCancelBtn", type: "button", text: "Cancel" });
    const viewer = el("div", { class: "formViewer newSessionViewer", id: "newSessionViewer", role: "dialog", "aria-modal": "true", "aria-label": "New session" }, [
      el("div", { class: "queueHeader" }, [el("div", { class: "newSessionHeaderLead" }, [el("div", { class: "title", text: "New session" }), backendTabs, backendName]), el("div", { class: "actions" }, [closeBtn])]),
      status,
      el("div", { class: "formBody" }, [
        el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Working directory" }), cwdField, cwdHint]),
        el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Session name" }), nameInput]),
        el("div", { class: "formGrid newSessionRunConfigRow" }, [el("label", { class: "field" }, [modelLabel, modelField]), el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Reasoning effort" }), reasoningField]), fastField]),
        el("label", { class: "field" }, [el("span", { class: "fieldLabel", text: "Resume conversation" }), resumeBtn]),
        el("div", { class: "formGrid newSessionOptionsRow" }, [tmuxField]),
        worktreeField,
      ]),
      el("div", { class: "formActions" }, [cancelBtn, startBtn]),
    ]);
    root.appendChild(backdrop);
    root.appendChild(viewer);
    viewer.appendChild(modelMenu);
    viewer.appendChild(reasoningMenu);
    viewer.appendChild(resumeMenu);

    function closeMenus() {
      cwdMenuOpen = false; cwdMenuFocus = -1;
      modelMenuOpen = false; modelMenuFocus = -1;
      reasoningMenuOpen = false;
      resumeMenuOpen = false;
    }

    function renderReasoningMenu() {
      reasoningMenu.innerHTML = "";
      for (const value of controller.currentReasoningChoices()) {
        const btn = el("button", { class: "fileMenuItem" + (reasoningEffort === value ? " active" : ""), type: "button", title: value });
        btn.appendChild(el("span", { class: "fileMenuPath", text: value }));
        btn.onclick = () => { controller.setNewSessionReasoningEffort(value); reasoningMenuOpen = false; applyDialogMenus(); };
        reasoningMenu.appendChild(btn);
      }
    }

    function renderModelMenu() {
      modelMenu.innerHTML = "";
      const items = controller.filteredNewSessionModelOptions();
      const raw = String(modelInput.value || "").trim();
      const configured = String(codoxearLaunch.defaultsForAgentBackend(backend, defaultsSource()).model || "").trim();
      if (modelMenuFocus < 0) {
        const selected = raw || configured;
        const index = selected ? items.findIndex((item) => item.displayText === selected || item.model === selected) : -1;
        if (index >= 0) modelMenuFocus = index;
      }
      if (modelMenuFocus >= items.length) modelMenuFocus = items.length ? items.length - 1 : -1;
      if (!items.length) { modelMenu.appendChild(el("div", { class: "pickerEmpty", text: "No matching models" })); modelInput.removeAttribute("aria-activedescendant"); return items; }
      for (const [index, item] of items.entries()) {
        const title = item.displayText || controller.newSessionProviderModelDisplay(item.model, item.providerChoice);
        const active = modelMenuFocus === index || (modelMenuFocus < 0 && (raw === title || raw === item.model));
        const btn = el("button", { id: `newSessionModelOption-${index}`, class: "fileMenuItem" + (active ? " active" : ""), type: "button", role: "option", "aria-selected": active ? "true" : "false", "aria-label": title, title });
        btn.appendChild(el("span", { class: "fileMenuPath", text: item.model }));
        if (item.providerChoice) btn.appendChild(el("span", { class: "fileMenuHint", text: item.providerChoice }));
        btn.onmousedown = (event) => event.preventDefault();
        btn.onclick = () => controller.selectNewSessionModel(item);
        modelMenu.appendChild(btn);
      }
      if (modelMenuFocus >= 0) modelInput.setAttribute("aria-activedescendant", `newSessionModelOption-${modelMenuFocus}`); else modelInput.removeAttribute("aria-activedescendant");
      return items;
    }

    const controller = createNewSessionController({
      backend: () => backend, provider: () => provider, reasoningEffort: () => reasoningEffort,
      literalModelInputValue: () => literalModelInputValue, launchPresetProviderAbsent: () => launchPresetProviderAbsent,
      defaultsSource, latestSessions, tmuxAvailable,
      assignProvider: (value) => { provider = value; }, assignReasoningEffort: (value) => { reasoningEffort = value; },
      assignLiteralModelInputValue: (value) => { literalModelInputValue = value; }, assignLaunchPresetProviderAbsent: (value) => { launchPresetProviderAbsent = Boolean(value); },
      modelInput, modelField, status, reasoningBtn, setPickerButtonContent, renderReasoningMenu, renderModelMenu,
      setFast: (value) => { fast = !!value; fastToggle.checked = fast; }, setBackend: (value, opts) => setBackend(value, opts), setTmuxChecked: (value) => { tmuxToggle.checked = value; }, applyDialogMenus,
      closeModelMenu: () => { modelMenuOpen = false; modelMenuFocus = -1; }, cwdInput, cwdMenu, cwdField, cwdHint, nameInput, recentCwds,
      cwdMenuFocus: () => cwdMenuFocus, assignCwdMenuFocus: (value) => { cwdMenuFocus = value; }, closeCwdMenu: () => { cwdMenuOpen = false; cwdMenuFocus = -1; }, el,
      resumeMenu, resumeBtn, closeResumeMenu: () => { resumeMenuOpen = false; }, fetchResumeCandidates,
      tmuxToggle, tmuxField, worktreeToggle, worktreeInput, worktreeField, startBtn,
    });

    function renderBackendTabs() {
      backendTabs.innerHTML = "";
      for (const value of ["pi", "codex", "cc"]) {
        const active = backend === value;
        const label = codoxearLaunch.agentBackendDisplayName(value);
        const btn = el("button", { class: `agentBackendTab${active ? " active" : ""}`, type: "button", title: label, "aria-label": label }, [el("img", { class: "agentBackendTabLogo", src: codoxearLaunch.agentBackendLogoPath(value), alt: `${label} logo`, width: "20", height: "20" })]);
        btn.onclick = () => setBackend(value, { resetSelections: true });
        backendTabs.appendChild(btn);
      }
      backendName.textContent = codoxearLaunch.agentBackendDisplayName(backend);
    }

    function syncRunConfigUi() {
      const defaults = codoxearLaunch.defaultsForAgentBackend(backend, defaultsSource());
      const hasProviders = controller.newSessionHasProviderChoices() || controller.newSessionAllowsCustomProvider();
      modelLabel.textContent = hasProviders ? "Provider / model" : "Model";
      modelInput.placeholder = hasProviders ? "provider/model or model" : "Model";
      fastField.style.display = defaults.supports_fast ? "" : "none";
      if (!defaults.supports_fast) { fast = false; fastToggle.checked = false; }
    }

    function setBackend(value, { resetSelections = false } = {}) {
      const next = codoxearLaunch.normalizeAgentBackendName(value);
      const previous = backend;
      backend = next;
      codoxearLaunch.rememberBackendChoice(next);
      const defaults = codoxearLaunch.defaultsForAgentBackend(next, defaultsSource());
      const providers = codoxearLaunch.providerChoicesForBackend(next, defaultsSource());
      const defaultProvider = String(defaults.provider_choice || "").trim();
      const rememberedProvider = codoxearLaunch.loadRememberedProviderChoice(next);
      controller.setNewSessionProvider((resetSelections || previous !== next || !providers.includes(provider)) ? ((rememberedProvider && providers.includes(rememberedProvider) ? rememberedProvider : "") || defaultProvider || providers[0] || "") : provider);
      const modelDefault = String(defaults.model || "").trim();
      if (resetSelections || previous !== next) {
        const remembered = controller.rememberedNewSessionProviderModelChoice();
        const pair = remembered || controller.parseNewSessionProviderModelInput(controller.newSessionProviderModelDisplay(modelDefault || "default", provider));
        if (pair.providerChoice && (providers.includes(pair.providerChoice) || controller.newSessionAllowsCustomProvider())) controller.setNewSessionProvider(pair.providerChoice);
        modelInput.value = controller.newSessionProviderModelDisplay(pair.model || modelDefault || "default", pair.providerAbsent ? "" : pair.providerChoice || provider);
        literalModelInputValue = pair.providerAbsent ? modelInput.value : "";
        launchPresetProviderAbsent = Boolean(pair.providerAbsent);
        controller.clearNewSessionProviderModelError();
      }
      const choices = controller.currentReasoningChoices();
      const defaultEffort = String(defaults.reasoning_effort || "").trim().toLowerCase();
      controller.setNewSessionReasoningEffort((resetSelections || previous !== next || !choices.includes(reasoningEffort)) ? defaultEffort || choices[0] || "high" : reasoningEffort);
      if (resetSelections || previous !== next) { fast = String(defaults.service_tier || "").trim().toLowerCase() === "fast"; fastToggle.checked = fast; }
      syncRunConfigUi(); renderBackendTabs(); renderReasoningMenu(); renderModelMenu(); controller.scheduleNewSessionResumeLoad();
    }

    function applyMenus() {
      cwdMenu.classList.toggle("open", cwdMenuOpen); modelMenu.classList.toggle("open", modelMenuOpen); reasoningMenu.classList.toggle("open", reasoningMenuOpen); resumeMenu.classList.toggle("open", resumeMenuOpen);
      cwdInput.setAttribute("aria-expanded", cwdMenuOpen ? "true" : "false"); if (!cwdMenuOpen && cwdMenuFocus < 0) cwdInput.removeAttribute("aria-activedescendant");
      modelInput.setAttribute("aria-expanded", modelMenuOpen ? "true" : "false"); if (!modelMenuOpen && modelMenuFocus < 0) modelInput.removeAttribute("aria-activedescendant");
      reasoningBtn.setAttribute("aria-expanded", reasoningMenuOpen ? "true" : "false"); resumeBtn.setAttribute("aria-expanded", resumeMenuOpen ? "true" : "false");
      if (cwdMenuOpen) positionDialogMenu(cwdMenu, cwdInput); if (modelMenuOpen) positionDialogMenu(modelMenu, modelInput); if (reasoningMenuOpen) positionDialogMenu(reasoningMenu, reasoningBtn); if (resumeMenuOpen) positionDialogMenu(resumeMenu, resumeBtn);
    }

    function close() {
      const wasOpen = isModalTargetOpen(viewer);
      controller.disposeResumeLoadTimer(); status.textContent = ""; closeMenus(); applyDialogMenus(); backdrop.style.display = "none"; viewer.style.display = "none"; afterModalVisibilityChanged();
      const target = returnFocusEl; returnFocusEl = null;
      if (wasOpen && target && target.isConnected && typeof target.focus === "function" && !target.disabled) win.requestAnimationFrame(() => { if (!isModalTargetOpen(viewer)) target.focus({ preventScroll: true }); });
    }

    function open({ cwd = null, statusText = "", likeSession = null, returnFocusEl: opener = null } = {}) {
      const active = selectedSession();
      const current = active ? sessionForId(active) : null;
      const like = likeSession && typeof likeSession === "object" ? likeSession : null;
      returnFocusEl = opener instanceof HTMLElement ? opener : doc.activeElement instanceof HTMLElement ? doc.activeElement : null;
      prepareModalOpen();
      const initialCwd = typeof cwd === "string" && cwd.trim() ? cwd.trim() : like && like.cwd && like.cwd !== "?" ? like.cwd : current && current.cwd && current.cwd !== "?" ? current.cwd : "";
      const initialBackend = (like ? codoxearLaunch.sessionAgentBackend(like) : current ? codoxearLaunch.sessionAgentBackend(current) : "") || codoxearLaunch.loadRememberedBackendChoice() || codoxearLaunch.normalizeAgentBackendName(defaultsSource().default_backend);
      status.textContent = String(statusText || controller.newSessionDefaultsWarningText() || ""); cwdInput.value = initialCwd; nameInput.value = ""; modelInput.value = ""; literalModelInputValue = ""; launchPresetProviderAbsent = false;
      controller.syncNewSessionNamePlaceholder(); controller.clearNewSessionResumeCandidates(); controller.setNewSessionResumeSelection(null); controller.setNewSessionCwdError(""); controller.clearNewSessionCwdInfo();
      tmuxToggle.checked = tmuxAvailable(); worktreeToggle.checked = false; worktreeInput.value = ""; worktreeInput.disabled = true; worktreeInput.style.display = "none"; worktreeField.style.display = "none"; closeMenus(); controller.renderRecentCwdMenu(); setBackend(initialBackend, { resetSelections: true }); if (like) controller.applyNewSessionLaunchPreset(like); controller.renderNewSessionResumeMenu();
      backdrop.style.display = "block"; viewer.style.display = "flex"; afterModalVisibilityChanged(); controller.scheduleNewSessionResumeLoad(); controller.syncNewSessionTmuxUi(); controller.syncNewSessionWorktreeUi();
      win.requestAnimationFrame(() => { if (!isModalTargetOpen(viewer)) return; const target = isMobile() ? closeBtn : cwdInput; target.focus({ preventScroll: true }); if (target === cwdInput) { const end = cwdInput.value.length; try { cwdInput.setSelectionRange(end, end); } catch (_) {} } });
    }

    function start() {
      if (startBusy) return;
      const cwd = String(cwdInput.value || "").trim(); const agentBackend = backend; controller.setNewSessionCwdError("");
      if (!cwd) { status.textContent = ""; controller.setNewSessionCwdError("Working directory is required."); return; }
      const parsed = controller.syncNewSessionProviderFromModelInput();
      if (parsed.providerError) { status.textContent = parsed.providerError; return; }
      const providerChoice = String(parsed.providerAbsent ? "" : parsed.providerChoice || provider || "").trim(); const model = String(parsed.model || "default").trim() || "default";
      codoxearLaunch.rememberProviderModelChoice(agentBackend, providerChoice, model, { providerAbsent: Boolean(parsed.providerAbsent) });
      const resumeSessionId = (controller.currentResumeSelection() || {}).session_id || null; const createInTmux = !!tmuxToggle.checked; const worktreeBranch = !resumeSessionId && worktreeToggle.checked ? String(worktreeInput.value || "").trim() : null;
      if (worktreeToggle.checked && !worktreeBranch) { status.textContent = "Branch name is required."; return; }
      status.textContent = resumeSessionId ? "Resuming..." : worktreeBranch ? "Creating worktree..." : createInTmux ? "Starting in tmux..." : "Starting...";
      let cwdStartError = false; let startErrorText = ""; startBusy = true; startBtn.disabled = true;
      Promise.resolve(spawnSession(cwd, resumeSessionId, worktreeBranch, String(nameInput.value || "").trim(), providerChoice, model, reasoningEffort, fast, createInTmux, (error) => { if (error && error.obj && error.obj.field === "cwd") { cwdStartError = true; status.textContent = ""; controller.setNewSessionCwdError(error.message); return; } const launchId = error && error.obj && error.obj.launch_id ? String(error.obj.launch_id) : ""; startErrorText = launchId ? `${error.message} (${launchId})` : error && error.message ? error.message : "Start failed."; }, agentBackend)).then((brokerPid) => { if (brokerPid) close(); else if (!cwdStartError) status.textContent = startErrorText || "Start failed."; }).finally(() => { startBusy = false; startBtn.disabled = false; });
    }

    closeBtn.onclick = close; cancelBtn.onclick = close; backdrop.onclick = close; viewer.onclick = (event) => event.stopPropagation();
    resumeBtn.onclick = (event) => { event.preventDefault(); event.stopPropagation(); controller.renderNewSessionResumeMenu(); resumeMenuOpen = !resumeMenuOpen; cwdMenuOpen = false; cwdMenuFocus = -1; modelMenuOpen = false; modelMenuFocus = -1; reasoningMenuOpen = false; applyDialogMenus(); };
    cwdInput.onclick = () => { cwdMenuFocus = -1; controller.renderRecentCwdMenu(); cwdMenuOpen = true; modelMenuOpen = false; modelMenuFocus = -1; resumeMenuOpen = false; applyDialogMenus(); };
    cwdInput.oninput = () => { cwdMenuFocus = -1; controller.setNewSessionCwdError(""); controller.syncNewSessionNamePlaceholder(); controller.renderRecentCwdMenu(); cwdMenuOpen = true; modelMenuOpen = false; modelMenuFocus = -1; controller.scheduleNewSessionResumeLoad(); applyDialogMenus(); };
    cwdInput.onblur = () => win.requestAnimationFrame(() => { if (!cwdField.contains(doc.activeElement)) { cwdMenuOpen = false; cwdMenuFocus = -1; applyDialogMenus(); } });
    cwdInput.onkeydown = (event) => { const items = controller.renderRecentCwdMenu(); if (event.key === "ArrowDown" || event.key === "ArrowUp") { if (!items.length) return; event.preventDefault(); cwdMenuOpen = true; modelMenuOpen = false; modelMenuFocus = -1; resumeMenuOpen = false; const delta = event.key === "ArrowDown" ? 1 : -1; cwdMenuFocus = cwdMenuFocus < 0 ? (delta > 0 ? 0 : items.length - 1) : (cwdMenuFocus + delta + items.length) % items.length; controller.renderRecentCwdMenu(); applyDialogMenus(); const active = doc.getElementById(`newSessionCwdOption-${cwdMenuFocus}`); if (active && typeof active.scrollIntoView === "function") active.scrollIntoView({ block: "nearest" }); return; } if (event.key === "Enter" && cwdMenuOpen && cwdMenuFocus >= 0 && items[cwdMenuFocus]) { event.preventDefault(); controller.applyNewSessionCwdSuggestion(items[cwdMenuFocus].cwd); return; } if (event.key === "Escape" && cwdMenuOpen) { event.preventDefault(); event.stopPropagation(); cwdMenuOpen = false; cwdMenuFocus = -1; applyDialogMenus(); } else if (event.key === "Tab" && cwdMenuOpen) { cwdMenuOpen = false; cwdMenuFocus = -1; applyDialogMenus(); } };
    modelInput.onclick = () => { modelMenuFocus = -1; renderModelMenu(); modelMenuOpen = true; cwdMenuOpen = false; cwdMenuFocus = -1; reasoningMenuOpen = false; resumeMenuOpen = false; applyDialogMenus(); };
    modelInput.oninput = () => { literalModelInputValue = ""; launchPresetProviderAbsent = false; modelMenuFocus = -1; controller.syncNewSessionProviderFromModelInput(); renderModelMenu(); controller.setNewSessionReasoningEffort(reasoningEffort); renderReasoningMenu(); modelMenuOpen = true; reasoningMenuOpen = false; applyDialogMenus(); };
    modelInput.onblur = () => win.requestAnimationFrame(() => { if (!modelField.contains(doc.activeElement)) { modelMenuOpen = false; modelMenuFocus = -1; applyDialogMenus(); } });
    modelInput.onkeydown = (event) => { const items = renderModelMenu(); if (event.key === "ArrowDown" || event.key === "ArrowUp") { if (!items.length) return; event.preventDefault(); modelMenuOpen = true; cwdMenuOpen = false; cwdMenuFocus = -1; reasoningMenuOpen = false; resumeMenuOpen = false; const delta = event.key === "ArrowDown" ? 1 : -1; modelMenuFocus = modelMenuFocus < 0 ? (delta > 0 ? 0 : items.length - 1) : (modelMenuFocus + delta + items.length) % items.length; renderModelMenu(); applyDialogMenus(); const active = doc.getElementById(`newSessionModelOption-${modelMenuFocus}`); if (active && typeof active.scrollIntoView === "function") active.scrollIntoView({ block: "nearest" }); return; } if (event.key === "Enter" && modelMenuOpen && modelMenuFocus >= 0 && items[modelMenuFocus]) { event.preventDefault(); controller.selectNewSessionModel(items[modelMenuFocus]); return; } if (event.key === "Escape" && modelMenuOpen) { event.preventDefault(); event.stopPropagation(); modelMenuOpen = false; modelMenuFocus = -1; applyDialogMenus(); } else if (event.key === "Tab" && modelMenuOpen) { modelMenuOpen = false; modelMenuFocus = -1; applyDialogMenus(); } };
    worktreeToggle.onchange = () => { controller.syncNewSessionWorktreeUi(); if (worktreeToggle.checked) worktreeInput.focus(); }; worktreeInput.oninput = () => controller.syncNewSessionWorktreeUi(); fastToggle.onchange = () => { fast = !!fastToggle.checked; };
    reasoningBtn.onclick = (event) => { event.preventDefault(); event.stopPropagation(); renderReasoningMenu(); reasoningMenuOpen = !reasoningMenuOpen; cwdMenuOpen = false; cwdMenuFocus = -1; modelMenuOpen = false; modelMenuFocus = -1; resumeMenuOpen = false; applyDialogMenus(); };
    startBtn.onclick = start;
    addEvent(doc, "mousedown", (event) => { if (!(cwdMenuOpen || modelMenuOpen || reasoningMenuOpen || resumeMenuOpen)) return; const target = event.target; const anchors = [modelField, reasoningBtn, resumeBtn, cwdInput]; if (anchors.some((node) => node.contains(target)) || [modelMenu, reasoningMenu, resumeMenu, cwdMenu].some((node) => node.contains(target))) return; event.preventDefault(); event.stopPropagation(); closeMenus(); applyDialogMenus(); }, true);
    addEvent(doc, "keydown", (event) => { if (event.key !== "Escape" || !isModalTargetOpen(viewer)) return; event.preventDefault(); event.stopPropagation(); close(); });

    return Object.freeze({ viewer, open, close, isOpen: () => isModalTargetOpen(viewer), closeMenus, applyMenus, refreshDefaults: () => { if (!isModalTargetOpen(viewer)) return; const text = String(status.textContent || "").trim(); controller.syncNewSessionTmuxUi(); renderModelMenu(); renderReasoningMenu(); syncRunConfigUi(); if (!text || text.startsWith("Launch defaults degraded for ")) status.textContent = controller.newSessionDefaultsWarningText(); } });
  }

  window.CodoxearNewSession = Object.freeze({ createNewSessionController, createNewSessionDialogController });
})();
