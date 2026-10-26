import * as CodoxearStorage from "./app_storage.js";
import * as CodoxearUrls from "./app_application.js";



function resolveAppUrl(path) {
    return CodoxearUrls.resolveAppUrl(path);
  }
  function storageGetItem(key) {
    return CodoxearStorage.getItem(key);
  }
  function storageSetItem(key, value) {
    return CodoxearStorage.setItem(key, value);
  }
  function storageRemoveItem(key) {
    return CodoxearStorage.removeItem(key);
  }

  const LAST_BACKEND_KEY = "codoxear.newSessionBackend";
  const LAST_LAUNCH_KEY = "codoxear.lastLaunch";

  // The dialog's successful launch is the only writer. Opening a dialog or
  // browsing backend tabs must never change the quick-start contract.
  function rememberLastLaunch(record) {
    storageSetItem(LAST_LAUNCH_KEY, JSON.stringify(record));
  }

  function loadLastLaunch(defaultsSource = {}, tmuxAvailable = false) {
    let record;
    try { record = JSON.parse(storageGetItem(LAST_LAUNCH_KEY)); } catch (_) {}
    if (record && ["codex", "pi", "cc"].includes(record.backend)
        && typeof record.model === "string" && typeof record.providerChoice === "string"
        && typeof record.reasoningEffort === "string"
        && typeof record.fast === "boolean" && typeof record.tmux === "boolean"
        && typeof record.providerAbsent === "boolean") {
      return { backend: record.backend, providerChoice: record.providerAbsent ? "" : record.providerChoice,
        model: record.model || "default", providerAbsent: record.providerAbsent,
        reasoningEffort: record.reasoningEffort, fast: record.fast, tmux: record.tmux };
    }
    const backend = loadRememberedBackendChoice() || normalizeAgentBackendName(defaultsSource.default_backend || "pi");
    const defaults = defaultsForAgentBackend(backend, defaultsSource);
    const providers = providerChoicesForBackend(backend, defaultsSource);
    const rememberedProvider = loadRememberedProviderChoice(backend);
    let providerChoice = (providers.includes(rememberedProvider) ? rememberedProvider : "") || String(defaults.provider_choice || "") || providers[0] || "";
    let model = String(defaults.model || "default");
    let providerAbsent = false;
    const remembered = loadRememberedProviderModelChoice(backend);
    const absent = rememberedProviderModelAbsentChoice(remembered);
    if (absent) { model = absent.model; providerChoice = ""; providerAbsent = true; }
    else if (remembered) {
      const slash = remembered.indexOf("/");
      if (slash > 0 && (providers.length || backend === "pi")) {
        const candidate = remembered.slice(0, slash);
        if (providers.includes(candidate) || backend === "pi") {
          providerChoice = candidate; model = remembered.slice(slash + 1) || model;
        }
      } else model = remembered;
    }
    const choices = reasoningChoicesForBackend(backend, defaultsSource, { provider: providerChoice, model });
    const effort = String(defaults.reasoning_effort || "").toLowerCase();
    return { backend, providerChoice, model, providerAbsent,
      reasoningEffort: choices.includes(effort) ? effort : choices[0] || "high",
      fast: !!defaults.supports_fast && defaults.service_tier === "fast", tmux: !!tmuxAvailable };
  }
  const NO_PROVIDER_MODEL_PREFIX = "__codoxear_no_provider__:";

  function lastProviderKey(backend) {
    return `codoxear.newSessionProvider.${normalizeAgentBackendName(backend)}`;
  }

  function lastProviderModelKey(backend) {
    return `codoxear.newSessionProviderModel.${normalizeAgentBackendName(backend)}`;
  }

  function loadRememberedBackendChoice() {
    const value = String(storageGetItem(LAST_BACKEND_KEY) || "").trim();
    return value ? normalizeAgentBackendName(value) : "";
  }

  function rememberBackendChoice(backend) {
    storageSetItem(LAST_BACKEND_KEY, normalizeAgentBackendName(backend));
  }

  function loadRememberedProviderChoice(backend) {
    return String(storageGetItem(lastProviderKey(backend)) || "").trim();
  }

  function rememberProviderChoice(backend, provider) {
    const value = String(provider || "").trim();
    if (value) storageSetItem(lastProviderKey(backend), value);
    else storageRemoveItem(lastProviderKey(backend));
  }

  function loadRememberedProviderModelChoice(backend) {
    return String(storageGetItem(lastProviderModelKey(backend)) || "").trim();
  }

  function rememberedProviderModelAbsentChoice(value) {
    const raw = String(value || "").trim();
    if (!raw.startsWith(NO_PROVIDER_MODEL_PREFIX)) return null;
    const model = raw.slice(NO_PROVIDER_MODEL_PREFIX.length).trim() || "default";
    return { providerChoice: "", model, providerError: "", providerAbsent: true };
  }

  function rememberProviderModelChoice(backend, provider, model, { providerAbsent = false } = {}) {
    const providerValue = String(provider || "").trim();
    const modelValue = String(model || "").trim() || "default";
    const value = providerAbsent ? `${NO_PROVIDER_MODEL_PREFIX}${modelValue}` : providerValue ? `${providerValue}/${modelValue}` : modelValue;
    if (value) storageSetItem(lastProviderModelKey(backend), value);
    else storageRemoveItem(lastProviderModelKey(backend));
  }

  function normalizeAgentBackendName(value) {
    const raw = String(value || "").trim().toLowerCase();
    if (raw === "pi") return "pi";
    if (raw === "cc" || raw === "claude" || raw === "claude-code") return "cc";
    return "codex";
  }

  function agentBackendDisplayName(value) {
    const backend = normalizeAgentBackendName(value);
    if (backend === "pi") return "Pi";
    if (backend === "cc") return "Claude";
    return "Codex";
  }

  function versionedStaticAssetPath(path) {
    const version = String(window.CODOXEAR_ASSET_VERSION || "").trim();
    if (!version) return path;
    const separator = String(path).includes("?") ? "&" : "?";
    return `${path}${separator}v=${encodeURIComponent(version)}`;
  }

  function agentBackendLogoPath(value) {
    const backend = normalizeAgentBackendName(value);
    return resolveAppUrl(versionedStaticAssetPath(`/static/logos/${backend}.svg`));
  }

  function sessionAgentBackend(s) {
    if (!s || typeof s !== "object") return "codex";
    return normalizeAgentBackendName(s.agent_backend);
  }

  function legacyCodexLaunchDefaults(seed = {}) {
    const raw = seed && typeof seed === "object" ? seed : {};
    const modelProviders = Array.isArray(raw.model_providers) ? raw.model_providers.slice() : ["chatgpt", "openai-api"];
    if (!modelProviders.includes("chatgpt")) modelProviders.unshift("chatgpt");
    if (!modelProviders.includes("openai-api")) modelProviders.splice(Math.min(1, modelProviders.length), 0, "openai-api");
    return {
      agent_backend: "codex",
      model_provider: typeof raw.model_provider === "string" ? raw.model_provider : "openai",
      preferred_auth_method: typeof raw.preferred_auth_method === "string" ? raw.preferred_auth_method : "chatgpt",
      provider_choice: typeof raw.provider_choice === "string" ? raw.provider_choice : "chatgpt",
      provider_choices: modelProviders,
      model: typeof raw.model === "string" ? raw.model : null,
      models: Array.isArray(raw.models) ? raw.models.slice() : [],
      model_providers: modelProviders,
      provider_models: raw.provider_models && typeof raw.provider_models === "object" ? raw.provider_models : null,
      reasoning_effort: typeof raw.reasoning_effort === "string" ? raw.reasoning_effort : "high",
      reasoning_efforts: Array.isArray(raw.reasoning_efforts) ? raw.reasoning_efforts.slice() : ["xhigh", "high", "medium", "low"],
      service_tier: typeof raw.service_tier === "string" ? raw.service_tier : "flex",
      supports_fast: raw.supports_fast !== false,
    };
  }

  function emptyPiLaunchDefaults(seed = {}) {
    const raw = seed && typeof seed === "object" ? seed : {};
    const providerChoices = Array.isArray(raw.provider_choices) ? raw.provider_choices.slice() : [];
    const modelChoices = Array.isArray(raw.models) ? raw.models.slice() : [];
    return {
      agent_backend: "pi",
      model_provider: typeof raw.model_provider === "string" ? raw.model_provider : null,
      preferred_auth_method: null,
      provider_choice: typeof raw.provider_choice === "string" ? raw.provider_choice : null,
      provider_choices: providerChoices,
      model: typeof raw.model === "string" ? raw.model : null,
      models: modelChoices,
      provider_models: raw.provider_models && typeof raw.provider_models === "object" ? raw.provider_models : null,
      reasoning_effort: typeof raw.reasoning_effort === "string" ? raw.reasoning_effort : "high",
      reasoning_efforts: Array.isArray(raw.reasoning_efforts) ? raw.reasoning_efforts.slice() : ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
      reasoning_efforts_by_model: raw.reasoning_efforts_by_model && typeof raw.reasoning_efforts_by_model === "object" ? raw.reasoning_efforts_by_model : {},
      service_tier: null,
      supports_fast: false,
    };
  }

  function emptyCcLaunchDefaults(seed = {}) {
    const raw = seed && typeof seed === "object" ? seed : {};
    const modelChoices = Array.isArray(raw.models) ? raw.models.slice() : [];
    return {
      agent_backend: "cc",
      model_provider: null,
      preferred_auth_method: null,
      provider_choice: null,
      provider_choices: [],
      model: typeof raw.model === "string" ? raw.model : null,
      models: modelChoices,
      reasoning_effort: typeof raw.reasoning_effort === "string" ? raw.reasoning_effort : "medium",
      reasoning_efforts: Array.isArray(raw.reasoning_efforts) ? raw.reasoning_efforts.slice() : ["low", "medium", "high", "xhigh", "max", "auto"],
      reasoning_efforts_by_model: raw.reasoning_efforts_by_model && typeof raw.reasoning_efforts_by_model === "object" ? raw.reasoning_efforts_by_model : {},
      service_tier: null,
      supports_fast: false,
    };
  }

  function defaultsForAgentBackend(backend, defaultsSource = null) {
    const normalized = normalizeAgentBackendName(backend);
    const raw = defaultsSource && typeof defaultsSource === "object" ? defaultsSource : {};
    if (raw.backends && typeof raw.backends === "object") {
      const item = raw.backends[normalized];
      if (item && typeof item === "object") {
        if (normalized === "pi") return emptyPiLaunchDefaults(item);
        if (normalized === "cc") return emptyCcLaunchDefaults(item);
        return legacyCodexLaunchDefaults(item);
      }
    }
    if (normalized === "pi") return emptyPiLaunchDefaults();
    if (normalized === "cc") return emptyCcLaunchDefaults();
    return legacyCodexLaunchDefaults(raw);
  }

  function providerChoicesForBackend(backend, defaultsSource = null) {
    const defaults = defaultsForAgentBackend(backend, defaultsSource);
    const out = [];
    for (const value of Array.isArray(defaults.provider_choices) ? defaults.provider_choices : []) {
      if (typeof value !== "string") continue;
      const trimmed = value.trim();
      if (!trimmed || out.includes(trimmed)) continue;
      out.push(trimmed);
    }
    return out;
  }

  function reasoningChoicesForBackend(backend, defaultsSource = null, { provider = null, model = null } = {}) {
    const defaults = defaultsForAgentBackend(backend, defaultsSource);
    let rawChoices = Array.isArray(defaults.reasoning_efforts) ? defaults.reasoning_efforts : [];
    const map = defaults.reasoning_efforts_by_model && typeof defaults.reasoning_efforts_by_model === "object" ? defaults.reasoning_efforts_by_model : null;
    const modelName = typeof model === "string" ? model.trim() : "";
    const providerName = typeof provider === "string" ? provider.trim() : "";
    if (map && modelName) {
      const providerKey = providerName ? `${providerName}/${modelName}` : "";
      if (providerKey && Array.isArray(map[providerKey])) rawChoices = map[providerKey];
      else if (!providerName && Array.isArray(map[modelName])) rawChoices = map[modelName];
    }
    const out = [];
    for (const value of rawChoices) {
      if (typeof value !== "string") continue;
      const trimmed = value.trim().toLowerCase();
      if (!trimmed || out.includes(trimmed)) continue;
      out.push(trimmed);
    }
    return out;
  }

  function backendSupportsFast(backend, defaultsSource = null) {
    return !!defaultsForAgentBackend(backend, defaultsSource).supports_fast;
  }

  function providerChoiceToSettings(choice, agentBackend = "codex") {
    const backend = normalizeAgentBackendName(agentBackend);
    const rawChoice = String(choice || "").trim();
    if (backend === "pi") return { model_provider: rawChoice || null, preferred_auth_method: null };
    if (backend === "cc") return { model_provider: null, preferred_auth_method: null };
    const codexChoice = rawChoice || "chatgpt";
    if (codexChoice === "chatgpt") return { model_provider: "openai", preferred_auth_method: "chatgpt" };
    if (codexChoice === "openai-api") return { model_provider: "openai", preferred_auth_method: "apikey" };
    return { model_provider: codexChoice, preferred_auth_method: "apikey" };
  }

  function sessionProviderChoice(s) {
    if (!s || typeof s !== "object") return "chatgpt";
    const backend = sessionAgentBackend(s);
    const provider = typeof s.model_provider === "string" ? s.model_provider.trim() : "";
    if (backend === "pi") return provider;
    if (backend === "cc") return "";
    const explicit = typeof s.provider_choice === "string" ? s.provider_choice.trim() : "";
    if (explicit) return explicit;
    const auth = typeof s.preferred_auth_method === "string" ? s.preferred_auth_method.trim() : "";
    if (provider === "openai") return auth === "chatgpt" ? "chatgpt" : "openai-api";
    return provider || "chatgpt";
  }

  function modelOptionMatches(option, query) {
    const text = String(option && option.searchText ? option.searchText : option && option.model ? option.model : "").toLowerCase();
    if (!query) return true;
    return text === query || text.startsWith(query) || text.includes(query);
  }

  function providerModelDisplay(model, providerChoice = "", { hasProviderChoices = false, allowCustomProvider = false } = {}) {
    const cleanModel = String(model || "").trim() || "default";
    const cleanProvider = String(providerChoice || "").trim();
    if ((hasProviderChoices || allowCustomProvider) && cleanProvider) return `${cleanProvider}/${cleanModel}`;
    return cleanModel;
  }

  function redactedLaunchErrorText(value) {
    let text = String(value || "").trim();
    if (!text) return "";
    const sensitiveKey = "[A-Z0-9_.-]*(?:TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_.-]*";
    const secretValue = "(?:(?:Bearer|Basic)\\s+[A-Za-z0-9._~+/=-]+|\\\"[^\\\"]*(?:\\\"|$)|'[^']*(?:'|$)|[^\\s\\\"',;}\\[\\]]+)";
    text = text.replace(new RegExp(`\\b(${sensitiveKey})\\s*=\\s*${secretValue}`, "gi"), "$1=[redacted]");
    text = text.replace(new RegExp(`(^|[^A-Z0-9_.-])([\\\"']?${sensitiveKey}[\\\"']?\\s*:\\s*)${secretValue}`, "gi"), "$1$2[redacted]");
    text = text.replace(new RegExp(`(^|[^A-Z0-9_.-])([\\\"']?${sensitiveKey}[\\\"']?\\s*[:=]\\s*)\\[redacted\\]\\s+[A-Za-z0-9._~+/=-]{12,}(?=$|[\\s,;}\\]])`, "gi"), "$1$2[redacted]");
    text = text.replace(/\b(Bearer|Basic)\s+[A-Za-z0-9._~+\/-]+=*/gi, "$1 [redacted]");
    text = text.replace(/\b(sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b/g, "[redacted-token]");
    return text;
  }

export { loadLastLaunch, rememberLastLaunch, lastProviderKey, lastProviderModelKey, loadRememberedBackendChoice, rememberBackendChoice, loadRememberedProviderChoice, rememberProviderChoice, loadRememberedProviderModelChoice, rememberedProviderModelAbsentChoice, rememberProviderModelChoice, normalizeAgentBackendName, agentBackendDisplayName, agentBackendLogoPath, sessionAgentBackend, legacyCodexLaunchDefaults, emptyPiLaunchDefaults, emptyCcLaunchDefaults, defaultsForAgentBackend, providerChoicesForBackend, reasoningChoicesForBackend, backendSupportsFast, providerChoiceToSettings, sessionProviderChoice, modelOptionMatches, providerModelDisplay, redactedLaunchErrorText };
