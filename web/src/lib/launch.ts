export function normalizeLaunchBackend(backend: string | null | undefined) {
  return String(backend || "codex").trim().toLowerCase() === "pi" ? "pi" : "codex";
}

export function providerChoiceToSettings(choice: string, backend: string) {
  const value = String(choice || "").trim();
  const normalizedBackend = normalizeLaunchBackend(backend);
  if (!value) {
    return { model_provider: undefined, preferred_auth_method: undefined };
  }
  if (normalizedBackend === "pi") {
    return { model_provider: value, preferred_auth_method: undefined };
  }
  if (value === "chatgpt") {
    return { model_provider: "openai", preferred_auth_method: "chatgpt" };
  }
  if (value === "openai-api") {
    return { model_provider: "openai", preferred_auth_method: "apikey" };
  }
  return { model_provider: value, preferred_auth_method: "apikey" };
}
