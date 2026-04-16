export interface RememberedFileSelection {
  line: number | null;
  path: string;
}

const rememberedFileSelectionBySession = new Map<string, RememberedFileSelection>();

export function normalizeRememberedLine(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.floor(value) : null;
}

export function rememberFileSelection(sessionId: string | null | undefined, path: string, line?: number | null) {
  const normalizedSessionId = String(sessionId || "").trim();
  const normalizedPath = String(path || "").trim();
  if (!normalizedSessionId || !normalizedPath) {
    return;
  }
  rememberedFileSelectionBySession.set(normalizedSessionId, {
    path: normalizedPath,
    line: normalizeRememberedLine(line),
  });
}

export function preferredFileSelectionForSession(sessionId: string | null | undefined) {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) {
    return null;
  }
  return rememberedFileSelectionBySession.get(normalizedSessionId) ?? null;
}

export function clearRememberedFileSelections() {
  rememberedFileSelectionBySession.clear();
}
