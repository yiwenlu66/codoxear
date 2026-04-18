import type { SessionSummary } from "./types";

export function getSessionRuntimeId(
  session: Pick<SessionSummary, "runtime_id"> | null | undefined,
): string | null {
  if (!session) {
    return null;
  }
  const runtimeId = String(session.runtime_id || "").trim();
  return runtimeId || null;
}

export function getSessionRouteId(
  sessionId: string,
  runtimeId?: string | null,
) {
  const runtime = String(runtimeId || "").trim();
  const durable = String(sessionId || "").trim();
  return runtime || durable;
}
