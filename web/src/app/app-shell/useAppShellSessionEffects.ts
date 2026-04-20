import { useEffect, useState } from "preact/hooks";
import type { LiveSessionStore } from "../../domains/live-session/store";
import type { SessionUiStore } from "../../domains/session-ui/store";
import type { SessionsStore } from "../../domains/sessions/store";
import { getSessionRuntimeId } from "../../lib/session-identity";
import type { SessionSummary } from "../../lib/types";

interface UseAppShellSessionEffectsOptions {
  activeSessionBackend?: string;
  activeSessionHistorical?: boolean;
  activeSessionPending?: boolean;
  activeSessionId: string | null;
  activeSessionRuntimeId?: string | null;
  activeSessionLiveBusy: boolean;
  items: SessionSummary[];
  liveSessionStoreApi: LiveSessionStore;
  realtimeConnected?: boolean;
  replySoundEnabled: boolean;
  sessionUiStoreApi: SessionUiStore;
  sessionsStoreApi: SessionsStore;
  workspaceOpen: boolean;
  activeSessionReplySoundPrimingRef: { current: string | null };
  backgroundReplySoundPrimedSessionIdsRef: { current: Set<string> };
  suppressedReplySoundSessionIdsRef: { current: Set<string> };
}

const BUSY_SESSIONS_REFRESH_MS = 5000;
const IDLE_SESSIONS_REFRESH_MS = 15000;
const ACTIVE_BUSY_LIVE_REFRESH_MS = 2000;
const ACTIVE_IDLE_LIVE_REFRESH_MS = 12000;
const BACKGROUND_BUSY_LIVE_REFRESH_MS = 5000;
const WORKSPACE_REFRESH_MS = 15000;
const SSE_SESSIONS_FALLBACK_MS = 60000;
const SSE_ACTIVE_LIVE_FALLBACK_MS = 30000;
const SSE_BACKGROUND_LIVE_FALLBACK_MS = 30000;
const SSE_WORKSPACE_FALLBACK_MS = 60000;

function isDocumentVisible() {
  if (typeof document === "undefined") {
    return true;
  }
  return document.visibilityState !== "hidden";
}

export function useAppShellSessionEffects({
  activeSessionBackend,
  activeSessionHistorical,
  activeSessionPending,
  activeSessionId,
  activeSessionRuntimeId,
  activeSessionLiveBusy,
  items,
  liveSessionStoreApi,
  realtimeConnected = false,
  replySoundEnabled,
  sessionUiStoreApi,
  sessionsStoreApi,
  workspaceOpen,
  activeSessionReplySoundPrimingRef,
  backgroundReplySoundPrimedSessionIdsRef,
  suppressedReplySoundSessionIdsRef,
}: UseAppShellSessionEffectsOptions) {
  const [pageVisible, setPageVisible] = useState(isDocumentVisible);
  const hasBusySession = items.some((session) => Boolean(session.busy || session.pending_startup));
  const sessionsRefreshIntervalMs = realtimeConnected
    ? SSE_SESSIONS_FALLBACK_MS
    : (hasBusySession ? BUSY_SESSIONS_REFRESH_MS : IDLE_SESSIONS_REFRESH_MS);
  const activeSessionBusy = activeSessionLiveBusy
    || items.some((session) => session.session_id === activeSessionId && session.busy);
  const activeLiveRefreshIntervalMs = realtimeConnected
    ? SSE_ACTIVE_LIVE_FALLBACK_MS
    : (activeSessionBusy ? ACTIVE_BUSY_LIVE_REFRESH_MS : ACTIVE_IDLE_LIVE_REFRESH_MS);

  useEffect(() => {
    const handleVisibilityChange = () => {
      setPageVisible(isDocumentVisible());
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, []);

  useEffect(() => {
    if (!pageVisible) {
      return undefined;
    }

    sessionsStoreApi.refresh().catch(() => undefined);
    const intervalId = window.setInterval(() => {
      sessionsStoreApi.refresh().catch(() => undefined);
    }, sessionsRefreshIntervalMs);
    return () => window.clearInterval(intervalId);
  }, [pageVisible, sessionsRefreshIntervalMs, sessionsStoreApi]);

  useEffect(() => {
    if (!pageVisible || !activeSessionId) {
      return undefined;
    }

    if ((activeSessionHistorical && activeSessionBackend === "pi") || activeSessionPending) {
      return undefined;
    }

    const recoverMissingSession = (error: unknown) => {
      if (!error || typeof error !== "object" || (error as { status?: unknown }).status !== 404) {
        return;
      }
      sessionsStoreApi.refresh().catch(() => undefined);
    };

    if (replySoundEnabled) {
      suppressedReplySoundSessionIdsRef.current.add(activeSessionId);
    }
    (activeSessionRuntimeId
      ? liveSessionStoreApi.loadInitial(activeSessionId, activeSessionRuntimeId)
      : liveSessionStoreApi.loadInitial(activeSessionId))
      .catch(recoverMissingSession);
    const intervalId = window.setInterval(() => {
      (activeSessionRuntimeId
        ? liveSessionStoreApi.poll(activeSessionId, activeSessionRuntimeId)
        : liveSessionStoreApi.poll(activeSessionId))
        .catch(recoverMissingSession)
        .finally(() => {
          if (activeSessionReplySoundPrimingRef.current === activeSessionId) {
            suppressedReplySoundSessionIdsRef.current.delete(activeSessionId);
            activeSessionReplySoundPrimingRef.current = null;
          }
        });
    }, activeLiveRefreshIntervalMs);
    return () => window.clearInterval(intervalId);
  }, [activeLiveRefreshIntervalMs, activeSessionBackend, activeSessionHistorical, activeSessionId, activeSessionPending, activeSessionReplySoundPrimingRef, activeSessionRuntimeId, liveSessionStoreApi, pageVisible, replySoundEnabled, sessionUiStoreApi, sessionsStoreApi, suppressedReplySoundSessionIdsRef, workspaceOpen]);

  useEffect(() => {
    if (!pageVisible || !workspaceOpen || !activeSessionId) {
      return undefined;
    }

    if ((activeSessionHistorical && activeSessionBackend === "pi") || activeSessionPending) {
      return undefined;
    }

    const recoverMissingSession = (error: unknown) => {
      if (!error || typeof error !== "object" || (error as { status?: unknown }).status !== 404) {
        return;
      }
      sessionsStoreApi.refresh().catch(() => undefined);
    };

    (activeSessionRuntimeId
      ? sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSessionBackend, runtimeId: activeSessionRuntimeId })
      : sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSessionBackend })).catch(recoverMissingSession);
    const intervalId = window.setInterval(() => {
      (activeSessionRuntimeId
        ? sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSessionBackend, runtimeId: activeSessionRuntimeId })
        : sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSessionBackend })).catch(recoverMissingSession);
    }, realtimeConnected ? SSE_WORKSPACE_FALLBACK_MS : WORKSPACE_REFRESH_MS);
    return () => window.clearInterval(intervalId);
  }, [activeSessionBackend, activeSessionHistorical, activeSessionId, activeSessionPending, activeSessionRuntimeId, pageVisible, realtimeConnected, sessionUiStoreApi, sessionsStoreApi, workspaceOpen]);

  useEffect(() => {
    if (!pageVisible || !replySoundEnabled) {
      return;
    }

    const backgroundBusySessions = items.filter((session) => session.session_id !== activeSessionId && session.busy);
    for (const session of backgroundBusySessions) {
      const sessionId = session.session_id;
      const runtimeId = getSessionRuntimeId(session);
      if (
        backgroundReplySoundPrimedSessionIdsRef.current.has(sessionId)
        || suppressedReplySoundSessionIdsRef.current.has(sessionId)
      ) {
        continue;
      }
      suppressedReplySoundSessionIdsRef.current.add(sessionId);
      (runtimeId
        ? liveSessionStoreApi.loadInitial(sessionId, runtimeId)
        : liveSessionStoreApi.loadInitial(sessionId))
        .catch(() => undefined)
        .finally(() => {
          suppressedReplySoundSessionIdsRef.current.delete(sessionId);
          backgroundReplySoundPrimedSessionIdsRef.current.add(sessionId);
        });
    }
  }, [activeSessionId, backgroundReplySoundPrimedSessionIdsRef, items, liveSessionStoreApi, pageVisible, replySoundEnabled, suppressedReplySoundSessionIdsRef]);

  useEffect(() => {
    if (!pageVisible || !replySoundEnabled) {
      return undefined;
    }

    const pollBackgroundBusySessions = () => {
      const backgroundBusySessions = items
        .filter((session) => session.session_id !== activeSessionId && session.busy)
        .filter((session) => backgroundReplySoundPrimedSessionIdsRef.current.has(session.session_id));
      for (const session of backgroundBusySessions) {
        const runtimeId = getSessionRuntimeId(session);
        (runtimeId
          ? liveSessionStoreApi.poll(session.session_id, runtimeId)
          : liveSessionStoreApi.poll(session.session_id)).catch(() => undefined);
      }
    };

    pollBackgroundBusySessions();
    const intervalId = window.setInterval(
      pollBackgroundBusySessions,
      realtimeConnected ? SSE_BACKGROUND_LIVE_FALLBACK_MS : BACKGROUND_BUSY_LIVE_REFRESH_MS,
    );
    return () => window.clearInterval(intervalId);
  }, [activeSessionId, backgroundReplySoundPrimedSessionIdsRef, items, liveSessionStoreApi, pageVisible, realtimeConnected, replySoundEnabled]);
}
