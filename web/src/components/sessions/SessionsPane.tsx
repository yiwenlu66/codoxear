import { useMemo, useState } from "preact/hooks";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

import { useSessionsStore, useSessionsStoreApi } from "../../app/providers";
import { api } from "../../lib/api";
import { normalizeLaunchBackend, providerChoiceToSettings } from "../../lib/launch";
import { getSessionRuntimeId } from "../../lib/session-identity";
import { getSessionDisplayName } from "../../lib/session-display";
import type { SessionSummary } from "../../lib/types";
import { EditSessionDialog } from "./EditSessionDialog";
import { SessionCard } from "./SessionCard";

interface SessionsPaneProps {
  onNewSession?: () => void;
}

type SessionsSurfaceTab = "sessions" | "focus";

function shortSessionId(sessionId: string) {
  const match = sessionId.match(/^([0-9a-f]{8})[0-9a-f-]{20,}$/i);
  return match ? match[1] : sessionId.slice(0, 8);
}

function deleteSessionConfirmText(session: SessionSummary) {
  const name = getSessionDisplayName({ ...session, session_id: "" }, "");
  const sid = shortSessionId(session.session_id);
  const target = name ? ` \"${name}\" (${sid})` : ` ${sid}`;
  if (session.owned) {
    return `Delete this web-owned session${target}? This will stop it and remove it from Codoxear.`;
  }
  return `Delete this terminal-owned session${target}? This will also stop the corresponding terminal session.`;
}

export function SessionsPane({ onNewSession }: SessionsPaneProps) {
  const { items, activeSessionId, remainingCount = 0 } = useSessionsStore();
  const sessionsStoreApi = useSessionsStoreApi();
  const [editingSession, setEditingSession] = useState<SessionSummary | null>(null);
  const [actionError, setActionError] = useState("");
  const [surfaceTab, setSurfaceTab] = useState<SessionsSurfaceTab>("sessions");
  const [loadingMore, setLoadingMore] = useState(false);

  const focusedItems = useMemo(
    () => items.filter((session) => session.focused === true),
    [items],
  );
  const visibleSessions = surfaceTab === "focus" ? focusedItems : items;
  const canLoadMoreSessions = surfaceTab === "sessions" && remainingCount > 0;

  const switchSurfaceTab = (nextTab: SessionsSurfaceTab) => {
    setSurfaceTab(nextTab);
    if (nextTab === "focus") {
      void sessionsStoreApi.refresh().catch(() => undefined);
    }
  };

  const loadMoreSessions = async () => {
    if (remainingCount <= 0) {
      return;
    }

    setLoadingMore(true);
    setActionError("");
    try {
      await sessionsStoreApi.loadMore();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to load more sessions");
    } finally {
      setLoadingMore(false);
    }
  };

  const toggleSessionFocus = async (session: SessionSummary) => {
    try {
      setActionError("");
      const runtimeId = getSessionRuntimeId(session);
      await api.setSessionFocus(session.session_id, session.focused !== true, runtimeId);
      await sessionsStoreApi.refresh();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to update Focus");
    }
  };

  const deleteSession = async (session: SessionSummary) => {
    const confirmed = typeof window === "undefined" || typeof window.confirm !== "function"
      ? true
      : window.confirm(deleteSessionConfirmText(session));
    if (!confirmed) {
      return;
    }
    try {
      setActionError("");
      await api.deleteSession(session.session_id);
      await sessionsStoreApi.refresh();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to delete session");
    }
  };

  const selectCreatedSession = async (response: { session_id?: string; broker_pid?: number }) => {
    await sessionsStoreApi.refresh();
    const returnedSessionId = String(response.session_id || "").trim();
    let createdSession = returnedSessionId
      ? sessionsStoreApi.getState().items.find((item) => item.session_id === returnedSessionId)
      : undefined;
    if (!createdSession) {
      createdSession = sessionsStoreApi.getState().items.find((item) => item.broker_pid === response.broker_pid);
    }
    if (!createdSession) {
      await sessionsStoreApi.refresh({ preferNewest: true });
      const state = sessionsStoreApi.getState();
      createdSession = (returnedSessionId
        ? state.items.find((item) => item.session_id === returnedSessionId)
        : undefined)
        ?? state.items.find((item) => item.broker_pid === response.broker_pid)
        ?? state.items.find((item) => item.session_id === state.activeSessionId)
        ?? state.items[0];
    }
    if (createdSession) {
      sessionsStoreApi.select(createdSession.session_id);
    }
  };

  const resumeHistoricalSession = async (session: SessionSummary) => {
    const cwd = String(session.cwd || "").trim();
    if (!cwd) {
      setActionError("This historical session is missing resume metadata.");
      return;
    }

    setActionError("");

    try {
      const details = await api.getSessionDetails(session.session_id);
      const source = details.session;
      const resumeSessionId = String(source.resume_session_id || "").trim();
      if (!resumeSessionId) {
        setActionError("This historical session is missing resume metadata.");
        return;
      }
      const backend = normalizeLaunchBackend(source.agent_backend);
      const response = await api.createSession({
        cwd,
        backend,
        resume_session_id: resumeSessionId,
      });
      await selectCreatedSession(response);
      await sessionsStoreApi.refreshBootstrap();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to resume session");
    }
  };

  const duplicateSession = async (session: SessionSummary) => {
    const cwd = String(session.cwd || "").trim();
    if (!cwd) {
      setActionError("This session does not have a working directory to duplicate.");
      return;
    }

    setActionError("");

    try {
      const details = await api.getSessionDetails(session.session_id);
      const source = details.session;
      const backend = normalizeLaunchBackend(source.agent_backend);
      const providerSettings = providerChoiceToSettings(String(source.provider_choice || ""), backend);
      const response = await api.createSession({
        cwd,
        backend,
        model: String(source.model || "").trim() || undefined,
        model_provider: providerSettings.model_provider,
        preferred_auth_method: providerSettings.preferred_auth_method,
        reasoning_effort: String(source.reasoning_effort || "").trim() || undefined,
        service_tier: String(source.service_tier || "").trim().toLowerCase() === "fast" ? "fast" : undefined,
        create_in_tmux: backend === "codex" && String(source.transport || "").trim().toLowerCase() === "tmux" ? true : undefined,
      });
      await selectCreatedSession(response);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to duplicate session");
    }
  };

  return (
    <>
      <aside className="sessionsPane" data-testid="sessions-surface">
        <div className="sessionsSurfaceHeader">
          <div>
            <p className="sessionsEyebrow">Continue where you left off</p>
            <h2 className="sessionsSurfaceTitle">Sessions</h2>
          </div>
          <Button type="button" size="sm" className="sessionsNewButton" onClick={() => onNewSession?.()}>
            New session
          </Button>
        </div>
        <div className="sessionsSurfaceTabs" role="tablist" aria-label="Session views">
          <Button
            type="button"
            size="sm"
            variant={surfaceTab === "sessions" ? "default" : "outline"}
            className="sessionsSurfaceTab"
            role="tab"
            aria-selected={surfaceTab === "sessions"}
            onClick={() => switchSurfaceTab("sessions")}
          >
            Sessions
          </Button>
          <Button
            type="button"
            size="sm"
            variant={surfaceTab === "focus" ? "default" : "outline"}
            className="sessionsSurfaceTab"
            role="tab"
            aria-selected={surfaceTab === "focus"}
            onClick={() => switchSurfaceTab("focus")}
          >
            Focus
          </Button>
        </div>
        {actionError ? <p className="px-1 pb-2 text-sm font-medium text-red-600">{actionError}</p> : null}
        <ScrollArea className="sessionsSurfaceBody">
          <div className="sessionsList">
            {visibleSessions.map((session) => (
              <SessionCard
                key={session.session_id}
                session={session}
                active={session.session_id === activeSessionId}
                onSelect={() => {
                  if (session.historical && normalizeLaunchBackend(session.agent_backend) !== "pi") {
                    void resumeHistoricalSession(session);
                    return;
                  }
                  sessionsStoreApi.select(session.session_id);
                }}
                onToggleFocus={() => { void toggleSessionFocus(session); }}
                onDuplicate={session.historical ? undefined : () => { void duplicateSession(session); }}
                onDelete={() => { void deleteSession(session); }}
                onEdit={session.historical ? undefined : () => {
                  setActionError("");
                  void api.getSessionDetails(session.session_id)
                    .then((details) => setEditingSession(details.session))
                    .catch((error) => {
                      setActionError(error instanceof Error ? error.message : "Failed to load session details");
                    });
                }}
              />
            ))}
            {surfaceTab === "focus" && focusedItems.length === 0 ? (
              <div className="focusRailEmpty">No sessions are in Focus.</div>
            ) : null}
            {canLoadMoreSessions ? (
              <button
                type="button"
                className="sessionGroupMoreButton"
                aria-label={`Load ${remainingCount} more sessions`}
                disabled={loadingMore}
                onClick={() => {
                  void loadMoreSessions();
                }}
              >
                {loadingMore ? "Loading..." : `Load ${remainingCount} more sessions`}
              </button>
            ) : null}
          </div>
        </ScrollArea>
      </aside>

      <EditSessionDialog
        key={editingSession?.session_id || "session-edit-dialog"}
        open={editingSession != null}
        session={editingSession}
        sessions={items}
        onClose={() => setEditingSession(null)}
        onSaved={async () => {
          await sessionsStoreApi.refresh();
        }}
      />
    </>
  );
}
