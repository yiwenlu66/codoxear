import { useMemo, useState } from "preact/hooks";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

import { useSessionsStore, useSessionsStoreApi } from "../../app/providers";
import { api } from "../../lib/api";
import { normalizeLaunchBackend, providerChoiceToSettings } from "../../lib/launch";
import { getSessionRuntimeId } from "../../lib/session-identity";
import { getSessionDisplayName } from "../../lib/session-display";
import type { CwdGroupMeta, SessionSummary } from "../../lib/types";
import { EditSessionDialog } from "./EditSessionDialog";
import { SessionCard } from "./SessionCard";
import { SessionGroup } from "./SessionGroup";

interface SessionsPaneProps {
  onNewSession?: () => void;
}

const FALLBACK_GROUP_KEY = "__no_working_directory__";
const FALLBACK_GROUP_TITLE = "No working directory";
const FALLBACK_GROUP_SUBTITLE = "Sessions without a cwd";

interface GroupedSessions {
  key: string;
  cwd: string | null;
  title: string;
  subtitle: string;
  collapsed: boolean;
  sessions: SessionSummary[];
}

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

function getGroupTitle(cwd: string | null) {
  if (!cwd) {
    return FALLBACK_GROUP_TITLE;
  }
  const parts = cwd.split(/[\\/]+/).filter(Boolean);
  return parts[parts.length - 1] || cwd;
}

function groupSessions(items: SessionSummary[], cwdGroups: Record<string, CwdGroupMeta>) {
  const groups = new Map<string, GroupedSessions>();

  items.forEach((session) => {
    const cwd = session.cwd?.trim() || null;
    const key = cwd || FALLBACK_GROUP_KEY;
    const meta = cwd ? cwdGroups[cwd] : undefined;
    const existing = groups.get(key);

    if (existing) {
      existing.sessions.push(session);
      return;
    }

    groups.set(key, {
      key,
      cwd,
      title: meta?.label?.trim() || getGroupTitle(cwd),
      subtitle: cwd || FALLBACK_GROUP_SUBTITLE,
      collapsed: Boolean(meta?.collapsed),
      sessions: [session],
    });
  });

  return Array.from(groups.values());
}

export function SessionsPane({ onNewSession }: SessionsPaneProps) {
  const { items, activeSessionId, cwdGroups = {}, remainingByGroup = {}, omittedGroupCount = 0 } = useSessionsStore();
  const sessionsStoreApi = useSessionsStoreApi();
  const [editingSession, setEditingSession] = useState<SessionSummary | null>(null);
  const [actionError, setActionError] = useState("");
  const [pendingGroupKey, setPendingGroupKey] = useState<string | null>(null);
  const [groupErrors, setGroupErrors] = useState<Record<string, string>>({});

  const groupedSessions = useMemo(() => groupSessions(items, cwdGroups), [cwdGroups, items]);

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
      const runtimeId = getSessionRuntimeId(session);
      await (runtimeId ? api.deleteSession(session.session_id, runtimeId) : api.deleteSession(session.session_id));
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

  async function saveGroupChange(group: GroupedSessions, payload: { label?: string; collapsed?: boolean }) {
    if (!group.cwd) {
      return false;
    }

    setPendingGroupKey(group.key);
    setGroupErrors((current) => ({ ...current, [group.key]: "" }));

    try {
      await api.editCwdGroup({ cwd: group.cwd, ...payload });
      await sessionsStoreApi.refreshBootstrap();
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to save group changes.";
      setGroupErrors((current) => ({ ...current, [group.key]: message }));
      return false;
    } finally {
      setPendingGroupKey((current) => (current === group.key ? null : current));
    }
  }

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
        {actionError ? <p className="px-1 pb-2 text-sm font-medium text-red-600">{actionError}</p> : null}
        <ScrollArea className="sessionsSurfaceBody">
          <div className="sessionsList">
            {groupedSessions.map((group) => {
              const visibleSessions = group.sessions;
              const hiddenSessionCount = Math.max(0, Number(remainingByGroup[group.key] || 0));
              const hasHiddenSessions = hiddenSessionCount > 0;

              return (
                <SessionGroup
                  key={group.key}
                  title={group.title}
                  subtitle={group.subtitle}
                  collapsed={group.collapsed}
                  canRename={Boolean(group.cwd)}
                  isSaving={pendingGroupKey === group.key}
                  errorMessage={groupErrors[group.key]}
                  onRename={
                    group.cwd
                      ? async (label) => saveGroupChange(group, { label: label.trim() })
                      : undefined
                  }
                  onToggle={
                    group.cwd
                      ? () => {
                          void saveGroupChange(group, { collapsed: !group.collapsed });
                        }
                      : undefined
                  }
                >
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
                      onToggleFocus={session.historical ? undefined : () => { void toggleSessionFocus(session); }}
                      onDuplicate={session.historical ? undefined : () => { void duplicateSession(session); }}
                      onDelete={session.historical ? undefined : () => { void deleteSession(session); }}
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
                  {hasHiddenSessions ? (
                    <button
                      type="button"
                      className="sessionGroupMoreButton"
                      aria-label={`Load ${hiddenSessionCount} more sessions in ${group.title}`}
                      onClick={() => {
                        void sessionsStoreApi.loadMoreGroup(group.key);
                      }}
                    >
                      ...
                    </button>
                  ) : null}
                </SessionGroup>
              );
            })}
            {omittedGroupCount > 0 ? (
              <button
                type="button"
                className="sessionGroupMoreButton"
                aria-label={`Load ${omittedGroupCount} more directories`}
                onClick={() => {
                  void sessionsStoreApi.loadMoreGroups();
                }}
              >
                Load {omittedGroupCount} more directories
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
