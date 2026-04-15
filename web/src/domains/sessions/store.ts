import { api } from "../../lib/api";
import type { CwdGroupMeta, NewSessionDefaults, SessionSummary } from "../../lib/types";

export interface SessionsState {
  items: SessionSummary[];
  activeSessionId: string | null;
  loading: boolean;
  bootstrapLoaded: boolean;
  remainingByGroup: Record<string, number>;
  omittedGroupCount: number;
  newSessionDefaults: NewSessionDefaults | null;
  recentCwds: string[];
  cwdGroups: Record<string, CwdGroupMeta>;
  tmuxAvailable: boolean;
}

export interface RefreshSessionsOptions {
  preferNewest?: boolean;
}

export interface SessionsStore {
  getState(): SessionsState;
  subscribe(listener: () => void): () => void;
  refresh(options?: RefreshSessionsOptions): Promise<void>;
  refreshBootstrap(): Promise<void>;
  loadMoreGroup(groupKey: string, limit?: number): Promise<void>;
  loadMoreGroups(limit?: number): Promise<void>;
  select(sessionId: string): void;
}

const FALLBACK_GROUP_KEY = "__no_working_directory__";
const GROUP_PAGE_SIZE = 5;
const GROUPS_PAGE_SIZE = 3;

function sessionGroupKey(session: SessionSummary) {
  const cwd = String(session.cwd || "").trim();
  return cwd || FALLBACK_GROUP_KEY;
}

function sessionDedupeKey(session: SessionSummary) {
  const threadId = String(session.thread_id || "").trim();
  if (threadId && !session.historical) {
    const backend = String(session.agent_backend || "codex").trim() || "codex";
    return `thread:${backend}:${threadId}`;
  }
  return `session:${String(session.session_id || "").trim()}`;
}

function dedupeSessions(items: SessionSummary[]) {
  const representativeByKey = new Map<string, SessionSummary>();
  const representativeBySessionId = new Map<string, string>();
  const unique: SessionSummary[] = [];
  for (const session of items) {
    const sessionId = String(session.session_id || "").trim();
    if (!sessionId) {
      continue;
    }
    const key = sessionDedupeKey(session);
    const representative = representativeByKey.get(key);
    if (representative) {
      representativeBySessionId.set(sessionId, representative.session_id);
      continue;
    }
    representativeByKey.set(key, session);
    representativeBySessionId.set(sessionId, session.session_id);
    unique.push(session);
  }
  return { sessions: unique, representativeBySessionId };
}

function orderedGroupKeys(items: SessionSummary[]) {
  const seen = new Set<string>();
  const keys: string[] = [];
  for (const session of items) {
    const key = sessionGroupKey(session);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    keys.push(key);
  }
  return keys;
}

function replaceGroupRows(items: SessionSummary[], groupKey: string, groupRows: SessionSummary[]) {
  const next: SessionSummary[] = [];
  let inserted = false;
  for (const session of items) {
    if (sessionGroupKey(session) === groupKey) {
      if (!inserted) {
        next.push(...groupRows);
        inserted = true;
      }
      continue;
    }
    next.push(session);
  }
  if (!inserted && groupRows.length > 0) {
    next.push(...groupRows);
  }
  return next;
}

export function createSessionsStore(): SessionsStore {
  let state: SessionsState = {
    items: [],
    activeSessionId: null,
    loading: false,
    bootstrapLoaded: false,
    remainingByGroup: {},
    omittedGroupCount: 0,
    newSessionDefaults: null,
    recentCwds: [],
    cwdGroups: {},
    tmuxAvailable: false,
  };
  const listeners = new Set<() => void>();
  let currentRefreshId = 0;
  let currentBootstrapRefreshId = 0;
  let hasResolvedInitialSelection = false;
  let revealedGroupKeys: string[] = [];
  let loadedGroupLimits: Record<string, number> = {};

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    async refresh(options?: RefreshSessionsOptions) {
      const refreshId = ++currentRefreshId;
      state = { ...state, loading: true };
      emit();

      try {
        const data = await api.listSessions();
        if (refreshId !== currentRefreshId) {
          return;
        }
        const baseDeduped = dedupeSessions(Array.isArray(data.sessions) ? data.sessions : []);
        let sessions = baseDeduped.sessions;
        const representativeBySessionId = new Map(baseDeduped.representativeBySessionId);
        const remainingByGroup: Record<string, number> = { ...(data.remaining_by_group ?? {}) };
        const baseGroupKeys = new Set(orderedGroupKeys(sessions));
        const extraGroupKeys = revealedGroupKeys.filter((groupKey) => !baseGroupKeys.has(groupKey));
        const groupKeysToRefresh = Array.from(new Set([
          ...Object.keys(loadedGroupLimits),
          ...extraGroupKeys,
        ]));

        if (groupKeysToRefresh.length > 0) {
          const groupResponses = await Promise.all(groupKeysToRefresh.map(async (groupKey) => {
            const desiredLimit = Math.max(GROUP_PAGE_SIZE, Number(loadedGroupLimits[groupKey] || 0) || GROUP_PAGE_SIZE);
            const response = await api.listSessions({ groupKey, offset: 0, limit: desiredLimit });
            const deduped = dedupeSessions(Array.isArray(response.sessions) ? response.sessions : []);
            for (const [sessionId, representativeId] of deduped.representativeBySessionId.entries()) {
              representativeBySessionId.set(sessionId, representativeId);
            }
            return {
              groupKey,
              sessions: deduped.sessions,
              remaining: Number(response.remaining_by_group?.[groupKey] || 0),
            };
          }));

          const aliveRevealedGroupKeys: string[] = [];
          for (const response of groupResponses) {
            if (response.sessions.length > 0) {
              sessions = replaceGroupRows(sessions, response.groupKey, response.sessions);
              if (revealedGroupKeys.includes(response.groupKey) && !baseGroupKeys.has(response.groupKey)) {
                aliveRevealedGroupKeys.push(response.groupKey);
              }
            }
            if (response.remaining > 0) {
              remainingByGroup[response.groupKey] = response.remaining;
            } else {
              delete remainingByGroup[response.groupKey];
            }
            if (response.sessions.length <= 0) {
              delete loadedGroupLimits[response.groupKey];
            }
          }

          revealedGroupKeys = revealedGroupKeys.filter((groupKey) => {
            if (baseGroupKeys.has(groupKey)) {
              return true;
            }
            return aliveRevealedGroupKeys.includes(groupKey);
          });
        }

        const combinedSessions = dedupeSessions(sessions).sessions;
        const revealedHiddenGroupCount = extraGroupKeys.filter((groupKey) => revealedGroupKeys.includes(groupKey)).length;
        const sessionIds = new Set(combinedSessions.map((session) => session.session_id));
        const activeRepresentativeSessionId = state.activeSessionId
          ? representativeBySessionId.get(state.activeSessionId) ?? state.activeSessionId
          : null;
        const preservedActiveSessionId = activeRepresentativeSessionId && sessionIds.has(activeRepresentativeSessionId)
          ? activeRepresentativeSessionId
          : null;
        const nextActiveSessionId = options?.preferNewest
          ? sessions[0]?.session_id ?? null
          : preservedActiveSessionId
            ?? (!hasResolvedInitialSelection ? sessions[0]?.session_id ?? null : null);
        if (nextActiveSessionId) {
          hasResolvedInitialSelection = true;
        }
        state = {
          ...state,
          items: combinedSessions,
          activeSessionId: nextActiveSessionId,
          loading: false,
          remainingByGroup,
          omittedGroupCount: Math.max(0, Number(data.omitted_group_count || 0) - revealedHiddenGroupCount),
        };
        emit();
      } catch (error) {
        if (refreshId === currentRefreshId) {
          state = { ...state, loading: false };
          emit();
          throw error;
        }
      }
    },
    async refreshBootstrap() {
      const refreshId = ++currentBootstrapRefreshId;
      const data = await api.getSessionsBootstrap();
      if (refreshId !== currentBootstrapRefreshId) {
        return;
      }
      state = {
        ...state,
        bootstrapLoaded: true,
        newSessionDefaults: data.new_session_defaults ?? state.newSessionDefaults,
        recentCwds: Array.isArray(data.recent_cwds)
          ? data.recent_cwds.filter((cwd): cwd is string => typeof cwd === "string")
          : state.recentCwds,
        cwdGroups: data.cwd_groups ?? state.cwdGroups,
        tmuxAvailable: typeof data.tmux_available === "boolean" ? data.tmux_available : state.tmuxAvailable,
      };
      emit();
    },
    async loadMoreGroup(groupKey: string, limit = GROUP_PAGE_SIZE) {
      const offset = state.items.filter((session) => sessionGroupKey(session) === groupKey).length;
      const data = await api.listSessions({ groupKey, offset, limit });
      const appended = Array.isArray(data.sessions) ? data.sessions : [];
      const existingIds = new Set(state.items.map((session) => session.session_id));
      const uniqueAppended = appended.filter((session) => !existingIds.has(session.session_id));
      loadedGroupLimits[groupKey] = offset + appended.length;
      const nextRemaining = { ...state.remainingByGroup };
      const reportedRemaining = data.remaining_by_group?.[groupKey] ?? 0;
      if (reportedRemaining > 0) {
        nextRemaining[groupKey] = reportedRemaining;
      } else {
        delete nextRemaining[groupKey];
      }
      state = {
        ...state,
        items: [...state.items, ...uniqueAppended],
        remainingByGroup: nextRemaining,
      };
      emit();
    },
    async loadMoreGroups(limit = GROUPS_PAGE_SIZE) {
      const loadedGroups = new Set(state.items.map((session) => sessionGroupKey(session)));
      const data = await api.listSessions({
        groupOffset: loadedGroups.size,
        groupLimit: limit,
      });
      const appended = Array.isArray(data.sessions) ? data.sessions : [];
      const existingIds = new Set(state.items.map((session) => session.session_id));
      const uniqueAppended = appended.filter((session) => !existingIds.has(session.session_id));
      for (const groupKey of orderedGroupKeys(uniqueAppended)) {
        if (!revealedGroupKeys.includes(groupKey)) {
          revealedGroupKeys.push(groupKey);
        }
      }
      state = {
        ...state,
        items: [...state.items, ...uniqueAppended],
        remainingByGroup: { ...state.remainingByGroup, ...(data.remaining_by_group ?? {}) },
        omittedGroupCount: typeof data.omitted_group_count === "number" ? data.omitted_group_count : state.omittedGroupCount,
      };
      emit();
    },
    select(sessionId: string) {
      hasResolvedInitialSelection = true;
      state = { ...state, activeSessionId: sessionId };
      emit();
    },
  };
}
