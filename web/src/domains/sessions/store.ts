import { api } from "../../lib/api";
import type { CwdGroupMeta, NewSessionDefaults, SessionSummary } from "../../lib/types";

export interface SessionsState {
  items: SessionSummary[];
  activeSessionId: string | null;
  loading: boolean;
  bootstrapLoaded: boolean;
  remainingCount: number;
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
  loadMore(limit?: number): Promise<void>;
  select(sessionId: string): void;
}

const PAGE_SIZE = 50;

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

export function createSessionsStore(): SessionsStore {
  let state: SessionsState = {
    items: [],
    activeSessionId: null,
    loading: false,
    bootstrapLoaded: false,
    remainingCount: 0,
    newSessionDefaults: null,
    recentCwds: [],
    cwdGroups: {},
    tmuxAvailable: false,
  };
  const listeners = new Set<() => void>();
  let currentRefreshId = 0;
  let currentBootstrapRefreshId = 0;
  let hasResolvedInitialSelection = false;
  let loadedLimit = PAGE_SIZE;
  let inFlightRefresh: { key: string; promise: Promise<void> } | null = null;

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  const refresh = async (options?: RefreshSessionsOptions) => {
    const refreshKey = `${loadedLimit}:${options?.preferNewest === true ? "newest" : "preserve"}`;
    if (inFlightRefresh && inFlightRefresh.key === refreshKey) {
      return inFlightRefresh.promise;
    }

    const refreshId = ++currentRefreshId;
    state = { ...state, loading: true };
    emit();

    let request: Promise<void> | null = null;
    request = (async () => {
      try {
        const data = await api.listSessions({ limit: loadedLimit });
        if (refreshId !== currentRefreshId) {
          return;
        }
        const deduped = dedupeSessions(Array.isArray(data.sessions) ? data.sessions : []);
        const sessions = deduped.sessions;
        const representativeBySessionId = deduped.representativeBySessionId;
        const sessionIds = new Set(sessions.map((session) => session.session_id));
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
          items: sessions,
          activeSessionId: nextActiveSessionId,
          loading: false,
          remainingCount: Math.max(0, Number(data.remaining_count || 0)),
        };
        emit();
      } catch (error) {
        if (refreshId === currentRefreshId) {
          state = { ...state, loading: false };
          emit();
          throw error;
        }
      } finally {
        if (inFlightRefresh?.promise === request) {
          inFlightRefresh = null;
        }
      }
    })();

    inFlightRefresh = { key: refreshKey, promise: request };
    return request;
  };

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    refresh,
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
    async loadMore(limit = PAGE_SIZE) {
      if (state.remainingCount <= 0) {
        return;
      }
      loadedLimit = Math.max(PAGE_SIZE, loadedLimit + Math.max(1, limit));
      await refresh();
    },
    select(sessionId: string) {
      hasResolvedInitialSelection = true;
      state = { ...state, activeSessionId: sessionId };
      emit();
    },
  };
}
