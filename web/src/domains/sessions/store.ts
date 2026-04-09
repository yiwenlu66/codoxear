import { api } from "../../lib/api";
import type { CwdGroupMeta, NewSessionDefaults, SessionSummary } from "../../lib/types";

export interface SessionsState {
  items: SessionSummary[];
  activeSessionId: string | null;
  loading: boolean;
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
  select(sessionId: string): void;
}

export function createSessionsStore(): SessionsStore {
  let state: SessionsState = {
    items: [],
    activeSessionId: null,
    loading: false,
    newSessionDefaults: null,
    recentCwds: [],
    cwdGroups: {},
    tmuxAvailable: false,
  };
  const listeners = new Set<() => void>();
  let currentRefreshId = 0;

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
        const sessionIds = new Set(data.sessions.map((session) => session.session_id));
        const nextActiveSessionId = options?.preferNewest
          ? data.sessions[0]?.session_id ?? null
          : state.activeSessionId && sessionIds.has(state.activeSessionId)
            ? state.activeSessionId
            : null;
        state = {
          items: data.sessions,
          activeSessionId: nextActiveSessionId,
          loading: false,
          newSessionDefaults: data.new_session_defaults ?? state.newSessionDefaults,
          recentCwds: Array.isArray(data.recent_cwds) ? data.recent_cwds.filter((cwd): cwd is string => typeof cwd === "string") : state.recentCwds,
          cwdGroups: data.cwd_groups ?? state.cwdGroups,
          tmuxAvailable: typeof data.tmux_available === "boolean" ? data.tmux_available : state.tmuxAvailable,
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
    select(sessionId: string) {
      state = { ...state, activeSessionId: sessionId };
      emit();
    },
  };
}
