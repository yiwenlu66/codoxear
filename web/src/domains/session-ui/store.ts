import { api } from "../../lib/api";

export interface SessionUiState {
  sessionId: string | null;
  diagnostics: Record<string, unknown> | null;
  queue: Record<string, unknown> | null;
  loading: boolean;
}

export interface SessionUiRefreshOptions {
  agentBackend?: string;
}

export interface SessionUiStore {
  getState(): SessionUiState;
  subscribe(listener: () => void): () => void;
  refresh(sessionId: string, options?: SessionUiRefreshOptions): Promise<void>;
}

export function createSessionUiStore(): SessionUiStore {
  let state: SessionUiState = {
    sessionId: null,
    diagnostics: null,
    queue: null,
    loading: false,
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
    async refresh(sessionId: string, _options?: SessionUiRefreshOptions) {
      const refreshId = ++currentRefreshId;
      const preserveCurrentState = state.sessionId === sessionId;
      state = {
        sessionId,
        diagnostics: preserveCurrentState ? state.diagnostics : null,
        queue: preserveCurrentState ? state.queue : null,
        loading: true,
      };
      emit();

      try {
        const workspace = await api.getWorkspace(sessionId);
        if (refreshId !== currentRefreshId) {
          return;
        }

        state = {
          sessionId,
          diagnostics: (workspace.diagnostics ?? null) as Record<string, unknown> | null,
          queue: (workspace.queue ?? null) as Record<string, unknown> | null,
          loading: false,
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
  };
}
