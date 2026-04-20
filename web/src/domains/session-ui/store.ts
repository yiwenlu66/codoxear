import { api } from "../../lib/api";

export interface SessionUiState {
  sessionId: string | null;
  runtimeId: string | null;
  diagnostics: Record<string, unknown> | null;
  queue: Record<string, unknown> | null;
  loading: boolean;
}

export interface SessionUiRefreshOptions {
  agentBackend?: string;
  runtimeId?: string | null;
}

export interface SessionUiStore {
  getState(): SessionUiState;
  subscribe(listener: () => void): () => void;
  refresh(sessionId: string, options?: SessionUiRefreshOptions): Promise<void>;
}

export function createSessionUiStore(): SessionUiStore {
  let state: SessionUiState = {
    sessionId: null,
    runtimeId: null,
    diagnostics: null,
    queue: null,
    loading: false,
  };
  const listeners = new Set<() => void>();
  let currentRefreshId = 0;
  let inFlightRefresh: { key: string; promise: Promise<void> } | null = null;

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
      const refreshKey = `${sessionId}:${String(_options?.runtimeId || "")}`;
      if (inFlightRefresh && inFlightRefresh.key === refreshKey) {
        return inFlightRefresh.promise;
      }

      const refreshId = ++currentRefreshId;
      const preserveCurrentState = state.sessionId === sessionId;
      state = {
        sessionId,
        runtimeId: preserveCurrentState ? state.runtimeId : (_options?.runtimeId ?? null),
        diagnostics: preserveCurrentState ? state.diagnostics : null,
        queue: preserveCurrentState ? state.queue : null,
        loading: true,
      };
      emit();

      let request: Promise<void> | null = null;
      request = (async () => {
        try {
          const workspace = _options?.runtimeId
            ? await api.getWorkspace(sessionId, undefined, _options.runtimeId)
            : await api.getWorkspace(sessionId);
          if (refreshId !== currentRefreshId) {
            return;
          }

          state = {
            sessionId,
            runtimeId: typeof workspace.runtime_id === "string" && workspace.runtime_id.trim()
              ? workspace.runtime_id
              : (_options?.runtimeId ?? null),
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
        } finally {
          if (inFlightRefresh?.promise === request) {
            inFlightRefresh = null;
          }
        }
      })();

      inFlightRefresh = { key: refreshKey, promise: request };
      return request;
    },
  };
}
