import { api } from "../../lib/api";
import type { MessageEvent } from "../../lib/types";

export interface MessagesState {
  bySessionId: Record<string, MessageEvent[]>;
  offsetsBySessionId: Record<string, number>;
  hasOlderBySessionId: Record<string, boolean>;
  olderBeforeBySessionId: Record<string, number>;
  loadingOlderBySessionId: Record<string, boolean>;
  loadingBySessionId: Record<string, boolean>;
  loadedBySessionId: Record<string, boolean>;
  loading: boolean;
}

export interface MessagesStore {
  getState(): MessagesState;
  subscribe(listener: () => void): () => void;
  applyLive(sessionId: string, events: MessageEvent[], options: { replace: boolean; offset?: number }): void;
  loadInitial(sessionId: string): Promise<void>;
  poll(sessionId: string): Promise<void>;
  loadOlder(sessionId: string, limit?: number): Promise<void>;
}

export function createMessagesStore(): MessagesStore {
  let state: MessagesState = {
    bySessionId: {},
    offsetsBySessionId: {},
    hasOlderBySessionId: {},
    olderBeforeBySessionId: {},
    loadingOlderBySessionId: {},
    loadingBySessionId: {},
    loadedBySessionId: {},
    loading: false,
  };
  const listeners = new Set<() => void>();
  const currentLoadIds: Record<string, number> = {};
  const currentOlderLoadIds: Record<string, number> = {};
  const inFlightLoads: Record<string, Promise<void> | undefined> = {};
  const inFlightLoadTokens: Record<string, object | undefined> = {};

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  const recomputeLoading = (loadingBySessionId: Record<string, boolean>) =>
    Object.values(loadingBySessionId).some((value) => value === true);

  const load = async (sessionId: string, init: boolean) => {
    if (!init) {
      const existing = inFlightLoads[sessionId];
      if (existing) {
        return existing;
      }
    }

    const loadId = (currentLoadIds[sessionId] || 0) + 1;
    currentLoadIds[sessionId] = loadId;
    const requestToken = {};
    inFlightLoadTokens[sessionId] = requestToken;
    const request = (async () => {
      state = {
        ...state,
        loadingBySessionId: {
          ...state.loadingBySessionId,
          [sessionId]: true,
        },
        loading: true,
      };
      emit();

      try {
        const data = await api.listMessages(sessionId, init, undefined, init ? undefined : state.offsetsBySessionId[sessionId]);
        if (loadId !== currentLoadIds[sessionId]) {
          return;
        }
        const priorEvents = init ? [] : state.bySessionId[sessionId] ?? [];
        const nextLoadingBySessionId = {
          ...state.loadingBySessionId,
          [sessionId]: false,
        };
        state = {
          bySessionId: {
            ...state.bySessionId,
            [sessionId]: init ? data.events : [...priorEvents, ...data.events],
          },
          offsetsBySessionId: {
            ...state.offsetsBySessionId,
            [sessionId]: typeof data.offset === "number" ? data.offset : state.offsetsBySessionId[sessionId] ?? 0,
          },
          hasOlderBySessionId: {
            ...state.hasOlderBySessionId,
            [sessionId]: init ? data.has_older === true : state.hasOlderBySessionId[sessionId] ?? false,
          },
          olderBeforeBySessionId: {
            ...state.olderBeforeBySessionId,
            [sessionId]: init
              ? typeof data.next_before === "number"
                ? data.next_before
                : 0
              : state.olderBeforeBySessionId[sessionId] ?? 0,
          },
          loadingOlderBySessionId: {
            ...state.loadingOlderBySessionId,
            [sessionId]: false,
          },
          loadingBySessionId: nextLoadingBySessionId,
          loadedBySessionId: {
            ...state.loadedBySessionId,
            [sessionId]: true,
          },
          loading: recomputeLoading(nextLoadingBySessionId),
        };
        emit();
      } catch (error) {
        if (loadId === currentLoadIds[sessionId]) {
          const nextLoadingBySessionId = {
            ...state.loadingBySessionId,
            [sessionId]: false,
          };
          state = {
            ...state,
            loadingBySessionId: nextLoadingBySessionId,
            loading: recomputeLoading(nextLoadingBySessionId),
          };
          emit();
          throw error;
        }
      } finally {
        if (inFlightLoadTokens[sessionId] === requestToken) {
          delete inFlightLoadTokens[sessionId];
          delete inFlightLoads[sessionId];
        }
      }
    })();
    inFlightLoads[sessionId] = request;
    return request;
  };

  const loadOlder = async (sessionId: string, limit = 80) => {
    const before = state.olderBeforeBySessionId[sessionId] ?? 0;
    if (before <= 0 && !state.hasOlderBySessionId[sessionId]) {
      return;
    }

    const loadId = (currentOlderLoadIds[sessionId] || 0) + 1;
    currentOlderLoadIds[sessionId] = loadId;
    state = {
      ...state,
      loadingOlderBySessionId: {
        ...state.loadingOlderBySessionId,
        [sessionId]: true,
      },
    };
    emit();

    try {
      const data = await api.listMessages(sessionId, true, undefined, undefined, before, limit);
      if (loadId !== currentOlderLoadIds[sessionId]) {
        return;
      }

      state = {
        ...state,
        bySessionId: {
          ...state.bySessionId,
          [sessionId]: [...data.events, ...(state.bySessionId[sessionId] ?? [])],
        },
        offsetsBySessionId: {
          ...state.offsetsBySessionId,
          [sessionId]: typeof data.offset === "number" ? data.offset : state.offsetsBySessionId[sessionId] ?? 0,
        },
        hasOlderBySessionId: {
          ...state.hasOlderBySessionId,
          [sessionId]: data.has_older === true,
        },
        olderBeforeBySessionId: {
          ...state.olderBeforeBySessionId,
          [sessionId]: typeof data.next_before === "number" ? data.next_before : 0,
        },
        loadingOlderBySessionId: {
          ...state.loadingOlderBySessionId,
          [sessionId]: false,
        },
      };
      emit();
    } catch (error) {
      if (loadId === currentOlderLoadIds[sessionId]) {
        state = {
          ...state,
          loadingOlderBySessionId: {
            ...state.loadingOlderBySessionId,
            [sessionId]: false,
          },
        };
        emit();
        throw error;
      }
    }
  };

  const applyLive = (sessionId: string, events: MessageEvent[], options: { replace: boolean; offset?: number }) => {
    const priorEvents = state.bySessionId[sessionId] ?? [];
    const nextLoadingBySessionId = {
      ...state.loadingBySessionId,
      [sessionId]: false,
    };
    state = {
      ...state,
      bySessionId: {
        ...state.bySessionId,
        [sessionId]: options.replace ? events : [...priorEvents, ...events],
      },
      offsetsBySessionId: {
        ...state.offsetsBySessionId,
        [sessionId]: typeof options.offset === "number" ? options.offset : state.offsetsBySessionId[sessionId] ?? 0,
      },
      loadingBySessionId: nextLoadingBySessionId,
      loadedBySessionId: {
        ...state.loadedBySessionId,
        [sessionId]: true,
      },
      loading: recomputeLoading(nextLoadingBySessionId),
    };
    emit();
  };

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    loadInitial(sessionId: string) {
      return load(sessionId, true);
    },
    poll(sessionId: string) {
      return load(sessionId, false);
    },
    applyLive,
    loadOlder(sessionId: string, limit?: number) {
      return loadOlder(sessionId, limit);
    },
  };
}
