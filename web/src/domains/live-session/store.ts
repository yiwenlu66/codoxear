import { api } from "../../lib/api";
import type { SessionUiRequest } from "../../lib/types";
import type { MessagesStore } from "../messages/store";

export interface LiveSessionState {
  offsetsBySessionId: Record<string, number>;
  liveOffsetsBySessionId: Record<string, number>;
  requestsBySessionId: Record<string, SessionUiRequest[]>;
  requestVersionsBySessionId: Record<string, string>;
  busyBySessionId: Record<string, boolean>;
  loadingBySessionId: Record<string, boolean>;
}

export interface LiveSessionStore {
  getState(): LiveSessionState;
  subscribe(listener: () => void): () => void;
  loadInitial(sessionId: string): Promise<void>;
  poll(sessionId: string): Promise<void>;
}

export function createLiveSessionStore(messagesStore: MessagesStore): LiveSessionStore {
  let state: LiveSessionState = {
    offsetsBySessionId: {},
    liveOffsetsBySessionId: {},
    requestsBySessionId: {},
    requestVersionsBySessionId: {},
    busyBySessionId: {},
    loadingBySessionId: {},
  };
  const listeners = new Set<() => void>();
  const inFlightBySessionId: Record<string, Promise<void> | undefined> = {};

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  const runLoad = async (sessionId: string, replace: boolean) => {
    const existing = inFlightBySessionId[sessionId];
    if (existing) {
      return existing;
    }

    const request = (async () => {
      state = {
        ...state,
        loadingBySessionId: {
          ...state.loadingBySessionId,
          [sessionId]: true,
        },
      };
      emit();

      try {
        const offset = replace ? undefined : state.offsetsBySessionId[sessionId];
        const liveOffset = replace ? undefined : state.liveOffsetsBySessionId[sessionId];
        const requestsVersion = replace ? undefined : state.requestVersionsBySessionId[sessionId];
        const payload = await api.getLiveSession(sessionId, offset, requestsVersion, undefined, liveOffset);
        messagesStore.applyLive(sessionId, payload.events ?? [], {
          replace,
          offset: typeof payload.offset === "number" ? payload.offset : offset,
        });
        const nextRequests = Array.isArray(payload.requests)
          ? payload.requests
          : state.requestsBySessionId[sessionId] ?? [];
        const nextRequestVersionsBySessionId = {
          ...state.requestVersionsBySessionId,
        };
        if (typeof payload.requests_version === "string") {
          nextRequestVersionsBySessionId[sessionId] = payload.requests_version;
        }
        state = {
          ...state,
          offsetsBySessionId: {
            ...state.offsetsBySessionId,
            [sessionId]: typeof payload.offset === "number" ? payload.offset : state.offsetsBySessionId[sessionId] ?? 0,
          },
          liveOffsetsBySessionId: {
            ...state.liveOffsetsBySessionId,
            [sessionId]: typeof payload.live_offset === "number" ? payload.live_offset : state.liveOffsetsBySessionId[sessionId] ?? 0,
          },
          requestsBySessionId: {
            ...state.requestsBySessionId,
            [sessionId]: nextRequests,
          },
          requestVersionsBySessionId: nextRequestVersionsBySessionId,
          busyBySessionId: {
            ...state.busyBySessionId,
            [sessionId]: payload.busy === true,
          },
          loadingBySessionId: {
            ...state.loadingBySessionId,
            [sessionId]: false,
          },
        };
        emit();
      } catch (error) {
        state = {
          ...state,
          loadingBySessionId: {
            ...state.loadingBySessionId,
            [sessionId]: false,
          },
        };
        emit();
        throw error;
      } finally {
        delete inFlightBySessionId[sessionId];
      }
    })();

    inFlightBySessionId[sessionId] = request;
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
    loadInitial(sessionId: string) {
      return runLoad(sessionId, true);
    },
    poll(sessionId: string) {
      return runLoad(sessionId, false);
    },
  };
}
