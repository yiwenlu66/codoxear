import { api } from "../../lib/api";
import { HttpError } from "../../lib/http";
import type { ContextUsagePayload, SessionUiRequest } from "../../lib/types";
import type { MessagesStore } from "../messages/store";

export interface LiveSessionState {
  offsetsBySessionId: Record<string, number>;
  liveOffsetsBySessionId: Record<string, number>;
  bridgeOffsetsBySessionId: Record<string, number>;
  requestsBySessionId: Record<string, SessionUiRequest[]>;
  requestVersionsBySessionId: Record<string, string>;
  busyBySessionId: Record<string, boolean>;
  loadingBySessionId: Record<string, boolean>;
  errorBySessionId: Record<string, string>;
  tokenBySessionId: Record<string, Record<string, unknown> | null>;
  contextUsageBySessionId: Record<string, ContextUsagePayload | null>;
}

export interface LiveSessionStore {
  getState(): LiveSessionState;
  subscribe(listener: () => void): () => void;
  loadInitial(sessionId: string, runtimeId?: string | null): Promise<void>;
  poll(sessionId: string, runtimeId?: string | null): Promise<void>;
}

function liveSessionErrorMessage(error: unknown): string {
  if (error instanceof HttpError && error.status === 404) {
    return "Pi RPC session ended or broker exited";
  }
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return "live session unavailable";
}

export function createLiveSessionStore(messagesStore: MessagesStore): LiveSessionStore {
  let state: LiveSessionState = {
    offsetsBySessionId: {},
    liveOffsetsBySessionId: {},
    bridgeOffsetsBySessionId: {},
    requestsBySessionId: {},
    requestVersionsBySessionId: {},
    busyBySessionId: {},
    loadingBySessionId: {},
    errorBySessionId: {},
    tokenBySessionId: {},
    contextUsageBySessionId: {},
  };
  const listeners = new Set<() => void>();
  const inFlightBySessionId: Record<string, Promise<void> | undefined> = {};

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  const runLoad = async (sessionId: string, replace: boolean, runtimeId?: string | null) => {
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
        const bridgeOffset = replace ? undefined : state.bridgeOffsetsBySessionId[sessionId];
        const requestsVersion = replace ? undefined : state.requestVersionsBySessionId[sessionId];
        const payload = runtimeId
          ? await api.getLiveSession(sessionId, offset, requestsVersion, undefined, liveOffset, runtimeId, bridgeOffset)
          : await api.getLiveSession(sessionId, offset, requestsVersion, undefined, liveOffset, undefined, bridgeOffset);
        messagesStore.applyLive(sessionId, payload.events ?? [], {
          replace,
          offset: typeof payload.offset === "number" ? payload.offset : offset,
          hasOlder: payload.has_older === true,
          nextBefore: typeof payload.next_before === "number" ? payload.next_before : undefined,
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
          bridgeOffsetsBySessionId: {
            ...state.bridgeOffsetsBySessionId,
            [sessionId]: typeof payload.bridge_offset === "number" ? payload.bridge_offset : state.bridgeOffsetsBySessionId[sessionId] ?? 0,
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
          errorBySessionId: {
            ...state.errorBySessionId,
            [sessionId]: "",
          },
          tokenBySessionId: {
            ...state.tokenBySessionId,
            [sessionId]: payload.token && typeof payload.token === "object" ? payload.token as Record<string, unknown> : null,
          },
          contextUsageBySessionId: {
            ...state.contextUsageBySessionId,
            [sessionId]: payload.context_usage && typeof payload.context_usage === "object" ? payload.context_usage : null,
          },
        };
        emit();
      } catch (error) {
        const message = liveSessionErrorMessage(error);
        state = {
          ...state,
          loadingBySessionId: {
            ...state.loadingBySessionId,
            [sessionId]: false,
          },
          errorBySessionId: {
            ...state.errorBySessionId,
            [sessionId]: message,
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
    loadInitial(sessionId: string, runtimeId?: string | null) {
      return runLoad(sessionId, true, runtimeId);
    },
    poll(sessionId: string, runtimeId?: string | null) {
      return runLoad(sessionId, false, runtimeId);
    },
  };
}
