import { api } from "../../lib/api";
import type { MessageEvent } from "../../lib/types";

const COMPOSER_DRAFTS_STORAGE_KEY = "codoxear.composerDrafts.v1";

export interface PendingComposerMessage {
  localId: string;
  role: "user";
  text: string;
  pending: true;
  requestId?: string;
  [key: string]: unknown;
}

export interface ComposerState {
  draftBySessionId: Record<string, string>;
  sending: boolean;
  pendingBySessionId: Record<string, PendingComposerMessage[]>;
}

export interface ComposerStore {
  getState(): ComposerState;
  subscribe(listener: () => void): () => void;
  setDraft(sessionId: string | null | undefined, value: string): void;
  submit(sessionId: string, runtimeId?: string | null): Promise<unknown>;
  clearAcknowledgedPending(sessionId: string, persistedEvents: MessageEvent[]): void;
}

function readPersistedDrafts(): Record<string, string> {
  if (typeof window === "undefined") {
    return {} as Record<string, string>;
  }

  try {
    const raw = window.localStorage.getItem(COMPOSER_DRAFTS_STORAGE_KEY);
    if (!raw) {
      return {} as Record<string, string>;
    }

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return {} as Record<string, string>;
    }

    const drafts: Record<string, string> = {};
    for (const [sessionId, value] of Object.entries(parsed)) {
      if (typeof sessionId === "string" && typeof value === "string" && value.length > 0) {
        drafts[sessionId] = value;
      }
    }

    return drafts;
  } catch {
    return {} as Record<string, string>;
  }
}

function persistDrafts(draftBySessionId: Record<string, string>) {
  if (typeof window === "undefined") {
    return;
  }

  try {
    if (Object.keys(draftBySessionId).length === 0) {
      window.localStorage.removeItem(COMPOSER_DRAFTS_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(COMPOSER_DRAFTS_STORAGE_KEY, JSON.stringify(draftBySessionId));
  } catch {
    // localStorage persistence is best-effort only.
  }
}

function nextDraftMap(draftBySessionId: Record<string, string>, sessionId: string | null | undefined, value: string) {
  if (!sessionId) {
    return draftBySessionId;
  }

  if (!value.length) {
    if (!(sessionId in draftBySessionId)) {
      return draftBySessionId;
    }

    const next = { ...draftBySessionId };
    delete next[sessionId];
    return next;
  }

  return {
    ...draftBySessionId,
    [sessionId]: value,
  };
}

export function createComposerStore(): ComposerStore {
  let state: ComposerState = { draftBySessionId: readPersistedDrafts(), sending: false, pendingBySessionId: {} };
  const listeners = new Set<() => void>();
  let nextPendingId = 0;

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  const updateDrafts = (sessionId: string | null | undefined, value: string) => {
    const draftBySessionId = nextDraftMap(state.draftBySessionId, sessionId, value);
    if (draftBySessionId === state.draftBySessionId) {
      return state.draftBySessionId;
    }

    persistDrafts(draftBySessionId);
    state = { ...state, draftBySessionId };
    return draftBySessionId;
  };

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    setDraft(sessionId: string | null | undefined, value: string) {
      updateDrafts(sessionId, value);
      emit();
    },
    async submit(sessionId: string, runtimeId?: string | null) {
      const text = state.draftBySessionId[sessionId] ?? "";
      if (!text.trim() || state.sending) return;

      nextPendingId += 1;
      const pendingMessage: PendingComposerMessage = {
        localId: `local-pending-${nextPendingId}`,
        role: "user",
        text,
        pending: true,
      };

      state = {
        ...state,
        draftBySessionId: nextDraftMap(state.draftBySessionId, sessionId, ""),
        sending: true,
        pendingBySessionId: {
          ...state.pendingBySessionId,
          [sessionId]: [...(state.pendingBySessionId[sessionId] ?? []), pendingMessage],
        },
      };
      persistDrafts(state.draftBySessionId);
      emit();

      try {
        const response = runtimeId
          ? await api.sendMessage(sessionId, text, runtimeId)
          : await api.sendMessage(sessionId, text);
        const requestId = response && typeof response === "object" && typeof (response as { request_id?: unknown }).request_id === "string"
          ? String((response as { request_id?: unknown }).request_id)
          : "";
        state = {
          ...state,
          sending: false,
          pendingBySessionId: {
            ...state.pendingBySessionId,
            [sessionId]: (state.pendingBySessionId[sessionId] ?? []).map((item) => item.localId === pendingMessage.localId
              ? { ...item, requestId: requestId || item.requestId }
              : item),
          },
        };
        emit();
        return response;
      } catch (error) {
        state = {
          ...state,
          draftBySessionId: nextDraftMap(state.draftBySessionId, sessionId, state.draftBySessionId[sessionId] ? state.draftBySessionId[sessionId] : text),
          sending: false,
          pendingBySessionId: {
            ...state.pendingBySessionId,
            [sessionId]: (state.pendingBySessionId[sessionId] ?? []).filter((item) => item.localId !== pendingMessage.localId),
          },
        };
        persistDrafts(state.draftBySessionId);
        emit();
        throw error;
      }
    },
    clearAcknowledgedPending(sessionId: string, persistedEvents: MessageEvent[]) {
      const pending = state.pendingBySessionId[sessionId] ?? [];
      if (!pending.length) return;

      const persistedUserTexts = persistedEvents
        .filter((event) => event?.role === "user" && typeof event?.text === "string")
        .map((event) => String(event.text));
      const failedRequestIds = new Set(
        persistedEvents
          .filter((event) => typeof event?.request_id === "string" && event?.request_state === "failed")
          .map((event) => String(event.request_id)),
      );
      const failedTexts = new Set(
        persistedEvents
          .filter((event) => event?.request_state === "failed" && typeof event?.pending_text === "string")
          .map((event) => String(event.pending_text)),
      );
      if (!persistedUserTexts.length && !failedRequestIds.size && !failedTexts.size) return;

      const acknowledgedLocalIds = new Set<string>();
      let persistedIdx = persistedUserTexts.length - 1;
      let pendingIdx = pending.length - 1;
      while (persistedIdx >= 0 && pendingIdx >= 0) {
        if (persistedUserTexts[persistedIdx] === pending[pendingIdx].text) {
          acknowledgedLocalIds.add(pending[pendingIdx].localId);
          pendingIdx -= 1;
        }
        persistedIdx -= 1;
      }
      for (const item of pending) {
        if ((item.requestId && failedRequestIds.has(item.requestId)) || failedTexts.has(item.text)) {
          acknowledgedLocalIds.add(item.localId);
        }
      }
      if (!acknowledgedLocalIds.size) return;

      state = {
        ...state,
        pendingBySessionId: {
          ...state.pendingBySessionId,
          [sessionId]: pending.filter((item) => !acknowledgedLocalIds.has(item.localId)),
        },
      };
      emit();
    },
  };
}
