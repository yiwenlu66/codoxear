import { api } from "../../lib/api";
import type { MessageEvent } from "../../lib/types";

export interface PendingComposerMessage {
  localId: string;
  role: "user";
  text: string;
  pending: true;
  [key: string]: unknown;
}

export interface ComposerState {
  draft: string;
  sending: boolean;
  pendingBySessionId: Record<string, PendingComposerMessage[]>;
}

export interface ComposerStore {
  getState(): ComposerState;
  subscribe(listener: () => void): () => void;
  setDraft(value: string): void;
  submit(sessionId: string): Promise<unknown>;
  clearAcknowledgedPending(sessionId: string, persistedEvents: MessageEvent[]): void;
}

export function createComposerStore(): ComposerStore {
  let state: ComposerState = { draft: "", sending: false, pendingBySessionId: {} };
  const listeners = new Set<() => void>();
  let nextPendingId = 0;

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
    setDraft(value: string) {
      state = { ...state, draft: value };
      emit();
    },
    async submit(sessionId: string) {
      if (!state.draft.trim() || state.sending) return;

      nextPendingId += 1;
      const text = state.draft;
      const pendingMessage: PendingComposerMessage = {
        localId: `local-pending-${nextPendingId}`,
        role: "user",
        text,
        pending: true,
      };

      state = {
        ...state,
        draft: "",
        sending: true,
        pendingBySessionId: {
          ...state.pendingBySessionId,
          [sessionId]: [...(state.pendingBySessionId[sessionId] ?? []), pendingMessage],
        },
      };
      emit();

      try {
        const response = await api.sendMessage(sessionId, text);
        state = {
          ...state,
          sending: false,
        };
        emit();
        return response;
      } catch (error) {
        state = {
          ...state,
          draft: state.draft ? state.draft : text,
          sending: false,
          pendingBySessionId: {
            ...state.pendingBySessionId,
            [sessionId]: (state.pendingBySessionId[sessionId] ?? []).filter((item) => item.localId !== pendingMessage.localId),
          },
        };
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
      if (!persistedUserTexts.length) return;

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
