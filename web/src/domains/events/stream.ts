export interface AppEventStreamEvent {
  seq?: number;
  type?: string;
  session_id?: string;
  runtime_id?: string | null;
  reason?: string;
  ts?: number;
  hints?: Record<string, unknown>;
}

export interface OpenAppEventStreamOptions {
  cursor?: number;
  onEvent(event: AppEventStreamEvent): void;
  onStateChange?(state: "connecting" | "open" | "error" | "closed"): void;
}

const EVENT_TYPES = [
  "sessions.invalidate",
  "session.live.invalidate",
  "session.workspace.invalidate",
  "notifications.invalidate",
  "attention.invalidate",
  "session.transport.invalidate",
  "stream.resync",
] as const;

function buildEventsUrl(cursor?: number) {
  if (typeof window === "undefined") {
    return "/api/events";
  }
  const url = new URL("/api/events", window.location.origin);
  if (typeof cursor === "number" && Number.isFinite(cursor) && cursor > 0) {
    url.searchParams.set("cursor", String(Math.floor(cursor)));
  }
  return `${url.pathname}${url.search}`;
}

function parseEventData(raw: string): AppEventStreamEvent | null {
  try {
    const payload = JSON.parse(raw);
    return payload && typeof payload === "object" ? payload as AppEventStreamEvent : null;
  } catch {
    return null;
  }
}

export function openAppEventStream({ cursor, onEvent, onStateChange }: OpenAppEventStreamOptions) {
  if (typeof window === "undefined" || typeof EventSource === "undefined") {
    return { close() {} };
  }

  onStateChange?.("connecting");
  const source = new EventSource(buildEventsUrl(cursor));
  const handleEvent = (event: MessageEvent<string>) => {
    const payload = parseEventData(String(event.data || ""));
    if (!payload) {
      return;
    }
    onEvent(payload);
  };

  source.onopen = () => {
    onStateChange?.("open");
  };
  source.onerror = () => {
    onStateChange?.("error");
  };
  source.onmessage = handleEvent;
  for (const eventType of EVENT_TYPES) {
    source.addEventListener(eventType, handleEvent as EventListener);
  }

  return {
    close() {
      source.close();
      onStateChange?.("closed");
    },
  };
}
