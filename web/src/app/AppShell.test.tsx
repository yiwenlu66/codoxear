import { render } from "preact";
import { flushSync } from "preact/compat";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "./providers";
import { AppShell } from "./AppShell";

const eventStreamMocks = vi.hoisted(() => ({
  openAppEventStream: vi.fn((_options: { onEvent?: (event: Record<string, unknown>) => void }) => ({
    close: vi.fn(),
  })),
}));

vi.mock("../domains/events/stream", () => ({
  openAppEventStream: eventStreamMocks.openAppEventStream,
}));

vi.mock("../lib/api", () => ({
  api: {
    getVoiceSettings: vi.fn().mockResolvedValue({
      tts_enabled_for_narration: false,
      tts_enabled_for_final_response: true,
      tts_base_url: "",
      tts_api_key: "",
      audio: { active_listener_count: 0, queue_depth: 0, segment_count: 0, stream_url: "/api/audio/live.m3u8" },
      notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" },
    }),
    saveVoiceSettings: vi.fn().mockResolvedValue({ ok: true }),
    setAudioListener: vi.fn().mockResolvedValue({ active_listener_count: 1 }),
    getNotificationsFeed: vi.fn().mockResolvedValue({ ok: true, items: [] }),
    getNotificationMessage: vi.fn().mockResolvedValue({ ok: true, notification_text: "" }),
    getNotificationSubscriptionState: vi.fn().mockResolvedValue({ ok: true, vapid_public_key: "", subscriptions: [] }),
    upsertNotificationSubscription: vi.fn().mockResolvedValue({ ok: true, vapid_public_key: "", subscriptions: [] }),
    toggleNotificationSubscription: vi.fn().mockResolvedValue({ ok: true, vapid_public_key: "", subscriptions: [] }),
    triggerTestPushNotification: vi.fn().mockResolvedValue({ ok: true, sent_count: 1, failed_count: 0, target_count: 1, notification_text: "回复完成" }),
    getFiles: vi.fn().mockResolvedValue({ ok: true, path: "", entries: [{ name: "src", path: "src", kind: "dir" }] }),
    getFileRead: vi.fn().mockResolvedValue({ ok: true, kind: "text", text: "console.log('viewer');" }),
    getGitFileVersions: vi.fn().mockResolvedValue({ ok: true, path: "src/main.tsx", base_text: "before", current_text: "after" }),
    getHarness: vi.fn().mockResolvedValue({ ok: true, enabled: true, request: "Keep going", cooldown_minutes: 15, remaining_injections: 2 }),
    saveHarness: vi.fn().mockResolvedValue({ ok: true, enabled: true, request: "Keep going", cooldown_minutes: 15, remaining_injections: 2 }),
    interruptSession: vi.fn().mockResolvedValue({ ok: true }),
    logout: vi.fn().mockResolvedValue({ ok: true }),
  },
}));

vi.mock("../components/workspace/MonacoWorkspace", () => ({
  MonacoWorkspace: (props: any) => (
    <div
      data-testid="monaco-workspace"
      data-line={props.line == null ? "" : String(props.line)}
      data-mode={props.mode}
      data-path={props.path}
    >
      {props.mode}:{props.path}
    </div>
  ),
}));

let root: HTMLDivElement | null = null;
let leakedNotification: typeof Notification | undefined;
let leakedPushManager: typeof PushManager | undefined;
let leakedServiceWorker: { register: ReturnType<typeof vi.fn> } | undefined;
let leakedPlay: ReturnType<typeof vi.fn> | undefined;
let leakedCanPlayType: ReturnType<typeof vi.fn> | undefined;

type OwnedPropertyDescriptor = {
  owner: object;
  descriptor: PropertyDescriptor;
};

function getOwnedPropertyDescriptor(target: object, key: PropertyKey): OwnedPropertyDescriptor | null {
  let current: object | null = target;
  while (current) {
    const descriptor = Object.getOwnPropertyDescriptor(current, key);
    if (descriptor) {
      return { owner: current, descriptor };
    }
    current = Object.getPrototypeOf(current);
  }
  return null;
}

function restoreProperty(target: object, key: PropertyKey, original: OwnedPropertyDescriptor | null) {
  if (!original) {
    Reflect.deleteProperty(target, key);
    return;
  }

  if (original.owner === target) {
    Object.defineProperty(target, key, original.descriptor);
    return;
  }

  Reflect.deleteProperty(target, key);
}

const originalNavigatorUserAgent = getOwnedPropertyDescriptor(window.navigator, "userAgent");
const originalNavigatorMaxTouchPoints = getOwnedPropertyDescriptor(window.navigator, "maxTouchPoints");
const originalNavigatorServiceWorker = getOwnedPropertyDescriptor(window.navigator, "serviceWorker");
const originalMediaPlay = getOwnedPropertyDescriptor(HTMLMediaElement.prototype, "play");
const originalMediaCanPlayType = getOwnedPropertyDescriptor(HTMLMediaElement.prototype, "canPlayType");

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

function setDocumentVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: state,
  });
  Object.defineProperty(document, "hidden", {
    configurable: true,
    value: state === "hidden",
  });
}

function createStaticStore<TState extends object, TActions extends Record<string, (...args: any[]) => any>>(
  state: TState,
  actions: TActions,
) {
  let currentState = state;
  const listeners = new Set<() => void>();

  return {
    getState: () => currentState,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setState: (nextState: TState) => {
      currentState = nextState;
      listeners.forEach((listener) => listener());
    },
    ...actions,
  };
}

function renderAppShell({
  activeSessionId = "sess-1",
  agentBackend = "pi",
  items,
  liveBusyBySessionId,
  messages,
  sessionUiSessionId,
  diagnostics = null,
  queue = null,
  files = [],
  requests = [],
}: {
  activeSessionId?: string | null;
  agentBackend?: string;
  items?: Array<{
    session_id: string;
    alias?: string;
    title?: string;
    first_user_message?: string;
    display_name?: string;
    agent_backend: string;
    busy: boolean;
  }>;
  liveBusyBySessionId?: Record<string, boolean>;
  messages?: Record<string, unknown[]>;
  sessionUiSessionId?: string | null;
  diagnostics?: Record<string, unknown> | null;
  queue?: Record<string, unknown> | null;
  files?: string[];
  requests?: unknown[];
} = {}) {
  const sessionItems = items ?? [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: agentBackend, busy: true }];
  const messageState = messages ?? Object.fromEntries(sessionItems.map((session) => [session.session_id, []]));
  const offsetState = Object.fromEntries(sessionItems.map((session) => [session.session_id, 0]));
  const sessionsStore = createStaticStore(
    {
      items: sessionItems,
      activeSessionId,
      loading: false,
      newSessionDefaults: null,
    },
    { refresh: vi.fn().mockResolvedValue(undefined), refreshBootstrap: vi.fn().mockResolvedValue(undefined), select: vi.fn() },
  );
  const messagesStore = createStaticStore(
    { bySessionId: messageState, offsetsBySessionId: offsetState, loading: false },
    { loadInitial: vi.fn().mockResolvedValue(undefined), poll: vi.fn().mockResolvedValue(undefined) },
  );
  const liveSessionStore = createStaticStore(
    {
      offsetsBySessionId: offsetState,
      requestsBySessionId: activeSessionId ? { [activeSessionId]: requests as any[] } : {},
      requestVersionsBySessionId: {},
      busyBySessionId: liveBusyBySessionId ?? Object.fromEntries(sessionItems.map((session) => [session.session_id, session.busy])),
      loadingBySessionId: {},
    },
    { loadInitial: vi.fn().mockResolvedValue(undefined), poll: vi.fn().mockResolvedValue(undefined) },
  );
  const composerStore = createStaticStore(
    { draft: "", sending: false },
    { setDraft: vi.fn(), submit: vi.fn() },
  );
  const sessionUiStore = createStaticStore(
    { sessionId: sessionUiSessionId ?? activeSessionId, diagnostics, queue, files, requests, loading: false },
    { refresh: vi.fn().mockResolvedValue(undefined) },
  );

  const mountNode = document.createElement("div");
  root = mountNode;
  document.body.appendChild(mountNode);
  act(() => {
    render(
      <AppProviders
        sessionsStore={sessionsStore as any}
        messagesStore={messagesStore as any}
        liveSessionStore={liveSessionStore as any}
        composerStore={composerStore as any}
        sessionUiStore={sessionUiStore as any}
      >
        <AppShell />
      </AppProviders>,
      mountNode,
    );
  });

  return { sessionsStore, messagesStore, liveSessionStore, composerStore, sessionUiStore };
}

function getRoot(): HTMLDivElement {
  if (!root) {
    throw new Error("AppShell test root is not mounted");
  }

  return root;
}

function findButtonByText(label: string) {
  return Array.from(getRoot().querySelectorAll<HTMLButtonElement>("button")).find((button) => button.textContent?.includes(label));
}

function findButtonByAriaLabel(label: string) {
  return getRoot().querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);
}

function requireButtonByText(label: string) {
  const button = findButtonByText(label);

  if (!button) {
    throw new Error(`Button containing \"${label}\" not found`);
  }

  return button;
}

function getToolbarTodoAnchor() {
  return getRoot().querySelector(".conversationToolbar .todoToolbarAnchor");
}

function getToolbarTodoButton() {
  return getRoot().querySelector<HTMLButtonElement>(".conversationToolbar .toolbarButton.todoToggle");
}

describe("AppShell", () => {
  afterEach(() => {
    localStorage.clear();
    document.documentElement.dataset.theme = "light";
    document.documentElement.style.colorScheme = "light";
    setDocumentVisibility("visible");
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
    vi.useRealTimers();
    vi.unstubAllGlobals();
    restoreProperty(window.navigator, "userAgent", originalNavigatorUserAgent);
    restoreProperty(window.navigator, "maxTouchPoints", originalNavigatorMaxTouchPoints);
    restoreProperty(window.navigator, "serviceWorker", originalNavigatorServiceWorker);
    restoreProperty(HTMLMediaElement.prototype, "play", originalMediaPlay);
    restoreProperty(HTMLMediaElement.prototype, "canPlayType", originalMediaCanPlayType);
    vi.clearAllMocks();
  });

  it("keeps one SSE connection while session state changes", async () => {
    const { liveSessionStore, sessionsStore } = renderAppShell({
      activeSessionId: "sess-1",
      items: [
        { session_id: "sess-1", alias: "Alpha", agent_backend: "pi", busy: true },
        { session_id: "sess-2", alias: "Beta", agent_backend: "pi", busy: false },
      ],
    });

    await flush();
    expect(eventStreamMocks.openAppEventStream).toHaveBeenCalledTimes(1);

    act(() => {
      sessionsStore.setState({
        ...sessionsStore.getState(),
        activeSessionId: "sess-2",
        items: [
          { session_id: "sess-1", alias: "Alpha", agent_backend: "pi", busy: false },
          { session_id: "sess-2", alias: "Beta", agent_backend: "pi", busy: true },
        ],
      });
    });
    await flush();

    act(() => {
      sessionsStore.setState({
        ...sessionsStore.getState(),
        items: [
          { session_id: "sess-2", alias: "Beta", agent_backend: "pi", busy: true },
          { session_id: "sess-1", alias: "Alpha", agent_backend: "pi", busy: false },
        ],
      });
    });
    await flush();

    expect(eventStreamMocks.openAppEventStream).toHaveBeenCalledTimes(1);

    const streamOptions = eventStreamMocks.openAppEventStream.mock.calls[0]?.[0];
    expect(streamOptions).toBeTruthy();
    if (!streamOptions?.onEvent) {
      throw new Error("SSE handler missing");
    }
    const onStreamEvent = streamOptions.onEvent;
    act(() => {
      onStreamEvent({
        type: "session.live.invalidate",
        session_id: "sess-2",
        runtime_id: null,
      });
    });
    await flush();

    expect(liveSessionStore.poll).toHaveBeenCalledWith("sess-2");
  });

  it("does not force a sessions refresh for transport-only invalidations", async () => {
    const { liveSessionStore, sessionsStore } = renderAppShell({
      activeSessionId: "sess-1",
      items: [{ session_id: "sess-1", alias: "Alpha", agent_backend: "pi", busy: true }],
    });

    await flush();
    vi.mocked(sessionsStore.refresh).mockClear();
    vi.mocked(liveSessionStore.poll).mockClear();

    const streamOptions = eventStreamMocks.openAppEventStream.mock.calls[0]?.[0];
    expect(streamOptions).toBeTruthy();
    if (!streamOptions?.onEvent) {
      throw new Error("SSE handler missing");
    }
    const onStreamEvent = streamOptions.onEvent;
    act(() => {
      onStreamEvent({
        type: "session.transport.invalidate",
        session_id: "sess-1",
        runtime_id: null,
      });
    });
    await flush();

    expect(liveSessionStore.poll).toHaveBeenCalledWith("sess-1");
    expect(sessionsStore.refresh).not.toHaveBeenCalled();
  });

  it("renders a two-part shell with sessions rail and conversation column", () => {
    renderAppShell({ activeSessionId: null, diagnostics: null });

    expect(getRoot().querySelector("[data-testid='app-shell']")).not.toBeNull();
    expect(getRoot().querySelector(".sidebarColumn.desktopSessionsRail")).not.toBeNull();
    expect(getRoot().querySelector(".conversationColumn")).not.toBeNull();
    expect(getRoot().querySelector("[data-testid='mobile-sessions-sheet']")).not.toBeNull();
    expect(getRoot().querySelector("[data-testid='workspace-rail']")).toBeNull();
    expect(getRoot().textContent).toContain("New session");
    expect(getRoot().textContent).toContain("Help");
    expect(getRoot().textContent).toContain("Settings");
    expect(getRoot().textContent).toContain("Log out");
    expect(getRoot().textContent).toContain("No session selected");
    expect(getRoot().querySelector(".mobileSheetTrigger")).toBeNull();
    expect(getRoot().querySelector(".mobileToolsTrigger")).toBeNull();
    expect(findButtonByAriaLabel("Files")).not.toBeNull();
    expect(findButtonByAriaLabel("Workspace")).not.toBeNull();
    expect(findButtonByAriaLabel("Harness mode")).not.toBeNull();
    expect(findButtonByAriaLabel("Interrupt (Esc)")).not.toBeNull();
    const notificationsButton = getRoot().querySelector<HTMLButtonElement>('[aria-label="Notifications off"]');
    const announcementsButton = getRoot().querySelector<HTMLButtonElement>('[aria-label="Announcements off"]');
    expect(notificationsButton).not.toBeNull();
    expect(announcementsButton).not.toBeNull();
    expect(notificationsButton?.title).toBe("Notifications off");
    expect(announcementsButton?.title).toBe("Announcements off");
    expect(notificationsButton?.querySelector("svg")).not.toBeNull();
    expect(announcementsButton?.querySelector("svg")).not.toBeNull();
  });

  it("renders bottom navigation on narrow viewports and defaults to the sessions page without an active session", () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === "(max-width: 880px)",
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn().mockReturnValue(false),
      })),
    });

    try {
      renderAppShell({ activeSessionId: null, diagnostics: null });
      expect(getRoot().querySelector('[data-testid="mobile-shell"]')).not.toBeNull();
      expect(findButtonByText("Sessions")).not.toBeNull();
      expect(findButtonByText("Read")).not.toBeNull();
      expect(findButtonByText("Chat")).not.toBeNull();
      expect(findButtonByText("Tools")).not.toBeNull();
      expect(findButtonByAriaLabel("Conversation tools")).toBeNull();
      expect(getRoot().querySelector('[data-testid="sessions-surface"]')).not.toBeNull();
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        value: originalMatchMedia,
      });
    }
  });

  it("renders direct toolbar icon actions on desktop without a grouped tools trigger", async () => {
    renderAppShell({ diagnostics: { status: "ok" } });
    await flush();

    expect(getRoot().querySelector(".mobileToolsTrigger")).toBeNull();
    expect(findButtonByAriaLabel("Files")?.querySelector("svg")).not.toBeNull();
    expect(findButtonByAriaLabel("Workspace")?.querySelector("svg")).not.toBeNull();
    expect(findButtonByAriaLabel("Harness mode")?.querySelector("svg")).not.toBeNull();
    expect(findButtonByAriaLabel("Interrupt (Esc)")?.querySelector("svg")).not.toBeNull();
  });

  it("defaults to read mode on narrow viewports when a session is active", async () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === "(max-width: 880px)",
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn().mockReturnValue(false),
      })),
    });

    try {
      renderAppShell({ diagnostics: { status: "ok" } });
      await flush();

      expect(getRoot().textContent).toContain("Read");
      expect(findButtonByAriaLabel("Conversation tools")).toBeNull();
      expect(getRoot().querySelector("[data-testid='composer-card']")).toBeNull();
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        value: originalMatchMedia,
      });
    }
  });

  it("switches to the tools page on narrow viewports", async () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === "(max-width: 880px)",
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn().mockReturnValue(false),
      })),
    });

    try {
      renderAppShell({ diagnostics: { status: "ok" } });
      await flush();

      expect(getRoot().textContent).toContain("Read");
      expect(findButtonByAriaLabel("Conversation tools")).toBeNull();

      act(() => {
        findButtonByText("Tools")?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      });
      await flush();

      expect(getRoot().textContent).toContain("Secondary actions");
      expect(findButtonByText("Files")).not.toBeNull();
      expect(findButtonByText("Workspace")).not.toBeNull();
      expect(findButtonByText("Harness")).not.toBeNull();
      expect(findButtonByText("Settings")).not.toBeNull();
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        value: originalMatchMedia,
      });
    }
  });

  it("opens workspace details in a dialog from the toolbar", async () => {
    renderAppShell({ diagnostics: { status: "ok" } });
    await flush();

    const button = findButtonByAriaLabel("Workspace");
    expect(button).not.toBeNull();

    act(() => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    const dialog = getRoot().querySelector("[data-testid='workspace-dialog']");
    expect(dialog).not.toBeNull();
    expect(dialog?.querySelector("[data-testid='workspace-card']")).not.toBeNull();
    expect(dialog?.textContent).toContain("Diagnostics");
  });

  it("opens workspace details from the tools page on narrow viewports", async () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === "(max-width: 880px)",
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn().mockReturnValue(false),
      })),
    });

    try {
      renderAppShell({ diagnostics: { status: "ok" } });
      await flush();

      act(() => {
        findButtonByText("Tools")?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      });
      await flush();

      const button = findButtonByText("Workspace");
      expect(button).not.toBeNull();
      act(() => {
        button?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      });
      await flush();

      const dialog = getRoot().querySelector("[data-testid='workspace-dialog']");
      expect(dialog).not.toBeNull();
      expect(getRoot().querySelector('[role="dialog"][aria-labelledby="mobile-workspace-title"]')).toBeNull();
      expect(dialog?.textContent).toContain("Workspace");
      expect(dialog?.textContent).toContain("Diagnostics");
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        value: originalMatchMedia,
      });
    }
  });

  it("switches to compact chat mode on narrow viewports without queue or attach actions", async () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === "(max-width: 880px)",
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn().mockReturnValue(false),
      })),
    });

    try {
      renderAppShell({ diagnostics: { status: "ok" } });
      await flush();

      act(() => {
        findButtonByText("Chat")?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      });
      await flush();

      expect(getRoot().querySelector("[data-testid='composer-card']")).not.toBeNull();
      expect(getRoot().textContent).toContain("Chat");
      expect(findButtonByText("Queue")).toBeUndefined();
      expect(findButtonByAriaLabel("Attach file")).toBeNull();
      expect(findButtonByAriaLabel("Cancel current loop")).not.toBeNull();
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        value: originalMatchMedia,
      });
    }
  });

  it("switches back to the sessions page from bottom navigation on narrow viewports", async () => {
    const originalMatchMedia = window.matchMedia;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === "(max-width: 880px)",
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn().mockReturnValue(false),
      })),
    });

    try {
      renderAppShell({ diagnostics: { status: "ok" } });
      await flush();

      expect(getRoot().textContent).toContain("Read");
      act(() => {
        findButtonByText("Sessions")?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      });
      await flush();

      expect(getRoot().querySelector('[data-testid="sessions-surface"]')).not.toBeNull();
      expect(getRoot().textContent).toContain("Continue where you left off");
    } finally {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        value: originalMatchMedia,
      });
    }
  });

  it("opens the todo viewer from the toolbar", async () => {
    renderAppShell({
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [
            { title: "Inspect session shortlist", status: "in-progress", description: "Check the active queue" },
            { title: "Draft focus metadata", status: "not-started" },
          ],
        },
      },
    });
    await flush();

    const button = findButtonByAriaLabel("Todo list");
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(getRoot().textContent).toContain("Todo list");
    expect(getRoot().textContent).toContain("1/2 completed");
    expect(getRoot().textContent).toContain("Inspect session shortlist");
  });

  it("opens the file viewer from the toolbar and requests the root directory listing", async () => {
    const { api } = await import("../lib/api");
    renderAppShell({ files: [] });
    await flush();

    const button = findButtonByAriaLabel("Files");
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();
    await flush();

    expect(api.getFiles).toHaveBeenCalledWith("sess-1", undefined, expect.any(AbortSignal));
    expect(getRoot().textContent).toContain("File viewer");
    expect(getRoot().textContent).toContain("Choose a file from the session.");
  });

  it("opens harness mode from the toolbar and saves harness settings", async () => {
    const { api } = await import("../lib/api");
    renderAppShell();
    await flush();

    const button = findButtonByAriaLabel("Harness mode");
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(api.getHarness).toHaveBeenCalledWith("sess-1");
    expect(getRoot().textContent).toContain("Harness mode");
    expect(getRoot().textContent).toContain("Additional request");

    const saveButton = findButtonByText("Save");
    expect(saveButton).not.toBeNull();
    act(() => {
      saveButton!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(api.saveHarness).toHaveBeenCalledWith("sess-1", expect.objectContaining({ enabled: true, request: "Keep going" }));
  });

  it("shows persisted title in the active header before first user message", async () => {
    renderAppShell({ items: [{ session_id: "sess-1", title: "Release checklist", first_user_message: "先整理一下今晚要发的内容", agent_backend: "pi", busy: false }] });
    await flush();

    expect(getRoot().querySelector(".conversationTitle")?.textContent).toContain("Release checklist");
    expect(getRoot().querySelector(".conversationTitle")?.textContent).not.toContain("先整理一下今晚要发的内容");
  });

  it("interrupts the active busy session from the toolbar", async () => {
    const { api } = await import("../lib/api");
    renderAppShell({ items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true }] });
    await flush();

    const button = findButtonByAriaLabel("Interrupt (Esc)");
    expect(button).not.toBeNull();
    expect(button?.disabled).toBe(false);

    act(() => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(api.interruptSession).toHaveBeenCalledWith("sess-1");
  });

  it("interrupts the active busy session when Escape is pressed", async () => {
    const { api } = await import("../lib/api");
    renderAppShell({ items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true }] });
    await flush();

    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
    });
    await flush();

    expect(api.interruptSession).toHaveBeenCalledWith("sess-1");
  });

  it("keeps the interrupt toolbar action when live session state is still busy", async () => {
    renderAppShell({
      items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: false }],
      liveBusyBySessionId: { "sess-1": true },
    });
    await flush();

    const button = findButtonByAriaLabel("Interrupt (Esc)");
    expect(button).not.toBeNull();
    expect(button?.disabled).toBe(false);
  });

  it("hides the interrupt toolbar action when the active session is idle", async () => {
    renderAppShell({ items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: false }] });
    await flush();

    expect(findButtonByAriaLabel("Interrupt (Esc)")).toBeNull();
  });

  it("opens announcement settings when announcements are enabled without credentials", async () => {
    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Announcements off"]');
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(getRoot().textContent).toContain("OpenAI-compatible API base URL");
    expect(getRoot().textContent).toContain("OpenAI-compatible API key");
    expect(getRoot().textContent).toContain("Announce narration messages");
    expect(getRoot().textContent).toContain("Press Enter to send");
    expect(getRoot().textContent).toContain("Play a short beep when the assistant finishes a reply");
    expect(getRoot().textContent).toContain("Theme");
    expect(getRoot().textContent).toContain("Paper-like surfaces with cobalt markdown accents.");
    expect(getRoot().textContent).toContain("Ink surfaces with brighter markdown contrast for long sessions.");
  });

  it("persists the selected theme mode from settings", async () => {
    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Announcements off"]');
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    const darkRadio = Array.from(getRoot().querySelectorAll<HTMLInputElement>('input[type="radio"]')).find(
      (input) => input.nextElementSibling?.textContent?.includes("Dark"),
    );
    expect(darkRadio).not.toBeUndefined();

    act(() => {
      darkRadio!.click();
    });
    await flush();

    expect(localStorage.getItem("codoxear.themeMode")).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");
  });

  it("persists the reply sound toggle from settings", async () => {
    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Announcements off"]');
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    const checkbox = Array.from(getRoot().querySelectorAll<HTMLInputElement>('input[type="checkbox"]')).find(
      (input) => input.nextElementSibling?.textContent?.includes("assistant finishes a reply"),
    );
    expect(checkbox).not.toBeUndefined();
    expect(checkbox?.checked).toBe(true);

    act(() => {
      checkbox!.checked = false;
      checkbox!.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await flush();

    expect(localStorage.getItem("codoxear.replySoundEnabled")).toBe("0");
  });

  it("requests a fixed-text mobile test push from settings", async () => {
    const { api } = await import("../lib/api");
    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Announcements off"]');
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    const pushButton = Array.from(getRoot().querySelectorAll<HTMLButtonElement>("button")).find(
      (item) => item.textContent?.includes("Test Push"),
    );
    expect(pushButton).not.toBeUndefined();

    await act(async () => {
      pushButton!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(api.triggerTestPushNotification).toHaveBeenCalled();
    expect(getRoot().textContent).toContain("Test push sent to 1 device");
  });

  it("sends announcement listener heartbeats when announcements are enabled", async () => {
    const { api } = await import("../lib/api");
    localStorage.setItem("codoxear.announcementEnabled", "1");

    renderAppShell();
    await flush();

    expect(api.setAudioListener).toHaveBeenCalledWith(expect.any(String), true);
  });

  it("toggles notifications on for supported browsers", async () => {
    const requestPermission = vi.fn().mockResolvedValue("granted");
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "default";
      static requestPermission = requestPermission;
      constructor() {}
    } as any);

    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Notifications off"]');
    expect(button).not.toBeNull();

    act(() => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(requestPermission).toHaveBeenCalled();
    const enabledButton = getRoot().querySelector<HTMLButtonElement>('[aria-label="Notifications on"]');
    expect(enabledButton).not.toBeNull();
    expect(enabledButton?.title).toBe("Notifications on");
  });

  it("polls the notification feed and shows desktop notifications for new items", async () => {
    vi.useFakeTimers();
    const { api } = await import("../lib/api");
    const notificationSpy = vi.fn();
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor(title: string, options?: NotificationOptions) {
        notificationSpy(title, options);
      }
    } as any);
    localStorage.setItem("codoxear.notificationEnabled", "1");
    vi.mocked(api.getNotificationsFeed)
      .mockResolvedValueOnce({ ok: true, items: [] })
      .mockResolvedValueOnce({
        ok: true,
        items: [{ message_id: "msg-1", session_display_name: "Legacy shell", notification_text: "finished task", updated_ts: 42 }],
      });

    renderAppShell();
    await flush();

    expect(api.getNotificationsFeed).toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(5000);
      await Promise.resolve();
    });
    await flush();

    expect(notificationSpy).toHaveBeenCalledWith("Legacy shell", expect.objectContaining({ body: "finished task" }));
    vi.useRealTimers();
  });

  it("pauses notification feed polling while hidden and catches up on resume", async () => {
    vi.useFakeTimers();
    const { api } = await import("../lib/api");
    const notificationSpy = vi.fn();
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor(title: string, options?: NotificationOptions) {
        notificationSpy(title, options);
      }
    } as any);
    localStorage.setItem("codoxear.notificationEnabled", "1");
    setDocumentVisibility("hidden");
    vi.mocked(api.getNotificationsFeed)
      .mockResolvedValueOnce({ ok: true, items: [] })
      .mockResolvedValueOnce({
        ok: true,
        items: [{ message_id: "msg-hidden", session_display_name: "Legacy shell", notification_text: "resumed", updated_ts: 43 }],
      });

    renderAppShell();
    await flush();

    expect(api.getNotificationsFeed).not.toHaveBeenCalled();

    await act(async () => {
      setDocumentVisibility("visible");
      document.dispatchEvent(new Event("visibilitychange"));
      await flush();
    });

    expect(api.getNotificationsFeed).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(5000);
      await Promise.resolve();
    });
    await flush();

    expect(notificationSpy).toHaveBeenCalledWith("Legacy shell", expect.objectContaining({ body: "resumed" }));
    vi.useRealTimers();
  });

  it("plays the reply beep for a background session when the notification feed reports a completed reply", async () => {
    vi.useFakeTimers();
    const { api } = await import("../lib/api");
    const oscillatorStart = vi.fn();
    const oscillatorStop = vi.fn();
    const close = vi.fn().mockResolvedValue(undefined);

    vi.stubGlobal("AudioContext", class AudioContextMock {
      currentTime = 0;
      destination = {};

      createOscillator() {
        return {
          connect: vi.fn(),
          type: "triangle",
          frequency: { setValueAtTime: vi.fn() },
          start: oscillatorStart,
          stop: oscillatorStop,
          onended: null,
        };
      }

      createGain() {
        return {
          connect: vi.fn(),
          gain: {
            setValueAtTime: vi.fn(),
            exponentialRampToValueAtTime: vi.fn(),
          },
        };
      }

      close = close;
    } as any);

    vi.mocked(api.getNotificationsFeed)
      .mockResolvedValueOnce({ ok: true, items: [] })
      .mockResolvedValueOnce({
        ok: true,
        items: [{ message_id: "msg-bg-1", session_display_name: "Background shell", notification_text: "done", updated_ts: 42 }],
      });

    renderAppShell({
      items: [
        { session_id: "sess-1", alias: "Active shell", agent_backend: "pi", busy: false },
        { session_id: "sess-2", alias: "Background shell", agent_backend: "pi", busy: false },
      ],
      activeSessionId: "sess-1",
      messages: {
        "sess-1": [],
        "sess-2": [],
      },
    });
    await flush();

    await act(async () => {
      vi.advanceTimersByTime(5000);
      await Promise.resolve();
    });
    await flush();

    expect(oscillatorStart).toHaveBeenCalledTimes(1);
    expect(oscillatorStop).toHaveBeenCalledTimes(1);
  });

  it("plays the reply beep for a background busy session when polling receives a final response", async () => {
    vi.useFakeTimers();
    const oscillatorStart = vi.fn();
    const oscillatorStop = vi.fn();
    const close = vi.fn().mockResolvedValue(undefined);

    vi.stubGlobal("AudioContext", class AudioContextMock {
      currentTime = 0;
      destination = {};

      createOscillator() {
        return {
          connect: vi.fn(),
          type: "triangle",
          frequency: { setValueAtTime: vi.fn() },
          start: oscillatorStart,
          stop: oscillatorStop,
          onended: null,
        };
      }

      createGain() {
        return {
          connect: vi.fn(),
          gain: {
            setValueAtTime: vi.fn(),
            exponentialRampToValueAtTime: vi.fn(),
          },
        };
      }

      close = close;
    } as any);

    const { liveSessionStore, messagesStore } = renderAppShell({
      items: [
        { session_id: "sess-1", alias: "Active shell", agent_backend: "pi", busy: true },
        { session_id: "sess-2", alias: "Background shell", agent_backend: "pi", busy: true },
      ],
      activeSessionId: "sess-1",
      messages: {
        "sess-1": [],
        "sess-2": [],
      },
    });
    await flush();

    (liveSessionStore as any).poll = vi.fn().mockImplementation(async (sessionId: string) => {
      if (sessionId !== "sess-2") {
        return undefined;
      }
      const state = (messagesStore as any).getState();
      (messagesStore as any).setState({
        ...state,
        bySessionId: {
          ...state.bySessionId,
          "sess-2": [{ role: "assistant", message_class: "final_response", message_id: "msg-bg-poll", text: "done" }],
        },
      });
      return undefined;
    });

    await act(async () => {
      vi.advanceTimersByTime(5000);
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect((liveSessionStore as any).poll).toHaveBeenCalledWith("sess-2");
    expect(oscillatorStart).toHaveBeenCalledTimes(1);
    expect(oscillatorStop).toHaveBeenCalledTimes(1);
  });

  it("does not replay the reply beep when activating a session whose completed reply already beeped in background", async () => {
    vi.useFakeTimers();
    const oscillatorStart = vi.fn();
    const oscillatorStop = vi.fn();
    const close = vi.fn().mockResolvedValue(undefined);

    vi.stubGlobal("AudioContext", class AudioContextMock {
      currentTime = 0;
      destination = {};

      createOscillator() {
        return {
          connect: vi.fn(),
          type: "triangle",
          frequency: { setValueAtTime: vi.fn() },
          start: oscillatorStart,
          stop: oscillatorStop,
          onended: null,
        };
      }

      createGain() {
        return {
          connect: vi.fn(),
          gain: {
            setValueAtTime: vi.fn(),
            exponentialRampToValueAtTime: vi.fn(),
          },
        };
      }

      close = close;
    } as any);

    const { liveSessionStore, messagesStore, sessionsStore } = renderAppShell({
      items: [
        { session_id: "sess-1", alias: "Active shell", agent_backend: "pi", busy: true },
        { session_id: "sess-2", alias: "Background shell", agent_backend: "pi", busy: true },
      ],
      activeSessionId: "sess-1",
      messages: {
        "sess-1": [],
        "sess-2": [],
      },
    });
    await flush();

    (liveSessionStore as any).poll = vi.fn().mockImplementation(async (sessionId: string) => {
      if (sessionId !== "sess-2") {
        return undefined;
      }
      const state = (messagesStore as any).getState();
      (messagesStore as any).setState({
        ...state,
        bySessionId: {
          ...state.bySessionId,
          "sess-2": [{ role: "assistant", message_class: "final_response", message_id: "msg-bg-poll-1", text: "ok" }],
        },
      });
      return undefined;
    });

    await act(async () => {
      vi.advanceTimersByTime(5000);
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(oscillatorStart).toHaveBeenCalledTimes(1);

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [
          { session_id: "sess-1", alias: "Active shell", agent_backend: "pi", busy: false },
          { session_id: "sess-2", alias: "Background shell", agent_backend: "pi", busy: false },
        ],
        activeSessionId: "sess-2",
        loading: false,
        newSessionDefaults: null,
      });
    });
    flushSync(() => {
      const state = (messagesStore as any).getState();
      (messagesStore as any).setState({
        ...state,
        bySessionId: {
          ...state.bySessionId,
          "sess-2": [{ role: "assistant", message_class: "final_response", message_id: "msg-bg-init-2", text: "ok" }],
        },
      });
    });
    await flush();
    await flush();

    expect(oscillatorStart).toHaveBeenCalledTimes(1);
    expect(oscillatorStop).toHaveBeenCalledTimes(1);
  });

  it("does not play the reply beep for final responses loaded during session activation", async () => {
    const oscillatorStart = vi.fn();
    const oscillatorStop = vi.fn();
    const close = vi.fn().mockResolvedValue(undefined);

    vi.stubGlobal("AudioContext", class AudioContextMock {
      currentTime = 0;
      destination = {};

      createOscillator() {
        return {
          connect: vi.fn(),
          type: "triangle",
          frequency: { setValueAtTime: vi.fn() },
          start: oscillatorStart,
          stop: oscillatorStop,
          onended: null,
        };
      }

      createGain() {
        return {
          connect: vi.fn(),
          gain: {
            setValueAtTime: vi.fn(),
            exponentialRampToValueAtTime: vi.fn(),
          },
        };
      }

      close = close;
    } as any);

    const { liveSessionStore, messagesStore, sessionsStore } = renderAppShell({
      items: [
        { session_id: "sess-1", alias: "Active shell", agent_backend: "pi", busy: false },
        { session_id: "sess-2", alias: "Background shell", agent_backend: "pi", busy: false },
      ],
      activeSessionId: "sess-1",
      messages: {
        "sess-1": [],
        "sess-2": [],
      },
    });
    await flush();

    (liveSessionStore as any).loadInitial = vi.fn().mockImplementation(async (sessionId: string) => {
      if (sessionId !== "sess-2") {
        return undefined;
      }
      const state = (messagesStore as any).getState();
      (messagesStore as any).setState({
        ...state,
        bySessionId: {
          ...state.bySessionId,
          "sess-2": [{ role: "assistant", message_class: "final_response", message_id: "msg-activate-1", text: "loaded on activate" }],
        },
      });
      return undefined;
    });

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [
          { session_id: "sess-1", alias: "Active shell", agent_backend: "pi", busy: false },
          { session_id: "sess-2", alias: "Background shell", agent_backend: "pi", busy: false },
        ],
        activeSessionId: "sess-2",
        loading: false,
        newSessionDefaults: null,
      });
    });
    await flush();
    await flush();

    expect(oscillatorStart).not.toHaveBeenCalled();
    expect(oscillatorStop).not.toHaveBeenCalled();
  });

  it("shows a desktop notification immediately for a live final response event", async () => {
    const notificationSpy = vi.fn();
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor(title: string, options?: NotificationOptions) {
        notificationSpy(title, options);
      }
    } as any);
    localStorage.setItem("codoxear.notificationEnabled", "1");

    renderAppShell({
      agentBackend: "codex",
      messages: {
        "sess-1": [{ role: "assistant", message_class: "final_response", message_id: "msg-live", notification_text: "done now" }],
      },
    });
    await flush();

    expect(notificationSpy).toHaveBeenCalledWith("Legacy shell", expect.objectContaining({ body: "done now" }));
  });

  it("resolves notification text for a live final response event by message id", async () => {
    const { api } = await import("../lib/api");
    const notificationSpy = vi.fn();
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor(title: string, options?: NotificationOptions) {
        notificationSpy(title, options);
      }
    } as any);
    localStorage.setItem("codoxear.notificationEnabled", "1");
    vi.mocked(api.getNotificationMessage).mockResolvedValue({
      ok: true,
      notification_text: "resolved summary",
      summary_status: "sent",
    } as any);

    renderAppShell({
      agentBackend: "codex",
      messages: {
        "sess-1": [{ role: "assistant", message_class: "final_response", message_id: "msg-live" }],
      },
    });
    await flush();
    await flush();

    expect(api.getNotificationMessage).toHaveBeenCalledWith("msg-live");
    expect(notificationSpy).toHaveBeenCalledWith("Legacy shell", expect.objectContaining({ body: "resolved summary" }));
  });

  it("backs off notification message lookups for unresolved final responses", async () => {
    vi.useFakeTimers();
    const { api } = await import("../lib/api");
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor() {}
    } as any);
    localStorage.setItem("codoxear.notificationEnabled", "1");

    const finalResponseEvent = { role: "assistant", message_class: "final_response", message_id: "msg-retry" };
    let resolveNotificationLookup: ((value: unknown) => void) | null = null;
    vi.mocked(api.getNotificationMessage).mockImplementation(() => new Promise((resolve) => {
      resolveNotificationLookup = resolve;
    }) as any);

    const { sessionsStore } = renderAppShell({
      agentBackend: "codex",
      messages: {
        "sess-1": [finalResponseEvent],
      },
    });
    await flush();
    await flush();

    expect(api.getNotificationMessage).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveNotificationLookup?.({
        ok: true,
        notification_text: "",
        summary_status: "pending",
      });
      await Promise.resolve();
      await Promise.resolve();
    });

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [{ session_id: "sess-1", alias: "Legacy shell retry", agent_backend: "codex", busy: true }],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      });
    });
    await flush();
    await flush();

    expect(api.getNotificationMessage).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.setSystemTime(Date.now() + 15000);
      vi.advanceTimersByTime(15000);
      await Promise.resolve();
    });

    vi.mocked(api.getNotificationMessage).mockResolvedValue({
      ok: true,
      notification_text: "",
      summary_status: "pending",
    } as any);

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [{ session_id: "sess-1", alias: "Legacy shell retry again", agent_backend: "codex", busy: true }],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      });
    });
    await flush();
    await flush();

    expect(api.getNotificationMessage).toHaveBeenCalledTimes(2);
  });

  it("enables mobile push notifications through a service worker subscription", async () => {
    const { api } = await import("../lib/api");
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor() {}
    } as any);
    vi.stubGlobal("PushManager", class PushManagerMock {} as any);
    Object.defineProperty(window.navigator, "userAgent", {
      configurable: true,
      value: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
    });
    Object.defineProperty(window.navigator, "maxTouchPoints", {
      configurable: true,
      value: 5,
    });

    const subscriptionJson = {
      endpoint: "https://push.example.test/sub/1",
      keys: { p256dh: "p256", auth: "auth" },
    };
    const subscribe = vi.fn().mockResolvedValue({
      endpoint: subscriptionJson.endpoint,
      toJSON: () => subscriptionJson,
    });
    const getSubscription = vi.fn().mockResolvedValue(null);
    const register = vi.fn().mockResolvedValue({ pushManager: { getSubscription, subscribe } });
    Object.defineProperty(window.navigator, "serviceWorker", {
      configurable: true,
      value: { register },
    });

    vi.mocked(api.getVoiceSettings).mockResolvedValueOnce({
      tts_enabled_for_narration: false,
      tts_enabled_for_final_response: true,
      tts_base_url: "https://example.test/v1",
      tts_api_key: "secret",
      audio: { active_listener_count: 0, queue_depth: 0, segment_count: 0, stream_url: "/api/audio/live.m3u8" },
      notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "ZmFrZS1rZXk" },
    } as any);
    vi.mocked(api.getNotificationSubscriptionState).mockResolvedValue({ ok: true, vapid_public_key: "ZmFrZS1rZXk", subscriptions: [] } as any);
    vi.mocked(api.upsertNotificationSubscription).mockResolvedValue({
      ok: true,
      vapid_public_key: "ZmFrZS1rZXk",
      subscriptions: [{ endpoint: subscriptionJson.endpoint, notifications_enabled: true, device_class: "mobile" }],
    } as any);

    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Notifications off"]');
    expect(button).not.toBeNull();

    await act(async () => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();
    await flush();

    expect(register).toHaveBeenCalledWith("/service-worker.js");
    expect(subscribe).toHaveBeenCalled();
    expect(api.upsertNotificationSubscription).toHaveBeenCalledWith(expect.objectContaining({
      subscription: subscriptionJson,
      device_class: "mobile",
    }));
    expect(getRoot().querySelector('[aria-label="Notifications on (push)"]')).not.toBeNull();
  });

  it("replaces stale mobile push subscriptions before enabling notifications", async () => {
    const { api } = await import("../lib/api");
    vi.stubGlobal("Notification", class NotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor() {}
    } as any);
    vi.stubGlobal("PushManager", class PushManagerMock {} as any);
    Object.defineProperty(window.navigator, "userAgent", {
      configurable: true,
      value: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
    });
    Object.defineProperty(window.navigator, "maxTouchPoints", {
      configurable: true,
      value: 5,
    });

    const staleSubscription = {
      endpoint: "https://permanently-removed.invalid/fcm/send/stale-token",
      unsubscribe: vi.fn().mockResolvedValue(true),
      toJSON: () => ({
        endpoint: "https://permanently-removed.invalid/fcm/send/stale-token",
        keys: { p256dh: "old-p256", auth: "old-auth" },
      }),
    };
    const freshSubscriptionJson = {
      endpoint: "https://push.example.test/sub/2",
      keys: { p256dh: "p256-new", auth: "auth-new" },
    };
    const freshSubscription = {
      endpoint: freshSubscriptionJson.endpoint,
      toJSON: () => freshSubscriptionJson,
    };
    const subscribe = vi.fn().mockResolvedValue(freshSubscription);
    const getSubscription = vi.fn().mockResolvedValue(staleSubscription);
    const register = vi.fn().mockResolvedValue({ pushManager: { getSubscription, subscribe } });
    Object.defineProperty(window.navigator, "serviceWorker", {
      configurable: true,
      value: { register },
    });

    vi.mocked(api.getVoiceSettings).mockResolvedValueOnce({
      tts_enabled_for_narration: false,
      tts_enabled_for_final_response: true,
      tts_base_url: "https://example.test/v1",
      tts_api_key: "secret",
      audio: { active_listener_count: 0, queue_depth: 0, segment_count: 0, stream_url: "/api/audio/live.m3u8" },
      notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "ZmFrZS1rZXk" },
    } as any);
    vi.mocked(api.getNotificationSubscriptionState).mockResolvedValue({ ok: true, vapid_public_key: "ZmFrZS1rZXk", subscriptions: [] } as any);
    vi.mocked(api.upsertNotificationSubscription).mockResolvedValue({
      ok: true,
      vapid_public_key: "ZmFrZS1rZXk",
      subscriptions: [{ endpoint: freshSubscriptionJson.endpoint, notifications_enabled: true, device_class: "mobile" }],
    } as any);

    renderAppShell();
    await flush();

    const button = getRoot().querySelector<HTMLButtonElement>('[aria-label="Notifications off"]');
    expect(button).not.toBeNull();

    await act(async () => {
      button!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();
    await flush();

    expect(staleSubscription.unsubscribe).toHaveBeenCalled();
    expect(subscribe).toHaveBeenCalled();
    expect(api.upsertNotificationSubscription).toHaveBeenCalledWith(expect.objectContaining({
      subscription: freshSubscriptionJson,
      device_class: "mobile",
    }));
  });

  it("autoplays announcement audio when enabled and live segments are ready", async () => {
    localStorage.setItem("codoxear.announcementEnabled", "1");
    const play = vi.fn().mockResolvedValue(undefined);
    const canPlayType = vi.fn().mockReturnValue("probably");
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: play,
    });
    Object.defineProperty(HTMLMediaElement.prototype, "canPlayType", {
      configurable: true,
      value: canPlayType,
    });
    const { api } = await import("../lib/api");
    vi.mocked(api.getVoiceSettings).mockResolvedValueOnce({
      tts_enabled_for_narration: false,
      tts_enabled_for_final_response: true,
      tts_base_url: "https://example.test/v1",
      tts_api_key: "secret",
      audio: { active_listener_count: 1, queue_depth: 0, segment_count: 2, stream_url: "/api/audio/live.m3u8" },
      notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" },
    } as any);

    renderAppShell();
    await flush();

    expect(play).toHaveBeenCalled();
  });

  it("covers browser-global mutations that should be cleaned up after each test", () => {
    class LeakedNotificationMock {
      static permission = "granted";
      static requestPermission = vi.fn().mockResolvedValue("granted");
      constructor() {}
    }
    class LeakedPushManagerMock {}

    leakedNotification = LeakedNotificationMock as any;
    leakedPushManager = LeakedPushManagerMock as any;
    leakedServiceWorker = { register: vi.fn() };
    leakedPlay = vi.fn().mockResolvedValue(undefined);
    leakedCanPlayType = vi.fn().mockReturnValue("probably");

    vi.stubGlobal("Notification", leakedNotification as any);
    vi.stubGlobal("PushManager", leakedPushManager as any);
    Object.defineProperty(window.navigator, "userAgent", {
      configurable: true,
      value: "cleanup-test-agent",
    });
    Object.defineProperty(window.navigator, "maxTouchPoints", {
      configurable: true,
      value: 7,
    });
    Object.defineProperty(window.navigator, "serviceWorker", {
      configurable: true,
      value: leakedServiceWorker,
    });
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: leakedPlay,
    });
    Object.defineProperty(HTMLMediaElement.prototype, "canPlayType", {
      configurable: true,
      value: leakedCanPlayType,
    });

    expect(window.navigator.userAgent).toBe("cleanup-test-agent");
    expect(window.navigator.maxTouchPoints).toBe(7);
  });

  it("restores browser globals after each test", () => {
    expect(globalThis.Notification).not.toBe(leakedNotification);
    expect(globalThis.PushManager).not.toBe(leakedPushManager);
    expect(window.navigator.userAgent).not.toBe("cleanup-test-agent");
    expect(window.navigator.maxTouchPoints).not.toBe(7);
    expect(window.navigator.serviceWorker).not.toBe(leakedServiceWorker);
    expect(HTMLMediaElement.prototype.play).not.toBe(leakedPlay);
    expect(HTMLMediaElement.prototype.canPlayType).not.toBe(leakedCanPlayType);
  });

  it("opens workspace diagnostics without rendering a persistent workspace rail", async () => {
    renderAppShell({ diagnostics: { status: "ok" } });

    act(() => {
      findButtonByAriaLabel("Workspace")?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(getRoot().querySelector("[data-testid='workspace-rail']")).toBeNull();
    expect(getRoot().querySelector("[data-testid='workspace-dialog']")?.textContent).toContain("Diagnostics");
  });

  it("keeps Workspace available while diagnostics are shown in the dialog", async () => {
    renderAppShell({ diagnostics: { status: "ok" } });

    const workspaceButton = findButtonByAriaLabel("Workspace");
    expect(workspaceButton).not.toBeNull();
    expect(workspaceButton?.disabled).toBe(false);

    act(() => {
      workspaceButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(getRoot().textContent).toContain("Workspace");
    expect(getRoot().querySelector("[data-testid='workspace-dialog']")?.textContent).toContain("Diagnostics");
  });

  it("does not render a Todo button in the toolbar", () => {
    renderAppShell({
      agentBackend: "pi",
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [{ title: "Shown in composer instead", status: "in-progress" }],
        },
      },
    });

    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();
    expect(getRoot().querySelector(".conversationToolbar")?.textContent).not.toContain("Todo");
  });

  it("keeps stale workspace state suppressed until session ui catches up", async () => {
    const { sessionsStore, sessionUiStore } = renderAppShell({
      activeSessionId: "sess-1",
      items: [
        { session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true },
        { session_id: "sess-2", alias: "Follow-up", agent_backend: "pi", busy: false },
      ],
      sessionUiSessionId: "sess-1",
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "Stale todo", status: "completed", description: "Should not bleed into the new session" }],
        },
      },
      requests: [{ id: "req-1", method: "confirm", title: "Old request" }],
    });
    await flush();

    act(() => {
      findButtonByAriaLabel("Workspace")?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(getRoot().querySelector("[data-testid='workspace-dialog']")).not.toBeNull();
    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [
          { session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true },
          { session_id: "sess-2", alias: "Follow-up", agent_backend: "pi", busy: false },
        ],
        activeSessionId: "sess-2",
        loading: false,
        newSessionDefaults: null,
      });
    });

    expect(getRoot().querySelector(".conversationTitle")?.textContent).toContain("Follow-up");
    expect(getRoot().textContent).not.toContain("Old request");
    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();

    expect(getRoot().querySelector("[data-testid='workspace-dialog']")?.textContent).toContain("No diagnostics available.");
    expect(getRoot().querySelector("[data-testid='workspace-dialog']")?.textContent).toContain("No pending requests");
    expect(getRoot().textContent).not.toContain("todo_snapshot");

    flushSync(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-2",
        diagnostics: {
          todo_snapshot: {
            available: true,
            error: false,
            progress_text: "2/2 completed",
            items: [{ title: "Fresh todo", status: "completed", description: "Visible after session ui catches up" }],
          },
          session_status: "ready",
        },
        queue: { items: [{ text: "Queued follow-up" }] },
        files: ["fresh-notes.md"],
        requests: [{ id: "req-2", method: "confirm", title: "Fresh request", message: "Visible again" }],
        loading: false,
      });
    });

    expect(getRoot().textContent).not.toContain("Old request");
    expect(getRoot().textContent).not.toContain("No diagnostics available.");
    expect(getRoot().textContent).not.toContain("No pending requests");
    expect(getRoot().textContent).toContain("Todo list");
    expect(getRoot().textContent).toContain("Session Status");
    expect(getRoot().textContent).toContain("Queued follow-up");
    expect(getRoot().textContent).toContain("fresh-notes.md");

    act(() => {
      requireButtonByText("UI Requests").dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    await flush();

    expect(getRoot().querySelector("[data-testid='workspace-dialog']")?.textContent).toContain("Fresh request");
    expect(getRoot().querySelector("[data-testid='workspace-dialog']")?.textContent).toContain("Visible again");
    expect(getRoot().textContent).not.toContain("Old request");
    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();
  });

  it("does not reintroduce a toolbar todo control when the active backend changes", async () => {
    const { sessionsStore } = renderAppShell({
      agentBackend: "pi",
      diagnostics: { todo_snapshot: { available: true, error: false, progress_text: "1/1 completed", items: [{ title: "Composer-owned todo", status: "completed" }] } },
    });
    await flush();

    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "codex", busy: true }],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      });
    });

    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();

    flushSync(() => {
      (sessionsStore as any).setState({
        items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true }],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      });
    });

    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();
  });

  it("does not render a toolbar todo control for non-pi sessions", () => {
    renderAppShell({
      agentBackend: "codex",
      diagnostics: { todo_snapshot: { available: true, error: false, progress_text: "1/2 complete", items: [{ title: "Hidden todo", status: "completed" }] } },
    });

    expect(getToolbarTodoAnchor()).toBeNull();
    expect(getToolbarTodoButton()).toBeNull();
  });

  it("uses the first user message as the active title when no alias is set", () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "4a145abccb9a48889dc7f3e5bed735f2", first_user_message: "我准备用 preact + vite 重构web端，请帮我出个规划", agent_backend: "pi", busy: true }],
        activeSessionId: "4a145abccb9a48889dc7f3e5bed735f2",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: vi.fn(), refreshBootstrap: vi.fn(), select: vi.fn() },
    );
    const messagesStore = createStaticStore(
      { bySessionId: { "4a145abccb9a48889dc7f3e5bed735f2": [] }, offsetsBySessionId: { "4a145abccb9a48889dc7f3e5bed735f2": 0 }, loading: false },
      { loadInitial: vi.fn(), poll: vi.fn() },
    );
    const liveSessionStore = createStaticStore(
      { offsetsBySessionId: {}, requestsBySessionId: {}, requestVersionsBySessionId: {}, busyBySessionId: {}, loadingBySessionId: {} },
      { loadInitial: vi.fn(), poll: vi.fn() },
    );
    const composerStore = createStaticStore(
      { draft: "", sending: false },
      { setDraft: vi.fn(), submit: vi.fn() },
    );
    const sessionUiStore = createStaticStore(
      { sessionId: "4a145abccb9a48889dc7f3e5bed735f2", diagnostics: null, queue: null, loading: false },
      { refresh: vi.fn() },
    );

    const mountNode = document.createElement("div");
    root = mountNode;
    document.body.appendChild(mountNode);
    render(
      <AppProviders
        sessionsStore={sessionsStore as any}
        messagesStore={messagesStore as any}
        liveSessionStore={liveSessionStore as any}
        composerStore={composerStore as any}
        sessionUiStore={sessionUiStore as any}
      >
        <AppShell />
      </AppProviders>,
      mountNode,
    );

    expect(getRoot().querySelector(".conversationTitle")?.textContent).toContain("我准备用 preact + vite 重构web端，请帮我出个规划");
  });

  it("refreshes sessions when active session polling returns 404", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "sess-1", alias: "Legacy shell", agent_backend: "pi", busy: true }],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh, refreshBootstrap: vi.fn().mockResolvedValue(undefined), select: vi.fn() },
    );
    const messagesStore = createStaticStore(
      { bySessionId: { "sess-1": [] }, offsetsBySessionId: { "sess-1": 0 }, loading: false },
      { loadInitial: vi.fn().mockResolvedValue(undefined), poll: vi.fn().mockResolvedValue(undefined) },
    );
    const liveSessionStore = createStaticStore(
      { offsetsBySessionId: {}, requestsBySessionId: {}, requestVersionsBySessionId: {}, busyBySessionId: {}, loadingBySessionId: {} },
      { loadInitial: vi.fn().mockRejectedValue({ status: 404 }), poll: vi.fn().mockResolvedValue(undefined) },
    );
    const composerStore = createStaticStore(
      { draft: "", sending: false },
      { setDraft: vi.fn(), submit: vi.fn() },
    );
    const sessionUiStore = createStaticStore(
      { sessionId: "sess-1", diagnostics: null, queue: null, loading: false },
      { refresh: vi.fn().mockResolvedValue(undefined) },
    );

    const mountNode = document.createElement("div");
    root = mountNode;
    document.body.appendChild(mountNode);
    act(() => {
      render(
        <AppProviders
          sessionsStore={sessionsStore as any}
          messagesStore={messagesStore as any}
          liveSessionStore={liveSessionStore as any}
          composerStore={composerStore as any}
          sessionUiStore={sessionUiStore as any}
        >
          <AppShell />
        </AppProviders>,
        mountNode,
      );
    });

    await flush();
    await flush();

    expect(refresh).toHaveBeenCalledTimes(2);
  });
});
