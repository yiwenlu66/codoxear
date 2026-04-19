import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "../../app/providers";
import { api } from "../../lib/api";

const { todoPanelRenderLog } = vi.hoisted(() => ({
  todoPanelRenderLog: [] as Array<{ expanded: boolean; progressText: unknown }>,
}));

function getLoggedProgressText(snapshot: unknown) {
  if (!snapshot || typeof snapshot !== "object") {
    return null;
  }

  return (snapshot as Record<string, unknown>).progress_text ?? null;
}

vi.mock("./TodoComposerPanel", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./TodoComposerPanel")>();

  return {
    ...actual,
    TodoComposerPanel: (props: Parameters<typeof actual.TodoComposerPanel>[0]) => {
      todoPanelRenderLog.push({ expanded: props.expanded, progressText: getLoggedProgressText(props.snapshot) });
      return <actual.TodoComposerPanel {...props} />;
    },
  };
});

import { Composer } from "./Composer";
import { createComposerStore } from "../../domains/composer/store";

interface RenderComposerOptions {
  activeSessionId?: string | null;
  items?: Array<{ session_id: string; agent_backend: string; busy: boolean; historical?: boolean; pending_startup?: boolean }>;
  liveBusyBySessionId?: Record<string, boolean>;
  liveContextUsageBySessionId?: Record<string, { used_tokens?: number; total_tokens?: number; percent_used?: number } | null>;
  messageEventsBySessionId?: Record<string, Array<Record<string, unknown>>>;
  sessionUiSessionId?: string | null;
  diagnostics?: Record<string, unknown> | null;
  draft?: string;
  submitResult?: unknown;
  composerStore?: any;
  compactMobile?: boolean;
}

let root: HTMLDivElement | null = null;

function getRoot() {
  if (!root) {
    throw new Error("Composer test root has not been initialized");
  }

  return root;
}

function createStore<TState extends object, TActions extends Record<string, (...args: any[]) => any>>(
  initialState: TState,
  actionFactory: (setState: (patch: Partial<TState>) => void, getState: () => TState) => TActions,
) {
  let state = initialState;
  const listeners = new Set<() => void>();
  const setState = (patch: Partial<TState>) => {
    state = { ...state, ...patch };
    listeners.forEach((listener) => listener());
  };
  const getState = () => state;
  return {
    getState,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    ...actionFactory(setState, getState),
  };
}

function mockViewportMatchMedia(matches: boolean) {
  const originalMatchMedia = window.matchMedia;

  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === "(max-width: 880px)" ? matches : false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn().mockReturnValue(false),
    })),
  });

  return () => {
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: originalMatchMedia,
    });
  };
}

function renderComposer(options: RenderComposerOptions = {}) {
  const {
    activeSessionId = "sess-1",
    items = [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
    liveBusyBySessionId = {},
    liveContextUsageBySessionId = {},
    messageEventsBySessionId = {},
    sessionUiSessionId = activeSessionId,
    diagnostics = null,
    draft = "Hello",
    submitResult,
    composerStore: providedComposerStore,
    compactMobile = false,
  } = options;
  const submit = vi.fn().mockResolvedValue(submitResult);
  const liveSessionStore = createStore(
    {
      offsetsBySessionId: {},
      liveOffsetsBySessionId: {},
      requestsBySessionId: {},
      requestVersionsBySessionId: {},
      busyBySessionId: liveBusyBySessionId,
      loadingBySessionId: {},
      contextUsageBySessionId: liveContextUsageBySessionId,
      tokenBySessionId: {},
      errorBySessionId: {},
    },
    () => ({ loadInitial: vi.fn(), poll: vi.fn() }),
  );
  const messagesStore = createStore(
    {
      bySessionId: messageEventsBySessionId,
      offsetsBySessionId: {},
      hasOlderBySessionId: {},
      olderBeforeBySessionId: {},
      loadingOlderBySessionId: {},
      loadingBySessionId: {},
      loadedBySessionId: {},
      loading: false,
    },
    () => ({ applyLive: vi.fn(), loadInitial: vi.fn(), poll: vi.fn(), loadOlder: vi.fn() }),
  );
  const sessionsStore = createStore(
    { items, activeSessionId, loading: false, newSessionDefaults: null },
    (setState) => ({ refresh: vi.fn(), select: vi.fn(), setState }),
  );
  const composerStore = providedComposerStore ?? createStore(
    { draftBySessionId: activeSessionId ? { [activeSessionId]: draft } : {}, sending: false, pendingBySessionId: {} },
    (setState, getState) => ({
      setDraft(sessionId: string | null | undefined, value: string) {
        if (!sessionId) {
          return;
        }
        const nextDraftBySessionId = { ...(getState().draftBySessionId ?? {}) };
        if (value.length) {
          nextDraftBySessionId[sessionId] = value;
        } else {
          delete nextDraftBySessionId[sessionId];
        }
        setState({ draftBySessionId: nextDraftBySessionId });
      },
      submit,
    }),
  );
  const sessionUiStore = createStore(
    {
      sessionId: sessionUiSessionId,
      diagnostics,
      queue: null,
      files: [],
      requests: [],
      loading: false,
    },
    (setState) => ({ refresh: vi.fn(), setState }),
  );

  root = document.createElement("div");
  document.body.appendChild(root);
  act(() => {
    render(
      <AppProviders
        sessionsStore={sessionsStore as any}
        messagesStore={messagesStore as any}
        composerStore={composerStore as any}
        liveSessionStore={liveSessionStore as any}
        sessionUiStore={sessionUiStore as any}
      >
        <Composer compactMobile={compactMobile} />
      </AppProviders>,
      root!,
    );
  });

  return { submit, sessionsStore, messagesStore, composerStore, liveSessionStore, sessionUiStore };
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await Promise.resolve();
  });
}

describe("Composer", () => {
  afterEach(() => {
    vi.useRealTimers();
    window.localStorage.clear();
    todoPanelRenderLog.length = 0;
    vi.restoreAllMocks();
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("shows Pi context usage in the composer gutter", async () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
      liveContextUsageBySessionId: {
        "sess-1": { used_tokens: 82000, total_tokens: 200000, percent_used: 41 },
      },
    });

    expect(getRoot().textContent).toContain("82K/200K 41%");
  });

  it("shows zero-used fallback when the total context is known", async () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
      liveContextUsageBySessionId: {
        "sess-1": { total_tokens: 200000 },
      },
    });

    expect(getRoot().textContent).toContain("0/200K 0%");
  });

  it("prefers diagnostics context usage when live usage is missing or still zero", async () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
      liveContextUsageBySessionId: {
        "sess-1": { used_tokens: 0, total_tokens: 272000, percent_used: 0 },
      },
      diagnostics: {
        context_usage: { used_tokens: 91000, total_tokens: 272000, percent_used: 33 },
      },
    });

    expect(getRoot().textContent).toContain("91K/272K 33%");
  });

  it("shows the active turn elapsed time in the composer gutter", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-19T10:00:12.000Z"));

    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: true }],
      liveBusyBySessionId: { "sess-1": true },
      messageEventsBySessionId: {
        "sess-1": [
          { role: "user", ts: Date.parse("2026-04-19T10:00:00.000Z") / 1000, text: "Run this" },
          { type: "reasoning", ts: Date.parse("2026-04-19T10:00:05.000Z") / 1000, text: "thinking" },
        ],
      },
    });

    expect(getRoot().textContent).toContain("Turn 12s");

    await act(async () => {
      vi.advanceTimersByTime(3000);
    });

    expect(getRoot().textContent).toContain("Turn 15s");
  });

  it("freezes the latest turn duration after the round finishes", async () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
      messageEventsBySessionId: {
        "sess-1": [
          { role: "user", ts: 100, text: "Run this" },
          { type: "tool", ts: 104, text: "bash" },
          { role: "assistant", ts: 109, text: "Finished" },
        ],
      },
    });

    expect(getRoot().textContent).toContain("Turn 9s");
  });

  it("keeps the textarea editable while a Pi session is still starting", () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false, pending_startup: true }],
      draft: "queued draft",
    });

    const composerRoot = getRoot();
    const textarea = composerRoot.querySelector("textarea") as HTMLTextAreaElement;
    const queueButton = Array.from(composerRoot.querySelectorAll("button")).find((button) => button.textContent?.includes("Queue")) as HTMLButtonElement;
    const sendButton = composerRoot.querySelector("button[type='submit']") as HTMLButtonElement;

    expect(textarea.disabled).toBe(false);
    expect(queueButton.disabled).toBe(false);
    expect(sendButton.disabled).toBe(true);
  });

  it("keeps separate drafts for different sessions", async () => {
    const sessionsStore = createStore(
      {
        items: [
          { session_id: "sess-1", agent_backend: "pi", busy: false },
          { session_id: "sess-2", agent_backend: "pi", busy: false },
        ],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      },
      (setState) => ({
        refresh: vi.fn(),
        select: vi.fn(),
        setState,
      }),
    );
    const composerStore = createComposerStore();
    root = document.createElement("div");
    document.body.appendChild(root);
    act(() => {
      render(
        <AppProviders sessionsStore={sessionsStore as any} composerStore={composerStore as any}>
          <Composer />
        </AppProviders>,
        root!,
      );
    });

    const textarea = getRoot().querySelector("textarea") as HTMLTextAreaElement;

    act(() => {
      textarea.value = "draft one";
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });

    act(() => {
      sessionsStore.setState({
        items: [
          { session_id: "sess-1", agent_backend: "pi", busy: false },
          { session_id: "sess-2", agent_backend: "pi", busy: false },
        ],
        activeSessionId: "sess-2",
        loading: false,
        newSessionDefaults: null,
      });
    });

    expect((getRoot().querySelector("textarea") as HTMLTextAreaElement).value).toBe("");

    act(() => {
      const currentTextarea = getRoot().querySelector("textarea") as HTMLTextAreaElement;
      currentTextarea.value = "draft two";
      currentTextarea.dispatchEvent(new Event("input", { bubbles: true }));
    });

    act(() => {
      sessionsStore.setState({
        items: [
          { session_id: "sess-1", agent_backend: "pi", busy: false },
          { session_id: "sess-2", agent_backend: "pi", busy: false },
        ],
        activeSessionId: "sess-1",
        loading: false,
        newSessionDefaults: null,
      });
    });

    expect((getRoot().querySelector("textarea") as HTMLTextAreaElement).value).toBe("draft one");
  });

  it("restores session drafts from localStorage on reload", () => {
    window.localStorage.setItem("codoxear.composerDrafts.v1", JSON.stringify({ "sess-1": "persisted draft" }));
    const composerStore = createComposerStore();
    renderComposer({ composerStore: composerStore as any });

    expect((getRoot().querySelector("textarea") as HTMLTextAreaElement).value).toBe("persisted draft");
  });

  it("submits on plain Enter when enter-to-send is enabled", async () => {
    window.localStorage.setItem("codoxear.enterToSend", "1");
    const { submit } = renderComposer({ items: [] });
    const composerRoot = getRoot();

    const textarea = composerRoot.querySelector("textarea") as HTMLTextAreaElement;
    expect(composerRoot.querySelector("[data-testid='composer-card']")).not.toBeNull();
    expect(composerRoot.querySelector("button[type='submit']")).not.toBeNull();
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    Object.defineProperty(event, "isComposing", { value: false });
    textarea.dispatchEvent(event);

    expect(submit).toHaveBeenCalledWith("sess-1");
    expect(event.defaultPrevented).toBe(true);
  });

  it("refreshes live session state immediately after a successful send", async () => {
    const { submit, liveSessionStore, sessionUiStore, sessionsStore } = renderComposer();
    const composerRoot = getRoot();

    await act(async () => {
      (composerRoot.querySelector("button[type='submit']") as HTMLButtonElement).click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(submit).toHaveBeenCalledWith("sess-1");
    expect(liveSessionStore.loadInitial).toHaveBeenCalledWith("sess-1");
    expect(sessionUiStore.refresh).toHaveBeenCalledWith("sess-1", { agentBackend: "pi" });
    expect(sessionsStore.refresh).toHaveBeenCalledTimes(1);
  });

  it("switches from a historical pi session to the resumed live session after send", async () => {
    const { submit, liveSessionStore, sessionUiStore, sessionsStore } = renderComposer({
      activeSessionId: "history:pi:resume-hist",
      items: [{ session_id: "history:pi:resume-hist", agent_backend: "pi", busy: false, historical: true }],
      submitResult: { ok: true, session_id: "live-pi-1" },
    });
    const composerRoot = getRoot();

    await act(async () => {
      (composerRoot.querySelector("button[type='submit']") as HTMLButtonElement).click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(submit).toHaveBeenCalledWith("history:pi:resume-hist");
    expect(sessionsStore.refresh).toHaveBeenCalledTimes(2);
    expect(sessionsStore.select).toHaveBeenCalledWith("live-pi-1");
    expect(liveSessionStore.loadInitial).toHaveBeenCalledWith("live-pi-1");
    expect(sessionUiStore.refresh).toHaveBeenCalledWith("live-pi-1", { agentBackend: "pi" });
  });

  it("continues polling after a successful send so new messages appear without a full refresh", async () => {
    vi.useFakeTimers();
    const { liveSessionStore, sessionUiStore, sessionsStore } = renderComposer();
    const composerRoot = getRoot();

    await act(async () => {
      (composerRoot.querySelector("button[type='submit']") as HTMLButtonElement).click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(liveSessionStore.loadInitial).toHaveBeenCalledWith("sess-1");
    expect(liveSessionStore.poll).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(1500);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(liveSessionStore.poll).toHaveBeenCalledWith("sess-1");
    expect(sessionUiStore.refresh).toHaveBeenCalledTimes(2);
    expect(sessionsStore.refresh).toHaveBeenCalledTimes(2);
  });

  it("submits on ctrl+enter when enter-to-send is disabled", () => {
    const { submit } = renderComposer({ items: [] });
    const composerRoot = getRoot();

    const textarea = composerRoot.querySelector("textarea") as HTMLTextAreaElement;
    const event = new KeyboardEvent("keydown", { key: "Enter", ctrlKey: true, bubbles: true, cancelable: true });
    Object.defineProperty(event, "isComposing", { value: false });
    textarea.dispatchEvent(event);

    expect(submit).toHaveBeenCalledWith("sess-1");
    expect(event.defaultPrevented).toBe(true);
  });

  it("does not submit on plain Enter when enter-to-send is disabled", () => {
    const { submit } = renderComposer({ items: [] });
    const composerRoot = getRoot();

    const textarea = composerRoot.querySelector("textarea") as HTMLTextAreaElement;
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    Object.defineProperty(event, "isComposing", { value: false });
    textarea.dispatchEvent(event);

    expect(submit).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
  });

  it("shows Pi slash command suggestions and applies the highlighted command", async () => {
    const getSessionCommands = vi.spyOn(api, "getSessionCommands").mockResolvedValue({
      commands: [
        { name: "reload", description: "Reload Pi runtime" },
        { name: "resume", description: "Resume a session" },
      ],
    });
    renderComposer({ draft: "/rel" });
    const composerRoot = getRoot();

    await flushEffects();
    await act(async () => {
      await getSessionCommands.mock.results[0]?.value;
    });
    await flushEffects();

    expect(composerRoot.textContent).toContain("/reload");
    expect(composerRoot.textContent).not.toContain("/resume");

    const textarea = composerRoot.querySelector("textarea") as HTMLTextAreaElement;
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    Object.defineProperty(event, "isComposing", { value: false });
    act(() => {
      textarea.dispatchEvent(event);
    });

    expect(textarea.value).toBe("/reload ");
  });

  it("does not request slash commands for non-pi sessions", async () => {
    const getSessionCommands = vi.spyOn(api, "getSessionCommands").mockResolvedValue({
      commands: [{ name: "reload" }],
    });
    renderComposer({
      draft: "/re",
      items: [{ session_id: "sess-1", agent_backend: "codex", busy: false }],
    });

    await flushEffects();

    expect(getSessionCommands).not.toHaveBeenCalled();
    expect(getRoot().querySelector("[data-testid='composer-command-menu']")).toBeNull();
  });

  it("caches Pi slash commands per session within the current page lifetime", async () => {
    const getSessionCommands = vi.spyOn(api, "getSessionCommands").mockResolvedValue({
      commands: [{ name: "reload", description: "Reload Pi runtime" }],
    });
    const { composerStore } = renderComposer({ draft: "/re" });

    await flushEffects();
    await act(async () => {
      await getSessionCommands.mock.results[0]?.value;
    });
    await flushEffects();

    expect(getSessionCommands).toHaveBeenCalledTimes(1);

    act(() => {
      composerStore.setDraft("sess-1", "hello");
    });
    await flushEffects();

    act(() => {
      composerStore.setDraft("sess-1", "/re");
    });
    await flushEffects();

    expect(getSessionCommands).toHaveBeenCalledTimes(1);
  });

  it("renders the input above a dedicated controls row for stacked mobile layout", () => {
    renderComposer({ items: [] });
    const composerRoot = getRoot();

    const form = composerRoot.querySelector("form.composerShell");
    const inputWrap = form?.querySelector(".composerInputWrap");
    const controlsRow = form?.querySelector(".composerControlsRow");

    expect(form?.firstElementChild).toBe(inputWrap ?? null);
    expect(controlsRow).not.toBeNull();
    expect(controlsRow?.querySelector(".composerAttachButton")).not.toBeNull();
    expect(controlsRow?.querySelector(".composerQueueButton")).not.toBeNull();
    expect(controlsRow?.querySelector(".sendButton")).not.toBeNull();
  });

  it("renders compact mobile controls with horizontal metadata and no queue or attach buttons", () => {
    renderComposer({
      compactMobile: true,
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: true }],
      liveContextUsageBySessionId: {
        "sess-1": { used_tokens: 82000, total_tokens: 200000, percent_used: 41 },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.querySelector(".composerControlsColumn.compactMobile")).not.toBeNull();
    expect(composerRoot.querySelector(".composerMetaRow.compactMobile")).not.toBeNull();
    expect(composerRoot.querySelector(".composerAttachButton")).toBeNull();
    expect(composerRoot.querySelector(".composerQueueButton")).toBeNull();
    expect(composerRoot.querySelector(".composerInterruptButton")).not.toBeNull();
    expect(composerRoot.querySelector(".sendButton")).not.toBeNull();
  });

  it("starts with two rows on mobile and caps textarea growth after the mobile max height", async () => {
    const restoreMatchMedia = mockViewportMatchMedia(true);

    try {
      renderComposer({ items: [], draft: "" });
      const composerRoot = getRoot();
      const textarea = composerRoot.querySelector("textarea") as HTMLTextAreaElement;

      expect(textarea.getAttribute("rows")).toBe("2");
      expect(textarea.style.minHeight).toBe("56px");
      expect(textarea.style.height).toBe("56px");

      Object.defineProperty(textarea, "scrollHeight", {
        configurable: true,
        get: () => 240,
      });

      act(() => {
        textarea.value = Array.from({ length: 12 }, (_, index) => `line ${index + 1}`).join("\n");
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
      });
      await flushEffects();

      expect(textarea.style.height).toBe("176px");
      expect(textarea.style.overflowY).toBe("auto");
    } finally {
      restoreMatchMedia();
    }
  });

  it("uploads a selected file for codex sessions and shows the attachment count", async () => {
    const attachSessionFile = vi.spyOn(api, "attachSessionFile").mockResolvedValue({ ok: true, path: "/tmp/notes.txt" } as any);
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "codex", busy: false }],
      draft: "Hello",
    });
    const composerRoot = getRoot();

    const attachButton = composerRoot.querySelector(".composerAttachButton") as HTMLButtonElement;
    const fileInput = composerRoot.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["hello"], "notes.txt", { type: "text/plain" });

    Object.defineProperty(fileInput, "files", {
      configurable: true,
      value: [file],
    });

    act(() => {
      attachButton.click();
      fileInput.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await flushEffects();

    expect(attachSessionFile).toHaveBeenCalledWith("sess-1", {
      filename: "notes.txt",
      data_b64: "aGVsbG8=",
      attachment_index: 1,
    });
    expect(composerRoot.querySelector(".composerAttachBadge")?.textContent).toBe("1");
  });

  it("disables attachments for pi sessions", () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
    });
    const composerRoot = getRoot();

    const attachButton = composerRoot.querySelector(".composerAttachButton") as HTMLButtonElement;

    expect(attachButton.disabled).toBe(true);
    expect(attachButton.title).toContain("Pi");
  });

  it("queues the current draft through the queue button and refreshes workspace state", async () => {
    const enqueueMessage = vi.spyOn(api, "enqueueMessage").mockResolvedValue({ ok: true } as any);
    const { composerStore, sessionUiStore } = renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: true }],
      draft: "After this turn, also inspect logs",
    });
    const composerRoot = getRoot();

    const queueButton = composerRoot.querySelector(".composerQueueButton") as HTMLButtonElement;

    await act(async () => {
      queueButton.click();
      await Promise.resolve();
    });
    await flushEffects();

    expect(enqueueMessage).toHaveBeenCalledWith("sess-1", "After this turn, also inspect logs");
    expect(sessionUiStore.refresh).toHaveBeenCalledWith("sess-1", { agentBackend: "pi" });
    expect(composerStore.getState().draftBySessionId?.["sess-1"] ?? "").toBe("");
  });

  it("shows a cancel-loop button for a busy session and interrupts the active loop", async () => {
    const interruptSession = vi.spyOn(api, "interruptSession").mockResolvedValue({ ok: true } as any);
    const { liveSessionStore, sessionUiStore, sessionsStore } = renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "pi", busy: false }],
      liveBusyBySessionId: { "sess-1": true },
    });
    const composerRoot = getRoot();

    const cancelButton = composerRoot.querySelector(".composerInterruptButton") as HTMLButtonElement | null;
    expect(cancelButton).not.toBeNull();
    expect(cancelButton?.textContent).toContain("Cancel loop");

    await act(async () => {
      cancelButton?.click();
      await Promise.resolve();
    });
    await flushEffects();

    expect(interruptSession).toHaveBeenCalledWith("sess-1");
    expect(sessionsStore.refresh).toHaveBeenCalledTimes(1);
    expect(liveSessionStore.loadInitial).toHaveBeenCalledWith("sess-1");
    expect(sessionUiStore.refresh).toHaveBeenCalledWith("sess-1", { agentBackend: "pi" });
  });

  it("shows a todo summary bar above the composer for a current pi session with todo items", () => {
    renderComposer({
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "2/3 completed",
          items: [{ title: "Move todo above composer", status: "in-progress" }],
        },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.querySelector(".composerTodoBar")).not.toBeNull();
    expect(composerRoot.querySelector(".composerTodoBar + [data-testid='composer-card']")).not.toBeNull();
    expect(composerRoot.querySelector(".composerTodoBar")?.nextElementSibling?.getAttribute("data-testid")).toBe("composer-card");
    expect(composerRoot.textContent).toContain("2/3 completed");
  });

  it("renders an aggregated Claude Todo V2 snapshot through the existing composer todo bar", async () => {
    renderComposer({
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [
            {
              id: "3",
              title: "Explore current todo implementation",
              description: "Inspect the existing web session todo feature.",
              status: "in-progress",
              owner: "Codex",
              assigned_by: "team-lead",
              source: "claude-todo-v2",
            },
            {
              id: "1",
              title: "Clarify compatibility goal",
              status: "completed",
              source: "claude-todo-v2",
            },
          ],
        },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.textContent).toContain("1/2 completed");

    const toggle = composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null;
    expect(toggle).not.toBeNull();

    toggle?.click();
    await Promise.resolve();

    expect(composerRoot.textContent).toContain("Explore current todo implementation");
    expect(composerRoot.textContent).toContain("Inspect the existing web session todo feature.");
    expect(composerRoot.textContent).toContain("Clarify compatibility goal");
  });

  it("renders a Claude Todo V2 current-state snapshot through the existing composer todo bar", async () => {
    renderComposer({
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [
            {
              id: "1",
              title: "Explore project context for todo state rendering",
              description: "Exploring project context for todo state rendering",
              status: "completed",
              source: "claude-todo-v2",
            },
            {
              id: "2",
              title: "Transition from design into implementation planning",
              description: "Transitioning from design into implementation planning",
              status: "not-started",
              source: "claude-todo-v2",
            },
          ],
        },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.textContent).toContain("1/2 completed");

    const toggle = composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null;
    expect(toggle).not.toBeNull();

    toggle?.click();
    await Promise.resolve();

    expect(composerRoot.textContent).toContain("Explore project context for todo state rendering");
    expect(composerRoot.textContent).toContain("Exploring project context for todo state rendering");
    expect(composerRoot.textContent).toContain("Transition from design into implementation planning");
    expect(composerRoot.textContent).toContain("not-started");
  });

  it("expands and collapses the todo panel when the summary bar is clicked", async () => {
    renderComposer({
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [{ title: "Keep toggle wired", status: "in-progress" }],
        },
      },
    });
    const composerRoot = getRoot();

    const toggle = composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null;

    expect(toggle).not.toBeNull();
    expect(composerRoot.querySelector(".composerTodoPanel")).toBeNull();

    toggle?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    toggle?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).toBeNull();
  });

  it("does not show a todo bar for non-pi sessions", () => {
    renderComposer({
      items: [{ session_id: "sess-1", agent_backend: "codex", busy: false }],
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "Should stay hidden", status: "completed" }],
        },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.querySelector(".composerTodoBar")).toBeNull();
  });

  it("does not show a todo bar when session ui state is stale", () => {
    renderComposer({
      sessionUiSessionId: "sess-2",
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "Should stay hidden", status: "completed" }],
        },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.querySelector(".composerTodoBar")).toBeNull();
  });

  it("does not show a todo bar when the snapshot has no valid items", () => {
    renderComposer({
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "   ", status: "   ", description: "   " }],
        },
      },
    });
    const composerRoot = getRoot();

    expect(composerRoot.querySelector(".composerTodoBar")).toBeNull();
  });

  it("preserves expansion when a session switch temporarily hides the todo bar", async () => {
    const { sessionsStore } = renderComposer({
      items: [
        { session_id: "sess-1", agent_backend: "pi", busy: false },
        { session_id: "sess-2", agent_backend: "pi", busy: false },
      ],
      diagnostics: {
        todo_snapshot: {
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [{ title: "Reset when switching", status: "in-progress" }],
        },
      },
    });
    const composerRoot = getRoot();

    (composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null)?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    (sessionsStore as any).setState({ activeSessionId: "sess-2" });
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoBar")).toBeNull();

    (sessionsStore as any).setState({ activeSessionId: "sess-1" });
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoBar")).not.toBeNull();
    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();
  });

  it("restores the todo panel expanded after same-session diagnostics temporarily hide it", async () => {
    const snapshot = {
      available: true,
      error: false,
      progress_text: "1/2 completed",
      items: [{ title: "Keep expansion through refresh", status: "in-progress" }],
    };
    const { sessionUiStore } = renderComposer({
      diagnostics: {
        todo_snapshot: snapshot,
      },
    });
    const composerRoot = getRoot();

    (composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null)?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    act(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-1",
        diagnostics: null,
      });
    });

    expect(composerRoot.querySelector(".composerTodoBar")).toBeNull();

    act(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-1",
        diagnostics: { todo_snapshot: snapshot },
      });
    });

    expect(composerRoot.querySelector(".composerTodoBar")).not.toBeNull();
    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();
  });

  it("remembers expanded state separately for each pi session when switching", async () => {
    const firstSnapshot = {
      available: true,
      error: false,
      progress_text: "1/2 completed",
      items: [{ title: "Open source session todo", status: "in-progress" }],
    };
    const secondSnapshot = {
      available: true,
      error: false,
      progress_text: "2/3 completed",
      items: [{ title: "Destination session todo", status: "in-progress" }],
    };
    const { sessionsStore, sessionUiStore } = renderComposer({
      items: [
        { session_id: "sess-1", agent_backend: "pi", busy: false },
        { session_id: "sess-2", agent_backend: "pi", busy: false },
      ],
      diagnostics: {
        todo_snapshot: firstSnapshot,
      },
    });
    const composerRoot = getRoot();

    (composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null)?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    const renderCountBeforeSwitch = todoPanelRenderLog.length;

    act(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-2",
        diagnostics: { todo_snapshot: secondSnapshot },
      });
      (sessionsStore as any).setState({ activeSessionId: "sess-2" });
    });

    const destinationRenders = todoPanelRenderLog
      .slice(renderCountBeforeSwitch)
      .filter((entry) => entry.progressText === "2/3 completed");

    expect(destinationRenders[0]?.expanded).toBe(false);
    expect(composerRoot.querySelector(".composerTodoBar")).not.toBeNull();
    expect(composerRoot.textContent).toContain("2/3 completed");
    expect(composerRoot.querySelector(".composerTodoPanel")).toBeNull();

    (composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null)?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    act(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-1",
        diagnostics: { todo_snapshot: firstSnapshot },
      });
      (sessionsStore as any).setState({ activeSessionId: "sess-1" });
    });

    expect(composerRoot.textContent).toContain("1/2 completed");
    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    act(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-2",
        diagnostics: { todo_snapshot: secondSnapshot },
      });
      (sessionsStore as any).setState({ activeSessionId: "sess-2" });
    });

    expect(composerRoot.textContent).toContain("2/3 completed");
    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();
  });

  it("expands on the first click after switching directly between displayable pi sessions", async () => {
    const firstSnapshot = {
      available: true,
      error: false,
      progress_text: "1/2 completed",
      items: [{ title: "Open source session todo", status: "in-progress" }],
    };
    const secondSnapshot = {
      available: true,
      error: false,
      progress_text: "2/3 completed",
      items: [{ title: "Destination session todo", status: "in-progress" }],
    };
    const { sessionsStore, sessionUiStore } = renderComposer({
      items: [
        { session_id: "sess-1", agent_backend: "pi", busy: false },
        { session_id: "sess-2", agent_backend: "pi", busy: false },
      ],
      diagnostics: {
        todo_snapshot: firstSnapshot,
      },
    });
    const composerRoot = getRoot();

    (composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null)?.click();
    await Promise.resolve();

    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();

    act(() => {
      (sessionUiStore as any).setState({
        sessionId: "sess-2",
        diagnostics: { todo_snapshot: secondSnapshot },
      });
      (sessionsStore as any).setState({ activeSessionId: "sess-2" });
      (composerRoot.querySelector(".composerTodoBarButton") as HTMLButtonElement | null)?.click();
    });

    expect(composerRoot.textContent).toContain("2/3 completed");
    expect(composerRoot.querySelector(".composerTodoPanel")).not.toBeNull();
  });
});
