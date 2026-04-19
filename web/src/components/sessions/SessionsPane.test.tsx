import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../../app/providers";
import { api } from "../../lib/api";
import { SessionsPane } from "./SessionsPane";

vi.mock("../../lib/api", () => ({
  api: {
    createSession: vi.fn().mockResolvedValue({ ok: true, session_id: "sess-2", broker_pid: 42 }),
    editSession: vi.fn().mockResolvedValue({ ok: true, alias: "Updated session" }),
    editCwdGroup: vi.fn().mockResolvedValue({ ok: true }),
    deleteSession: vi.fn().mockResolvedValue({ ok: true }),
    setSessionFocus: vi.fn().mockResolvedValue({ ok: true, focused: true }),
    getSessionDetails: vi.fn().mockResolvedValue({ ok: true, session: { session_id: "sess-1", alias: "Inbox cleanup", agent_backend: "pi", priority_offset: 0 } }),
  },
}));

let root: HTMLDivElement | null = null;

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

async function click(element: Element) {
  await act(async () => {
    (element as HTMLElement).dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

async function setInputValue(element: HTMLInputElement, value: string) {
  await act(async () => {
    element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function pressKey(element: Element, key: string) {
  await act(async () => {
    element.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
  });
}

function createSessionsStore(initialState: any, options?: { onRefresh?: () => void | Promise<void> }) {
  let state = initialState;
  const listeners = new Set<() => void>();

  const emit = () => listeners.forEach((listener) => listener());

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    refresh: vi.fn(async (_options?: { preferNewest?: boolean }) => {
      await options?.onRefresh?.();
      emit();
    }),
    refreshBootstrap: vi.fn(async () => {
      await options?.onRefresh?.();
      emit();
    }),
    loadMoreGroup: vi.fn(async () => {
      await options?.onRefresh?.();
      emit();
    }),
    loadMoreGroups: vi.fn(async () => {
      await options?.onRefresh?.();
      emit();
    }),
    select: vi.fn((sessionId: string) => {
      state = { ...state, activeSessionId: sessionId };
      emit();
    }),
    setState(next: any) {
      state = next;
      emit();
    },
  };
}

function renderSessionsPane(state: any, options?: { onRefresh?: () => void | Promise<void> }) {
  const sessionsStore = createSessionsStore(state, options);
  root = document.createElement("div");
  document.body.appendChild(root);
  render(
    <AppProviders sessionsStore={sessionsStore as any}>
      <SessionsPane />
    </AppProviders>,
    root,
  );
  return sessionsStore;
}

describe("SessionsPane", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders active session badges", () => {
    renderSessionsPane({
      items: [{ session_id: "sess-1", alias: "Inbox cleanup", agent_backend: "pi", busy: true, owned: true, queue_len: 2 }],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    expect(root?.querySelector("[data-testid='sessions-surface']")).not.toBeNull();
    expect(root?.querySelectorAll("[data-testid='session-card']")).toHaveLength(1);
    expect(root?.querySelector("[data-testid='session-card'][aria-current='true']")).not.toBeNull();
    expect(root?.textContent).toContain("Inbox cleanup");
    expect(root?.textContent).toContain("pi");
    expect(root?.textContent).toContain("web");
  });

  it("uses first user message as title and hides cwd in compact row", () => {
    renderSessionsPane({
      items: [{ session_id: "sess-1", first_user_message: "我准备用 preact + vite 重构web端，请帮我出个规划", cwd: "/Users/huapeixuan/Documents/Code/codoxear", agent_backend: "pi" }],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    expect(root?.querySelector(".sessionTitle")?.textContent).toContain("我准备用 preact + vite 重构web端，请帮我出个规划");
    expect(root?.textContent).not.toContain("/Users/huapeixuan/Documents/Code/codoxear");
  });

  it("prefers persisted title over first user message when alias is missing", () => {
    renderSessionsPane({
      items: [{ session_id: "sess-1", title: "Release checklist", first_user_message: "先整理一下今晚要发的内容", agent_backend: "pi" }],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    expect(root?.querySelector(".sessionTitle")?.textContent).toContain("Release checklist");
    expect(root?.querySelector(".sessionTitle")?.textContent).not.toContain("先整理一下今晚要发的内容");
  });

  it("switches between Sessions and Focus tabs", async () => {
    const sessionsStore = renderSessionsPane({
      items: [
        { session_id: "sess-1", alias: "Inbox cleanup", agent_backend: "pi", focused: true, cwd: "/tmp/a" },
        { session_id: "sess-2", alias: "Release prep", agent_backend: "pi", focused: false, cwd: "/tmp/b" },
      ],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    expect(root?.textContent).toContain("Inbox cleanup");
    expect(root?.textContent).toContain("Release prep");

    const focusTab = Array.from(root?.querySelectorAll<HTMLButtonElement>("button") || []).find((button) => button.textContent?.includes("Focus"));
    expect(focusTab).toBeDefined();
    await click(focusTab!);
    await flush();

    expect(root?.textContent).toContain("Inbox cleanup");
    expect(root?.textContent).not.toContain("Release prep");
    expect(sessionsStore.refresh).toHaveBeenCalled();
  });

  it("toggles Focus from the session rail", async () => {
    const sessionsStore = renderSessionsPane({
      items: [{ session_id: "sess-1", alias: "Inbox cleanup", agent_backend: "pi", focused: false }],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const focusButton = root?.querySelector<HTMLButtonElement>('button[aria-label="Add to Focus"]');
    expect(focusButton).not.toBeNull();
    await click(focusButton!);
    await flush();

    expect(api.setSessionFocus).toHaveBeenCalledWith("sess-1", true, null);
    expect(sessionsStore.refresh).toHaveBeenCalled();
  });

  it("deletes a historical session after confirmation", async () => {
    const confirm = vi.fn().mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    const sessionsStore = renderSessionsPane({
      items: [{ session_id: "history:pi:resume-1", alias: "Recovered", agent_backend: "pi", historical: true }],
      activeSessionId: "history:pi:resume-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const deleteButton = root?.querySelector<HTMLButtonElement>('button[aria-label="Delete session"]');
    expect(deleteButton).not.toBeNull();
    await click(deleteButton!);
    await flush();

    expect(api.deleteSession).toHaveBeenCalledWith("history:pi:resume-1");
    expect(sessionsStore.refresh).toHaveBeenCalled();
    expect(confirm).toHaveBeenCalled();
  });

  it("deletes a session after confirmation", async () => {
    const confirm = vi.fn().mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    const sessionsStore = renderSessionsPane({
      items: [{ session_id: "sess-1", alias: "Inbox cleanup", agent_backend: "pi" }],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const deleteButton = root?.querySelector<HTMLButtonElement>('button[aria-label="Delete session"]');
    expect(deleteButton).not.toBeNull();
    await click(deleteButton!);
    await flush();

    expect(api.deleteSession).toHaveBeenCalledWith("sess-1");
    expect(sessionsStore.refresh).toHaveBeenCalled();
    expect(confirm).toHaveBeenCalled();
  });

  it("duplicates a session from details and selects returned session", async () => {
    vi.mocked(api.getSessionDetails).mockResolvedValue({
      ok: true,
      session: {
        session_id: "sess-1",
        cwd: "/tmp/project",
        agent_backend: "codex",
        provider_choice: "openai-api",
        model: "gpt-5.4",
        reasoning_effort: "high",
        service_tier: "fast",
        transport: "tmux",
      },
    } as any);

    const sessionsStore = renderSessionsPane({
      items: [{ session_id: "sess-1", alias: "Inbox cleanup", cwd: "/tmp/project", agent_backend: "codex" }],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: ["/tmp/project"],
      cwdGroups: {},
      tmuxAvailable: true,
    });

    sessionsStore.refresh = vi.fn(async () => {
      sessionsStore.setState({
        ...sessionsStore.getState(),
        items: [...sessionsStore.getState().items, { session_id: "sess-2", alias: "Inbox cleanup copy", cwd: "/tmp/project", agent_backend: "codex" }],
      });
    });

    const duplicateButton = root?.querySelector<HTMLButtonElement>('button[aria-label="Duplicate session"]');
    expect(duplicateButton).not.toBeNull();
    await click(duplicateButton!);
    await flush();

    expect(api.getSessionDetails).toHaveBeenCalledWith("sess-1");
    expect(api.createSession).toHaveBeenCalledWith({
      cwd: "/tmp/project",
      backend: "codex",
      model: "gpt-5.4",
      model_provider: "openai",
      preferred_auth_method: "apikey",
      reasoning_effort: "high",
      service_tier: "fast",
      create_in_tmux: true,
    });
    expect(sessionsStore.select).toHaveBeenCalledWith("sess-2");
  });

  it("selects historical pi sessions without resuming them immediately", async () => {
    const sessionsStore = renderSessionsPane({
      items: [{ session_id: "history:pi:resume-hist", alias: "Recovered planning thread", cwd: "/tmp/project", agent_backend: "pi", historical: true }],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: ["/tmp/project"],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const sessionButton = root?.querySelector<HTMLButtonElement>(".sessionCardButton");
    expect(sessionButton).not.toBeNull();
    await click(sessionButton!);
    await flush();

    expect(api.createSession).not.toHaveBeenCalled();
    expect(sessionsStore.select).toHaveBeenCalledWith("history:pi:resume-hist");
  });

  it("opens edit dialog from icon action and saves fields", async () => {
    vi.mocked(api.getSessionDetails).mockResolvedValue({
      ok: true,
      session: { session_id: "sess-1", alias: "Inbox cleanup", first_user_message: "整理一下今天的会话", agent_backend: "pi", priority_offset: 0 },
    } as any);

    const sessionsStore = renderSessionsPane({
      items: [
        { session_id: "sess-1", alias: "Inbox cleanup", first_user_message: "整理一下今天的会话", agent_backend: "pi" },
        { session_id: "sess-2", alias: "Release prep", agent_backend: "pi" },
      ],
      activeSessionId: "sess-1",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const editButton = root?.querySelector<HTMLButtonElement>('button[aria-label="Edit session"]');
    expect(editButton).not.toBeNull();
    await click(editButton!);
    await flush();

    const dependencySelect = root?.querySelector('select[name="dependencySessionId"]') as HTMLSelectElement;
    await act(async () => {
      dependencySelect.value = "sess-2";
      dependencySelect.dispatchEvent(new Event("change", { bubbles: true }));
    });

    const saveButton = Array.from(root?.querySelectorAll("button") || []).find((button) => button.textContent?.includes("Save changes"));
    expect(saveButton).toBeDefined();
    saveButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    await flush();

    expect(api.editSession).toHaveBeenCalledWith("sess-1", {
      name: "Inbox cleanup",
      priority_offset: 0,
      snooze_until: null,
      dependency_session_id: "sess-2",
    });
    expect(sessionsStore.refresh).toHaveBeenCalled();
  });

  it("groups by cwd preserving first appearance order", () => {
    renderSessionsPane({
      items: [
        { session_id: "sess-1", alias: "Docs polish", cwd: "/work/docs", agent_backend: "pi", updated_ts: 30 },
        { session_id: "sess-2", alias: "Bug bash", cwd: "/work/api", agent_backend: "codex", updated_ts: 120 },
        { session_id: "sess-3", alias: "Release notes", cwd: "/work/docs", agent_backend: "pi", updated_ts: 20 },
      ],
      activeSessionId: "sess-3",
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const groups = Array.from(root?.querySelectorAll<HTMLElement>(".sessionGroup") || []);
    expect(groups).toHaveLength(2);
    expect(groups.map((group) => group.querySelector(".sessionGroupTitle")?.textContent?.trim())).toEqual(["docs", "api"]);
  });

  it("selects grouped session on card click", async () => {
    const sessionsStore = renderSessionsPane({
      items: [
        { session_id: "sess-1", alias: "Docs polish", cwd: "/work/docs", agent_backend: "pi" },
        { session_id: "sess-2", alias: "Release notes", cwd: "/work/docs", agent_backend: "pi" },
      ],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const cardButtons = root?.querySelectorAll<HTMLButtonElement>(".sessionCardButton") || [];
    await click(cardButtons[1]!);
    await flush();

    expect(sessionsStore.select).toHaveBeenCalledWith("sess-2");
  });

  it("renames cwd group and refreshes bootstrap", async () => {
    const sessionsStore = renderSessionsPane(
      {
        items: [{ session_id: "sess-1", alias: "Docs polish", cwd: "/work/docs", agent_backend: "pi" }],
        activeSessionId: null,
        loading: false,
        newSessionDefaults: null,
        recentCwds: [],
        cwdGroups: {},
        tmuxAvailable: false,
      },
      {
        onRefresh: () => {
          sessionsStore.setState({ ...sessionsStore.getState(), cwdGroups: { "/work/docs": { label: "Knowledge Base" } } });
        },
      },
    );

    const renameButton = root?.querySelector<HTMLButtonElement>(".sessionGroupRenameButton");
    expect(renameButton).not.toBeNull();
    await click(renameButton!);

    const input = root?.querySelector<HTMLInputElement>(".sessionGroupRenameInput");
    expect(input).not.toBeNull();
    await setInputValue(input!, "Knowledge Base");
    await pressKey(input!, "Enter");
    await flush();

    expect(api.editCwdGroup).toHaveBeenCalledWith({ cwd: "/work/docs", label: "Knowledge Base" });
    expect(sessionsStore.refreshBootstrap).toHaveBeenCalledTimes(1);
  });

  it("collapses cwd group and hides cards", async () => {
    const sessionsStore = renderSessionsPane(
      {
        items: [{ session_id: "sess-1", alias: "Docs polish", cwd: "/work/projects/docs", agent_backend: "pi" }],
        activeSessionId: null,
        loading: false,
        newSessionDefaults: null,
        recentCwds: [],
        cwdGroups: {},
        tmuxAvailable: false,
      },
      {
        onRefresh: () => {
          sessionsStore.setState({ ...sessionsStore.getState(), cwdGroups: { "/work/projects/docs": { collapsed: true } } });
        },
      },
    );

    const titleButton = root?.querySelector<HTMLButtonElement>(".sessionGroupTitleButton");
    expect(titleButton).not.toBeNull();
    await click(titleButton!);
    await flush();

    expect(api.editCwdGroup).toHaveBeenCalledWith({ cwd: "/work/projects/docs", collapsed: true });
    expect(root?.querySelectorAll("[data-testid='session-card']")).toHaveLength(0);
  });

  it("loads more sessions and directories when pagination controls are clicked", async () => {
    const sessionsStore = renderSessionsPane({
      items: [{ session_id: "sess-1", alias: "Session 1", cwd: "/work/docs", agent_backend: "pi" }],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
      remainingByGroup: { "/work/docs": 1 },
      omittedGroupCount: 2,
    });

    const groupMore = Array.from(root?.querySelectorAll<HTMLButtonElement>("button") || []).find((button) => (button.textContent || "").trim() === "...");
    const dirsMore = Array.from(root?.querySelectorAll<HTMLButtonElement>("button") || []).find((button) => (button.textContent || "").includes("Load 2 more directories"));

    expect(groupMore).toBeDefined();
    expect(dirsMore).toBeDefined();

    await click(groupMore!);
    await click(dirsMore!);
    await flush();

    expect(sessionsStore.loadMoreGroup).toHaveBeenCalledWith("/work/docs");
    expect(sessionsStore.loadMoreGroups).toHaveBeenCalledTimes(1);
  });

  it("renders fallback group without rename/toggle controls", () => {
    renderSessionsPane({
      items: [{ session_id: "sess-1", alias: "Inbox", agent_backend: "pi" }],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });

    const group = root?.querySelector<HTMLElement>(".sessionGroup");
    expect(group?.querySelector(".sessionGroupTitle")?.textContent).toContain("No working directory");
    expect(group?.querySelector(".sessionGroupRenameButton")).toBeNull();
  });
});
