import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../../lib/api";
import { createSessionsStore } from "./store";

vi.mock("../../lib/api", () => ({
  api: {
    listSessions: vi.fn(),
    getSessionsBootstrap: vi.fn(),
  },
}));

describe("createSessionsStore", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("selects the newest session on the first refresh", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ session_id: "s1" }, { session_id: "s2" }],
    } as never);
    const store = createSessionsStore();
    const snapshots: string[] = [];

    store.subscribe(() => {
      const state = store.getState();
      snapshots.push(`${state.loading}:${state.activeSessionId}`);
    });

    await store.refresh();

    expect(snapshots).toEqual(["true:null", "false:s1"]);
    expect(store.getState()).toEqual({
      items: [{ session_id: "s1" }, { session_id: "s2" }],
      activeSessionId: "s1",
      loading: false,
      bootstrapLoaded: false,
      remainingByGroup: {},
      omittedGroupCount: 0,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });
  });

  it("refreshes bootstrap metadata without touching the selected session", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ session_id: "s1" }, { session_id: "s2" }],
    } as never);
    vi.mocked(api.getSessionsBootstrap).mockResolvedValue({
      recent_cwds: ["/tmp/project"],
      cwd_groups: { "/tmp/project": { label: "Project", collapsed: true } },
      new_session_defaults: { default_backend: "pi" },
      tmux_available: true,
    } as never);
    const store = createSessionsStore();

    await store.refresh();
    store.select("s2");
    await store.refreshBootstrap();

    expect(store.getState()).toEqual({
      items: [{ session_id: "s1" }, { session_id: "s2" }],
      activeSessionId: "s2",
      loading: false,
      bootstrapLoaded: true,
      remainingByGroup: {},
      omittedGroupCount: 0,
      newSessionDefaults: { default_backend: "pi" },
      recentCwds: ["/tmp/project"],
      cwdGroups: { "/tmp/project": { label: "Project", collapsed: true } },
      tmuxAvailable: true,
    });
  });

  it("loads more rows for one cwd group and updates remaining counts", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "docs-1", cwd: "/work/docs" },
          { session_id: "docs-2", cwd: "/work/docs" },
          { session_id: "docs-3", cwd: "/work/docs" },
          { session_id: "docs-4", cwd: "/work/docs" },
          { session_id: "docs-5", cwd: "/work/docs" },
        ],
        remaining_by_group: { "/work/docs": 2 },
      } as never)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "docs-6", cwd: "/work/docs" },
          { session_id: "docs-7", cwd: "/work/docs" },
        ],
        remaining_by_group: {},
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    await store.loadMoreGroup("/work/docs");

    expect(api.listSessions).toHaveBeenNthCalledWith(2, {
      groupKey: "/work/docs",
      offset: 5,
      limit: 5,
    });
    expect(store.getState().items).toEqual([
      { session_id: "docs-1", cwd: "/work/docs" },
      { session_id: "docs-2", cwd: "/work/docs" },
      { session_id: "docs-3", cwd: "/work/docs" },
      { session_id: "docs-4", cwd: "/work/docs" },
      { session_id: "docs-5", cwd: "/work/docs" },
      { session_id: "docs-6", cwd: "/work/docs" },
      { session_id: "docs-7", cwd: "/work/docs" },
    ]);
    expect(store.getState().remainingByGroup).toEqual({});
  });

  it("loads more omitted groups and updates the omitted group count", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "group-1", cwd: "/work/group-1" },
          { session_id: "group-2", cwd: "/work/group-2" },
          { session_id: "group-3", cwd: "/work/group-3" },
        ],
        omitted_group_count: 2,
      } as never)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "group-4", cwd: "/work/group-4" },
          { session_id: "group-5", cwd: "/work/group-5" },
        ],
        omitted_group_count: 0,
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    await store.loadMoreGroups();

    expect(api.listSessions).toHaveBeenNthCalledWith(2, {
      groupOffset: 3,
      groupLimit: 3,
    });
    expect(store.getState().items.map((session) => session.session_id)).toEqual([
      "group-1",
      "group-2",
      "group-3",
      "group-4",
      "group-5",
    ]);
    expect(store.getState().omittedGroupCount).toBe(0);
  });

  it("preserves revealed groups and expanded group rows across refreshes", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "docs-1", cwd: "/work/docs" },
          { session_id: "docs-2", cwd: "/work/docs" },
          { session_id: "ops-1", cwd: "/work/ops" },
          { session_id: "lab-1", cwd: "/work/lab" },
        ],
        remaining_by_group: { "/work/docs": 1 },
        omitted_group_count: 1,
      } as never)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "play-1", cwd: "/work/play" }],
        omitted_group_count: 0,
      } as never)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "docs-3", cwd: "/work/docs" }],
        remaining_by_group: {},
      } as never)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "docs-1", cwd: "/work/docs", busy: true },
          { session_id: "docs-2", cwd: "/work/docs" },
          { session_id: "ops-1", cwd: "/work/ops" },
          { session_id: "lab-1", cwd: "/work/lab" },
        ],
        remaining_by_group: { "/work/docs": 1 },
        omitted_group_count: 1,
      } as never)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "docs-1", cwd: "/work/docs", busy: true },
          { session_id: "docs-2", cwd: "/work/docs" },
          { session_id: "docs-3", cwd: "/work/docs" },
        ],
        remaining_by_group: {},
      } as never)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "play-1", cwd: "/work/play", busy: true }],
        remaining_by_group: {},
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    await store.loadMoreGroups(1);
    await store.loadMoreGroup("/work/docs", 5);
    await store.refresh();

    expect(store.getState().items).toEqual([
      { session_id: "docs-1", cwd: "/work/docs", busy: true },
      { session_id: "docs-2", cwd: "/work/docs" },
      { session_id: "docs-3", cwd: "/work/docs" },
      { session_id: "ops-1", cwd: "/work/ops" },
      { session_id: "lab-1", cwd: "/work/lab" },
      { session_id: "play-1", cwd: "/work/play", busy: true },
    ]);
    expect(store.getState().remainingByGroup).toEqual({});
    expect(store.getState().omittedGroupCount).toBe(0);
  });

  it("keeps an explicit selection across refreshes", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "s1" }, { session_id: "s2" }],
      } as never)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "s1" }, { session_id: "s2" }, { session_id: "s3" }],
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    store.select("s2");
    await store.refresh();

    expect(store.getState()).toEqual({
      items: [{ session_id: "s1" }, { session_id: "s2" }, { session_id: "s3" }],
      activeSessionId: "s2",
      loading: false,
      bootstrapLoaded: false,
      remainingByGroup: {},
      omittedGroupCount: 0,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });
  });

  it("clears loading when refresh fails", async () => {
    vi.mocked(api.listSessions).mockRejectedValue(new Error("boom"));
    const store = createSessionsStore();

    await expect(store.refresh()).rejects.toThrow("boom");
    expect(store.getState()).toEqual({
      items: [],
      activeSessionId: null,
      loading: false,
      bootstrapLoaded: false,
      remainingByGroup: {},
      omittedGroupCount: 0,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });
  });

  it("clears the selection when the selected session disappears", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "s1" }, { session_id: "s2" }],
      } as never)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "s3" }, { session_id: "s4" }],
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    store.select("s2");
    await store.refresh();

    expect(store.getState()).toEqual({
      items: [{ session_id: "s3" }, { session_id: "s4" }],
      activeSessionId: null,
      loading: false,
      bootstrapLoaded: false,
      remainingByGroup: {},
      omittedGroupCount: 0,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });
  });

  it("can prefer the newest session after a create flow", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "s1" }, { session_id: "s2" }],
      } as never)
      .mockResolvedValueOnce({
        sessions: [{ session_id: "s3" }, { session_id: "s1" }, { session_id: "s2" }],
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    store.select("s2");
    await store.refresh({ preferNewest: true });

    expect(store.getState().activeSessionId).toBe("s3");
  });

  it("ignores stale refresh responses", async () => {
    let resolveFirst: (v: any) => void;
    let resolveSecond: (v: any) => void;
    vi.mocked(api.listSessions)
      .mockReturnValueOnce(new Promise((r) => { resolveFirst = r; }) as never)
      .mockReturnValueOnce(new Promise((r) => { resolveSecond = r; }) as never);
    const store = createSessionsStore();

    const firstPromise = store.refresh();
    const secondPromise = store.refresh();

    resolveSecond!({ sessions: [{ session_id: "s2" }] });
    await secondPromise;
    expect(store.getState().activeSessionId).toBe("s2");

    resolveFirst!({ sessions: [{ session_id: "s1" }] });
    await firstPromise;
    expect(store.getState().activeSessionId).toBe("s2");
  });

  it("dedupes repeated session rows from the API by session id", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        { session_id: "s1", alias: "Newest" },
        { session_id: "s1", alias: "Newest" },
        { session_id: "s1", alias: "Newest" },
        { session_id: "s2", alias: "Older" },
      ],
    } as never);
    const store = createSessionsStore();

    await store.refresh();

    expect(store.getState().items).toEqual([
      { session_id: "s1", alias: "Newest" },
      { session_id: "s2", alias: "Older" },
    ]);
    expect(store.getState().activeSessionId).toBe("s1");
  });

  it("dedupes live session rows that point at the same backend thread", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        { session_id: "broker-a", thread_id: "thread-1", agent_backend: "pi", alias: "Newest" },
        { session_id: "broker-b", thread_id: "thread-1", agent_backend: "pi", alias: "Duplicate" },
        { session_id: "broker-c", thread_id: "thread-1", agent_backend: "pi", alias: "Duplicate again" },
        { session_id: "broker-d", thread_id: "thread-2", agent_backend: "pi", alias: "Other" },
      ],
    } as never);
    const store = createSessionsStore();

    await store.refresh();

    expect(store.getState().items).toEqual([
      { session_id: "broker-a", thread_id: "thread-1", agent_backend: "pi", alias: "Newest" },
      { session_id: "broker-d", thread_id: "thread-2", agent_backend: "pi", alias: "Other" },
    ]);
    expect(store.getState().activeSessionId).toBe("broker-a");
  });

  it("keeps a selected duplicate thread on its visible representative", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "broker-a", thread_id: "thread-1", agent_backend: "pi", alias: "Newest" },
          { session_id: "broker-b", thread_id: "thread-1", agent_backend: "pi", alias: "Duplicate" },
        ],
      } as never)
      .mockResolvedValueOnce({
        sessions: [
          { session_id: "broker-a", thread_id: "thread-1", agent_backend: "pi", alias: "Newest" },
          { session_id: "broker-b", thread_id: "thread-1", agent_backend: "pi", alias: "Duplicate" },
        ],
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    store.select("broker-b");
    await store.refresh();

    expect(store.getState().items).toEqual([
      { session_id: "broker-a", thread_id: "thread-1", agent_backend: "pi", alias: "Newest" },
    ]);
    expect(store.getState().activeSessionId).toBe("broker-a");
  });
});
