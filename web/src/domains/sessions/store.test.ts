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
      remainingCount: 0,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });
    expect(api.listSessions).toHaveBeenCalledWith({ limit: 50 });
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
      remainingCount: 0,
      newSessionDefaults: { default_backend: "pi" },
      recentCwds: ["/tmp/project"],
      cwdGroups: { "/tmp/project": { label: "Project", collapsed: true } },
      tmuxAvailable: true,
    });
  });

  it("loads more rows through flat pagination", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: Array.from({ length: 50 }, (_, index) => ({ session_id: `sess-${index + 1}` })),
        remaining_count: 2,
      } as never)
      .mockResolvedValueOnce({
        sessions: Array.from({ length: 52 }, (_, index) => ({ session_id: `sess-${index + 1}` })),
        remaining_count: 0,
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    await store.loadMore();

    expect(api.listSessions).toHaveBeenNthCalledWith(1, { limit: 50 });
    expect(api.listSessions).toHaveBeenNthCalledWith(2, { limit: 100 });
    expect(store.getState().items).toHaveLength(52);
    expect(store.getState().remainingCount).toBe(0);
  });

  it("preserves the expanded flat-list limit across refreshes", async () => {
    vi.mocked(api.listSessions)
      .mockResolvedValueOnce({
        sessions: Array.from({ length: 50 }, (_, index) => ({ session_id: `sess-${index + 1}`, focused: index === 10 })),
        remaining_count: 1,
      } as never)
      .mockResolvedValueOnce({
        sessions: Array.from({ length: 51 }, (_, index) => ({ session_id: `sess-${index + 1}`, focused: index === 10 })),
        remaining_count: 0,
      } as never)
      .mockResolvedValueOnce({
        sessions: Array.from({ length: 51 }, (_, index) => ({ session_id: `sess-${index + 1}`, focused: index === 10 })),
        remaining_count: 0,
      } as never);
    const store = createSessionsStore();

    await store.refresh();
    await store.loadMore();
    await store.refresh();

    expect(api.listSessions).toHaveBeenNthCalledWith(3, { limit: 100 });
    expect(store.getState().items).toHaveLength(51);
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

    expect(store.getState().activeSessionId).toBe("s2");
  });

  it("prefers the newest session when requested", async () => {
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

  it("dedupes live rows by backend thread id", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        { session_id: "runtime-1", thread_id: "thread-a", agent_backend: "pi" },
        { session_id: "runtime-2", thread_id: "thread-a", agent_backend: "pi" },
        { session_id: "runtime-3", thread_id: "thread-b", agent_backend: "pi" },
      ],
    } as never);
    const store = createSessionsStore();

    await store.refresh();

    expect(store.getState().items.map((session) => session.session_id)).toEqual(["runtime-1", "runtime-3"]);
  });

  it("reuses an in-flight refresh for repeated identical requests", async () => {
    let resolveRefresh!: (value: unknown) => void;
    vi.mocked(api.listSessions).mockReturnValueOnce(new Promise((resolve) => {
      resolveRefresh = resolve;
    }) as never);
    const store = createSessionsStore();

    const first = store.refresh();
    const second = store.refresh();

    expect(api.listSessions).toHaveBeenCalledTimes(1);
    resolveRefresh({ sessions: [{ session_id: "s1" }] });
    await Promise.all([first, second]);

    expect(store.getState().items).toEqual([{ session_id: "s1" }]);
  });
});
