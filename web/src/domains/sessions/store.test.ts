import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../../lib/api";
import { createSessionsStore } from "./store";

vi.mock("../../lib/api", () => ({
  api: {
    listSessions: vi.fn(),
  },
}));

describe("createSessionsStore", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("keeps no session selected on the first refresh until the user explicitly picks one", async () => {
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

    expect(snapshots).toEqual(["true:null", "false:null"]);
    expect(store.getState()).toEqual({
      items: [{ session_id: "s1" }, { session_id: "s2" }],
      activeSessionId: null,
      loading: false,
      newSessionDefaults: null,
      recentCwds: [],
      cwdGroups: {},
      tmuxAvailable: false,
    });
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
      .mockReturnValueOnce(new Promise((r) => { resolveFirst = r; }))
      .mockReturnValueOnce(new Promise((r) => { resolveSecond = r; }));
    const store = createSessionsStore();

    const firstPromise = store.refresh();
    const secondPromise = store.refresh();

    resolveSecond!({ sessions: [{ session_id: "s2" }] });
    await secondPromise;
    expect(store.getState().activeSessionId).toBeNull();

    resolveFirst!({ sessions: [{ session_id: "s1" }] });
    await firstPromise;
    expect(store.getState().activeSessionId).toBeNull();
  });

  it("stores cwd groups from the API", async () => {
    const cwdGroups = { "/tmp": { label: "Temp", collapsed: true } };
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [],
      cwd_groups: cwdGroups,
    } as never);
    const store = createSessionsStore();

    await store.refresh();

    expect(store.getState().cwdGroups).toEqual(cwdGroups);
  });
});
