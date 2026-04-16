import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../../lib/api";
import { createMessagesStore } from "./store";

vi.mock("../../lib/api", () => ({
  api: {
    listMessages: vi.fn(),
  },
}));

describe("createMessagesStore", () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  it("loads initial messages for a session and clears loading state", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      events: [{ id: "m1" }, { id: "m2" }],
      offset: 2,
    } as never);
    const store = createMessagesStore();
    const snapshots: Array<Record<string, unknown>> = [];

    store.subscribe(() => {
      const state = store.getState();
      snapshots.push({
        loading: state.loading,
        messages: state.bySessionId.s1 ?? [],
      });
    });

    await store.loadInitial("s1");

    expect(api.listMessages).toHaveBeenCalledWith("s1", true, undefined, undefined);
    expect(snapshots).toEqual([
      { loading: true, messages: [] },
      { loading: false, messages: [{ id: "m1" }, { id: "m2" }] },
    ]);
    expect(store.getState()).toEqual({
      bySessionId: {
        s1: [{ id: "m1" }, { id: "m2" }],
      },
      offsetsBySessionId: {
        s1: 2,
      },
      hasOlderBySessionId: {
        s1: false,
      },
      olderBeforeBySessionId: {
        s1: 0,
      },
      loadingOlderBySessionId: {
        s1: false,
      },
      loadingBySessionId: {
        s1: false,
      },
      loadedBySessionId: {
        s1: true,
      },
      loading: false,
    });
  });

  it("replaces messages for a polled session without touching other sessions", async () => {
    vi.mocked(api.listMessages)
      .mockResolvedValueOnce({
        events: [{ id: "m1" }],
        offset: 1,
      } as never)
      .mockResolvedValueOnce({
        events: [{ id: "other" }],
        offset: 1,
      } as never)
      .mockResolvedValueOnce({
        events: [{ id: "m2" }],
        offset: 2,
      } as never);
    const store = createMessagesStore();

    await store.loadInitial("s1");
    await store.loadInitial("s2");
    await store.poll("s1");

    expect(api.listMessages).toHaveBeenNthCalledWith(1, "s1", true, undefined, undefined);
    expect(api.listMessages).toHaveBeenNthCalledWith(2, "s2", true, undefined, undefined);
    expect(api.listMessages).toHaveBeenNthCalledWith(3, "s1", false, undefined, 1);
    expect(store.getState()).toEqual({
      bySessionId: {
        s1: [{ id: "m1" }, { id: "m2" }],
        s2: [{ id: "other" }],
      },
      offsetsBySessionId: {
        s1: 2,
        s2: 1,
      },
      hasOlderBySessionId: {
        s1: false,
        s2: false,
      },
      olderBeforeBySessionId: {
        s1: 0,
        s2: 0,
      },
      loadingOlderBySessionId: {
        s1: false,
        s2: false,
      },
      loadingBySessionId: {
        s1: false,
        s2: false,
      },
      loadedBySessionId: {
        s1: true,
        s2: true,
      },
      loading: false,
    });
  });

  it("clears loading when message fetch fails", async () => {
    vi.mocked(api.listMessages).mockRejectedValue(new Error("boom"));
    const store = createMessagesStore();

    await expect(store.loadInitial("s1")).rejects.toThrow("boom");
    expect(store.getState()).toEqual({
      bySessionId: {},
      offsetsBySessionId: {},
      hasOlderBySessionId: {},
      olderBeforeBySessionId: {},
      loadingOlderBySessionId: {},
      loadingBySessionId: {
        s1: false,
      },
      loadedBySessionId: {},
      loading: false,
    });
  });

  it("ignores stale message responses when a newer explicit reload supersedes them", async () => {
    let resolveFirst: (v: any) => void;
    let resolveSecond: (v: any) => void;
    vi.mocked(api.listMessages)
      .mockReturnValueOnce(new Promise((r) => { resolveFirst = r; }))
      .mockReturnValueOnce(new Promise((r) => { resolveSecond = r; }));
    const store = createMessagesStore();

    const firstPromise = store.loadInitial("s1");
    const secondPromise = store.loadInitial("s1");

    resolveSecond!({ events: [{ id: "m2" }], offset: 2 });
    await secondPromise;
    expect(store.getState().bySessionId.s1).toEqual([{ id: "m2" }]);

    resolveFirst!({ events: [{ id: "m1" }], offset: 1 });
    await firstPromise;
    expect(store.getState().bySessionId.s1).toEqual([{ id: "m2" }]);
  });

  it("does not start an overlapping poll while the same session is still loading", async () => {
    let resolveFirst: (value: any) => void;
    vi.mocked(api.listMessages).mockReturnValueOnce(new Promise((resolve) => {
      resolveFirst = resolve;
    }) as never);
    const store = createMessagesStore();

    const initialPromise = store.loadInitial("s1");
    const pollPromise = store.poll("s1");

    expect(api.listMessages).toHaveBeenCalledTimes(1);
    expect(api.listMessages).toHaveBeenCalledWith("s1", true, undefined, undefined);

    resolveFirst!({ events: [{ id: "m1" }], offset: 1 });
    await Promise.all([initialPromise, pollPromise]);

    expect(store.getState().bySessionId.s1).toEqual([{ id: "m1" }]);
    expect(store.getState().offsetsBySessionId.s1).toBe(1);
  });

  it("passes the saved offset when polling", async () => {
    vi.mocked(api.listMessages)
      .mockResolvedValueOnce({ events: [{ id: "m1" }], offset: 4 } as never)
      .mockResolvedValueOnce({ events: [{ id: "m2" }], offset: 5 } as never);
    const store = createMessagesStore();

    await store.loadInitial("s1");
    await store.poll("s1");

    expect(api.listMessages).toHaveBeenNthCalledWith(2, "s1", false, undefined, 4);
    expect(store.getState().bySessionId.s1).toEqual([{ id: "m1" }, { id: "m2" }]);
  });

  it("applies live payloads by replacing then appending messages", () => {
    const store = createMessagesStore();

    store.applyLive("s1", [{ id: "m1" } as any], { replace: true, offset: 4 });
    store.applyLive("s1", [{ id: "m2" } as any], { replace: false, offset: 5 });

    expect(store.getState().bySessionId.s1).toEqual([{ id: "m1" }, { id: "m2" }]);
    expect(store.getState().offsetsBySessionId.s1).toBe(5);
    expect(store.getState().loadedBySessionId.s1).toBe(true);
  });

  it("upserts one streamed assistant row per stream_id and replaces it with durable history", () => {
    const store = createMessagesStore();

    store.applyLive(
      "s1",
      [{ role: "assistant", text: "hel", streaming: true, stream_id: "pi-stream:turn-001", turn_id: "turn-001" } as any],
      { replace: true, offset: 1 },
    );

    store.applyLive(
      "s1",
      [{ role: "assistant", text: "hello", streaming: true, stream_id: "pi-stream:turn-001", turn_id: "turn-001" } as any],
      { replace: false, offset: 2 },
    );

    expect(store.getState().bySessionId.s1).toEqual([
      { role: "assistant", text: "hello", streaming: true, stream_id: "pi-stream:turn-001", turn_id: "turn-001" },
    ]);

    store.applyLive(
      "s1",
      [{ role: "assistant", text: "hello", turn_id: "turn-001" } as any],
      { replace: false, offset: 3 },
    );

    expect(store.getState().bySessionId.s1).toEqual([
      { role: "assistant", text: "hello", turn_id: "turn-001" },
    ]);
  });

  it("prepends older replay pages and tracks the next history cursor", async () => {
    vi.mocked(api.listMessages)
      .mockResolvedValueOnce({
        events: [{ id: "m2" }, { id: "m3" }],
        offset: 4,
        has_older: true,
        next_before: 2,
      } as never)
      .mockResolvedValueOnce({
        events: [{ id: "m0" }, { id: "m1" }],
        offset: 4,
        has_older: false,
        next_before: 0,
      } as never);
    const store = createMessagesStore();

    await store.loadInitial("s1");
    await store.loadOlder("s1");

    expect(api.listMessages).toHaveBeenNthCalledWith(2, "s1", true, undefined, undefined, 2, 80);
    expect(store.getState().bySessionId.s1).toEqual([{ id: "m0" }, { id: "m1" }, { id: "m2" }, { id: "m3" }]);
    expect(store.getState().hasOlderBySessionId.s1).toBe(false);
    expect(store.getState().olderBeforeBySessionId.s1).toBe(0);
  });
});
