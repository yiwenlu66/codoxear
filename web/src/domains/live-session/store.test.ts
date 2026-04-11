import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../../lib/api";
import { createMessagesStore } from "../messages/store";
import { createLiveSessionStore } from "./store";

vi.mock("../../lib/api", () => ({
  api: {
    getLiveSession: vi.fn(),
  },
}));

describe("createLiveSessionStore", () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  it("loads initial live data into messages and live state", async () => {
    vi.mocked(api.getLiveSession).mockResolvedValue({
      events: [{ id: "m1" }],
      requests: [{ id: "r1", method: "select" }],
      busy: true,
      offset: 3,
    } as never);
    const messagesStore = createMessagesStore();
    const liveStore = createLiveSessionStore(messagesStore);

    await liveStore.loadInitial("s1");

    expect(api.getLiveSession).toHaveBeenCalledWith("s1", undefined, undefined);
    expect(messagesStore.getState().bySessionId.s1).toEqual([{ id: "m1" }]);
    expect(liveStore.getState().offsetsBySessionId.s1).toBe(3);
    expect(liveStore.getState().requestsBySessionId.s1).toEqual([{ id: "r1", method: "select" }]);
    expect(liveStore.getState().busyBySessionId.s1).toBe(true);
  });

  it("polls live data with the saved offset and appends messages", async () => {
    vi.mocked(api.getLiveSession)
      .mockResolvedValueOnce({
        events: [{ id: "m1" }],
        requests: [{ id: "r1" }],
        busy: true,
        offset: 3,
      } as never)
      .mockResolvedValueOnce({
        events: [{ id: "m2" }],
        requests: [{ id: "r2" }],
        busy: false,
        offset: 4,
      } as never);
    const messagesStore = createMessagesStore();
    const liveStore = createLiveSessionStore(messagesStore);

    await liveStore.loadInitial("s1");
    await liveStore.poll("s1");

    expect(api.getLiveSession).toHaveBeenNthCalledWith(2, "s1", 3, undefined);
    expect(messagesStore.getState().bySessionId.s1).toEqual([{ id: "m1" }, { id: "m2" }]);
    expect(liveStore.getState().offsetsBySessionId.s1).toBe(4);
    expect(liveStore.getState().requestsBySessionId.s1).toEqual([{ id: "r2" }]);
    expect(liveStore.getState().busyBySessionId.s1).toBe(false);
  });

  it("preserves cached requests when the server omits unchanged requests", async () => {
    vi.mocked(api.getLiveSession)
      .mockResolvedValueOnce({
        events: [{ id: "m1" }],
        requests: [{ id: "r1" }],
        requests_version: "v1",
        busy: true,
        offset: 3,
      } as never)
      .mockResolvedValueOnce({
        events: [{ id: "m2" }],
        requests_version: "v1",
        busy: false,
        offset: 4,
      } as never);
    const messagesStore = createMessagesStore();
    const liveStore = createLiveSessionStore(messagesStore);

    await liveStore.loadInitial("s1");
    await liveStore.poll("s1");

    expect(api.getLiveSession).toHaveBeenNthCalledWith(2, "s1", 3, "v1");
    expect(liveStore.getState().requestsBySessionId.s1).toEqual([{ id: "r1" }]);
  });

  it("does not start overlapping live polls for the same session", async () => {
    let resolveFirst!: (value: unknown) => void;
    vi.mocked(api.getLiveSession).mockReturnValueOnce(new Promise((resolve) => {
      resolveFirst = resolve;
    }) as never);
    const messagesStore = createMessagesStore();
    const liveStore = createLiveSessionStore(messagesStore);

    const first = liveStore.poll("s1");
    const second = liveStore.poll("s1");

    expect(api.getLiveSession).toHaveBeenCalledTimes(1);
    resolveFirst({ events: [{ id: "m1" }], requests: [], busy: false, offset: 1 });
    await Promise.all([first, second]);

    expect(messagesStore.getState().bySessionId.s1).toEqual([{ id: "m1" }]);
    expect(liveStore.getState().loadingBySessionId.s1).toBe(false);
  });
});
