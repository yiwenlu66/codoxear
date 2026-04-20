import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../../lib/api";
import { createSessionUiStore } from "./store";

vi.mock("../../lib/api", () => ({
  api: {
    getWorkspace: vi.fn(),
  },
}));

describe("createSessionUiStore", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  function createDeferred<T>() {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }

  it("refreshes all workspace panels and emits loading transitions", async () => {
    vi.mocked(api.getWorkspace).mockResolvedValue({ diagnostics: { status: "ok" }, queue: { items: [] } } as any);

    const store = createSessionUiStore();
    await store.refresh("s1");

    expect(api.getWorkspace).toHaveBeenCalledWith("s1");
    expect(store.getState()).toEqual({
      sessionId: "s1",
      runtimeId: null,
      diagnostics: { status: "ok" },
      queue: { items: [] },
      loading: false,
    });
  });

  it("refreshes workspace data the same way for non-pi sessions", async () => {
    vi.mocked(api.getWorkspace).mockResolvedValue({ diagnostics: { status: "ok" }, queue: { items: [] } } as any);

    const store = createSessionUiStore();
    await store.refresh("s1", { agentBackend: "codex" });

    expect(api.getWorkspace).toHaveBeenCalledWith("s1");
    expect(store.getState().diagnostics).toEqual({ status: "ok" });
  });

  it("keeps same-session workspace data visible while a refresh is in flight", async () => {
    vi.mocked(api.getWorkspace).mockResolvedValueOnce({
      diagnostics: { todo_snapshot: { progress_text: "1/2 completed" } },
      queue: { items: [] },
    } as any);

    const nextWorkspace = createDeferred<Record<string, unknown>>();

    vi.mocked(api.getWorkspace).mockReturnValueOnce(nextWorkspace.promise as any);

    const store = createSessionUiStore();

    await store.refresh("s1");

    const refreshPromise = store.refresh("s1");

    expect(store.getState()).toEqual({
      sessionId: "s1",
      runtimeId: null,
      diagnostics: { todo_snapshot: { progress_text: "1/2 completed" } },
      queue: { items: [] },
      loading: true,
    });

    nextWorkspace.resolve({
      diagnostics: { todo_snapshot: { progress_text: "2/2 completed" } },
      queue: { items: ["queued"] },
    });
    await refreshPromise;

    expect(store.getState()).toEqual({
      sessionId: "s1",
      runtimeId: null,
      diagnostics: { todo_snapshot: { progress_text: "2/2 completed" } },
      queue: { items: ["queued"] },
      loading: false,
    });
  });

  it("clears workspace data immediately when switching sessions", async () => {
    vi.mocked(api.getWorkspace).mockResolvedValueOnce({
      diagnostics: { todo_snapshot: { progress_text: "1/2 completed" } },
      queue: { items: [] },
    } as any);

    const nextWorkspace = createDeferred<Record<string, unknown>>();

    vi.mocked(api.getWorkspace).mockReturnValueOnce(nextWorkspace.promise as any);

    const store = createSessionUiStore();

    await store.refresh("s1");

    const refreshPromise = store.refresh("s2");

    expect(store.getState()).toEqual({
      sessionId: "s2",
      runtimeId: null,
      diagnostics: null,
      queue: null,
      loading: true,
    });

    nextWorkspace.resolve({
      diagnostics: { todo_snapshot: { progress_text: "0/1 completed" } },
      queue: { items: [] },
    });
    await refreshPromise;

    expect(store.getState()).toEqual({
      sessionId: "s2",
      runtimeId: null,
      diagnostics: { todo_snapshot: { progress_text: "0/1 completed" } },
      queue: { items: [] },
      loading: false,
    });
  });

  it("reuses an in-flight refresh for the same session and runtime", async () => {
    const deferred = createDeferred<Record<string, unknown>>();
    vi.mocked(api.getWorkspace).mockReturnValueOnce(deferred.promise as any);

    const store = createSessionUiStore();
    const first = store.refresh("s1", { runtimeId: "rt-1" });
    const second = store.refresh("s1", { runtimeId: "rt-1" });

    expect(api.getWorkspace).toHaveBeenCalledTimes(1);

    deferred.resolve({
      runtime_id: "rt-1",
      diagnostics: { todo_snapshot: { progress_text: "1/1 completed" } },
      queue: { items: ["queued"] },
    });
    await Promise.all([first, second]);

    expect(store.getState()).toEqual({
      sessionId: "s1",
      runtimeId: "rt-1",
      diagnostics: { todo_snapshot: { progress_text: "1/1 completed" } },
      queue: { items: ["queued"] },
      loading: false,
    });
  });
});
