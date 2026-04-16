/** @jsxImportSource preact */
import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, expect, it, vi } from "vitest";
import { useAppShellSessionEffects } from "./useAppShellSessionEffects";

function Harness(props: Parameters<typeof useAppShellSessionEffects>[0]) {
  useAppShellSessionEffects(props);
  return <div data-testid="session-effects" />;
}

function baseProps(overrides: Partial<Parameters<typeof useAppShellSessionEffects>[0]> = {}): Parameters<typeof useAppShellSessionEffects>[0] {
  return {
    activeSessionBackend: "pi",
    activeSessionHistorical: false,
    activeSessionId: "sess-1",
    activeSessionLiveBusy: false,
    items: [{ session_id: "sess-1", busy: false }] as any,
    liveSessionStoreApi: { loadInitial: vi.fn().mockResolvedValue(undefined), poll: vi.fn().mockResolvedValue(undefined) } as any,
    replySoundEnabled: false,
    sessionUiStoreApi: { refresh: vi.fn().mockResolvedValue(undefined) } as any,
    sessionsStoreApi: { refresh: vi.fn().mockResolvedValue(undefined) } as any,
    workspaceOpen: false,
    activeSessionReplySoundPrimingRef: { current: null },
    backgroundReplySoundPrimedSessionIdsRef: { current: new Set<string>() },
    suppressedReplySoundSessionIdsRef: { current: new Set<string>() },
    ...overrides,
  };
}

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

afterEach(() => {
  document.body.innerHTML = "";
  setDocumentVisibility("visible");
  vi.useRealTimers();
  vi.clearAllMocks();
});

it("refreshes sessions immediately and every 5 seconds while any session is busy", async () => {
  vi.useFakeTimers();
  const sessionsStoreApi = { refresh: vi.fn().mockResolvedValue(undefined) } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionBackend="pi"
        activeSessionId={null}
        activeSessionLiveBusy={false}
        items={[{ session_id: "sess-1", busy: true }] as any}
        liveSessionStoreApi={{ loadInitial: vi.fn(), poll: vi.fn() } as any}
        replySoundEnabled={false}
        sessionUiStoreApi={{ refresh: vi.fn() } as any}
        sessionsStoreApi={sessionsStoreApi}
        workspaceOpen={false}
        activeSessionReplySoundPrimingRef={{ current: null }}
        backgroundReplySoundPrimedSessionIdsRef={{ current: new Set<string>() }}
        suppressedReplySoundSessionIdsRef={{ current: new Set<string>() }}
      />,
      root,
    );
    await flush();
  });

  expect(sessionsStoreApi.refresh).toHaveBeenCalledTimes(1);

  await act(async () => {
    vi.advanceTimersByTime(5000);
    await Promise.resolve();
  });

  expect(sessionsStoreApi.refresh).toHaveBeenCalledTimes(2);
});

it("polls the active busy session every 2 seconds and workspace every 15 seconds when open", async () => {
  vi.useFakeTimers();
  const liveSessionStoreApi = {
    loadInitial: vi.fn().mockResolvedValue(undefined),
    poll: vi.fn().mockResolvedValue(undefined),
  } as any;
  const sessionUiStoreApi = { refresh: vi.fn().mockResolvedValue(undefined) } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionBackend="pi"
        activeSessionId="sess-1"
        activeSessionLiveBusy={false}
        items={[{ session_id: "sess-1", busy: true }] as any}
        liveSessionStoreApi={liveSessionStoreApi}
        replySoundEnabled={false}
        sessionUiStoreApi={sessionUiStoreApi}
        sessionsStoreApi={{ refresh: vi.fn().mockResolvedValue(undefined) } as any}
        workspaceOpen={true}
        activeSessionReplySoundPrimingRef={{ current: null }}
        backgroundReplySoundPrimedSessionIdsRef={{ current: new Set<string>() }}
        suppressedReplySoundSessionIdsRef={{ current: new Set<string>() }}
      />,
      root,
    );
    await flush();
  });

  expect(liveSessionStoreApi.loadInitial).toHaveBeenCalledWith("sess-1");
  expect(sessionUiStoreApi.refresh).toHaveBeenCalledWith("sess-1", { agentBackend: "pi" });

  await act(async () => {
    vi.advanceTimersByTime(2000);
    await Promise.resolve();
  });
  expect(liveSessionStoreApi.poll).toHaveBeenCalledWith("sess-1");
  expect(sessionUiStoreApi.refresh).toHaveBeenCalledTimes(1);

  await act(async () => {
    vi.advanceTimersByTime(13000);
    await Promise.resolve();
  });
  expect(sessionUiStoreApi.refresh).toHaveBeenCalledTimes(2);
});

it("slows active idle live polling to every 12 seconds", async () => {
  vi.useFakeTimers();
  const liveSessionStoreApi = {
    loadInitial: vi.fn().mockResolvedValue(undefined),
    poll: vi.fn().mockResolvedValue(undefined),
  } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionBackend="pi"
        activeSessionId="sess-1"
        activeSessionLiveBusy={false}
        items={[{ session_id: "sess-1", busy: false }] as any}
        liveSessionStoreApi={liveSessionStoreApi}
        replySoundEnabled={false}
        sessionUiStoreApi={{ refresh: vi.fn() } as any}
        sessionsStoreApi={{ refresh: vi.fn().mockResolvedValue(undefined) } as any}
        workspaceOpen={false}
        activeSessionReplySoundPrimingRef={{ current: null }}
        backgroundReplySoundPrimedSessionIdsRef={{ current: new Set<string>() }}
        suppressedReplySoundSessionIdsRef={{ current: new Set<string>() }}
      />,
      root,
    );
    await flush();
  });

  await act(async () => {
    vi.advanceTimersByTime(11999);
    await Promise.resolve();
  });
  expect(liveSessionStoreApi.poll).toHaveBeenCalledTimes(0);

  await act(async () => {
    vi.advanceTimersByTime(1);
    await Promise.resolve();
  });
  expect(liveSessionStoreApi.poll).toHaveBeenCalledTimes(1);
});

it("switches to fast live polling as soon as live state reports the active session busy", async () => {
  vi.useFakeTimers();
  const liveSessionStoreApi = {
    loadInitial: vi.fn().mockResolvedValue(undefined),
    poll: vi.fn().mockResolvedValue(undefined),
  } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(<Harness {...baseProps({ liveSessionStoreApi })} />, root);
    await flush();
  });

  await act(async () => {
    render(<Harness {...baseProps({ liveSessionStoreApi, activeSessionLiveBusy: true })} />, root);
    await flush();
  });

  await act(async () => {
    vi.advanceTimersByTime(2000);
    await Promise.resolve();
  });

  expect(liveSessionStoreApi.poll).toHaveBeenCalledTimes(1);
  expect(liveSessionStoreApi.poll).toHaveBeenCalledWith("sess-1");
});

it("skips live and workspace polling for historical pi sessions", async () => {
  vi.useFakeTimers();
  const liveSessionStoreApi = {
    loadInitial: vi.fn().mockResolvedValue(undefined),
    poll: vi.fn().mockResolvedValue(undefined),
  } as any;
  const sessionUiStoreApi = { refresh: vi.fn().mockResolvedValue(undefined) } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionBackend="pi"
        activeSessionHistorical={true}
        activeSessionId="history:pi:resume-hist"
        activeSessionLiveBusy={false}
        items={[{ session_id: "history:pi:resume-hist", busy: false, historical: true, agent_backend: "pi" }] as any}
        liveSessionStoreApi={liveSessionStoreApi}
        replySoundEnabled={false}
        sessionUiStoreApi={sessionUiStoreApi}
        sessionsStoreApi={{ refresh: vi.fn().mockResolvedValue(undefined) } as any}
        workspaceOpen={true}
        activeSessionReplySoundPrimingRef={{ current: null }}
        backgroundReplySoundPrimedSessionIdsRef={{ current: new Set<string>() }}
        suppressedReplySoundSessionIdsRef={{ current: new Set<string>() }}
      />,
      root,
    );
    await flush();
  });

  expect(liveSessionStoreApi.loadInitial).not.toHaveBeenCalled();
  expect(liveSessionStoreApi.poll).not.toHaveBeenCalled();
  expect(sessionUiStoreApi.refresh).not.toHaveBeenCalled();

  await act(async () => {
    vi.advanceTimersByTime(12000);
    await Promise.resolve();
  });

  expect(liveSessionStoreApi.poll).not.toHaveBeenCalled();
});

it("pauses live and workspace polling while hidden and refreshes immediately on resume", async () => {
  vi.useFakeTimers();
  setDocumentVisibility("hidden");
  const sessionsStoreApi = { refresh: vi.fn().mockResolvedValue(undefined) } as any;
  const liveSessionStoreApi = {
    loadInitial: vi.fn().mockResolvedValue(undefined),
    poll: vi.fn().mockResolvedValue(undefined),
  } as any;
  const sessionUiStoreApi = { refresh: vi.fn().mockResolvedValue(undefined) } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionBackend="pi"
        activeSessionId="sess-1"
        activeSessionLiveBusy={false}
        items={[{ session_id: "sess-1", busy: true }] as any}
        liveSessionStoreApi={liveSessionStoreApi}
        replySoundEnabled={false}
        sessionUiStoreApi={sessionUiStoreApi}
        sessionsStoreApi={sessionsStoreApi}
        workspaceOpen={true}
        activeSessionReplySoundPrimingRef={{ current: null }}
        backgroundReplySoundPrimedSessionIdsRef={{ current: new Set<string>() }}
        suppressedReplySoundSessionIdsRef={{ current: new Set<string>() }}
      />,
      root,
    );
    await flush();
  });

  expect(sessionsStoreApi.refresh).toHaveBeenCalledTimes(0);
  expect(liveSessionStoreApi.loadInitial).toHaveBeenCalledTimes(0);
  expect(sessionUiStoreApi.refresh).toHaveBeenCalledTimes(0);

  await act(async () => {
    setDocumentVisibility("visible");
    document.dispatchEvent(new Event("visibilitychange"));
    await flush();
  });

  expect(sessionsStoreApi.refresh).toHaveBeenCalledTimes(1);
  expect(liveSessionStoreApi.loadInitial).toHaveBeenCalledWith("sess-1");
  expect(sessionUiStoreApi.refresh).toHaveBeenCalledWith("sess-1", { agentBackend: "pi" });
});

it("primes and polls background busy sessions every 5 seconds when reply sounds are enabled", async () => {
  vi.useFakeTimers();
  const liveSessionStoreApi = {
    loadInitial: vi.fn().mockResolvedValue(undefined),
    poll: vi.fn().mockResolvedValue(undefined),
  } as any;

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionBackend="pi"
        activeSessionId="sess-1"
        activeSessionLiveBusy={false}
        items={[
          { session_id: "sess-1", busy: true },
          { session_id: "sess-2", busy: true },
        ] as any}
        liveSessionStoreApi={liveSessionStoreApi}
        replySoundEnabled={true}
        sessionUiStoreApi={{ refresh: vi.fn() } as any}
        sessionsStoreApi={{ refresh: vi.fn().mockResolvedValue(undefined) } as any}
        workspaceOpen={false}
        activeSessionReplySoundPrimingRef={{ current: null }}
        backgroundReplySoundPrimedSessionIdsRef={{ current: new Set<string>() }}
        suppressedReplySoundSessionIdsRef={{ current: new Set<string>() }}
      />,
      root,
    );
    await flush();
  });

  expect(liveSessionStoreApi.loadInitial).toHaveBeenCalledWith("sess-1");
  expect(liveSessionStoreApi.loadInitial).toHaveBeenCalledWith("sess-2");
  await flush();

  await act(async () => {
    vi.advanceTimersByTime(5000);
    await Promise.resolve();
    await Promise.resolve();
  });

  expect(liveSessionStoreApi.poll.mock.calls.map((call: [string]) => call[0])).toContain("sess-2");
});
