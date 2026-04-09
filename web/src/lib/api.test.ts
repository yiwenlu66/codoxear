import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { getJson } from "./http";
import type { MessagesResponse, SessionUiStateResponse, SessionsResponse } from "./types";

describe("getJson", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("throws server error messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        text: async () => '{"error":"bad request"}',
      }),
    );

    await expect(getJson("/api/sessions")).rejects.toThrow("bad request");
  });

  it("parses successful json responses", async () => {
    const payload: SessionsResponse = {
      sessions: [{ session_id: "s1", agent_backend: "pi", busy: true }],
      new_session_defaults: { default_backend: "pi" },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        text: async () => JSON.stringify(payload),
      }),
    );

    await expect(getJson<SessionsResponse>("/api/sessions")).resolves.toEqual(payload);
  });
});

describe("api", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests sessions with the provided abort signal", async () => {
    const signal = new AbortController().signal;
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"sessions":[]}',
    });
    vi.stubGlobal("fetch", fetchMock);

    const payload = await api.listSessions(signal);

    expect(payload).toEqual({ sessions: [] });
    expect(fetchMock).toHaveBeenCalledWith("api/sessions", {
      headers: { Accept: "application/json" },
      signal,
    });
  });

  it("builds the init messages route", async () => {
    const payload: MessagesResponse = { events: [], offset: 0, ui_version: "v1" };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payload),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.listMessages("session-1", true)).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/session-1/messages?init=1", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("includes offsets when polling messages", async () => {
    const payload: MessagesResponse = { events: [{ id: "m1" }], offset: 9 };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payload),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.listMessages("session-1", false, undefined, 4)).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/session-1/messages?offset=4", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("includes before and limit when loading older history pages", async () => {
    const payload: MessagesResponse = { events: [{ id: "m0" }], offset: 9, has_older: true, next_before: 12 };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payload),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.listMessages("session-1", true, undefined, undefined, 6, 40)).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/session-1/messages?init=1&before=6&limit=40", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests the session ui state", async () => {
    const payload: SessionUiStateResponse = {
      requests: [{ id: "ui-req-1", method: "select" }],
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payload),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getSessionUiState("pi-session")).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/pi-session/ui_state", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests file reads for a session path", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"kind":"text","text":"hello"}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getFileRead("pi-session", "src/main.tsx")).resolves.toEqual({ ok: true, kind: "text", text: "hello" });
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/pi-session/file/read?path=src%2Fmain.tsx", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests the session file list", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"files":["src/main.tsx"]}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getFiles("pi-session")).resolves.toEqual({ ok: true, files: ["src/main.tsx"] });
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/pi-session/file/list", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests git file versions for a session path", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"path":"src/main.tsx","current_text":"x"}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getGitFileVersions("pi-session", "src/main.tsx")).resolves.toEqual({ ok: true, path: "src/main.tsx", current_text: "x" });
    expect(fetchMock).toHaveBeenCalledWith("api/sessions/pi-session/git/file_versions?path=src%2Fmain.tsx", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests and saves harness settings", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        text: async () => '{"ok":true,"enabled":true}',
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        text: async () => '{"ok":true,"enabled":false}',
      });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getHarness("sess-1")).resolves.toEqual({ ok: true, enabled: true });
    await expect(api.saveHarness("sess-1", { enabled: false })).resolves.toEqual({ ok: true, enabled: false });

    expect(fetchMock).toHaveBeenNthCalledWith(1, "api/sessions/sess-1/harness", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
    expect(fetchMock).toHaveBeenNthCalledWith(2, "api/sessions/sess-1/harness", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ enabled: false }),
    }));
  });

  it("posts session edit payloads", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"alias":"Updated"}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.editSession("sess-1", {
      name: "Updated",
      priority_offset: 0.25,
      snooze_until: null,
      dependency_session_id: "sess-2",
    })).resolves.toEqual({ ok: true, alias: "Updated" });

    expect(fetchMock).toHaveBeenCalledWith("api/sessions/sess-1/edit", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        name: "Updated",
        priority_offset: 0.25,
        snooze_until: null,
        dependency_session_id: "sess-2",
      }),
    }));
  });

  it("requests voice settings", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"tts_base_url":"https://example.test/v1"}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getVoiceSettings()).resolves.toEqual({ ok: true, tts_base_url: "https://example.test/v1" });
    expect(fetchMock).toHaveBeenCalledWith("api/settings/voice", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("posts announcement listener heartbeats", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"active_listener_count":1}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.setAudioListener("listener-1", true)).resolves.toEqual({ ok: true, active_listener_count: 1 });
    expect(fetchMock).toHaveBeenCalledWith("api/audio/listener", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ client_id: "listener-1", enabled: true }),
    }));
  });

  it("requests the notification feed with a since cursor", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"items":[]}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getNotificationsFeed(123.5)).resolves.toEqual({ ok: true, items: [] });
    expect(fetchMock).toHaveBeenCalledWith("api/notifications/feed?since=123.5", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests notification message details by id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"notification_text":"ready"}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getNotificationMessage("msg-1")).resolves.toEqual({ ok: true, notification_text: "ready" });
    expect(fetchMock).toHaveBeenCalledWith("api/notifications/message?message_id=msg-1", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("requests notification subscription state", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"subscriptions":[]}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.getNotificationSubscriptionState()).resolves.toEqual({ ok: true, subscriptions: [] });
    expect(fetchMock).toHaveBeenCalledWith("api/notifications/subscription", {
      headers: { Accept: "application/json" },
      signal: undefined,
    });
  });

  it("posts notification subscriptions", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"subscriptions":[]}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.upsertNotificationSubscription({ subscription: { endpoint: "https://push.test/sub/1" } })).resolves.toEqual({ ok: true, subscriptions: [] });
    expect(fetchMock).toHaveBeenCalledWith("api/notifications/subscription", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ subscription: { endpoint: "https://push.test/sub/1" } }),
    }));
  });

  it("toggles notification subscriptions by endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"ok":true,"subscriptions":[]}',
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.toggleNotificationSubscription("https://push.test/sub/1", false)).resolves.toEqual({ ok: true, subscriptions: [] });
    expect(fetchMock).toHaveBeenCalledWith("api/notifications/subscription/toggle", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ endpoint: "https://push.test/sub/1", enabled: false }),
    }));
  });

  it("edits cwd group metadata", async () => {
    const payload = { ok: true, cwd: "/tmp", label: "New Label", collapsed: true };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payload),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.editCwdGroup({ cwd: "/tmp", label: "New Label", collapsed: true })).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("api/cwd_groups/edit", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ cwd: "/tmp", label: "New Label", collapsed: true }),
    }));
  });
});
