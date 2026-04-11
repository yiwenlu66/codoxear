/** @jsxImportSource preact */
import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, expect, it, vi } from "vitest";
import { useAppShellNotifications } from "./useAppShellNotifications";

vi.mock("../../lib/api", () => ({
  api: {
    getNotificationMessage: vi.fn().mockResolvedValue({ ok: true, notification_text: "" }),
    getNotificationSubscriptionState: vi.fn().mockResolvedValue({ ok: true, subscriptions: [] }),
    getNotificationsFeed: vi.fn().mockResolvedValue({ ok: true, items: [] }),
    toggleNotificationSubscription: vi.fn().mockResolvedValue({ ok: true, subscriptions: [] }),
    upsertNotificationSubscription: vi.fn().mockResolvedValue({ ok: true, subscriptions: [] }),
  },
}));

function Harness({
  activeSessionId,
  bySessionId,
  voiceSettings,
}: {
  activeSessionId: string | null;
  bySessionId: Record<string, unknown[]>;
  voiceSettings: any;
}) {
  const state = useAppShellNotifications({
    activeSessionId,
    activeTitle: "Legacy shell",
    bySessionId,
    playReplyBeep: vi.fn(),
    suppressedReplySoundSessionIdsRef: { current: new Set<string>() },
    voiceSettings,
  });

  return (
    <div>
      <div
        data-label={state.notificationLabel}
        data-push-enabled={state.pushNotificationsEnabled ? "yes" : "no"}
        data-reply-sound={state.replySoundEnabled ? "yes" : "no"}
      />
      <button type="button" onClick={() => { void state.toggleNotifications(); }}>Toggle</button>
    </div>
  );
}

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

afterEach(() => {
  document.body.innerHTML = "";
  localStorage.clear();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

it("dedupes desktop notifications by message id", async () => {
  const notificationSpy = vi.fn();
  vi.stubGlobal("Notification", class NotificationMock {
    static permission = "granted";
    static requestPermission = vi.fn().mockResolvedValue("granted");

    constructor(title: string, options?: NotificationOptions) {
      notificationSpy(title, options);
    }
  } as any);
  localStorage.setItem("codoxear.notificationEnabled", "1");

  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionId="sess-1"
        bySessionId={{
          "sess-1": [
            { role: "assistant", message_class: "final_response", message_id: "msg-1", notification_text: "done" },
            { role: "assistant", message_class: "final_response", message_id: "msg-1", notification_text: "done" },
          ],
        }}
        voiceSettings={{ notifications: { vapid_public_key: "" } }}
      />,
      root,
    );
    await flush();
  });

  expect(notificationSpy).toHaveBeenCalledTimes(1);
  expect(notificationSpy).toHaveBeenCalledWith("Legacy shell", expect.objectContaining({ body: "done" }));
});

it("reads the persisted reply-sound preference", async () => {
  localStorage.setItem("codoxear.replySoundEnabled", "0");
  const root = document.createElement("div");
  document.body.appendChild(root);

  await act(async () => {
    render(
      <Harness
        activeSessionId="sess-1"
        bySessionId={{ "sess-1": [] }}
        voiceSettings={{ notifications: { vapid_public_key: "ZmFrZS1rZXk" } }}
      />,
      root,
    );
    await flush();
  });

  expect(root.querySelector("[data-reply-sound]")?.getAttribute("data-reply-sound")).toBe("no");
});
