from frontend_module_loader import module_path
import json
import subprocess
import textwrap


HELPERS_SOURCE = module_path("app_voice_helpers.js").read_text(encoding="utf-8")
NOTIFICATIONS_SOURCE = module_path("app_notifications.js").read_text(encoding="utf-8")


def run_notification_module_harness() -> dict:
    """Exercise the notification runtime's delivery and toggle behavior.

    The bell is a pure browser-notification toggle: it mounts into the voice
    host, never builds a panel, delivers feed items as desktop notifications
    with sound, dedups repeats, and suppresses delivery while priming.
    """
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const storage = new Map([["codoxear.desktopNotificationsEnabled", "1"]]);
        const notifications = [];
        const sound = {{ resumes: 0, starts: 0, stops: 0 }};
        const toasts = [];
        const feedResponses = [];
        const subscriptionCalls = [];
        const createdIds = [];
        const byId = new Map();
        function node(tag = "div") {{
          return {{
            tag, style: {{}}, dataset: {{}}, title: "", textContent: "", disabled: false,
            _children: [], _attrs: {{}},
            classList: {{
              values: new Set(),
              toggle(name, enabled) {{ enabled ? this.values.add(name) : this.values.delete(name); }},
              contains(name) {{ return this.values.has(name); }},
            }},
            setAttribute(name, value) {{ this._attrs[name] = String(value); }},
            append(...children) {{ this._children.push(...children); }},
            appendChild(child) {{ this._children.push(child); return child; }},
            insertBefore(child, before) {{
              const index = before ? this._children.indexOf(before) : -1;
              if (index < 0) this._children.push(child); else this._children.splice(index, 0, child);
              return child;
            }},
            get firstChild() {{ return this._children[0] || null; }},
            replaceChildren(...children) {{ this._children = children; }},
          }};
        }}
        class NotificationCtor {{
          constructor(title, options) {{ this.title = title; this.body = options.body; this.tag = options.tag; notifications.push(this); }}
          close() {{}}
        }}
        NotificationCtor.permission = "granted";
        class AudioContextCtor {{
          constructor() {{ this.state = "suspended"; this.currentTime = 0; this.destination = {{}}; }}
          resume() {{ this.state = "running"; sound.resumes += 1; return Promise.resolve(); }}
          createOscillator() {{ return {{ frequency: {{}}, connect() {{}}, start() {{ sound.starts += 1; }}, stop() {{ sound.stops += 1; }} }}; }}
          createGain() {{ return {{ gain: {{ setValueAtTime() {{}}, exponentialRampToValueAtTime() {{}} }}, connect() {{}} }}; }}
          close() {{ return Promise.resolve(); }}
        }}
        const root = node("root");
        const voiceHost = node("host");
        voiceHost.appendChild(node("announceBtn"));
        const el = (tag, attrs = {{}}, children = []) => {{
          const created = node(tag);
          for (const [name, value] of Object.entries(attrs)) {{
            if (name === "class") created.className = value;
            else if (name === "text") created.textContent = value;
            else if (name === "html") created.innerHTML = value;
            else created.setAttribute(name, value);
          }}
          if (attrs.id) {{ createdIds.push(attrs.id); byId.set(attrs.id, created); }}
          created.append(...children);
          return created;
        }};
        const ctx = {{
          console, Notification: NotificationCtor,
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11 Linux x86_64" }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(HELPERS_SOURCE)}, ctx);
        vm.runInContext({json.dumps(NOTIFICATIONS_SOURCE)}, ctx);
        const runtime = ctx.window.CodoxearNotifications.createNotificationRuntime({{
          root, voiceHost, el, iconSvg: () => "<svg></svg>",
          isAppDisposed: () => false,
          api: async (url, options = {{}}) => {{
            if (url.includes("/api/notifications/feed")) return {{ items: feedResponses.shift() || [] }};
            subscriptionCalls.push([url, options.method || "GET", options.body || null]);
            return {{ subscriptions: [], vapid_public_key: "key" }};
          }},
          setToast(message) {{ toasts.push(message); }}, handleAppAuthLoss() {{}}, resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
          storageGetItem: (key) => storage.get(key) || null,
          storageSetItem: (key, value) => storage.set(key, String(value)),
          storageRemoveItem: (key) => storage.delete(key),
          eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
          windowTarget: ctx.window, navigatorTarget: ctx.navigator,
          Notification: NotificationCtor, AudioContext: AudioContextCtor, clearTimeout() {{}},
        }});
        (async () => {{
          const notificationBtn = byId.get("notificationBtn");
          await runtime.syncState({{ subscriptions: [], vapid_public_key: "key" }});
          const disabledBeforeToggle = {{
            enabled: runtime.enabledLocally(),
            active: notificationBtn.classList.contains("active"),
            title: notificationBtn.title,
            ariaLabel: notificationBtn._attrs["aria-label"],
          }};

          // Bell click enables notifications on this device.
          await notificationBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
          const enabledAfterToggle = {{
            enabled: runtime.enabledLocally(),
            active: notificationBtn.classList.contains("active"),
            title: notificationBtn.title,
            stored: storage.get("codoxear.notificationEnabled") || null,
          }};

          // Priming records known ids without delivering desktop notifications.
          feedResponses.push([
            {{ message_id: "primed", updated_ts: 1, session_display_name: "Agent", notification_text: "Booted", session_id: "s1" }},
          ]);
          await runtime.pollFeed({{ prime: true }});

          // A fresh feed item delivers one desktop notification with sound.
          feedResponses.push([
            {{ message_id: "fresh", updated_ts: 2, session_display_name: "Agent", notification_text: "Needs attention", session_id: "s1" }},
          ]);
          await runtime.pollFeed();

          // A server replay of the same item id must not deliver again.
          feedResponses.push([
            {{ message_id: "fresh", updated_ts: 2, session_display_name: "Agent", notification_text: "Needs attention", session_id: "s1" }},
          ]);
          await runtime.pollFeed();

          // Bell click again disables notifications on this device.
          await notificationBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
          const disabledAfterSecondToggle = {{
            enabled: runtime.enabledLocally(),
            active: notificationBtn.classList.contains("active"),
            title: notificationBtn.title,
            stored: storage.get("codoxear.notificationEnabled") || null,
          }};

          process.stdout.write(JSON.stringify({{
            api: Object.keys(runtime).sort(),
            disabledBeforeToggle,
            delivered: notifications.map((entry) => [entry.title, entry.body, entry.tag]),
            sound,
            enabledAfterToggle,
            disabledAfterSecondToggle,
            toasts,
            dom: {{
              createdIds,
              hostIds: voiceHost._children.map((child) => child._attrs.id || child.tag),
              rootChildren: root._children.length,
            }},
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def run_mobile_toggle_endpoint_harness() -> dict:
    """Bell toggling on a mobile-class device drives the push endpoints.

    Enabling must subscribe through the service worker and POST the
    subscription; disabling must POST the subscription toggle with the
    registered endpoint.
    """
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const storage = new Map();
        const apiCalls = [];
        const toasts = [];
        const byId = new Map();
        function node(tag = "div") {{
          return {{
            tag, style: {{}}, dataset: {{}}, title: "", textContent: "", disabled: false,
            _children: [], _attrs: {{}},
            classList: {{
              values: new Set(),
              toggle(name, enabled) {{ enabled ? this.values.add(name) : this.values.delete(name); }},
              contains(name) {{ return this.values.has(name); }},
            }},
            setAttribute(name, value) {{ this._attrs[name] = String(value); }},
            append(...children) {{ this._children.push(...children); }},
            appendChild(child) {{ this._children.push(child); return child; }},
            insertBefore(child, before) {{
              const index = before ? this._children.indexOf(before) : -1;
              if (index < 0) this._children.push(child); else this._children.splice(index, 0, child);
              return child;
            }},
            get firstChild() {{ return this._children[0] || null; }},
            replaceChildren(...children) {{ this._children = children; }},
          }};
        }}
        class NotificationCtor {{
          constructor(title, options) {{ this.title = title; this.body = options.body; }}
          close() {{}}
        }}
        NotificationCtor.permission = "granted";
        class AudioContextCtor {{
          constructor() {{ this.state = "running"; this.currentTime = 0; this.destination = {{}}; }}
          resume() {{ return Promise.resolve(); }}
          createOscillator() {{ return {{ frequency: {{}}, connect() {{}}, start() {{}}, stop() {{}} }}; }}
          createGain() {{ return {{ gain: {{ setValueAtTime() {{}}, exponentialRampToValueAtTime() {{}} }}, connect() {{}} }}; }}
          close() {{ return Promise.resolve(); }}
        }}
        const subscription = {{
          endpoint: "https://push.example/sub/1",
          toJSON() {{ return {{ endpoint: this.endpoint, keys: {{ p256dh: "k", auth: "a" }} }}; }},
        }};
        const registration = {{
          pushManager: {{
            _subscription: null,
            async getSubscription() {{ return this._subscription; }},
            async subscribe() {{ this._subscription = subscription; return subscription; }},
          }},
        }};
        const navigatorTarget = {{
          userAgent: "iPhone",
          serviceWorker: {{ register: async () => registration }},
        }};
        const windowTarget = {{ isSecureContext: true, PushManager: function PushManager() {{}} }};
        const voiceHost = node("host");
        voiceHost.appendChild(node("announceBtn"));
        const el = (tag, attrs = {{}}, children = []) => {{
          const created = node(tag);
          for (const [name, value] of Object.entries(attrs)) {{
            if (name === "class") created.className = value;
            else if (name === "text") created.textContent = value;
            else if (name === "html") created.innerHTML = value;
            else created.setAttribute(name, value);
          }}
          if (attrs.id) byId.set(attrs.id, created);
          created.append(...children);
          return created;
        }};
        const ctx = {{ console, Notification: NotificationCtor, atob: () => "raw-bytes", window: windowTarget, navigator: navigatorTarget }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(HELPERS_SOURCE)}, ctx);
        vm.runInContext({json.dumps(NOTIFICATIONS_SOURCE)}, ctx);
        const runtime = ctx.window.CodoxearNotifications.createNotificationRuntime({{
          root: node("root"), voiceHost, el, iconSvg: () => "",
          isAppDisposed: () => false,
          api: async (url, options = {{}}) => {{
            apiCalls.push([url, options.method || "GET", options.body || null]);
            return {{ subscriptions: [], vapid_public_key: "key" }};
          }},
          setToast(message) {{ toasts.push(message); }}, handleAppAuthLoss() {{}}, resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
          storageGetItem: (key) => storage.get(key) || null,
          storageSetItem: (key, value) => storage.set(key, String(value)),
          storageRemoveItem: (key) => storage.delete(key),
          eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
          focusSessionFromNotification() {{}},
          windowTarget, navigatorTarget,
          Notification: NotificationCtor, AudioContext: AudioContextCtor, clearTimeout() {{}},
        }});
        (async () => {{
          const notificationBtn = byId.get("notificationBtn");
          await notificationBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
          const afterEnable = {{
            enabled: runtime.enabledLocally(),
            active: notificationBtn.classList.contains("active"),
            subscribed: registration.pushManager._subscription === subscription,
          }};
          await notificationBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
          const afterDisable = {{
            enabled: runtime.enabledLocally(),
            active: notificationBtn.classList.contains("active"),
          }};
          process.stdout.write(JSON.stringify({{
            afterEnable,
            afterDisable,
            toasts,
            posts: apiCalls.filter((call) => call[1] === "POST"),
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def test_bell_toggle_posts_push_subscription_and_toggle_endpoints() -> None:
    result = run_mobile_toggle_endpoint_harness()

    posts = result["posts"]
    assert len(posts) == 2
    subscribe_url, subscribe_method, subscribe_body = posts[0]
    assert subscribe_url == "/api/notifications/subscription"
    assert subscribe_method == "POST"
    assert subscribe_body["subscription"]["endpoint"] == "https://push.example/sub/1"
    assert subscribe_body["device_class"] == "mobile"
    toggle_url, toggle_method, toggle_body = posts[1]
    assert toggle_url == "/api/notifications/subscription/toggle"
    assert toggle_method == "POST"
    assert toggle_body == {"endpoint": "https://push.example/sub/1", "enabled": False}
    assert result["afterEnable"] == {"enabled": True, "active": True, "subscribed": True}
    assert result["afterDisable"] == {"enabled": False, "active": False}
    assert result["toasts"] == []


def test_notification_runtime_is_a_delivery_and_toggle_surface_without_panel() -> None:
    result = run_notification_module_harness()

    assert result["api"] == ["dispose", "enabledLocally", "pollFeed", "syncState"]
    # The only DOM the runtime creates is the bell button, hosted before the
    # announce button in the voice host; nothing is appended to the app root.
    assert result["dom"] == {
        "createdIds": ["notificationBtn"],
        "hostIds": ["notificationBtn", "announceBtn"],
        "rootChildren": 0,
    }
    # Disabled device: bell is idle with the off label.
    assert result["disabledBeforeToggle"] == {
        "enabled": False,
        "active": False,
        "title": "Notifications off",
        "ariaLabel": "Notifications off",
    }
    # Priming suppressed delivery; exactly one desktop notification for the
    # fresh item carries the feed's session name and text, and the replayed
    # item id delivered nothing. Sound was primed once on enable, then played
    # once for the delivered item.
    assert result["delivered"] == [["Agent", "Needs attention", "fresh"]]
    assert result["sound"] == {"starts": 1, "stops": 1, "resumes": 1}
    # Bell click enables: local flag persisted, active state and label flip.
    assert result["enabledAfterToggle"] == {
        "enabled": True,
        "active": True,
        "title": "Notifications on",
        "stored": "1",
    }
    # Second click disables: flag cleared, active state drops, label flips back.
    assert result["disabledAfterSecondToggle"] == {
        "enabled": False,
        "active": False,
        "title": "Notifications off",
        "stored": None,
    }
    assert result["toasts"] == []
