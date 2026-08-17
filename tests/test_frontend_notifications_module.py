from frontend_module_loader import module_path
import json
import subprocess
import textwrap


HELPERS_SOURCE = module_path("app_voice_helpers.js").read_text(encoding="utf-8")
NOTIFICATIONS_SOURCE = module_path("app_notifications.js").read_text(encoding="utf-8")


def test_notification_runtime_has_narrow_api_and_unread_owns_panel_active_state() -> None:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const storage = new Map([
          ["codoxear.notificationEnabled", "1"],
          ["codoxear.desktopNotificationsEnabled", "1"],
        ]);
        let feed = [];
        function node() {{
          return {{
            style: {{}}, dataset: {{}}, title: "", textContent: "", disabled: false,
            _children: [], _attrs: {{}},
            classList: {{
              values: new Set(),
              toggle(name, enabled) {{ enabled ? this.values.add(name) : this.values.delete(name); }},
              contains(name) {{ return this.values.has(name); }},
            }},
            setAttribute(name, value) {{ this._attrs[name] = String(value); }},
            append(...children) {{ this._children.push(...children); }},
            appendChild(child) {{ this._children.push(child); return child; }},
            replaceChildren(...children) {{ this._children = children; }},
          }};
        }}
        class NotificationCtor {{}}
        NotificationCtor.permission = "granted";
        const dom = {{
          notificationBtn: node(), notificationPanel: node(), notificationList: node(),
          notificationEmpty: node(), notificationClearBtn: node(), notificationEnableBtn: node(),
        }};
        const documentTarget = {{ createElement: () => node() }};
        const ctx = {{
          console, Notification: NotificationCtor,
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11 Linux" }}, document: documentTarget,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(HELPERS_SOURCE)}, ctx);
        vm.runInContext({json.dumps(NOTIFICATIONS_SOURCE)}, ctx);
        const runtime = ctx.window.CodoxearNotifications.createNotificationRuntime({{
          ...dom,
          isAppDisposed: () => false,
          api: async (url) => url.includes("/feed") ? {{ items: feed.splice(0) }} : {{ subscriptions: [] }},
          setToast() {{}}, handleAppAuthLoss() {{}}, resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
          storageGetItem: (key) => storage.get(key) || null,
          storageSetItem: (key, value) => storage.set(key, String(value)),
          storageRemoveItem: (key) => storage.delete(key),
          eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
          windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
          Notification: NotificationCtor, clearTimeout() {{}},
        }});
        (async () => {{
          await runtime.syncState({{ subscriptions: [], vapid_public_key: "key" }});
          const enabledWithoutUnread = {{
            enabled: runtime.enabledLocally(),
            active: dom.notificationBtn.classList.contains("active"),
            unread: dom.notificationBtn.dataset.unread,
          }};
          feed = [{{ message_id: "fresh", updated_ts: 1, notification_text: "Done", session_display_name: "Agent" }}];
          await runtime.pollFeed();
          const withUnread = {{
            active: dom.notificationBtn.classList.contains("active"),
            unread: dom.notificationBtn.dataset.unread,
          }};
          process.stdout.write(JSON.stringify({{
            api: Object.keys(runtime).sort(),
            enabledWithoutUnread,
            withUnread,
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)

    assert result["api"] == ["dispose", "enabledLocally", "pollFeed", "syncState"]
    assert result["enabledWithoutUnread"] == {"enabled": True, "active": False, "unread": "0"}
    assert result["withUnread"] == {"active": True, "unread": "1"}
