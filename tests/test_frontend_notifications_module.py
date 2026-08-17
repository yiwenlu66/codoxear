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
        class NotificationCtor {{}}
        NotificationCtor.permission = "granted";
        const root = node("root");
        const voiceHost = node("host");
        const announceBtn = node("button");
        announceBtn.setAttribute("id", "announceBtn");
        voiceHost.appendChild(announceBtn);
        const byId = new Map();
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
        const documentTarget = {{ createElement: () => node() }};
        const ctx = {{
          console, Notification: NotificationCtor,
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11 Linux" }}, document: documentTarget,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(HELPERS_SOURCE)}, ctx);
        vm.runInContext({json.dumps(NOTIFICATIONS_SOURCE)}, ctx);
        const runtime = ctx.window.CodoxearNotifications.createNotificationRuntime({{
          root, voiceHost, el, iconSvg: () => "<svg></svg>",
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
          const notificationBtn = byId.get("notificationBtn");
          const notificationPanel = byId.get("notificationPanel");
          await runtime.syncState({{ subscriptions: [], vapid_public_key: "key" }});
          const enabledWithoutUnread = {{
            enabled: runtime.enabledLocally(),
            active: notificationBtn.classList.contains("active"),
            unread: notificationBtn.dataset.unread,
          }};
          feed = [{{ message_id: "fresh", updated_ts: 1, notification_text: "Done", session_display_name: "Agent" }}];
          await runtime.pollFeed();
          const withUnread = {{
            active: notificationBtn.classList.contains("active"),
            unread: notificationBtn.dataset.unread,
          }};
          process.stdout.write(JSON.stringify({{
            api: Object.keys(runtime).sort(),
            enabledWithoutUnread,
            withUnread,
            dom: {{
              hostIds: voiceHost._children.map((child) => child._attrs.id || ""),
              rootIds: root._children.map((child) => child._attrs.id || ""),
              panelClass: notificationPanel.className,
            }},
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    completed = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)

    assert result["api"] == ["dispose", "enabledLocally", "pollFeed", "syncState"]
    assert result["enabledWithoutUnread"] == {"enabled": True, "active": False, "unread": "0"}
    assert result["withUnread"] == {"active": True, "unread": "1"}
    assert result["dom"] == {
        "hostIds": ["notificationBtn", "announceBtn"],
        "rootIds": ["notificationPanel"],
        "panelClass": "notificationPanel",
    }
