self.addEventListener("push", (event) => {
  let payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch (_error) {
    payload = {};
  }
  const sessionName = typeof payload.session_display_name === "string" && payload.session_display_name ? payload.session_display_name : "Codoxear";
  const body = typeof payload.notification_text === "string" && payload.notification_text ? payload.notification_text : "New final response";
  const sessionId = typeof payload.session_id === "string" ? payload.session_id : "";
  const targetUrl = new URL("./", self.registration.scope);
  if (sessionId) {
    targetUrl.hash = `session=${encodeURIComponent(sessionId)}`;
  }
  event.waitUntil(
    self.registration.showNotification(sessionName, {
      body,
      tag: sessionId || "codoxear-final-response",
      data: { url: targetUrl.toString() },
      renotify: true,
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const rawUrl = event.notification && event.notification.data ? event.notification.data.url : "";
  const target = typeof rawUrl === "string" && rawUrl ? rawUrl : new URL("./", self.registration.scope).toString();
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((windows) => {
      for (const client of windows) {
        if ("focus" in client) {
          client.navigate(target);
          return client.focus();
        }
      }
      return clients.openWindow(target);
    })
  );
});
