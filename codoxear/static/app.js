import { createApplicationController } from "./app_application.js";

/* Boot is deliberately small: application composition lives in app_application.js. */
  "use strict";

  const application = { createApplicationController };
  if (!application || typeof application.createApplicationController !== "function")
    throw new Error("Codoxear application controller failed to load");

  const controller = application.createApplicationController({
    windowTarget: window,
    documentTarget: document,
    navigatorTarget: typeof navigator === "undefined" ? undefined : navigator,
    EventSource: typeof EventSource === "function" ? EventSource : null,
    AbortController: typeof AbortController === "function" ? AbortController : null,
  });
  const { api, renderApp, renderLogin } = controller;

  (async function boot() {
    try {
      await api("/api/me");
      renderApp();
    } catch (error) {
      if (error && error.status === 401) {
        renderLogin(renderApp);
        return;
      }
      console.error("boot auth check failed", error);
      const err = document.createElement("pre");
      err.textContent = `error: unable to contact server (${error && error.message ? error.message : "unknown error"})`;
      document.body.innerHTML = "";
      document.body.appendChild(err);
    }
  })();
