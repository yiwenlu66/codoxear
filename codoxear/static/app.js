import { createApplicationController } from "./app_application.js";

/* Boot is deliberately small: application composition lives in app_application.js. */
  const controller = createApplicationController({
    windowTarget: window,
    documentTarget: document,
    navigatorTarget: typeof navigator === "undefined" ? undefined : navigator,
    EventSource: typeof EventSource === "function" ? EventSource : null,
    AbortController: typeof AbortController === "function" ? AbortController : null,
  });
  const { api, renderApp, renderLogin } = controller;

  function showPostLoginBootstrapFailure(error) {
    console.error("application bootstrap failed after login", error);
    const target = document.getElementById("root") || document.body;
    const message = error && error.message ? error.message : "unknown error";
    const panel = document.createElement("section");
    panel.setAttribute("data-codoxear-boot-error", "true");
    panel.setAttribute("role", "alert");
    panel.style.cssText = [
      "box-sizing:border-box",
      "min-height:100%",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "padding:16px",
      "background:#f6f5f1",
      "color:#2f2b26",
      "border:1px solid #2f2b26",
      "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
      "white-space:pre-wrap",
    ].join(";");
    panel.textContent = `Codoxear failed to start\n\n${message}\n\nReload to retry.`;
    target.replaceChildren(panel);
  }

  (async function boot() {
    try {
      await api("/api/me");
      renderApp();
    } catch (error) {
      if (error && error.status === 401) {
        renderLogin(() => {
          try {
            renderApp();
          } catch (bootstrapError) {
            showPostLoginBootstrapFailure(bootstrapError);
          }
        });
        return;
      }
      console.error("boot auth check failed", error);
      const err = document.createElement("pre");
      err.textContent = `error: unable to contact server (${error && error.message ? error.message : "unknown error"})`;
      document.body.innerHTML = "";
      document.body.appendChild(err);
    }
  })();
