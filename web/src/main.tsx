import { render } from "preact";
import App from "./app/App";
import "./styles/index.css";
import { toPublicAssetUrl } from "./lib/publicAssetUrl";
import { installViewportCssVars } from "./lib/viewport";

installViewportCssVars();
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register(toPublicAssetUrl("service-worker.js")).catch(() => undefined);
  });
}
render(<App />, document.getElementById("root")!);
