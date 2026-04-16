export function installViewportCssVars() {
  const update = () => {
    const vv = window.visualViewport;
    const layoutH = Math.round(window.innerHeight);
    const visualH = Math.round(vv ? vv.height : window.innerHeight);
    document.documentElement.style.setProperty("--appH", `${visualH}px`);
    document.documentElement.style.setProperty("--layoutH", `${layoutH}px`);
  };
  update();
  window.addEventListener("resize", update);
  window.visualViewport?.addEventListener("resize", update);
}
