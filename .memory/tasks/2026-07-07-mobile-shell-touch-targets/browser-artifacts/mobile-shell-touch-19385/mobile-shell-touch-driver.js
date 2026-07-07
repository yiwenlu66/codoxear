(async () => {
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const visible = (el) => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return cs.display !== "none" && cs.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  };
  const measure = (label, el) => {
    const rect = el ? el.getBoundingClientRect() : null;
    const cs = el ? getComputedStyle(el) : null;
    return {
      label,
      exists: Boolean(el),
      visible: visible(el),
      width: rect ? Math.round(rect.width * 100) / 100 : 0,
      height: rect ? Math.round(rect.height * 100) / 100 : 0,
      display: cs ? cs.display : "",
      visibility: cs ? cs.visibility : "",
      disabled: Boolean(el && el.disabled),
      aria: el ? el.getAttribute("aria-label") || "" : "",
      title: el ? el.getAttribute("title") || "" : "",
    };
  };
  const byId = (id) => measure(`#${id}`, document.getElementById(id));
  const targets = [];
  for (const id of [
    "toggleSidebarBtn",
    "interruptBtn",
    "fileBtn",
    "copyConversationBtn",
    "diagBtn",
    "unattendedBtn",
    "chatSearchBtn",
    "prevUserBtn",
    "nextUserBtn",
  ]) targets.push(byId(id));

  const toggle = document.getElementById("toggleSidebarBtn");
  if (toggle) toggle.click();
  await wait(350);
  for (const id of ["newBtn", "announceBtn", "notificationBtn"]) targets.push(byId(id));

  const newBtn = document.getElementById("newBtn");
  if (newBtn) newBtn.click();
  await wait(350);
  const backendTabs = Array.from(document.querySelectorAll(".agentBackendTab")).map((el, idx) => measure(`.agentBackendTab[${idx}]`, el));
  const backendLabels = backendTabs.map((tab, idx) => ({ idx, text: document.querySelectorAll(".agentBackendTab")[idx]?.textContent?.trim() || "", ...tab }));
  const visibleTargets = targets.filter((item) => item.visible);
  const visibleBackendTabs = backendLabels.filter((item) => item.visible);
  const tooSmall = visibleTargets.concat(visibleBackendTabs).filter((item) => item.width < 44 || item.height < 44);
  return {
    viewport: { innerWidth, innerHeight, devicePixelRatio },
    href: location.href,
    bodyOverflow: {
      scrollWidth: document.scrollingElement ? document.scrollingElement.scrollWidth : document.documentElement.scrollWidth,
      clientWidth: document.scrollingElement ? document.scrollingElement.clientWidth : document.documentElement.clientWidth,
      innerWidth,
    },
    selectedSession: document.querySelector(".session.active")?.dataset?.sid || null,
    targets,
    backendTabs: backendLabels,
    tooSmall,
  };
})()
