(() => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const textOf = (node) => node ? (node.innerText || node.textContent || "") : "";
  const rowByText = (needle) => [...document.querySelectorAll(".session")].find((row) => textOf(row).includes(needle)) || null;
  const dialogState = () => {
    const dialog = document.querySelector("#appConfirm");
    const backdrop = document.querySelector("#appConfirmBackdrop");
    const confirmBtn = document.querySelector("#appConfirmConfirmBtn");
    const cancelBtn = document.querySelector("#appConfirmCancelBtn");
    const title = document.querySelector("#appConfirmTitle");
    const msg = document.querySelector("#appConfirmMessage");
    const active = document.activeElement;
    const rect = dialog ? dialog.getBoundingClientRect() : null;
    return {
      exists: !!dialog,
      display: dialog ? getComputedStyle(dialog).display : null,
      backdropDisplay: backdrop ? getComputedStyle(backdrop).display : null,
      title: textOf(title).trim(),
      message: textOf(msg).trim(),
      confirmText: textOf(confirmBtn).trim(),
      cancelText: textOf(cancelBtn).trim(),
      activeId: active ? active.id : null,
      activeText: textOf(active).trim(),
      role: dialog ? dialog.getAttribute("role") : null,
      ariaModal: dialog ? dialog.getAttribute("aria-modal") : null,
      rect: rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom } : null,
      bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth || document.body.scrollWidth > document.body.clientWidth,
      nativeConfirmCount: window.__nativeConfirmCount || 0,
    };
  };
  const sessionState = async () => {
    let apiIds = [];
    let apiPending = null;
    try {
      const apiSessions = await fetch("/api/sessions").then((r) => r.json());
      apiIds = (apiSessions.sessions || []).map((s) => s.session_id);
      const row = (apiSessions.sessions || []).find((s) => s.session_id === "confirm-focus-session");
      apiPending = row ? Boolean(row.pending_attachment) : null;
    } catch (_err) {}
    return {
      rowPresent: !!rowByText("confirm-focus-proof"),
      apiIds,
      apiHasProof: apiIds.includes("confirm-focus-session"),
      apiPendingAttachment: apiPending,
      selectedHash: location.hash,
      nativeConfirmCount: window.__nativeConfirmCount || 0,
    };
  };
  window.__confirmFocusProof = { sleep, textOf, rowByText, dialogState, sessionState };
})()
