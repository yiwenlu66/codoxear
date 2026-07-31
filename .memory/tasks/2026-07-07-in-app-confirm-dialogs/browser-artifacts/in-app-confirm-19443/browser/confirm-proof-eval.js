(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  window.__nativeConfirmCount = 0;
  window.confirm = () => {
    window.__nativeConfirmCount += 1;
    throw new Error("native confirm should not be called");
  };
  const rows = () => Array.from(document.querySelectorAll(".session"));
  const rowByText = (needle) => rows().find((row) => (row.innerText || row.textContent || "").includes(needle)) || null;
  const textOf = (node) => node ? (node.innerText || node.textContent || "") : "";
  const dialogState = () => {
    const dialog = document.querySelector("#appConfirm");
    const backdrop = document.querySelector("#appConfirmBackdrop");
    const confirmBtn = document.querySelector("#appConfirmConfirmBtn");
    const cancelBtn = document.querySelector("#appConfirmCancelBtn");
    const title = document.querySelector("#appConfirmTitle");
    const msg = document.querySelector("#appConfirmMessage");
    return {
      exists: !!dialog,
      display: dialog ? getComputedStyle(dialog).display : null,
      backdropDisplay: backdrop ? getComputedStyle(backdrop).display : null,
      role: dialog ? dialog.getAttribute("role") : null,
      ariaModal: dialog ? dialog.getAttribute("aria-modal") : null,
      labelledby: dialog ? dialog.getAttribute("aria-labelledby") : null,
      describedby: dialog ? dialog.getAttribute("aria-describedby") : null,
      title: textOf(title).trim(),
      message: textOf(msg).trim(),
      confirmText: textOf(confirmBtn).trim(),
      cancelText: textOf(cancelBtn).trim(),
      activeId: document.activeElement ? document.activeElement.id : null,
      bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth || document.body.scrollWidth > document.body.clientWidth,
    };
  };
  const sessionSummaries = () => rows().map((row) => ({ text: textOf(row).trim(), html: row.outerHTML.slice(0, 1200) }));

  await sleep(800);
  const initial = sessionSummaries();
  const cancelRow = rowByText("confirm-cancel-project");
  const deleteRow = rowByText("confirm-delete-project");
  if (!cancelRow || !deleteRow) return { ok: false, reason: "proof rows not found", initial };

  cancelRow.querySelector(".sessionDel").click();
  await sleep(150);
  const cancelDialog = dialogState();
  document.querySelector("#appConfirmCancelBtn").click();
  await sleep(350);
  const afterCancel = {
    dialog: dialogState(),
    cancelRowStillPresent: !!rowByText("confirm-cancel-project"),
    deleteRowStillPresent: !!rowByText("confirm-delete-project"),
    sessionCount: rows().length,
    nativeConfirmCount: window.__nativeConfirmCount,
  };

  rowByText("confirm-delete-project").querySelector(".sessionDel").click();
  await sleep(150);
  const confirmDialog = dialogState();
  document.querySelector("#appConfirmConfirmBtn").click();
  await sleep(1000);
  const apiSessions = await fetch("/api/sessions").then((r) => r.json());
  const apiIds = (apiSessions.sessions || []).map((s) => s.session_id);
  const afterConfirm = {
    dialog: dialogState(),
    cancelRowStillPresent: !!rowByText("confirm-cancel-project"),
    deleteRowStillPresent: !!rowByText("confirm-delete-project"),
    sessionCount: rows().length,
    nativeConfirmCount: window.__nativeConfirmCount,
    apiIds,
    apiHasCancel: apiIds.includes("launch-confirm-cancel"),
    apiHasDelete: apiIds.includes("launch-confirm-delete"),
  };

  return { ok: true, initial, cancelDialog, afterCancel, confirmDialog, afterConfirm };
})()
