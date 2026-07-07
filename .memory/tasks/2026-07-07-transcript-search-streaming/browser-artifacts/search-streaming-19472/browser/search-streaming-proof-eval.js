(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const sid = "search-streaming-session";
  const by = (sel) => document.querySelector(sel);
  const setInput = (el, value) => {
    el.focus();
    el.value = value;
    el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
  };
  if (!location.hash.includes(sid)) location.hash = `#session=${sid}`;
  await sleep(900);
  const row = [...document.querySelectorAll(".session")].find((node) => (node.innerText || "").includes("large-search-proof"));
  if (row) row.click();
  await sleep(1200);
  const initialRows = [...document.querySelectorAll(".msg-row")].map((row) => row.innerText || "");
  const searchBtn = by("#chatSearchBtn");
  if (!searchBtn) return { ok: false, reason: "missing chatSearchBtn" };
  if (searchBtn.disabled) return { ok: false, reason: "chatSearchBtn disabled", rowCount: initialRows.length, hash: location.hash };
  searchBtn.click();
  await sleep(250);
  const input = by("#chatSearchInput");
  if (!input) return { ok: false, reason: "missing chatSearchInput" };

  setInput(input, "needle");
  await sleep(950);
  const needleStatus = (by("#chatSearchStatus") || {}).innerText || "";
  const needleHint = (by("#chatSearchAllHint") || {}).innerText || "";
  const needleLoadedHits = document.querySelectorAll(".msg-row.chat-search-hit").length;
  const needleCurrentText = (by(".msg-row.chat-search-current") || {}).innerText || "";
  const needleApi = await (await fetch(`/api/sessions/${sid}/messages/search?q=needle&limit=1&text_max=96&count_max=1000`)).json();

  setInput(input, "EARLY_ONLY_TARGET");
  await sleep(950);
  const earlyBeforeStatus = (by("#chatSearchStatus") || {}).innerText || "";
  const earlyBeforeHint = (by("#chatSearchAllHint") || {}).innerText || "";
  const earlyBeforeHits = document.querySelectorAll(".msg-row.chat-search-hit").length;
  const nextBtn = by("#chatSearchNextBtn");
  if (nextBtn) nextBtn.click();
  await sleep(1800);
  const earlyAfterStatus = (by("#chatSearchStatus") || {}).innerText || "";
  const earlyAfterHint = (by("#chatSearchAllHint") || {}).innerText || "";
  const earlyRows = [...document.querySelectorAll(".msg-row")].map((row) => row.innerText || "");
  const earlyLoadedHits = document.querySelectorAll(".msg-row.chat-search-hit").length;
  const earlyCurrentText = (by(".msg-row.chat-search-current") || {}).innerText || "";
  const toast = (by("#toast") || {}).innerText || "";
  const earlyApi = await (await fetch(`/api/sessions/${sid}/messages/search?q=EARLY_ONLY_TARGET&limit=1&text_max=96`)).json();

  return {
    ok: true,
    selectedHash: location.hash,
    initialRowCount: initialRows.length,
    initialHasTailNeedle: initialRows.some((text) => text.includes("bulk needle search row")),
    initialHasEarlyTarget: initialRows.some((text) => text.includes("EARLY_ONLY_TARGET")),
    needle: {
      status: needleStatus,
      hint: needleHint,
      loadedHits: needleLoadedHits,
      currentText: needleCurrentText,
      apiMatchCount: needleApi.match_count,
      apiTruncated: needleApi.match_count_truncated,
      apiFirstText: needleApi.matches && needleApi.matches[0] && needleApi.matches[0].text,
    },
    early: {
      beforeStatus: earlyBeforeStatus,
      beforeHint: earlyBeforeHint,
      beforeLoadedHits: earlyBeforeHits,
      afterStatus: earlyAfterStatus,
      afterHint: earlyAfterHint,
      loadedHits: earlyLoadedHits,
      currentText: earlyCurrentText,
      rowContainsEarlyTarget: earlyRows.some((text) => text.includes("EARLY_ONLY_TARGET")),
      apiMatchCount: earlyApi.match_count,
      apiTruncated: earlyApi.match_count_truncated,
      apiFirstText: earlyApi.matches && earlyApi.matches[0] && earlyApi.matches[0].text,
    },
    toast,
    bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth || document.body.scrollWidth > document.body.clientWidth,
  };
})()
