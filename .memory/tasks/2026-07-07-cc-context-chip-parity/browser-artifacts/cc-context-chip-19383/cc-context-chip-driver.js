(async () => {
  const sid = "cc-context-proof";
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let sessions;
  let row;
  for (let i = 0; i < 40; i += 1) {
    sessions = await fetch("/api/sessions").then((response) => response.json());
    row = (sessions.sessions || []).find((item) => item.session_id === sid);
    if (row && row.token) break;
    await wait(250);
  }
  const tail = await fetch(`/api/sessions/${sid}/messages/tail?limit=50`).then((response) => response.json());
  await wait(500);
  const chip = document.querySelector("#ctxChip");
  return {
    row,
    tailToken: tail.token || null,
    events: tail.events || [],
    chip: chip ? { text: chip.textContent, title: chip.title, display: getComputedStyle(chip).display } : null,
    sessionsToken: row && row.token || null,
    href: location.href,
  };
})();
