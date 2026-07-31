(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const sid = "code-copy-session";
  window.__copiedTexts = [];
  try {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text) => {
          window.__copiedTexts.push(String(text));
        },
      },
    });
  } catch (_err) {}
  const by = (sel) => document.querySelector(sel);
  if (!location.hash.includes(sid)) location.hash = `#session=${sid}`;
  await sleep(900);
  const row = [...document.querySelectorAll(".session")].find((node) => (node.innerText || "").includes("code-copy-proof"));
  if (row) row.click();
  await sleep(1200);
  const buttons = [...document.querySelectorAll(".md pre .code-copy-btn")];
  const pres = [...document.querySelectorAll(".md pre")];
  const codeTexts = pres.map((pre) => (pre.querySelector("code") || {}).textContent || "");
  const rectsBefore = buttons.map((btn) => {
    const r = btn.getBoundingClientRect();
    const cs = getComputedStyle(btn);
    return { width: r.width, height: r.height, opacity: cs.opacity, aria: btn.getAttribute("aria-label"), title: btn.getAttribute("title") };
  });
  const msgCopy = [...document.querySelectorAll(".msg-copy-btn")].find((btn) => !btn.disabled && btn.getAttribute("aria-hidden") !== "true") || by(".msg-copy-btn");
  if (buttons[0]) buttons[0].click();
  await sleep(180);
  const firstAfter = buttons[0]
    ? { copiedClass: buttons[0].classList.contains("copied"), aria: buttons[0].getAttribute("aria-label"), title: buttons[0].getAttribute("title") }
    : null;
  if (buttons[1]) buttons[1].click();
  await sleep(180);
  if (msgCopy) msgCopy.click();
  await sleep(300);
  const toast = (by("#toast") || {}).innerText || "";
  const copied = window.__copiedTexts.slice();
  return {
    ok: true,
    selectedHash: location.hash,
    codeButtonCount: buttons.length,
    preCount: pres.length,
    codeTexts,
    rectsBefore,
    firstAfter,
    copied,
    firstCopyExact: copied[0] === "printf 'alpha <tag> & value'",
    secondCopyExact: copied[1] === '{"beta": 2, "note": "second block"}',
    blockCopiesExcludeProse: copied.slice(0, 2).every((text) => !/prose between|Here are two|End of answer/.test(text || "")),
    messageCopyIncludesProseAndBothBlocks: Boolean(copied[2] && copied[2].includes("The prose between code blocks") && copied[2].includes("```bash") && copied[2].includes("```json")),
    toast,
    bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth || document.body.scrollWidth > document.body.clientWidth,
  };
})()
