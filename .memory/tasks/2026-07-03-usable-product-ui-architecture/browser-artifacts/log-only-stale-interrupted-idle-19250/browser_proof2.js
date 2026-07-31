// Browser DOM proof v2: correct .session/.stateDot selectors.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const PORT = '19250', BASE = `http://127.0.0.1:${PORT}`, PASSWORD = 'test-password';
const ART = '/home/yiwen/codex-web-product-recovery/.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/log-only-stale-interrupted-idle-19250';
const CHROMIUM = '/usr/bin/chromium';
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const log = (...a) => console.log('[br]', ...a);

async function snap(page) {
  return page.evaluate(() => {
    const card = document.querySelector('.session');
    if (!card) return { found: false, sessionCount: document.querySelectorAll('.session').length };
    const dot = card.querySelector('.stateDot');
    const cs = dot ? window.getComputedStyle(dot) : null;
    const sendBtn = document.querySelector('#sendBtn');
    return {
      found: true,
      cardClass: card.className || null,
      dotClass: dot ? dot.className : null,
      dotBg: cs ? cs.backgroundColor : null,
      dotIsBusy: dot ? dot.classList.contains('busy') : null,
      dotIsIdle: dot ? dot.classList.contains('idle') : null,
      sendDisabled: sendBtn ? sendBtn.disabled : null,
      titleText: card.querySelector('.titleText') ? card.querySelector('.titleText').textContent : null,
    };
  });
}

async function apiBusy(page) {
  return page.evaluate(async () => {
    const r = await fetch('/api/sessions', { credentials: 'include' });
    const d = await r.json();
    const s = (d.sessions || []).find(x => x.session_id === 'cert-stale-interrupt');
    return s ? { busy: s.busy, missing: false } : { missing: true };
  });
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROMIUM, headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  await page.goto(BASE + '/', { waitUntil: 'networkidle2' });
  await page.waitForSelector('input[name="password"]', { timeout: 15000 });
  await page.type('input[name="password"]', PASSWORD);
  await page.click('#loginBtn');
  await page.waitForSelector('#sendBtn', { timeout: 15000 });
  await sleep(2500);

  // Select the fake session card.
  await page.evaluate(() => { const c = document.querySelector('.session'); if (c) c.click(); });
  await sleep(2500);

  const p1Dom = await snap(page);
  const p1Api = await apiBusy(page);
  log('phase1 dom:', JSON.stringify(p1Dom), 'api:', JSON.stringify(p1Api));
  await page.screenshot({ path: `${ART}/browser-phase1-idle.png` });

  // Ask host to append post-interrupt activity.
  fs.writeFileSync(`${ART}/.browser-append-request`, 'now');
  let waited = 0;
  while (waited < 30) {
    if (fs.existsSync(`${ART}/.browser-append-done`)) break;
    await sleep(500); waited += 0.5;
  }
  await sleep(3000); // let polling refresh DOM

  const p2Dom = await snap(page);
  const p2Api = await apiBusy(page);
  log('phase2 dom:', JSON.stringify(p2Dom), 'api:', JSON.stringify(p2Api));
  await page.screenshot({ path: `${ART}/browser-phase2-busy.png` });

  await sleep(2000);
  const p2bDom = await snap(page);
  const p2bApi = await apiBusy(page);

  fs.writeFileSync(`${ART}/browser-dom.json`, JSON.stringify({
    phase1: { dom: p1Dom, api: p1Api },
    phase2: { dom: p2Dom, api: p2Api },
    phase2_repoll: { dom: p2bDom, api: p2bApi },
  }, null, 2));
  log('done');
  await browser.close();
})().catch(e => { console.error('FATAL', e); process.exit(1); });
