// Claim 5 browser evidence: sidebar state dot across interrupt-idle -> resumed -> complete.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const PORT = '19200', BASE = `http://127.0.0.1:${PORT}`, PASSWORD = 'test-password';
const ART = '/tmp/codoxear-cert-19200/cert-artifacts', CHROMIUM = '/usr/bin/chromium';
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const log = (...a) => console.log('[c5]', ...a);

async function dotInfo(page) {
  return page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll('div.session'));
    const card = cards.find(c => (c.textContent || '').includes('interrupt'));
    if (!card) return { found: false };
    const dot = card.querySelector('.stateDot');
    const cs = dot ? window.getComputedStyle(dot) : null;
    return { found: true, dotClass: dot ? dot.className : null,
      dotColor: cs ? cs.backgroundColor : null,
      titleText: card.querySelector('.titleText') ? card.querySelector('.titleText').textContent : null };
  });
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROMIUM, headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  await page.goto(BASE + '/', { waitUntil: 'networkidle2' });
  if (!await page.$('#sendBtn')) {
    await page.waitForSelector('input[name="password"]', { timeout: 15000 });
    await page.type('input[name="password"]', PASSWORD);
    await page.click('#loginBtn');
    await page.waitForSelector('#sendBtn', { timeout: 15000 });
  }
  await sleep(2500);
  // Phase 0: select cert-interrupt (post-interrupt idle).
  await page.evaluate(() => { const c = Array.from(document.querySelectorAll('div.session')).find(x => (x.textContent||'').includes('interrupt')); if (c) c.click(); });
  await sleep(2500);
  const p0 = await dotInfo(page);
  log('phase0 dot:', JSON.stringify(p0));
  await page.screenshot({ path: `${ART}/claim5-phase0-idle.png` });

  // Trigger resumed activity via /send from page context.
  const send = await page.evaluate(async () => {
    const r = await fetch('/api/sessions/cert-interrupt/send', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: 'resume', allow_pending_attachment: false }), credentials: 'include' });
    return { status: r.status, json: await r.json().catch(() => null) };
  });
  log('send:', JSON.stringify(send));
  await sleep(3000);
  const p1 = await dotInfo(page);
  log('phase1 dot:', JSON.stringify(p1));
  await page.screenshot({ path: `${ART}/claim5-phase1-running.png` });

  // Wait for auto-completion.
  await sleep(6000);
  const p2 = await dotInfo(page);
  log('phase2 dot:', JSON.stringify(p2));
  await page.screenshot({ path: `${ART}/claim5-phase2-complete.png` });

  fs.writeFileSync(`${ART}/claim5-browser.json`, JSON.stringify({ phase0: p0, send, phase1: p1, phase2: p2 }, null, 2));
  await browser.close();
  log('done');
})().catch(e => { console.error('FATAL', e); process.exit(1); });
