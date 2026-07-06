// Codoxear certification browser harness (claims 1, 3, 4-browser).
const puppeteer = require('puppeteer-core');
const fs = require('fs');

const PORT = process.env.PORT || '19200';
const BASE = `http://127.0.0.1:${PORT}`;
const PASSWORD = 'test-password';
const ART = process.env.ART_DIR || '/tmp/codoxear-cert-19200/cert-artifacts';
const CHROMIUM = process.env.CHROMIUM || '/usr/bin/chromium';
function log(...a) { console.log('[harness]', ...a); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function login(page) {
  await page.goto(BASE + '/', { waitUntil: 'networkidle2' });
  // Already authenticated? (#sendBtn present => no login form)
  const authed = await page.$('#sendBtn');
  if (authed) { log('already authed'); return; }
  await page.waitForSelector('input[name="password"]', { timeout: 15000 });
  await page.type('input[name="password"]', PASSWORD);
  await page.click('#loginBtn');
  await page.waitForSelector('#sendBtn', { timeout: 15000 });
  log('logged in');
}

async function badgeInfo(page) {
  return page.evaluate(() => {
    const b = document.getElementById('attachBadge');
    if (!b) return { present: false };
    const cs = window.getComputedStyle(b);
    const r = b.getBoundingClientRect();
    return { present: true, display: cs.display, textContent: b.textContent,
             visible: cs.display !== 'none' && r.width > 0 && r.height > 0 };
  });
}

async function apiFromPage(page, path, opts = {}) {
  return page.evaluate(async (u, o) => {
    const res = await fetch(u, { method: o.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: o.body ? JSON.stringify(o.body) : undefined, credentials: 'include' });
    return { status: res.status, json: await res.json().catch(() => null) };
  }, BASE + path, opts);
}

async function selectSession(page, sid) {
  // Sidebar cards are keyed by display name = basename(cwd); fake sessions use
  // distinct cwds so the card text uniquely contains a recognizable token.
  // Map sid -> cwd basename token used for matching.
  const tokenMap = {
    'cert-noresp': 'noresp', 'cert-normal': 'normal', 'cert-resume': 'resume',
    'cert-attach-a': 'attachA', 'cert-attach-b': 'attachB', 'cert-cleanup': 'cleanup',
  };
  const token = tokenMap[sid] || sid;
  const clicked = await page.evaluate((token) => {
    const cards = Array.from(document.querySelectorAll('div.session'));
    let target = cards.find(c => (c.textContent || '').includes(token));
    if (!target) target = cards[0];
    if (target) { target.click(); return target.textContent.trim().slice(0, 60); }
    return null;
  }, token);
  await sleep(1500);
  return clicked;
}

(async () => {
  const results = { claim1: {}, claim3: {}, claim4Browser: {} };
  const browser = await puppeteer.launch({
    executablePath: CHROMIUM, headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--window-size=1280,900'],
  });

  // ===== CLAIM 1: Attachment indicator truth (desktop 1280x900) =====
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await login(page);

    // Ensure clean pending state on both attach sessions.
    await apiFromPage(page, '/api/sessions/cert-attach-a/pending_attachment/clear', { method: 'POST', body: {} });
    await apiFromPage(page, '/api/sessions/cert-attach-b/pending_attachment/clear', { method: 'POST', body: {} });
    await sleep(1500);

    // --- cert-attach-a: appear + survive reload ---
    await selectSession(page, 'cert-attach-a');
    await sleep(2500);
    const before = await badgeInfo(page);
    results.claim1.beforeAttach = before;
    log('A before attach:', JSON.stringify(before));

    const b64 = Buffer.from('hello attachment indicator').toString('base64');
    const inj = await apiFromPage(page, '/api/sessions/cert-attach-a/inject_file',
      { method: 'POST', body: { filename: 'note.txt', attachment_index: 1, data_b64: b64 } });
    results.claim1.injectFile = { status: inj.status, ok: inj.json && inj.json.ok };
    log('A inject_file:', results.claim1.injectFile);
    await sleep(3200); // wait for the 2500ms session poll
    const afterAttach = await badgeInfo(page);
    results.claim1.afterAttach = afterAttach;
    log('A after attach:', JSON.stringify(afterAttach));
    await page.screenshot({ path: `${ART}/claim1-after-attach.png` });

    // Survive reload from server pending state.
    await page.reload({ waitUntil: 'networkidle2' });
    await page.waitForSelector('#sendBtn', { timeout: 15000 });
    await selectSession(page, 'cert-attach-a');
    await sleep(3200);
    const afterReload = await badgeInfo(page);
    results.claim1.afterReload = afterReload;
    log('A after reload:', JSON.stringify(afterReload));
    await page.screenshot({ path: `${ART}/claim1-after-reload.png` });

    // --- Disappears immediately after sending the pending attachment ---
    await page.evaluate(() => { window.confirm = () => true; });
    await page.type('#msg', 'consume the pending attachment');
    const sendTransition = await page.evaluate(async () => {
      const beforeBadge = (() => { const b = document.getElementById('attachBadge'); const cs = window.getComputedStyle(b); return { display: cs.display, text: b.textContent }; })();
      document.getElementById('sendBtn').click();
      let hiddenAt = null;
      const t0 = Date.now();
      while (Date.now() - t0 < 4000) {
        await new Promise(r => setTimeout(r, 40));
        const b = document.getElementById('attachBadge');
        if (window.getComputedStyle(b).display === 'none') { hiddenAt = Date.now() - t0; break; }
      }
      const afterBadge = (() => { const b = document.getElementById('attachBadge'); const cs = window.getComputedStyle(b); return { display: cs.display, text: b.textContent }; })();
      return { beforeBadge, afterBadge, hiddenAtMs: hiddenAt };
    });
    results.claim1.sendIndicator = sendTransition;
    log('A send transition:', JSON.stringify(sendTransition));
    await sleep(800);
    await page.screenshot({ path: `${ART}/claim1-after-send.png` });

    // --- cert-attach-b: appear + disappears after clearing ---
    await selectSession(page, 'cert-attach-b');
    await sleep(2500);
    const inj2 = await apiFromPage(page, '/api/sessions/cert-attach-b/inject_file',
      { method: 'POST', body: { filename: 'note.txt', attachment_index: 1, data_b64: b64 } });
    results.claim1.injectFileB = { status: inj2.status, ok: inj2.json && inj2.json.ok };
    await sleep(3200);
    const afterAttachB = await badgeInfo(page);
    results.claim1.afterAttachB = afterAttachB;
    log('B after attach:', JSON.stringify(afterAttachB));

    const clearResp = await apiFromPage(page, '/api/sessions/cert-attach-b/pending_attachment/clear', { method: 'POST', body: {} });
    results.claim1.clearResp = { status: clearResp.status, json: clearResp.json };
    // Capture badge within ~400ms (before the next poll) to prove immediate.
    let clearHiddenAt = null;
    const ct0 = Date.now();
    while (Date.now() - ct0 < 3000) {
      await sleep(40);
      const bi = await badgeInfo(page);
      if (!bi.visible) { clearHiddenAt = Date.now() - ct0; break; }
    }
    const afterClearB = await badgeInfo(page);
    results.claim1.afterClearB = { ...afterClearB, hiddenAtMs: clearHiddenAt };
    log('B after clear:', JSON.stringify(results.claim1.afterClearB));
    await page.screenshot({ path: `${ART}/claim1-after-clear.png` });

    await page.close();
  } catch (e) { results.claim1.error = String(e && e.stack || e); log('claim1 err', results.claim1.error); }

  // ===== CLAIM 3: Mobile composer controls >= 44x44 (390x844) =====
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1, isMobile: false, hasTouch: false });
    await login(page);
    await selectSession(page, 'cert-attach-a');
    await sleep(2500);
    await page.evaluate(() => { const s = document.getElementById('composerStopBtn'); if (s) s.classList.add('is-visible'); });
    const measure = await page.evaluate(() => {
      function rect(id) {
        const e = document.getElementById(id);
        if (!e) return { id, missing: true };
        const r = e.getBoundingClientRect(); const cs = window.getComputedStyle(e);
        return { id, width: Math.round(r.width * 100) / 100, height: Math.round(r.height * 100) / 100,
          minWidth: cs.minWidth, minHeight: cs.minHeight, display: cs.display,
          meets44: r.width >= 44 && r.height >= 44 };
      }
      const composer = document.querySelector('.composer');
      const formEl = composer && composer.querySelector('form');
      return {
        controls: [rect('attachBtn'), rect('queueBtn'), rect('sendBtn'), rect('composerStopBtn')],
        composerScrollWidth: composer ? composer.scrollWidth : null,
        composerClientWidth: composer ? composer.clientWidth : null,
        formScrollWidth: formEl ? formEl.scrollWidth : null,
        formClientWidth: formEl ? formEl.clientWidth : null,
        bodyScrollWidth: document.body.scrollWidth,
        viewportWidth: window.innerWidth,
        horizontalOverflow: (composer && composer.scrollWidth > composer.clientWidth) || document.body.scrollWidth > window.innerWidth,
      };
    });
    results.claim3.desktopPointer = measure;
    await page.screenshot({ path: `${ART}/claim3-mobile-composer.png` });
    // Touch-emulated variant.
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
    await sleep(600);
    const measureTouch = await page.evaluate(() => {
      function rect(id) {
        const e = document.getElementById(id); if (!e) return { id, missing: true };
        const r = e.getBoundingClientRect();
        return { id, width: Math.round(r.width * 100) / 100, height: Math.round(r.height * 100) / 100, meets44: r.width >= 44 && r.height >= 44 };
      }
      const composer = document.querySelector('.composer');
      return { controls: [rect('attachBtn'), rect('queueBtn'), rect('sendBtn'), rect('composerStopBtn')],
        composerScrollWidth: composer ? composer.scrollWidth : null, composerClientWidth: composer ? composer.clientWidth : null,
        horizontalOverflow: composer && composer.scrollWidth > composer.clientWidth };
    });
    results.claim3.touchEmulated = measureTouch;
    await page.close();
  } catch (e) { results.claim3.error = String(e && e.stack || e); log('claim3 err', results.claim3.error); }

  // ===== CLAIM 4 (browser): no-response transcript message rendered =====
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    await login(page);
    await selectSession(page, 'cert-noresp');
    await sleep(2500);
    const transcript = await page.evaluate(() => {
      const rows = Array.from(document.querySelectorAll('.msg, .message, [class*="msg"], [class*="transcript"] > *'));
      const out = [];
      for (const r of rows) { const t = r.textContent && r.textContent.trim(); if (t) out.push({ cls: r.className, text: t.slice(0, 200) }); }
      const all = Array.from(document.querySelectorAll('*'));
      const hit = all.find(e => e.children.length === 0 && e.textContent && e.textContent.includes('without producing a response'));
      return { rowCount: out.length, rows: out.slice(0, 24), hasNoResponseText: !!hit,
        hitText: hit ? hit.textContent.trim() : null, hitClass: hit ? hit.className : null };
    });
    results.claim4Browser = transcript;
    log('noresp transcript rows:', transcript.rowCount, 'hasText:', transcript.hasNoResponseText);
    await page.screenshot({ path: `${ART}/claim4-browser-noresp.png` });
    await page.close();
  } catch (e) { results.claim4Browser.error = String(e && e.stack || e); log('claim4 err', results.claim4Browser.error); }

  await browser.close();
  fs.writeFileSync(`${ART}/browser-results.json`, JSON.stringify(results, null, 2));
  log('done');
})().catch(e => { console.error('FATAL', e); process.exit(1); });
