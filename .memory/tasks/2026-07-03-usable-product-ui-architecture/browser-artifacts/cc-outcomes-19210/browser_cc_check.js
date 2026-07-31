const puppeteer = require('/tmp/codoxear-cert-19200/cert-artifacts/node_modules/puppeteer-core');
const fs = require('fs');
const [,, base, password, outPath] = process.argv;
(async () => {
  const browser = await puppeteer.launch({executablePath:'/usr/bin/chromium', headless:true, args:['--no-sandbox','--disable-dev-shm-usage']});
  const page = await browser.newPage();
  await page.goto(base, {waitUntil:'networkidle2'});
  const loginResult = await page.evaluate(async (password) => {
    const r = await fetch('/api/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({password}), credentials:'same-origin'});
    return {status:r.status, text: await r.text()};
  }, password);
  if (loginResult.status !== 200) throw new Error('login failed '+JSON.stringify(loginResult));
  await page.goto(base, {waitUntil:'networkidle2'});
  await page.waitForFunction(() => document.querySelectorAll('.session').length >= 3, {timeout:10000});
  const sessionOrder = await page.evaluate(async () => {
    const r = await fetch('/api/sessions', {credentials:'same-origin'});
    const obj = await r.json();
    return obj.sessions.map(s => s.session_id);
  });
  async function selectAndRead(sid, expected) {
    const index = sessionOrder.indexOf(sid);
    if (index < 0) throw new Error('session not in API order '+sid+' order='+sessionOrder.join(','));
    await page.evaluate((index) => {
      const cards = Array.from(document.querySelectorAll('.session'));
      if (!cards[index]) throw new Error('session card index missing '+index+' count='+cards.length);
      cards[index].click();
    }, index);
    await page.waitForFunction((expected) => document.body.innerText.includes(expected), {timeout:10000}, expected);
    return await page.evaluate((sid, expected) => {
      const text = document.body.innerText;
      const rows = Array.from(document.querySelectorAll('.msg,.msg-row,.msg-shell')).map(e => ({cls:String(e.className), text:e.textContent})).filter(r => r.text.includes(expected));
      const errorRows = Array.from(document.querySelectorAll('.msg.error,.assistant.error,.msg-row.assistant')).map(e => ({cls:String(e.className), text:e.textContent})).filter(r => r.text.includes(expected));
      return {sid, expected, hasExpected:text.includes(expected), rows, errorRows};
    }, sid, expected);
  }
  const result = {
    loginResult,
    sessionOrder,
    noresp: await selectAndRead('cc-noresp', 'The backend completed this turn without producing a response.'),
    apierr: await selectAndRead('cc-apierr', 'API Error: 503 Service Unavailable'),
    normal: await selectAndRead('cc-normal', 'CC-ANSWER-OK'),
  };
  await page.screenshot({path: outPath.replace(/\.json$/, '.png'), fullPage:true});
  await browser.close();
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
})();
