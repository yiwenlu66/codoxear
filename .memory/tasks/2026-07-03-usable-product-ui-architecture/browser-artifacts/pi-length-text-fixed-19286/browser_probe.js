(async () => {
const sessions = [
  'pi-length-text-prefix-fixed',
  'pi-length-text-continuation-fixed',
  'pi-stop-text-control'
];
const sleep = ms => new Promise(r => setTimeout(r, ms));
function collect(sid) {
  const rows = Array.from(document.querySelectorAll('.msg-row')).map(row => {
    const msg = row.querySelector('.msg');
    return {rowClass: row.className, msgClass: msg ? msg.className : null, text: (msg || row).innerText.trim()};
  });
  const msgBox = document.querySelector('#msg');
  const sendBtn = document.querySelector('#sendBtn, #send');
  return {sid, href: location.href, rowCount: rows.length, rowClasses: rows.map(r => r.msgClass || r.rowClass), rowTexts: rows.map(r => r.text), composerDisabled: msgBox ? msgBox.disabled : null, sendDisabled: sendBtn ? sendBtn.disabled : null, loadError: window.__codoxearLoadError || null};
}
const out = {};
for (const sid of sessions) {
  location.hash = `session=${sid}`;
  await sleep(900);
  out[sid] = collect(sid);
}
out.title = document.title;
out.url = location.href;
out.viewport = {w: innerWidth, h: innerHeight};
return out;
})()
