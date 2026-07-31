(async () => {
const sessions = [
  'pi-stop-empty-regression',
  'pi-length-prefix-control',
  'pi-length-continuation-control'
];
const sleep = ms => new Promise(r => setTimeout(r, ms));
function collect(sid) {
  const rows = Array.from(document.querySelectorAll('.msg-row')).map(row => {
    const msg = row.querySelector('.msg');
    return {rowClass: row.className, msgClass: msg ? msg.className : null, text: (msg || row).innerText.trim()};
  });
  const msgBox = document.querySelector('#msg');
  const sendBtn = document.querySelector('#sendBtn, #send');
  const placeholder = document.querySelector('#msgPh');
  return {
    sid,
    href: location.href,
    rows,
    rowCount: rows.length,
    rowTexts: rows.map(r => r.text),
    rowClasses: rows.map(r => r.msgClass || r.rowClass),
    composerDisabled: msgBox ? msgBox.disabled : null,
    composerAria: msgBox ? msgBox.getAttribute('aria-label') : null,
    placeholderText: placeholder ? placeholder.textContent : null,
    sendDisabled: sendBtn ? sendBtn.disabled : null,
    loadError: window.__codoxearLoadError || null,
  };
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
