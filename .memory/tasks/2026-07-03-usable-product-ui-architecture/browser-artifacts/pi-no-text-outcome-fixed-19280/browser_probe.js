(async () => {
const sessions = [
  'pi-no-text-stop-empty',
  'pi-no-text-end-turn-empty',
  'pi-no-text-stop-thinking',
  'pi-nonterminal-thinking-control',
  'pi-tool-use-control'
];
const sleep = ms => new Promise(r => setTimeout(r, ms));
function collect(sid) {
  const rows = Array.from(document.querySelectorAll('.msg-row')).map(row => {
    const msg = row.querySelector('.msg');
    return {rowClass: row.className, msgClass: msg ? msg.className : null, text: (msg || row).innerText.trim()};
  });
  const matchingSession = Array.from(document.querySelectorAll('[data-session-id]')).find(el => el.getAttribute('data-session-id') === sid);
  const stateDot = matchingSession && matchingSession.querySelector('.stateDot, .status-dot, .session-dot');
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
    matchingSessionClass: matchingSession ? matchingSession.className : null,
    matchingSessionText: matchingSession ? matchingSession.innerText.trim() : null,
    stateDotClass: stateDot ? stateDot.className : null,
    composerDisabled: msgBox ? msgBox.disabled : null,
    composerAria: msgBox ? msgBox.getAttribute('aria-label') : null,
    placeholderText: placeholder ? placeholder.textContent : null,
    sendDisabled: sendBtn ? sendBtn.disabled : null,
    loadError: window.__codoxearLoadError || null,
    bootstrapped: window.__codoxearBootstrapped || false,
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
