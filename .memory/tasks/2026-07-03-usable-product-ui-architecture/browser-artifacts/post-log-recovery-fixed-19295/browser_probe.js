(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const sid = 'post-log-recovery-fixed';
  location.hash = `session=${sid}`;
  await sleep(1200);
  const rows = Array.from(document.querySelectorAll('.msg-row')).map(row => {
    const msg = row.querySelector('.msg');
    return { rowClass: row.className, msgClass: msg ? msg.className : null, text: (msg || row).innerText.trim(), historyCursor: row.dataset.historyCursor || '' };
  });
  const msg = document.querySelector('#msg');
  const sendBtn = document.querySelector('#sendBtn, #send');
  const queueBtn = document.querySelector('#queueBtn');
  const attachBtn = document.querySelector('#attachBtn, #attachFilesBtn');
  const unattendedBtn = document.querySelector('#unattendedBtn');
  const exportBtn = document.querySelector('#copyConversationBtn, #exportTranscriptBtn');
  return {
    href: location.href,
    title: document.title,
    viewport: { w: innerWidth, h: innerHeight },
    loadError: window.__codoxearLoadError || null,
    rows,
    composer: { disabled: msg ? msg.disabled : null, placeholder: msg ? msg.getAttribute('placeholder') : null, title: msg ? msg.getAttribute('title') : null, value: msg ? msg.value : null },
    controls: {
      sendDisabled: sendBtn ? sendBtn.disabled : null,
      queueDisabled: queueBtn ? queueBtn.disabled : null,
      attachDisabled: attachBtn ? attachBtn.disabled : null,
      unattendedDisabled: unattendedBtn ? unattendedBtn.disabled : null,
      exportDisabled: exportBtn ? exportBtn.disabled : null,
    },
    bodyTextIncludes: {
      sentinel: document.body.innerText.includes('POST_LOG_BOUND_DEATH_SENTINEL'),
      stopped: document.body.innerText.includes('The backend process stopped before completing this turn.'),
      postLogCopy: document.body.innerText.includes('stopped after binding a transcript log, before the turn completed'),
    }
  };
})()
