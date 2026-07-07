// Sanitized browser eval snippets used for evidence capture.

// Before Start Session: selected backend/options.
(() => ({
  backend: [...document.querySelectorAll('.agentBackendTab')].map(b => ({
    label: b.getAttribute('aria-label') || b.textContent,
    active: b.classList.contains('active')
  })),
  cwd: document.querySelector('#newSessionCwdInput')?.value,
  name: document.querySelector('#newSessionNameInput')?.value,
  model: document.querySelector('#newSessionModelInput')?.value,
  reasoning: document.querySelector('#newSessionReasoningBtn')?.textContent,
  tmux: document.querySelector('#newSessionTmuxToggle')?.checked
}))();

// Bound/idle session state after sentinel send.
(() => ({
  url: location.href,
  transcript: document.body.innerText,
  ctxChip: {
    text: document.querySelector('#ctxChip')?.textContent,
    title: document.querySelector('#ctxChip')?.getAttribute('title'),
    disabled: document.querySelector('#ctxChip')?.disabled
  },
  controls: {
    send: { disabled: document.querySelector('#sendBtn')?.disabled, label: document.querySelector('#sendBtn')?.getAttribute('aria-label') || document.querySelector('#sendBtn')?.textContent },
    file: { disabled: document.querySelector('#fileBtn')?.disabled, label: document.querySelector('#fileBtn')?.getAttribute('aria-label') },
    attach: { disabled: document.querySelector('#attachBtn')?.disabled, label: document.querySelector('#attachBtn')?.getAttribute('aria-label') },
    capture: { disabled: document.querySelector('#captureBtn')?.disabled, label: document.querySelector('#captureBtn')?.getAttribute('aria-label') },
    queue: { disabled: document.querySelector('#queueBtn')?.disabled, label: document.querySelector('#queueBtn')?.getAttribute('aria-label') },
    unattended: { disabled: document.querySelector('#unattendedBtn')?.disabled, label: document.querySelector('#unattendedBtn')?.getAttribute('aria-label') }
  },
  body: {
    scrollWidth: document.body.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    overflow: document.body.scrollWidth > document.documentElement.clientWidth
  }
}))();
