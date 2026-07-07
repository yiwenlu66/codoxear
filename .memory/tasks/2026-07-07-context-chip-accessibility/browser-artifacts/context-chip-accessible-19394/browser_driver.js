(() => {
  const chip = document.getElementById('ctxChip');
  const toast = document.getElementById('toast');
  const sessions = Array.from(document.querySelectorAll('.session.desktop, .session.mobile, [data-session-id]')).map((el, index) => ({
    index,
    tagName: el.tagName,
    className: String(el.className || ''),
    text: (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 180),
    datasetSessionId: el.dataset ? el.dataset.sessionId || null : null,
    ariaLabel: el.getAttribute('aria-label'),
    role: el.getAttribute('role'),
  }));
  const rect = chip ? chip.getBoundingClientRect() : null;
  const styles = chip ? getComputedStyle(chip) : null;
  const focusBefore = document.activeElement === chip;
  let focusAfterProgrammatic = null;
  if (chip) {
    try {
      chip.focus();
      focusAfterProgrammatic = document.activeElement === chip;
    } catch (e) {
      focusAfterProgrammatic = false;
    }
  }
  const body = document.body;
  const docEl = document.documentElement;
  return {
    url: location.href,
    hash: location.hash,
    title: document.title,
    viewport: { width: window.innerWidth, height: window.innerHeight, devicePixelRatio: window.devicePixelRatio },
    loggedInShellVisible: !!document.getElementById('newBtn') && !!document.getElementById('sendBtn'),
    activeElement: document.activeElement ? { tagName: document.activeElement.tagName, id: document.activeElement.id, ariaLabel: document.activeElement.getAttribute('aria-label'), text: (document.activeElement.textContent || '').trim().slice(0, 80) } : null,
    chip: chip ? {
      exists: true,
      tagName: chip.tagName,
      type: chip.getAttribute('type'),
      typeProperty: chip.type,
      disabled: chip.disabled === true,
      ariaLabel: chip.getAttribute('aria-label'),
      accessibleNameFromAriaLabel: chip.getAttribute('aria-label'),
      text: chip.textContent,
      title: chip.title,
      display: styles.display,
      visibility: styles.visibility,
      pointerEvents: styles.pointerEvents,
      tabIndex: chip.tabIndex,
      focusBefore,
      focusAfterProgrammatic,
      rect: rect ? { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) } : null,
      visibleByLayout: !!(rect && rect.width > 0 && rect.height > 0 && styles.display !== 'none' && styles.visibility !== 'hidden'),
    } : { exists: false },
    toast: toast ? { text: toast.textContent, role: toast.getAttribute('role'), ariaLive: toast.getAttribute('aria-live') } : null,
    overflow: {
      bodyScrollWidth: body ? body.scrollWidth : null,
      bodyClientWidth: body ? body.clientWidth : null,
      docScrollWidth: docEl ? docEl.scrollWidth : null,
      docClientWidth: docEl ? docEl.clientWidth : null,
      windowInnerWidth: window.innerWidth,
      horizontalOverflow: !!(docEl && docEl.scrollWidth > window.innerWidth + 1),
    },
    sessions,
  };
})()