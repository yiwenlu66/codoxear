#!/usr/bin/env python3
"""Real-browser geometry verification for the in-flight attachment chip.

Run via scripts/docker_upload_pill_geometry_verify.sh. The upload POST is held
open by the Docker-only fault proxy, so the production in-flight chip can be
measured with getBoundingClientRect plus elementFromPoint hit testing, then the
same file is measured again once the commit replaces the chip. No UI mocks, no
provider credentials, no prompt submission.

--expect fixed     pending chip must match the completed chip's geometry while
                   keeping the 44px cancel target through its hit lane
--expect baseline  reproduce the pre-fix mismatch (pending taller than done)
"""
import argparse
import http.cookiejar
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, required=True)
p.add_argument('--artifacts', required=True)
p.add_argument('--session', required=True)
p.add_argument('--expect', choices=['fixed', 'baseline'], default='fixed')
a = p.parse_args()
assert a.port != 8743
base = f'http://127.0.0.1:{a.port}'
art = Path(a.artifacts)
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
report = {}

SESSION = 'activity-peek'
PASSWORD = 'pill-verify-password'

# The chip's own box, its computed box constraints, and the remove button's
# real clickable extent (walked one pixel at a time with elementFromPoint, so a
# clipped or covered hit lane is reported as clipped).
MEASURE = r'''(() => {
  // The chip's entrance animation is a 0.3s transform scale; settle it first so
  // a measurement is the chip's geometry, not a frame of its animation.
  document.querySelectorAll('.stagedAttachmentChip').forEach((chip) => {
    chip.getAnimations().forEach((animation) => { try { animation.finish(); } catch (_) {} });
  });
  const tray = document.querySelector('#stagedAttachments');
  const form = document.querySelector('.composer form');
  const owns = (btn, x, y) => {
    const hit = document.elementFromPoint(x, y);
    return !!hit && (hit === btn || btn.contains(hit));
  };
  const hitExtent = (btn) => {
    if (!btn) return null;
    const r = btn.getBoundingClientRect();
    const cx = Math.round(r.left + r.width / 2);
    const cy = Math.round(r.top + r.height / 2);
    const center = document.elementFromPoint(cx, cy);
    if (!owns(btn, cx, cy)) {
      return { ok: false, center: center ? center.tagName + '.' + (typeof center.className === 'string' ? center.className : '') : null };
    }
    let l = 0, rr = 0, u = 0, d = 0;
    while (l < 60 && owns(btn, cx - l - 1, cy)) l += 1;
    while (rr < 60 && owns(btn, cx + rr + 1, cy)) rr += 1;
    while (u < 60 && owns(btn, cx, cy - u - 1)) u += 1;
    while (d < 60 && owns(btn, cx, cy + d + 1)) d += 1;
    return { ok: true, w: l + rr + 1, h: u + d + 1, left: l, right: rr, up: u, down: d };
  };
  const parseInset = (value) => {
    const parts = String(value).trim().split(/\s+/).map((part) => parseFloat(part));
    if (parts.length === 1) return { top: parts[0], right: parts[0], bottom: parts[0], left: parts[0] };
    if (parts.length === 2) return { top: parts[0], right: parts[1], bottom: parts[0], left: parts[1] };
    if (parts.length === 3) return { top: parts[0], right: parts[1], bottom: parts[2], left: parts[1] };
    return { top: parts[0], right: parts[1], bottom: parts[2], left: parts[3] };
  };
  const describe = (c) => {
    const r = c.getBoundingClientRect();
    const cs = getComputedStyle(c);
    const btn = c.querySelector('.stagedAttachmentRemove');
    const thumb = c.querySelector('.stagedAttachmentThumb');
    const name = c.querySelector('.stagedAttachmentName');
    const meta = c.querySelector('.stagedAttachmentMeta');
    const phase = c.querySelector('.stagedAttachmentPhase');
    return {
      text: c.textContent.trim(),
      pending: c.dataset.pending || null,
      failed: c.dataset.failed || null,
      rect: { x: +r.left.toFixed(2), y: +r.top.toFixed(2), w: +r.width.toFixed(2), h: +r.height.toFixed(2) },
      computed: { height: cs.height, minHeight: cs.minHeight, fontSize: cs.fontSize },
      nameFontSize: name ? getComputedStyle(name).fontSize : null,
      metaFontSize: meta ? getComputedStyle(meta).fontSize : null,
      phaseFontSize: phase ? getComputedStyle(phase).fontSize : null,
      hasThumb: !!thumb,
      thumbHeight: thumb ? +thumb.getBoundingClientRect().height.toFixed(2) : null,
      hasRetry: !!c.querySelector('.stagedAttachmentRetry'),
      remove: btn ? {
        w: +btn.getBoundingClientRect().width.toFixed(2),
        h: +btn.getBoundingClientRect().height.toFixed(2),
        afterInset: getComputedStyle(btn, '::after').inset,
      } : null,
      removeLane: btn ? (() => {
        const inset = parseInset(getComputedStyle(btn, '::after').inset);
        const r = btn.getBoundingClientRect();
        return { inset, w: +(r.width - inset.left - inset.right).toFixed(2),
                 h: +(r.height - inset.top - inset.bottom).toFixed(2) };
      })() : null,
      removeHit: hitExtent(btn),
      retryHit: hitExtent(c.querySelector('.stagedAttachmentRetry')),
      // Overflow of the chip's own content boxes (never the hit lane's
      // absolutely positioned box, which is allowed to leave the chip).
      contentOverflowY: +Math.max(0, ...Array.from(c.children, (child) =>
        child.getBoundingClientRect().bottom - r.bottom)).toFixed(2),
      scrollOverflowY: c.scrollHeight - Math.ceil(r.height),
    };
  };
  // The cancel lane is invisible and may leave the chip. Measure its real box
  // against the two surfaces it must respect: the composer band above the chip
  // and the composer's entry row below it.
  const probeLane = (btn, textarea, composer) => {
    if (!btn || !textarea || !composer) return null;
    const b = btn.getBoundingClientRect();
    const t = textarea.getBoundingClientRect();
    const inset = parseInset(getComputedStyle(btn, '::after').inset);
    // inset offsets are measured inward from each edge, so a negative (outset)
    // value adds on the top/left edges and the box's own bottom/right edges
    // extend by the negative of the bottom/right inset.
    const laneTop = b.top + inset.top;
    const laneBottom = b.bottom - inset.bottom;
    const laneLeft = b.left + inset.left;
    const laneRight = b.right - inset.right;
    const x = Math.round(b.left + b.width / 2);
    const y = Math.round(t.top + 1);
    const hit = document.elementFromPoint(x, y);
    const owns = (px, py) => {
      const el = document.elementFromPoint(px, py);
      return !!el && (el === btn || btn.contains(el));
    };
    return {
      inset, lane: { w: +(laneRight - laneLeft).toFixed(2), h: +(laneBottom - laneTop).toFixed(2) },
      gapToEntryRow: +(t.top - b.bottom).toFixed(2),
      clearsEntryRowBy: +(t.top - laneBottom).toFixed(2),
      insideComposerBandBy: +(laneTop - composer.getBoundingClientRect().top).toFixed(2),
      probe: { x, y },
      probeHit: hit ? hit.tagName + '.' + (typeof hit.className === 'string' ? hit.className : '') : null,
      probeOwnedByCancel: owns(x, y),
      laneTopOwnedByCancel: owns(x, Math.round(laneTop + 1)),
      laneBottomOwnedByCancel: owns(x, Math.round(laneBottom - 1)),
    };
  };
  const chips = Array.from((tray || document).querySelectorAll('.stagedAttachmentChip'));
  const textarea = document.querySelector('.composer textarea');
  const composer = document.querySelector('.composer');
  const lastChip = chips[chips.length - 1];
  const lastRemove = lastChip ? lastChip.querySelector('.stagedAttachmentRemove') : null;
  return {
    viewport: { w: innerWidth, h: innerHeight, coarse: matchMedia('(pointer: coarse)').matches, narrow: matchMedia('(max-width: 880px)').matches },
    tray: tray ? { h: +tray.getBoundingClientRect().height.toFixed(2), display: getComputedStyle(tray).display, count: chips.length } : null,
    formHeight: form ? +form.getBoundingClientRect().height.toFixed(2) : null,
    chips: chips.map(describe),
    laneProbe: probeLane(lastRemove, textarea, composer),
  };
})()'''


# The coarse-pointer emulation is a browser launch flag (Chromium's Blink
# setting, the same signal `(pointer: coarse)` reads), so the mobile pass needs
# its own browser session; agent-browser's device emulation alone reports a fine
# pointer here.
BROWSER = {"session": a.session, "coarse": False}
COARSE_ARGS = "--blink-settings=primaryPointerType=2"


def use_coarse(enable):
    BROWSER["session"] = f"{a.session}-coarse" if enable else a.session
    BROWSER["coarse"] = enable


def ab(*args):
    env = dict(os.environ)
    if BROWSER["coarse"]:
        env["AGENT_BROWSER_ARGS"] = COARSE_ARGS
    r = subprocess.run(['agent-browser', '--session', BROWSER["session"], *args],
                       capture_output=True, text=True, timeout=90, env=env)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def js(code):
    return json.loads(ab('eval', code, '--json'))['data']['result']


def http(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=40) as r:
        return json.loads(r.read())


def wait(fn, seconds=30, what='condition'):
    until = time.monotonic() + seconds
    value = None
    while time.monotonic() < until:
        value = fn()
        if value:
            return value
        time.sleep(.3)
    raise AssertionError(f'timed out waiting for {what}: last={value}')


def check(name, value):
    report[name] = value
    (art / 'report.json').write_text(json.dumps(report, indent=2))
    print(name, json.dumps(value), flush=True)


def measure():
    return js(MEASURE)


def chips():
    return measure()['chips']


def fault(**kwargs):
    http('/__fault', kwargs)


def require(condition, message, detail):
    if not condition:
        raise AssertionError(f'{message}: {json.dumps(detail)}')


def ensure_app():
    """Land on the composer with a selected session, across emulation reloads."""
    wait(lambda: js('Boolean(document.querySelector("#pw") || document.querySelector("#sessions .session"))'), 40, 'boot')
    if js('Boolean(document.querySelector("#pw"))'):
        ab('fill', '#pw', PASSWORD)
        ab('click', '#loginBtn')
    wait(lambda: js('Boolean(document.querySelector("#sessions .session"))'), 40, 'session list')
    if js('Boolean(document.querySelector("#attachBtn") && document.querySelector("#attachBtn").disabled)'):
        ab('click', '#sessions .session')
    wait(lambda: js('!document.querySelector("#attachBtn").disabled'), 40, 'attach button ready')
    wait(lambda: js('Boolean(document.querySelector("#stagedAttachments"))'), 30, 'attachment tray')


def clear_attachments():
    if not chips():
        return
    http(f'/api/sessions/{SESSION}/attachments/clear', {})
    wait(lambda: not chips(), 40, 'attachments cleared')


def png_bytes():
    buf = io.BytesIO()
    Image.new('RGB', (16, 16), (180, 40, 40)).save(buf, 'PNG')
    return buf.getvalue()


def upload_file(name, payload):
    path = art / name
    path.write_bytes(payload)
    ab('upload', '#imgInput', str(path))
    return name


def find_chip(name, pending):
    exact = [c for c in chips() if bool(c['pending']) is pending and name[:12] in c['text']]
    return exact[0] if exact else None


def wait_settled(name, seconds=40):
    """The completed state the user sees: the in-flight entry retired and one
    committed chip left. The origin commits before its (held) response reaches
    the browser, so a catalog poll can briefly render the committed chip beside
    the still-in-flight entry; only the settled state is comparable."""
    def probe():
        cs = chips()
        if len(cs) == 1 and not cs[0]['pending'] and not cs[0]['failed'] and name[:12] in cs[0]['text']:
            return cs[0]
        return None
    return wait(probe, seconds, f'settled committed chip {name}')


def crop_pill(shot, chip):
    """Crop the measured chip (plus margin) out of the viewport screenshot."""
    path = art / shot
    img = Image.open(path)
    scale = img.width / js('innerWidth')
    box = chip['rect']
    pad = 16
    crop = img.crop((max(0, int((box['x'] - pad) * scale)), max(0, int((box['y'] - pad) * scale)),
                     min(img.width, int((box['x'] + box['w'] + pad) * scale)),
                     min(img.height, int((box['y'] + box['h'] + pad) * scale))))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    crop.save(art / shot.replace('.png', '-crop.png'))


def scenario(label, file_name, payload, hold=8):
    """Upload `file_name` with its response held open, measuring the in-flight
    chip and then the settled committed chip of the same file."""
    ensure_app()
    fault(delay_upload=hold)
    upload_file(file_name, payload)
    pend = wait(lambda: find_chip(file_name, True), 30, f'in-flight chip {file_name}')
    pend_view = measure()
    ab('screenshot', str(art / f'{label}-pending.png'))
    crop_pill(f'{label}-pending.png', pend)
    done = wait_settled(file_name, hold + 40)
    fault(delay_upload=0)
    ab('screenshot', str(art / f'{label}-completed.png'))
    crop_pill(f'{label}-completed.png', done)
    view = measure()
    check(f'{label}_transition', {
        'pending': pend, 'committed': done,
        'pendingLayout': {'tray': pend_view['tray'], 'formHeight': pend_view['formHeight'],
                          'laneProbe': pend_view['laneProbe']},
        'settledLayout': {'tray': view['tray'], 'formHeight': view['formHeight'],
                          'laneProbe': view['laneProbe']},
        'viewport': view['viewport'],
    })
    return pend, done, view, pend_view


http('/api/login', {'password': PASSWORD})
ab('open', base)
ensure_app()

# ---------------------------------------------------------------- desktop
ab('set', 'viewport', '1280', '800')
fault(delay_upload=0)
desktop_pend, desktop_done, desktop_view, desktop_pend_view = scenario(
    'desktop-nonimage', 'The_49ers_ZDW_-_Cool_Down_.mp4', b'x' * 4096)
clear_attachments()

img_pend, img_done, img_view, img_pend_view = scenario('desktop-image', 'team-photo.png', png_bytes())
clear_attachments()

# ------------------------------------------------- mobile (coarse pointer)
use_coarse(True)
ab('open', base)
ab('set', 'viewport', '390', '844')
mobile_pend, mobile_done, mobile_view, mobile_pend_view = scenario(
    'mobile-nonimage', 'The_49ers_ZDW_-_Cool_Down_.mp4', b'x' * 4096)
clear_attachments()
use_coarse(False)

# ------------------------------------ failure/recovery layout is intentional
# content: it may wrap and grow, and it must stay actionable.
ensure_app()
fault(lose_next=True, fail_status=True)
failed_name = upload_file('The_49ers_ZDW_-_Cool_Down_.mp4', b'x' * 4096)
failed = wait(lambda: (lambda f: f[0] if f else None)(
    [c for c in chips() if c['failed'] and c['hasRetry']]), 40, 'failed chip with Retry')
ab('screenshot', str(art / 'failed-recovery.png'))
crop_pill('failed-recovery.png', failed)
check('failed_recovery_layout', failed)
fault(lose_next=False, fail_status=False)
ab('click', '.stagedAttachmentRetry')
failed_done = wait_settled(failed_name, 40)
check('failed_retry_commits', failed_done)
clear_attachments()

# ------------------------------------------------------------- assertions
heights = {
    'desktop-nonimage': (desktop_pend['rect']['h'], desktop_done['rect']['h']),
    'desktop-image': (img_pend['rect']['h'], img_done['rect']['h']),
    'mobile-nonimage': (mobile_pend['rect']['h'], mobile_done['rect']['h']),
}
check('chip_heights_pending_vs_committed', heights)
check('viewport_probes', {
    'desktop': desktop_view['viewport'], 'mobile': mobile_view['viewport'],
})
check('pending_remove_hit_targets', {
    'desktop-nonimage': desktop_pend['removeHit'],
    'desktop-image': img_pend['removeHit'],
    'mobile-nonimage': mobile_pend['removeHit'],
})
check('committed_remove_hit_targets', {
    'desktop-nonimage': desktop_done['removeHit'],
    'desktop-image': img_done['removeHit'],
    'mobile-nonimage': mobile_done['removeHit'],
})

if a.expect == 'baseline':
    for label, (pend_h, done_h) in heights.items():
        require(pend_h > done_h + 4, f'baseline should show a taller in-flight chip ({label})',
                {'pending': pend_h, 'committed': done_h})
    require(desktop_pend['remove']['w'] == 44, 'baseline in-flight remove keeps its 44px box', desktop_pend['remove'])
    print('BASELINE reproduced: the in-flight chip is taller than the committed chip', flush=True)
else:
    for label, (pend_h, done_h) in heights.items():
        require(pend_h == 32 and done_h == 32,
                f'chip height must stay 32px in both states ({label})',
                {'pending': pend_h, 'committed': done_h})
    require(desktop_view['viewport']['narrow'] is False and desktop_view['viewport']['coarse'] is False,
            'desktop pass must be a fine-pointer desktop viewport', desktop_view['viewport'])
    require(mobile_view['viewport']['narrow'] is True and mobile_view['viewport']['coarse'] is True,
            'mobile emulation probe', mobile_view['viewport'])
    for label, pend in (('desktop-nonimage', desktop_pend), ('desktop-image', img_pend), ('mobile-nonimage', mobile_pend)):
        hit = pend['removeHit']
        require(hit and hit['ok'], f'in-flight cancel must be hit-testable ({label})', pend['remove'])
        # A full 44x44 of usable target, measured by walking elementFromPoint.
        require(hit['w'] >= 44 and hit['h'] >= 44,
                f'in-flight cancel target must be a full 44x44 ({label})', hit)
        require(pend['removeLane']['w'] >= 44 and pend['removeLane']['h'] >= 44,
                f'in-flight cancel lane must be laid out 44x44 ({label})', pend['removeLane'])
        require(pend['contentOverflowY'] <= 0.5,
                f'in-flight chip content must fit its box ({label})',
                {'overflowY': pend['contentOverflowY'], 'scrollOverflow': pend['scrollOverflowY']})
    for label, done in (('desktop-nonimage', desktop_done), ('desktop-image', img_done), ('mobile-nonimage', mobile_done)):
        require(done['remove']['w'] == 18 and done['remove']['h'] == 18,
                f'committed remove visual box must stay 18x18 ({label})', done['remove'])
        require(done['removeHit'] and done['removeHit']['w'] >= 32,
                f'committed remove target must not shrink ({label})', done['removeHit'])
    for label, pend, done in (('desktop-nonimage', desktop_pend, desktop_done),
                              ('desktop-image', img_pend, img_done),
                              ('mobile-nonimage', mobile_pend, mobile_done)):
        require(pend['remove']['w'] == 18 and pend['remove']['h'] == 18,
                f'in-flight cancel box must match the completed box ({label})', pend['remove'])
        require(pend['computed']['height'] == done['computed']['height'],
                f'in-flight and settled chips must share one computed height ({label})',
                {'pending': pend['computed'], 'committed': done['computed']})
    # No layout jump between the in-flight and settled states of one upload.
    for label, pv, sv in (('desktop-nonimage', desktop_pend_view, desktop_view),
                          ('desktop-image', img_pend_view, img_view),
                          ('mobile-nonimage', mobile_pend_view, mobile_view)):
        require(pv['formHeight'] == sv['formHeight'] and pv['tray']['h'] == sv['tray']['h'] == 32,
                f'the composer and tray must not change height across the transition ({label})',
                {'pending': {'form': pv['formHeight'], 'tray': pv['tray']},
                 'settled': {'form': sv['formHeight'], 'tray': sv['tray']}})
    require(img_pend['hasThumb'] and img_done['hasThumb'], 'image chips keep their thumbnail',
            {'pending': img_pend['hasThumb'], 'committed': img_done['hasThumb']})
    require(img_pend['thumbHeight'] == 22 and img_done['thumbHeight'] == 22,
            'image thumbnails keep their 22px box',
            {'pending': img_pend['thumbHeight'], 'committed': img_done['thumbHeight']})
    # The wider lane must still leave the composer's entry row its own pixels:
    # its bottom edge stops above the textarea, its top edge stays in the band.
    for label, view in (('desktop-nonimage', desktop_pend_view), ('desktop-image', img_pend_view),
                        ('mobile-nonimage', mobile_pend_view)):
        probe = view['laneProbe']
        require(probe['lane']['w'] >= 44 and probe['lane']['h'] >= 44,
                f'the measured lane box must be 44x44 ({label})', probe['lane'])
        require(probe and probe['clearsEntryRowBy'] >= 1,
                f'the cancel lane must end above the composer entry row ({label})', probe)
        require(probe['insideComposerBandBy'] >= 0,
                f'the cancel lane must stay inside the composer band ({label})', probe)
        require(probe['laneTopOwnedByCancel'] and probe['laneBottomOwnedByCancel'],
                f'both lane ends must be hit-testable ({label})', probe)
        require(probe['probeOwnedByCancel'] is False and str(probe['probeHit']).startswith('TEXTAREA'),
                f'the composer entry row keeps the pixels below the lane ({label})', probe)
    require(mobile_pend['removeHit']['w'] >= 44 and mobile_pend['removeHit']['h'] >= 44,
            'the coarse-pointer lane owns a full 44x44', mobile_pend['removeHit'])
    require(failed['hasRetry'] and failed['removeHit']['ok'] and failed['removeHit']['w'] >= 44,
            'failure recovery keeps a hit-testable >=44px cancel target', failed)
    require(failed['retryHit'] and failed['retryHit']['ok'] and failed['retryHit']['w'] >= 44
            and failed['retryHit']['h'] >= 44,
            'Retry stays a >=44px hit-testable target', failed['retryHit'])
    require(failed['rect']['h'] >= 44, 'failure recovery keeps its wrapping height', failed['rect'])
    require(failed_done['rect']['h'] == 32, 'recovery settles back to the single-line chip', failed_done['rect'])
    print('PASS in-flight chip matches committed chip geometry; 44px cancel target preserved', flush=True)
print('artifacts', art, flush=True)
