#!/usr/bin/env bash
# Docker-only sidebar verification. All application bytes come from the commit
# archive; the only substitute is the idle backend CLI (the broker is real).
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: scripts/docker_sidebar_snooze_drag_verify.sh [commit] (default HEAD)'
  exit 0
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ref="${1:-HEAD}"
commit="$(git -C "$repo" rev-parse --verify "${ref}^{commit}")"
# The snapshot must be a commit, never the working tree: other work is in flight
# in the checkout and `git archive` is what keeps it out of the verified bytes.
[[ -n "$commit" ]] || { echo "cannot resolve $ref to a commit" >&2; exit 2; }
printf 'verifying %s (%s)\n' "$ref" "$commit" >&2
[[ -z "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]] || { echo 'Unset host app-dir overrides' >&2; exit 2; }
artifacts="$repo/.memory/tasks/2026-11-13-sidebar-snooze-and-drag/verify/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$artifacts/build"
# Read-only Git commands are confined to resolving/exporting the build snapshot.
git -C "$repo" archive --format=tar "$commit" | tar -x -C "$artifacts/build"
printf '%s\n' "$commit" > "$artifacts/commit.txt"
container="codoxear-sidebar-verify-$$"
image="codoxear-sidebar-verify:${commit:0:12}"
browser_session="codoxear-sidebar-verify-$$"
cleanup() {
  agent-browser --session "$browser_session" close >/dev/null 2>&1 || true
  sudo docker logs "$container" > "$artifacts/container.log" 2>&1 || true
  sudo docker cp "$container:/home/tester/server.log" "$artifacts/server.log" >/dev/null 2>&1 || true
  sudo docker rm -f "$container" > "$artifacts/cleanup.txt" 2>&1 || true
  python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1])' "$artifacts/build"
}
trap cleanup EXIT
# Self-contained tmux rig from the archived Dockerfile. Build the exact archived
# frontend sources too, matching deploy.sh rather than trusting a stale bundle.
cat > "$artifacts/build/docker/sidebar-verify.Dockerfile" <<'DOCKER'
FROM node:25-slim AS assets
WORKDIR /source
COPY . .
RUN npm ci && npx esbuild codoxear/static/app.js --bundle --minify --format=esm --target=es2020 --outfile=codoxear/static/dist/app.bundle.js
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tmux ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir 'Pillow>=9.0' 'py-vapid>=1.9.2' 'pywebpush>=2.3.0' 'websockets>=14,<17'
COPY . /workspace
COPY --from=assets /source/codoxear/static/dist /workspace/codoxear/static/dist
RUN useradd --create-home --uid 1000 tester && chmod -R a+rX /workspace
USER tester
ENV HOME=/home/tester PYTHONPATH=/workspace PYTHONDONTWRITEBYTECODE=1 CODEX_WEB_PASSWORD=sidebar-verify-password CODEX_WEB_HOST=0.0.0.0 CODEX_WEB_PORT=19558 CODEX_WEB_DEFAULT_AGENT_BACKEND=pi CODEX_WEB_TMUX_SESSION=codoxear PI_BIN=/home/tester/fake-agent
WORKDIR /workspace
CMD ["sleep", "infinity"]
DOCKER
sudo docker build -f "$artifacts/build/docker/sidebar-verify.Dockerfile" -t "$image" "$artifacts/build" > "$artifacts/docker-build.log" 2>&1
# Docker allocates an unused host loopback port. Never bind or request 8743.
sudo docker run -d --init --name "$container" -p 127.0.0.1::19558 "$image" > "$artifacts/container-id.txt"
port="$(sudo docker port "$container" 19558/tcp | sed 's/127.0.0.1://')"
[[ "$port" != 8743 && "$port" =~ ^[0-9]+$ ]] || { echo 'Invalid isolated port' >&2; exit 2; }
export VERIFY_ARTIFACTS="$artifacts" VERIFY_CONTAINER="$container" VERIFY_PORT="$port" VERIFY_BROWSER="$browser_session" VERIFY_COMMIT="$commit"
echo "Artifacts: $artifacts; isolated loopback port: $port"
python3 - <<'PY'
import json, os, pathlib, subprocess, time, traceback
A = pathlib.Path(os.environ['VERIFY_ARTIFACTS'])
C = os.environ['VERIFY_CONTAINER']
B = os.environ['VERIFY_BROWSER']
BASE = 'http://127.0.0.1:' + os.environ['VERIFY_PORT']
PASSWORD = 'sidebar-verify-password'
report = {'commit': os.environ['VERIFY_COMMIT'], 'base': BASE, 'container': C,
          'steps': {str(i): {'status': 'not-run', 'evidence': []} for i in range(1,15)},
          'residualRisks': ['Linux Chromium with synthetic pointers does not prove physical iOS/Android picker or scroll arbitration.']}
current = None
counter = 0

def save(name, obj):
    p = A / (name + '.json'); p.write_text(json.dumps(obj, indent=2)); return p.name

def flush(): save('report', report)

def cmd(args, **kwargs):
    result=subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        save('command-failure',{'command':args,'status':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        raise RuntimeError(f'command failed ({result.returncode}): {args[0]}; {result.stdout} {result.stderr}')
    return result.stdout

def dx(*args): return cmd(['sudo','docker','exec', C, *args])

def check(ok, msg):
    if not ok: raise AssertionError(msg)

def start(n):
    global current
    current = n
    print(f'STEP {n}', flush=True)

def passed(n, *evidence):
    report['steps'][str(n)] = {'status': 'pass', 'evidence': list(evidence)}; flush()
    print(f'PASS {n}: {evidence}', flush=True)

def api(method, path, body=None, status=200, auth=True, label=None):
    global counter
    counter += 1
    args = ['curl','-sS','--max-time','40','-w','\n%{http_code}','-X',method]
    if auth: args += ['-b',str(A/'cookies.txt')]
    if path == '/api/login': args += ['-c',str(A/'cookies.txt')]
    if body is not None: args += ['-H','Content-Type: application/json','--data',json.dumps(body)]
    raw = cmd(args + [BASE+path]); text, code = raw.rsplit('\n',1)
    try: data=json.loads(text)
    except ValueError: data=text
    name=save(label or f'http-{counter:03}', {'method':method,'path':path,'request':body,'auth':auth,'status':int(code),'response':data})
    check(int(code)==status, f'{name}: expected {status}, got {code}: {data}')
    return data

def rows(label=None): return api('GET','/api/sessions',label=label)['sessions']
def row(sid, listing=None): return next(r for r in (listing if listing is not None else rows()) if r['session_id']==sid)
def ids(listing): return [r['session_id'] for r in listing]
def group(r): return 'waiting' if r.get('blocked') else 'later' if r.get('snoozed') else 'now'
def edit(r, **changes):
    body={'name':r.get('alias') or '', 'priority_offset':r.get('priority_offset',0), 'snooze_until':r.get('snooze_until'), 'snoozed_indefinitely':r.get('snoozed_indefinitely',False), 'dependency_session_id':r.get('dependency_session_id')}
    body.update(changes)
    return api('POST',f"/api/sessions/{r['session_id']}/edit",body)

def wait(fn, timeout=45):
    end=time.monotonic()+timeout; last=None
    while time.monotonic()<end:
        last=fn()
        if last: return last
        time.sleep(.4)
    raise AssertionError(f'timed out after {timeout}s; last={last}')

def browser(*args): return cmd(['agent-browser','--session',B,*args], timeout=60)
def ev(js):
    result=json.loads(browser('eval',js,'--json'))
    check(result.get('success'), f'browser eval failed: {result}')
    return result['data']['result']
def shot(name): browser('screenshot',str(A/(name+'.png')))
def dom():
    return ev('''(() => { let g='now'; return [...document.querySelector('#sessions').children].flatMap(n=>{if(n.matches('.sessionGroupHeader')){g=n.dataset.sessionGroup;return []}return n.matches('.session[data-session-id]')?[{id:n.dataset.sessionId,group:g,active:n.classList.contains('active')}]:[]}) })()''')
def show_sidebar():
    ev("(() => { const b=document.querySelector('#toggleSidebarBtn'); if(b && getComputedStyle(b).display!=='none' && !document.body.classList.contains('sidebar-open')) b.click(); return true })()")
def open_edit(sid):
    # Invoke the actual card action (same controller in both locked DOM branches).
    ev(f'''(() => {{ const c=document.querySelector('.session[data-session-id="{sid}"]'); if(!c)throw Error('missing card'); c.querySelector('[aria-label="Edit conversation"]').click(); return true }})()''')
    wait(lambda: ev("document.querySelector('#editViewer').open"), 5)
def chip(mode): browser('click',f'[data-snooze-mode="{mode}"]')
def toast(): return ev("document.querySelector('#toast')?.textContent || ''")
def settle_order(expected):
    wait(lambda: ids(rows())==expected)
    wait(lambda: [r['id'] for r in dom()]==expected)

def drag(sid, target, pointer, label):
    # Mouse activation is bound to the press target, not document: deliver its
    # initial 8px pull there, then body moves exercise document capture handlers.
    js=f'''(async () => {{
      const c=document.querySelector('.session[data-session-id="{sid}"]');
      const p=c.classList.contains('desktop')?c:c.querySelector('.sessionContent');
      const r=p.getBoundingClientRect(), x=r.left+Math.min(75,r.width/2), y=r.top+r.height/2;
      const opts={{bubbles:true,cancelable:true,pointerId:79,pointerType:'{pointer}',button:0,isPrimary:true}};
      const fire=(n,t,x,y)=>n.dispatchEvent(new PointerEvent(t,{{...opts,clientX:x,clientY:y,buttons:t==='pointerup'?0:1}}));
      fire(p,'pointerdown',x,y);
      if('{pointer}'==='touch')await new Promise(r=>setTimeout(r,400));
      else fire(p,'pointermove',x,y-8);
      const active=document.body.dataset.dragging==='true';
      const t=document.querySelector({json.dumps(target)}); if(!t)throw Error('missing target '+{json.dumps(target)});
      const tr=t.getBoundingClientRect(); const ty=t.matches('.session')?tr.top+5:tr.top+tr.height/2;
      fire(document.body,'pointermove',x,ty);
      const drop=[...document.querySelectorAll('[data-drop]')].map(n=>({{group:n.dataset.sessionGroup,kind:n.dataset.drop}}));
      fire(document.body,'pointerup',x,ty);
      // A browser-generated post-drop click must not select the dragged card.
      c.click();
      return {{sid:'{sid}',pointer:'{pointer}',desktop:c.classList.contains('desktop'),viewport:{{width:innerWidth,height:innerHeight}},active,source:{{x,y}},target:{{x,y:ty}},drop,selected:[...document.querySelectorAll('.session.active')].map(n=>n.dataset.sessionId)}};
    }})()'''
    data=ev(js); save(label,data); check(data['active'],f'drag did not activate: {data}')
    return data

try:
    flush()
    # A real server + real broker + real tmux; backend is an idle, offline stub.
    dx('sh','-c', '''cat > "$HOME/fake-agent" <<'EOF'
#!/bin/sh
case "$1" in --help|-h|--version|-v) echo 'fake agent'; exit 0;; esac
while :; do printf 'AGENT:%s\\n' "$PWD"; sleep 2; done
EOF
chmod +x "$HOME/fake-agent"
mkdir -p "$HOME/.local/share/codoxear" /tmp/ws/alpha /tmp/ws/beta /tmp/ws/gamma
''')
    def launch_server():
        cmd(['sudo','docker','exec','-d',C,'sh','-c','python3 -m codoxear.server >> "$HOME/server.log" 2>&1 & echo $! > "$HOME/server.pid"; wait'])
        wait(lambda: subprocess.run(['curl','-s','--max-time','2','-o','/dev/null','-w','%{http_code}',BASE+'/api/me'],text=True,capture_output=True).stdout=='401',60)
    launch_server()
    api('POST','/api/login',{'password':PASSWORD},label='login')
    for name in ['alpha','beta','gamma']:
        api('POST','/api/sessions',{'cwd':'/tmp/ws/'+name,'agent_backend':'pi','create_in_tmux':True},label='launch-'+name)
    wait(lambda: len(rows())==3,60)
    # Deliberately below the clipped-score ceiling. Each session retains its own
    # user offset; this avoids the documented score=1.0 tiebreak non-placement.
    for r, offset in zip(rows(),[-.25,-.45,-.65]): edit(r,priority_offset=offset)
    initial=rows('initial-three-sessions'); sid, second, third=ids(initial)

    start(1)
    response=api('POST',f'/api/sessions/{sid}/snooze',{'indefinite':True},label='step01-set')
    check(response=={'ok':True,'snooze_until':None,'snoozed_indefinitely':True},str(response))
    listing=rows('step01-list'); s=row(sid,listing)
    check(s['snoozed'] and group(s)=='later' and ids(listing)[-1]==sid,'indefinite row not after Now')
    passed(1,'step01-set.json','step01-list.json')

    start(2)
    disk=json.loads(dx('cat','/home/tester/.local/share/codoxear/session_sidebar.json')); save('step02-disk',disk)
    check(disk[sid]['snoozed_indefinitely'] is True and 'snooze_until' not in disk[sid],str(disk))
    brokers_before=dx('sh','-c','for f in "$HOME"/.local/share/codoxear/socks/*.json; do cat "$f"; done')
    server_pid=dx('cat','/home/tester/server.pid').strip()
    dx('python3','-c','import os,signal,sys; os.kill(int(sys.argv[1]),signal.SIGTERM)',server_pid)
    wait(lambda: dx('sh','-c',f'if kill -0 {server_pid} 2>/dev/null; then echo alive; else echo gone; fi').strip()=='gone')
    launch_server()
    api('POST','/api/login',{'password':PASSWORD})
    listing=rows('step02-restarted-list')
    check(row(sid,listing)['snoozed'],'restart lost indefinite flag')
    brokers_after=dx('sh','-c','for f in "$HOME"/.local/share/codoxear/socks/*.json; do cat "$f"; done')
    save('step02-processes',{'terminated_server_pid':server_pid,'restarted_server_pid':dx('cat','/home/tester/server.pid').strip(),'brokers_before':brokers_before,'brokers_after':brokers_after})
    passed(2,'step02-disk.json','step02-restarted-list.json','step02-processes.json')

    start(3)
    edit(row(second),snooze_until=int(time.time())-3600,snoozed_indefinitely=False)
    listing=rows('step03-expiry-list')
    disk=json.loads(dx('cat','/home/tester/.local/share/codoxear/session_sidebar.json')); save('step03-pruned-disk',disk)
    check(not row(second,listing)['snoozed'] and row(second,listing)['snooze_until'] is None,'past timed snooze did not expire')
    check(row(sid,listing)['snoozed'] and disk[sid].get('snoozed_indefinitely') is True,'indefinite expired')
    passed(3,'step03-expiry-list.json','step03-pruned-disk.json')

    start(4)
    api('POST',f'/api/sessions/{sid}/snooze',{'indefinite':False},label='step04-clear')
    check(group(row(sid,rows('step04-list')))=='now','clear did not return to Now')
    passed(4,'step04-clear.json','step04-list.json')

    start(5)
    before=rows('step05-before'); above, below, moving=ids(before)
    result=api('POST',f'/api/sessions/{moving}/reorder',{'above':above,'below':below},label='step05-reorder')
    after=rows('step05-after')
    check(result['applied'] is True and ids(after)==[above,moving,below],str(result))
    changed=[r['session_id'] for r in after if r['priority_offset'] != row(r['session_id'],before)['priority_offset']]
    check(changed==[moving],f'offsets changed: {changed}')
    save('step05-offset-diff',{'changed':changed,'before':{r['session_id']:r['priority_offset'] for r in before},'after':{r['session_id']:r['priority_offset'] for r in after}})
    api('POST',f'/api/sessions/{moving}/reorder',{'above':None,'below':None},400,label='step05-both-null')
    api('POST',f'/api/sessions/{moving}/reorder',{'above':'unknown','below':below},400,label='step05-unknown-neighbor')
    api('POST','/api/sessions/unknown/reorder',{'above':above,'below':below},404,label='step05-unknown-session')
    api('POST',f'/api/sessions/{moving}/reorder',{'above':above,'below':below},401,False,label='step05-no-auth')
    passed(5,'step05-before.json','step05-reorder.json','step05-after.json','step05-offset-diff.json','step05-{both-null,unknown-neighbor,unknown-session,no-auth}.json')

    start(6)
    def tmux_matches():
        listing=rows(); names=[r['display_name'] for r in listing]
        actual=dx('tmux','list-windows','-t','codoxear','-F','#{window_name}').splitlines()
        save('step06-tmux',{'sidebar_ids':ids(listing),'sidebar_names':names,'tmux_names':actual})
        return names==actual
    wait(tmux_matches,90)
    passed(6,'step06-tmux.json')

    # B. Browser matrix. Each later step only becomes pass when all variants pass.
    browser('set','viewport','1100','900')
    browser('open',BASE+'/')
    browser('errors','--clear','--json'); browser('console','--clear','--json')
    browser('fill','#pw',PASSWORD); browser('click','#loginBtn')
    wait(lambda: ev("document.querySelectorAll('#sessions .session').length===3"),30)
    ev("window.__verifyUnhandled=[]; window.addEventListener('unhandledrejection',e=>window.__verifyUnhandled.push(String(e.reason))); true")
    matrix=[]
    for width,height in [(393,852),(1100,900)]:
        browser('set','viewport',str(width),str(height))
        for family in ['clay','paper','slate']:
            tag=f'{width}-{family}'
            # Theme through the real Settings writer, not DOM theme spoofing.
            ev("document.querySelector('#settingsBtnSide').click(); true")
            browser('click',f'[data-theme-family="{family}"]'); browser('click','[data-theme-mode="light"]')
            time.sleep(.5); browser('click','#settingsCloseBtn')
            open_edit(sid)
            start(7)
            chip('indefinitely')
            indefinite_hidden=ev("getComputedStyle(document.querySelector('#editSnoozeCustomRow')).display==='none'")
            chip('custom')
            data=ev('''(() => { const row=document.querySelector('#editSnoozeCustomRow'); const inputs=[...row.querySelectorAll('input')]; return { chips:[...document.querySelectorAll('#editSnoozeButtons .choiceChip')].map(n=>n.querySelector('.chipLabel').textContent), display:getComputedStyle(row).display, fields:inputs.map(n=>{ const r=n.getBoundingClientRect(),w=n.parentElement.getBoundingClientRect(),cs=getComputedStyle(n),indicator=getComputedStyle(n,'::-webkit-calendar-picker-indicator');return {id:n.id,type:n.type,value:n.value,height:r.height,top:r.top,left:r.left,width:r.width,wrapperHeight:w.height,scrollWidth:n.scrollWidth,clientWidth:n.clientWidth,fontSize:cs.fontSize,indicator:{display:indicator.display,width:indicator.width,height:indicator.height,opacity:indicator.opacity,pointerEvents:indicator.pointerEvents},right:r.right,bottom:r.bottom}}),chipHeight:document.querySelector('#editSnoozeButtons .choiceChip').getBoundingClientRect().height, rowColumns:getComputedStyle(row).gridTemplateColumns,
              reset:(()=>{const n=document.querySelector('#editPriorityResetBtn'),cs=getComputedStyle(n),b=n.getBoundingClientRect(),r=document.createRange();r.selectNodeContents(n);return {height:b.height,width:b.width,textWidth:r.getBoundingClientRect().width,padLeft:cs.paddingLeft,padRight:cs.paddingRight};})(),
              depends:(()=>{const n=document.querySelector('#editDependencyBtn'),b=n.getBoundingClientRect();return {height:b.height,width:b.width,scrollWidth:n.scrollWidth,clientWidth:n.clientWidth};})()}; })()''')
            data['indefiniteHidesCustom']=indefinite_hidden
            save('step07-08-'+tag,data); shot('step08-'+tag)
            check(data['chips']==['No snooze','This evening','Tomorrow','This weekend','Next week','Indefinitely','Custom…'] and indefinite_hidden and data['display']=='grid' and [f['type'] for f in data['fields']]==['date','time'] and data['fields'][0]['left']<data['fields'][1]['left'] and data['fields'][0]['top']==data['fields'][1]['top'],f'{tag}: incorrect snooze field: {data}')
            start(8)
            reset=data['reset']; depends=data['depends']
            # A text button must fit its own label; a 32px square did not.
            check(reset['textWidth'] <= reset['width']-float(reset['padLeft'].removesuffix('px'))-float(reset['padRight'].removesuffix('px'))-2, f'{tag}: reset label overflows its box: {reset}')
            # The dependency picker rides the chip plane, like the snooze fields.
            check(abs(depends['height']-data['chipHeight'])<0.5 and depends['scrollWidth']<=depends['clientWidth']+1, f'{tag}: dependency picker off the chip plane or clipped: {depends}')
            for f in data['fields']:
                check(f['height']==32 and f['wrapperHeight']==44 and f['height']==data['chipHeight'] and f['scrollWidth']<=f['clientWidth'], f'{tag}: sizing/clipping defect: {f}')
                check(f['indicator']['display']!='none' and float(f['indicator']['width'].removesuffix('px'))>0 and float(f['indicator']['opacity'])>0, f'{tag}: indicator absent: {f}')
                # Trusted mouse click at the actual indicator region; screenshot
                # captures the native popup, which is outside the DOM tree.
                browser('mouse','move',str(round(f['right']-15)),str(round(f['top']+16)))
                browser('mouse','down'); browser('mouse','up'); time.sleep(.25)
                shot('step08-picker-'+tag+'-'+f['type']); browser('press','Escape')
            start(9)
            chip('indefinitely'); browser('click','#editSaveBtn')
            wait(lambda: not ev("document.querySelector('#editViewer').open"),10)
            notice=toast(); listing=rows('step09-snoozed-'+tag)
            wait(lambda: any(r['id']==sid and r['group']=='later' for r in dom()))
            check(notice and row(sid,listing)['snoozed_indefinitely'] is True,f'{tag}: save/toast failed')
            open_edit(sid)
            selected=ev("document.querySelector('[data-snooze-mode=indefinitely]').classList.contains('active')")
            check(selected,f'{tag}: reopen lost selected indefinite')
            chip('none'); browser('click','#editSaveBtn')
            wait(lambda: not ev("document.querySelector('#editViewer').open"),10)
            listing=rows('step09-cleared-'+tag)
            wait(lambda: any(r['id']==sid and r['group']=='now' for r in dom()))
            check(group(row(sid,listing))=='now',f'{tag}: clear failed')
            matrix.append({'variant':tag,'toast':notice,'reopenedIndefiniteActive':selected,'fields':'step07-08-'+tag+'.json'})
            save('step09-dialog-matrix',matrix)
    passed(7,'step07-08-{393,1100}-{clay,paper,slate}.json')
    passed(8,'step07-08-{393,1100}-{clay,paper,slate}.json','step08-picker-*.png')
    passed(9,'step09-dialog-matrix.json','step09-{snoozed,cleared}-*.json')
    start(10)
    errors=json.loads(browser('errors','--json')); console=json.loads(browser('console','--json'))
    unhandled=ev('window.__verifyUnhandled')
    save('step10-browser',{'errors':errors,'console':console,'unhandledRejections':unhandled})
    check(not errors.get('data',{}).get('errors') and not unhandled, 'browser errors/unhandled rejections; see step10-browser.json')
    passed(10,'step10-browser.json')

    # C. Drag: touch at phone size, mouse at desktop size.
    for width,height,pointer in [(393,852,'touch'),(1100,900,'mouse')]:
        tag=f'{width}-{pointer}'
        browser('set','viewport',str(width),str(height)); show_sidebar()
        wait(lambda: ev(f"document.querySelectorAll('#sessions .session.desktop').length === {3 if width > 880 else 0}"),30)
        time.sleep(.5)
        start(11)
        before=rows('step11-before-'+tag); a,b,m=ids(before)
        save('step11-dom-before-'+tag,dom())
        data=drag(m,f'.session[data-session-id="{a}"]',pointer,'step11-gesture-'+tag)
        check(m not in data['selected'],'post-drop click incorrectly selected dragged row')
        expected=[m,a,b]; settle_order(expected)
        save('step11-after-'+tag,{'api':rows(),'dom':dom(),'before':ids(before),'after':expected}); shot('step11-'+tag)
        # Time-based 500ms guard must have expired before ordinary tap selection.
        time.sleep(.55)
        ev(f'document.querySelector(\'.session[data-session-id="{m}"]\').click(); true')
        wait(lambda: any(r['id']==m and r['active'] for r in dom()))
        save('step11-next-tap-'+tag,{'clicked':m,'delayAfterPollSeconds':.55,'dom':dom()})
        show_sidebar()
        start(12)
        drag(m,'.sessionGroupHeader[data-session-group="later"]',pointer,'step12-gesture-'+tag)
        wait(lambda: row(m)['snoozed_indefinitely'] is True)
        wait(lambda: any(r['id']==m and r['group']=='later' for r in dom()))
        save('step12-after-'+tag,{'api':rows(),'dom':dom()}); shot('step12-'+tag)
        start(13)
        drag(m,'.sessionGroupHeader[data-session-group="now"]',pointer,'step13-gesture-'+tag)
        settle_order([m,a,b])
        check(not row(m)['snoozed_indefinitely'] and group(row(m))=='now','wake failed')
        save('step13-after-'+tag,{'api':rows(),'dom':dom()}); shot('step13-'+tag)
    passed(11,'step11-{before,dom-before,gesture,after}-{393-touch,1100-mouse}.json')
    passed(12,'step12-{gesture,after}-{393-touch,1100-mouse}.json')
    passed(13,'step13-{gesture,after}-{393-touch,1100-mouse}.json')
    start(14)
    browser('set','viewport','393','852'); show_sidebar()
    wait(lambda: ev("document.querySelectorAll('#sessions .sessionSwipe').length===3"),30)
    time.sleep(.5)
    swipe=ev('''(() => { const c=document.querySelector('#sessions .session'), p=c.querySelector('.sessionContent'),r=p.getBoundingClientRect(); const x=r.left+r.width/2,y=r.top+r.height/2; const fire=(t,x)=>p.dispatchEvent(new PointerEvent(t,{bubbles:true,cancelable:true,pointerType:'touch',pointerId:88,button:0,clientX:x,clientY:y})); fire('pointerdown',x);fire('pointermove',x-100);fire('pointerup',x-100); return {id:c.dataset.sessionId,swipeX:p.dataset.swipeX,deleteRect:c.querySelector('.sessionDel').getBoundingClientRect().toJSON(),contentRect:p.getBoundingClientRect().toJSON()}; })()''')
    save('step14-left-swipe',swipe); shot('step14-left-swipe')
    # The requested left swipe must expose the delete hit region, not merely move.
    deletion=ev('''(() => { const c=document.querySelector('#sessions .session'), d=c.querySelector('.sessionDel'),r=d.getBoundingClientRect(); const hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2); return {hit:hit?.outerHTML,deleteHit:hit===d||d.contains(hit)}; })()''')
    save('step14-delete-hit',deletion)
    check(deletion['deleteHit'],'left swipe did not reveal delete action (step14-delete-hit.json)')
    passed(14,'step11-gesture-*.json','step14-left-swipe.json','step14-delete-hit.json')
    save('final-browser-errors',json.loads(browser('errors','--json')))
    report['result']='pass'
except Exception as exc:
    report['result']='fail'; report['failure']={'step':current,'message':str(exc),'traceback':traceback.format_exc()}
    if current: report['steps'][str(current)]={'status':'fail','evidence':[str(exc)]}
    try: shot('failure')
    except Exception: pass
    try: save('failure-browser-errors',json.loads(browser('errors','--json')))
    except Exception: pass
    print(traceback.format_exc(),flush=True)
finally:
    # Never leave a cookie-bearing artifact behind.
    (A/'cookies.txt').unlink(missing_ok=True)
    flush()
    print('Report: '+str(A/'report.json'),flush=True)
if report['result']!='pass': raise SystemExit(1)
PY
