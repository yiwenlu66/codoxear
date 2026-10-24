#!/usr/bin/env bash
# Focused Docker/browser cwd-label proof. prepare starts an isolated server;
# verify rebuilds edited ESM and records real glyph geometry + screenshots.
# No host runtime, credentials, backend processes, or live deployment are used.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_CWD_SUFFIX_PORT:-19674}"
root="${CODOXEAR_CWD_SUFFIX_ROOT:-/tmp/codoxear-cwd-suffix-${port}}"
name="codoxear-cwd-suffix-${port}"
session="codoxear-cwd-suffix-${port}"
image="${CODOXEAR_CWD_SUFFIX_IMAGE:-codoxear-sandbox:latest}"
[[ "$port" =~ ^[1-9][0-9]{0,4}$ && "$port" != 8743 && "$port" -le 65535 ]] || { echo 'invalid/non-isolated port' >&2; exit 2; }
[[ "$root" == /tmp/codoxear-cwd-suffix-* && -z "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]] || { echo 'refusing non-temporary root/app-dir override' >&2; exit 2; }
browser() { agent-browser --session "$session" "$@"; }
stop() { browser close >/dev/null 2>&1 || true; sudo docker rm -f "$name" >/dev/null 2>&1 || true; }
bundle() {
  npx --yes esbuild "$repo/codoxear/static/app.js" --bundle --minify --format=esm --outfile="$root/app.bundle.js" > "$root/artifacts/esbuild.log" 2>&1
}
prepare() {
  mkdir -p "$root/home/.local/share/codoxear" "$root/artifacts"
  python3 - "$root" <<'PY'
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
base = '/tmp/cwd-proof/'
parent = base + 'shared-long-prefix-for-working-directory-comparison/' * 3 + '研究（中文）-ancestor/'
recent = [base + 'short', parent + 'recent-project-Alpha_123', parent + 'recent-中文-leaf-Beta_456']
filesystem = [parent + 'typed-parent/' + x for x in ['filesystem-project-Gamma_789', 'filesystem-中文-leaf-Delta_012']]
config = dict(recent=recent, filesystem=filesystem, query=parent + 'typed-parent/')
(root / 'scenario.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
(root / 'home/.local/share/codoxear/recent_cwds.json').write_text(json.dumps({p: time.time() - i for i, p in enumerate(recent)}, ensure_ascii=False))
PY
  bundle
  sudo docker run -d --rm --name "$name" -p "127.0.0.1:${port}:${port}" \
    -e HOME=/home/tester -e CODEX_WEB_PASSWORD=test-password -e CODEX_WEB_HOST=0.0.0.0 -e CODEX_WEB_PORT="$port" \
    -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$repo:/workspace:ro" -v "$root/home:/home/tester" \
    -v "$root/app.bundle.js:/workspace/codoxear/static/dist/app.bundle.js:ro" \
    -v "$root/scenario.json:/scenario.json:ro" -w /workspace "$image" bash -c '
      python3 -c '\''import json,pathlib; s=json.load(open("/scenario.json")); [pathlib.Path(p).mkdir(parents=True,exist_ok=True) for p in s["recent"]+s["filesystem"]]'\''
      exec python3 -m codoxear.server
    ' > "$root/artifacts/container-id.txt"
  for _ in $(seq 1 60); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/api/me" || true)" == 401 ]] && break
    sleep 0.5
  done
  [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/api/me")" == 401 ]]
  printf 'ready: http://127.0.0.1:%s/\nscenario: %s/scenario.json\nverify: %s verify\n' "$port" "$root" "$0"
}
probe='(() => {
  const menu = document.querySelector("#newSessionCwdMenu");
  const rect = r => ({left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height});
  const rows = [...menu.querySelectorAll("[role=option]")].map(row => {
    const label = row.querySelector(".fileMenuPath"), bounds = label.getBoundingClientRect();
    let left = bounds.left, right = bounds.right;
    for(let p=label; p && p!==document.body; p=p.parentElement) {
      if(["hidden","clip","auto","scroll"].includes(getComputedStyle(p).overflowX)) {
        const b=p.getBoundingClientRect(); left=Math.max(left,b.left); right=Math.min(right,b.right);
      }
    }
    const walker=document.createTreeWalker(label,NodeFilter.SHOW_TEXT), chars=[];
    while(walker.nextNode()) {
      const node=walker.currentNode;
      for(let i=0;i<node.length;i++) {
        const range=document.createRange(); range.setStart(node,i); range.setEnd(node,i+1);
        const r=range.getBoundingClientRect();
        chars.push({char:node.data[i],left:r.left,right:r.right,visible:r.width>0 && r.left>=left-0.6 && r.right<=right+0.6});
      }
    }
    const range=document.createRange();range.selectNodeContents(label);
    const selection=getSelection();selection.removeAllRanges();selection.addRange(range);
    const copyText=selection.toString();selection.removeAllRanges();
    const badge=row.querySelector(".cwdSuggestionSource");
    return {title:row.title,text:label.textContent,copyText,bounds:rect(bounds),clip:{left,right},chars,
      visible:chars.filter(c=>c.visible).map(c=>c.char).join(""),
      naturalOrder:chars.every((c,i)=>!i || c.left>=chars[i-1].left-0.6),
      badge:badge ? {text:badge.textContent,direction:getComputedStyle(badge).direction,bounds:rect(badge.getBoundingClientRect())}:null};
  });
  const r=menu.getBoundingClientRect();
  return {width:innerWidth,height:innerHeight,menu:rect(r),menuClient:menu.clientWidth,menuScroll:menu.scrollWidth,
    documentClient:document.documentElement.clientWidth,documentScroll:document.documentElement.scrollWidth,rows};
})()'
verify() {
  trap 'sudo docker logs "$name" > "$root/artifacts/server.log" 2>&1 || true; stop' EXIT
  bundle # regenerate from the current edited source, never a stale tracked bundle
  browser open "http://127.0.0.1:$port/" >/dev/null
  browser fill '#pw' test-password >/dev/null
  browser click '#loginBtn' >/dev/null
  browser wait 1800 >/dev/null
  for mode in desktop mobile; do
    if [[ "$mode" == desktop ]]; then browser set viewport 1280 1000 >/dev/null; else
      browser set viewport 390 844 >/dev/null
      # Narrow viewports keep the sidebar as an off-canvas drawer; open it first.
      browser find role button click --name "Toggle sidebar" >/dev/null
      browser wait 400 >/dev/null
    fi
    browser click '#newBtn' >/dev/null
    browser wait 500 >/dev/null
    browser click '#newSessionCwdInput' >/dev/null
    browser wait 500 >/dev/null
    browser eval "$probe" --json > "$root/artifacts/$mode-recent.json"
    browser screenshot "$root/artifacts/$mode-recent.png" >/dev/null
    # Select by full identity, independent of API/history ordering.
    recent_path="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["recent"][1])' "$root/scenario.json")"
    browser click "#newSessionCwdMenu [role=option][title=\"$recent_path\"]" >/dev/null
    browser eval 'document.querySelector("#newSessionCwdInput").value' --json > "$root/artifacts/$mode-recent-selection.json"
    query="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["query"])' "$root/scenario.json")"
    browser fill '#newSessionCwdInput' "$query" >/dev/null
    browser wait 1200 >/dev/null
    browser eval "$probe" --json > "$root/artifacts/$mode-filesystem.json"
    browser screenshot "$root/artifacts/$mode-filesystem.png" >/dev/null
    filesystem_path="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["filesystem"][0])' "$root/scenario.json")"
    browser click "#newSessionCwdMenu [role=option][title=\"$filesystem_path\"]" >/dev/null
    browser eval 'document.querySelector("#newSessionCwdInput").value' --json > "$root/artifacts/$mode-filesystem-selection.json"
    browser click '#newSessionCloseBtn' >/dev/null
    if [[ "$mode" == mobile ]]; then browser find role button click --name "Toggle sidebar" >/dev/null; fi
  done
  python3 - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); artifacts = root/'artifacts'; scenario=json.loads((root/'scenario.json').read_text())
def load(name): return json.loads((artifacts/name).read_text())['data']['result']
checks={}
for mode in ['desktop','mobile']:
    for source in ['recent','filesystem']:
        p=load(f'{mode}-{source}.json'); expected=scenario[source]
        titles=[r['title'] for r in p['rows']]
        if source=='recent':
            # A plain empty field is designed to show the badged recents plus
            # unbadged /api/cwd-suggest entries of "/"; require every scenario
            # recent to be present and badged, and every extra row unbadged.
            checks[f'{mode}-{source}-paths']=set(expected)<=set(titles) and all(
                (r['badge'] or {}).get('text')=='Recent' if r['title'] in expected else r['badge'] is None
                for r in p['rows'])
        else:
            checks[f'{mode}-{source}-paths']=set(titles)==set(expected)
        checks[f'{mode}-{source}-no-overflow']=p['documentScroll']<=p['documentClient']+1 and p['menuScroll']<=p['menuClient']+1 and p['menu']['right']<=p['width']+1
        for i,r in enumerate(p['rows']):
            key=f'{mode}-{source}-{i}'
            checks[key+'-natural-ltr']=r['naturalOrder']
            if r['title'] not in expected: continue  # extra root suggestions: order + overflow invariants only
            checks[key+'-full-dom-copy']=r['text']==r['copyText']==r['title']
            checks[key+'-rightmost-retained']=bool(r['chars']) and r['chars'][-1]['visible']
            if r['title'].endswith('/short'):
                checks[key+'-short-unchanged']=all(c['visible'] for c in r['chars'])
            else:
                checks[key+'-prefix-clipped']=not r['chars'][0]['visible']
            if source=='recent':
                checks[key+'-badge']=r['badge'] is not None and r['badge']['text']=='Recent' and r['badge']['direction']=='ltr' and r['badge']['bounds']['right']<=p['menu']['right']+1
            else:
                checks[key+'-badge']=r['badge'] is None
        selection=load(f'{mode}-{source}-selection.json')
        checks[f'{mode}-{source}-complete-selection']=selection==expected[1 if source=='recent' else 0]
report={'checks':checks,'passed':all(checks.values()),'visual_review_required':'Inspect screenshots for a leading ellipsis and natural mixed Chinese/ASCII punctuation order; glyph ranges alone cannot prove painted ellipsis.'}
(artifacts/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
assert report['passed'], 'browser contract failed'
PY
  printf 'artifacts: %s/artifacts\n' "$root"
}
case "${1:-}" in
 prepare) prepare;;
 verify) verify;;
 stop) stop;;
 *) echo "Usage: $0 {prepare|verify|stop}"; exit 2;;
esac
