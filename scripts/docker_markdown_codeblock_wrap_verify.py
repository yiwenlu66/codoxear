#!/usr/bin/env python3
"""Docker-only real transcript wrap/copy regression; no backend credentials needed.

Runs the working tree read-only. Retains geometry, clipboard round trips and
screenshots in /tmp/codoxear-codeblock-wrap-results.*. Optional environment
CODOXEAR_CODEBLOCK_WRAP_BASELINE_REF compares the exact Git stylesheet from that
revision, not an approximation of its declarations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("CODOXEAR_CODEBLOCK_WRAP_PORT", "19673"))
assert 1 <= PORT <= 65535 and PORT != 8743, "refusing invalid/live port"
assert not (os.environ.get("CODEXEAR_APP_DIR") or os.environ.get("CODEX_WEB_APP_DIR")), "refusing host app-dir"
ARTIFACTS = Path(tempfile.mkdtemp(prefix="codoxear-codeblock-wrap-results."))
TEMP = Path(tempfile.mkdtemp(prefix="codoxear-codeblock-wrap-build."))
CONTAINER = f"codoxear-codeblock-wrap-{os.getpid()}"
SESSION = CONTAINER
IMAGE = "codoxear-codeblock-wrap:working-tree"
BASE = f"http://127.0.0.1:{PORT}"
PASSWORD = "codeblock-wrap-password"
BASELINE_REF = os.environ.get("CODOXEAR_CODEBLOCK_WRAP_BASELINE_REF")
BLOCKS = [
    "    " + "alpha beta gamma delta " * 60 + "\n    explicit second line\n",
    "    " + "UnbrokenToken" * 60 + "\n",
    "\tif value < 3 && ready:\n\t\treturn 'preserve tabs & <tags>'\n    trailing spaces  \n\n    end\n",
]
MARKDOWN = "Inline `inline  code` stays unchanged.\n\n" + "\n".join(
    f"```text\n{block}```\n" for block in BLOCKS
)
# A socket fixture owns only an idle transcript and records every control call.
FIXTURE = r'''
import json, os, socket
from pathlib import Path
home = Path.home()
socks = home / '.local/share/codoxear/socks'
socks.mkdir(parents=True, exist_ok=True)
log = home / 'wrap.jsonl'
text = json.loads(Path('/proof/markdown.json').read_text())
rows = [
 {'type':'session','version':3,'id':'wrap-thread','cwd':'/workspace','timestamp':'2026-10-23T10:00:00Z'},
 {'type':'message','id':'u1','timestamp':'2026-10-23T10:00:00Z','message':{'role':'user','content':[{'type':'text','text':'Show wrapping fixtures.'}]}},
 {'type':'message','id':'a1','parentId':'u1','timestamp':'2026-10-23T10:00:01Z','message':{'role':'assistant','model':'fixture-model','stopReason':'stop','content':[{'type':'text','text':text}]}},
]
log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
sock = socks / 'wrap.sock'
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(sock)); server.listen(16)
meta = {'session_id':'wrap','agent_backend':'pi','owner':'web','broker_pid':os.getpid(),'codex_pid':os.getpid(),'cwd':'/workspace','start_ts':1792749600.0,'log_path':str(log),'control_protocol_version':2,'control_capabilities':{'sync_send':True,'key_write_errors':True}}
sock.with_suffix('.json').write_text(json.dumps(meta))
while True:
 conn, _ = server.accept()
 with conn:
  data = b''
  while not data.endswith(b'\n'):
   chunk = conn.recv(65536)
   if not chunk: break
   data += chunk
  request = json.loads(data)
  with (home / 'control.jsonl').open('a') as out: out.write(json.dumps(request)+'\n')
  response = {'busy':False,'turn_open':False,'queue_len':0,'interrupted_idle':False,'token':None} if request.get('cmd') == 'state' else {'error':'read-only fixture'}
  conn.sendall(json.dumps(response).encode()+b'\n')
'''
PROBE = r'''(() => {
 const nodes = [...document.querySelectorAll('.msg.assistant .md pre')];
 const style = n => {const s=getComputedStyle(n); return {whiteSpace:s.whiteSpace,overflowWrap:s.overflowWrap,wordBreak:s.wordBreak};};
 return {width:innerWidth,coarse:matchMedia('(pointer: coarse)').matches,hoverNone:matchMedia('(hover: none)').matches,
  pageOverflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
  inline:style(document.querySelector('.msg.assistant .md p code')),
  blocks:nodes.map(pre => {
   const code=pre.querySelector('code'), text=code.textContent;
   const firstLine=text.indexOf('\n'), range=document.createRange();
   range.setStart(code.firstChild,0); range.setEnd(code.firstChild,firstLine);
   const rects=[...range.getClientRects()].map(r=>({x:r.x,y:r.y,width:r.width,height:r.height}));
   const indentRange=document.createRange(); indentRange.setStart(code.firstChild,0); indentRange.setEnd(code.firstChild,4);
   return {pre:style(pre),code:style(code),text,clientWidth:pre.clientWidth,scrollWidth:pre.scrollWidth,
    height:pre.getBoundingClientRect().height,rects,indentWidth:indentRange.getBoundingClientRect().width};
  })};
})()'''


def run(args, *, check=True):
    return subprocess.run([str(a) for a in args], check=check, capture_output=True, text=True)


def docker(*args, check=True):
    return run(["sudo", "docker", *args], check=check)


def browser(*args):
    result = run(["agent-browser", "--session", SESSION, *args, "--json"])
    payload = json.loads(result.stdout)
    assert payload.get("success"), payload
    return payload.get("data", {})


def evaluate(js):
    return browser("eval", js).get("result")


def grant_clipboard():
    # agent-browser's daemon supports permissions, but its CLI has no command.
    app_dir = Path(os.environ["XDG_RUNTIME_DIR"]) / "agent-browser" if os.environ.get("XDG_RUNTIME_DIR") else Path.home() / ".agent-browser"
    socket_dir = Path(os.environ.get("AGENT_BROWSER_SOCKET_DIR", app_dir))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_dir / f"{SESSION}.sock"))
        connection.sendall(json.dumps({"id": "clipboard-permission", "action": "permissions",
                                      "permissions": ["clipboard-read", "clipboard-write"], "grant": True}).encode() + b"\n")
        response = json.loads(connection.makefile().readline())
        assert response["success"], response


def save(name, value):
    (ARTIFACTS / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    print(f"artifacts={ARTIFACTS}", flush=True)
    TEMP.chmod(0o755)
    (TEMP / "fixture.py").write_text(FIXTURE)
    (TEMP / "markdown.json").write_text(json.dumps(MARKDOWN))
    old_css = run(["git", "-C", ROOT, "show", f"{BASELINE_REF}:codoxear/static/app.css"]).stdout if BASELINE_REF else None
    build = docker("build", "-f", ROOT / "docker/sandbox.Dockerfile", "-t", IMAGE, ROOT)
    (ARTIFACTS / "docker-build.log").write_text(build.stdout + build.stderr)
    docker("run", "-d", "--name", CONTAINER, "-p", f"127.0.0.1:{PORT}:{PORT}",
           "-e", "HOME=/home/tester", "-e", f"CODEX_WEB_PASSWORD={PASSWORD}",
           "-e", "CODEX_WEB_HOST=0.0.0.0", "-e", f"CODEX_WEB_PORT={PORT}",
           "-e", "PYTHONPATH=/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "--mount", f"type=bind,src={ROOT},dst=/workspace,readonly",
           "--mount", f"type=bind,src={TEMP},dst=/proof,readonly", IMAGE,
           "bash", "-c", "python3 /proof/fixture.py >/tmp/fixture.log 2>&1 & exec python3 -m codoxear.server")
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/api/me", timeout=1)
        except urllib.error.HTTPError as err:
            if err.code == 401:
                break
        except (OSError, TimeoutError):
            pass
        time.sleep(.25)
    else:
        raise AssertionError("Docker server did not become ready")
    unit = docker("exec", CONTAINER, "python3", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                  "tests/test_markdown_codeblock_wrap_css.py", "tests/test_code_block_copy_source.py",
                  "tests/test_app_markdown_extended.py")
    (ARTIFACTS / "unit-tests.txt").write_text(unit.stdout + unit.stderr)
    probes = {}
    for device in ("desktop", "phone"):
        # Separate browser contexts ensure phone really uses coarse pointer, not just width.
        if device == "desktop":
            browser("set", "viewport", "1280", "900")
        else:
            # set device only changes metrics in this agent-browser version.
            # Blink's coarse primary-pointer setting also selects hover:none.
            browser("--args", "--blink-settings=primaryPointerType=2", "set", "device", "iPhone 15 Pro")
        browser("open", BASE)
        browser("fill", "#pw", PASSWORD)
        browser("click", "#loginBtn")
        browser("wait", "#sessions .session")
        if device == "phone":
            browser("click", "#toggleSidebarBtn")
        browser("click", "#sessions .session")
        browser("wait", ".msg.assistant .md pre")
        for family in ("clay", "paper", "slate"):
            evaluate(f"document.documentElement.dataset.theme={json.dumps(family)}; document.getElementById('codoxearThemeLink').href={json.dumps('/static/themes/' + family + '.css')}")
            browser("wait", "200")
            for mode in ("light", "dark"):
                evaluate(f"document.documentElement.dataset.mode={json.dumps(mode)}")
                key = f"{device}-{family}-{mode}"
                # Swap only the served base stylesheet; themes and DOM remain identical.
                before = None
                if old_css is not None:
                    evaluate("window.__wrapOldStyle=document.createElement('style'); __wrapOldStyle.textContent=" + json.dumps(old_css) + "; document.head.append(__wrapOldStyle)")
                    before = evaluate(PROBE)
                    evaluate("__wrapOldStyle.remove()")
                after = evaluate(PROBE)
                assert len(after["blocks"]) == 3, after
                assert after["pageOverflow"] == 0, after
                if before:
                    assert after["inline"] == before["inline"]
                    assert all(b["pre"]["whiteSpace"] == "pre" for b in before["blocks"])
                for index, block in enumerate(after["blocks"]):
                    assert block["text"] == BLOCKS[index], (index, block["text"])
                    assert block["scrollWidth"] <= block["clientWidth"], block
                    assert block["pre"]["whiteSpace"] == block["code"]["whiteSpace"] == "pre-wrap"
                    assert block["pre"]["overflowWrap"] == "anywhere"
                    if index < 2:
                        assert len({round(r["y"], 1) for r in block["rects"]}) > 1, block
                        if before:
                            assert block["height"] > before["blocks"][index]["height"], block
                            assert before["blocks"][index]["scrollWidth"] > before["blocks"][index]["clientWidth"]
                        assert block["indentWidth"] > 20, block
                if device == "phone":
                    assert after["width"] <= 430 and after["coarse"] and after["hoverNone"], after
                probes[key] = {"before": before, "after": after}
                save("geometry.json", probes)
                if family == "clay" and mode == "light":
                    for block_index, label in ((0, "wrapped"), (1, "unbroken-token"), (2, "whitespace")):
                        evaluate(f"document.querySelectorAll('.msg.assistant .md pre')[{block_index}].scrollIntoView({{block:'start'}})")
                        browser("screenshot", str(ARTIFACTS / f"{device}-{label}.png"))
        # Real clipboard, explicitly permitted in the ephemeral automation context.
        grant_clipboard()
        copies = []
        for index, expected in enumerate(BLOCKS):
            # Assign stable test ids without changing text or copy behavior.
            evaluate(f"document.querySelectorAll('.msg.assistant .codeBlockWrap')[{index}].id='wrap-copy-target'")
            evaluate("document.querySelector('#wrap-copy-target pre').classList.add('show-copy')")
            browser("click", "#wrap-copy-target .code-copy-btn")
            browser("fill", "#msg", "")
            browser("focus", "#msg")
            browser("press", "Control+v")
            browser("wait", "150")
            pasted = evaluate("document.querySelector('#msg').value")
            assert evaluate("navigator.clipboard.readText()") == expected
            assert pasted == expected, (device, index, pasted)
            copies.append({"block": index, "clipboardPasteExact": True, "length": len(pasted)})
            browser("fill", "#msg", "")
            evaluate("document.querySelector('#wrap-copy-target').removeAttribute('id')")
        # Raw Markdown copy stays independent of visual line wrapping.
        evaluate("document.querySelector('.msg-shell.assistant .msg-copy-btn').click()")
        browser("focus", "#msg")
        browser("press", "Control+v")
        browser("wait", "150")
        assert evaluate("document.querySelector('#msg').value") == MARKDOWN
        browser("fill", "#msg", "")
        # Drafts are shared by the server; do not carry clipboard proof text into
        # the next device's screenshot via the real draft persistence flow.
        browser("wait", "1000")
        save(f"{device}-clipboard.json", {"codeBlocks": copies, "rawMarkdownExact": True})
        errors = browser("errors")
        save(f"{device}-errors.json", errors)
        assert errors.get("errors", []) == [], errors
        browser("close")
    control = docker("exec", CONTAINER, "cat", "/home/tester/control.jsonl").stdout
    commands = [json.loads(line)["cmd"] for line in control.splitlines()]
    assert all(cmd == "state" for cmd in commands), commands
    save("report.json", {"pass": True, "geometry": probes, "brokerMutations": 0,
                         "clipboard": "exact real Copy code and raw Markdown -> clipboard -> paste round trips on both devices"})
    print("PASS: all six theme/mode combinations at desktop and coarse-pointer phone; exact clipboard round trips", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        logs = docker("logs", CONTAINER, check=False)
        (ARTIFACTS / "server.log").write_text(logs.stdout + logs.stderr)
        run(["agent-browser", "--session", SESSION, "close"], check=False)
        docker("rm", "-f", CONTAINER, check=False)
        shutil.rmtree(TEMP)
        print(f"artifacts={ARTIFACTS}", flush=True)
