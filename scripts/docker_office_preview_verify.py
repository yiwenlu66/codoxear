#!/usr/bin/env python3
"""Drive scripts/docker_office_preview_verify.sh through HTTP and a real browser.

Assumes the container started by the .sh orchestrator: a Codoxear server on
127.0.0.1:<port>, one Pi session whose cwd is the generated Office fixture at
/workspace/demo, and LibreOffice + poppler-utils inside the container.

Endpoint contract:
  session read payload: GET /api/sessions/<id>/file/read?path=<rel>
      -> kind "office", preview_url, and an original-download URL
  session preview:      GET /api/sessions/<id>/file/office_preview?path=<rel>
  global preview:       GET /api/files/office_preview?path=<abs>
  global read payload:  POST /api/files/read {path}
  original download:    GET /api/sessions/<id>/file/download?path=<rel>
"""
import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--password", required=True)
ap.add_argument("--container", required=True)
ap.add_argument("--artifacts", required=True)
ap.add_argument("--commit", required=True)
ap.add_argument("--browser-session", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--helper-cmd", default="python3 -m codoxear.dependency_doctor",
                help="in-container provisioning helper command")
args = ap.parse_args()

ART = Path(args.artifacts)
BASE = f"http://127.0.0.1:{args.port}"
DEMO = "/workspace/demo"
T0 = time.time()

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
anon = urllib.request.build_opener()  # no cookies: auth-negative checks

report: dict = {"commit": args.commit, "base_url": BASE, "started_at": T0, "checks": {}, "observed": {}}


def check(name: str, ok: bool, evidence) -> None:
    report["checks"][name] = bool(ok)
    report["observed"][name] = evidence
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)


def req(op, method: str, path: str, *, body=None, headers=None, raw=False):
    url = path if path.startswith("http") else BASE + path
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with op.open(request) as response:
            payload = response.read()
            status = response.status
            rhdrs = dict(response.headers)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        status = exc.code
        rhdrs = dict(exc.headers)
    if raw:
        return status, rhdrs, payload
    try:
        return status, rhdrs, json.loads(payload.decode("utf-8"))
    except Exception:
        return status, rhdrs, {"_raw": payload[:400].decode("utf-8", "replace")}


def api(method, path, **kw):
    return req(opener, method, path, **kw)


def api_anon(method, path, **kw):
    return req(anon, method, path, **kw)


def docker(*cmd, check_rc=True, capture=True) -> subprocess.CompletedProcess:
    full = ["sudo", "docker", "exec", args.container, *cmd]
    proc = subprocess.run(full, capture_output=capture, text=True, timeout=900)
    if check_rc and proc.returncode != 0:
        raise RuntimeError(f"docker exec failed: {' '.join(full)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def docker_pdf_text(pdf_host_path: Path, stem: str) -> dict:
    """Copy a downloaded preview PDF into the container and extract text/info."""
    inner = f"/tmp/office-verify-pdfs/{stem}.pdf"
    docker("mkdir", "-p", "/tmp/office-verify-pdfs")
    subprocess.run(["sudo", "docker", "cp", str(pdf_host_path), f"{args.container}:{inner}"],
                   check=True, capture_output=True, text=True)
    text = docker("pdftotext", inner, "-", check_rc=False).stdout
    info = docker("pdfinfo", "-f", "1", "-l", "100000", inner, check_rc=False).stdout
    pages = 0
    max_dim = 0.0
    for line in info.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*pts", line)
        if line.startswith("Page") and match:
            max_dim = max(max_dim, float(match.group(1)), float(match.group(2)))
    out = {"text": text, "pages": pages, "max_page_dimension_pts": max_dim}
    (ART / f"{stem}.txt").write_text(text, encoding="utf-8")
    return out


def browser(*cli) -> dict:
    env = dict(os.environ, AGENT_BROWSER_SESSION=args.browser_session)
    proc = subprocess.run(["agent-browser", *cli], capture_output=True, text=True, env=env, timeout=300)
    stem = f"browser-{len(list(ART.glob('browser-*.txt'))) + 1:02d}-{'-'.join(cli[:1])}"
    (ART / f"{stem}.txt").write_text(
        f"$ agent-browser {' '.join(cli)}\nexit={proc.returncode}\n{proc.stdout}\n{proc.stderr}",
        encoding="utf-8")
    parsed: dict
    try:
        parsed = json.loads(proc.stdout)
    except Exception:
        marker = proc.stdout.find("{")
        try:
            parsed = json.loads(proc.stdout[marker:]) if marker >= 0 else {}
        except Exception:
            parsed = {}
    if not parsed:
        # Drive commands (open/click/fill/wait) answer plain text, not JSON.
        parsed = {"success": proc.returncode == 0, "raw": (proc.stdout + proc.stderr).strip()[-500:]}
    parsed["_rc"] = proc.returncode
    return parsed


def browser_eval(expr: str):
    out = browser("eval", expr, "--json")
    return out.get("data", {}).get("result")


def browser_ok(*cli) -> bool:
    out = browser(*cli)
    return out.get("_rc") == 0 and out.get("success") is not False


# --- login + session discovery ---------------------------------------------
status, _, login_body = api("POST", "/api/login", body={"password": args.password})
if status != 200:
    raise SystemExit(f"login failed: {status} {login_body}")

session_id = None
for _ in range(120):
    status, _, payload = api("GET", "/api/sessions")
    if status == 200:
        for row in payload.get("sessions", []):
            if isinstance(row, dict) and row.get("agent_backend") == "pi" and row.get("session_id"):
                session_id = row["session_id"]
                break
    if session_id:
        break
    time.sleep(1)
if not session_id:
    raise SystemExit("no Pi session appeared in /api/sessions")
(ART / "session-id.txt").write_text(session_id + "\n", encoding="utf-8")

manifest = json.loads(docker("cat", f"{DEMO}/manifest.json").stdout)
(ART / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
MARKERS = manifest["markers"]

OFFICE_RELS = ["report.docx", "report.doc", "data.xlsx", "data.xls", "deck.pptx", "deck.ppt"]
EXCEL_RELS = ["data.xlsx", "data.xls"]
EXCEL_MARKER_KEYS = ["xlsx_main", "xlsx_tail", "xlsx_hidden", "xlsx_veryhidden", "xlsx_print_in", "xlsx_print_out"]
# deck.ppt is reserved for the browser "Preparing..." observation: it must stay
# unconverted until the browser opens it.
API_PREVIEW_RELS = ["report.docx", "report.doc", "data.xlsx", "data.xls", "deck.pptx"]


def preview_cache_files() -> list[str]:
    proc = docker("bash", "-c",
                  'find "$HOME/.local/share/codoxear" /tmp -maxdepth 6 -type f -name "*.pdf" 2>/dev/null | sort',
                  check_rc=False)
    return [line for line in proc.stdout.splitlines() if line.strip() and "/office-verify-pdfs/" not in line]


def read_payload(rel: str):
    return api("GET", f"/api/sessions/{session_id}/file/read?" + urllib.parse.urlencode({"path": rel}))


def find_download_url(payload: dict, rel: str) -> str | None:
    for key, value in payload.items():
        if isinstance(value, str) and key.endswith("_url") and "/file/download" in value:
            return value
    for key, value in payload.items():
        if isinstance(value, str) and key.endswith("_url") and "/file/blob" in value:
            return value
    return None


try:
    # --- 1. Read payloads: kind office, lazy (no conversion on read) ------------
    baseline_pdfs = preview_cache_files()
    payloads: dict[str, dict] = {}
    for rel in OFFICE_RELS:
        status, _, payload = read_payload(rel)
        payloads[rel] = {"status": status, "payload": payload}
    lazy_pdfs = preview_cache_files()
    check("1_read_kind_office_and_lazy",
          all(p["status"] == 200 and p["payload"].get("kind") == "office" and p["payload"].get("preview_url")
              for p in payloads.values()) and lazy_pdfs == baseline_pdfs,
          {"payloads": {k: {"status": v["status"], "kind": v["payload"].get("kind"),
                            "preview_url": v["payload"].get("preview_url")} for k, v in payloads.items()},
           "pdf_files_before": baseline_pdfs, "pdf_files_after_reads": lazy_pdfs})

    # --- 2. Auth, preview PDF, Range 206, digest-preserved download, markers ----
    per_file: dict[str, dict] = {}
    for rel in API_PREVIEW_RELS:
        entry: dict = {}
        payload = payloads[rel]["payload"]
        preview_url = payload.get("preview_url") or ""
        download_url = find_download_url(payload, rel)
        entry["preview_url"] = preview_url
        entry["download_url"] = download_url
        if not preview_url:
            entry["skipped"] = "read payload carried no preview_url"
            per_file[rel] = entry
            continue

        status_anon, _, _ = api_anon("GET", preview_url, raw=True)
        entry["anon_status"] = status_anon

        status, hdrs, pdf = api("GET", preview_url, raw=True)
        entry["preview_status"] = status
        entry["preview_content_type"] = hdrs.get("Content-Type") or hdrs.get("content-type")
        stem = rel.replace(".", "-")
        (ART / f"preview-{stem}.pdf").write_bytes(pdf)
        entry["preview_sha256"] = hashlib.sha256(pdf).hexdigest()

        status_r, hdrs_r, chunk = api("GET", preview_url, headers={"Range": "bytes=0-1023"}, raw=True)
        entry["range_status"] = status_r
        entry["range_content_range"] = hdrs_r.get("Content-Range") or hdrs_r.get("content-range")
        entry["range_bytes"] = len(chunk)

        digest_ok = None
        if download_url:
            status_d, _, original = api("GET", download_url, raw=True)
            entry["download_status"] = status_d
            digest_ok = (status_d == 200
                         and hashlib.sha256(original).hexdigest() == manifest["files"][rel]["sha256"])
            entry["download_digest_matches_source"] = digest_ok

        # Cache identity: a second fetch returns byte-identical output.
        status2, _, pdf2 = api("GET", preview_url, raw=True)
        entry["second_fetch_identical"] = status2 == 200 and pdf2 == pdf

        info = docker_pdf_text(ART / f"preview-{stem}.pdf", f"preview-{stem}")
        expected_marker_keys = EXCEL_MARKER_KEYS if rel in EXCEL_RELS else (
            ["docx"] if rel.startswith("report") else ["pptx"])
        marker_result = {key: all(m in info["text"] for m in MARKERS[key]) for key in expected_marker_keys}
        entry["markers"] = marker_result
        entry["pages"] = info["pages"]
        entry["max_page_dimension_pts"] = info["max_page_dimension_pts"]

        ok = (status_anon == 401
              and status == 200 and "pdf" in str(entry["preview_content_type"]).lower()
              and status_r == 206 and entry["range_bytes"] == 1024
              and digest_ok is True
              and entry["second_fetch_identical"]
              and all(marker_result.values()))
        if rel in EXCEL_RELS:
            # Ordinary readable pagination: several pages, none enormous.
            ok = ok and info["pages"] >= 3 and 0 < info["max_page_dimension_pts"] <= 1300
        per_file[rel] = entry

    check("2_session_preview_auth_pdf_range_download_markers",
          all(e.get("anon_status") == 401 and e.get("preview_status") == 200
              and "pdf" in str(e.get("preview_content_type")).lower()
              and e.get("range_status") == 206 and e.get("range_bytes") == 1024
              and e.get("download_digest_matches_source") is True
              and e.get("second_fetch_identical") and all(e.get("markers", {}).values())
              for e in per_file.values()),
          per_file)
    check("2b_excel_all_sheets_and_pagination",
          all(per_file[rel].get("pages", 0) >= 3 and 0 < per_file[rel].get("max_page_dimension_pts", 0) <= 1300
              for rel in EXCEL_RELS),
          {rel: {"pages": per_file[rel].get("pages"), "max_dim": per_file[rel].get("max_page_dimension_pts"),
                 "markers": per_file[rel].get("markers"),
                 "baseline_raw_lo": {k: manifest["baseline_pdf"][rel]["markers"][k]
                                     for k in ("xlsx_hidden", "xlsx_veryhidden", "xlsx_print_out")}}
           for rel in EXCEL_RELS})

    # --- 3. Global/absolute endpoint ---------------------------------------------
    g_status, _, g_payload = api("POST", "/api/files/read", body={"path": f"{DEMO}/report.docx"})
    g_preview = (g_payload.get("preview_url") or "") if isinstance(g_payload, dict) else ""
    ga_status, _, _ = api_anon("GET", g_preview, raw=True) if g_preview else (None, None, None)
    gp_status, gp_hdrs, gp_pdf = api("GET", g_preview, raw=True) if g_preview else (None, None, b"")
    check("3_global_office_preview",
          g_status == 200 and g_payload.get("kind") == "office" and g_preview.startswith("/api/files/office_preview")
          and ga_status == 401 and gp_status == 200 and "pdf" in str(gp_hdrs.get("Content-Type")).lower()
          and hashlib.sha256(gp_pdf).hexdigest() == per_file["report.docx"].get("preview_sha256"),
          {"read_status": g_status, "kind": g_payload.get("kind") if isinstance(g_payload, dict) else None,
           "preview_url": g_preview, "anon_status": ga_status, "preview_status": gp_status})

    # --- 4. Two distinct simultaneous conversions --------------------------------
    docker("bash", "-c", f"cp {DEMO}/report.doc {DEMO}/simultaneous-a.doc && cp {DEMO}/deck.ppt {DEMO}/simultaneous-b.ppt")
    sim_results: dict[str, dict] = {}


    def convert_one(rel: str) -> None:
        url = f"/api/sessions/{session_id}/file/office_preview?" + urllib.parse.urlencode({"path": rel})
        started = time.time()
        status, hdrs, pdf = api("GET", url, raw=True)
        sim_results[rel] = {"status": status, "content_type": hdrs.get("Content-Type"),
                            "elapsed_s": round(time.time() - started, 2), "bytes": len(pdf),
                            "pdf": pdf}


    simultaneous_rels = ("simultaneous-a.doc", "simultaneous-b.ppt")
    threads = [threading.Thread(target=convert_one, args=(rel,))
               for rel in simultaneous_rels]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sim_observed = {}
    # A worker-thread exception produces no entry, so both expected keys must
    # be present before the per-result assertions can hold.
    sim_ok = set(sim_results) == set(simultaneous_rels)
    for rel, res in sim_results.items():
        key = "docx" if rel.endswith(".doc") else "pptx"
        stem = f"sim-{rel.replace('.', '-')}"
        pdf_path = ART / f"preview-{stem}.pdf"
        pdf_path.write_bytes(res["pdf"])
        marker_ok = all(m in docker_pdf_text(pdf_path, f"preview-{stem}")["text"] for m in MARKERS[key])
        sim_observed[rel] = {k: v for k, v in res.items() if k != "pdf"} | {"markers": marker_ok}
        sim_ok = sim_ok and res["status"] == 200 and "pdf" in str(res["content_type"]).lower() and marker_ok
    check("4_simultaneous_distinct_conversions", sim_ok, sim_observed)

    # --- 5. Source rewrite invalidates the cached preview ------------------------
    docker("bash", "-c", f"cp {DEMO}/data.xlsx {DEMO}/rewrite.xlsx")
    rw_url = f"/api/sessions/{session_id}/file/office_preview?" + urllib.parse.urlencode({"path": "rewrite.xlsx"})
    rw_status1, _, rw_pdf1 = api("GET", rw_url, raw=True)
    (ART / "preview-rewrite-before.pdf").write_bytes(rw_pdf1)
    docker("python3", "-c", (
        "import zipfile, shutil\n"
        "src = '/workspace/demo/rewrite.xlsx'\n"
        "tmp = src + '.new'\n"
        "zin = zipfile.ZipFile(src)\n"
        "zout = zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED)\n"
        "for item in zin.infolist():\n"
        "    data = zin.read(item.filename)\n"
        "    if item.filename == 'xl/worksheets/sheet1.xml':\n"
        "        data = data.replace(b'XLSXMAIN7F41', b'XLSXMAIN8F52')\n"
        "    zout.writestr(item, data)\n"
        "zout.close(); zin.close()\n"
        "shutil.move(tmp, src)\n"))
    rw_status2, _, rw_pdf2 = api("GET", rw_url, raw=True)
    (ART / "preview-rewrite-after.pdf").write_bytes(rw_pdf2)
    rw_after = docker_pdf_text(ART / "preview-rewrite-after.pdf", "preview-rewrite-after")
    rw_dl_status, _, rw_dl = api("GET", f"/api/sessions/{session_id}/file/download?" +
                                 urllib.parse.urlencode({"path": "rewrite.xlsx"}), raw=True)
    new_source_digest = hashlib.sha256(rw_dl).hexdigest()
    check("5_source_rewrite_invalidates_preview",
          rw_status1 == 200 and rw_status2 == 200
          and rw_pdf1 != rw_pdf2
          and "XLSXMAIN8F52" in rw_after["text"] and "XLSXMAIN7F41" not in rw_after["text"]
          and new_source_digest != manifest["files"]["data.xlsx"]["sha256"],
          {"first_status": rw_status1, "second_status": rw_status2,
           "bytes_differ": rw_pdf1 != rw_pdf2,
           "new_marker_present": "XLSXMAIN8F52" in rw_after["text"],
           "old_marker_absent": "XLSXMAIN7F41" not in rw_after["text"],
           "new_source_digest": new_source_digest})

    # --- 6. Failure paths: corrupt + encrypted ------------------------------------
    corrupt_url = payloads.get("broken.docx", {}).get("payload", {}).get("preview_url")
    if not corrupt_url:
        cb_status, _, corrupt_payload = read_payload("broken.docx")
        corrupt_url = corrupt_payload.get("preview_url")
        payloads["broken.docx"] = {"status": cb_status, "payload": corrupt_payload}
    c_status, c_error = None, None
    if corrupt_url:
        c_status, _, c_body = api("GET", corrupt_url, raw=True)
        try:
            c_error = json.loads(c_body.decode("utf-8")).get("error")
        except Exception:
            c_error = None
    check("6a_corrupt_file_fails_explicitly",
          c_status in (400, 409, 422, 500) and bool(c_error),
          {"preview_url": corrupt_url, "status": c_status, "error": c_error})

    encrypted_observed: dict = {}
    encrypted_ok = False
    if "secret.xlsx" in manifest["files"]:
        _, _, enc_payload = read_payload("secret.xlsx")
        enc_url = enc_payload.get("preview_url")
        e_status, e_error = None, None
        if enc_url:
            e_status, _, e_body = api("GET", enc_url, raw=True)
            try:
                e_error = json.loads(e_body.decode("utf-8")).get("error")
            except Exception:
                e_error = None
        encrypted_observed = {"preview_url": enc_url, "status": e_status, "error": e_error}
        encrypted_ok = e_status in (400, 409, 422, 500) and bool(e_error)
    else:
        encrypted_observed = {"error": "secret.xlsx missing from fixture manifest; the test image installs msoffcrypto-tool"}
    check("6b_encrypted_file_fails_explicitly", encrypted_ok, encrypted_observed)

    # --- 7. Browser: authenticated viewer opens Office previews --------------------
    browser("open", BASE + "/")
    browser("errors", "--clear", "--json")
    browser("console", "--clear", "--json")
    browser("fill", "#pw", args.password)
    browser("click", "#loginBtn")
    browser("wait", "3500")
    card = f'#sessions .session[data-session-id="{session_id}"]'
    if not browser_ok("click", card):
        check("7_browser_session_card", False, {"selector": card})
    browser("wait", "1600")
    browser("fill", "#msg", "preview deck.ppt and data.xlsx please")
    browser("click", "#sendBtn")
    browser("wait", "3000")
    links = browser_eval("Array.from(document.querySelectorAll('a.inlineFileLink')).map((a) => a.dataset.filePath)")

    # deck.ppt has never been converted: observe the Preparing state, then render.
    browser("click", 'a.inlineFileLink[data-file-path$="/deck.ppt"]')
    preparing_seen = False
    rendered = False
    status_text = ""
    for _ in range(120):  # up to 60s
        state = browser_eval("""(() => {
          const vis = (n) => !!n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0;
          const viewer = document.querySelector("#fileViewer");
          const status = document.querySelector("#fileStatus");
          const canvases = Array.from(document.querySelectorAll("#fileViewer canvas")).filter(vis);
          return {open: vis(viewer), status: status ? status.textContent : "",
                  preparing: /preparing/i.test(viewer ? viewer.textContent : ""),
                  canvases: canvases.length};
        })()""")
        if isinstance(state, dict):
            preparing_seen = preparing_seen or bool(state.get("preparing"))
            status_text = state.get("status") or status_text
            if state.get("canvases"):
                rendered = True
                break
        time.sleep(0.5)
    browser("screenshot", "--full", str(ART / "office-browser-preview.png"))
    ppt_download_present = browser_eval("""(() => {
      const viewer = document.querySelector("#fileViewer");
      if (!viewer) return false;
      const links = Array.from(viewer.querySelectorAll("a[download], a[href*='/file/download'], a[href*='/file/blob']"));
      const buttons = Array.from(viewer.querySelectorAll("button")).filter((b) => /download/i.test(b.textContent || b.title || b.getAttribute("aria-label") || ""));
      return links.length + buttons.length > 0;
    })()""")
    check("7_browser_office_pdf_render",
          rendered and ppt_download_present is True,
          {"links": links, "preparing_seen": preparing_seen, "rendered": rendered,
           "status_text": status_text, "download_present": ppt_download_present})
    report["observed"]["7_browser_preparing_state_seen"] = preparing_seen

    # Browser error path: corrupt file keeps the original download reachable.
    browser("click", "#fileCloseBtn")
    browser("wait", "800")
    browser("click", "#fileBtn")
    browser("wait", "600")
    browser("fill", "#filePickerInput", "broken.docx")
    browser("wait", "900")
    broken_option = browser_eval("""(() => {
      const item = Array.from(document.querySelectorAll("#filePickerMenu .fileMenuItem"))
        .find((n) => (n.innerText || "").includes("broken.docx"));
      return item ? item.id : null;
    })()""")
    if broken_option:
        browser("click", f"#{broken_option}")
    else:
        browser("click", "#filePickerOption-1")
    browser("wait", "6000")
    error_view = browser_eval("""(() => {
      const vis = (n) => !!n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0;
      const viewer = document.querySelector("#fileViewer");
      const text = viewer ? viewer.textContent : "";
      const fallback = viewer ? viewer.querySelector(".fileDownloadFallback a, a[href*='/file/download']") : null;
      return {open: vis(viewer), text: text.slice(0, 400),
              canvases: viewer ? viewer.querySelectorAll("canvas").length : 0,
              fallbackHref: fallback ? fallback.href : null};
    })()""")
    browser("screenshot", "--full", str(ART / "office-browser-error.png"))
    check("7b_browser_error_keeps_download",
          isinstance(error_view, dict) and error_view.get("open") is True
          and error_view.get("canvases") == 0 and bool(error_view.get("fallbackHref")),
          error_view)

    errors_payload = browser("errors", "--json")
    console_payload = browser("console", "--json")
    page_errors = errors_payload.get("data", {}).get("errors", [])
    console_msgs = console_payload.get("data", {}).get("messages", [])
    fatal_console = [m for m in console_msgs if isinstance(m, dict)
                     and re.search(r"\b(?:ReferenceError|TypeError|SyntaxError)\b|\bCodoxear\b.*\bfailed to load\b",
                                   str(m.get("text") or ""), re.I)]
    check("7c_browser_no_fatal_errors", page_errors == [] and not fatal_console,
          {"page_errors": page_errors, "fatal_console": fatal_console})

    # --- 8. Provisioning helper (in-container only) ----------------------

    helper_cmd = args.helper_cmd
    helper_observed: dict = {"cmd": helper_cmd}
    helper_ok = True

    def helper_run(*argv, path_override: str = ""):
        env = f"env PATH={path_override}" if path_override else ""
        proc = docker("bash", "-c", f"{env} {helper_cmd} {' '.join(argv)}",
                      check_rc=False)
        return {"argv": list(argv), "rc": proc.returncode,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-1000:],
                "path_override": path_override or None}

    help_res = helper_run("--help")
    helper_observed["help"] = help_res
    helper_ok = helper_ok and help_res["rc"] == 0

    json_res = helper_run("--json")
    helper_observed["check_json"] = json_res
    try:
        parsed = json.loads(json_res["stdout"])
        groups = parsed.get("checks", {})
        helper_observed["check_json_groups"] = {g: v.get("status") for g, v in groups.items()}
        # The verify image honestly lacks ffmpeg/tmux; the doctor must report
        # rc 1 with office/fonts/git ok and the real gaps named.
        json_ok = (json_res["rc"] in (0, 1) and set(groups) == {"office", "fonts", "media", "git", "tmux"}
                   and groups.get("office", {}).get("status") == "ok")
    except Exception:
        json_ok = False
    helper_ok = helper_ok and json_ok

    probe_res = helper_run("--probe-office", "--office", "--json")
    helper_observed["probe_office"] = probe_res
    try:
        probe_ok = probe_res["rc"] == 0 and json.loads(probe_res["stdout"]).get("checks", {}).get("office", {}).get("status") == "ok"
    except Exception:
        probe_ok = False
    helper_ok = helper_ok and probe_ok

    dry_res = helper_run("--install", "--dry-run", "--office", "--fonts", "--json")
    helper_observed["dry_run"] = dry_res
    which_after = docker("bash", "-c", "which soffice", check_rc=False).stdout.strip()
    helper_observed["soffice_after_dry_run"] = which_after
    helper_ok = helper_ok and dry_res["rc"] == 0 and which_after == "/usr/bin/soffice"

    # Missing capability: the doctor deliberately resolves absolute native
    # locations (not just PATH), so absence must be real. A one-shot throwaway
    # container from the same image renames soffice/LibreOffice away; the
    # long-lived verification container is untouched.
    missing_proc = subprocess.run(
        ["sudo", "docker", "run", "--rm", "--user", "root", args.image, "bash", "-c",
         "mv /usr/bin/soffice /usr/bin/soffice.moved; "
         "mv /usr/lib/libreoffice /usr/lib/libreoffice.moved; "
         f"{helper_cmd} --json"],
        capture_output=True, text=True, timeout=300)
    missing_res = {"rc": missing_proc.returncode, "stdout": missing_proc.stdout[-2000:],
                   "stderr": missing_proc.stderr[-1000:], "via": f"throwaway {args.image} without soffice"}
    helper_observed["check_missing_libreoffice"] = missing_res
    try:
        missing_groups = json.loads(missing_res["stdout"]).get("checks", {})
        office_status = missing_groups.get("office", {}).get("status")
    except Exception:
        office_status = None
    helper_observed["missing_office_status"] = office_status
    helper_ok = helper_ok and missing_res["rc"] == 1 and office_status == "missing"

    # Cache maintenance after font/LibreOffice changes: clear the generated
    # PDFs (JSON reports the exact count), then prove the next preview
    # performs a fresh conversion with the cache empty.
    clear_res = helper_run("--clear-office-cache", "--json")
    helper_observed["clear_office_cache"] = clear_res
    try:
        cleared = json.loads(clear_res["stdout"]).get("office_cache", {}).get("removed")
    except Exception:
        cleared = None
    helper_observed["clear_office_cache_removed"] = cleared
    rc_status, rc_hdrs, rc_pdf = api("GET", per_file["report.docx"]["preview_url"], raw=True)
    helper_observed["reconvert_after_clear"] = {"status": rc_status, "bytes": len(rc_pdf)}
    helper_ok = (helper_ok and clear_res["rc"] == 0 and isinstance(cleared, int) and cleared >= 1
                 and rc_status == 200
                 and "pdf" in str(rc_hdrs.get("Content-Type")).lower() and rc_pdf.startswith(b"%PDF"))
    check("8_provisioning_helper", helper_ok, helper_observed)

except Exception as exc:
    import traceback
    report['driver_error'] = ''.join(traceback.format_exception(exc))[-3000:]
    print('DRIVER ERROR:', exc, flush=True)

# --- report --------------------------------------------------------------------
report["finished_at"] = time.time()
report["elapsed_s"] = round(report["finished_at"] - T0, 1)
report["pass"] = "driver_error" not in report and all(report["checks"].values())
(ART / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"pass": report["pass"], "checks": report["checks"]}, indent=2))
raise SystemExit(0 if report["pass"] else 1)
