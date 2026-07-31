#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import requests
import websocket

PORT = int(os.environ.get("CODOXEAR_BROWSER_CDP_PORT", "19533"))
APP_PORT = int(os.environ.get("CODOXEAR_DOCKER_PORT", "19532"))
ORIGIN = f"http://127.0.0.1:{APP_PORT}"
SESSION_ID = "coarse-code-copy-session"
PASSWORD = os.environ.get("CODOXEAR_DOCKER_PASSWORD", "<sandbox-password>")
ART = Path(os.environ["CODOXEAR_BROWSER_ARTIFACT_DIR"])
PROFILE = ART / "chrome-profile"
RESULTS = ART / "browser-results.json"
EXPECTED_FIRST_BLOCK = "printf 'alpha <tag> & value'"
CHROME = os.environ.get("CHROME", "/usr/bin/chromium")


def http_json(url: str, method: str = "GET") -> Any:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Cdp:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=5)
        self.next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        msg_id = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = self.ws.recv()
            msg = json.loads(raw)
            if msg.get("id") != msg_id:
                continue
            if "error" in msg:
                raise RuntimeError(f"CDP {method} failed: {msg['error']}")
            return msg.get("result")

    def close(self) -> None:
        self.ws.close()


def wait_for(predicate, timeout=15.0, interval=0.1):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except Exception as exc:
            last = exc
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting; last={last!r}")


def eval_js(page: Cdp, expression: str, await_promise: bool = True) -> Any:
    res = page.call(
        "Runtime.evaluate",
        {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True,
            "userGesture": True,
        },
    )
    if res.get("exceptionDetails"):
        raise RuntimeError(json.dumps(res["exceptionDetails"], sort_keys=True))
    return res.get("result", {}).get("value")


def navigate(page: Cdp, url: str) -> None:
    page.call("Page.navigate", {"url": url})
    wait_for(lambda: eval_js(page, "document.readyState", False) in ("interactive", "complete"), timeout=10)


def apply_emulation(page: Cdp, scenario: dict[str, Any]) -> None:
    page.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": scenario["width"],
            "height": scenario["height"],
            "deviceScaleFactor": 1,
            "mobile": bool(scenario.get("mobile", False)),
            "screenWidth": scenario["width"],
            "screenHeight": scenario["height"],
        },
    )
    touch_enabled = bool(scenario.get("touch", False))
    touch_params = {"enabled": touch_enabled}
    if touch_enabled:
        touch_params["maxTouchPoints"] = 5
    page.call("Emulation.setTouchEmulationEnabled", touch_params)
    page.call(
        "Emulation.setEmulatedMedia",
        {
            "media": "screen",
            "features": [
                {"name": "pointer", "value": scenario["pointer"]},
                {"name": "hover", "value": scenario["hover"]},
                {"name": "any-pointer", "value": scenario["pointer"]},
                {"name": "any-hover", "value": scenario["hover"]},
            ],
        },
    )


def wait_for_code_button(page: Cdp) -> None:
    def ready():
        return eval_js(
            page,
            "(() => ({buttons: document.querySelectorAll('.code-copy-btn').length, sessions: document.querySelectorAll('.session').length, text: document.body.innerText.slice(0, 500)}))()",
        )

    try:
        wait_for(lambda: (ready()["buttons"] > 0), timeout=8)
        return
    except TimeoutError:
        eval_js(page, "(() => { const s = document.querySelector('.session'); if (s) s.click(); return !!s; })()")
        wait_for(lambda: (ready()["buttons"] > 0), timeout=10)


def page_metrics(page: Cdp, label: str) -> dict[str, Any]:
    js = r"""
(() => {
  const btn = document.querySelector('.code-copy-btn');
  if (!btn) return {label: LABEL, error: 'missing .code-copy-btn', bodyText: document.body.innerText};
  const pre = btn.closest('pre') || document.querySelector('.md pre');
  const code = pre ? pre.querySelector('code') : null;
  const rect = btn.getBoundingClientRect();
  const bcs = getComputedStyle(btn);
  const pcs = pre ? getComputedStyle(pre) : null;
  const de = document.documentElement;
  const body = document.body;
  const maxScrollWidth = Math.max(de.scrollWidth || 0, body ? body.scrollWidth || 0 : 0);
  const clientWidth = de.clientWidth || window.innerWidth;
  return {
    label: LABEL,
    url: location.href,
    title: document.title,
    media: {
      coarse: matchMedia('(pointer: coarse)').matches,
      fine: matchMedia('(pointer: fine)').matches,
      hoverNone: matchMedia('(hover: none)').matches,
      hoverHover: matchMedia('(hover: hover)').matches,
      anyCoarse: matchMedia('(any-pointer: coarse)').matches,
      anyFine: matchMedia('(any-pointer: fine)').matches,
      anyHoverNone: matchMedia('(any-hover: none)').matches,
      anyHoverHover: matchMedia('(any-hover: hover)').matches
    },
    viewport: {innerWidth, innerHeight, devicePixelRatio, visualViewportWidth: visualViewport && visualViewport.width, visualViewportHeight: visualViewport && visualViewport.height},
    buttonCount: document.querySelectorAll('.code-copy-btn').length,
    rect: {width: rect.width, height: rect.height, left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom},
    computed: {width: bcs.width, height: bcs.height, minWidth: bcs.minWidth, minHeight: bcs.minHeight, padding: bcs.padding},
    pre: pre ? {paddingRight: pcs.paddingRight, scrollWidth: pre.scrollWidth, clientWidth: pre.clientWidth, rectWidth: pre.getBoundingClientRect().width} : null,
    overflow: {documentScrollWidth: de.scrollWidth, bodyScrollWidth: body ? body.scrollWidth : null, clientWidth, innerWidth, pageHorizontalOverflow: maxScrollWidth > clientWidth + 1},
    codeText: code ? code.textContent : null,
    bodyTextSample: document.body.innerText.slice(0, 1000)
  };
})()
""".replace("LABEL", json.dumps(label))
    return eval_js(page, js)


def activate_copy_and_read(page: Cdp, metrics: dict[str, Any], touch: bool) -> str:
    x = metrics["rect"]["left"] + metrics["rect"]["width"] / 2
    y = metrics["rect"]["top"] + metrics["rect"]["height"] / 2
    if touch:
        page.call("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y, "radiusX": 4, "radiusY": 4, "force": 1}], "modifiers": 0})
        page.call("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": [], "modifiers": 0})
    else:
        page.call("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
        page.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
    time.sleep(0.25)
    return eval_js(page, "navigator.clipboard.readText()")


def assert_scenario(scenario: dict[str, Any], metrics: dict[str, Any], copied: str) -> list[str]:
    failures: list[str] = []
    label = scenario["label"]
    if metrics.get("error"):
        failures.append(f"{label}: {metrics['error']}")
        return failures
    if scenario["pointer"] == "coarse":
        if not (metrics["media"]["coarse"] and metrics["media"]["hoverNone"]):
            failures.append(f"{label}: did not match coarse pointer + hover none media: {metrics['media']}")
        if metrics["rect"]["width"] < 44 or metrics["rect"]["height"] < 44:
            failures.append(f"{label}: button rect below 44px: {metrics['rect']}")
        for prop in ("width", "height", "minWidth", "minHeight"):
            val = float(str(metrics["computed"][prop]).replace("px", ""))
            if val < 44:
                failures.append(f"{label}: computed {prop} below 44px: {metrics['computed'][prop]}")
        padding = float(str(metrics["pre"]["paddingRight"]).replace("px", "")) if metrics.get("pre") else 0
        if scenario.get("requirePrePadding58") and padding < 58:
            failures.append(f"{label}: pre padding-right below 58px: {padding}")
    else:
        if metrics["media"].get("coarse"):
            failures.append(f"{label}: desktop unexpectedly matched coarse pointer: {metrics['media']}")
        if not (metrics["media"].get("fine") or not metrics["media"].get("coarse")):
            failures.append(f"{label}: desktop did not prove fine/non-coarse pointer: {metrics['media']}")
        if metrics["rect"]["width"] >= 44 or metrics["rect"]["height"] >= 44:
            failures.append(f"{label}: desktop button not compact: {metrics['rect']}")
    if metrics["overflow"]["pageHorizontalOverflow"]:
        failures.append(f"{label}: page-level horizontal overflow: {metrics['overflow']}")
    if metrics.get("codeText") != EXPECTED_FIRST_BLOCK:
        failures.append(f"{label}: first code block text mismatch: {metrics.get('codeText')!r}")
    if copied != EXPECTED_FIRST_BLOCK:
        failures.append(f"{label}: clipboard mismatch: {copied!r}")
    return failures


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    PROFILE.mkdir(parents=True)
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={PORT}",
        "--remote-allow-origins=*",
        f"--user-data-dir={PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=(ART / "chromium-stderr.log").open("w"))
    results: dict[str, Any] = {"chrome_pid": proc.pid, "chrome_cmd": cmd, "origin": ORIGIN, "session_id": SESSION_ID, "scenarios": []}
    page = None
    browser = None
    try:
        version = wait_for(lambda: http_json(f"http://127.0.0.1:{PORT}/json/version"), timeout=10)
        results["browser_version"] = version
        browser = Cdp(version["webSocketDebuggerUrl"])
        permission_error = None
        try:
            browser.call("Browser.grantPermissions", {"origin": ORIGIN, "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"]})
        except Exception as exc:
            permission_error = repr(exc)
        results["permission_error"] = permission_error
        target = requests.put(f"http://127.0.0.1:{PORT}/json/new?about:blank", timeout=5).json()
        page = Cdp(target["webSocketDebuggerUrl"])
        page.call("Page.enable")
        page.call("Runtime.enable")
        navigate(page, ORIGIN + "/")
        login = eval_js(page, f"fetch('/api/login', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{'password': {json.dumps(PASSWORD)}}})}}).then(async r => ({{status:r.status, body: await r.text()}}))")
        results["browser_login"] = {"status": login.get("status"), "body_len": len(login.get("body") or "")}
        scenarios = [
            {"label": "touch-tablet-768x1024", "width": 768, "height": 1024, "mobile": True, "touch": True, "pointer": "coarse", "hover": "none", "requirePrePadding58": True},
            {"label": "touch-phone-390x844", "width": 390, "height": 844, "mobile": True, "touch": True, "pointer": "coarse", "hover": "none", "requirePrePadding58": False},
            {"label": "fine-desktop-1280x800", "width": 1280, "height": 800, "mobile": False, "touch": False, "pointer": "fine", "hover": "hover", "requirePrePadding58": False},
        ]
        failures: list[str] = []
        for scenario in scenarios:
            apply_emulation(page, scenario)
            media_probe = eval_js(page, "({coarse: matchMedia('(pointer: coarse)').matches, fine: matchMedia('(pointer: fine)').matches, hoverNone: matchMedia('(hover: none)').matches, hoverHover: matchMedia('(hover: hover)').matches, innerWidth, innerHeight})")
            navigate(page, ORIGIN + "/#session=" + SESSION_ID)
            wait_for_code_button(page)
            metrics = page_metrics(page, scenario["label"])
            copied = activate_copy_and_read(page, metrics, bool(scenario.get("touch")))
            metrics_after = page_metrics(page, scenario["label"] + "-after-copy")
            scenario_result = {"config": scenario, "pre_navigation_media_probe": media_probe, "metrics": metrics, "copied": copied, "metrics_after_copy": metrics_after}
            scenario_result["failures"] = assert_scenario(scenario, metrics, copied)
            failures.extend(scenario_result["failures"])
            results["scenarios"].append(scenario_result)
        results["failures"] = failures
        RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"results": str(RESULTS), "failures": failures}, indent=2, sort_keys=True))
        return 1 if failures else 0
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        results["chrome_cleanup_returncode"] = proc.returncode
        try:
            if RESULTS.exists():
                current = json.loads(RESULTS.read_text(encoding="utf-8"))
                current["chrome_cleanup_returncode"] = proc.returncode
                RESULTS.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
