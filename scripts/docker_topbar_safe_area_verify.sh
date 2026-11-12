#!/usr/bin/env bash
# Docker-isolated real-browser regression for the iPhone header safe-area geometry.
#
# The container runs the CURRENT working tree (bundle rebuilt into the build
# context; the checkout is never touched) on a loopback port that is never the
# live 8743, and Chromium is driven over CDP so the renderer's top safe-area inset
# is real (`Emulation.setSafeAreaInsetsOverride`) - the only way to reproduce the
# reported symptom, where a fixed header folded the inset into its own padding,
# collapsed its content box, and clipped the 32px controls against overflow.
#
# Every assertion is a property of the current tree, so the script stays valid
# after the change is committed:
#   * with no inset the header is the ordinary 56px band (phone and desktop);
#   * with a 47px notch inset the band grows by exactly that much, keeps its 40px
#     content box, and every control is fully visible, clear of the status bar and
#     hit-testable at both edges;
#   * an over-subscribed column (maximal tall composer) does not squeeze the band
#     back to the clipped geometry, and the composer's budget gives back the
#     height the header takes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_TOPBAR_VERIFY_PORT:-19676}"
password="${CODOXEAR_TOPBAR_VERIFY_PASSWORD:-topbar-safearea-password}"
image="codoxear-topbar-safearea:working-tree"
container="codoxear-topbar-safearea-${port}"
root="$(mktemp -d /tmp/codoxear-topbar-safearea-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-topbar-safearea-results.XXXXXX)"
home_dir="$root/home"
node_modules="${CODOXEAR_AGENT_BROWSER_NODE_MODULES:-$(npm root -g)/agent-browser/node_modules}"
readonly repo_root port password image container root artifacts home_dir node_modules
readonly -a docker=(sudo docker)

case "${1:-}" in
  -h|--help|help)
    echo "Usage: scripts/docker_topbar_safe_area_verify.sh"
    echo "Results land in a new /tmp/codoxear-topbar-safearea-results.* directory."
    exit 0
    ;;
esac

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override" >&2
  exit 2
fi

fail() {
  echo "FAIL: $*" >&2
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  echo "artifacts=$artifacts" >&2
  exit 1
}

cleanup() {
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT

# ---- Build context from the working tree ------------------------------------
build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" ls-files -z --cached --others --exclude-standard | python3 -c '
import sys
skip = (b".memory/", b"review/")
parts = sys.stdin.buffer.read().split(b"\0")
keep = [p for p in parts if p and not p.startswith(skip)]
sys.stdout.buffer.write(b"\0".join(keep) + b"\0")
' > "$root/context-files.nul"
tar --null --no-recursion -C "$repo_root" -T "$root/context-files.nul" -cf "$root/context.tar"
tar -C "$build_context" -xf "$root/context.tar"
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"

mkdir -p "$home_dir/.local/share"
sudo chown -R 1000:1000 "$home_dir"
"${docker[@]}" build --file "$build_context/docker/sandbox.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed"
"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach \
  --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" \
  --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$build_context,dst=/workspace,readonly" \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" \
  bash -c 'exec python3 -m codoxear.server' \
  > "$artifacts/container-id.txt" || fail "container start failed"

for _ in $(seq 1 120); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
  [[ "$code" == "401" ]] && break
  sleep 0.25
done
[[ "${code:-}" == "401" ]] || fail "server readiness failed"

# ---- The real-browser geometry probe ---------------------------------------
cat > "$root/probe.js" <<'JS'
const pw = require(process.env.PW_PATH);

const MEASURE = () => {
  const topbar = document.querySelector(".topbar");
  const rect = topbar.getBoundingClientRect();
  const style = getComputedStyle(topbar);
  const borderTop = parseFloat(style.borderTopWidth) || 0;
  const borderBottom = parseFloat(style.borderBottomWidth) || 0;
  // overflow: hidden clips at the padding box; 1px absorbs the border seam.
  const clipTop = rect.top + borderTop - 1;
  const clipBottom = rect.bottom - borderBottom + 1;
  const controls = Array.from(topbar.querySelectorAll("button"))
    .map((button) => {
      const box = button.getBoundingClientRect();
      const cx = box.left + box.width / 2;
      const hitTop = document.elementFromPoint(cx, box.top + 2);
      const hitBottom = document.elementFromPoint(cx, box.bottom - 2);
      return {
        id: button.id,
        rendered: box.height > 0,
        top: box.top,
        bottom: box.bottom,
        visibleHeight: Math.max(0, Math.min(box.bottom, clipBottom) - Math.max(box.top, clipTop)),
        topEdgeHit: Boolean(hitTop) && (hitTop === button || button.contains(hitTop)),
        bottomEdgeHit: Boolean(hitBottom) && (hitBottom === button || button.contains(hitBottom)),
      };
    })
    .filter((control) => control.rendered);
  const chatHeader = document.querySelector(".chatHeader");
  return {
    topbarHeight: rect.height,
    topbarBottom: rect.bottom,
    paddingTop: parseFloat(style.paddingTop),
    paddingBottom: parseFloat(style.paddingBottom),
    contentHeight: Math.max(0, rect.height - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom)),
    controls,
    chatHeaderTop: chatHeader ? chatHeader.getBoundingClientRect().top : null,
  };
};

(async () => {
  const browser = await pw.chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 393, height: 852 },
    deviceScaleFactor: 3,
    isMobile: true,
    hasTouch: true,
    userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
  });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
  const setInsets = (top, bottom = 34) =>
    cdp.send("Emulation.setSafeAreaInsetsOverride", { insets: { top, bottom, left: 0, right: 0 } });

  const out = { errors: [] };
  page.on("pageerror", (error) => out.errors.push(String(error.message || error)));

  const measure = async () => {
    // The renderer's own view of the inset, so a measurement can never be a
    // stale layout read: a probe box sized by env() must report the inset.
    const insetSeen = await page.evaluate(() => {
      const probe = document.createElement("div");
      probe.style.cssText = "position:absolute;top:0;left:0;width:1px;height:env(safe-area-inset-top,0px)";
      document.body.appendChild(probe);
      const height = probe.getBoundingClientRect().height;
      probe.remove();
      return height;
    });
    await page.waitForTimeout(200);
    return { ...(await page.evaluate(MEASURE)), insetSeen };
  };

  // A dragged-to-maximum tall composer, measured through the real height-mode
  // control (its click is what clears autoGrow's inline textarea height), then
  // returned to normal mode.
  const measureTall = () => page.evaluate(() => {
    const composer = document.querySelector(".composer");
    const expand = document.querySelector("#expandBtn");
    const expanderDone = () => {
      composer.style.removeProperty("--composer-tall-h");
      for (let step = 0; step < 3 && composer.dataset.height !== "normal"; step += 1) expand.click();
    };
    expand.click();
    composer.style.setProperty("--composer-tall-h", `${window.innerHeight}px`);
    const textarea = document.querySelector("#msg");
    const box = composer.getBoundingClientRect();
    const sendBox = document.querySelector("#sendBtn").getBoundingClientRect();
    const result = {
      topbarHeight: document.querySelector(".topbar").getBoundingClientRect().height,
      textareaMaxHeight: parseFloat(getComputedStyle(textarea).maxHeight),
      overflowingBottom: Math.max(0, box.bottom - window.innerHeight),
      sendVisible: sendBox.bottom <= window.innerHeight && sendBox.top >= 0,
    };
    expanderDone();
    return result;
  });

  await page.goto(process.env.APP_URL, { waitUntil: "domcontentloaded" });
  await page.fill("#pw", process.env.APP_PASSWORD);
  await page.click("#loginBtn");
  await page.waitForSelector(".topbar", { timeout: 20000 });
  await page.waitForFunction("window.__codoxearAppBootstrapped === true", null, { timeout: 30000 });

  await setInsets(0, 0);
  out.phone_no_inset = await measure();
  await page.screenshot({ path: process.env.SHOT_DIR + "/phone-no-inset.png" });

  await setInsets(47, 34);
  out.phone_notch_inset = await measure();
  await page.screenshot({ path: process.env.SHOT_DIR + "/phone-notch-inset.png" });

  await setInsets(0, 0);
  await page.setViewportSize({ width: 1280, height: 800 });
  out.desktop_no_inset = await measure();

  await page.setViewportSize({ width: 393, height: 852 });
  await setInsets(0, 0);
  out.tall_composer_plain = await measureTall();
  await setInsets(47, 34);
  out.tall_composer_notch = await measureTall();
  await page.screenshot({ path: process.env.SHOT_DIR + "/phone-notch-tall-composer.png" });

  await browser.close();
  process.stdout.write(JSON.stringify(out, null, 2));
})().catch((error) => {
  console.error("PROBE ERROR: " + (error && error.stack ? error.stack : String(error)));
  process.exit(1);
});
JS

PW_PATH="${node_modules}/playwright-core" \
APP_URL="http://127.0.0.1:${port}/" \
APP_PASSWORD="$password" \
SHOT_DIR="$artifacts" \
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}" \
node "$root/probe.js" > "$artifacts/probe.json" 2> "$artifacts/probe.err" \
  || fail "browser probe failed: $(tail -3 "$artifacts/probe.err")"

# ---- Assertions -------------------------------------------------------------
python3 - "$artifacts" <<'PY' || fail "geometry assertions failed (see $artifacts/report.json)"
import json
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])
CONTROL_HEIGHT = 32
NOTCH = 47

probe = json.loads((artifacts / "probe.json").read_text(encoding="utf-8"))
checks: dict[str, bool] = {}


def fits(state: dict) -> bool:
    return bool(state["controls"]) and all(
        control["visibleHeight"] >= CONTROL_HEIGHT for control in state["controls"]
    )


def tappable(state: dict) -> bool:
    return all(control["topEdgeHit"] and control["bottomEdgeHit"] for control in state["controls"])


# No inset: the ordinary band, on the phone and on the desktop.
for state_name in ("phone_no_inset", "desktop_no_inset"):
    state = probe[state_name]
    checks[f"{state_name}:height_56"] = abs(state["topbarHeight"] - 56) < 0.5
    checks[f"{state_name}:content_box_40"] = abs(state["contentHeight"] - 40) < 0.5
    checks[f"{state_name}:controls_fully_visible"] = fits(state)
    checks[f"{state_name}:controls_tappable"] = tappable(state)
    checks[f"{state_name}:inset_is_zero"] = state["insetSeen"] == 0

# The notch inset is space the header adds above its controls, never a slice
# taken out of them.
notch = probe["phone_notch_inset"]
checks["notch:inset_is_47"] = notch["insetSeen"] == NOTCH
checks["notch:header_grows_by_inset"] = abs(notch["topbarHeight"] - (56 + NOTCH)) < 0.5
checks["notch:content_box_preserved"] = abs(notch["contentHeight"] - 40) < 0.5
checks["notch:controls_fully_visible"] = fits(notch)
checks["notch:controls_clear_status_bar"] = all(control["top"] >= NOTCH for control in notch["controls"])
checks["notch:controls_tappable"] = tappable(notch)
checks["notch:chat_header_below_header"] = notch["chatHeaderTop"] >= notch["topbarBottom"] - 0.5

# An over-subscribed column must not squeeze the band back out of shape, and the
# composer's budget must give back the height the header takes.
tall_plain = probe["tall_composer_plain"]
tall_notch = probe["tall_composer_notch"]
checks["tall:header_holds_height_when_column_over_subscribed"] = abs(tall_notch["topbarHeight"] - (56 + NOTCH)) < 0.5
checks["tall:no_overflow_without_the_bottom_inset"] = tall_plain["overflowingBottom"] == 0
checks["tall:budget_gives_back_the_header_inset"] = (
    abs((tall_plain["textareaMaxHeight"] - tall_notch["textareaMaxHeight"]) - NOTCH) < 0.5
)
checks["tall:send_control_on_screen"] = tall_notch["sendVisible"] is True
checks["no_page_errors"] = not probe["errors"]

summary = {
    "pass": all(checks.values()),
    "checks": checks,
    "measurements": {
        name: probe[name]
        for name in ("phone_no_inset", "phone_notch_inset", "desktop_no_inset")
    },
    "tall_composer": {"plain": tall_plain, "notch": tall_notch},
}
(artifacts / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pass": summary["pass"], "failed": [name for name, ok in checks.items() if not ok]}, indent=2))
if not summary["pass"]:
    raise SystemExit(1)
PY

printf 'PASS: url=http://127.0.0.1:%s/ artifacts=%s screenshots=%s/{phone-no-inset,phone-notch-inset,phone-notch-tall-composer}.png\n' \
  "$port" "$artifacts" "$artifacts"
