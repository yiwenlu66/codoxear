from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ASSET_VERSION_PLACEHOLDER = "__CODOXEAR_ASSET_VERSION__"
STATIC_ATTACH_MAX_BYTES_PLACEHOLDER = "__CODOXEAR_ATTACH_MAX_BYTES__"
FRONTEND_ASSET_FILES = (
    "app_url.js",
    "app_storage.js",
    "app_perf.js",
    "app_api.js",
    "app_markdown.js",
    "app_launch.js",
    "app_display.js",
    "app_dom.js",
    "app_file_helpers.js",
    "app_session_helpers.js",
    "app_viewport.js",
    "app_polling.js",
    "app_transcript.js",
    "app_message_identity.js",
    "app_message_rows.js",
    "app_conversation_copy.js",
    "app_modal.js",
    "app_clipboard.js",
    "app_voice_helpers.js",
    "app.js",
    "app.css",
)
SHELL_ASSET_FILES = (
    "favicon.png",
    "manifest.webmanifest",
    "service-worker.js",
)
UI_IMAGE_ASSET_FILES = (
    "codoxear-icon.png",
    "logos/codex.svg",
    "logos/pi.svg",
    "logos/cc.svg",
)
STATIC_ASSET_VERSION_FILES = FRONTEND_ASSET_FILES + SHELL_ASSET_FILES + UI_IMAGE_ASSET_FILES
TOP_LEVEL_STATIC_ASSETS = (
    ("/favicon.ico", "favicon.png"),
    ("/manifest.webmanifest", "manifest.webmanifest"),
    ("/service-worker.js", "service-worker.js"),
    *((f"/{name}", name) for name in FRONTEND_ASSET_FILES),
    ("/favicon.png", "favicon.png"),
    ("/", "index.html"),
)
CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"


@dataclass(frozen=True)
class StaticRouteDeps:
    static_dir: Path
    top_level_static_assets: tuple[tuple[str, str], ...]
    read_static_bytes: Callable[[Path], bytes]
    static_cache_control_headers: Callable[[], dict[str, str]]
    content_security_policy: str


def static_cache_control_headers(*, enabled: bool | None = None) -> dict[str, str]:
    if enabled is None:
        enabled = str(os.environ.get("CODEX_WEB_STATIC_CACHE") or "").strip() == "1"
    if enabled:
        return {"Cache-Control": "public, max-age=31536000, immutable"}
    return {
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def static_asset_version(static_dir: Path = STATIC_DIR) -> str:
    base = static_dir.resolve()
    digest = hashlib.sha256()
    for rel in STATIC_ASSET_VERSION_FILES:
        path = (base / rel).resolve()
        if not str(path).startswith(str(base)):
            raise ValueError(f"static asset escaped static dir: {path}")
        if not path.is_file():
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def read_static_bytes(path: Path, *, attach_upload_max_bytes: int) -> bytes:
    data = path.read_bytes()
    if path.suffix != ".html":
        return data
    replacements = {
        STATIC_ASSET_VERSION_PLACEHOLDER.encode("ascii"): static_asset_version(path.parent).encode("ascii"),
        STATIC_ATTACH_MAX_BYTES_PLACEHOLDER.encode("ascii"): str(attach_upload_max_bytes).encode("ascii"),
    }
    for placeholder, value in replacements.items():
        if placeholder in data:
            data = data.replace(placeholder, value)
    return data


def static_content_type(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".js":
        return "text/javascript; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    if path.suffix == ".webmanifest":
        return "application/manifest+json; charset=utf-8"
    if path.suffix == ".png":
        return "image/png"
    if path.suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if path.suffix == ".webp":
        return "image/webp"
    if path.suffix == ".svg":
        return "image/svg+xml; charset=utf-8"
    if path.suffix == ".ico":
        return "image/x-icon"
    return "application/octet-stream"


def handle_static_get_route(handler: Any, *, path: str, deps: StaticRouteDeps) -> bool:
    rel = static_route_asset(path, top_level_static_assets=deps.top_level_static_assets)
    if rel is None:
        return False
    send_static_file(handler, rel, deps=deps)
    return True


def static_route_asset(path: str, *, top_level_static_assets: tuple[tuple[str, str], ...] = TOP_LEVEL_STATIC_ASSETS) -> str | None:
    for route, asset in top_level_static_assets:
        if path == route:
            return asset
    if path.startswith("/static/"):
        return path[len("/static/") :]
    return None


def send_static_file(handler: Any, rel: str, *, deps: StaticRouteDeps) -> None:
    static_root = deps.static_dir.resolve()
    path = (static_root / rel.lstrip("/")).resolve()
    try:
        path.relative_to(static_root)
    except ValueError:
        handler.send_error(404)
        return
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return
    data = deps.read_static_bytes(path)
    handler.send_response(200)
    handler.send_header("Content-Type", static_content_type(path))
    if path.suffix == ".html":
        handler.send_header("Content-Security-Policy", deps.content_security_policy)
        handler.send_header("X-Frame-Options", "DENY")
    handler.send_header("Content-Length", str(len(data)))
    # UI is used for interactive debugging; serve assets without caching by
    # default so changes (including inline JS) show up immediately on
    # refresh. Packaged deployments may opt into immutable static caching
    # with CODEX_WEB_STATIC_CACHE=1.
    for name, value in deps.static_cache_control_headers().items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(data)
