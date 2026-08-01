from __future__ import annotations

import hashlib
import urllib.parse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .server_http import gzip_response_body


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
    "app_new_session.js",
    "app_dom.js",
    "app_file_helpers.js",
    "app_file_picker.js",
    "app_file_viewer.js",
    "app_file_editor.js",
    "app_session_helpers.js",
    "app_viewport.js",
    "app_polling.js",
    "app_transcript.js",
    "app_message_identity.js",
    "app_message_rows.js",
    "app_conversation_copy.js",
    "app_modal.js",
    "app_clipboard.js",
    "app_code_copy.js",
    "app_hint_mode.js",
    "app_voice_helpers.js",
    "app_voice.js",
    "app_queue.js",
    "app_diagnostics.js",
    "app_recovery.js",
    "app_unattended.js",
    "app_chat_navigation.js",
    "app_chat_search.js",
    "app_shell.js",
    "app_composer.js",
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
MONACO_ASSET_ROOT = "monaco"
TOP_LEVEL_STATIC_ASSETS = (
    ("/favicon.ico", "favicon.png"),
    ("/manifest.webmanifest", "manifest.webmanifest"),
    ("/service-worker.js", "service-worker.js"),
    *((f"/{name}", name) for name in FRONTEND_ASSET_FILES),
    ("/favicon.png", "favicon.png"),
    ("/", "index.html"),
)
CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"


@dataclass(frozen=True)
class StaticRouteDeps:
    static_dir: Path
    top_level_static_assets: tuple[tuple[str, str], ...]
    read_static_bytes: Callable[[Path], bytes]
    static_cache_control_headers: Callable[..., dict[str, str]]
    content_security_policy: str


def static_cache_control_headers(*, versioned: bool, is_html: bool) -> dict[str, str]:
    if versioned and not is_html:
        return {"Cache-Control": "public, max-age=31536000, immutable"}
    return {"Cache-Control": "no-cache"}


def _static_version_asset_paths(base: Path) -> list[tuple[str, Path]]:
    assets: list[tuple[str, Path]] = []
    for rel in STATIC_ASSET_VERSION_FILES:
        assets.append((rel, base / rel))
    monaco_root = base / MONACO_ASSET_ROOT
    if monaco_root.is_dir():
        for path in sorted((p for p in monaco_root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(base).as_posix()):
            assets.append((path.relative_to(base).as_posix(), path))
    return assets


@lru_cache(maxsize=None)
def static_asset_version(static_dir: Path = STATIC_DIR) -> str:
    base = static_dir.resolve()
    digest = hashlib.sha256()
    for rel, raw_path in _static_version_asset_paths(base):
        path = raw_path.resolve()
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
    if path.suffix == ".json" or path.suffix == ".map":
        return "application/json; charset=utf-8"
    if path.suffix == ".ttf":
        return "font/ttf"
    if path.suffix == ".wasm":
        return "application/wasm"
    return "application/octet-stream"


def handle_static_get_route(handler: Any, *, path: str, query: str, deps: StaticRouteDeps) -> bool:
    rel = static_route_asset(path, top_level_static_assets=deps.top_level_static_assets)
    if rel is None:
        return False
    send_static_file(handler, rel, query=query, deps=deps)
    return True


def static_route_asset(path: str, *, top_level_static_assets: tuple[tuple[str, str], ...] = TOP_LEVEL_STATIC_ASSETS) -> str | None:
    for route, asset in top_level_static_assets:
        if path == route:
            return asset
    if path.startswith("/static/"):
        return path[len("/static/") :]
    if path.startswith("/monaco/"):
        return f"monaco/{path[len('/monaco/') :]}"
    return None


def send_static_file(handler: Any, rel: str, *, query: str, deps: StaticRouteDeps) -> None:
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
    body, gzip_encoded = gzip_response_body(handler, data)
    handler.send_response(200)
    handler.send_header("Content-Type", static_content_type(path))
    if gzip_encoded:
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Vary", "Accept-Encoding")
    if path.suffix == ".html":
        handler.send_header("Content-Security-Policy", deps.content_security_policy)
        handler.send_header("X-Frame-Options", "DENY")
    handler.send_header("Content-Length", str(len(body)))
    for name, value in deps.static_cache_control_headers(
        versioned="v" in urllib.parse.parse_qs(query, keep_blank_values=True),
        is_html=path.suffix == ".html",
    ).items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(body)
