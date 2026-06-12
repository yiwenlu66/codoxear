from __future__ import annotations

import base64
import hashlib
import hmac
import http.server
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CookieAuthSettings:
    cookie_name: str
    cookie_path: str
    cookie_expires: str
    cookie_secure: bool
    secret: bytes


def _chmod_private_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def load_or_create_hmac_secret(*, app_dir: Path, secret_path: Path) -> bytes:
    app_dir.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        _chmod_private_file(secret_path)
        b = secret_path.read_bytes()
        if len(b) < 32:
            raise ValueError(f"invalid hmac secret (too short): {secret_path}")
        return b[:64]
    secret = os.urandom(64)
    secret_path.write_bytes(secret)
    os.chmod(secret_path, 0o600)
    return secret


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64u_dec(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def sign_cookie(payload: dict[str, Any], *, secret: bytes) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return f"{_b64u(raw)}.{_b64u(sig)}"


def verify_cookie(value: str, *, secret: bytes) -> dict[str, Any] | None:
    try:
        a, b = value.split(".", 1)
        raw = _b64u_dec(a)
        sig = _b64u_dec(b)
        want = hmac.new(secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, want):
            return None
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def parse_cookies(header: str | None) -> dict[str, str]:
    if not header:
        return {}
    out: dict[str, str] = {}
    parts = header.split(";")
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def require_auth(
    handler: http.server.BaseHTTPRequestHandler,
    *,
    settings: CookieAuthSettings,
    verify: Callable[[str], dict[str, Any] | None] | None = None,
) -> bool:
    cookies = parse_cookies(handler.headers.get("Cookie"))
    token = cookies.get(settings.cookie_name)
    if not token:
        return False
    payload = (verify or (lambda value: verify_cookie(value, secret=settings.secret)))(token)
    if payload is None:
        return False
    if payload.get("v") != 1:
        setattr(handler, "_codoxear_refresh_auth_cookie", True)
    return True


def set_auth_cookie(handler: http.server.BaseHTTPRequestHandler, *, settings: CookieAuthSettings) -> None:
    token = sign_cookie({"v": 1}, secret=settings.secret)
    attrs = [
        f"{settings.cookie_name}={token}",
        f"Path={settings.cookie_path}",
        "HttpOnly",
        "SameSite=Strict",
        f"Expires={settings.cookie_expires}",
    ]
    forwarded_proto_raw = handler.headers.get("X-Forwarded-Proto")
    forwarded_proto = str(forwarded_proto_raw).lower() if forwarded_proto_raw is not None else ""
    if settings.cookie_secure or forwarded_proto == "https":
        attrs.append("Secure")
    handler.send_header("Set-Cookie", "; ".join(attrs))
