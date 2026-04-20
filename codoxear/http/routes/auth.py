from __future__ import annotations

import json

_SERVER = None


def bind_server_runtime(runtime) -> None:
    global _SERVER
    _SERVER = runtime



def _sv():
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER


def handle_get(handler, path: str, _u) -> bool:
    sv = _sv()
    if path != "/api/me":
        return False
    if not sv._require_auth(handler):
        handler._unauthorized()
        return True
    sv._json_response(handler, 200, {"ok": True})
    return True



def handle_post(handler, path: str, _u) -> bool:
    sv = _sv()
    if path == "/api/login":
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        pw = obj.get("password")
        if not isinstance(pw, str) or not sv._is_same_password(pw):
            sv._json_response(handler, 403, {"error": "bad password"})
            return True
        handler.send_response(200)
        sv._set_auth_cookie(handler)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.end_headers()
        handler.wfile.write(b'{"ok":true}')
        return True
    if path == "/api/logout":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        handler.send_response(200)
        handler.send_header(
            "Set-Cookie",
            f"{sv.COOKIE_NAME}=deleted; Path={sv.COOKIE_PATH}; Max-Age=0; HttpOnly; SameSite=Strict",
        )
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.end_headers()
        handler.wfile.write(b'{"ok":true}')
        return True
    return False
