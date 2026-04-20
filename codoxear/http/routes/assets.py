from __future__ import annotations

_SERVER = None


def bind_server_runtime(runtime) -> None:
    global _SERVER
    _SERVER = runtime



def _sv():
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER


def handle_get(handler, path: str, u) -> bool:
    sv = _sv()
    if path == "/favicon.ico":
        resolved = sv._resolve_public_web_asset("favicon.ico")
        if resolved is not None:
            handler._send_path(resolved)
            return True
        handler._send_static("favicon.png")
        return True
    if path == "/manifest.webmanifest":
        resolved = sv._resolve_public_web_asset("manifest.webmanifest")
        if resolved is None:
            handler.send_error(404)
            return True
        handler._send_path(resolved)
        return True
    if path == "/service-worker.js":
        resolved = sv._resolve_public_web_asset("service-worker.js")
        if resolved is None:
            handler.send_error(404)
            return True
        handler._send_path(resolved)
        return True
    if path == "/app.js":
        handler._send_static("app.js")
        return True
    if path == "/app.css":
        handler._send_static("app.css")
        return True
    if path == "/favicon.png":
        resolved = sv._resolve_public_web_asset("favicon.png")
        if resolved is not None:
            handler._send_path(resolved)
            return True
        handler._send_static("favicon.png")
        return True
    if path == "/":
        body, ctype = sv._read_web_index()
        handler._send_bytes(body.encode("utf-8"), ctype)
        return True
    if path.startswith("/assets/") and not sv.USE_LEGACY_WEB:
        served_dist_dir = sv._served_web_dist_dir()
        if served_dist_dir is not None:
            candidate = (served_dist_dir / path.lstrip("/")).resolve()
            if sv._is_path_within(served_dist_dir.resolve(), candidate) and candidate.is_file():
                handler._send_path(candidate)
                return True
        handler.send_error(404)
        return True
    if (
        not sv.USE_LEGACY_WEB
        and path.startswith("/")
        and "/" not in path[1:]
        and not path.startswith("/api/")
    ):
        resolved = sv._resolve_public_web_asset(path)
        if resolved is not None:
            handler._send_path(resolved)
            return True
    if path.startswith("/static/"):
        handler._send_static(path[len("/static/") :])
        return True
    return False
