from __future__ import annotations


def normalize_url_prefix(raw: str | None) -> str:
    if raw is None:
        return ""
    value = str(raw).strip()
    if not value or value == "/":
        return ""
    if "://" in value:
        raise ValueError("CODEX_WEB_URL_PREFIX must be a path prefix (not a URL)")
    if "?" in value or "#" in value:
        raise ValueError("CODEX_WEB_URL_PREFIX must not include '?' or '#'")
    if not value.startswith("/"):
        raise ValueError("CODEX_WEB_URL_PREFIX must start with '/'")
    while len(value) > 1 and value.endswith("/"):
        value = value[:-1]
    if value == "/":
        return ""
    return value


def match_session_route(path: str, *suffix: str) -> str | None:
    parts = path.split("/")
    if len(parts) != 4 + len(suffix):
        return None
    if parts[:3] != ["", "api", "sessions"]:
        return None
    session_id = parts[3]
    if not session_id:
        return None
    if tuple(parts[4:]) != tuple(suffix):
        return None
    return session_id


def strip_url_prefix(prefix: str, path: str) -> str | None:
    if not prefix:
        return path
    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        return path[len(prefix) :]
    return None
