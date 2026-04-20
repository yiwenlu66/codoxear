from __future__ import annotations

import base64
import json
import urllib.parse
from pathlib import Path

_SERVER = None


def bind_server_runtime(runtime) -> None:
    global _SERVER
    _SERVER = runtime



def _sv():
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER



def _send_inline_blob(handler, path_obj: Path) -> None:
    sv = _sv()
    raw = path_obj.read_bytes()
    kind, ctype = sv._file_kind(path_obj, raw)
    if kind not in {"image", "pdf"} or ctype is None:
        sv._json_response(handler, 400, {"error": "file is not previewable inline"})
        return
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header(
        "Content-Disposition",
        f"inline; filename*=UTF-8''{urllib.parse.quote(path_obj.name, safe='')}",
    )
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    handler.wfile.write(raw)



def handle_get(handler, path: str, u) -> bool:
    sv = _sv()
    if path.startswith("/api/sessions/") and path.endswith("/file/read"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        qs = urllib.parse.parse_qs(u.query)
        path_q = qs.get("path")
        if not path_q or not path_q[0]:
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        rel = path_q[0]
        base = sv._safe_expanduser(Path(s.cwd))
        if not base.is_absolute():
            base = base.resolve()
        p = sv._resolve_session_path(base, rel)
        if not p.exists():
            sv._json_response(handler, 404, {"error": "file not found"})
            return True
        if not p.is_file():
            sv._json_response(handler, 400, {"error": "path is not a file"})
            return True
        try:
            view = sv._read_client_file_view(p)
        except PermissionError as e:
            sv._json_response(handler, 403, {"error": str(e)})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        try:
            sv.MANAGER.files_add(session_id, str(p))
        except KeyError:
            pass
        if view.kind == "image":
            sv._json_response(handler, 200, {
                "ok": True,
                "kind": "image",
                "content_type": view.content_type,
                "path": str(p),
                "rel": str(rel),
                "size": int(view.size),
                "image_url": f"/api/sessions/{session_id}/file/blob?path={urllib.parse.quote(rel)}",
            })
            return True
        if view.kind == "pdf":
            sv._json_response(handler, 200, {
                "ok": True,
                "kind": "pdf",
                "content_type": view.content_type,
                "path": str(p),
                "rel": str(rel),
                "size": int(view.size),
                "pdf_url": f"/api/sessions/{session_id}/file/blob?path={urllib.parse.quote(rel)}",
            })
            return True
        if view.kind == "download_only":
            sv._json_response(handler, 200, {
                "ok": True,
                "kind": "download_only",
                "path": str(p),
                "rel": str(rel),
                "size": int(view.size),
                "reason": view.blocked_reason,
                "viewer_max_bytes": view.viewer_max_bytes,
            })
            return True
        sv._json_response(handler, 200, {
            "ok": True,
            "kind": view.kind,
            "path": str(p),
            "rel": str(rel),
            "size": int(view.size),
            "text": view.text,
            "editable": bool(view.editable),
            "version": view.version,
        })
        return True
    if path.startswith("/api/sessions/") and path.endswith("/file/search"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        qs = urllib.parse.parse_qs(u.query)
        query_raw = qs.get("q")
        if not query_raw or not query_raw[0].strip():
            sv._json_response(handler, 400, {"error": "q required"})
            return True
        limit_raw = qs.get("limit", [str(sv.FILE_SEARCH_LIMIT)])[0]
        try:
            limit = int(str(limit_raw).strip() or str(sv.FILE_SEARCH_LIMIT))
        except ValueError:
            sv._json_response(handler, 400, {"error": "limit must be an integer"})
            return True
        if limit < 1:
            sv._json_response(handler, 400, {"error": "limit must be >= 1"})
            return True
        base = sv._safe_expanduser(Path(s.cwd))
        if not base.is_absolute():
            base = base.resolve()
        try:
            result = sv._search_session_relative_files(base, query=query_raw[0], limit=limit)
        except FileNotFoundError as e:
            sv._json_response(handler, 404, {"error": str(e)})
            return True
        except PermissionError as e:
            sv._json_response(handler, 403, {"error": str(e)})
            return True
        except (RuntimeError, ValueError) as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {
            "ok": True,
            "cwd": str(base),
            "query": result["query"],
            "mode": result["mode"],
            "matches": result["matches"],
            "scanned": result["scanned"],
            "truncated": result["truncated"],
        })
        return True
    if path.startswith("/api/sessions/") and path.endswith("/file/list"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        qs = urllib.parse.parse_qs(u.query)
        raw_rel = qs.get("path", [""])[0]
        base = sv._safe_expanduser(Path(s.cwd))
        if not base.is_absolute():
            base = base.resolve()
        try:
            entries = sv._list_session_directory_entries(base, raw_rel)
        except FileNotFoundError as e:
            sv._json_response(handler, 404, {"error": str(e)})
            return True
        except PermissionError as e:
            sv._json_response(handler, 403, {"error": str(e)})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {
            "ok": True,
            "cwd": str(base),
            "path": str(raw_rel or ""),
            "entries": entries,
        })
        return True
    if path.startswith("/api/sessions/") and path.endswith("/file/blob"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        qs = urllib.parse.parse_qs(u.query)
        path_q = qs.get("path")
        if not path_q or not path_q[0]:
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        rel = path_q[0]
        base = sv._safe_expanduser(Path(s.cwd))
        if not base.is_absolute():
            base = base.resolve()
        p = sv._resolve_session_path(base, rel)
        if not p.exists():
            sv._json_response(handler, 404, {"error": "file not found"})
            return True
        if not p.is_file():
            sv._json_response(handler, 400, {"error": "path is not a file"})
            return True
        _send_inline_blob(handler, p)
        return True
    if path == "/api/files/blob":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        qs = urllib.parse.parse_qs(u.query)
        path_q = qs.get("path")
        if not path_q or not path_q[0]:
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        path_obj = Path(path_q[0]).expanduser().resolve()
        if not path_obj.exists():
            sv._json_response(handler, 404, {"error": "file not found"})
            return True
        if not path_obj.is_file():
            sv._json_response(handler, 400, {"error": "path is not a file"})
            return True
        _send_inline_blob(handler, path_obj)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/file/download"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        qs = urllib.parse.parse_qs(u.query)
        path_q = qs.get("path")
        if not path_q or not path_q[0]:
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        rel = path_q[0]
        base = sv._safe_expanduser(Path(s.cwd))
        if not base.is_absolute():
            base = base.resolve()
        p = sv._resolve_session_path(base, rel)
        try:
            raw, size = sv._read_downloadable_file(p)
        except FileNotFoundError as e:
            sv._json_response(handler, 404, {"error": str(e)})
            return True
        except PermissionError as e:
            sv._json_response(handler, 403, {"error": str(e)})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", str(size))
        handler.send_header("Content-Disposition", sv._download_disposition(p))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Expires", "0")
        handler.end_headers()
        handler.wfile.write(raw)
        return True
    return False



def handle_post(handler, path: str, u) -> bool:
    sv = _sv()
    if path == "/api/files/read":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        path_raw = obj.get("path")
        if not isinstance(path_raw, str) or not path_raw.strip():
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        session_id_raw = obj.get("session_id")
        session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
        try:
            path_obj = sv._resolve_client_file_path(session_id=session_id, raw_path=path_raw)
            view = sv._read_client_file_view(path_obj)
        except FileNotFoundError as e:
            sv._json_response(handler, 404, {"error": str(e)})
            return True
        except PermissionError as e:
            sv._json_response(handler, 403, {"error": str(e)})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        if session_id:
            try:
                sv.MANAGER.files_add(session_id, str(path_obj))
            except KeyError:
                pass
        if view.kind == "image":
            sv._json_response(handler, 200, {
                "ok": True,
                "kind": "image",
                "content_type": view.content_type,
                "path": str(path_obj),
                "size": int(view.size),
                "image_url": f"/api/files/blob?path={urllib.parse.quote(str(path_obj))}",
            })
            return True
        if view.kind == "pdf":
            sv._json_response(handler, 200, {
                "ok": True,
                "kind": "pdf",
                "content_type": view.content_type,
                "path": str(path_obj),
                "size": int(view.size),
                "pdf_url": f"/api/files/blob?path={urllib.parse.quote(str(path_obj))}",
            })
            return True
        if view.kind == "download_only":
            sv._json_response(handler, 200, {
                "ok": True,
                "kind": "download_only",
                "path": str(path_obj),
                "size": int(view.size),
                "reason": view.blocked_reason,
                "viewer_max_bytes": view.viewer_max_bytes,
            })
            return True
        sv._json_response(handler, 200, {
            "ok": True,
            "kind": view.kind,
            "path": str(path_obj),
            "size": int(view.size),
            "text": view.text,
            "editable": bool(view.editable),
            "version": view.version,
        })
        return True
    if path == "/api/files/inspect":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        path_raw = obj.get("path")
        if not isinstance(path_raw, str) or not path_raw.strip():
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        session_id_raw = obj.get("session_id")
        session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
        try:
            path_obj = sv._resolve_client_file_path(session_id=session_id, raw_path=path_raw)
            view = sv._read_client_file_view(path_obj)
        except FileNotFoundError as e:
            sv._json_response(handler, 404, {"error": str(e)})
            return True
        except PermissionError as e:
            sv._json_response(handler, 403, {"error": str(e)})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {
            "ok": True,
            "path": str(path_obj),
            "kind": view.kind,
            "content_type": view.content_type,
            "size": int(view.size),
            "reason": view.blocked_reason,
            "viewer_max_bytes": view.viewer_max_bytes,
        })
        return True
    if path == "/api/files/blob":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        qs = urllib.parse.parse_qs(u.query)
        path_q = qs.get("path")
        if not path_q or not path_q[0]:
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        path_obj = sv._safe_expanduser(Path(path_q[0])).resolve()
        if not path_obj.exists():
            sv._json_response(handler, 404, {"error": "file not found"})
            return True
        if not path_obj.is_file():
            sv._json_response(handler, 400, {"error": "path is not a file"})
            return True
        _send_inline_blob(handler, path_obj)
        return True
    session_id = sv._match_session_route(path, "file", "write")
    if session_id is not None:
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        path_raw = obj.get("path")
        if not isinstance(path_raw, str) or not path_raw.strip():
            sv._json_response(handler, 400, {"error": "path required"})
            return True
        text_raw = obj.get("text")
        if not isinstance(text_raw, str):
            sv._json_response(handler, 400, {"error": "text must be a string"})
            return True
        create_raw = obj.get("create")
        create = create_raw if isinstance(create_raw, bool) else False
        version_raw = obj.get("version")
        if not create and (not isinstance(version_raw, str) or not version_raw.strip()):
            sv._json_response(handler, 400, {"error": "version required"})
            return True
        sv.MANAGER.refresh_session_meta(session_id)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        base = Path(s.cwd).expanduser()
        if not base.is_absolute():
            base = base.resolve()
        if create:
            try:
                p = sv._resolve_under(base, path_raw)
            except ValueError as e:
                sv._json_response(handler, 400, {"error": str(e)})
                return True
            try:
                size, next_version = sv._write_new_text_file_atomic(p, text=text_raw)
            except FileExistsError:
                payload = {"error": "file already exists", "conflict": True, "path": str(p)}
                if p.is_file():
                    try:
                        _current_text, _current_size, current_version = sv._read_text_file_for_write(
                            p, max_bytes=sv.FILE_READ_MAX_BYTES
                        )
                        payload["version"] = current_version
                    except (FileNotFoundError, PermissionError, ValueError):
                        pass
                sv._json_response(handler, 409, payload)
                return True
            except FileNotFoundError as e:
                sv._json_response(handler, 404, {"error": str(e)})
                return True
            except PermissionError as e:
                sv._json_response(handler, 403, {"error": str(e)})
                return True
            except ValueError as e:
                sv._json_response(handler, 400, {"error": str(e)})
                return True
        else:
            p = sv._resolve_session_path(base, path_raw)
            try:
                _current_text, _current_size, current_version = sv._read_text_file_for_write(
                    p, max_bytes=sv.FILE_READ_MAX_BYTES
                )
            except FileNotFoundError as e:
                sv._json_response(handler, 404, {"error": str(e)})
                return True
            except PermissionError as e:
                sv._json_response(handler, 403, {"error": str(e)})
                return True
            except ValueError as e:
                sv._json_response(handler, 400, {"error": str(e)})
                return True
            if current_version != version_raw:
                sv._json_response(handler, 409, {
                    "error": "file changed on disk",
                    "conflict": True,
                    "path": str(p),
                    "version": current_version,
                })
                return True
            try:
                size, next_version = sv._write_text_file_atomic(p, text=text_raw)
            except FileNotFoundError as e:
                sv._json_response(handler, 404, {"error": str(e)})
                return True
            except PermissionError as e:
                sv._json_response(handler, 403, {"error": str(e)})
                return True
            except ValueError as e:
                sv._json_response(handler, 400, {"error": str(e)})
                return True
        try:
            sv.MANAGER.files_add(session_id, str(p))
        except KeyError:
            pass
        sv._json_response(handler, 200, {
            "ok": True,
            "path": str(p),
            "rel": str(path_raw),
            "size": int(size),
            "version": next_version,
            "editable": True,
        })
        return True
    if path.startswith("/api/sessions/") and (path.endswith("/inject_file") or path.endswith("/inject_image")):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        s = sv.MANAGER.get_session(session_id)
        if not s:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        if s.backend == "pi":
            sv._json_response(handler, 409, {
                "error": "attachment injection is not supported for Pi sessions",
                "backend": "pi",
                "operation": "attachment_injection",
            })
            return True
        try:
            body = sv._read_body(handler, limit=sv.ATTACH_UPLOAD_BODY_MAX_BYTES)
        except ValueError:
            sv._json_response(handler, 413, {"error": f"file too large (max {sv.ATTACH_UPLOAD_MAX_BYTES} bytes)"})
            return True
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        data_b64 = obj.get("data_b64")
        filename = obj.get("filename")
        attachment_index = obj.get("attachment_index")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename required")
        if isinstance(attachment_index, bool) or not isinstance(attachment_index, int):
            sv._json_response(handler, 400, {"error": "attachment_index must be an integer"})
            return True
        if not isinstance(data_b64, str) or not data_b64:
            sv._json_response(handler, 400, {"error": "data_b64 required"})
            return True
        try:
            raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except Exception:
            sv._json_response(handler, 400, {"error": "invalid base64"})
            return True
        try:
            out_path = sv._stage_uploaded_file(session_id, filename, raw)
        except ValueError as e:
            status = 413 if str(e).startswith("file too large") else 400
            sv._json_response(handler, status, {"error": str(e)})
            return True
        try:
            inject_text = sv._attachment_inject_text(attachment_index, out_path)
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        seq = f"\x1b[200~{inject_text}\x1b[201~"
        try:
            resp = sv.MANAGER.inject_keys(session_id, seq)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {
            "ok": True,
            "path": str(out_path),
            "inject_text": inject_text,
            "broker": resp,
        })
        return True
    return False
