from __future__ import annotations

import time
import urllib.parse
from pathlib import Path
from typing import Any

_SERVER = None


def bind_server_runtime(runtime: Any) -> None:
    global _SERVER
    _SERVER = runtime



def _sv() -> Any:
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER



def handle_get(handler: Any, path: str, u: Any) -> bool:
    sv = _sv()
    if path == "/api/sessions/bootstrap":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        qs = urllib.parse.parse_qs(u.query)
        refresh_pi_models = (qs.get("refresh_pi_models") or ["0"])[0] == "1"
        sv._json_response(
            handler,
            200,
            {
                "recent_cwds": sv.MANAGER.recent_cwds(),
                "cwd_groups": sv.MANAGER.cwd_groups_get(),
                "new_session_defaults": sv._read_new_session_defaults(
                    page_state_db=getattr(sv.MANAGER, "_page_state_db", None),
                    refresh_pi_models=refresh_pi_models,
                ),
                "tmux_available": sv._tmux_available(),
            },
        )
        return True
    if path == "/api/sessions":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        t0 = time.perf_counter()
        qs = urllib.parse.parse_qs(u.query)
        group_key_q = qs.get("group_key")
        group_key = group_key_q[0] if group_key_q else None
        offset = max(0, int(qs.get("offset", ["0"])[0] or "0"))
        limit_default = sv.SESSION_LIST_PAGE_SIZE
        if group_key is not None:
            limit_default = sv.SESSION_LIST_GROUP_PAGE_SIZE
        limit = max(1, min(200, int(qs.get("limit", [str(limit_default)])[0] or str(limit_default))))
        group_offset = max(0, int(qs.get("group_offset", ["0"])[0] or "0"))
        group_limit = max(
            1,
            min(
                20,
                int(
                    qs.get("group_limit", [str(sv.SESSION_LIST_RECENT_GROUP_LIMIT)])[0]
                    or str(sv.SESSION_LIST_RECENT_GROUP_LIMIT)
                ),
            ),
        )
        payload = sv._session_list_payload(
            sv.MANAGER.list_sessions(),
            group_key=group_key,
            offset=offset,
            limit=limit,
            group_offset=group_offset,
            group_limit=group_limit,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sv._record_metric("api_sessions_ms", dt_ms)
        sv._json_response(handler, 200, payload)
        return True
    if path == "/api/session_resume_candidates":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        qs = urllib.parse.parse_qs(u.query)
        cwd_raw = qs.get("cwd", [""])[0]
        backend_raw = qs.get("backend", ["codex"])[0]
        offset_raw = qs.get("offset", ["0"])[0]
        limit_raw = qs.get("limit", ["20"])[0]
        try:
            agent_backend = sv.normalize_agent_backend(
                qs.get("agent_backend", [""])[0],
                default=sv.DEFAULT_AGENT_BACKEND,
            )
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        try:
            cwd_path = sv._resolve_dir_target(str(cwd_raw), field_name="cwd")
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e), "field": "cwd"})
            return True
        try:
            backend = sv._normalize_requested_backend(backend_raw)
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e), "field": "backend"})
            return True
        try:
            offset = max(0, int(offset_raw))
        except ValueError:
            sv._json_response(handler, 400, {"error": "offset must be an integer", "field": "offset"})
            return True
        try:
            limit = max(1, min(100, int(limit_raw)))
        except ValueError:
            sv._json_response(handler, 400, {"error": "limit must be an integer", "field": "limit"})
            return True
        info = sv._describe_session_cwd(cwd_path)
        all_rows = sv._list_resume_candidates_for_cwd(info["cwd"], backend=backend, limit=100000) if info["exists"] else []
        rows = all_rows[offset : offset + limit]
        remaining = max(0, len(all_rows) - (offset + len(rows)))
        for row in rows:
            sid = row.get("session_id")
            alias = sv.MANAGER.alias_get(sid) if isinstance(sid, str) and sid else ""
            preview = ""
            log_path_raw = row.get("log_path")
            session_path_raw = row.get("session_path")
            if isinstance(log_path_raw, str) and log_path_raw:
                preview = sv._first_user_message_preview_from_log(Path(log_path_raw))
            elif isinstance(session_path_raw, str) and session_path_raw:
                preview = sv._first_user_message_preview_from_pi_session(Path(session_path_raw))
            row["alias"] = alias
            row["first_user_message"] = preview
        sv._json_response(
            handler,
            200,
            {
                "ok": True,
                **info,
                "sessions": rows,
                "offset": offset,
                "limit": limit,
                "remaining": remaining,
                "agent_backend": agent_backend,
            },
        )
        return True
    if path == "/api/metrics":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        sv._json_response(handler, 200, {"metrics": sv._metrics_snapshot()})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/live"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        qs = urllib.parse.parse_qs(u.query)
        offset_q = qs.get("offset")
        live_offset_q = qs.get("live_offset")
        bridge_offset_q = qs.get("bridge_offset")
        requests_version_q = qs.get("requests_version")
        offset = 0 if offset_q is None else int(offset_q[0])
        live_offset = 0 if live_offset_q is None else int(live_offset_q[0])
        bridge_offset = 0 if bridge_offset_q is None else int(bridge_offset_q[0])
        requests_version = str(requests_version_q[0] or "").strip() or None if requests_version_q else None
        try:
            payload = sv._session_live_payload(
                sv.MANAGER,
                session_id,
                offset=offset,
                live_offset=live_offset,
                bridge_offset=bridge_offset,
                requests_version=requests_version,
            )
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, payload)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/workspace"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        try:
            payload = sv._session_workspace_payload(sv.MANAGER, session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, payload)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/details"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        try:
            payload = sv._session_details_payload(sv.MANAGER, session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        sv._json_response(handler, 200, payload)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/diagnostics"):
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
        try:
            state = sv._validated_session_state(sv.MANAGER.get_state(session_id))
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        token_val: dict[str, Any] | None = None
        st_token = state.get("token")
        if isinstance(st_token, dict) or st_token is None:
            token_val = st_token if isinstance(st_token, dict) else (s.token if isinstance(s.token, dict) else None)
        model_provider = s.model_provider
        preferred_auth_method = s.preferred_auth_method
        model = s.model
        reasoning_effort = s.reasoning_effort
        service_tier = s.service_tier
        if ((model_provider is None or model is None or reasoning_effort is None) and s.log_path is not None and s.log_path.exists()):
            log_provider, log_model, log_effort = sv._read_run_settings_from_log(s.log_path, agent_backend=s.agent_backend)
            if model_provider is None:
                model_provider = log_provider
            if model is None:
                model = log_model
            if reasoning_effort is None:
                reasoning_effort = log_effort
        sidebar_meta = sv.MANAGER.sidebar_meta_get(session_id)
        cwd_path = sv._safe_expanduser(Path(s.cwd))
        if not cwd_path.is_absolute():
            cwd_path = cwd_path.resolve()
        git_branch = sv._current_git_branch(cwd_path)
        updated_ts = sv._display_updated_ts(s)
        elapsed_s = max(0.0, time.time() - updated_ts)
        time_priority = sv._priority_from_elapsed_seconds(elapsed_s)
        base_priority = sv._clip01(time_priority + float(sidebar_meta["priority_offset"]))
        blocked = sidebar_meta["dependency_session_id"] is not None
        snoozed = sidebar_meta["snooze_until"] is not None and float(sidebar_meta["snooze_until"]) > time.time()
        final_priority = 0.0 if (snoozed or blocked) else base_priority
        broker_busy = sv._state_busy_value(state)
        busy = sv._display_pi_busy(s, broker_busy=broker_busy) if s.backend == "pi" else broker_busy
        if s.backend != "pi" and s.log_path is not None and s.log_path.exists():
            idle_val = sv.MANAGER.idle_from_log(session_id)
            busy = broker_busy or (not bool(idle_val))
        sv._json_response(
            handler,
            200,
            {
                "session_id": sv._durable_session_id_for_live_session(s),
                "runtime_id": s.session_id,
                "thread_id": s.thread_id,
                "agent_backend": s.agent_backend,
                "backend": s.backend,
                "owned": bool(s.owned),
                "transport": s.transport,
                "cwd": s.cwd,
                "start_ts": float(s.start_ts),
                "updated_ts": updated_ts,
                "log_path": str(s.log_path) if s.log_path is not None else None,
                "session_file_path": sv._display_source_path(s),
                "broker_pid": int(s.broker_pid),
                "codex_pid": int(s.codex_pid),
                "busy": bool(busy),
                "broker_busy": broker_busy,
                "queue_len": sv.MANAGER._queue_len(session_id),
                "token": token_val,
                "model_provider": model_provider,
                "preferred_auth_method": preferred_auth_method,
                "provider_choice": sv._provider_choice_for_backend(
                    backend=s.backend,
                    model_provider=model_provider,
                    preferred_auth_method=preferred_auth_method,
                ),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
                "tmux_session": s.tmux_session,
                "tmux_window": s.tmux_window,
                "git_branch": git_branch,
                "time_priority": time_priority,
                "base_priority": base_priority,
                "final_priority": final_priority,
                "priority_offset": sidebar_meta["priority_offset"],
                "snooze_until": sidebar_meta["snooze_until"],
                "dependency_session_id": sidebar_meta["dependency_session_id"],
                "todo_snapshot": sv._todo_snapshot_payload_for_session(s),
            },
        )
        return True
    if path.startswith("/api/sessions/") and path.endswith("/queue"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        try:
            q = sv.MANAGER.queue_list(session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, "queue": q})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/ui_state"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        if not session_id:
            handler.send_error(404)
            return True
        try:
            payload = sv.MANAGER.get_ui_state(session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, payload)
        return True
    session_id = sv._match_session_route(path, "commands")
    if session_id is not None:
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        try:
            payload = sv.MANAGER.get_session_commands(session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, payload)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/git/changed_files"):
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
        cwd = sv._safe_expanduser(Path(s.cwd))
        if not cwd.is_absolute():
            cwd = cwd.resolve()
        try:
            sv._require_git_repo(cwd)
        except RuntimeError as e:
            sv._json_response(handler, 409, {"error": str(e)})
            return True
        unstaged = sv._run_git(cwd, ["diff", "--name-only"], timeout_s=sv.GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).splitlines()
        staged = sv._run_git(cwd, ["diff", "--name-only", "--cached"], timeout_s=sv.GIT_DIFF_TIMEOUT_SECONDS, max_bytes=64 * 1024).splitlines()
        unstaged_numstat = sv._run_git(cwd, ["diff", "--numstat"], timeout_s=sv.GIT_DIFF_TIMEOUT_SECONDS, max_bytes=128 * 1024)
        staged_numstat = sv._run_git(cwd, ["diff", "--numstat", "--cached"], timeout_s=sv.GIT_DIFF_TIMEOUT_SECONDS, max_bytes=128 * 1024)

        def _norm_list(xs: list[str]) -> list[str]:
            out: list[str] = []
            for x in xs:
                t = x.strip()
                if not t:
                    continue
                out.append(t)
                if len(out) >= sv.GIT_CHANGED_FILES_MAX:
                    break
            return out

        unstaged2 = _norm_list(unstaged)
        staged2 = _norm_list(staged)
        seen: set[str] = set()
        merged: list[str] = []
        for x in [*unstaged2, *staged2]:
            if x in seen:
                continue
            seen.add(x)
            merged.append(x)
        stats = sv._parse_git_numstat(unstaged_numstat)
        for path_key, vals in sv._parse_git_numstat(staged_numstat).items():
            prev = stats.get(path_key)
            if prev is None:
                stats[path_key] = vals
                continue
            add_prev = prev.get("additions")
            del_prev = prev.get("deletions")
            add_new = vals.get("additions")
            del_new = vals.get("deletions")
            prev["additions"] = None if add_prev is None or add_new is None else int(add_prev) + int(add_new)
            prev["deletions"] = None if del_prev is None or del_new is None else int(del_prev) + int(del_new)
        entries: list[dict[str, Any]] = []
        for path_key in merged:
            vals = stats.get(path_key, {})
            entries.append({
                "path": path_key,
                "additions": vals.get("additions"),
                "deletions": vals.get("deletions"),
                "changed": True,
            })
        sv._json_response(handler, 200, {
            "ok": True,
            "cwd": str(cwd),
            "files": merged,
            "entries": entries,
            "unstaged": unstaged2,
            "staged": staged2,
        })
        return True
    if path.startswith("/api/sessions/") and path.endswith("/git/diff"):
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
        staged_q = qs.get("staged")
        staged = bool(staged_q and staged_q[0] == "1")
        cwd = sv._safe_expanduser(Path(s.cwd))
        if not cwd.is_absolute():
            cwd = cwd.resolve()
        try:
            sv._require_git_repo(cwd)
        except RuntimeError as e:
            sv._json_response(handler, 409, {"error": str(e)})
            return True
        try:
            _target, _repo_root, rel = sv._resolve_git_path(cwd, rel)
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        args = ["diff", "-U3"]
        if staged:
            args.append("--cached")
        args.extend(["--", rel])
        diff = sv._run_git(cwd, args, timeout_s=sv.GIT_DIFF_TIMEOUT_SECONDS, max_bytes=sv.GIT_DIFF_MAX_BYTES)
        sv._json_response(handler, 200, {"ok": True, "cwd": str(cwd), "path": rel, "staged": staged, "diff": diff})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/git/file_versions"):
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
        cwd = sv._safe_expanduser(Path(s.cwd))
        if not cwd.is_absolute():
            cwd = cwd.resolve()
        try:
            sv._require_git_repo(cwd)
        except RuntimeError as e:
            sv._json_response(handler, 409, {"error": str(e)})
            return True
        try:
            p, _repo_root, rel = sv._resolve_git_path(cwd, rel)
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        current_text = ""
        current_size = 0
        current_exists = bool(p.exists() and p.is_file())
        if current_exists:
            current_text, current_size = sv._read_text_file_strict(p, max_bytes=sv.FILE_READ_MAX_BYTES)
        try:
            sv.MANAGER.files_add(session_id, str(p))
        except KeyError:
            pass
        base_exists = False
        base_text = ""
        try:
            base_text = sv._run_git(cwd, ["show", f"HEAD:{rel}"], timeout_s=sv.GIT_DIFF_TIMEOUT_SECONDS, max_bytes=sv.FILE_READ_MAX_BYTES)
            base_exists = True
        except RuntimeError:
            base_exists = False
            base_text = ""
        sv._json_response(handler, 200, {
            "ok": True,
            "cwd": str(cwd),
            "path": rel,
            "abs_path": str(p),
            "current_exists": current_exists,
            "current_size": int(current_size),
            "current_text": current_text,
            "base_exists": base_exists,
            "base_text": base_text,
        })
        return True
    if path.startswith("/api/sessions/") and path.endswith("/messages"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        t0_total = time.perf_counter()
        parts = path.split("/")
        if len(parts) < 4:
            handler.send_error(404)
            return True
        session_id = parts[3]
        t0_meta = time.perf_counter()
        sv.MANAGER.refresh_session_meta(session_id, strict=False)
        dt_meta_ms = (time.perf_counter() - t0_meta) * 1000.0
        s = sv.MANAGER.get_session(session_id)
        historical_row = sv._historical_session_row(session_id)
        if (not s) and historical_row is None:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        qs = urllib.parse.parse_qs(u.query)
        offset_q = qs.get("offset")
        offset = 0 if offset_q is None else int(offset_q[0])
        if offset < 0:
            offset = 0
        init_q = qs.get("init")
        init = bool(init_q and init_q[0] == "1")
        before_q = qs.get("before")
        before = 0 if before_q is None else int(before_q[0])
        before = max(0, before)
        limit_q = qs.get("limit")
        limit = sv.SESSION_HISTORY_PAGE_SIZE if limit_q is None else int(limit_q[0])
        limit = max(20, min(sv.SESSION_HISTORY_PAGE_SIZE, limit))
        payload = sv.MANAGER.get_messages_page(session_id, offset=offset, init=init, limit=limit, before=before)
        if isinstance(payload.get("diag"), dict) and s is not None and s.backend != "pi":
            payload["diag"]["meta_refresh_ms"] = round(dt_meta_ms, 3)
        sv._json_response(handler, 200, payload)
        dt_total_ms = (time.perf_counter() - t0_total) * 1000.0
        sv._record_metric("api_messages_init_ms" if init else "api_messages_poll_ms", dt_total_ms)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/tail"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        try:
            tail = sv.MANAGER.get_tail(session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        sv._json_response(handler, 200, {"tail": tail})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/harness"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        if len(parts) < 4:
            handler.send_error(404)
            return True
        session_id = parts[3]
        try:
            cfg = sv.MANAGER.harness_get(session_id)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        sv._json_response(handler, 200, {"ok": True, **cfg})
        return True
    return False
