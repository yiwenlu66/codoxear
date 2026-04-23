from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runtime import ServerRuntime


def pi_native_session_dir_for_cwd(runtime: ServerRuntime, cwd: str | Path) -> Path:
    sv = runtime
    cwd_path = sv._safe_expanduser(Path(cwd)).resolve()
    slug = str(cwd_path).strip("/").replace("/", "-")
    return sv.PI_NATIVE_SESSIONS_DIR / f"--{slug}--"


def pi_new_session_file_for_cwd(runtime: ServerRuntime, cwd: str | Path) -> Path:
    sv = runtime
    now = float(sv._now())
    millis = int(round((now - sv.math.floor(now)) * 1000))
    if millis >= 1000:
        now = sv.math.floor(now) + 1.0
        millis = 0
    stamp = sv.time.strftime("%Y-%m-%dT%H-%M-%S", sv.time.gmtime(now))
    name = f"{stamp}-{millis:03d}Z_{sv.uuid.uuid4()}.jsonl"
    return pi_native_session_dir_for_cwd(runtime, cwd) / name


def write_pi_session_header(
    runtime: ServerRuntime,
    session_path: Path,
    *,
    session_id: str,
    cwd: str,
    parent_session: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    thinking_level: str | None = None,
) -> None:
    sv = runtime
    session_path.parent.mkdir(parents=True, exist_ok=True)
    now = float(sv._now())
    millis = int(round((now - sv.math.floor(now)) * 1000))
    if millis >= 1000:
        now = sv.math.floor(now) + 1.0
        millis = 0
    timestamp = f"{sv.time.strftime('%Y-%m-%dT%H:%M:%S', sv.time.gmtime(now))}.{millis:03d}Z"
    header: dict[str, Any] = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": timestamp,
        "cwd": cwd,
    }
    if isinstance(parent_session, str) and parent_session.strip():
        header["parentSession"] = parent_session.strip()
    if isinstance(provider, str) and provider.strip():
        header["provider"] = provider.strip()
    if isinstance(model_id, str) and model_id.strip():
        header["modelId"] = model_id.strip()
    if isinstance(thinking_level, str) and thinking_level.strip():
        header["thinkingLevel"] = thinking_level.strip()
    with session_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")


def pi_session_history_glob(session_path: Path) -> str:
    return f"{session_path.name}.history*"


def pi_session_has_handoff_history(session_path: Path) -> bool:
    try:
        return any(session_path.parent.glob(pi_session_history_glob(session_path)))
    except OSError:
        return False


def next_pi_handoff_history_path(session_path: Path) -> Path:
    base_name = f"{session_path.name}.history"
    candidate = session_path.with_name(base_name)
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = session_path.with_name(f"{base_name}.{i}")
        if not candidate.exists():
            return candidate
        i += 1


def copy_file_atomic(runtime: ServerRuntime, source_path: Path, target_path: Path) -> None:
    sv = runtime
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        raise FileExistsError(f"target already exists: {target_path}")
    tmp_path = target_path.with_name(
        f".{target_path.name}.codoxear-tmp-{sv.secrets.token_hex(6)}"
    )
    try:
        with source_path.open("rb") as src, tmp_path.open("xb") as dst:
            sv.shutil.copyfileobj(src, dst)
        try:
            st = source_path.stat()
            sv.os.chmod(tmp_path, st.st_mode & 0o777)
        except OSError:
            pass
        sv.os.replace(tmp_path, target_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def append_pi_user_message(
    runtime: ServerRuntime,
    session_path: Path,
    *,
    text: str,
) -> None:
    sv = runtime
    now = float(sv._now())
    millis = int(round(now * 1000))
    entry = {
        "type": "message",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "timestamp": millis,
        },
    }
    with session_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def pi_handoff_message_text(
    *, source_session_id: str, history_path: Path, cwd: str
) -> str:
    return "\n".join(
        [
            "Handoff context:",
            f"- Source session id: {source_session_id}",
            f"- Archived history file: {history_path}",
            f"- Working directory: {cwd}",
            "- This is a fresh session with no inherited chat context.",
            "- Read the archived history file only when you need prior context.",
        ]
    )


def write_pi_handoff_session(
    runtime: ServerRuntime,
    session_path: Path,
    *,
    session_id: str,
    cwd: str,
    source_session_id: str,
    history_path: Path,
    provider: str | None = None,
    model_id: str | None = None,
    thinking_level: str | None = None,
) -> None:
    write_pi_session_header(
        runtime,
        session_path,
        session_id=session_id,
        cwd=cwd,
        parent_session=source_session_id,
        provider=provider,
        model_id=model_id,
        thinking_level=thinking_level,
    )
    append_pi_user_message(
        runtime,
        session_path,
        text=pi_handoff_message_text(
            source_session_id=source_session_id,
            history_path=history_path,
            cwd=cwd,
        ),
    )


def pi_session_name_from_session_file(
    runtime: ServerRuntime,
    session_path: Path,
    *,
    max_scan_bytes: int = 512 * 1024,
) -> str:
    try:
        objs = runtime._read_jsonl_tail(session_path, max_scan_bytes)
    except Exception:
        return ""
    for obj in reversed(objs):
        if not isinstance(obj, dict) or obj.get("type") != "session_info":
            continue
        name = obj.get("name")
        if isinstance(name, str):
            return name.strip()
    return ""
