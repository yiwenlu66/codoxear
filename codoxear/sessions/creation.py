from __future__ import annotations

from typing import Any

from ..runtime import ServerRuntime


def parse_create_session_request(
    runtime: ServerRuntime, obj: dict[str, Any]
) -> dict[str, Any]:
    sv = runtime
    cwd = obj.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("cwd required")

    name_raw = obj.get("name")
    if name_raw is None:
        name = None
    elif isinstance(name_raw, str):
        name = name_raw
    else:
        raise ValueError("name must be a string")

    backend = sv.normalize_agent_backend(
        obj.get("backend"),
        default=sv.normalize_agent_backend(obj.get("agent_backend"), default="codex"),
    )
    args = obj.get("args")
    if args is None:
        args_list = None
    elif isinstance(args, list) and all(isinstance(x, str) for x in args):
        args_list = [x for x in args if x]
    else:
        raise ValueError("args must be a list of strings")

    resume_session_id = sv._clean_optional_resume_session_id(obj.get("resume_session_id"))

    create_in_tmux_raw = obj.get("create_in_tmux")
    if create_in_tmux_raw is None:
        create_in_tmux = False
    elif isinstance(create_in_tmux_raw, bool):
        create_in_tmux = create_in_tmux_raw
    else:
        raise ValueError("create_in_tmux must be a boolean")

    if backend == "pi":
        pi_provider_choices = {
            str(value)
            for value in (
                read_pi_launch_defaults(runtime).get("provider_choices") or []
            )
            if isinstance(value, str) and value.strip()
        }
        model_provider = sv._normalize_requested_model_provider(
            obj.get("model_provider"),
            allowed=pi_provider_choices or None,
        )
        model = sv._normalize_requested_model(obj.get("model"))
        reasoning_effort = sv._normalize_requested_pi_reasoning_effort(
            obj.get("reasoning_effort")
        )
        return {
            "cwd": cwd,
            "name": name,
            "backend": backend,
            "args": args_list,
            "resume_session_id": resume_session_id,
            "worktree_branch": None,
            "model_provider": model_provider,
            "preferred_auth_method": None,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "service_tier": None,
            "create_in_tmux": create_in_tmux,
        }

    allowed_providers = set(
        read_codex_launch_defaults(runtime).get("model_providers") or ["openai"]
    )
    model_provider = sv._normalize_requested_model_provider(
        obj.get("model_provider"),
        allowed=set(
            [
                "openai",
                *[p for p in allowed_providers if p not in {"chatgpt", "openai-api"}],
            ]
        ),
    )
    preferred_auth_method = sv._normalize_requested_preferred_auth_method(
        obj.get("preferred_auth_method")
    )
    model = sv._normalize_requested_model(obj.get("model"))
    reasoning_effort = sv._normalize_requested_reasoning_effort(
        obj.get("reasoning_effort")
    )
    service_tier = sv._normalize_requested_service_tier(obj.get("service_tier"))

    worktree_branch_raw = obj.get("worktree_branch")
    if worktree_branch_raw is None:
        worktree_branch = None
    elif isinstance(worktree_branch_raw, str):
        worktree_branch = worktree_branch_raw.strip() or None
    else:
        raise ValueError("worktree_branch must be a string")

    return {
        "cwd": cwd,
        "name": name,
        "backend": backend,
        "args": args_list,
        "resume_session_id": resume_session_id,
        "worktree_branch": worktree_branch,
        "model_provider": model_provider,
        "preferred_auth_method": preferred_auth_method,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "create_in_tmux": create_in_tmux,
    }


def configured_model_providers(data: dict[str, Any]) -> list[str]:
    providers = ["openai"]
    seen = {"openai"}
    raw = data.get("model_providers")
    if not isinstance(raw, dict):
        return providers
    for key in raw.keys():
        if not isinstance(key, str):
            continue
        name = key.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        providers.append(name)
    return providers


def read_codex_launch_defaults(runtime: ServerRuntime) -> dict[str, Any]:
    sv = runtime
    configured_model = None
    configured_effort = None
    configured_provider = "openai"
    configured_auth_method = "apikey"
    configured_service_tier = "flex"
    configured_providers = ["chatgpt", "openai-api"]
    if sv.CODEX_CONFIG_PATH.exists():
        data = sv.tomllib.loads(sv.CODEX_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Codex config in {sv.CODEX_CONFIG_PATH}")
        configured_model = sv._clean_optional_text(data.get("model"))
        configured_effort = sv._display_reasoning_effort(
            data.get("model_reasoning_effort")
        )
        configured_auth_method = (
            sv._normalize_requested_preferred_auth_method(
                data.get("preferred_auth_method")
            )
            or configured_auth_method
        )
        configured_providers = [
            "chatgpt",
            "openai-api",
            *[p for p in configured_model_providers(data) if p != "openai"],
        ]
        configured_provider = (
            sv._normalize_requested_model_provider(
                data.get("model_provider") or data.get("model_provider_id"),
                allowed=set(
                    [
                        "openai",
                        *[
                            p
                            for p in configured_providers
                            if p not in {"chatgpt", "openai-api"}
                        ],
                    ]
                ),
            )
            or configured_provider
        )
        configured_service_tier = (
            sv._normalize_requested_service_tier(data.get("service_tier"))
            or configured_service_tier
        )
    defaults: dict[str, Any] = {
        "model_provider": configured_provider,
        "preferred_auth_method": configured_auth_method,
        "provider_choice": sv._provider_choice_for_settings(
            model_provider=configured_provider,
            preferred_auth_method=configured_auth_method,
        ),
        "model": configured_model,
        "model_providers": configured_providers,
        "service_tier": configured_service_tier,
    }
    if configured_effort is not None:
        defaults["reasoning_effort"] = configured_effort
        return defaults
    if not sv.MODELS_CACHE_PATH.exists():
        defaults["reasoning_effort"] = None
        return defaults
    cache = sv.json.loads(sv.MODELS_CACHE_PATH.read_text(encoding="utf-8"))
    models = cache.get("models") if isinstance(cache, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"invalid models cache in {sv.MODELS_CACHE_PATH}")
    rows: list[dict[str, Any]] = [row for row in models if isinstance(row, dict)]
    if not rows:
        defaults["reasoning_effort"] = None
        return defaults
    if configured_model is not None:
        for row in rows:
            names = {
                sv._clean_optional_text(row.get("slug")),
                sv._clean_optional_text(row.get("display_name")),
            }
            if configured_model in names:
                defaults["reasoning_effort"] = sv._display_reasoning_effort(
                    row.get("default_reasoning_level")
                )
                return defaults
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("priority") if isinstance(row.get("priority"), int) else 999999,
            sv._clean_optional_text(row.get("slug")) or "",
        ),
    )
    defaults["reasoning_effort"] = sv._display_reasoning_effort(
        ranked[0].get("default_reasoning_level")
    )
    return defaults


def normalize_pi_provider_models_snapshot(
    runtime: ServerRuntime, raw: Any
) -> dict[str, Any] | None:
    sv = runtime
    if not isinstance(raw, dict):
        return None
    provider_models_raw = raw.get("provider_models")
    if not isinstance(provider_models_raw, dict):
        return None
    provider_choices_raw = raw.get("provider_choices")
    provider_choices: list[str] = []
    if isinstance(provider_choices_raw, list):
        for value in provider_choices_raw:
            name = sv._clean_optional_text(value)
            if name is None or name in provider_choices:
                continue
            provider_choices.append(name)
    provider_models: dict[str, list[str]] = {}
    for key, value in provider_models_raw.items():
        name = sv._clean_optional_text(key)
        if name is None:
            continue
        model_choices: list[str] = []
        if isinstance(value, list):
            for item in value:
                model_id = sv._clean_optional_text(item)
                if model_id is None or model_id in model_choices:
                    continue
                model_choices.append(model_id)
        provider_models[name] = model_choices
        if name not in provider_choices:
            provider_choices.append(name)
    cached_at = raw.get("cached_at")
    return {
        "provider_choices": provider_choices,
        "provider_models": provider_models,
        "cached_at": float(cached_at) if isinstance(cached_at, (int, float)) else None,
    }


def read_pi_provider_models_snapshot_from_source(
    runtime: ServerRuntime,
) -> dict[str, Any]:
    sv = runtime
    provider_choices: list[str] = []
    provider_models: dict[str, list[str]] = {}
    if sv.PI_MODELS_PATH.exists():
        data = sv.json.loads(sv.PI_MODELS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi models config in {sv.PI_MODELS_PATH}")
        providers = data.get("providers")
        if isinstance(providers, dict):
            for key, value in providers.items():
                name = sv._clean_optional_text(key)
                if name is None or name in provider_choices:
                    continue
                provider_choices.append(name)
                model_choices: list[str] = []
                if isinstance(value, dict):
                    models = value.get("models")
                    if isinstance(models, list):
                        for row in models:
                            if not isinstance(row, dict):
                                continue
                            model_id = sv._clean_optional_text(row.get("id"))
                            if model_id is None or model_id in model_choices:
                                continue
                            model_choices.append(model_id)
                provider_models[name] = model_choices
    return {
        "provider_choices": provider_choices,
        "provider_models": provider_models,
        "cached_at": sv.time.time(),
    }


def read_pi_provider_models_snapshot(
    runtime: ServerRuntime,
    *,
    page_state_db: Any = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    sv = runtime
    if not force_refresh and page_state_db is not None:
        cached = normalize_pi_provider_models_snapshot(
            runtime,
            page_state_db.load_app_kv(sv.PI_MODELS_CACHE_NAMESPACE).get("snapshot"),
        )
        if cached is not None:
            return cached
    snapshot = read_pi_provider_models_snapshot_from_source(runtime)
    if page_state_db is not None:
        page_state_db.save_app_kv(sv.PI_MODELS_CACHE_NAMESPACE, {"snapshot": snapshot})
    return snapshot


def read_pi_launch_defaults(
    runtime: ServerRuntime,
    *,
    page_state_db: Any = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    sv = runtime
    configured_provider: str | None = None
    configured_model: str | None = None
    configured_effort: str | None = "high"

    if sv.PI_SETTINGS_PATH.exists():
        data = sv.json.loads(sv.PI_SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi settings in {sv.PI_SETTINGS_PATH}")
        configured_provider = sv._clean_optional_text(data.get("defaultProvider"))
        configured_model = sv._clean_optional_text(data.get("defaultModel"))

    snapshot = read_pi_provider_models_snapshot(
        runtime,
        page_state_db=page_state_db,
        force_refresh=force_refresh,
    )
    provider_choices = list(snapshot.get("provider_choices") or [])
    provider_models = {
        str(key): [str(item) for item in value]
        for key, value in dict(snapshot.get("provider_models") or {}).items()
        if isinstance(key, str) and isinstance(value, list)
    }

    fallback_provider = next(
        (name for name in provider_choices if provider_models.get(name)),
        provider_choices[0] if provider_choices else None,
    )
    selected_provider = (
        configured_provider if configured_provider in provider_choices else fallback_provider
    )
    selected_models = provider_models.get(selected_provider or "", [])
    selected_model = (
        configured_model
        if configured_model in selected_models
        else (selected_models[0] if selected_models else None)
    )

    return {
        "agent_backend": "pi",
        "model_provider": selected_provider,
        "preferred_auth_method": None,
        "provider_choice": selected_provider,
        "provider_choices": provider_choices,
        "model": selected_model,
        "models": selected_models,
        "provider_models": provider_models,
        "reasoning_effort": configured_effort,
        "reasoning_efforts": list(sv.SUPPORTED_PI_REASONING_EFFORTS),
        "service_tier": None,
        "supports_fast": False,
        "models_cached_at": snapshot.get("cached_at"),
    }


def read_new_session_defaults(
    runtime: ServerRuntime,
    *,
    page_state_db: Any = None,
    refresh_pi_models: bool = False,
) -> dict[str, Any]:
    sv = runtime
    codex = read_codex_launch_defaults(runtime)
    codex["agent_backend"] = "codex"
    codex["provider_choices"] = list(codex.get("model_providers") or [])
    codex["reasoning_efforts"] = list(sv.SUPPORTED_REASONING_EFFORTS)
    codex["supports_fast"] = True
    pi = read_pi_launch_defaults(
        runtime,
        page_state_db=page_state_db,
        force_refresh=refresh_pi_models,
    )
    return {
        "default_backend": sv.DEFAULT_AGENT_BACKEND,
        "backends": {
            "codex": codex,
            "pi": pi,
        },
    }
