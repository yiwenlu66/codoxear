from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .agent_backend import get_agent_backend
from .agent_backend import normalize_agent_backend
from .cc_log import CC_SUPPORTED_REASONING_EFFORTS

SUPPORTED_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
SUPPORTED_PI_REASONING_EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
SUPPORTED_CC_REASONING_EFFORTS = CC_SUPPORTED_REASONING_EFFORTS


@dataclass(frozen=True)
class LaunchConfigPaths:
    codex_config_path: Path
    models_cache_path: Path
    pi_settings_path: Path
    pi_models_path: Path
    pi_auth_path: Path
    cc_settings_path: Path


class LaunchRequestValidationError(ValueError):
    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class NewSessionLaunchRequest:
    cwd: str
    args: list[str] | None
    agent_backend: str
    resume_session_id: str | None
    worktree_branch: str | None
    model_provider: str | None
    preferred_auth_method: str | None
    model: str | None
    reasoning_effort: str | None
    service_tier: str | None
    create_in_tmux: bool


def clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def normalize_requested_model(value: Any) -> str | None:
    out = clean_optional_text(value)
    if out is None:
        return None
    return None if out.lower() == "default" else out


def display_reasoning_effort(value: Any) -> str | None:
    out = clean_optional_text(value)
    if out is None:
        return None
    lowered = out.lower()
    return lowered if lowered in SUPPORTED_REASONING_EFFORTS else None


def display_pi_reasoning_effort(value: Any) -> str | None:
    out = clean_optional_text(value)
    if out is None:
        return None
    lowered = out.lower()
    return lowered if lowered in SUPPORTED_PI_REASONING_EFFORTS else None


def normalize_requested_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string")
    out = value.strip().lower()
    if not out:
        return None
    if out not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {', '.join(SUPPORTED_REASONING_EFFORTS)}")
    return out


def clean_reasoning_effort_list(raw: Any, *, supported: tuple[str, ...]) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        value = item.strip().lower()
        if not value or value not in supported or value in out:
            continue
        out.append(value)
    return out or None


def pi_reasoning_efforts_for_model_row(row: dict[str, Any]) -> list[str] | None:
    explicit = (
        clean_reasoning_effort_list(row.get("reasoning_efforts"), supported=SUPPORTED_PI_REASONING_EFFORTS)
        or clean_reasoning_effort_list(row.get("reasoningEfforts"), supported=SUPPORTED_PI_REASONING_EFFORTS)
        or clean_reasoning_effort_list(row.get("thinking_efforts"), supported=SUPPORTED_PI_REASONING_EFFORTS)
        or clean_reasoning_effort_list(row.get("thinkingEfforts"), supported=SUPPORTED_PI_REASONING_EFFORTS)
    )
    if explicit is not None:
        return explicit
    reasoning = row.get("reasoning")
    if reasoning is False:
        return ["off"]
    if reasoning is True:
        return list(SUPPORTED_PI_REASONING_EFFORTS)
    return None


def pi_reasoning_effort_key(provider: str | None, model: str | None) -> str | None:
    model_clean = clean_optional_text(model)
    if model_clean is None:
        return None
    provider_clean = clean_optional_text(provider)
    return f"{provider_clean}/{model_clean}" if provider_clean else model_clean


def read_pi_reasoning_efforts_by_model(paths: LaunchConfigPaths) -> dict[str, list[str]]:
    if not paths.pi_models_path.exists():
        return {}
    data = json.loads(paths.pi_models_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid Pi models config in {paths.pi_models_path}")
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return {}
    out: dict[str, list[str]] = {}
    for provider, value in providers.items():
        provider_name = provider.strip() if isinstance(provider, str) else ""
        if not provider_name or not isinstance(value, dict):
            continue
        models = value.get("models")
        if not isinstance(models, list):
            continue
        for row in models:
            if not isinstance(row, dict):
                continue
            model_id = clean_optional_text(row.get("id"))
            if model_id is None:
                continue
            efforts = pi_reasoning_efforts_for_model_row(row)
            if efforts is None:
                continue
            out.setdefault(model_id, list(efforts))
            out[f"{provider_name}/{model_id}"] = list(efforts)
    return out


def pi_allowed_reasoning_efforts_for_model(
    *,
    model_provider: str | None,
    model: str | None,
    reasoning_efforts_by_model: Mapping[str, list[str]] | None = None,
    paths: LaunchConfigPaths | None = None,
) -> list[str] | None:
    if reasoning_efforts_by_model is not None:
        mapping = reasoning_efforts_by_model
    elif paths is not None:
        mapping = read_pi_reasoning_efforts_by_model(paths)
    else:
        mapping = {}
    provider_clean = clean_optional_text(model_provider)
    key = pi_reasoning_effort_key(model_provider, model)
    if key and key in mapping:
        return list(mapping[key])
    if provider_clean:
        return None
    model_clean = clean_optional_text(model)
    if model_clean and model_clean in mapping:
        return list(mapping[model_clean])
    return None


def normalize_requested_pi_reasoning_effort(
    value: Any,
    *,
    model_provider: str | None = None,
    model: str | None = None,
    reasoning_efforts_by_model: Mapping[str, list[str]] | None = None,
    paths: LaunchConfigPaths | None = None,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string")
    out = value.strip().lower()
    if not out:
        return None
    allowed = pi_allowed_reasoning_efforts_for_model(
        model_provider=model_provider,
        model=model,
        reasoning_efforts_by_model=reasoning_efforts_by_model,
        paths=paths,
    ) or list(SUPPORTED_PI_REASONING_EFFORTS)
    if out not in allowed:
        model_label = clean_optional_text(model) or "selected model"
        raise ValueError(f"reasoning_effort must be one of {', '.join(allowed)} for Pi model {model_label}")
    return out


def normalize_requested_cc_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string")
    out = value.strip().lower()
    if not out:
        return None
    if out not in SUPPORTED_CC_REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {', '.join(SUPPORTED_CC_REASONING_EFFORTS)}")
    return out


def normalize_requested_model_provider(value: Any, *, allowed: set[str] | None = None) -> str | None:
    provider = clean_optional_text(value)
    if provider is None:
        return None
    if allowed is not None and provider not in allowed:
        allowed_txt = ", ".join(sorted(allowed))
        raise ValueError(f"model_provider must be one of {allowed_txt}")
    return provider


def normalize_requested_service_tier(value: Any) -> str | None:
    tier = clean_optional_text(value)
    if tier is None:
        return None
    if tier not in {"fast", "flex"}:
        raise ValueError("service_tier must be one of fast, flex")
    return tier


def normalize_requested_preferred_auth_method(value: Any) -> str | None:
    method = clean_optional_text(value)
    if method is None:
        return None
    if method not in {"chatgpt", "apikey"}:
        raise ValueError("preferred_auth_method must be one of chatgpt, apikey")
    return method


def configured_model_providers(data: dict[str, Any]) -> list[str]:
    providers = ["openai"]
    seen = {"openai"}
    raw = data.get("model_providers")
    if not isinstance(raw, dict):
        return providers
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        name = key.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        providers.append(name)
    return providers


def provider_models_from_config(data: dict[str, Any]) -> list[str]:
    """Extract per-provider model lists declared in config.toml model_providers."""
    raw = data.get("model_providers")
    if not isinstance(raw, dict):
        return []
    models: list[str] = []
    seen: set[str] = set()
    for value in raw.values():
        if not isinstance(value, dict):
            continue
        for model in value.get("models", []):
            name = clean_optional_text(model)
            if name and name not in seen:
                seen.add(name)
                models.append(name)
    return models


def provider_choice_for_settings(*, model_provider: str | None, preferred_auth_method: str | None) -> str:
    provider = model_provider or "openai"
    if provider == "openai":
        return "chatgpt" if preferred_auth_method == "chatgpt" else "openai-api"
    return provider


def fallback_codex_launch_defaults() -> dict[str, Any]:
    return {
        "model_provider": "openai",
        "preferred_auth_method": "apikey",
        "provider_choice": "openai-api",
        "model": None,
        "model_providers": ["chatgpt", "openai-api"],
        "service_tier": "flex",
        "reasoning_effort": None,
    }


def fallback_pi_launch_defaults() -> dict[str, Any]:
    return {
        "agent_backend": "pi",
        "model_provider": None,
        "preferred_auth_method": None,
        "provider_choice": None,
        "provider_choices": [],
        "model": None,
        "models": [],
        "reasoning_effort": "high",
        "reasoning_efforts": list(SUPPORTED_PI_REASONING_EFFORTS),
        "reasoning_efforts_by_model": {},
        "service_tier": None,
        "supports_fast": False,
    }


def fallback_cc_launch_defaults() -> dict[str, Any]:
    return {
        "agent_backend": "cc",
        "model_provider": None,
        "preferred_auth_method": None,
        "provider_choice": None,
        "provider_choices": [],
        "model": None,
        "models": ["sonnet", "opus", "fable"],
        "reasoning_effort": "medium",
        "reasoning_efforts": list(SUPPORTED_CC_REASONING_EFFORTS),
        "service_tier": None,
        "supports_fast": False,
    }


def read_codex_launch_defaults(paths: LaunchConfigPaths) -> dict[str, Any]:
    configured_model = None
    configured_effort = None
    configured_provider = "openai"
    configured_auth_method = "apikey"
    configured_service_tier = "flex"
    configured_providers = ["chatgpt", "openai-api"]
    provider_models: list[str] = []
    if paths.codex_config_path.exists():
        data = tomllib.loads(paths.codex_config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Codex config in {paths.codex_config_path}")
        configured_model = clean_optional_text(data.get("model"))
        configured_effort = display_reasoning_effort(data.get("model_reasoning_effort"))
        configured_auth_method = normalize_requested_preferred_auth_method(data.get("preferred_auth_method")) or configured_auth_method
        configured_providers = ["chatgpt", "openai-api", *[p for p in configured_model_providers(data) if p != "openai"]]
        configured_provider = normalize_requested_model_provider(
            data.get("model_provider") or data.get("model_provider_id"),
            allowed=set(["openai", *[p for p in configured_providers if p not in {"chatgpt", "openai-api"}]]),
        ) or configured_provider
        configured_service_tier = normalize_requested_service_tier(data.get("service_tier")) or configured_service_tier
        provider_models = provider_models_from_config(data)
    defaults: dict[str, Any] = {
        "model_provider": configured_provider,
        "preferred_auth_method": configured_auth_method,
        "provider_choice": provider_choice_for_settings(model_provider=configured_provider, preferred_auth_method=configured_auth_method),
        "model": configured_model,
        "model_providers": configured_providers,
        "service_tier": configured_service_tier,
    }
    if provider_models:
        defaults["models"] = provider_models
    if configured_effort is not None:
        defaults["reasoning_effort"] = configured_effort
        return defaults
    if not paths.models_cache_path.exists():
        defaults["reasoning_effort"] = None
        return defaults
    cache = json.loads(paths.models_cache_path.read_text(encoding="utf-8"))
    models = cache.get("models") if isinstance(cache, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"invalid models cache in {paths.models_cache_path}")
    rows: list[dict[str, Any]] = [row for row in models if isinstance(row, dict)]
    if not rows:
        defaults["reasoning_effort"] = None
        return defaults
    if configured_model is not None:
        for row in rows:
            names = {clean_optional_text(row.get("slug")), clean_optional_text(row.get("display_name"))}
            if configured_model in names:
                defaults["reasoning_effort"] = display_reasoning_effort(row.get("default_reasoning_level"))
                return defaults
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row.get("priority")) if isinstance(row.get("priority"), int) else 999999,
            clean_optional_text(row.get("slug")) or "",
        ),
    )
    defaults["reasoning_effort"] = display_reasoning_effort(ranked[0].get("default_reasoning_level"))
    return defaults


def read_pi_launch_defaults(paths: LaunchConfigPaths) -> dict[str, Any]:
    configured_provider: str | None = None
    configured_model: str | None = None
    configured_effort: str | None = "high"
    provider_choices: list[str] = []
    model_choices: list[str] = []
    provider_models: dict[str, list[str]] = {}
    reasoning_efforts_by_model = read_pi_reasoning_efforts_by_model(paths)

    if paths.pi_settings_path.exists():
        data = json.loads(paths.pi_settings_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi settings in {paths.pi_settings_path}")
        configured_provider = clean_optional_text(data.get("defaultProvider"))
        configured_model = clean_optional_text(data.get("defaultModel"))

    if paths.pi_models_path.exists():
        data = json.loads(paths.pi_models_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi models config in {paths.pi_models_path}")
        providers = data.get("providers")
        if isinstance(providers, dict):
            for key, value in providers.items():
                if not isinstance(key, str):
                    continue
                name = key.strip()
                if not name or name in provider_choices:
                    continue
                provider_choices.append(name)
                provider_models[name] = []
                if not isinstance(value, dict):
                    continue
                models = value.get("models")
                if not isinstance(models, list):
                    continue
                for row in models:
                    if not isinstance(row, dict):
                        continue
                    model_id = clean_optional_text(row.get("id"))
                    if model_id is None or model_id in model_choices:
                        continue
                    model_choices.append(model_id)
                    provider_models[name].append(model_id)

    if paths.pi_auth_path.exists():
        data = json.loads(paths.pi_auth_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Pi auth config in {paths.pi_auth_path}")
        for key, value in data.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, dict):
                continue
            auth_type = clean_optional_text(value.get("type"))
            access = clean_optional_text(value.get("access"))
            refresh = clean_optional_text(value.get("refresh"))
            if auth_type != "oauth":
                continue
            if access is None and refresh is None:
                continue
            name = key.strip()
            if name not in provider_choices:
                provider_choices.append(name)

    if configured_provider is not None and configured_provider not in provider_choices:
        provider_choices.insert(0, configured_provider)
    if configured_model is not None and configured_model not in model_choices:
        model_choices.insert(0, configured_model)
    configured_efforts = pi_allowed_reasoning_efforts_for_model(
        model_provider=configured_provider,
        model=configured_model,
        reasoning_efforts_by_model=reasoning_efforts_by_model,
    )
    if configured_efforts is None:
        configured_efforts = list(SUPPORTED_PI_REASONING_EFFORTS)
    if configured_effort not in configured_efforts:
        configured_effort = configured_efforts[0] if configured_efforts else None

    return {
        "agent_backend": "pi",
        "model_provider": configured_provider,
        "preferred_auth_method": None,
        "provider_choice": configured_provider,
        "provider_choices": provider_choices,
        "model": configured_model,
        "models": model_choices,
        "provider_models": provider_models,
        "reasoning_effort": configured_effort,
        "reasoning_efforts": configured_efforts,
        "reasoning_efforts_by_model": reasoning_efforts_by_model,
        "service_tier": None,
        "supports_fast": False,
    }


def read_cc_launch_defaults(paths: LaunchConfigPaths) -> dict[str, Any]:
    configured_model: str | None = None
    configured_effort: str | None = "medium"
    if paths.cc_settings_path.exists():
        data = json.loads(paths.cc_settings_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid Claude Code settings in {paths.cc_settings_path}")
        configured_model = clean_optional_text(data.get("model")) or clean_optional_text(data.get("defaultModel"))
        configured_effort = normalize_requested_cc_reasoning_effort(data.get("effortLevel") or data.get("effort") or data.get("thinkingLevel")) or configured_effort
    return {
        "agent_backend": "cc",
        "model_provider": None,
        "preferred_auth_method": None,
        "provider_choice": None,
        "provider_choices": [],
        "model": configured_model,
        "models": [m for m in [configured_model, "sonnet", "opus", "fable"] if isinstance(m, str) and m],
        "reasoning_effort": configured_effort,
        "reasoning_efforts": list(SUPPORTED_CC_REASONING_EFFORTS),
        "service_tier": None,
        "supports_fast": False,
    }


def launch_defaults_warning(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def read_new_session_defaults(paths: LaunchConfigPaths, *, default_agent_backend: str) -> dict[str, Any]:
    warnings: dict[str, str] = {}
    readers = {
        "codex": read_codex_launch_defaults,
        "pi": read_pi_launch_defaults,
        "cc": read_cc_launch_defaults,
    }
    fallbacks = {
        "codex": fallback_codex_launch_defaults,
        "pi": fallback_pi_launch_defaults,
        "cc": fallback_cc_launch_defaults,
    }
    reasoning_efforts = {
        "codex": SUPPORTED_REASONING_EFFORTS,
        "pi": SUPPORTED_PI_REASONING_EFFORTS,
        "cc": SUPPORTED_CC_REASONING_EFFORTS,
    }
    backends: dict[str, dict[str, Any]] = {}
    for backend_name in ("codex", "pi", "cc"):
        try:
            defaults = readers[backend_name](paths)
        except Exception as exc:
            defaults = fallbacks[backend_name]()
            warnings[backend_name] = launch_defaults_warning(exc)
        backends[backend_name] = get_agent_backend(backend_name).project_launch_defaults(
            defaults,
            reasoning_efforts=reasoning_efforts[backend_name],
        )
    out = {
        "default_backend": default_agent_backend,
        "backends": backends,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def parse_new_session_launch_request(
    obj: dict[str, Any],
    *,
    default_agent_backend: str,
    codex_launch_defaults_provider: Callable[[], dict[str, Any]],
    pi_launch_defaults_provider: Callable[[], dict[str, Any]],
) -> NewSessionLaunchRequest:
    try:
        agent_backend = normalize_agent_backend(obj.get("agent_backend"), default=default_agent_backend)
    except ValueError as e:
        raise LaunchRequestValidationError(str(e)) from e
    cwd = obj.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip():
        raise LaunchRequestValidationError("cwd required", field="cwd")
    model = normalize_requested_model(obj.get("model"))
    options = get_agent_backend(agent_backend).normalize_launch_request_options(
        obj,
        model=model,
        validation_error_type=LaunchRequestValidationError,
        normalize_model_provider=normalize_requested_model_provider,
        normalize_preferred_auth_method=normalize_requested_preferred_auth_method,
        normalize_reasoning_effort=normalize_requested_reasoning_effort,
        normalize_pi_reasoning_effort=normalize_requested_pi_reasoning_effort,
        normalize_cc_reasoning_effort=normalize_requested_cc_reasoning_effort,
        normalize_service_tier=normalize_requested_service_tier,
        codex_launch_defaults_provider=codex_launch_defaults_provider,
        pi_launch_defaults_provider=pi_launch_defaults_provider,
    )
    model_provider = options["model_provider"]
    preferred_auth_method = options["preferred_auth_method"]
    reasoning_effort = options["reasoning_effort"]
    service_tier = options["service_tier"]

    create_in_tmux_raw = obj.get("create_in_tmux")
    if create_in_tmux_raw is None:
        create_in_tmux = False
    elif isinstance(create_in_tmux_raw, bool):
        create_in_tmux = create_in_tmux_raw
    else:
        raise LaunchRequestValidationError("create_in_tmux must be a boolean")
    resume_session_id_raw = obj.get("resume_session_id")
    if resume_session_id_raw is None:
        resume_session_id = None
    elif isinstance(resume_session_id_raw, str):
        resume_session_id = resume_session_id_raw.strip() or None
    else:
        raise LaunchRequestValidationError("resume_session_id must be a string")
    worktree_branch_raw = obj.get("worktree_branch")
    if worktree_branch_raw is None:
        worktree_branch = None
    elif isinstance(worktree_branch_raw, str):
        worktree_branch = worktree_branch_raw.strip() or None
    else:
        raise LaunchRequestValidationError("worktree_branch must be a string")
    args = obj.get("args")
    if args is None:
        args_list = None
    elif isinstance(args, list) and all(isinstance(x, str) for x in args):
        args_list = [x for x in args if x]
    else:
        raise LaunchRequestValidationError("args must be a list of strings")
    return NewSessionLaunchRequest(
        cwd=cwd,
        args=args_list,
        agent_backend=agent_backend,
        resume_session_id=resume_session_id,
        worktree_branch=worktree_branch,
        model_provider=model_provider,
        preferred_auth_method=preferred_auth_method,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        create_in_tmux=create_in_tmux,
    )
