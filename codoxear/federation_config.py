"""Federation settings: persistence, precedence, and effective config.

Settings live in ``APP_DIR/federation.json``::

    {
      "hub":  {"enabled": bool, "listen": "0.0.0.0:8744", "token": "..."},
      "node": {"enabled": bool, "hub": "host:8744", "token": "...", "name": "..."}
    }

Precedence is saved file over env vars (CODEX_WEB_FEDERATION_LISTEN /
CODEX_WEB_FEDERATION_HUB / CODEX_WEB_FEDERATION_TOKEN / CODEX_WEB_NODE_NAME):
a section present in the file wins for that section, with individual missing
fields falling back to env. With no file at all, behavior is exactly the
original env-only mode. The token is never logged.
"""

from __future__ import annotations

import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from .federation_hub import ENV_FEDERATION_LISTEN as ENV_LISTEN
from .federation_node import ENV_FEDERATION_HUB as ENV_HUB
from .federation_node import ENV_FEDERATION_TOKEN as ENV_TOKEN
from .federation_node import ENV_NODE_NAME
from .json_state import atomic_write_json, load_json_file

DEFAULT_HUB_LISTEN = "0.0.0.0:8744"

TOKEN_HEX_CHARS = 24


@dataclass(frozen=True)
class EffectiveFederationConfig:
    hub_enabled: bool
    hub_listen: str
    hub_token: str
    node_enabled: bool
    node_hub: str
    node_token: str
    node_name: str
    # "file" when federation.json drove at least one section, else "env".
    source: str


def _clean_text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _clean_section(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    if isinstance(raw.get("enabled"), bool):
        out["enabled"] = raw["enabled"]
    for key in ("listen", "hub", "token", "name"):
        text = _clean_text(raw.get(key))
        if text:
            out[key] = text
    return out


def normalize_federation_settings(raw: Any) -> dict[str, Any]:
    """Validate/trim a settings object into the persisted shape."""
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    hub = _clean_section(raw.get("hub"))
    node = _clean_section(raw.get("node"))
    if hub:
        out["hub"] = hub
    if node:
        out["node"] = node
    return out


def load_federation_settings(path: Path) -> dict[str, Any] | None:
    """Return the saved settings, or None when no file exists.

    A malformed file raises (json.JSONDecodeError) so callers surface the
    problem instead of silently dropping the user's config.
    """
    raw = load_json_file(path, default=None)
    if raw is None:
        return None
    return normalize_federation_settings(raw)


def save_federation_settings(path: Path, settings: dict[str, Any]) -> None:
    atomic_write_json(path, normalize_federation_settings(settings))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def generate_token() -> str:
    """Fresh random token: 24 lowercase hex chars (96 bits)."""
    return secrets.token_hex(TOKEN_HEX_CHARS // 2)


def _default_node_name() -> str:
    return socket.gethostname()


def resolve_effective_config(
    saved: dict[str, Any] | None,
    environ: MutableMapping[str, str] | None = None,
    *,
    default_node_name: str | None = None,
) -> EffectiveFederationConfig:
    """Merge saved settings over env vars into the effective config."""
    env = os.environ if environ is None else environ
    env_listen = _clean_text(env.get(ENV_LISTEN))
    env_hub = _clean_text(env.get(ENV_HUB))
    env_token = _clean_text(env.get(ENV_TOKEN))
    env_node_name = _clean_text(env.get(ENV_NODE_NAME))

    saved = normalize_federation_settings(saved) if saved else {}
    hub_section = saved.get("hub") if isinstance(saved.get("hub"), dict) else None
    node_section = saved.get("node") if isinstance(saved.get("node"), dict) else None

    if hub_section is not None:
        hub_enabled = bool(hub_section.get("enabled", False))
        hub_listen = _clean_text(hub_section.get("listen")) or env_listen or DEFAULT_HUB_LISTEN
        hub_token = _clean_text(hub_section.get("token")) or env_token
    else:
        hub_enabled = bool(env_listen and env_token)
        hub_listen = env_listen or DEFAULT_HUB_LISTEN
        hub_token = env_token

    if node_section is not None:
        node_enabled = bool(node_section.get("enabled", False))
        node_hub = _clean_text(node_section.get("hub")) or env_hub
        node_token = _clean_text(node_section.get("token")) or env_token
        node_name = _clean_text(node_section.get("name")) or env_node_name
    else:
        node_enabled = bool(env_hub and env_token)
        node_hub = env_hub
        node_token = env_token
        node_name = env_node_name

    if not node_name:
        node_name = default_node_name or _default_node_name()

    return EffectiveFederationConfig(
        hub_enabled=hub_enabled,
        hub_listen=hub_listen,
        hub_token=hub_token,
        node_enabled=node_enabled,
        node_hub=node_hub,
        node_token=node_token,
        node_name=node_name,
        source="file" if saved else "env",
    )


def public_config_payload(config: EffectiveFederationConfig) -> dict[str, Any]:
    """The admin-facing config view (tokens included by design: the UI is
    password-protected and admins copy the hub token to nodes)."""
    return {
        "hub": {"enabled": config.hub_enabled, "listen": config.hub_listen, "token": config.hub_token},
        "node": {
            "enabled": config.node_enabled,
            "hub": config.node_hub,
            "token": config.node_token,
            "name": config.node_name,
        },
        "source": config.source,
    }
