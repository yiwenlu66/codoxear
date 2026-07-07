from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TokenObservationKind = Literal["none", "update", "clear"]


@dataclass(frozen=True)
class TokenObservation:
    """Internal token signal.

    Public API token fields remain only dict-or-null.  This object exists so
    log readers can distinguish an absent token observation from a newer
    observation that explicitly clears stale token pressure.
    """

    kind: TokenObservationKind
    token: dict[str, Any] | None = None

    @property
    def observed(self) -> bool:
        return self.kind != "none"

    @property
    def public_token(self) -> dict[str, Any] | None:
        return self.token if self.kind == "update" else None


TOKEN_NONE = TokenObservation("none")
TOKEN_CLEAR = TokenObservation("clear")


def token_update_observation(token: dict[str, Any]) -> TokenObservation:
    return TokenObservation("update", token)


def coerce_token_observation(value: TokenObservation | dict[str, Any] | None) -> TokenObservation:
    if isinstance(value, TokenObservation):
        return value
    if isinstance(value, dict):
        return token_update_observation(value)
    return TOKEN_NONE
