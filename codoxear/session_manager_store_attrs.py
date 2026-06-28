from __future__ import annotations

from typing import Any


def store_backed_attr(store_attr: str) -> property:
    def getter(manager: Any) -> Any:
        return getattr(manager._session_store_for_manager(), store_attr)

    def setter(manager: Any, value: Any) -> None:
        setattr(manager._session_store_for_manager(), store_attr, value)

    return property(getter, setter)
