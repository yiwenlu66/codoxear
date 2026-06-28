from __future__ import annotations

from typing import Any


def store_backed_attr(store_attr: str) -> property:
    def getter(manager: Any) -> Any:
        return getattr(manager._session_store_for_manager(), store_attr)

    def setter(manager: Any, value: Any) -> None:
        setattr(manager._session_store_for_manager(), store_attr, value)

    return property(getter, setter)


def load_store_attr(manager_attr: str, loader_name: str) -> Any:
    def load(manager: Any) -> None:
        cleaned = getattr(manager._session_store_for_manager(), loader_name)()
        with manager._lock:
            setattr(manager, manager_attr, cleaned)

    return load


def save_dict_store_attr(manager_attr: str, saver_name: str) -> Any:
    def save(manager: Any) -> None:
        with manager._lock:
            obj = dict(getattr(manager, manager_attr))
        getattr(manager._session_store_for_manager(), saver_name)(obj)

    return save


def save_set_store_attr(manager_attr: str, saver_name: str) -> Any:
    def save(manager: Any) -> None:
        with manager._lock:
            obj = set(getattr(manager, manager_attr, set()))
        getattr(manager._session_store_for_manager(), saver_name)(obj)

    return save


def save_pending_attachment_ids_attr(manager_attr: str, saver_name: str) -> Any:
    def save(manager: Any) -> None:
        with manager._lock:
            ids = set(str(item) for item in getattr(manager, manager_attr, set()) if str(item).strip())
        getattr(manager._session_store_for_manager(), saver_name)(ids)

    return save
