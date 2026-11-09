from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import codoxear.file_complete as file_complete
from codoxear.file_complete import complete_session_path
from codoxear.file_routes import FileGetRouteDeps
from codoxear.file_routes import handle_file_get_route


def _entry_names(result) -> list[str]:
    return [str(entry["name"]) for entry in result.entries]


def test_completion_resolves_relative_parent_absolute_and_home_prefixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    session = tmp_path / "workspace" / "project"
    parent = session.parent
    for directory in (home, session, parent):
        directory.mkdir(parents=True, exist_ok=True)
    (session / "alpha.txt").write_text("a", encoding="utf-8")
    (parent / "beta.txt").write_text("b", encoding="utf-8")
    (home / "gamma.txt").write_text("c", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    current = complete_session_path(session, token="./al")
    assert current.directory == session.resolve()
    assert current.entries == [{"name": "alpha.txt", "completion": "./alpha.txt", "is_dir": False}]

    parent_result = complete_session_path(session, token="../be")
    assert parent_result.directory == parent.resolve()
    assert parent_result.entries == [{"name": "beta.txt", "completion": "../beta.txt", "is_dir": False}]

    absolute = complete_session_path(session, token=f"{parent}/be")
    assert absolute.directory == parent.resolve()
    assert absolute.entries == [{"name": "beta.txt", "completion": f"{parent}/beta.txt", "is_dir": False}]

    home_result = complete_session_path(session, token="~/ga")
    assert home_result.directory == home.resolve()
    assert home_result.entries == [{"name": "gamma.txt", "completion": "~/gamma.txt", "is_dir": False}]


def test_completion_lists_visible_children_and_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    (tmp_path / "afile.txt").write_text("file", encoding="utf-8")
    (tmp_path / ".secret").write_text("hidden", encoding="utf-8")

    visible = complete_session_path(tmp_path, token="./")
    assert _entry_names(visible) == ["adir", "afile.txt"]
    assert visible.entries[0] == {"name": "adir", "completion": "./adir/", "is_dir": True}
    assert visible.entries[1] == {"name": "afile.txt", "completion": "./afile.txt", "is_dir": False}

    hidden = complete_session_path(tmp_path, token="./.s")
    assert hidden.entries == [{"name": ".secret", "completion": "./.secret", "is_dir": False}]


def test_completion_accepts_a_slash_less_cwd_fragment(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    (tmp_path / "afile.txt").write_text("file", encoding="utf-8")
    (tmp_path / ".secret").write_text("hidden", encoding="utf-8")

    fragment = complete_session_path(tmp_path, token="a")
    assert fragment.directory == tmp_path.resolve()
    assert fragment.entries == [
        {"name": "adir", "completion": "adir/", "is_dir": True},
        {"name": "afile.txt", "completion": "afile.txt", "is_dir": False},
    ]

    # An empty token is the cwd listing itself: the composer sends it for a bare
    # `@` mention before any name character exists.
    listing = complete_session_path(tmp_path, token="")
    assert _entry_names(listing) == ["adir", "afile.txt"]
    assert listing.entries[0]["completion"] == "adir/"
    assert listing.entries[1]["completion"] == "afile.txt"

    assert complete_session_path(tmp_path, token=".s").entries == [
        {"name": ".secret", "completion": ".secret", "is_dir": False},
    ]
    assert complete_session_path(tmp_path, token="qqq").entries == []


def test_completion_preserves_each_typed_prefix_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    parent = session.parent
    home = tmp_path / "home"
    session.mkdir()
    home.mkdir()
    for directory in (session, parent, home):
        (directory / "xray").write_text("x", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    assert complete_session_path(session, token="./x").entries[0]["completion"] == "./xray"
    assert complete_session_path(session, token="../x").entries[0]["completion"] == "../xray"
    assert complete_session_path(session, token=f"{parent}/x").entries[0]["completion"] == f"{parent}/xray"
    assert complete_session_path(session, token="~/x").entries[0]["completion"] == "~/xray"


def test_completion_uses_fuzzy_scores_and_returns_no_match(tmp_path: Path) -> None:
    names = ["apple.txt", "alpaca.md", "snap.log", "zeta"]
    for name in names:
        (tmp_path / name).write_text(name, encoding="utf-8")

    result = complete_session_path(tmp_path, token="./ap")
    expected = sorted(
        (name for name in names if file_complete.file_search_score(name, "ap") >= 0),
        key=lambda name: (-file_complete.file_search_score(name, "ap"), name.casefold()),
    )
    assert _entry_names(result) == expected
    assert complete_session_path(tmp_path, token="./qqq").entries == []


def test_completion_skips_non_utf8_filenames(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("good", encoding="utf-8")
    raw_path = os.fsencode(tmp_path) + b"/bad-\xff.txt"
    descriptor = os.open(raw_path, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(descriptor)

    result = complete_session_path(tmp_path, token="./")
    assert _entry_names(result) == ["good.txt"]


def test_completion_applies_limit_after_sorting_and_reports_truncation(tmp_path: Path) -> None:
    for name in ("z-dir", "a-dir", "b-file", "c-file"):
        path = tmp_path / name
        if name.endswith("dir"):
            path.mkdir()
        else:
            path.write_text(name, encoding="utf-8")

    result = complete_session_path(tmp_path, token="./", limit=2)
    assert _entry_names(result) == ["a-dir", "z-dir"]
    assert result.truncated is True

    minimum = complete_session_path(tmp_path, token="./", limit=0)
    assert _entry_names(minimum) == ["a-dir"]
    assert minimum.truncated is True

    maximum = complete_session_path(tmp_path, token="./", limit=999)
    assert _entry_names(maximum) == ["a-dir", "z-dir", "b-file", "c-file"]
    assert maximum.truncated is False


def test_completion_turns_unreachable_or_invalid_directories_into_empty_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain_file = tmp_path / "plain.txt"
    plain_file.write_text("text", encoding="utf-8")
    assert complete_session_path(tmp_path, token="./missing/").entries == []
    assert complete_session_path(tmp_path, token="./plain.txt/").entries == []

    def denied(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(file_complete.os, "scandir", denied)
    denied_result = complete_session_path(tmp_path, token="./")
    assert denied_result.entries == []
    assert denied_result.truncated is False


class _TimedOutFuture:
    def __init__(self) -> None:
        self.cancelled = False

    def result(self, *, timeout: float):
        assert timeout == file_complete.FILE_COMPLETE_TIMEOUT_SECONDS
        raise FutureTimeoutError()

    def cancel(self) -> None:
        self.cancelled = True


class _TimedOutExecutor:
    def __init__(self) -> None:
        self.future = _TimedOutFuture()

    def submit(self, *_args, **_kwargs):
        return self.future


def test_completion_timeout_returns_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _TimedOutExecutor()
    monkeypatch.setattr(file_complete, "_FILE_COMPLETE_EXECUTOR", executor)
    result = complete_session_path(tmp_path, token="./")
    assert result.entries == []
    assert result.truncated is False
    assert result.directory is None
    assert executor.future.cancelled is True


class _Handler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class _Manager:
    def __init__(self, cwd: Path | None) -> None:
        self.cwd = cwd
        self.refreshed: list[str] = []

    def refresh_session_meta(self, session_id: str) -> None:
        self.refreshed.append(session_id)

    def get_session(self, _session_id: str):
        return None if self.cwd is None else SimpleNamespace(cwd=str(self.cwd))


def _match_session_route(path: str, *parts: str) -> str | None:
    return "s" if path == "/api/sessions/s/" + "/".join(parts) else None


def _route_deps(responses: list[tuple[int, dict[str, object]]]) -> FileGetRouteDeps:
    return FileGetRouteDeps(
        require_auth=lambda _handler: True,
        json_response=lambda _handler, status, payload: responses.append((status, payload)),
        resolve_session_cwd=lambda raw: Path(raw).resolve(),
        resolve_existing_session_file=lambda base, rel: base / rel,
        resolve_session_path=lambda base, rel: base / rel,
        resolve_git_client_file_view=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        resolve_git_existing_regular_file=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        resolve_existing_absolute_file=Path,
        read_client_file_view=lambda _path: (_ for _ in ()).throw(AssertionError("unexpected")),
        read_regular_file_prefix=lambda _path, _count: (_ for _ in ()).throw(AssertionError("unexpected")),
        search_session_relative_files=lambda _base, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        list_session_relative_files=lambda _base: [],
        list_session_relative_file_entries=lambda _base: [],
        file_kind=lambda _path, _prefix: ("text", None),
        ensure_video_preview=lambda path: path,
        ensure_office_preview=lambda path: path,
        inspect_downloadable_file=lambda _path: 0,
        download_disposition=lambda _path: "attachment",
        send_inline_file_response=lambda *_args: None,
        send_attachment_file_response=lambda *_args, **_kwargs: None,
        file_search_limit=120,
    )


def test_complete_route_forwards_the_prefix_only_query_flag(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "xalpha.log").write_text("b", encoding="utf-8")
    responses: list[tuple[int, dict[str, object]]] = []
    seen: list[bool] = []

    def spy(base: Path, *, token: str, limit: int, prefix_only: bool = False):
        seen.append(prefix_only)
        return complete_session_path(base, token=token, limit=limit, prefix_only=prefix_only)

    deps = dataclasses.replace(_route_deps(responses), complete_session_path=spy)
    for query in ("path=al&prefix_only=1", "path=al"):
        handle_file_get_route(
            _Handler(),
            path="/api/sessions/s/file/complete",
            query=query,
            manager=_Manager(tmp_path),
            deps=deps,
            match_session_route=_match_session_route,
        )
    assert seen == [True, False]
    prefix_entry_names = [entry["name"] for entry in responses[0][1]["entries"]]
    fuzzy_entry_names = [entry["name"] for entry in responses[1][1]["entries"]]
    assert prefix_entry_names == ["alpha.txt"]
    assert "xalpha.log" in fuzzy_entry_names


def test_complete_route_returns_session_completion_and_validates_request(tmp_path: Path) -> None:
    (tmp_path / "a-dir").mkdir()
    (tmp_path / "b-file").write_text("b", encoding="utf-8")
    responses: list[tuple[int, dict[str, object]]] = []
    deps = _route_deps(responses)
    handler = _Handler()

    handled = handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="path=./&limit=1",
        manager=_Manager(tmp_path),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [
        (
            200,
            {
                "ok": True,
                "dir": str(tmp_path.resolve()),
                "entries": [{"name": "a-dir", "completion": "./a-dir/", "is_dir": True}],
                "truncated": True,
            },
        )
    ]

    responses.clear()
    handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="path=filename&limit=not-a-number",
        manager=_Manager(tmp_path),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert responses == [(400, {"error": "limit must be an integer"})]

    # A slash-less token is a fragment in the session cwd (the composer's `@`
    # mention sends exactly this form), and an empty token lists the cwd itself.
    responses.clear()
    handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="path=b-f",
        manager=_Manager(tmp_path),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert responses == [
        (
            200,
            {
                "ok": True,
                "dir": str(tmp_path.resolve()),
                "entries": [{"name": "b-file", "completion": "b-file", "is_dir": False}],
                "truncated": False,
            },
        )
    ]

    responses.clear()
    handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="path=&limit=1",
        manager=_Manager(tmp_path),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert responses == [
        (
            200,
            {
                "ok": True,
                "dir": str(tmp_path.resolve()),
                "entries": [{"name": "a-dir", "completion": "a-dir/", "is_dir": True}],
                "truncated": True,
            },
        )
    ]

    responses.clear()
    handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="limit=50",
        manager=_Manager(tmp_path),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert responses == [(400, {"error": "path required"})]

    responses.clear()
    handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="path=./&limit=not-a-number",
        manager=_Manager(tmp_path),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert responses == [(400, {"error": "limit must be an integer"})]

    responses.clear()
    handle_file_get_route(
        handler,
        path="/api/sessions/s/file/complete",
        query="path=./",
        manager=_Manager(None),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert responses == [(404, {"error": "unknown session"})]


def test_prefix_only_completion_returns_only_names_sharing_the_fragment(tmp_path: Path) -> None:
    names = ["alpha.txt", "alpine.md", "xalpha.log", "beta.txt"]
    for name in names:
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "alder").mkdir()

    fuzzy = complete_session_path(tmp_path, token="al")
    assert "xalpha.log" in _entry_names(fuzzy)

    prefix = complete_session_path(tmp_path, token="al", prefix_only=True)
    # Prefix mode answers the tab-completion question: the complete set of names
    # that start with the fragment, dirs first, alphabetical.
    assert _entry_names(prefix) == ["alder", "alpha.txt", "alpine.md"]
    assert prefix.entries[0] == {"name": "alder", "completion": "alder/", "is_dir": True}
    assert prefix.truncated is False

    assert complete_session_path(tmp_path, token="qqq", prefix_only=True).entries == []


def test_prefix_only_completion_truncates_over_matches_not_over_fuzzy_score(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"app-{index}.js").write_text("x", encoding="utf-8")
    for index in range(5):
        (tmp_path / f"other-{index}.js").write_text("x", encoding="utf-8")

    capped = complete_session_path(tmp_path, token="app", prefix_only=True, limit=2)
    assert _entry_names(capped) == ["app-0.js", "app-1.js"]
    assert capped.truncated is True

    whole = complete_session_path(tmp_path, token="app", prefix_only=True, limit=999)
    assert _entry_names(whole) == [f"app-{index}.js" for index in range(5)]
    assert whole.truncated is False


def test_server_adapter_forwards_the_prefix_only_flag(tmp_path: Path) -> None:
    # The route reaches the completer through codoxear.server's adapter, so the
    # flag must survive that hop: a wrapper that drops it silently turns Tab
    # completion back into fuzzy matching (or raises).
    from codoxear.server import _complete_session_path

    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "xalpha.log").write_text("b", encoding="utf-8")

    forwarded = _complete_session_path(tmp_path, token="al", limit=20, prefix_only=True)
    assert _entry_names(forwarded) == ["alpha.txt"]

    fuzzy = _complete_session_path(tmp_path, token="al", limit=20)
    assert "xalpha.log" in _entry_names(fuzzy)


def test_completion_never_suggests_build_or_tooling_artifacts(tmp_path: Path) -> None:
    """`__pycache__`, `node_modules`, `.git` are noise, not candidates.

    Reported from use: a cwd/completion list that offers them wastes attention on
    names nobody picks. Exact names only, so a real project called e.g. `venvs`
    or `targets` is untouched.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "venvs").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "app.py").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__" / "app.cpython-311.pyc").write_text("x", encoding="utf-8")

    listing = _entry_names(complete_session_path(tmp_path, token="./", limit=50))
    assert "src" in listing and "venvs" in listing and "app.py" in listing
    assert "__pycache__" not in listing
    assert "node_modules" not in listing
    assert ".git" not in listing

    # Even asked for directly, artifact names are not suggested.
    assert complete_session_path(tmp_path, token="./__p", prefix_only=True).entries == []
    assert _entry_names(complete_session_path(tmp_path, token="./node", prefix_only=True)) == []
    # …and their contents are not listed either.
    assert complete_session_path(tmp_path, token="./__pycache__/").entries == []
