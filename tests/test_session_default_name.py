from __future__ import annotations

from pathlib import Path

from codoxear.session_default_name import default_session_name


def test_the_default_name_is_the_working_directory_name() -> None:
    assert default_session_name("/home/tester/codoxear", []) == "codoxear"
    assert default_session_name("/home/tester/codoxear/", []) == "codoxear"
    assert default_session_name("~/nested work/quick-07", []) == "quick-07"


def test_a_name_another_live_session_holds_is_numbered() -> None:
    assert default_session_name("/a/repo", ["repo"]) == "repo-2"
    assert default_session_name("/a/repo", ["repo", "repo-2"]) == "repo-3"
    # Gaps are filled, so names stay dense as sessions come and go.
    assert default_session_name("/a/repo", ["repo", "repo-3"]) == "repo-2"


def test_numbering_ignores_case_and_other_basenames() -> None:
    assert default_session_name("/a/Repo", ["repo"]) == "Repo-2"
    assert default_session_name("/a/repo", ["someone else's name", ""]) == "repo"


def test_an_unknown_cwd_yields_no_name() -> None:
    assert default_session_name("", []) == ""
    assert default_session_name("/", []) == ""
    assert default_session_name(None, []) == ""


def test_a_quick_workspace_uses_the_label_its_launch_already_declares() -> None:
    quick = str(Path.home() / "codoxear-quick" / "quick-07")
    assert default_session_name(quick, []) == "Quick Session 07"
    assert default_session_name(quick, ["Quick Session 07"]) == "Quick Session 07-2"


def test_a_quick_like_directory_outside_the_root_is_just_its_own_name() -> None:
    assert default_session_name("/a/repo/quick-07", []) == "quick-07"
