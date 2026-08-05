from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts" / "check_js_refs.py"
SPEC = importlib.util.spec_from_file_location("check_js_refs", CHECKER_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def _write_static_shell(tmp_path: Path, app_source: str) -> Path:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "app.js").write_text(app_source)
    (static_dir / "index.html").write_text("<!doctype html><script src=\"app.js\"></script>")
    return static_dir


def test_reference_checker_accepts_destructured_const_bindings(tmp_path: Path) -> None:
    static_dir = _write_static_shell(
        tmp_path,
        """
        function boot(controller) {
            const { renderLogin, renderShell: renderApp, retry = renderLogin, ...actions } = controller;
            renderLogin(renderApp);
            retry();
            actions();
        }
        """,
    )

    assert checker.undefined_call_references(
        static_dir / "app.js", static_dir, static_dir / "index.html"
    ) == []


def test_reference_checker_keeps_unrelated_calls_undefined(tmp_path: Path) -> None:
    static_dir = _write_static_shell(
        tmp_path,
        """
        function boot(controller) {
            const { renderLogin } = controller;
            renderLogin();
            missingRenderer();
        }
        """,
    )

    undefined = checker.undefined_call_references(static_dir / "app.js", static_dir, static_dir / "index.html")

    assert [name for name, _ in undefined] == ["missingRenderer"]


def test_reference_checker_accepts_the_app_shell_controller_bindings() -> None:
    static_dir = ROOT / "codoxear" / "static"

    assert checker.undefined_call_references(
        static_dir / "app.js", static_dir, static_dir / "index.html"
    ) == []
