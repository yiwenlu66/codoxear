from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_pytest_imports_current_checkout_by_default() -> None:
    source = PYPROJECT.read_text(encoding="utf-8")
    assert "[tool.pytest.ini_options]" in source
    assert 'pythonpath = ["."]' in source
