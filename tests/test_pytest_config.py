import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def test_pytest_imports_current_checkout_by_default() -> None:
    """A clean pytest process resolves the checkout package through its config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        probe_test = Path(temp_dir) / "test_probe.py"
        probe_test.write_text(
            "from pathlib import Path\n"
            "import codoxear\n\n"
            "def test_imports_current_checkout():\n"
            f"    assert Path(codoxear.__file__).resolve().parent.parent == Path({str(ROOT)!r})\n",
            encoding="utf-8",
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTEST_ADDOPTS"}
        }
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-c", str(PYPROJECT), str(probe_test)],
            cwd=temp_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
