import subprocess
import sys
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"


class TestSessiondPackaging(unittest.TestCase):
    def test_sessiond_module_help_does_not_launch_backend(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "codoxear.sessiond", "--help"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        self.assertIn("--cwd", proc.stdout)
        self.assertIn("usage:", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
