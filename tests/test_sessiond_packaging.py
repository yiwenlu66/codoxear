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
    def test_sessiond_console_script_is_declared(self) -> None:
        data = PYPROJECT.read_text(encoding="utf-8")
        self.assertIn('codoxear-sessiond = "codoxear.sessiond:main"', data)
        if tomllib is not None:
            parsed = tomllib.loads(data)
            self.assertEqual(parsed["project"]["scripts"]["codoxear-sessiond"], "codoxear.sessiond:main")

    def test_readme_documents_installed_sessiond_command(self) -> None:
        data = README.read_text(encoding="utf-8")
        self.assertIn("installs `codoxear-server`, `codoxear-broker`, and `codoxear-sessiond`", data)
        self.assertIn("codoxear-sessiond --cwd /path/to/repo -- codex", data)
        self.assertIn("CODEX_WEB_AGENT_BACKEND=pi codoxear-sessiond --cwd /path/to/repo -- pi", data)
        self.assertIn("CODEX_WEB_AGENT_BACKEND=cc codoxear-sessiond --cwd /path/to/repo -- claude", data)

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
