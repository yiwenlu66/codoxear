import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codoxear-docker-sandbox"
DOCKERFILE = ROOT / "docker" / "sandbox.Dockerfile"


def _run_preflight(env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the sandbox `preflight` subcommand with a custom environment.

    Uses an isolated HOME so the guard's notion of the "live runtime" points at
    a throwaway directory and the test never touches the real host live state.
    """
    with tempfile.TemporaryDirectory(prefix="codoxear-sandbox-test-") as tmp:
        home = Path(tmp) / "home"
        (home / ".local" / "share").mkdir(parents=True, exist_ok=True)
        clean_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
        }
        clean_env.update(env)
        return subprocess.run(
            ["bash", str(SCRIPT), "preflight"],
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=20,
        )


class TestDockerSandboxSource(unittest.TestCase):





    def test_preflight_passes_with_default_throwaway_root(self) -> None:
        result = _run_preflight({})
        self.assertEqual(
            result.returncode, 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        self.assertContains("preflight ok", result.stdout)

    def test_preflight_refuses_root_inside_live_app_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codoxear-sandbox-test-") as tmp:
            home = Path(tmp) / "home"
            live_app = home / ".local" / "share" / "codoxear"
            live_app.mkdir(parents=True, exist_ok=True)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(home),
                "CODOXEAR_DOCKER_ROOT": str(live_app),
            }
            result = subprocess.run(
                ["bash", str(SCRIPT), "preflight"],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(result.returncode, 2, f"stderr={result.stderr!r}")
        self.assertContains("isolation guard failed", result.stderr)

    def test_preflight_refuses_live_app_dir_env_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codoxear-sandbox-test-") as tmp:
            home = Path(tmp) / "home"
            live_app = home / ".local" / "share" / "codoxear"
            live_app.mkdir(parents=True, exist_ok=True)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(home),
                "CODEX_WEB_APP_DIR": str(live_app),
            }
            result = subprocess.run(
                ["bash", str(SCRIPT), "preflight"],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(result.returncode, 2, f"stderr={result.stderr!r}")
        self.assertContains("CODEX_WEB_APP_DIR", result.stderr)

    def test_preflight_allows_throwaway_app_dir_env_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codoxear-sandbox-test-") as tmp:
            home = Path(tmp) / "home"
            (home / ".local" / "share").mkdir(parents=True, exist_ok=True)
            throwaway = Path(tmp) / "throwaway-app"
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(home),
                "CODEXEAR_APP_DIR": str(throwaway),
                "CODOXEAR_DOCKER_ROOT": str(Path(tmp) / "sandbox-root"),
            }
            result = subprocess.run(
                ["bash", str(SCRIPT), "preflight"],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(result.returncode, 0, f"stderr={result.stderr!r}")
        self.assertContains("preflight ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
