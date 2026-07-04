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
    def test_usage_lists_supported_commands(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "Usage: scripts/codoxear-docker-sandbox [build|smoke|start|stop|logs|preflight|test]",
            source,
        )
        for command in ("build", "smoke", "start", "stop", "logs", "preflight", "test"):
            self.assertIn(f"  {command})", source)

    def test_video_transcoding_dependency_is_available_in_sandbox(self) -> None:
        source = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("ffmpeg", source)

    def test_isolation_guard_function_exists(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ensure_isolation()", source)
        # Guard must run before any directory creation in start/unit paths.
        self.assertIn("ensure_safe_port\n  ensure_isolation\n  prepare_root", source)
        # Guard must cover both current and legacy live app dirs.
        self.assertIn("codoxear", source)
        self.assertIn("codex-web", source)
        # Guard must check both app-dir override env names.
        self.assertIn("CODEXEAR_APP_DIR", source)
        self.assertIn("CODEX_WEB_APP_DIR", source)

    def test_container_does_not_forward_app_dir_env(self) -> None:
        """The container launch must not forward app-dir overrides, so APP_DIR
        always resolves under the throwaway container HOME."""
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("-e CODEXEAR_APP_DIR", source)
        self.assertNotIn("-e CODEX_WEB_APP_DIR", source)

    def test_dockerfile_does_not_pin_app_dir(self) -> None:
        source = DOCKERFILE.read_text(encoding="utf-8")
        self.assertNotIn("CODEXEAR_APP_DIR", source)
        self.assertNotIn("CODEX_WEB_APP_DIR", source)

    def test_preflight_passes_with_default_throwaway_root(self) -> None:
        result = _run_preflight({})
        self.assertEqual(
            result.returncode, 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        self.assertIn("preflight ok", result.stdout)

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
        self.assertIn("isolation guard failed", result.stderr)

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
        self.assertIn("CODEX_WEB_APP_DIR", result.stderr)

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
        self.assertIn("preflight ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
