import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codoxear-docker-sandbox"
DOCKERFILE = ROOT / "docker" / "sandbox.Dockerfile"


class TestDockerSandboxSource(unittest.TestCase):
    def test_usage_lists_supported_commands(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Usage: scripts/codoxear-docker-sandbox [build|smoke|start|stop|logs|test]", source)
        for command in ("build", "smoke", "start", "stop", "logs", "test"):
            self.assertIn(f"  {command})", source)

    def test_video_transcoding_dependency_is_available_in_sandbox(self) -> None:
        source = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("ffmpeg", source)


if __name__ == "__main__":
    unittest.main()
