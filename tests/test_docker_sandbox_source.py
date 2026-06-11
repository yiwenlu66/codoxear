import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "codoxear-docker-sandbox"


class TestDockerSandboxSource(unittest.TestCase):
    def test_usage_lists_supported_commands(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Usage: scripts/codoxear-docker-sandbox [build|smoke|start|stop|logs|test]", source)
        for command in ("build", "smoke", "start", "stop", "logs", "test"):
            self.assertIn(f"  {command})", source)


if __name__ == "__main__":
    unittest.main()
