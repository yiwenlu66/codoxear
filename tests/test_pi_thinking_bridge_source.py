import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "codoxear" / "pi_active_session_bridge.ts"


class TestPiThinkingBridgeSource(unittest.TestCase):
    def test_thinking_command_uses_pi_setter_and_reports_effective_level(self) -> None:
        source = BRIDGE.read_text(encoding="utf-8")

        self.assertIn('pi.registerCommand("thinking"', source)
        self.assertIn("setThinkingLevel(level: ThinkingLevel): void", source)
        self.assertIn("getThinkingLevel(): ThinkingLevel", source)
        self.assertIn("pi.setThinkingLevel(requested as ThinkingLevel)", source)
        self.assertIn("const effective = pi.getThinkingLevel()", source)
        self.assertIn("adjusted for the current model", source)
        self.assertIn("Choose one of: ${THINKING_LEVELS.join", source)
        self.assertIn('ctx.ui.notify(message, "info")', source)

    def test_marker_writes_remain_guarded_and_registered(self) -> None:
        source = BRIDGE.read_text(encoding="utf-8")

        self.assertIn("if (!markerPath) return", source)
        self.assertIn("fs.renameSync(tmp, markerPath)", source)
        self.assertIn('pi.on("session_start"', source)
        self.assertIn('pi.on("session_switch"', source)
        self.assertIn('pi.on("session_fork"', source)


if __name__ == "__main__":
    unittest.main()
