import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_LAUNCH_PY = ROOT / "codoxear" / "broker_launch.py"


class TestBrokerLaunchSource(unittest.TestCase):
    def test_launch_helper_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        launch_tree = ast.parse(BROKER_LAUNCH_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        launch_defs = {node.name for node in launch_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        moved_names = {
            "_session_log_path_from_args",
            "_pi_session_dir_name",
            "_pi_session_dir_from_args",
            "_pi_new_session_log_path",
            "_pi_active_session_marker_path",
            "_pi_bridge_extension_path",
            "_reset_pi_active_session_marker",
            "_read_pi_active_session_marker",
            "_expand_cwd",
            "_user_shell",
            "_agent_shell_command",
        }
        self.assertTrue(moved_names <= launch_defs)
        self.assertFalse(moved_names & broker_defs)

    def test_broker_keeps_patch_sensitive_launch_wrappers(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_launch import SHELL_PRE_EXEC_MARKER", broker_source)
        self.assertIn("from codoxear.broker_launch import _resume_session_id_from_args as _resume_session_id_from_args_impl", broker_source)
        self.assertIn("return _resume_session_id_from_args_impl(args, agent_backend=AGENT_BACKEND)", broker_source)
        self.assertIn("return _ensure_pi_bridge_args_impl(args=args, marker_path=marker_path, agent_backend=AGENT_BACKEND)", broker_source)
        self.assertIn("return _ensure_pi_session_arg_impl(args=args, cwd=cwd, sessions_dir=sessions_dir, agent_backend=AGENT_BACKEND)", broker_source)
        self.assertIn("return _shell_argv_for_command_impl(cmd, user_shell=_user_shell)", broker_source)


if __name__ == "__main__":
    unittest.main()
