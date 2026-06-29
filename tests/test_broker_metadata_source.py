import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_METADATA_PY = ROOT / "codoxear" / "broker_metadata.py"


class TestBrokerMetadataSource(unittest.TestCase):
    def test_sidecar_metadata_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        metadata_tree = ast.parse(BROKER_METADATA_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        metadata_defs = {node.name for node in metadata_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        owned_names = {
            "_claimed_log_paths_from_sock_meta",
            "_broker_sidecar_meta",
            "_write_broker_sidecar_meta",
        }
        self.assertTrue(owned_names <= metadata_defs)
        self.assertFalse(owned_names & broker_defs)

    def test_broker_write_meta_is_projection_wrapper(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        self.assertNotIn("import json", broker_source)
        self.assertIn("from codoxear.broker_metadata import _claimed_log_paths_from_sock_meta", broker_source)
        self.assertIn("from codoxear.broker_metadata import _write_broker_sidecar_meta", broker_source)
        self.assertIn("_write_broker_sidecar_meta(", broker_source)
        self.assertIn("owner_tag=OWNER_TAG", broker_source)
        self.assertIn("agent_backend=AGENT_BACKEND", broker_source)
        self.assertIn("service_tier=SERVICE_TIER_OVERRIDE", broker_source)


if __name__ == "__main__":
    unittest.main()
