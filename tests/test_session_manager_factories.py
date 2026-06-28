import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FACTORIES_PY = ROOT / "codoxear" / "session_manager_factories.py"
SERVER_PY = ROOT / "codoxear" / "server.py"


def test_factories_use_explicit_caps_after_construction() -> None:
    module = ast.parse(FACTORIES_PY.read_text(encoding="utf-8"))

    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "session_manager_factory_caps":
            continue
        arg_names = [arg.arg for arg in node.args.args]
        assert "server" not in arg_names, node.name
        for child in ast.walk(node):
            assert not (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "server"
            ), node.name


def test_server_exports_factory_caps_for_late_binding() -> None:
    server_source = SERVER_PY.read_text(encoding="utf-8")
    binding_source = (ROOT / "codoxear" / "session_manager_method_bindings.py").read_text(encoding="utf-8")

    assert "from .session_manager_factories import session_manager_factory_caps as _session_manager_factory_caps_impl" in server_source
    assert "factory_caps = server_module._session_manager_factory_caps_impl(server_module)" in binding_source
