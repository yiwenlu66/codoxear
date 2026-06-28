import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTE_DEPS_PY = ROOT / "codoxear" / "server_route_deps.py"
SERVER_PY = ROOT / "codoxear" / "server.py"


def test_route_deps_use_explicit_caps_after_construction() -> None:
    module = ast.parse(ROUTE_DEPS_PY.read_text(encoding="utf-8"))

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ServerRouteDepsFactory":
            for child in ast.walk(node):
                assert not (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "server"
                )
                assert not (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Attribute)
                    and child.value.attr == "server"
                )
            return
    raise AssertionError("ServerRouteDepsFactory missing")


def test_server_builds_route_deps_from_live_caps() -> None:
    server_source = SERVER_PY.read_text(encoding="utf-8")

    assert "from .server_route_deps import server_route_caps as _server_route_caps_impl" in server_source
    assert "server_module = sys.modules[__name__]" in server_source
    assert "return ServerRouteDepsFactory(_server_route_caps_impl(server_module))" in server_source
