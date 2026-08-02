import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTE_DEPS_PY = ROOT / "codoxear" / "server_route_deps.py"


def test_route_deps_accept_server_config_directly() -> None:
    module = ast.parse(ROUTE_DEPS_PY.read_text(encoding="utf-8"))

    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "ServerRouteCaps"
        for node in module.body
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "server_route_caps"
        for node in module.body
    )

    factory = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerRouteDepsFactory"
    )
    config_field = next(
        node
        for node in factory.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "config"
    )
    assert isinstance(config_field.annotation, ast.Name)
    assert config_field.annotation.id == "ServerConfig"
