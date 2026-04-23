from __future__ import annotations

import ast
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "codoxear"


def _module_name(path: Path) -> str:
    return "codoxear." + str(path.relative_to(ROOT).with_suffix("")).replace("/", ".")


def _resolve_from(module_name: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = module_name.split(".")[:-1]
    if node.level > len(parts) + 1:
        return ""
    base_parts = parts[: len(parts) - node.level + 1]
    if node.module:
        base_parts += node.module.split(".")
    return ".".join(base_parts)


def _module_graph() -> tuple[dict[str, Path], dict[str, set[str]]]:
    modules: dict[str, Path] = {}
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        modules[_module_name(path)] = path

    edges: dict[str, set[str]] = defaultdict(set)
    for module_name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if not name.startswith("codoxear"):
                        continue
                    candidates = [
                        m for m in modules if m == name or m.startswith(name + ".")
                    ]
                    if candidates:
                        edges[module_name].add(sorted(candidates, key=len)[0])
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(module_name, node)
                if not base.startswith("codoxear"):
                    continue
                candidates = [
                    m for m in modules if m == base or m.startswith(base + ".")
                ]
                if candidates:
                    edges[module_name].add(sorted(candidates, key=len)[0])
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{base}.{alias.name}"
                    if candidate in modules:
                        edges[module_name].add(candidate)

    return modules, edges


def _focus_modules(modules: dict[str, Path]) -> list[str]:
    out: list[str] = []
    for name in modules:
        if name == "codoxear.server":
            out.append(name)
            continue
        if name.startswith("codoxear.http."):
            out.append(name)
            continue
        if name.startswith("codoxear.sessions."):
            out.append(name)
            continue
        if name.startswith("codoxear.workspace."):
            out.append(name)
            continue
    return sorted(out)


def _scc(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    onstack: set[str] = set()
    idx: dict[str, int] = {}
    low: dict[str, int] = {}
    comps: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in edges.get(v, set()):
            if w not in nodes:
                continue
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in nodes:
        if v not in idx:
            strongconnect(v)
    return comps


class TestDependencyDagSource(unittest.TestCase):
    def test_sessions_and_workspace_do_not_import_server(self) -> None:
        modules, edges = _module_graph()
        offenders: list[str] = []
        for module_name in sorted(modules):
            if not (
                module_name.startswith("codoxear.sessions.")
                or module_name.startswith("codoxear.workspace.")
            ):
                continue
            if "codoxear.server" in edges.get(module_name, set()):
                offenders.append(module_name)
        self.assertEqual(offenders, [], f"owner modules import server: {offenders}")

    def test_server_sessions_workspace_http_focus_graph_is_acyclic(self) -> None:
        modules, edges = _module_graph()
        focus = _focus_modules(modules)
        comps = _scc(focus, edges)
        cyclic = [sorted(comp) for comp in comps if len(comp) > 1]
        self.assertEqual(cyclic, [], f"focus graph contains cycles: {cyclic}")


if __name__ == "__main__":
    unittest.main()
