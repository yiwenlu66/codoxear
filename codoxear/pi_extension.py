"""Install a live loader, not a stale copy, of Codoxear's packaged Pi bridge."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


def loader_path(*, home: Path | None = None) -> Path:
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    return (Path(agent_dir).expanduser() if agent_dir else (home or Path.home()) / ".pi/agent") / "extensions/codoxear-bridge.ts"


def bridge_path() -> Path:
    return Path(__file__).resolve().with_name("pi_active_session_bridge.ts")


def loader_content(target: Path) -> str:
    return "// Managed by codoxear-pi-extension; uninstall with that command.\nexport { default } from " + json.dumps(str(target)) + ";\n"


def extension_status(*, home: Path | None = None) -> dict:
    loader = loader_path(home=home)
    if not loader.exists():
        return {"status": "missing", "detail": f"Pi bridge not installed: {loader}", "loader": str(loader)}
    try:
        content = loader.read_text(encoding="utf-8")
        # Parse our tiny generated loader so status also catches an old pipx
        # environment's dangling absolute target after a reinstall.
        line = next(line for line in content.splitlines() if line.startswith("export { default } from "))
        target = Path(json.loads(line.removeprefix("export { default } from ").removesuffix(";")))
    except (OSError, ValueError, StopIteration, TypeError):
        return {"status": "error", "detail": f"Unrecognized Pi bridge loader: {loader}", "loader": str(loader)}
    ok = target.is_file()
    return {"status": "ok" if ok else "error", "detail": f"{loader} → {target}" + ("" if ok else " (dangling target; reinstall)"), "loader": str(loader), "target": str(target)}


def install(*, home: Path | None = None) -> dict:
    loader = loader_path(home=home)
    if loader.exists() and extension_status(home=home)["status"] == "error" and "target" not in extension_status(home=home):
        raise ValueError(f"Refusing to replace an unmanaged file: {loader}")
    loader.parent.mkdir(parents=True, exist_ok=True)
    temp = loader.with_suffix(".tmp")
    temp.write_text(loader_content(bridge_path()), encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(loader)
    return extension_status(home=home)


def uninstall(*, home: Path | None = None) -> dict:
    status = extension_status(home=home)
    if status["status"] != "missing":
        if "target" not in status:
            raise ValueError(f"Refusing to remove an unmanaged file: {status['loader']}")
        loader_path(home=home).unlink()
    return extension_status(home=home)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/uninstall Codoxear's global Pi extension. Existing Pi sessions need /reload or restart.")
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        status = {"install": install, "uninstall": uninstall, "status": extension_status}[args.action]()
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(status) if args.json else status["detail"])
    return 0 if args.action == "uninstall" or status["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
