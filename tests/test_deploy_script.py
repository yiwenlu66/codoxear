from __future__ import annotations

import configparser
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_deploy_script_updates_only_a_clean_snapshot_before_service_operations(tmp_path: Path) -> None:
    deploy_dir = tmp_path / "deploy"
    unit_path = tmp_path / "codoxear-server.service"
    unit_path.write_text(
        "[Service]\n"
        "WorkingDirectory=/editable/checkout\n"
        "ExecStart=/old/venv/bin/python -u -m codoxear.server\n"
        "EnvironmentFile=-/editable/checkout/.env\n"
        "Environment=CODEX_WEB_PORT=9876 PRESERVE_ME=yes\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pipx_home = tmp_path / "pipx-home"
    (pipx_home / "venvs" / "codoxear" / "bin").mkdir(parents=True)
    (pipx_home / "venvs" / "codoxear" / "bin" / "python").touch(mode=0o755)
    operation_log = tmp_path / "operations.log"

    _write_executable(
        fake_bin / "pipx",
        f"""#!/usr/bin/env bash
set -eu
printf 'pipx %s\\n' "$*" >> {operation_log!s}
if [[ "$1" == "environment" ]]; then
  printf '%s\\n' {pipx_home!s}
fi
""",
    )
    _write_executable(
        fake_bin / "systemctl",
        f"""#!/usr/bin/env bash
set -eu
printf 'systemctl %s\\n' "$*" >> {operation_log!s}
if [[ "$*" == *"show codoxear-server.service -p Environment --value"* ]]; then
  printf 'CODEX_WEB_PORT=9876\\n'
fi
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -eu
url="${!#}"
case "$url" in
  */api/sessions) printf '401' ;;
  *) printf '200' ;;
esac
""",
    )

    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODOXEAR_DEPLOY_DIR": str(deploy_dir),
        "CODOXEAR_SERVICE_UNIT": str(unit_path),
        # The sandbox has no real server for the headless boot check to load.
        "CODOXEAR_SKIP_BOOT_CHECK": "1",
    }
    try:
        completed = subprocess.run(
            [str(DEPLOY_SCRIPT), commit],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )

        assert f"deployed {commit}" in completed.stdout
        assert subprocess.check_output(["git", "-C", deploy_dir, "rev-parse", "HEAD"], text=True).strip() == commit
        unit_config = configparser.ConfigParser(interpolation=None, strict=False)
        unit_config.optionxform = str
        unit_config.read(unit_path)
        service = unit_config["Service"]
        assert service["WorkingDirectory"] == str(deploy_dir)
        assert service["ExecStart"] == f"{pipx_home}/venvs/codoxear/bin/python -u -m codoxear.server"
        assert service["EnvironmentFile"] == "-/editable/checkout/.env"
        assert service["Environment"] == "CODEX_WEB_PORT=9876 PRESERVE_ME=yes"
        operations = operation_log.read_text()
        assert "pipx install --force" in operations
        assert "systemctl --user restart codoxear-server.service" in operations

        (deploy_dir / "must-stay-unmodified").write_text("dirty")
        operation_log.write_text("")
        failed = subprocess.run(
            [str(DEPLOY_SCRIPT), commit],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert failed.returncode != 0
        assert "refusing to replace a dirty deploy worktree" in failed.stderr
        assert operation_log.read_text() == ""
    finally:
        if deploy_dir.exists():
            subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", str(deploy_dir)], check=False)
