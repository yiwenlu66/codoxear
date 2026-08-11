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
    environment_file = tmp_path / "codoxear-server.env"
    environment_file.write_text("CODEX_WEB_PASSWORD=test-password\n")
    unit_path.write_text(
        "[Service]\n"
        "WorkingDirectory=/editable/checkout\n"
        "ExecStart=/old/venv/bin/python -u -m codoxear.server\n"
        "EnvironmentFile=-" + str(environment_file) + "\n"
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
if [[ "$*" == *"show codoxear-server.service -p EnvironmentFiles --value"* ]]; then
  printf '%s (ignore_errors=no)\\n' {environment_file!s}
elif [[ "$*" == *"show codoxear-server.service -p Environment --value"* ]]; then
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

    _write_executable(
        fake_bin / "node",
        f"""#!/usr/bin/env bash
set -eu
printf 'node %s\\n' "$*" >> {operation_log!s}
[[ "$1" == "--check" ]]
[[ -f "$2" ]]
""",
    )
    _write_executable(
        fake_bin / "npx",
        f"""#!/usr/bin/env bash
set -eu
printf 'npx %s\\n' "$*" >> {operation_log!s}
for arg in "$@"; do
  case "$arg" in
    --outfile=*) output="${{arg#--outfile=}}"; mkdir -p "$(dirname "$output")"; printf 'bundle' > "$output" ;;
  esac
done
""",
    )
    _write_executable(
        fake_bin / "agent-browser",
        f"""#!/usr/bin/env bash
set -eu
printf 'agent-browser %s\\n' "$*" >> {operation_log!s}
case "$1" in
  eval) printf '\"OK\"\\n' ;;
  errors) printf '%s\\n' '{{"success":true,"data":{{"errors":[]}},"error":null}}' ;;
esac
""",
    )

    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODOXEAR_DEPLOY_DIR": str(deploy_dir),
        "CODOXEAR_SERVICE_UNIT": str(unit_path),
        "CODOXEAR_BOOT_CHECK_PASSWORD": "",
        "CODEX_WEB_PASSWORD": "",
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
        assert service["EnvironmentFile"] == f"-{environment_file}"
        assert service["Environment"] == "CODEX_WEB_PORT=9876 PRESERVE_ME=yes"
        operations = operation_log.read_text()
        static_js_paths = sorted((deploy_dir / "codoxear" / "static").rglob("*.js"))
        checked_paths = [line.removeprefix("node --check ") for line in operations.splitlines() if line.startswith("node --check ")]
        assert set(checked_paths) == {str(path) for path in static_js_paths}
        assert len(checked_paths) == len(static_js_paths)
        assert "pipx install --force" in operations
        assert "systemctl --user restart codoxear-server.service" in operations
        assert f"node --check {deploy_dir}/codoxear/static/app.js" in operations
        assert "agent-browser fill #pw test-password" in operations
        assert "agent-browser click #loginBtn" in operations
        assert "agent-browser eval" in operations
        assert operations.index(f"node --check {deploy_dir}/codoxear/static/app.js") < operations.index("pipx install --force")

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


def test_deploy_rejects_undefined_app_call_before_service_operations(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(source_root)],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", source_root, "config", "user.email", "tests@example.invalid"], check=True)
    subprocess.run(["git", "-C", source_root, "config", "user.name", "deploy test"], check=True)
    app_path = source_root / "codoxear" / "static" / "app.js"
    app_path.write_text(app_path.read_text() + "\nfoo();\n")
    subprocess.run(["git", "-C", source_root, "add", "--", "codoxear/static/app.js"], check=True)
    subprocess.run(["git", "-C", source_root, "commit", "--quiet", "-m", "inject undefined app reference"], check=True)
    commit = subprocess.check_output(["git", "-C", source_root, "rev-parse", "HEAD"], text=True).strip()

    deploy_dir = tmp_path / "deploy"
    unit_path = tmp_path / "codoxear-server.service"
    unit_path.write_text(
        "[Service]\n"
        "WorkingDirectory=/editable/checkout\n"
        "ExecStart=/old/venv/bin/python -u -m codoxear.server\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    operation_log = tmp_path / "operations.log"
    _write_executable(
        fake_bin / "node",
        f"""#!/usr/bin/env bash
set -eu
printf 'node %s\\n' "$*" >> {operation_log!s}
[[ "$1" == "--check" ]]
[[ -f "$2" ]]
""",
    )
    for command in ("pipx", "systemctl", "curl"):
        _write_executable(
            fake_bin / command,
            f"""#!/usr/bin/env bash
set -eu
printf '{command} %s\\n' "$*" >> {operation_log!s}
exit 91
""",
        )

    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODOXEAR_DEPLOY_DIR": str(deploy_dir),
        "CODOXEAR_SERVICE_UNIT": str(unit_path),
    }
    try:
        completed = subprocess.run(
            [str(source_root / "scripts" / "deploy.sh"), commit],
            cwd=source_root,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert completed.returncode != 0
        assert "undefined function reference: foo" in completed.stderr
        assert "app.js reference check failed" in completed.stderr
        operations = operation_log.read_text()
        assert "node --check" in operations
        assert "pipx " not in operations
        assert "systemctl " not in operations
        assert "curl " not in operations
    finally:
        if deploy_dir.exists():
            subprocess.run(["git", "-C", source_root, "worktree", "remove", "--force", str(deploy_dir)], check=False)
