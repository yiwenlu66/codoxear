"""Explicit, optional native prerequisites doctor; never called by the service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Callable, Sequence

from .system_dependencies import resolve_executable

GROUPS = ("office", "fonts", "media", "git", "tmux")
# Native distribution packages only: no repository additions or system upgrades.
PACKAGES = {
    "apt": {
        "office": ["libreoffice-writer", "libreoffice-calc", "libreoffice-impress"],
        "fonts": ["fonts-noto-core", "fonts-noto-cjk"],
        "media": ["ffmpeg"], "git": ["git"], "tmux": ["tmux"],
    },
    "dnf": {
        "office": ["libreoffice-writer", "libreoffice-calc", "libreoffice-impress"],
        "fonts": ["google-noto-sans-fonts", "google-noto-sans-cjk-vf-fonts"],
        "media": ["ffmpeg-free"], "git": ["git"], "tmux": ["tmux"],
    },
    "pacman": {
        "office": ["libreoffice-fresh"], "fonts": ["noto-fonts", "noto-fonts-cjk"],
        "media": ["ffmpeg"], "git": ["git"], "tmux": ["tmux"],
    },
    "zypper": {
        "office": ["libreoffice-writer", "libreoffice-calc", "libreoffice-impress"],
        "fonts": ["google-noto-sans-fonts", "google-noto-sans-cjk-fonts"],
        "media": ["ffmpeg"], "git": ["git"], "tmux": ["tmux"],
    },
    "brew": {
        "office": ["libreoffice"], "fonts": ["font-noto-sans", "font-noto-sans-cjk"],
        "media": ["ffmpeg"], "git": ["git"], "tmux": ["tmux"],
    },
}
MANAGER_BIN = {"apt": "apt-get", "dnf": "dnf", "pacman": "pacman", "zypper": "zypper", "brew": "brew"}


def _command(args: Sequence[str], run: Callable, *, timeout: int = 20) -> tuple[bool, str]:
    try:
        result = run(list(args), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output[-2000:]


def _result(status: str, detail: str, **extra) -> dict:
    return {"status": status, "detail": detail, **extra}


def check_fonts(*, resolve: Callable, run: Callable, platform: str, home: Path, font_roots: Sequence[Path] | None = None) -> dict:
    fc_list = resolve("fc-list")
    inventory = ""
    if fc_list:
        ok, inventory = _command([fc_list, "--format=%{family}\\n"], run)
        if not ok:
            return _result("error", f"Font inventory failed: {inventory}")
    else:
        roots = font_roots if font_roots is not None else (
            [home / "Library/Fonts", Path("/Library/Fonts"), Path("/System/Library/Fonts")]
            if platform == "darwin" else
            [home / ".local/share/fonts", home / ".fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
        )
        names = []
        try:
            for root in roots:
                if root.is_dir():
                    names.extend(p.name for p in root.rglob("*") if p.suffix.lower() in {".ttf", ".otf", ".ttc"})
        except OSError as exc:
            return _result("unknown", f"Cannot inventory fonts: {exc}")
        inventory = "\n".join(names)
        if not inventory:
            return _result("unknown", "No font inventory available (fc-list absent); font availability is unverified.")
    normalized = re.sub(r"[ _-]", "", inventory).lower()
    core = bool(re.search(r"notosans(?:regular|bold|italic|\[|,|\n|$)", normalized))
    cjk = bool(re.search(r"notosans(?:cjk|sc|tc|hk|jp|kr)", normalized))
    return _result("ok" if core and cjk else "missing", f"Noto Sans baseline: {'yes' if core else 'not found'}; Noto Sans CJK: {'yes' if cjk else 'not found'}. Other scripts may need additional Noto families.")


def check_dependencies(
    groups: Sequence[str] = GROUPS, *, resolve: Callable = resolve_executable,
    run: Callable = subprocess.run, platform: str | None = None,
    home: Path | None = None, probe_office: bool = False,
    font_roots: Sequence[Path] | None = None,
) -> dict[str, dict]:
    platform = sys.platform if platform is None else platform
    home = Path.home() if home is None else Path(home)
    checks = {}
    for group in groups:
        if group == "fonts":
            checks[group] = check_fonts(resolve=resolve, run=run, platform=platform, home=home, font_roots=font_roots)
            continue
        names = {"office": ["soffice"], "media": ["ffmpeg", "ffprobe"], "git": ["git"], "tmux": ["tmux"]}[group]
        paths = {name: resolve(name) for name in names}
        missing = [name for name, path in paths.items() if path is None]
        if missing:
            checks[group] = _result("missing", "Not found: " + ", ".join(missing), executables=paths)
            continue
        versions = {}
        failure = ""
        for name, path in paths.items():
            flag = "-V" if name == "tmux" else "-version" if name in {"ffmpeg", "ffprobe"} else "--version"
            ok, output = _command([path, flag], run)
            versions[name] = output.splitlines()[0] if output else ""
            if not ok or not output:
                failure = f"{name} version check failed: {output}"
                break
        if not failure and group == "media":
            # Exercise both actual runtime encoders, rather than trusting a binary
            # name or the compile-time encoder list. No persistent output is made.
            ok, output = _command([
                paths["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "lavfi", "-i", "color=c=black:s=16x16:r=10",
                "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.1",
                "-c:v", "libx264", "-c:a", "aac", "-f", "null", "-",
            ], run)
            if not ok:
                failure = f"libx264 + AAC encoding probe failed: {output}"
        if not failure and group == "git":
            with tempfile.TemporaryDirectory(prefix="codoxear-doctor-git-") as temp:
                ok, output = _command([paths["git"], "init", "--bare", str(Path(temp) / "probe.git")], run)
            if not ok:
                failure = f"Git cannot initialize a temporary repository: {output}"
        if not failure and group == "office" and probe_office:
            with tempfile.TemporaryDirectory(prefix="codoxear-doctor-office-") as temp:
                root = Path(temp)
                source = root / "probe.txt"
                source.write_text("Codoxear Office conversion probe\n", encoding="utf-8")
                ok, output = _command([
                    paths["soffice"], f"-env:UserInstallation={(root / 'profile').as_uri()}",
                    "--headless", "--convert-to", "pdf", "--outdir", str(root), str(source),
                ], run, timeout=90)
                pdf = root / "probe.pdf"
                if not ok or not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
                    failure = f"Office text-to-PDF probe failed: {output}"
        detail = failure or "; ".join(versions.values())
        if not failure and group == "media":
            detail += "; libx264 + AAC encoding succeeded"
        if not failure and group == "office":
            detail += "; basic text-to-PDF probe succeeded" if probe_office else "; conversion not probed (use --probe-office)"
            detail += "; Writer/Calc/Impress module capabilities unverified; explicit --install --office always ensures module packages"
        checks[group] = _result("error" if failure else "ok", detail, executables=paths)
    return checks


def detect_manager(*, platform: str, os_release: str, resolve: Callable) -> str | None:
    if platform == "darwin":
        return "brew" if resolve("brew") else None
    if platform != "linux":
        return None
    fields = {}
    for line in os_release.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key] = value.strip().strip('"').strip("'")
    families = (fields.get("ID", "") + " " + fields.get("ID_LIKE", "")).split()
    for family in families:
        manager = {"debian": "apt", "ubuntu": "apt", "fedora": "dnf", "rhel": "dnf", "centos": "dnf", "arch": "pacman", "manjaro": "pacman", "opensuse": "zypper", "opensuse-leap": "zypper", "opensuse-tumbleweed": "zypper", "sles": "zypper", "suse": "zypper"}.get(family)
        if manager and resolve(MANAGER_BIN[manager]):
            return manager
    return None


def install_plan(checks: dict, *, manager: str | None, resolve: Callable = resolve_executable, uid: int = 0) -> dict:
    # A working soffice/version or text conversion does not prove Calc/Impress
    # are installed. Explicit Office provisioning must always ensure all modules.
    needed = [group for group, check in checks.items() if group == "office" or check["status"] != "ok"]
    plan = {"manager": manager, "groups": needed, "commands": [], "notes": []}
    if not needed:
        return plan
    if manager not in PACKAGES:
        plan["notes"].append("Unsupported OS or native package manager unavailable. Install LibreOffice (Writer/Calc/Impress), Noto Sans + CJK fonts, FFmpeg with libx264/AAC and ffprobe, Git, or tmux manually for the selected groups; rerun codoxear-doctor. On macOS install Homebrew yourself from https://brew.sh first.")
        return plan
    binary = resolve(MANAGER_BIN[manager])
    if binary is None:
        plan["notes"].append(f"Cannot resolve {MANAGER_BIN[manager]}; install it manually.")
        return plan
    prefix = []
    if manager != "brew" and uid != 0:
        sudo = resolve("sudo")
        if sudo is None:
            plan["notes"].append("sudo is unavailable; run the explicit install as root or install the listed native packages manually.")
            return plan
        prefix = [sudo]
    if manager == "brew" and uid == 0:
        plan["notes"].append("Homebrew must run as a normal user, not root.")
        return plan
    if manager == "brew":
        # Casks and formulae are separate manager invocations by contract.
        for cask in (False, True):
            packages = [p for group in needed if (group in {"office", "fonts"}) == cask for p in PACKAGES[manager][group]]
            if packages:
                plan["commands"].append([binary, "install", *(["--cask"] if cask else []), *packages])
    else:
        packages = list(dict.fromkeys(p for group in needed for p in PACKAGES[manager][group]))
        flags = {"apt": ["install", "-y"], "dnf": ["install", "-y"], "pacman": ["-S", "--needed", "--noconfirm"], "zypper": ["--non-interactive", "install"]}[manager]
        plan["commands"].append([*prefix, binary, *flags, *packages])
    if manager == "dnf" and "media" in needed:
        plan["notes"].append("Fedora's native ffmpeg-free may lack libx264. No third-party repositories are added. Choose a full-codec FFmpeg source yourself if the encoding probe fails; this helper will report failure, not claim media support.")
    if manager == "pacman":
        plan["notes"].append("Uses the existing package database, without refresh or system upgrade. On an out-of-date Arch system, follow your distribution's full-upgrade policy yourself before installing; never use pacman -Sy alone.")
    plan["notes"].append("Only non-ok selected groups are planned, except Office: explicit Office provisioning always ensures Writer/Calc/Impress module packages. Already-installed packages are left to the package manager; unavailable packages/repos fail explicitly. No repository changes, index refresh, or whole-system upgrade is performed.")
    return plan


def clear_office_cache() -> dict:
    from .office_preview import clear_office_preview_cache
    from .util import default_app_dir

    directory = default_app_dir() / "office_previews"
    return {"directory": str(directory), "removed": clear_office_preview_cache(directory)}


def main(argv: Sequence[str] | None = None, *, run: Callable = subprocess.run, resolve: Callable = resolve_executable,
         platform: str | None = None, os_release: str | None = None, uid: int | None = None,
         confirm: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Check optional native prerequisites; installation requires --install and explicit groups.")
    parser.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    parser.add_argument("--install", action="store_true", help="explicitly install non-ok selected groups")
    parser.add_argument("--dry-run", action="store_true", help="show install commands without running them")
    parser.add_argument("--yes", action="store_true", help="confirm the displayed install plan non-interactively")
    parser.add_argument("--probe-office", action="store_true", help="also convert a temporary text document to PDF (not a Calc/Impress capability check)")
    parser.add_argument("--clear-office-cache", action="store_true", help="explicitly remove only generated Office preview PDFs; use after font/LibreOffice changes")
    for group in GROUPS:
        parser.add_argument(f"--{group}", action="store_true", help=f"select {group} (default check: all groups)")
    args = parser.parse_args(argv)
    selected = [group for group in GROUPS if getattr(args, group)]
    if args.clear_office_cache:
        if selected or args.install or args.dry_run or args.probe_office or args.yes:
            parser.error("--clear-office-cache is a standalone operation; only --json may accompany it")
        try:
            cleared = clear_office_cache()
        except OSError as exc:
            error = {"office_cache": {"error": str(exc)}}
            print(json.dumps(error) if args.json else f"Office cache clear failed: {exc}")
            return 2
        print(json.dumps({"office_cache": cleared}, indent=2) if args.json else f"Removed {cleared['removed']} generated Office PDFs from {cleared['directory']}; originals untouched.")
        return 0
    if (args.install or args.dry_run) and not selected:
        parser.error("installation/planning requires explicit groups, e.g. --office --fonts")
    if args.yes and not args.install:
        parser.error("--yes requires --install")
    platform = sys.platform if platform is None else platform
    if os_release is None:
        try:
            os_release = Path("/etc/os-release").read_text(encoding="utf-8") if platform == "linux" else ""
        except OSError:
            os_release = ""
    checks = check_dependencies(selected or GROUPS, resolve=resolve, run=run, platform=platform, probe_office=args.probe_office)
    report = {"platform": platform, "checks": checks}
    code = 0 if all(check["status"] == "ok" for check in checks.values()) else 1
    if args.install or args.dry_run:
        manager = detect_manager(platform=platform, os_release=os_release, resolve=resolve)
        plan = install_plan(checks, manager=manager, resolve=resolve, uid=os.geteuid() if uid is None else uid)
        report["plan"] = plan
        if not args.json or (args.install and not args.dry_run):
            for command in plan["commands"]:
                print(shlex.join(command), file=sys.stderr)
            for note in plan["notes"]:
                print(note, file=sys.stderr)
        if plan["groups"] and not plan["commands"]:
            code = 2
        elif args.dry_run:
            code = 0
        elif plan["commands"]:
            try:
                if not args.yes:
                    print("Install this plan? [y/N] ", end="", file=sys.stderr, flush=True)
                approved = args.yes or confirm("").strip().lower() in {"y", "yes"}
            except EOFError:
                approved = False
            if not approved:
                report["installation"] = "cancelled"
                code = 2
            else:
                report["installation"] = "succeeded"
                for command in plan["commands"]:
                    try:
                        result = run(command, stdout=sys.stderr, check=False)
                    except OSError as exc:
                        report["installation"] = f"failed: {exc}"
                        code = 2
                        break
                    if result.returncode != 0:
                        report["installation"] = f"failed: exit {result.returncode}"
                        code = 2
                        break
                else:
                    report["checks"] = check_dependencies(selected, resolve=resolve, run=run, platform=platform, probe_office=args.probe_office)
                    code = 0 if all(check["status"] == "ok" for check in report["checks"].values()) else 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for group, check in report["checks"].items():
            print(f"{group}: {check['status']} — {check['detail']}")
        if "installation" in report:
            print(f"Installation: {report['installation']}")
        if code == 1:
            print("Some selected features are unavailable or unverified. Use --install with explicit groups, or --dry-run to inspect a native package plan.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
