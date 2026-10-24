from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codoxear import dependency_doctor as doctor
from codoxear import video_preview, voice_hls
from codoxear.system_dependencies import resolve_executable


def fake_resolve(name):
    return "/native/bin/" + name


def successful_run(args, **kwargs):
    return subprocess.CompletedProcess(args, 0, "version 1\n", "")


class ResolverTests(unittest.TestCase):
    def test_path_wins_over_known_fallbacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            user = root / ".local/bin"
            custom = root / "custom"
            user.mkdir(parents=True)
            custom.mkdir()
            for directory in (custom, user):
                binary = directory / "codoxear-test-tool"
                binary.write_text("#!/bin/sh\nexit 0\n")
                binary.chmod(0o755)
            self.assertEqual(resolve_executable("codoxear-test-tool", path=str(custom), home=root), str(custom / "codoxear-test-tool"))
            self.assertEqual(resolve_executable("codoxear-test-tool", path="/nonexistent", home=root), str(user / "codoxear-test-tool"))

    def test_empty_path_never_implicitly_searches_current_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "codoxear-cwd-only-test-tool"
            binary = root / name
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            previous = Path.cwd()
            try:
                os.chdir(root)
                self.assertIsNone(resolve_executable(name, path="", home=root))
                fallback = root / ".local/bin"
                fallback.mkdir(parents=True)
                target = fallback / name
                binary.replace(target)
                self.assertEqual(resolve_executable(name, path="", home=root), str(target))
            finally:
                os.chdir(previous)

    def test_homebrew_and_mac_application_fallback(self):
        seen = []
        def which(name, *, path=None):
            seen.append((name, path))
            if name == "ffmpeg" and "/opt/homebrew/bin" in path.split(":"):
                return "/opt/homebrew/bin/ffmpeg"
            if name == "/Applications/LibreOffice.app/Contents/MacOS/soffice":
                return name
            return None
        self.assertEqual(resolve_executable("ffmpeg", platform="darwin", path="/narrow", which=which), "/opt/homebrew/bin/ffmpeg")
        self.assertEqual(resolve_executable("soffice", platform="darwin", path="/narrow", which=which), "/Applications/LibreOffice.app/Contents/MacOS/soffice")
        self.assertIsNone(resolve_executable("soffice", platform="linux", path="/narrow", which=which))


class DoctorTests(unittest.TestCase):
    def test_versions_git_usability_and_real_codec_command(self):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            return successful_run(args, **kwargs)
        checks = doctor.check_dependencies(["office", "media", "git", "tmux"], resolve=fake_resolve, run=run)
        self.assertTrue(all(c["status"] == "ok" for c in checks.values()))
        self.assertIn(["/native/bin/tmux", "-V"], calls)
        self.assertTrue(any(c[1:3] == ["init", "--bare"] for c in calls))
        encode = next(c for c in calls if "libx264" in c)
        self.assertIn("aac", encode)
        self.assertEqual(encode[-3:], ["-f", "null", "-"])

    def test_codec_failure_is_not_binary_success(self):
        def run(args, **kwargs):
            if "libx264" in args:
                return subprocess.CompletedProcess(args, 1, "", "Unknown encoder 'libx264'")
            return successful_run(args, **kwargs)
        checks = doctor.check_dependencies(["media"], resolve=fake_resolve, run=run)
        self.assertEqual(checks["media"]["status"], "error")
        self.assertIn("libx264", checks["media"]["detail"])

    def test_broken_git_and_missing_executable(self):
        def run(args, **kwargs):
            if "init" in args:
                return subprocess.CompletedProcess(args, 1, "", "broken git")
            return successful_run(args, **kwargs)
        self.assertEqual(doctor.check_dependencies(["git"], resolve=fake_resolve, run=run)["git"]["status"], "error")
        self.assertEqual(doctor.check_dependencies(["media"], resolve=lambda _: None)["media"]["status"], "missing")

    def test_mac_font_files_without_fontconfig_and_unknown_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "NotoSans-Regular.ttf").touch()
            (root / "NotoSansCJK.ttc").touch()
            checks = doctor.check_dependencies(["fonts"], resolve=lambda _: None, platform="darwin", font_roots=[root])
            self.assertEqual(checks["fonts"]["status"], "ok")
            checks = doctor.check_dependencies(["fonts"], resolve=lambda _: None, platform="linux", font_roots=[root / "absent"])
            self.assertEqual(checks["fonts"]["status"], "unknown")

    def test_fontconfig_inventory(self):
        def run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, "Noto Sans\nNoto Sans CJK SC\n", "")
        self.assertEqual(doctor.check_dependencies(["fonts"], resolve=fake_resolve, run=run)["fonts"]["status"], "ok")

    def test_office_probe_validates_pdf_not_just_exit_code(self):
        checks = doctor.check_dependencies(["office"], resolve=fake_resolve, run=successful_run, probe_office=True)
        self.assertEqual(checks["office"]["status"], "error")
        def run(args, **kwargs):
            if "--convert-to" in args:
                Path(args[-1]).with_suffix(".pdf").write_bytes(b"%PDF-1.7\n")
            return successful_run(args, **kwargs)
        checks = doctor.check_dependencies(["office"], resolve=fake_resolve, run=run, probe_office=True)
        self.assertEqual(checks["office"]["status"], "ok")

    def test_distro_detection_does_not_guess_from_available_managers(self):
        for distro, manager in [("ubuntu", "apt"), ("fedora", "dnf"), ("arch", "pacman"), ("opensuse-tumbleweed", "zypper")]:
            self.assertEqual(doctor.detect_manager(platform="linux", os_release=f"ID={distro}", resolve=fake_resolve), manager)
        self.assertEqual(doctor.detect_manager(platform="linux", os_release='ID=derivative\nID_LIKE="debian ubuntu"', resolve=fake_resolve), "apt")
        self.assertIsNone(doctor.detect_manager(platform="win32", os_release="", resolve=fake_resolve))
        self.assertIsNone(doctor.detect_manager(platform="linux", os_release="ID=unknown", resolve=fake_resolve))

    def test_package_plans_are_selected_and_native(self):
        checks = {"office": {"status": "missing"}, "fonts": {"status": "unknown"}, "git": {"status": "ok"}}
        for manager in ("apt", "dnf", "pacman", "zypper"):
            plan = doctor.install_plan(checks, manager=manager, resolve=fake_resolve, uid=1000)
            self.assertEqual(plan["groups"], ["office", "fonts"])
            self.assertEqual(plan["commands"][0][0], "/native/bin/sudo")
            self.assertNotIn("git", plan["commands"][0])
        brew = doctor.install_plan({**checks, "tmux": {"status": "missing"}}, manager="brew", resolve=fake_resolve, uid=1000)
        self.assertEqual(brew["commands"], [["/native/bin/brew", "install", "tmux"], ["/native/bin/brew", "install", "--cask", "libreoffice", "font-noto-sans", "font-noto-sans-cjk"]])
        fedora = doctor.install_plan({"media": {"status": "error"}}, manager="dnf", resolve=fake_resolve)
        self.assertEqual(fedora["commands"], [["/native/bin/dnf", "install", "-y", "ffmpeg-free"]])
        self.assertIn("libx264", fedora["notes"][0])

    def test_cli_check_dry_run_cancel_install_and_failure(self):
        for args, expected, approved in [
            (["--git", "--json"], 1, False),
            (["--git", "--dry-run", "--json"], 0, False),
            (["--git", "--install", "--json"], 2, False),
            (["--git", "--install", "--yes", "--json"], 0, True),
        ]:
            installed = False
            installer_calls = []
            def resolve(name):
                return None if name == "git" and not installed else fake_resolve(name)
            def run(command, **kwargs):
                nonlocal installed
                if "install" in command:
                    installer_calls.append(command)
                    installed = True
                return successful_run(command, **kwargs)
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = doctor.main(args, resolve=resolve, run=run, platform="linux", os_release="ID=ubuntu", uid=0, confirm=lambda _: "n")
            self.assertEqual(code, expected)
            report = json.loads(out.getvalue())
            self.assertEqual(bool(installer_calls), approved)
            if approved:
                self.assertEqual(report["checks"]["git"]["status"], "ok")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                doctor.main(["--install"], resolve=lambda _: None)
        self.assertEqual(exc.exception.code, 2)

    def test_healthy_soffice_does_not_skip_module_install(self):
        plan = doctor.install_plan({"office": {"status": "ok"}}, manager="apt", resolve=fake_resolve)
        self.assertEqual(plan["groups"], ["office"])
        self.assertEqual(plan["commands"][0][-3:], ["libreoffice-writer", "libreoffice-calc", "libreoffice-impress"])

    def test_clear_cache_uses_runtime_directory_and_preserves_unowned_files(self):
        with tempfile.TemporaryDirectory() as temp, patch("codoxear.util.default_app_dir", return_value=Path(temp)):
            cache = Path(temp) / "office_previews"
            cache.mkdir()
            (cache / ("a" * 64 + ".pdf")).write_bytes(b"%PDF-owned")
            original = cache / "original.pdf"
            original.write_bytes(b"%PDF-original")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(doctor.main(["--clear-office-cache", "--json"]), 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report["office_cache"]["removed"], 1)
            self.assertEqual(original.read_bytes(), b"%PDF-original")

    def test_installer_failure_and_unsupported_os_are_explicit(self):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 4, "", "failed")
        for platform, release in [("linux", "ID=ubuntu"), ("win32", "")]:
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = doctor.main(["--install", "--git", "--yes", "--json"], resolve=lambda n: None if n == "git" else fake_resolve(n), run=run, platform=platform, os_release=release, uid=0)
            self.assertEqual(code, 2)
            report = json.loads(out.getvalue())
            self.assertTrue(report.get("installation", "").startswith("failed") or report["plan"]["notes"])


class RuntimeResolutionTests(unittest.TestCase):
    def test_git_invokes_resolved_path(self):
        from codoxear import git_ops
        result = subprocess.CompletedProcess([], 0, b"main\n", b"")
        with patch.object(git_ops, "resolve_executable", side_effect=fake_resolve), patch.object(git_ops.subprocess, "run", return_value=result) as run:
            self.assertEqual(git_ops.run_git(Path("."), ["branch", "--show-current"], timeout_s=1, max_bytes=1024), "main\n")
            self.assertEqual(run.call_args.args[0][0], "/native/bin/git")

    def test_server_tmux_probe_and_factories_share_resolver(self):
        from codoxear import server, session_manager_factories as factories
        with patch.object(server, "_TMUX_AVAILABLE_CACHE", None), patch.object(server, "resolve_executable", return_value="/native/bin/tmux") as resolve:
            self.assertTrue(server._tmux_available())
            resolve.assert_called_once_with("tmux")
        with patch.object(factories, "resolve_executable", side_effect=fake_resolve):
            deps = factories.session_manager_coordinator_deps(server)
            self.assertEqual(deps.prune_coordinator.which_tmux("tmux"), "/native/bin/tmux")
            self.assertEqual(deps.web_launch_coordinator.which_tmux("tmux"), "/native/bin/tmux")

    def test_hls_invokes_resolved_paths(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, b"", b"")
        with tempfile.TemporaryDirectory() as temp, patch.object(voice_hls, "resolve_executable", side_effect=fake_resolve), patch.object(voice_hls.subprocess, "run", side_effect=run), patch.object(voice_hls.subprocess, "check_output", return_value="6.0\n") as probe:
            stream = voice_hls.MergedHLSStream(root_dir=Path(temp))
            self.assertTrue(stream.append_silence(force=True))
            self.assertEqual(calls[0][0], "/native/bin/ffmpeg")
            self.assertEqual(probe.call_args.args[0][0], "/native/bin/ffprobe")

    def test_video_reports_missing_resolved_binary(self):
        with patch.object(video_preview, "resolve_executable", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg is required"):
                video_preview._generate_video_preview(Path("source.mov"), Path("preview.mp4"), preview_dir=Path("/unused"))


if __name__ == "__main__":
    unittest.main()
