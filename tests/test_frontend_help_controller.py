from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap


APP_MODAL_JS = module_path("app_modal.js")
APP_HELP_JS = module_path("app_help.js")


def test_help_controller_owns_open_close_coordination_and_focus_restore() -> None:
    sources = [APP_MODAL_JS.read_text(encoding="utf-8"), APP_HELP_JS.read_text(encoding="utf-8")]
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const sources = __SOURCES__;
        class ElementStub {}
        const calls = [];
        function node(name) {
          const listeners = {};
          return Object.assign(new ElementStub(), {
            name,
            style: { display: "none" },
            isConnected: true,
            disabled: false,
            addEventListener(type, handler) { listeners[type] = handler; },
            emit(type, event = {}) { return listeners[type](event); },
            focus() { calls.push(`focus:${name}`); },
          });
        }
        const opener = node("opener");
        const fallback = node("fallback");
        const backdrop = node("backdrop");
        const viewer = node("viewer");
        const closeButton = node("close");
        const ctx = { window: {}, requestAnimationFrame: (callback) => callback() };
        vm.createContext(ctx);
        for (const source of sources) vm.runInContext(source, ctx);
        const controller = ctx.window.CodoxearHelp.createHelpController({
          backdrop,
          viewer,
          closeButton,
          openButton: opener,
          documentTarget: { activeElement: fallback },
          ElementCtor: ElementStub,
          prepareModalOpen: () => calls.push("prepare"),
          afterModalVisibilityChanged: () => calls.push("visibility"),
          addEvent: (target, type, handler) => target.addEventListener(type, handler),
          focusModalCloseButton: (_viewer, button) => { calls.push("focus-close"); button.focus(); },
          isModalTargetOpen: (target) => target.style.display === "flex",
          restoreModalFocus: (target, stillOpen) => { calls.push(`restore:${stillOpen()}`); target.focus(); },
        });
        const click = {
          currentTarget: opener,
          preventDefault() { calls.push("prevent"); },
          stopPropagation() { calls.push("stop"); },
        };
        opener.emit("click", click);
        const openSnapshot = { backdrop: backdrop.style.display, viewer: viewer.style.display, isOpen: controller.isOpen() };
        closeButton.emit("click", click);
        const closeSnapshot = { backdrop: backdrop.style.display, viewer: viewer.style.display, isOpen: controller.isOpen() };
        controller.show();
        backdrop.emit("click");
        process.stdout.write(JSON.stringify({
          frozen: Object.isFrozen(controller),
          openSnapshot,
          closeSnapshot,
          calls,
        }));
        """
    ).replace("__SOURCES__", json.dumps(sources))
    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    data = json.loads(result.stdout)
    assert data["frozen"] is True
    assert data["openSnapshot"] == {"backdrop": "block", "viewer": "flex", "isOpen": True}
    assert data["closeSnapshot"] == {"backdrop": "none", "viewer": "none", "isOpen": False}
    assert data["calls"] == [
        "prevent", "stop", "prepare", "visibility", "focus-close", "focus:close",
        "prevent", "stop", "visibility", "restore:false", "focus:opener",
        "prepare", "visibility", "focus-close", "focus:close",
        "visibility", "restore:false", "focus:fallback",
    ]
