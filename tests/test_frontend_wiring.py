import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"


def test_wiring_factories_preserve_explicit_controller_dependencies() -> None:
    sources = {
        name: (STATIC / name).read_text(encoding="utf-8")
        for name in ("app_wiring.js", "app_event_bindings.js")
    }
    program = """
const vm = require("vm");
const sources = __SOURCES__;
const context = { window: {} };
vm.createContext(context);
for (const source of Object.values(sources)) vm.runInContext(source, context);
const wiring = context.window.CodoxearWiring.createWiring();
const selected = () => "s-1";
const flow = wiring.createMessageFlowOptions({
  getSelected: selected,
  api: () => null,
  accidental: "must not reach controller",
});
const lifecycle = wiring.createSessionLifecycleOptions({
  getSelected: selected,
  setSelected: () => null,
  clearDeletedSessionClientState: "not a lifecycle dependency",
});
const listeners = [];
const target = {
  addEventListener(type, handler, options) { listeners.push(["add", type, handler, options]); },
  removeEventListener(type, handler, options) { listeners.push(["remove", type, handler, options]); },
};
const registered = [];
const events = context.window.CodoxearEventBindings.createEventBindings({
  addEvent(target, type, handler, options) {
    registered.push([target, type, handler, options]);
    target.addEventListener(type, handler, options);
    return handler;
  },
});
const handler = () => "clicked";
const bound = events.onClick(target, handler, { capture: true });
process.stdout.write(JSON.stringify({
  flowKeys: Object.keys(flow).sort(),
  flowSelected: flow.getSelected(),
  lifecycleKeys: Object.keys(lifecycle).sort(),
  event: { registered: registered.length, type: listeners[0][1], sameHandler: bound === handler, capture: listeners[0][3].capture },
}));
""".replace("__SOURCES__", json.dumps(sources))
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    result = json.loads(completed.stdout)

    assert result["flowKeys"] == ["api", "getSelected"]
    assert result["flowSelected"] == "s-1"
    assert result["lifecycleKeys"] == ["getSelected", "setSelected"]
    assert result["event"] == {"registered": 1, "type": "click", "sameHandler": True, "capture": True}
