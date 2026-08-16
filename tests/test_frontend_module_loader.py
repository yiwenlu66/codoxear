from __future__ import annotations

import json
import subprocess
import textwrap

from frontend_module_loader import strip_esm


def test_strip_esm_preserves_esm_strict_assignment_behavior() -> None:
    transformed = strip_esm(
        "export { writeUndeclared };\nfunction writeUndeclared() { undeclaredProjectionValue = 1; }\nwriteUndeclared();\n",
        "fixture.js",
    )
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const context = {{ window: {{}} }};
        vm.createContext(context);
        let errorName = "";
        try {{ vm.runInContext({json.dumps(transformed)}, context); }} catch (error) {{ errorName = error.name; }}
        process.stdout.write(JSON.stringify({{ errorName, leaked: "undeclaredProjectionValue" in context }}));
        """
    )

    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"errorName": "ReferenceError", "leaked": False}
