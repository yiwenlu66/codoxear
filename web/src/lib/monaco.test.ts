import { describe, expect, it } from "vitest";

import { workerModuleKeyForLabel } from "./monaco";

describe("monaco loader", () => {
  it("maps Monaco worker labels to the expected worker bundles", () => {
    expect(workerModuleKeyForLabel("json")).toBe("json");
    expect(workerModuleKeyForLabel("css")).toBe("css");
    expect(workerModuleKeyForLabel("scss")).toBe("css");
    expect(workerModuleKeyForLabel("html")).toBe("html");
    expect(workerModuleKeyForLabel("javascript")).toBe("typescript");
    expect(workerModuleKeyForLabel("typescript")).toBe("typescript");
    expect(workerModuleKeyForLabel("markdown")).toBe("editor");
  });
});
