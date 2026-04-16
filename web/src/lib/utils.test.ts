import { describe, expect, it } from "vitest";
import { cn } from "./utils";

describe("cn", () => {
  it("merges Tailwind classes predictably", () => {
    expect(cn("px-3 py-2", "px-4", false && "hidden", "text-sm", "text-sm")).toBe("py-2 px-4 text-sm");
  });
});
