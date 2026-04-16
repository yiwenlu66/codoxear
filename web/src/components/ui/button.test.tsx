import { render } from "preact";
import { afterEach, describe, expect, it } from "vitest";
import { Button } from "./button";

describe("Button", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("applies shadcn-style variants and preserves custom classes", () => {
    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <Button variant="secondary" size="sm" className="tracking-wide">
        Save
      </Button>,
      root,
    );

    const button = root.querySelector("button");
    expect(button?.className).toContain("inline-flex");
    expect(button?.className).toContain("bg-secondary");
    expect(button?.className).toContain("h-9");
    expect(button?.className).toContain("tracking-wide");
  });
});
