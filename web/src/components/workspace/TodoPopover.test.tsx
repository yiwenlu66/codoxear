import { render } from "preact";
import { afterEach, describe, expect, it } from "vitest";
import { TodoPopover } from "./TodoPopover";

describe("TodoPopover", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders progress text and todo items when a snapshot is available", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoPopover
        snapshot={{
          available: true,
          error: false,
          progress_text: "2/3 completed",
          items: [
            { title: "Explore project context", status: "completed", description: "Read the relevant UI files" },
            { title: "Restore todo popover", status: "in-progress" },
          ],
        }}
      />,
      root,
    );

    expect(root.textContent).toContain("Todo");
    expect(root.textContent).toContain("2/3 completed");
    expect(root.textContent).toContain("Explore project context");
    expect(root.textContent).toContain("completed");
    expect(root.textContent).toContain("Read the relevant UI files");
    expect(root.textContent).toContain("Restore todo popover");
    expect(root.querySelectorAll(".todoPopoverItem")).toHaveLength(2);
  });

  it("renders the empty state when no todo snapshot is available yet", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(<TodoPopover snapshot={{ available: false, error: false, items: [] }} />, root);

    expect(root.textContent).toContain("No todo list yet");
  });

  it("renders the unavailable state when snapshot loading failed", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(<TodoPopover snapshot={{ available: false, error: true, items: [] }} />, root);

    expect(root.textContent).toContain("Todo list unavailable");
  });

  it("falls back safely when items are malformed or missing titles", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoPopover
        snapshot={{
          available: true,
          error: false,
          items: [{ status: "not-started", description: "Needs a title fallback" }],
        }}
      />,
      root,
    );

    expect(root.textContent).toContain("Untitled todo");
    expect(root.textContent).toContain("Needs a title fallback");
  });
});
