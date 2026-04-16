import { render } from "preact";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TodoComposerPanel } from "./TodoComposerPanel";

describe("TodoComposerPanel", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders a compact summary row when collapsed", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "2/4 completed",
          items: [
            { title: "Explore project context", status: "completed" },
            { title: "Move todo above composer", status: "in-progress" },
          ],
        }}
        expanded={false}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.querySelector(".composerTodoBar")).not.toBeNull();
    expect(root.querySelector(".composerTodoBarButton")).not.toBeNull();
    expect(root.querySelector(".composerTodoPanel")).toBeNull();
    expect(root.textContent).toContain("2/4 completed");
  });

  it("renders the full list when expanded", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [
            { title: "Restore legacy placement", status: "completed", description: "Move it above the input" },
            { title: "Polish summary bar", status: "in-progress" },
          ],
        }}
        expanded={true}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.querySelector(".composerTodoPanel")).not.toBeNull();
    expect(root.querySelectorAll(".composerTodoItem")).toHaveLength(2);
    expect(root.textContent).toContain("Restore legacy placement");
    expect(root.textContent).toContain("Move it above the input");
    expect(root.textContent).toContain("in-progress");
  });

  it("renders nothing when snapshot.available is not true", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: false,
          error: false,
          progress_text: "should stay hidden",
          items: [{ title: "Hidden item", status: "completed" }],
        }}
        expanded={false}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.innerHTML).toBe("");
  });

  it("renders nothing when normalization leaves no valid items", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          items: [{ title: "   ", status: "   ", description: "   " }],
        }}
        expanded={true}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.innerHTML).toBe("");
  });

  it("keeps raw Pi status strings and normalization fallbacks", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "1/1 completed",
          items: [{ title: "  ", status: "  custom-status  ", description: "  Needs trimming  " }],
        }}
        expanded={true}
        onToggle={() => undefined}
      />,
      root,
    );

    expect(root.textContent).toContain("Untitled todo");
    expect(root.textContent).toContain("custom-status");
    expect(root.textContent).toContain("Needs trimming");
  });

  it("calls onToggle when the summary button is clicked and reflects expanded state", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    const onToggle = vi.fn();

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [{ title: "Keep toggle wired", status: "in-progress" }],
        }}
        expanded={false}
        onToggle={onToggle}
      />,
      root,
    );

    let button = root.querySelector(".composerTodoBarButton") as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.getAttribute("aria-expanded")).toBe("false");

    button?.click();

    expect(onToggle).toHaveBeenCalledTimes(1);

    render(
      <TodoComposerPanel
        snapshot={{
          available: true,
          error: false,
          progress_text: "1/2 completed",
          items: [{ title: "Keep toggle wired", status: "in-progress" }],
        }}
        expanded={true}
        onToggle={onToggle}
      />,
      root,
    );

    button = root.querySelector(".composerTodoBarButton") as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.getAttribute("aria-expanded")).toBe("true");

    button?.click();

    expect(onToggle).toHaveBeenCalledTimes(2);
  });
});
