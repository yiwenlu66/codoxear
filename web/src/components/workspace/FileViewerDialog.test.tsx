import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FileViewerDialog } from "./FileViewerDialog";
import { clearRememberedFileSelections } from "./fileSelectionState";

vi.mock("../../lib/api", () => ({
  api: {
    getFiles: vi.fn(),
    getFileRead: vi.fn(),
    getGitFileVersions: vi.fn(),
  },
}));

vi.mock("./MonacoWorkspace", () => ({
  MonacoWorkspace: (props: any) => (
    <div
      data-testid="monaco-workspace"
      data-line={props.line == null ? "" : String(props.line)}
      data-mode={props.mode}
      data-path={props.path}
    >
      {props.mode}:{props.path}
    </div>
  ),
}));

let root: HTMLDivElement | null = null;

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

async function settle(count = 4) {
  for (let index = 0; index < count; index += 1) {
    await flush();
  }
}

describe("FileViewerDialog", () => {
  afterEach(() => {
    clearRememberedFileSelections();
    vi.clearAllMocks();
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("loads the root directory, expands a folder, and opens a selected file in diff mode", async () => {
    const { api } = await import("../../lib/api");
    (api as any).getFiles.mockImplementation((_sessionId: string, nextPath?: string) => Promise.resolve(
      nextPath === "src"
        ? {
            ok: true,
            path: "src",
            entries: [{ name: "main.tsx", path: "src/main.tsx", kind: "file" }],
          }
        : {
            ok: true,
            path: "",
            entries: [
              { name: "src", path: "src", kind: "dir" },
              { name: "README.md", path: "README.md", kind: "file" },
            ],
          },
    ));
    (api as any).getGitFileVersions.mockResolvedValue({
      ok: true,
      path: "src/main.tsx",
      base_exists: true,
      current_exists: true,
      base_text: "const before = true;",
      current_text: "const after = true;",
    } as any);
    (api as any).getFileRead.mockResolvedValue({ ok: true, kind: "text", text: "const after = true;" });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <FileViewerDialog open sessionId="sess-diff" onClose={() => undefined} />,
        root!,
      );
      await settle(8);
    });
    await settle(8);

    expect((api as any).getFiles).toHaveBeenCalledWith("sess-diff", undefined, expect.any(AbortSignal));
    expect(root?.textContent).toContain("src");
    expect(root?.textContent).toContain("README.md");

    const expandButton = root?.querySelector('button[aria-label="Expand src"]') as HTMLButtonElement | null;
    expect(expandButton).not.toBeNull();
    act(() => {
      expandButton?.click();
    });
    await settle(8);

    expect((api as any).getFiles).toHaveBeenCalledWith("sess-diff", "src", expect.any(AbortSignal));

    const fileButton = Array.from(root?.querySelectorAll("button") || []).find((button) => button.textContent === "main.tsx") as HTMLButtonElement | undefined;
    expect(fileButton).toBeDefined();
    act(() => {
      fileButton?.click();
    });
    await settle(8);

    expect((api as any).getGitFileVersions).toHaveBeenCalledWith("sess-diff", "src/main.tsx", expect.any(AbortSignal));
    expect(root.textContent).toContain("Diff");
  });

  it("loads a directory only once when it is collapsed and re-expanded", async () => {
    const { api } = await import("../../lib/api");
    (api as any).getFiles.mockImplementation((_sessionId: string, nextPath?: string) => Promise.resolve(
      nextPath === "docs"
        ? {
            ok: true,
            path: "docs",
            entries: [{ name: "intro.md", path: "docs/intro.md", kind: "file" }],
          }
        : {
            ok: true,
            path: "",
            entries: [{ name: "docs", path: "docs", kind: "dir" }],
          },
    ));

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <FileViewerDialog open sessionId="sess-cache" onClose={() => undefined} />,
        root!,
      );
      await settle(8);
    });
    await settle(8);

    const expandButton = root?.querySelector('button[aria-label="Expand docs"]') as HTMLButtonElement | null;
    expect(expandButton).not.toBeNull();

    act(() => {
      expandButton?.click();
    });
    await settle(8);

    const collapseButton = root?.querySelector('button[aria-label="Collapse docs"]') as HTMLButtonElement | null;
    expect(collapseButton).not.toBeNull();
    act(() => {
      collapseButton?.click();
    });
    await settle(4);

    const reExpandButton = root?.querySelector('button[aria-label="Expand docs"]') as HTMLButtonElement | null;
    expect(reExpandButton).not.toBeNull();
    act(() => {
      reExpandButton?.click();
    });
    await settle(8);

    expect((api as any).getFiles).toHaveBeenCalledTimes(2);
  });

  it("can switch from diff mode to file and markdown preview modes", async () => {
    const { api } = await import("../../lib/api");
    (api as any).getFiles.mockResolvedValue({ ok: true, path: "", entries: [] });
    (api as any).getGitFileVersions.mockResolvedValue({
      ok: true,
      path: "docs/intro.md",
      base_exists: true,
      current_exists: true,
      base_text: "# Before",
      current_text: "# After",
    } as any);
    (api as any).getFileRead.mockResolvedValue({ ok: true, kind: "text", text: "# Hello\n\nBody" });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <FileViewerDialog open sessionId="sess-preview" initialPath="docs/intro.md" onClose={() => undefined} />,
        root!,
      );
      await settle(8);
    });

    const fileButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "File") as HTMLButtonElement | undefined;
    const previewButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "Preview") as HTMLButtonElement | undefined;
    expect(fileButton).toBeDefined();
    expect(previewButton).toBeDefined();

    act(() => {
      fileButton?.click();
    });
    await settle(6);
    expect(api.getFileRead).toHaveBeenCalledWith("sess-preview", "docs/intro.md", expect.any(AbortSignal));
    expect(root.textContent).toContain("docs/intro.md");

    act(() => {
      previewButton?.click();
    });
    await settle(6);
    expect(root.querySelector(".filePreview article h1")?.textContent).toBe("Hello");
    expect(root.querySelector(".filePreview article")?.textContent).toContain("Body");
  });

  it("opens explicit file references in file mode and preserves the requested line", async () => {
    const { api } = await import("../../lib/api");
    (api as any).getFiles.mockResolvedValue({ ok: true, path: "", entries: [] });
    (api as any).getFileRead.mockResolvedValue({ ok: true, kind: "text", text: "line 1\nline 2" });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <FileViewerDialog
          open
          sessionId="sess-line"
         
          initialPath="src/main.tsx"
          initialLine={18}
          onClose={() => undefined}
        />,
        root!,
      );
      await settle(8);
    });

    expect((api as any).getFileRead).toHaveBeenCalledWith("sess-line", "src/main.tsx", expect.any(AbortSignal));
    expect(root.textContent).toContain("line 18");
  });

  it("shows a friendly error when the file list payload is malformed", async () => {
    const { api } = await import("../../lib/api");
    (api as any).getFiles.mockResolvedValue({ ok: true, path: "", entries: null });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <FileViewerDialog open sessionId="sess-bad-files" onClose={() => undefined} />,
        root!,
      );
      await settle(8);
    });
    await settle(8);

    expect(root.textContent).toContain("Unable to list files");
    expect(root.textContent).not.toContain("Cannot read properties of null");
  });
});
