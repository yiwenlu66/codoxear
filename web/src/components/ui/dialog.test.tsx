import { render } from "preact";
import { useState } from "preact/hooks";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./dialog";
import { Sheet, SheetContent } from "./sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./tabs";

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

async function pressKey(target: EventTarget | null, key: string, options: KeyboardEventInit = {}) {
  await act(async () => {
    target?.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key, ...options }));
  });
}

function DismissibleDialogHarness() {
  const [open, setOpen] = useState(true);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent titleId="dismissible-dialog-title">
        <DialogHeader>
          <DialogTitle id="dismissible-dialog-title">Dialog title</DialogTitle>
        </DialogHeader>
        <button type="button">Dialog action</button>
      </DialogContent>
    </Dialog>
  );
}

describe("overlay primitives", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("does not render dialog or sheet content while closed", () => {
    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <>
        <Dialog open={false}>
          <DialogContent>Dialog body</DialogContent>
        </Dialog>
        <Sheet open={false}>
          <SheetContent side="left">Sheet body</SheetContent>
        </Sheet>
      </>,
      root,
    );

    expect(root.textContent).not.toContain("Dialog body");
    expect(root.textContent).not.toContain("Sheet body");
    expect(root.querySelector('[role="dialog"]')).toBeNull();
  });

  it("renders dialog and sheet content only while open with modal labelling", () => {
    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <>
        <Dialog open>
          <DialogContent titleId="dialog-title">
            <DialogHeader>
              <DialogTitle id="dialog-title">Dialog title</DialogTitle>
            </DialogHeader>
            Dialog body
          </DialogContent>
        </Dialog>
        <Sheet open>
          <SheetContent side="left" titleId="sheet-title">
            <h2 id="sheet-title">Sheet title</h2>
            Sheet body
          </SheetContent>
        </Sheet>
      </>,
      root,
    );

    expect(root.textContent).toContain("Dialog body");
    expect(root.textContent).toContain("Sheet body");

    const dialog = root.querySelector('[role="dialog"][aria-labelledby="dialog-title"]');
    const sheet = root.querySelector('[role="dialog"][aria-labelledby="sheet-title"]');

    expect(dialog?.getAttribute("aria-modal")).toBe("true");
    expect(sheet?.getAttribute("aria-modal")).toBe("true");
  });

  it("moves focus into the dialog, traps Tab navigation, and restores focus on close", async () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    const opener = document.createElement("button");
    opener.textContent = "Open dialog";
    document.body.appendChild(opener);
    opener.focus();

    render(
      <Dialog open>
        <DialogContent titleId="dialog-title">
          <DialogHeader>
            <DialogTitle id="dialog-title">Dialog title</DialogTitle>
          </DialogHeader>
          <button type="button">First action</button>
          <button type="button">Last action</button>
        </DialogContent>
      </Dialog>,
      root,
    );

    await flush();

    const dialog = root.querySelector('[role="dialog"][aria-labelledby="dialog-title"]') as HTMLDivElement | null;
    const dialogButtons = dialog ? (Array.from(dialog.querySelectorAll("button")) as HTMLButtonElement[]) : [];

    expect(document.activeElement).toBe(dialogButtons[0]);

    dialogButtons[1]?.focus();
    await pressKey(dialogButtons[1], "Tab");
    expect(document.activeElement).toBe(dialogButtons[0]);

    dialogButtons[0]?.focus();
    await pressKey(dialogButtons[0], "Tab", { shiftKey: true });
    expect(document.activeElement).toBe(dialogButtons[1]);

    render(
      <Dialog open={false}>
        <DialogContent titleId="dialog-title">
          <DialogHeader>
            <DialogTitle id="dialog-title">Dialog title</DialogTitle>
          </DialogHeader>
        </DialogContent>
      </Dialog>,
      root,
    );

    await flush();

    expect(document.activeElement).toBe(opener);

    opener.remove();
  });

  it("dismisses the dialog when the backdrop is clicked", async () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(<DismissibleDialogHarness />, root);
    await flush();

    const backdrop = root.querySelector('[data-testid="dialog-backdrop"]');
    expect(backdrop).not.toBeNull();
    expect(root.querySelector('[role="dialog"]')?.textContent).toContain("Dialog action");

    await act(async () => {
      backdrop?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    await flush();

    expect(root.querySelector('[role="dialog"]')).toBeNull();
    expect(root.querySelector('[data-testid="dialog-backdrop"]')).toBeNull();
  });

  it("dismisses the dialog when Escape is pressed", async () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(<DismissibleDialogHarness />, root);
    await flush();

    const dialog = root.querySelector('[role="dialog"]') as HTMLDivElement | null;
    expect(dialog).not.toBeNull();

    await pressKey(dialog, "Escape");
    await flush();

    expect(root.querySelector('[role="dialog"]')).toBeNull();
    expect(root.querySelector('[data-testid="dialog-backdrop"]')).toBeNull();
  });

  it("moves focus into the sheet, traps Tab navigation, and restores focus on close", async () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    const opener = document.createElement("button");
    opener.textContent = "Open sheet";
    document.body.appendChild(opener);
    opener.focus();

    render(
      <Sheet open>
        <SheetContent side="left" titleId="sheet-title">
          <h2 id="sheet-title">Sheet title</h2>
          <button type="button">Sheet first</button>
          <button type="button">Sheet last</button>
        </SheetContent>
      </Sheet>,
      root,
    );

    await flush();

    const sheet = root.querySelector('[role="dialog"][aria-labelledby="sheet-title"]') as HTMLDivElement | null;
    const sheetButtons = sheet ? (Array.from(sheet.querySelectorAll("button")) as HTMLButtonElement[]) : [];

    expect(document.activeElement).toBe(sheetButtons[0]);

    sheetButtons[1]?.focus();
    await pressKey(sheetButtons[1], "Tab");
    expect(document.activeElement).toBe(sheetButtons[0]);

    sheetButtons[0]?.focus();
    await pressKey(sheetButtons[0], "Tab", { shiftKey: true });
    expect(document.activeElement).toBe(sheetButtons[1]);

    render(
      <Sheet open={false}>
        <SheetContent side="left" titleId="sheet-title">
          <h2 id="sheet-title">Sheet title</h2>
        </SheetContent>
      </Sheet>,
      root,
    );

    await flush();

    expect(document.activeElement).toBe(opener);

    opener.remove();
  });

  it("renders tabs with tab semantics and selected metadata", async () => {
    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>
        <TabsContent value="overview">Overview panel</TabsContent>
        <TabsContent value="details">Details panel</TabsContent>
      </Tabs>,
      root,
    );

    const tablist = root.querySelector('[role="tablist"]');
    const overviewTab = Array.from(root.querySelectorAll('[role="tab"]')).find((node) => node.textContent === "Overview") as HTMLButtonElement | undefined;
    const detailsTab = Array.from(root.querySelectorAll('[role="tab"]')).find((node) => node.textContent === "Details") as HTMLButtonElement | undefined;
    const panel = root.querySelector('[role="tabpanel"]');

    expect(tablist).not.toBeNull();
    expect(overviewTab?.getAttribute("aria-selected")).toBe("true");
    expect(overviewTab?.getAttribute("data-state")).toBe("active");
    expect(detailsTab?.getAttribute("aria-selected")).toBe("false");
    expect(detailsTab?.getAttribute("data-state")).toBe("inactive");
    expect(panel?.getAttribute("aria-labelledby")).toBe(overviewTab?.id);
    expect(panel?.textContent).toContain("Overview panel");

    detailsTab?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await Promise.resolve();

    const nextDetailsTab = Array.from(root.querySelectorAll('[role="tab"]')).find((node) => node.textContent === "Details") as HTMLButtonElement | undefined;

    expect(nextDetailsTab?.getAttribute("aria-selected")).toBe("true");
    expect(nextDetailsTab?.getAttribute("data-state")).toBe("active");
    expect(root.querySelector('[role="tabpanel"]')?.textContent).toContain("Details panel");
  });

  it("supports arrow, home, and end keyboard navigation for tabs", async () => {
    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
          <TabsTrigger value="activity">Activity</TabsTrigger>
        </TabsList>
        <TabsContent value="overview">Overview panel</TabsContent>
        <TabsContent value="details">Details panel</TabsContent>
        <TabsContent value="activity">Activity panel</TabsContent>
      </Tabs>,
      root,
    );

    const tabs = Array.from(root.querySelectorAll('[role="tab"]')) as HTMLButtonElement[];
    const [overviewTab, detailsTab, activityTab] = tabs;

    overviewTab?.focus();
    await pressKey(overviewTab, "ArrowRight");
    expect(document.activeElement).toBe(detailsTab);
    expect(detailsTab?.getAttribute("aria-selected")).toBe("true");
    expect(root.querySelector('[role="tabpanel"]')?.textContent).toContain("Details panel");

    await pressKey(detailsTab, "End");
    expect(document.activeElement).toBe(activityTab);
    expect(activityTab?.getAttribute("aria-selected")).toBe("true");
    expect(root.querySelector('[role="tabpanel"]')?.textContent).toContain("Activity panel");

    await pressKey(activityTab, "ArrowRight");
    expect(document.activeElement).toBe(overviewTab);
    expect(overviewTab?.getAttribute("aria-selected")).toBe("true");
    expect(root.querySelector('[role="tabpanel"]')?.textContent).toContain("Overview panel");

    await pressKey(overviewTab, "Home");
    expect(document.activeElement).toBe(overviewTab);
    expect(overviewTab?.getAttribute("aria-selected")).toBe("true");

    await pressKey(overviewTab, "ArrowLeft");
    expect(document.activeElement).toBe(activityTab);
    expect(activityTab?.getAttribute("aria-selected")).toBe("true");
    expect(root.querySelector('[role="tabpanel"]')?.textContent).toContain("Activity panel");
  });
});
