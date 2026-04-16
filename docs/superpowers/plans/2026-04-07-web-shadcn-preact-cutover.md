# Web Shadcn-Preact Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the `web/` frontend on top of a `shadcn-preact`-style component foundation while preserving the current `Preact` runtime, existing store boundaries, and current product behavior.

**Architecture:** Keep the current `web/src/app`, `web/src/domains`, and `web/src/lib` boundaries intact, but replace the presentation layer with a Tailwind-token-driven `ui` primitive set under `web/src/components/ui`. Perform the cutover in one shipping pass, while still implementing internally in layers: foundation first, then primitives, then shell and feature surfaces, then cleanup and verification.

**Tech Stack:** `preact`, `typescript`, `vite`, `vitest`, `tailwindcss@3`, `postcss`, `autoprefixer`, `class-variance-authority`, `clsx`, `tailwind-merge`, `tailwindcss-animate`, existing `web/src` stores, Python-served frontend build

---

## File Map

### Existing files to modify

- `web/package.json` - add Tailwind and `shadcn-preact` support dependencies
- `web/vite.config.ts` - add `@` and `@ui` aliases used by the new primitive layer
- `web/tsconfig.app.json` - add path aliases for app code
- `web/tsconfig.test.json` - mirror path aliases for vitest
- `web/tsconfig.node.json` - include alias-friendly config for `vite.config.ts`
- `web/src/main.tsx` - switch from direct `theme.css`/`global.css` imports to a single Tailwind entry stylesheet
- `web/src/styles/theme.css` - replace legacy token names with `shadcn`-compatible CSS variables
- `web/src/styles/global.css` - shrink to app-wide shell/viewport/resets instead of owning every component
- `web/src/app/AppShell.tsx` - rebuild the multi-pane shell around cards, separators, and sheets
- `web/src/components/sessions/SessionsPane.tsx` - render a structured sidebar surface
- `web/src/components/sessions/SessionCard.tsx` - render the new card-like session rows
- `web/src/components/conversation/ConversationPane.tsx` - render the timeline with reusable message surfaces
- `web/src/components/composer/Composer.tsx` - rebuild the prompt workbench around `textarea`, `button`, and cards
- `web/src/components/workspace/SessionWorkspace.tsx` - move diagnostics, queue, files, and requests into tabs/cards
- `web/src/components/new-session/NewSessionDialog.tsx` - rebuild the form on `dialog`-style primitives and grouped field sections
- `web/src/app/AppShell.test.tsx` - update shell assertions to the new structure
- `web/src/components/sessions/SessionsPane.test.tsx` - update session list assertions to the new card structure
- `web/src/components/conversation/ConversationPane.test.tsx` - update message-surface assertions
- `web/src/components/composer/Composer.test.tsx` - update composer structure assertions without losing keyboard coverage
- `web/src/components/workspace/SessionWorkspace.test.tsx` - update workspace structure assertions
- `web/src/components/new-session/NewSessionDialog.test.tsx` - update dialog structure assertions while preserving submit flow coverage

### New files to create

- `web/postcss.config.js` - Tailwind PostCSS pipeline
- `web/tailwind.config.js` - Tailwind theme + content config wired to `shadcn` tokens
- `web/src/styles/index.css` - Tailwind entry sheet importing base layers plus the remaining global CSS
- `web/src/lib/utils.ts` - `cn()` helper using `clsx` + `tailwind-merge`
- `web/src/lib/utils.test.ts` - lock class merging behavior before adding primitives
- `web/src/components/ui/button.tsx` - base action primitive
- `web/src/components/ui/button.test.tsx` - verify variants and class merging
- `web/src/components/ui/card.tsx` - card/content/header/footer primitives
- `web/src/components/ui/badge.tsx` - compact status pill primitive
- `web/src/components/ui/input.tsx` - shared text input primitive
- `web/src/components/ui/textarea.tsx` - shared multiline input primitive
- `web/src/components/ui/separator.tsx` - structural divider primitive
- `web/src/components/ui/skeleton.tsx` - loading placeholder primitive
- `web/src/components/ui/dialog.tsx` - modal wrapper used by new-session flows
- `web/src/components/ui/sheet.tsx` - mobile off-canvas panel wrapper for sidebar/workspace
- `web/src/components/ui/tabs.tsx` - workspace tab primitives
- `web/src/components/ui/scroll-area.tsx` - scroll container primitive for sidebar/timeline/workspace panes
- `web/src/components/ui/dialog.test.tsx` - lock dialog/sheet open-state behavior used by the cutover

### Verification commands used repeatedly

- `cd web && npx vitest run src/lib/utils.test.ts`
- `cd web && npx vitest run src/components/ui/button.test.tsx src/components/ui/dialog.test.tsx`
- `cd web && npx vitest run src/app/AppShell.test.tsx src/components/sessions/SessionsPane.test.tsx`
- `cd web && npx vitest run src/components/conversation/ConversationPane.test.tsx src/components/composer/Composer.test.tsx`
- `cd web && npx vitest run src/components/workspace/SessionWorkspace.test.tsx src/components/new-session/NewSessionDialog.test.tsx`
- `cd web && npm test`
- `cd web && npm run build`
- `python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py tests/test_frontend_contract_source.py -q`

## Task 1: Add the Tailwind and token foundation

**Files:**
- Create: `web/postcss.config.js`
- Create: `web/tailwind.config.js`
- Create: `web/src/styles/index.css`
- Create: `web/src/lib/utils.ts`
- Create: `web/src/lib/utils.test.ts`
- Modify: `web/package.json`
- Modify: `web/vite.config.ts`
- Modify: `web/tsconfig.app.json`
- Modify: `web/tsconfig.test.json`
- Modify: `web/tsconfig.node.json`
- Modify: `web/src/main.tsx`
- Modify: `web/src/styles/theme.css`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Write the failing test for the shared `cn()` helper before adding any Tailwind-driven components**

```ts
// web/src/lib/utils.test.ts
import { describe, expect, it } from "vitest";
import { cn } from "./utils";

describe("cn", () => {
  it("merges Tailwind classes predictably", () => {
    expect(cn("px-3 py-2", "px-4", false && "hidden", "text-sm", "text-sm")).toBe("py-2 px-4 text-sm");
  });
});
```

- [ ] **Step 2: Run the helper test and confirm it fails because the utility and dependencies do not exist yet**

Run: `cd web && npx vitest run src/lib/utils.test.ts`
Expected: FAIL with `Cannot find module './utils'` or `Cannot resolve 'clsx'`.

- [ ] **Step 3: Add Tailwind/PostCSS config, alias support, the `cn()` helper, and the new stylesheet entry point**

```json
// web/package.json
{
  "dependencies": {
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "preact": "^10.26.4",
    "tailwind-merge": "^3.3.1"
  },
  "devDependencies": {
    "autoprefixer": "^10.4.21",
    "postcss": "^8.5.6",
    "tailwindcss": "^3.4.17",
    "tailwindcss-animate": "^1.0.7"
  }
}
```

```js
// web/postcss.config.js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

```js
// web/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" },
        popover: { DEFAULT: "hsl(var(--popover))", foreground: "hsl(var(--popover-foreground))" },
        primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
        destructive: { DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [await import("tailwindcss-animate").then((mod) => mod.default)],
};
```

```ts
// web/src/lib/utils.ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

```css
/* web/src/styles/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@import "./theme.css";
@import "./global.css";

@layer base {
  * {
    @apply border-border;
  }

  html,
  body,
  #root {
    min-height: 100%;
  }

  body {
    @apply bg-background text-foreground antialiased;
  }
}
```

```ts
// web/src/main.tsx
import { render } from "preact";
import App from "./app/App";
import "./styles/index.css";
```

```ts
// web/vite.config.ts
import { resolve } from "node:path";

resolve: {
  alias: {
    "@": resolve(__dirname, "./src"),
    "@ui": resolve(__dirname, "./src/components/ui"),
  },
},
```

```json
// web/tsconfig.app.json and web/tsconfig.test.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@ui/*": ["./src/components/ui/*"]
    }
  }
}
```

```css
/* web/src/styles/theme.css */
:root {
  color-scheme: light;
  --background: 210 35% 96%;
  --foreground: 222 47% 11%;
  --card: 0 0% 100%;
  --card-foreground: 222 47% 11%;
  --popover: 0 0% 100%;
  --popover-foreground: 222 47% 11%;
  --primary: 221 83% 53%;
  --primary-foreground: 210 40% 98%;
  --secondary: 210 40% 96%;
  --secondary-foreground: 222 47% 11%;
  --muted: 214 32% 91%;
  --muted-foreground: 215 16% 40%;
  --accent: 206 80% 94%;
  --accent-foreground: 222 47% 11%;
  --destructive: 0 72% 51%;
  --destructive-foreground: 210 40% 98%;
  --border: 214 24% 84%;
  --input: 214 24% 84%;
  --ring: 221 83% 53%;
  --radius: 1rem;
}
```

```css
/* web/src/styles/global.css */
:root {
  --sidebar-w: 23rem;
  --workspace-w: 24rem;
}

body {
  margin: 0;
  min-width: 320px;
  background:
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(241, 245, 249, 0.98));
}

.liveAudioElement {
  display: none;
}
```

- [ ] **Step 4: Re-run the helper test and a full frontend build to verify the foundation is wired correctly**

Run: `cd web && npx vitest run src/lib/utils.test.ts && npm run build`
Expected: PASS for the helper test, then PASS for the build with generated assets in `web/dist`.

- [ ] **Step 5: Commit the foundation before adding any primitives**

```bash
git add web/package.json web/postcss.config.js web/tailwind.config.js web/src/styles/index.css web/src/lib/utils.ts web/src/lib/utils.test.ts web/vite.config.ts web/tsconfig.app.json web/tsconfig.test.json web/tsconfig.node.json web/src/main.tsx web/src/styles/theme.css web/src/styles/global.css
git commit -m "feat(web): add shadcn-preact styling foundation"
```

## Task 2: Add the reusable primitive layer under `web/src/components/ui`

**Files:**
- Create: `web/src/components/ui/button.tsx`
- Create: `web/src/components/ui/button.test.tsx`
- Create: `web/src/components/ui/card.tsx`
- Create: `web/src/components/ui/badge.tsx`
- Create: `web/src/components/ui/input.tsx`
- Create: `web/src/components/ui/textarea.tsx`
- Create: `web/src/components/ui/separator.tsx`
- Create: `web/src/components/ui/skeleton.tsx`

- [ ] **Step 1: Write the failing primitive test around `Button` variants and class merging**

```tsx
// web/src/components/ui/button.test.tsx
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
```

- [ ] **Step 2: Run the primitive test and confirm it fails because the UI layer does not exist yet**

Run: `cd web && npx vitest run src/components/ui/button.test.tsx`
Expected: FAIL with `Cannot find module './button'`.

- [ ] **Step 3: Add the shared UI primitives used by every major surface in the cutover**

```tsx
// web/src/components/ui/button.tsx
import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "preact/compat";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground shadow hover:opacity-95",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        outline: "border bg-card text-card-foreground hover:bg-accent hover:text-accent-foreground",
        ghost: "hover:bg-accent hover:text-accent-foreground",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 px-3",
        lg: "h-11 px-6",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
```

```tsx
// web/src/components/ui/card.tsx
import type { ComponentChildren } from "preact";
import { cn } from "@/lib/utils";

export function Card({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("rounded-2xl border bg-card text-card-foreground shadow-sm", className)}>{children}</div>;
}

export function CardHeader({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("flex flex-col space-y-1.5 p-5", className)}>{children}</div>;
}

export function CardTitle({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <h3 className={cn("text-sm font-semibold tracking-tight", className)}>{children}</h3>;
}

export function CardContent({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("p-5 pt-0", className)}>{children}</div>;
}
```

```tsx
// web/src/components/ui/badge.tsx
import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "preact/compat";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium", {
  variants: {
    variant: {
      default: "bg-primary/10 text-primary",
      secondary: "bg-secondary text-secondary-foreground",
      outline: "border border-border text-muted-foreground",
    },
  },
  defaultVariants: { variant: "default" },
});

export function Badge({ className, variant, ...props }: HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeVariants>) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
```

```tsx
// web/src/components/ui/input.tsx and textarea.tsx
import type { JSX } from "preact";
import { cn } from "@/lib/utils";

export function Input(props: JSX.HTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn("flex h-10 w-full rounded-xl border bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", props.className)} />;
}

export function Textarea(props: JSX.HTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cn("flex min-h-[120px] w-full rounded-2xl border bg-background px-3 py-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", props.className)} />;
}
```

```tsx
// web/src/components/ui/separator.tsx and skeleton.tsx
import { cn } from "@/lib/utils";

export function Separator({ className = "", orientation = "horizontal" }: { className?: string; orientation?: "horizontal" | "vertical" }) {
  return <div className={cn(orientation === "horizontal" ? "h-px w-full" : "h-full w-px", "bg-border", className)} aria-hidden="true" />;
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-muted", className)} />;
}
```

- [ ] **Step 4: Re-run the primitive test to verify the new base layer is usable before feature components depend on it**

Run: `cd web && npx vitest run src/components/ui/button.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit the base primitive layer**

```bash
git add web/src/components/ui/button.tsx web/src/components/ui/button.test.tsx web/src/components/ui/card.tsx web/src/components/ui/badge.tsx web/src/components/ui/input.tsx web/src/components/ui/textarea.tsx web/src/components/ui/separator.tsx web/src/components/ui/skeleton.tsx
git commit -m "feat(web): add base shadcn-preact primitives"
```

## Task 3: Add overlay and layout primitives for the shell cutover

**Files:**
- Create: `web/src/components/ui/dialog.tsx`
- Create: `web/src/components/ui/sheet.tsx`
- Create: `web/src/components/ui/tabs.tsx`
- Create: `web/src/components/ui/scroll-area.tsx`
- Create: `web/src/components/ui/dialog.test.tsx`

- [ ] **Step 1: Write the failing test for modal and sheet open-state behavior used by the shell and dialog surfaces**

```tsx
// web/src/components/ui/dialog.test.tsx
import { render } from "preact";
import { afterEach, describe, expect, it } from "vitest";
import { Dialog, DialogContent } from "./dialog";
import { Sheet, SheetContent } from "./sheet";

describe("overlay primitives", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders dialog and sheet content only while open", () => {
    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <>
        <Dialog open>
          <DialogContent>Dialog body</DialogContent>
        </Dialog>
        <Sheet open>
          <SheetContent side="left">Sheet body</SheetContent>
        </Sheet>
      </>,
      root,
    );

    expect(root.textContent).toContain("Dialog body");
    expect(root.textContent).toContain("Sheet body");
  });
});
```

- [ ] **Step 2: Run the overlay test and confirm it fails before those primitives are created**

Run: `cd web && npx vitest run src/components/ui/dialog.test.tsx`
Expected: FAIL with `Cannot find module './dialog'` or `./sheet`.

- [ ] **Step 3: Add the shell-facing primitives with a minimal, project-owned API surface**

```tsx
// web/src/components/ui/dialog.tsx
import type { ComponentChildren } from "preact";
import { cn } from "@/lib/utils";

export function Dialog({ open, children }: { open?: boolean; children?: ComponentChildren }) {
  if (!open) return null;
  return <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/45 p-4">{children}</div>;
}

export function DialogContent({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("w-full max-w-3xl rounded-3xl border bg-card text-card-foreground shadow-2xl", className)}>{children}</div>;
}

export function DialogHeader({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("flex flex-col gap-1 p-6 pb-0", className)}>{children}</div>;
}

export function DialogTitle({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <h2 className={cn("text-lg font-semibold", className)}>{children}</h2>;
}
```

```tsx
// web/src/components/ui/sheet.tsx
import type { ComponentChildren } from "preact";
import { cn } from "@/lib/utils";

export function Sheet({ open, children }: { open?: boolean; children?: ComponentChildren }) {
  if (!open) return null;
  return <div className="fixed inset-0 z-40 bg-slate-950/40">{children}</div>;
}

export function SheetContent({ side = "left", className, children }: { side?: "left" | "right"; className?: string; children?: ComponentChildren }) {
  const sideClass = side === "right" ? "right-0" : "left-0";
  return <div className={cn("absolute top-0 h-full w-[min(24rem,92vw)] border bg-card p-4 shadow-xl", sideClass, className)}>{children}</div>;
}
```

```tsx
// web/src/components/ui/tabs.tsx
import { createContext } from "preact";
import { useContext, useState, type ComponentChildren } from "preact/hooks";
import { cn } from "@/lib/utils";

const TabsContext = createContext<{ value: string; setValue: (value: string) => void } | null>(null);

export function Tabs({ defaultValue, children, className }: { defaultValue: string; children?: ComponentChildren; className?: string }) {
  const [value, setValue] = useState(defaultValue);
  return (
    <TabsContext.Provider value={{ value, setValue }}>
      <div className={cn("flex h-full flex-col gap-4", className)}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("inline-flex h-10 items-center rounded-xl bg-secondary p-1 text-secondary-foreground", className)}>{children}</div>;
}

export function TabsTrigger({ value, className, children }: { value: string; className?: string; children?: ComponentChildren }) {
  const context = useContext(TabsContext);
  const active = context?.value === value;
  return (
    <button
      type="button"
      data-testid="workspace-tab"
      className={cn("inline-flex items-center rounded-lg px-3 py-1.5 text-sm font-medium transition-colors", active && "bg-background shadow-sm", className)}
      onClick={() => context?.setValue(value)}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, className, children }: { value: string; className?: string; children?: ComponentChildren }) {
  const context = useContext(TabsContext);
  if (context?.value !== value) return null;
  return <div className={cn("min-h-0 flex-1", className)}>{children}</div>;
}
```

```tsx
// web/src/components/ui/scroll-area.tsx
import type { ComponentChildren } from "preact";
import { cn } from "@/lib/utils";

export function ScrollArea({ className, children }: { className?: string; children?: ComponentChildren }) {
  return <div className={cn("min-h-0 overflow-auto", className)}>{children}</div>;
}
```

- [ ] **Step 4: Re-run the overlay test so shell and dialog tasks can build on a stable base**

Run: `cd web && npx vitest run src/components/ui/dialog.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit the overlay/layout primitive layer**

```bash
git add web/src/components/ui/dialog.tsx web/src/components/ui/sheet.tsx web/src/components/ui/tabs.tsx web/src/components/ui/scroll-area.tsx web/src/components/ui/dialog.test.tsx
git commit -m "feat(web): add overlay and layout ui primitives"
```

## Task 4: Rebuild the app shell and sessions sidebar on the new primitives

**Files:**
- Modify: `web/src/app/AppShell.tsx`
- Modify: `web/src/app/AppShell.test.tsx`
- Modify: `web/src/components/sessions/SessionsPane.tsx`
- Modify: `web/src/components/sessions/SessionCard.tsx`
- Modify: `web/src/components/sessions/SessionsPane.test.tsx`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Tighten the shell and session tests so they fail until the new card-and-sheet layout exists**

```tsx
// web/src/components/sessions/SessionsPane.test.tsx
expect(root.querySelector("[data-testid='sessions-surface']")).not.toBeNull();
expect(root.querySelectorAll("[data-testid='session-card']")).toHaveLength(1);
expect(root.querySelector("[data-testid='session-card'].ring-2")).not.toBeNull();
expect(root.textContent).toContain("Inbox cleanup");
expect(root.textContent).toContain("pi");
expect(root.textContent).toContain("web");
```

```tsx
// web/src/app/AppShell.test.tsx
expect(getRoot().querySelector("[data-testid='app-shell']")).not.toBeNull();
expect(getRoot().querySelector("[data-testid='mobile-sessions-sheet']")).not.toBeNull();
expect(getRoot().querySelector("[data-testid='workspace-rail']")).not.toBeNull();
expect(getRoot().textContent).toContain("New session");
```

- [ ] **Step 2: Run the shell/sidebar tests and confirm they fail against the current CSS-driven layout**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx src/components/sessions/SessionsPane.test.tsx`
Expected: FAIL because the new `data-testid` hooks and primitive-driven layout are not present yet.

- [ ] **Step 3: Rebuild the shell and session surfaces using `Card`, `Button`, `Badge`, `ScrollArea`, and `Sheet`**

```tsx
// web/src/components/sessions/SessionCard.tsx
import { Badge } from "@ui/badge";
import { Card, CardContent } from "@ui/card";
import { cn } from "@/lib/utils";

export function SessionCard({ session, active, onSelect }: SessionCardProps) {
  const title = session.alias || session.first_user_message || session.title || shortSessionId(session.session_id);
  const preview = (session.alias ? session.first_user_message : session.cwd) || session.cwd || session.title || session.session_id;

  return (
    <button type="button" data-testid="session-card" onClick={onSelect} className="w-full text-left">
      <Card className={cn("transition-all hover:-translate-y-0.5 hover:shadow-md", active && "ring-2 ring-primary") }>
        <CardContent className="space-y-3 p-4">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className={cn("h-2.5 w-2.5 rounded-full bg-emerald-500", session.busy && "animate-pulse bg-amber-500")} />
            <Badge variant="outline">{session.agent_backend || "codex"}</Badge>
            {session.owned ? <Badge variant="secondary">web</Badge> : null}
            {session.queue_len ? <Badge>{session.queue_len}</Badge> : null}
          </div>
          <div className="space-y-1">
            <div className="line-clamp-2 text-sm font-semibold text-foreground">{title}</div>
            <div className="line-clamp-2 text-xs text-muted-foreground">{preview}</div>
          </div>
        </CardContent>
      </Card>
    </button>
  );
}
```

```tsx
// web/src/components/sessions/SessionsPane.tsx
import { Button } from "@ui/button";
import { ScrollArea } from "@ui/scroll-area";

export function SessionsPane() {
  const { items, activeSessionId } = useSessionsStore();
  const sessionsStoreApi = useSessionsStoreApi();

  return (
    <aside data-testid="sessions-surface" className="flex h-full min-h-0 flex-col gap-4 rounded-[28px] border bg-card/90 p-4 shadow-sm backdrop-blur">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-[0.24em] text-muted-foreground">Codoxear</div>
          <div className="text-lg font-semibold">Sessions</div>
        </div>
        <Button size="sm">New session</Button>
      </div>
      <ScrollArea className="flex-1 pr-1">
        <div className="space-y-3">
          {items.map((session) => (
            <SessionCard
              key={session.session_id}
              session={session}
              active={session.session_id === activeSessionId}
              onSelect={() => sessionsStoreApi.select(session.session_id)}
            />
          ))}
        </div>
      </ScrollArea>
    </aside>
  );
}
```

```tsx
// web/src/app/AppShell.tsx
import { Button } from "@ui/button";
import { Sheet, SheetContent } from "@ui/sheet";

export function AppShell() {
  const { activeSessionId } = useSessionsStore();
  const [newSessionOpen, setNewSessionOpen] = useState(false);
  const [mobileSessionsOpen, setMobileSessionsOpen] = useState(false);
  const [mobileWorkspaceOpen, setMobileWorkspaceOpen] = useState(false);

  return (
    <div data-testid="app-shell" className="grid h-dvh grid-cols-1 gap-4 p-3 lg:grid-cols-[23rem_minmax(0,1fr)_24rem] lg:p-4">
      <div className="hidden lg:block">
        <SessionsPane />
      </div>
      <section className="flex min-h-0 flex-col gap-4">
        <div className="flex items-center justify-between lg:hidden">
          <Button variant="outline" size="sm" onClick={() => setMobileSessionsOpen(true)}>Sessions</Button>
          <Button variant="outline" size="sm" onClick={() => setNewSessionOpen(true)}>New session</Button>
          <Button variant="outline" size="sm" onClick={() => setMobileWorkspaceOpen(Boolean(activeSessionId))}>Workspace</Button>
        </div>
        <ConversationPane />
        <Composer />
      </section>
      <aside data-testid="workspace-rail" className="hidden min-h-0 lg:block">
        <SessionWorkspace />
      </aside>
      <Sheet open={mobileSessionsOpen}>
        <SheetContent side="left" className="p-0" >
          <div data-testid="mobile-sessions-sheet" className="h-full p-4"><SessionsPane /></div>
        </SheetContent>
      </Sheet>
      <Sheet open={mobileWorkspaceOpen}>
        <SheetContent side="right" className="p-4">
          <SessionWorkspace />
        </SheetContent>
      </Sheet>
      <NewSessionDialog open={newSessionOpen} onClose={() => setNewSessionOpen(false)} />
    </div>
  );
}
```

- [ ] **Step 4: Re-run the shell/sidebar tests to lock the new shell before touching the conversation and dialog surfaces**

Run: `cd web && npx vitest run src/app/AppShell.test.tsx src/components/sessions/SessionsPane.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit the shell/sidebar cutover**

```bash
git add web/src/app/AppShell.tsx web/src/app/AppShell.test.tsx web/src/components/sessions/SessionsPane.tsx web/src/components/sessions/SessionCard.tsx web/src/components/sessions/SessionsPane.test.tsx web/src/styles/global.css
git commit -m "feat(web): rebuild shell and sessions sidebar"
```

## Task 5: Rebuild the conversation timeline and composer workbench

**Files:**
- Modify: `web/src/components/conversation/ConversationPane.tsx`
- Modify: `web/src/components/conversation/ConversationPane.test.tsx`
- Modify: `web/src/components/composer/Composer.tsx`
- Modify: `web/src/components/composer/Composer.test.tsx`

- [ ] **Step 1: Update the timeline and composer tests so they fail until the new primitives are actually being used**

```tsx
// web/src/components/conversation/ConversationPane.test.tsx
expect(root.querySelectorAll("[data-testid='message-surface']").length).toBeGreaterThanOrEqual(4);
expect(root.querySelector("[data-testid='message-surface'][data-kind='user']")).not.toBeNull();
expect(root.querySelector("[data-testid='message-surface'][data-kind='assistant']")).not.toBeNull();
expect(root.querySelector("[data-testid='message-surface'][data-kind='ask_user']")).not.toBeNull();
```

```tsx
// web/src/components/composer/Composer.test.tsx
expect(root.querySelector("[data-testid='composer-card']")).not.toBeNull();
expect(root.querySelector("textarea.rounded-2xl")).not.toBeNull();
expect(root.querySelector("button[type='submit']")).not.toBeNull();
```

- [ ] **Step 2: Run the conversation/composer tests and confirm they fail before the view rewrite starts**

Run: `cd web && npx vitest run src/components/conversation/ConversationPane.test.tsx src/components/composer/Composer.test.tsx`
Expected: FAIL because the new `data-testid` hooks and primitive-driven markup do not exist yet.

- [ ] **Step 3: Rewrite the timeline and composer around cards, badges, scroll areas, and the shared textarea/button primitives**

```tsx
// web/src/components/conversation/ConversationPane.tsx
import { Badge } from "@ui/badge";
import { Card, CardContent } from "@ui/card";
import { ScrollArea } from "@ui/scroll-area";
import { Skeleton } from "@ui/skeleton";
import { cn } from "@/lib/utils";

function surfaceTone(kind: string) {
  if (kind === "user") return "bg-primary/10 border-primary/20";
  if (kind === "assistant") return "bg-card";
  return "bg-muted/60";
}

function renderConversationEvent(event: MessageEvent, kind: string) {
  return (
    <Card data-testid="message-surface" data-kind={kind} className={cn("overflow-hidden border", surfaceTone(kind))}>
      <CardContent className="space-y-3 p-4">
        <div className="flex items-center justify-between gap-2">
          <Badge variant={kind === "assistant" ? "secondary" : "outline"}>{eventLabel(kind)}</Badge>
          {event.summary ? <span className="text-xs text-muted-foreground">{event.summary}</span> : null}
        </div>
        {renderRichText(contentTextFromMessage(event), "prose prose-sm max-w-none text-foreground")}
      </CardContent>
    </Card>
  );
}

export function ConversationPane() {
  if (loading) {
    return <div className="space-y-3">{Array.from({ length: 4 }, (_, index) => <Skeleton key={index} className="h-24 rounded-2xl" />)}</div>;
  }

  return (
    <Card className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[28px] bg-card/90 backdrop-blur">
      <div className="border-b px-5 py-4">
        <div className="text-sm font-semibold">Conversation</div>
      </div>
      <ScrollArea className="flex-1 px-4 py-4">
        <div className="space-y-4">{timeline.map((item) => renderConversationEvent(item.event, item.kind))}</div>
      </ScrollArea>
    </Card>
  );
}
```

```tsx
// web/src/components/composer/Composer.tsx
import { Button } from "@ui/button";
import { Card, CardContent } from "@ui/card";
import { Textarea } from "@ui/textarea";

export function Composer() {
  return (
    <Card data-testid="composer-card" className="rounded-[28px] bg-card/95 shadow-sm">
      <CardContent className="space-y-3 p-4">
        {todoSnapshot ? (
          <TodoComposerPanel
            snapshot={todoSnapshot}
            expanded={visibleTodoExpanded}
            onToggle={() => {
              if (!activeSessionId) return;
              setTodoExpandedBySessionId((value) => ({ ...value, [activeSessionId]: !value[activeSessionId] }));
            }}
          />
        ) : null}
        <form
          className="flex items-end gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (activeSessionId) {
              composerStoreApi.submit(activeSessionId);
            }
          }}
        >
          <Button type="button" variant="outline" size="icon" aria-label="Attach file">
            <span className="buttonGlyph">📎</span>
          </Button>
          <div className="flex-1">
            <Textarea
              value={draft}
              className="min-h-[96px] resize-none border-0 bg-secondary/60 shadow-none"
              onInput={(event) => composerStoreApi.setDraft(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key !== "Enter" || event.isComposing || event.shiftKey) return;
                if (!enterToSendEnabled() && !event.ctrlKey && !event.metaKey) return;
                if (!activeSessionId) return;
                event.preventDefault();
                composerStoreApi.submit(activeSessionId).catch(() => undefined);
              }}
              disabled={sending}
              placeholder="Enter your instructions here"
            />
          </div>
          <div className="flex gap-2">
            <Button type="button" variant="outline" size="icon" aria-label="Queued messages">
              <span className="buttonGlyph">≡</span>
            </Button>
            <Button type="submit" disabled={sending || !draft.trim()}>
              {sending ? "Sending…" : "Send"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 4: Re-run the timeline and composer tests to confirm the new shell preserves behavior while changing presentation**

Run: `cd web && npx vitest run src/components/conversation/ConversationPane.test.tsx src/components/composer/Composer.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit the conversation/composer rewrite**

```bash
git add web/src/components/conversation/ConversationPane.tsx web/src/components/conversation/ConversationPane.test.tsx web/src/components/composer/Composer.tsx web/src/components/composer/Composer.test.tsx
git commit -m "feat(web): rebuild conversation and composer surfaces"
```

## Task 6: Rebuild the workspace and new-session dialog, then verify the full cutover

**Files:**
- Modify: `web/src/components/workspace/SessionWorkspace.tsx`
- Modify: `web/src/components/workspace/SessionWorkspace.test.tsx`
- Modify: `web/src/components/new-session/NewSessionDialog.tsx`
- Modify: `web/src/components/new-session/NewSessionDialog.test.tsx`
- Modify: `web/src/styles/global.css`

- [ ] **Step 1: Update the workspace and new-session dialog tests so they fail until the new tabbed/card modal structure is present**

```tsx
// web/src/components/workspace/SessionWorkspace.test.tsx
expect(root.querySelector("[data-testid='workspace-card']")).not.toBeNull();
expect(root.querySelectorAll("[data-testid='workspace-tab']").length).toBeGreaterThanOrEqual(3);
expect(root.textContent).toContain("Diagnostics");
expect(root.textContent).toContain("Queue");
```

```tsx
// web/src/components/new-session/NewSessionDialog.test.tsx
expect(root.querySelector("[data-testid='new-session-dialog']")).not.toBeNull();
expect(root.querySelector("[data-testid='backend-tab-codex']")).not.toBeNull();
expect(root.querySelector("input[name='cwd']")).not.toBeNull();
expect(root.querySelector("button[type='submit']")).not.toBeNull();
```

- [ ] **Step 2: Run the workspace/dialog tests and confirm they fail before the final surface rewrite**

Run: `cd web && npx vitest run src/components/workspace/SessionWorkspace.test.tsx src/components/new-session/NewSessionDialog.test.tsx`
Expected: FAIL because the tabbed workspace card and dialog markup are not present yet.

- [ ] **Step 3: Rebuild the workspace and dialog on `Card`, `Tabs`, `Dialog`, `Input`, `Button`, `Badge`, and `Separator`**

```tsx
// web/src/components/workspace/SessionWorkspace.tsx
import { Badge } from "@ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@ui/card";
import { ScrollArea } from "@ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@ui/tabs";

export function SessionWorkspace({ mode = "default" }: SessionWorkspaceProps) {
  return (
    <Card data-testid="workspace-card" className="flex h-full min-h-0 flex-col rounded-[28px] bg-card/95 shadow-sm">
      <CardHeader className="border-b pb-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Workspace</CardTitle>
            <p className="text-sm text-muted-foreground">Files, queue, diagnostics, and session requests.</p>
          </div>
          {loading ? <Badge variant="secondary">Loading</Badge> : null}
        </div>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 p-4">
        <Tabs defaultValue="diagnostics" className="h-full">
          <TabsList>
            <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
            <TabsTrigger value="queue">Queue</TabsTrigger>
            <TabsTrigger value="requests">Requests</TabsTrigger>
          </TabsList>
          <TabsContent value="diagnostics">
            <ScrollArea className="h-full pr-1">
              <div className="space-y-3 text-sm text-muted-foreground">{diagnosticsEntries.map(([key, value]) => <div key={key}>{key}: {String(value)}</div>)}</div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="queue">
            <ScrollArea className="h-full pr-1">
              <div className="space-y-3 text-sm text-muted-foreground">{queueItems.length ? queueItems.map((item, index) => <div key={`${item}-${index}`}>{item}</div>) : <div>No queued items</div>}</div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="requests">
            <ScrollArea className="h-full pr-1">
              <div className="space-y-3 text-sm text-muted-foreground">{requests.length ? `${requests.length} pending request(s)` : "No pending requests"}</div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
```

```tsx
// web/src/components/new-session/NewSessionDialog.tsx
import { Button } from "@ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@ui/dialog";
import { Input } from "@ui/input";
import { Badge } from "@ui/badge";
import { Separator } from "@ui/separator";

export function NewSessionDialog({ open, onClose }: NewSessionDialogProps) {
  return (
    <Dialog open={open}>
      <DialogContent className="overflow-hidden" >
        <div data-testid="new-session-dialog" className="grid gap-0 lg:grid-cols-[16rem_minmax(0,1fr)]">
          <div className="border-b bg-secondary/50 p-6 lg:border-b-0 lg:border-r">
            <DialogHeader className="p-0">
              <DialogTitle>New session</DialogTitle>
            </DialogHeader>
            <p className="mt-2 text-sm text-muted-foreground">Launch a new Codex or Pi session with project-aware defaults.</p>
          </div>
          <form
            className="space-y-5 p-6"
            onSubmit={(event) => {
              event.preventDefault();
              void handleSubmit(event);
            }}
          >
            <div className="flex gap-2">
              {backendNames.map((name) => (
                <button key={name} type="button" data-testid={`backend-tab-${name}`} className={name === backend ? "rounded-full bg-primary px-3 py-1 text-primary-foreground" : "rounded-full bg-secondary px-3 py-1"} onClick={() => setBackend(name)}>
                  {name}
                </button>
              ))}
            </div>
            <Separator />
            <label className="space-y-2 text-sm font-medium">
              Working directory
              <Input name="cwd" value={cwd} onInput={(event) => setCwd(event.currentTarget.value)} />
            </label>
            <label className="space-y-2 text-sm font-medium">
              Session name
              <Input value={sessionName} onInput={(event) => setSessionName(event.currentTarget.value)} />
            </label>
            <div className="flex items-center justify-between gap-3">
              <div className="flex gap-2">
                {supportsFast ? <Badge variant="outline">Fast mode available</Badge> : null}
                {supportsWorktree ? <Badge variant="outline">Worktree supported</Badge> : null}
              </div>
              <div className="flex gap-2">
                <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
                <Button type="submit" disabled={submitting}>{submitting ? "Starting…" : "Start session"}</Button>
              </div>
            </div>
          </form>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

- [ ] **Step 4: Run the final targeted tests, then the full frontend and backend verification suite**

Run: `cd web && npx vitest run src/components/workspace/SessionWorkspace.test.tsx src/components/new-session/NewSessionDialog.test.tsx && npm test && npm run build`
Expected: PASS across the targeted tests, the full frontend test suite, and the production build.

Run: `python3 -m pytest tests/test_vite_dist_serving.py tests/test_vite_asset_versioning.py tests/test_frontend_contract_source.py -q`
Expected: PASS, confirming the Python side still serves the Vite build contract correctly.

- [ ] **Step 5: Commit the final surface rewrite and verification pass**

```bash
git add web/src/components/workspace/SessionWorkspace.tsx web/src/components/workspace/SessionWorkspace.test.tsx web/src/components/new-session/NewSessionDialog.tsx web/src/components/new-session/NewSessionDialog.test.tsx web/src/styles/global.css
git commit -m "feat(web): complete shadcn-preact ui cutover"
```

## Spec Coverage Check

- Tailwind, token model, and `shadcn-preact` foundation are covered in Task 1.
- Project-owned primitive layer under `web/src/components/ui/` is covered in Tasks 2 and 3.
- App shell, mobile sheets, sessions pane, and session cards are covered in Task 4.
- Conversation pane and composer workbench are covered in Task 5.
- Workspace and new-session dialog cutover are covered in Task 6.
- Desktop/mobile verification, full build verification, and Python contract verification are covered in Task 6.

## Placeholder Scan

- No `TODO`, `TBD`, `implement later`, or `similar to Task N` placeholders remain.
- Every task includes exact file paths, commands, and concrete code snippets.

## Type Consistency Check

- Aliases consistently use `@/` for app code and `@ui/` for primitives.
- The primitive names used later (`Button`, `Card`, `Dialog`, `Sheet`, `Tabs`, `ScrollArea`) are all defined in earlier tasks.
- Shell and feature tasks depend only on the primitives introduced earlier in the plan.
