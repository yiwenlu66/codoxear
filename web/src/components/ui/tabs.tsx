import { createContext, type ComponentChildren } from "preact";
import { useContext, useId, useState } from "preact/hooks";

import { cn } from "@/lib/utils";

interface TabsContextValue {
  baseId: string;
  value: string;
  setValue: (value: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

export interface TabsProps {
  defaultValue: string;
  children?: ComponentChildren;
  className?: string;
}

export interface TabsListProps {
  className?: string;
  children?: ComponentChildren;
}

export interface TabsTriggerProps {
  value: string;
  className?: string;
  children?: ComponentChildren;
}

export interface TabsContentProps {
  value: string;
  className?: string;
  children?: ComponentChildren;
}

function tabValueSlug(value: string) {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "") || "tab";
}

function triggerId(baseId: string, value: string) {
  return `${baseId}-tab-${tabValueSlug(value)}`;
}

function panelId(baseId: string, value: string) {
  return `${baseId}-panel-${tabValueSlug(value)}`;
}

export function Tabs({ defaultValue, children, className }: TabsProps) {
  const baseId = useId();
  const [value, setValue] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ baseId, value, setValue }}>
      <div className={cn("flex h-full flex-col gap-4", className)}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ className, children }: TabsListProps) {
  const context = useContext(TabsContext);

  function handleKeyDown(event: KeyboardEvent) {
    const target = event.target instanceof HTMLElement ? event.target.closest<HTMLElement>('[role="tab"]') : null;

    if (!context || !target) {
      return;
    }

    const tablist = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    const tabs = tablist ? (Array.from(tablist.querySelectorAll<HTMLElement>('[role="tab"]')) as HTMLElement[]) : [];
    const currentIndex = tabs.indexOf(target);

    if (currentIndex < 0) {
      return;
    }

    let nextIndex = -1;

    switch (event.key) {
      case "ArrowRight":
        nextIndex = (currentIndex + 1) % tabs.length;
        break;
      case "ArrowLeft":
        nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        break;
      case "Home":
        nextIndex = 0;
        break;
      case "End":
        nextIndex = tabs.length - 1;
        break;
      default:
        return;
    }

    event.preventDefault();

    const nextTab = tabs[nextIndex];
    const nextValue = nextTab?.dataset.value;

    if (!nextTab || !nextValue) {
      return;
    }

    context.setValue(nextValue);
    nextTab.focus();
  }

  return (
    <div
      role="tablist"
      onKeyDown={handleKeyDown}
      className={cn("inline-flex h-10 items-center rounded-xl bg-secondary p-1 text-secondary-foreground", className)}
    >
      {children}
    </div>
  );
}

export function TabsTrigger({ value, className, children }: TabsTriggerProps) {
  const context = useContext(TabsContext);
  const active = context?.value === value;
  const currentTriggerId = context ? triggerId(context.baseId, value) : undefined;
  const currentPanelId = context ? panelId(context.baseId, value) : undefined;

  return (
    <button
      type="button"
      id={currentTriggerId}
      role="tab"
      tabIndex={active ? 0 : -1}
      aria-selected={active}
      aria-controls={currentPanelId}
      data-value={value}
      data-state={active ? "active" : "inactive"}
      data-testid="workspace-tab"
      className={cn("inline-flex items-center rounded-lg px-3 py-1.5 text-sm font-medium transition-colors", active && "bg-background shadow-sm", className)}
      onClick={() => context?.setValue(value)}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, className, children }: TabsContentProps) {
  const context = useContext(TabsContext);

  if (!context || context.value !== value) {
    return null;
  }

  return (
    <div
      id={panelId(context.baseId, value)}
      role="tabpanel"
      aria-labelledby={triggerId(context.baseId, value)}
      data-state="active"
      className={cn("min-h-0 flex-1", className)}
    >
      {children}
    </div>
  );
}
