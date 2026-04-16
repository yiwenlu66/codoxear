import type { ComponentChildren } from "preact";
import { useLayoutEffect, useRef } from "preact/hooks";

import { cn } from "@/lib/utils";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

function getFocusableElements(container: HTMLElement) {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) => !element.hasAttribute("disabled") && element.getAttribute("aria-hidden") !== "true",
  );
}

export interface SheetProps {
  open?: boolean;
  children?: ComponentChildren;
}

export interface SheetContentProps {
  side?: "left" | "right";
  className?: string;
  children?: ComponentChildren;
  titleId?: string;
  ariaLabel?: string;
}

export function Sheet({ open, children }: SheetProps) {
  if (!open) {
    return null;
  }

  return <div className="fixed inset-0 z-40 bg-slate-950/40">{children}</div>;
}

export function SheetContent({ side = "left", className, children, titleId, ariaLabel }: SheetContentProps) {
  const sideClass = side === "right" ? "right-0" : "left-0";
  const contentRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useLayoutEffect(() => {
    const content = contentRef.current;

    if (!content) {
      return undefined;
    }

    previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const [firstFocusable] = getFocusableElements(content);
    (firstFocusable ?? content).focus();

    return () => {
      const previousFocus = previousFocusRef.current;
      const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;

      if (
        previousFocus &&
        previousFocus.isConnected &&
        (!activeElement || activeElement === document.body || content.contains(activeElement))
      ) {
        queueMicrotask(() => {
          if (previousFocus.isConnected) {
            previousFocus.focus();
          }
        });
      }
    };
  }, []);

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key !== "Tab") {
      return;
    }

    const content = contentRef.current;

    if (!content) {
      return;
    }

    const focusableElements = getFocusableElements(content);

    if (focusableElements.length === 0) {
      event.preventDefault();
      content.focus();
      return;
    }

    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];
    const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    if (event.shiftKey) {
      if (!activeElement || activeElement === firstFocusable || !content.contains(activeElement)) {
        event.preventDefault();
        lastFocusable.focus();
      }
      return;
    }

    if (!activeElement || activeElement === lastFocusable || !content.contains(activeElement)) {
      event.preventDefault();
      firstFocusable.focus();
    }
  }

  return (
    <div
      ref={contentRef}
      role="dialog"
      tabIndex={-1}
      aria-modal="true"
      aria-labelledby={titleId}
      aria-label={ariaLabel}
      onKeyDown={handleKeyDown}
      className={cn("absolute top-0 h-full w-[min(24rem,92vw)] border bg-card p-4 shadow-xl", sideClass, className)}
    >
      {children}
    </div>
  );
}
