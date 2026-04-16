import { createContext, type ComponentChildren } from "preact";
import { useContext, useLayoutEffect, useRef } from "preact/hooks";

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

const DialogContext = createContext<{ onOpenChange?: (open: boolean) => void }>({});

interface DialogBaseProps {
  className?: string;
  children?: ComponentChildren;
}

export interface DialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children?: ComponentChildren;
}

export interface DialogContentProps extends DialogBaseProps {
  titleId?: string;
  ariaLabel?: string;
}

export interface DialogTitleProps extends DialogBaseProps {
  id?: string;
}

export function Dialog({ open, onOpenChange, children }: DialogProps) {
  if (!open) {
    return null;
  }

  return (
    <DialogContext.Provider value={{ onOpenChange }}>
      <div className="fixed inset-0 z-50 grid place-items-center p-4">
        <button
          type="button"
          data-testid="dialog-backdrop"
          data-dialog-owned="true"
          aria-label="Close dialog"
          onClick={() => onOpenChange?.(false)}
          className="absolute inset-0 bg-slate-950/45"
        />
        <div className="relative z-10 flex w-full justify-center">{children}</div>
      </div>
    </DialogContext.Provider>
  );
}

export function DialogContent({ className, children, titleId, ariaLabel }: DialogContentProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const { onOpenChange } = useContext(DialogContext);

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
      const activeElementIsDialogOwned = activeElement?.closest('[data-dialog-owned="true"]') != null;

      if (
        previousFocus &&
        previousFocus.isConnected &&
        (!activeElement || activeElement === document.body || content.contains(activeElement) || activeElementIsDialogOwned)
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
    if (event.key === "Escape") {
      event.preventDefault();
      onOpenChange?.(false);
      return;
    }

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
      className={cn("w-full max-w-3xl rounded-3xl border bg-card text-card-foreground shadow-2xl", className)}
    >
      {children}
    </div>
  );
}

export function DialogHeader({ className, children }: DialogBaseProps) {
  return <div className={cn("flex flex-col gap-1 p-6 pb-0", className)}>{children}</div>;
}

export function DialogTitle({ id, className, children }: DialogTitleProps) {
  return (
    <h2 id={id} className={cn("text-lg font-semibold", className)}>
      {children}
    </h2>
  );
}
