import type { ComponentChildren } from "preact";

import { cn } from "@/lib/utils";

export interface ScrollAreaProps {
  className?: string;
  children?: ComponentChildren;
}

export function ScrollArea({ className, children }: ScrollAreaProps) {
  return <div className={cn("min-h-0 overflow-auto", className)}>{children}</div>;
}
