import type { JSX } from "preact";

import { cn } from "@/lib/utils";

export interface SeparatorProps extends JSX.HTMLAttributes<HTMLDivElement> {
  decorative?: boolean;
  orientation?: "horizontal" | "vertical";
}

export function Separator({ className, decorative = true, orientation = "horizontal", ...props }: SeparatorProps) {
  return (
    <div
      aria-hidden={decorative ? "true" : undefined}
      aria-orientation={decorative ? undefined : orientation}
      className={cn("shrink-0 bg-border", orientation === "horizontal" ? "h-px w-full" : "h-full w-px", className)}
      data-orientation={orientation}
      role={decorative ? undefined : "separator"}
      {...props}
    />
  );
}
