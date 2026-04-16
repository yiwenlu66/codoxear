import type { JSX } from "preact";

import { cn } from "@/lib/utils";

export function Skeleton({ className, ...props }: JSX.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("animate-pulse rounded-md bg-muted", className)} {...props} />;
}
