import type { JSX } from "preact";

import { cn } from "@/lib/utils";

type DivProps = JSX.HTMLAttributes<HTMLDivElement>;
type HeadingProps = JSX.HTMLAttributes<HTMLHeadingElement>;

export function Card({ className, ...props }: DivProps) {
  return <div className={cn("rounded-xl border bg-card text-card-foreground shadow", className)} {...props} />;
}

export function CardHeader({ className, ...props }: DivProps) {
  return <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />;
}

export function CardTitle({ className, ...props }: HeadingProps) {
  return <h3 className={cn("font-semibold leading-none tracking-tight", className)} {...props} />;
}

export function CardContent({ className, ...props }: DivProps) {
  return <div className={cn("p-6 pt-0", className)} {...props} />;
}
