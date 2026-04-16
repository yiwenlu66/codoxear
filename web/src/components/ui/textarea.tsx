import type { JSX, Ref } from "preact";

import { cn } from "@/lib/utils";

export type TextareaProps = JSX.TextareaHTMLAttributes<HTMLTextAreaElement> & {
  textareaRef?: Ref<HTMLTextAreaElement>;
};

export function Textarea({ className, textareaRef, ...props }: TextareaProps) {
  return (
    <textarea
      ref={textareaRef}
      className={cn(
        "flex min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}
