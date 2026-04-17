import { Fragment, h, type ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

import { AskUserCard, askUserHistorySignature, isUnresolvedAskUserEvent } from "./AskUserCard";
import { useComposerStore, useComposerStoreApi, useLiveSessionStore, useLiveSessionStoreApi, useMessagesStore, useMessagesStoreApi, useSessionsStore } from "../../app/providers";
import type { MessageEvent } from "../../lib/types";

const MAIN_TIMELINE_KINDS = new Set([
  "user",
  "assistant",
  "ask_user",
  "reasoning",
  "tool",
  "tool_result",
  "subagent",
  "todo_snapshot",
  "custom_message",
  "pi_session",
  "pi_model_change",
  "pi_thinking_level_change",
  "pi_event",
  "event",
]);

const CHAT_GROUPABLE_KINDS = new Set(["user", "assistant", "ask_user"]);
const COLLAPSIBLE_LINE_THRESHOLD = 8;
const COLLAPSIBLE_CHAR_THRESHOLD = 420;

const EVENT_LABELS: Record<string, string> = {
  ask_user: "Question",
  reasoning: "Reasoning",
  tool: "Tool",
  tool_result: "Tool Result",
  subagent: "Subagent",
  todo_snapshot: "Todo Progress",
  custom_message: "Custom Message",
  pi_session: "Session",
  pi_model_change: "Model Change",
  pi_thinking_level_change: "Thinking Level",
  pi_event: "System Event",
  event: "Event",
};

interface MarkdownRenderOptions {
  sessionId?: string;
  cwd?: string;
  onOpenLocalFile?: (path: string, line?: number | null) => void;
}

function baseName(value: string): string {
  const normalized = value.replace(/[\\/]+$/, "");
  const parts = normalized.split(/[\\/]+/);
  return parts[parts.length - 1] || normalized;
}

function normalizePathSeparators(value: string): string {
  return value.replace(/\\/g, "/");
}

function isProbablyUrl(value: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(value);
}

function isAbsolutePath(value: string): boolean {
  return value.startsWith("/") || /^[A-Za-z]:[\\/]/.test(value) || value.startsWith("~/");
}

function joinPaths(baseDir: string, target: string): string {
  const baseParts = normalizePathSeparators(baseDir).split("/").filter(Boolean);
  const targetParts = normalizePathSeparators(target).split("/");
  for (const part of targetParts) {
    if (!part || part === ".") continue;
    if (part === "..") {
      baseParts.pop();
      continue;
    }
    baseParts.push(part);
  }
  return `${baseDir.startsWith("/") ? "/" : ""}${baseParts.join("/")}` || "/";
}

function resolvePathTarget(rawTarget: string, cwd?: string): string {
  const target = rawTarget.trim();
  if (!target) return "";
  if (isProbablyUrl(target)) return target;
  if (isAbsolutePath(target)) return normalizePathSeparators(target);
  if (!cwd) return normalizePathSeparators(target);
  return joinPaths(cwd, target);
}

function parseLocalFileRef(rawValue: string, cwd?: string): { path: string; line?: number } | null {
  const trimmed = rawValue.trim();
  if (!trimmed || isProbablyUrl(trimmed) || trimmed.endsWith(":")) {
    return null;
  }

  let pathPart = trimmed;
  let line: number | undefined;

  const hashMatch = pathPart.match(/^(.*)#L(\d+)(?:-\d+)?$/i);
  if (hashMatch) {
    pathPart = hashMatch[1] || "";
    line = Number.parseInt(hashMatch[2] || "", 10);
  } else {
    const lineMatch = pathPart.match(/^(.*):(\d+)$/);
    if (lineMatch) {
      pathPart = lineMatch[1] || "";
      line = Number.parseInt(lineMatch[2] || "", 10);
    }
  }

  const resolvedPath = resolvePathTarget(pathPart, cwd);
  if (!resolvedPath || isProbablyUrl(resolvedPath)) {
    return null;
  }

  return Number.isFinite(line) ? { path: resolvedPath, line } : { path: resolvedPath };
}

function fileBlobHref(sessionId: string, path: string): string {
  return `api/sessions/${encodeURIComponent(sessionId)}/file/blob?path=${encodeURIComponent(path)}`;
}

function normalizeLineNumber(value: string | null): number | undefined {
  const line = Number.parseInt(String(value || "").trim(), 10);
  return Number.isFinite(line) && line > 0 ? line : undefined;
}

function rewriteOaiMemCitations(rawText: string): string {
  const raw = String(rawText ?? "");
  if (!raw.includes("<oai-mem-citation>")) {
    return raw;
  }

  const blockRegex = /<oai-mem-citation>\s*<citation_entries>\s*([\s\S]*?)\s*<\/citation_entries>\s*<rollout_ids>[\s\S]*?<\/rollout_ids>\s*<\/oai-mem-citation>/g;
  return raw.replace(blockRegex, (_whole, body) => {
    const lines = String(body || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (!lines.length) {
      return _whole;
    }

    const items = lines.map((line) => {
      const match = line.match(/^(.*?):(\d+)(?:-(\d+))?\|note=\[(.*)\]$/);
      if (!match) {
        return null;
      }
      const relPath = String(match[1] || "").trim().replace(/^\.?\//, "");
      const startLine = normalizeLineNumber(match[2]);
      const endLine = normalizeLineNumber(match[3]);
      const note = String(match[4] || "").trim();
      if (!relPath || !startLine || !note) {
        return null;
      }
      const lineSuffix = endLine && endLine >= startLine ? `#L${startLine}-${endLine}` : `#L${startLine}`;
      return `${note}|~/.codex/memories/${relPath}${lineSuffix}`;
    });

    if (items.some((item) => !item)) {
      return _whole;
    }

    const list = items
      .map((item, index) => {
        const [note, target] = String(item).split("|");
        return `${index + 1}. [${note}](${target})`;
      })
      .join("\n");

    return `\n---\n\nMemory citations:\n${list}`;
  });
}

type MarkdownNode = {
  type: string;
  alt?: string | null;
  checked?: boolean | null;
  children?: MarkdownNode[];
  depth?: number;
  identifier?: string;
  lang?: string | null;
  ordered?: boolean;
  start?: number | null;
  title?: string | null;
  url?: string;
  value?: string;
  align?: Array<"left" | "right" | "center" | null>;
};

type MarkdownDefinition = {
  title?: string | null;
  url: string;
};

const markdownProcessor = unified().use(remarkParse).use(remarkGfm).use(remarkBreaks);

function textFromChildren(children: ComponentChildren): string {
  if (children == null || typeof children === "boolean") {
    return "";
  }
  if (typeof children === "string" || typeof children === "number") {
    return String(children);
  }
  if (Array.isArray(children)) {
    return children.map((child) => textFromChildren(child)).join("");
  }
  if (typeof children === "object" && "props" in children) {
    return textFromChildren((children as { props?: { children?: ComponentChildren } }).props?.children ?? null);
  }
  return "";
}

function definitionId(value: string | undefined): string {
  return String(value || "").trim().toLowerCase();
}

function collectMarkdownDefinitions(root: MarkdownNode): Map<string, MarkdownDefinition> {
  const definitions = new Map<string, MarkdownDefinition>();
  for (const child of root.children || []) {
    if (child.type !== "definition") {
      continue;
    }
    const key = definitionId(child.identifier);
    if (!key || !child.url) {
      continue;
    }
    definitions.set(key, { url: child.url, title: child.title });
  }
  return definitions;
}

function renderMarkdownLink(target: string, children: ComponentChildren, options: MarkdownRenderOptions, title?: string | null) {
  const fileRef = parseLocalFileRef(target, options.cwd);
  if (fileRef && options.sessionId) {
    const displayLabel = textFromChildren(children).trim() || baseName(fileRef.path);
    const text = fileRef.line && displayLabel === baseName(fileRef.path) ? `${displayLabel}#L${fileRef.line}` : displayLabel;
    return (
      <a
        className="messageFileLink underline decoration-dotted underline-offset-4"
        data-file-path={fileRef.path}
        data-file-line={fileRef.line ? String(fileRef.line) : undefined}
        href={fileBlobHref(options.sessionId, fileRef.path)}
        rel="noreferrer"
        target="_blank"
        title={title || undefined}
      >
        {text}
      </a>
    );
  }

  const resolvedHref = resolvePathTarget(target, options.cwd);
  return (
    <a
      className="messageInlineLink underline decoration-dotted underline-offset-4"
      href={resolvedHref}
      rel="noreferrer"
      target="_blank"
      title={title || undefined}
    >
      {children}
    </a>
  );
}

function renderMarkdownImage(target: string, altText: string, options: MarkdownRenderOptions, title?: string | null) {
  const resolvedPath = resolvePathTarget(target, options.cwd);
  const src = options.sessionId && !isProbablyUrl(resolvedPath) ? fileBlobHref(options.sessionId, resolvedPath) : resolvedPath;
  return (
    <img
      alt={altText}
      className="messageImage max-h-80 rounded-2xl border border-border/60 bg-background/70 object-contain"
      loading="lazy"
      src={src}
      title={title || undefined}
    />
  );
}

function renderMarkdownChildren(children: MarkdownNode[] | undefined, options: MarkdownRenderOptions, definitions: Map<string, MarkdownDefinition>, keyPrefix: string): ComponentChildren {
  return (children || []).map((child, index) => (
    <Fragment key={`${keyPrefix}-${index}`}>{renderMarkdownNode(child, options, definitions, `${keyPrefix}-${index}`)}</Fragment>
  ));
}

function renderMarkdownTable(node: MarkdownNode, options: MarkdownRenderOptions, definitions: Map<string, MarkdownDefinition>, keyPrefix: string) {
  const rows = node.children || [];
  const headerRow = rows[0];
  const bodyRows = rows.slice(1);
  const alignments = Array.isArray(node.align) ? node.align : [];

  return (
    <div className="mdTableWrap overflow-x-auto rounded-2xl border border-border/60 bg-background/70">
      <table>
        {headerRow ? (
          <thead>
            <tr>
              {(headerRow.children || []).map((cell, index) => (
                <th key={`${keyPrefix}-head-${index}`} style={alignments[index] ? { textAlign: alignments[index] } : undefined}>
                  {renderMarkdownChildren(cell.children, options, definitions, `${keyPrefix}-head-${index}`)}
                </th>
              ))}
            </tr>
          </thead>
        ) : null}
        {bodyRows.length ? (
          <tbody>
            {bodyRows.map((row, rowIndex) => (
              <tr key={`${keyPrefix}-row-${rowIndex}`}>
                {(row.children || []).map((cell, cellIndex) => (
                  <td key={`${keyPrefix}-row-${rowIndex}-cell-${cellIndex}`} style={alignments[cellIndex] ? { textAlign: alignments[cellIndex] } : undefined}>
                    {renderMarkdownChildren(cell.children, options, definitions, `${keyPrefix}-row-${rowIndex}-cell-${cellIndex}`)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        ) : null}
      </table>
    </div>
  );
}

function renderMarkdownNode(node: MarkdownNode, options: MarkdownRenderOptions, definitions: Map<string, MarkdownDefinition>, keyPrefix: string): ComponentChildren {
  switch (node.type) {
    case "root":
      return renderMarkdownChildren(node.children, options, definitions, keyPrefix);
    case "definition":
      return null;
    case "paragraph":
      return <p>{renderMarkdownChildren(node.children, options, definitions, keyPrefix)}</p>;
    case "text":
    case "html":
      return node.value || "";
    case "strong":
      return <strong>{renderMarkdownChildren(node.children, options, definitions, keyPrefix)}</strong>;
    case "emphasis":
      return <em>{renderMarkdownChildren(node.children, options, definitions, keyPrefix)}</em>;
    case "delete":
      return <del>{renderMarkdownChildren(node.children, options, definitions, keyPrefix)}</del>;
    case "break":
      return <br />;
    case "inlineCode":
      return <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.92em]">{(node.value || "").replace(/\n$/, "")}</code>;
    case "code": {
      const className = node.lang ? `language-${node.lang}` : undefined;
      return (
        <pre className="overflow-x-auto rounded-2xl border border-border/60 bg-background/70 p-4">
          <code className={cn("font-mono text-sm", className)}>{(node.value || "").replace(/\n$/, "")}</code>
        </pre>
      );
    }
    case "heading": {
      const depth = Math.min(Math.max(node.depth || 1, 1), 6);
      return h(`h${depth}`, null, renderMarkdownChildren(node.children, options, definitions, keyPrefix));
    }
    case "blockquote":
      return <blockquote className="border-l-2 border-border/70 pl-4 text-muted-foreground">{renderMarkdownChildren(node.children, options, definitions, keyPrefix)}</blockquote>;
    case "list": {
      const ListTag = node.ordered ? "ol" : "ul";
      return <ListTag start={node.ordered && node.start && node.start !== 1 ? node.start : undefined}>{renderMarkdownChildren(node.children, options, definitions, keyPrefix)}</ListTag>;
    }
    case "listItem": {
      const checked = typeof node.checked === "boolean" ? node.checked : null;
      return (
        <li>
          {checked === null ? null : <input checked={checked} disabled readOnly type="checkbox" />}
          {renderMarkdownChildren(node.children, options, definitions, `${keyPrefix}-item`)}
        </li>
      );
    }
    case "thematicBreak":
      return <hr />;
    case "link":
      return renderMarkdownLink(node.url || "", renderMarkdownChildren(node.children, options, definitions, keyPrefix), options, node.title);
    case "image":
      return renderMarkdownImage(node.url || "", node.alt || "", options, node.title);
    case "linkReference": {
      const definition = definitions.get(definitionId(node.identifier));
      if (!definition) {
        return renderMarkdownChildren(node.children, options, definitions, keyPrefix);
      }
      return renderMarkdownLink(definition.url, renderMarkdownChildren(node.children, options, definitions, keyPrefix), options, definition.title);
    }
    case "imageReference": {
      const definition = definitions.get(definitionId(node.identifier));
      if (!definition) {
        return node.alt || "";
      }
      return renderMarkdownImage(definition.url, node.alt || "", options, definition.title);
    }
    case "table":
      return renderMarkdownTable(node, options, definitions, keyPrefix);
    default:
      if (node.children?.length) {
        return renderMarkdownChildren(node.children, options, definitions, keyPrefix);
      }
      return node.value || "";
  }
}

function MarkdownContent({ value, options = {} }: { value: string; options?: MarkdownRenderOptions }) {
  const normalized = rewriteOaiMemCitations(value).replace(/\r\n?/g, "\n");
  const root = markdownProcessor.runSync(markdownProcessor.parse(normalized)) as MarkdownNode;
  const definitions = collectMarkdownDefinitions(root);
  return <>{renderMarkdownNode(root, options, definitions, "md")}</>;
}

function messageContentParts(event: MessageEvent): string[] {
  const message = event.message;
  if (!message || !Array.isArray(message.content)) {
    return [];
  }
  return message.content
    .map((item) => (typeof item?.text === "string" ? item.text.trim() : ""))
    .filter(Boolean);
}

function firstNonEmptyText(...values: Array<unknown>): string {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}


function detailsSummary(details: Record<string, unknown> | undefined): string {
  if (!details) {
    return "";
  }
  if (typeof details.summary === "string" && details.summary.trim()) {
    return details.summary.trim();
  }
  if (typeof details.error === "string" && details.error.trim()) {
    return details.error.trim();
  }
  if (Array.isArray(details.todos) && details.todos.length) {
    return `${details.todos.length} todo item${details.todos.length === 1 ? "" : "s"}`;
  }
  const keys = Object.keys(details);
  if (keys.length) {
    return `Details: ${keys.join(", ")}`;
  }
  return "";
}

function contentTextFromMessage(event: MessageEvent): string {
  const kind = eventKind(event);
  if (kind === "ask_user") {
    return firstNonEmptyText(event.question, event.text, "Prompt");
  }
  if (typeof event.text === "string" && event.text.trim()) {
    return event.text;
  }
  const contentParts = messageContentParts(event);
  if (contentParts.length) {
    return contentParts.join("\n");
  }
  if (typeof event.output === "string" && event.output.trim()) {
    return event.output;
  }
  if (typeof event.summary === "string" && event.summary.trim()) {
    return event.summary;
  }
  if (typeof event.question === "string" && event.question.trim()) {
    return event.question;
  }
  if (typeof event.context === "string" && event.context.trim()) {
    return event.context;
  }
  if (event.details) {
    return detailsSummary(event.details) || JSON.stringify(event.details, null, 2);
  }
  return JSON.stringify(event, null, 2);
}

function eventKind(event: MessageEvent): string {
  if (typeof event.role === "string" && event.role) {
    return event.role;
  }
  if (typeof event.message?.role === "string" && event.message.role) {
    return event.message.role;
  }
  if (event.toolName === "ask_user") {
    return "ask_user";
  }
  return typeof event.type === "string" && event.type ? event.type : "event";
}

function shouldRenderInMainConversation(event: MessageEvent): boolean {
  const kind = eventKind(event);
  if (MAIN_TIMELINE_KINDS.has(kind)) {
    return true;
  }
  return Boolean(firstNonEmptyText(event.text, event.summary, event.question, event.context));
}

function canGroupEvent(kind: string): boolean {
  return CHAT_GROUPABLE_KINDS.has(kind);
}

function eventLabel(kind: string): string {
  return EVENT_LABELS[kind] || kind.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function surfaceBadgeVariant(kind: string): "default" | "secondary" | "outline" {
  switch (kind) {
    case "user":
      return "default";
    case "assistant":
    case "tool_result":
    case "todo_snapshot":
      return "secondary";
    default:
      return "outline";
  }
}

function messageSurfaceTone(kind: string, isError = false): string {
  if (isError) {
    return "border-destructive/40 bg-destructive/5";
  }

  switch (kind) {
    case "user":
      return "border-primary/30 bg-primary/10 text-foreground";
    case "assistant":
      return "border-border/70 bg-card/95";
    case "ask_user":
      return "border-amber-300/70 bg-amber-50/90";
    case "reasoning":
      return "border-sky-200/80 bg-sky-50/80";
    case "tool":
      return "border-indigo-200/80 bg-indigo-50/80";
    case "tool_result":
      return "border-emerald-200/80 bg-emerald-50/80";
    case "subagent":
      return "border-slate-200/80 bg-slate-50/85";
    case "todo_snapshot":
      return "border-teal-200/80 bg-teal-50/85";
    default:
      return "border-border/60 bg-muted/30";
  }
}

function isDisplayableEpochTs(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 1_000_000_000;
}

function formatMessageTimestamp(ts: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(ts * 1000));
}

function messageDayKey(ts: number): string {
  const date = new Date(ts * 1000);
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

function formatDaySeparator(ts: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(ts * 1000));
}

function handleRichTextClick(event: MouseEvent, options: MarkdownRenderOptions) {
  if (!options.onOpenLocalFile) {
    return;
  }

  const target = event.target instanceof Element ? event.target : null;
  const link = target?.closest("a[data-file-path]") as HTMLAnchorElement | null;
  if (!link) {
    return;
  }

  const path = String(link.getAttribute("data-file-path") || "").trim();
  if (!path) {
    return;
  }

  event.preventDefault();
  options.onOpenLocalFile(path, normalizeLineNumber(link.getAttribute("data-file-line")));
}

function renderRichText(value: string, className = "messageBody", options: MarkdownRenderOptions = {}) {
  if (!value.trim()) {
    return null;
  }
  return (
    <div className={className} onClick={(event) => handleRichTextClick(event as MouseEvent, options)}>
      <MarkdownContent value={value} options={options} />
    </div>
  );
}

function shouldCollapseContent(value: string): boolean {
  const normalized = value.trim();
  if (!normalized) {
    return false;
  }
  const lineCount = normalized.split("\n").length;
  return lineCount > COLLAPSIBLE_LINE_THRESHOLD || normalized.length > COLLAPSIBLE_CHAR_THRESHOLD;
}

function compactSingleLine(value: string, maxLength = 140): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 1).trimEnd()}...`;
}

function toolSummaryText(event: MessageEvent): string {
  return compactSingleLine(
    firstNonEmptyText(event.summary, event.text, event.context, detailsSummary(event.details)),
  );
}

function ExpandableRichText({
  value,
  className = "messageBody",
  options = {},
}: {
  value: string;
  className?: string;
  options?: MarkdownRenderOptions;
}) {
  const collapsible = shouldCollapseContent(value);
  const [expanded, setExpanded] = useState(false);
  const previousValueRef = useRef(value);
  const contentClassName = cn("messageExpandableContent", collapsible && !expanded && "isCollapsed");

  useEffect(() => {
    if (previousValueRef.current !== value) {
      previousValueRef.current = value;
      setExpanded(false);
    }
  }, [value]);

  return (
    <div className="messageExpandable space-y-3">
      <div className={contentClassName}>{renderRichText(value, className, options)}</div>
      {collapsible ? (
        <button
          type="button"
          className="messageExpandButton inline-flex items-center rounded-full border border-border/70 px-3 py-1 text-xs font-medium text-muted-foreground transition hover:bg-accent hover:text-accent-foreground"
          aria-expanded={expanded ? "true" : "false"}
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </div>
  );
}

function renderCardHeader(kind: string, title?: string, summary?: string, ts?: number) {
  const showTimestamp = isDisplayableEpochTs(ts);
  return (
    <header className="messageCardHeader flex flex-col gap-2">
      <div className="messageCardHeaderRow flex flex-wrap items-center gap-2">
        <Badge variant={surfaceBadgeVariant(kind)}>{eventLabel(kind)}</Badge>
        {title ? <div className="messageCardTitle text-sm font-semibold text-foreground">{title}</div> : null}
        {showTimestamp ? (
          <time className="messageTimestamp ml-auto text-xs text-muted-foreground" dateTime={new Date(ts * 1000).toISOString()}>
            {formatMessageTimestamp(ts)}
          </time>
        ) : null}
      </div>
      {summary ? <div className="messageCardSummary text-sm text-muted-foreground">{summary}</div> : null}
    </header>
  );
}

function CopyMessageIcon() {
  return (
    <svg className="messageCopyIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M7 15H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function ChatMessageCard({
  event,
  kind,
  options,
}: {
  event: MessageEvent;
  kind: "user" | "assistant";
  options: MarkdownRenderOptions;
}) {
  const label = kind === "user" ? "You" : "Assistant";
  const text = contentTextFromMessage(event);
  const [copied, setCopied] = useState(false);
  const resetTimerRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (resetTimerRef.current !== null) {
      window.clearTimeout(resetTimerRef.current);
    }
  }, []);

  const handleCopy = async () => {
    if (!text.trim() || !navigator.clipboard || typeof navigator.clipboard.writeText !== "function") {
      return;
    }
    await navigator.clipboard.writeText(text);
    setCopied(true);
    if (resetTimerRef.current !== null) {
      window.clearTimeout(resetTimerRef.current);
    }
    resetTimerRef.current = window.setTimeout(() => {
      setCopied(false);
      resetTimerRef.current = null;
    }, 1200);
  };

  return (
    <MessageSurface kind={kind}>
      {renderCardHeader(kind, label, undefined, event.ts)}
      {renderRichText(text, "messageBody prose prose-sm max-w-none", options)}
      <div className="messageBubbleActions">
        <button
          type="button"
          className={cn("messageCopyButton", copied && "isCopied")}
          aria-label={`Copy ${kind} message`}
          onClick={() => {
            void handleCopy();
          }}
        >
          <CopyMessageIcon />
        </button>
      </div>
    </MessageSurface>
  );
}

function MessageSurface({
  kind,
  children,
  grouped = false,
  isError = false,
  compact = false,
  className,
  contentClassName,
}: {
  kind: string;
  children: ComponentChildren;
  grouped?: boolean;
  isError?: boolean;
  compact?: boolean;
  className?: string;
  contentClassName?: string;
}) {
  const isChatSurface = kind === "user" || kind === "assistant" || kind === "ask_user";

  return (
    <Card
      data-testid="message-surface"
      data-kind={kind}
      className={cn(
        "messageSurface rounded-[1.35rem] border shadow-sm backdrop-blur-sm transition-colors",
        isChatSurface ? "max-w-3xl" : compact ? "max-w-[56rem]" : "max-w-4xl",
        kind === "user" ? "ml-auto messageBubble user" : undefined,
        kind === "assistant" ? "mr-auto messageBubble assistant" : undefined,
        kind === "ask_user" ? "mr-auto messageBubble messageCard ask_user" : undefined,
        !isChatSurface && !compact ? "messageCard" : undefined,
        compact ? "border-0 bg-transparent shadow-none backdrop-blur-none" : undefined,
        grouped && "grouped",
        isError && "isError",
        !compact ? messageSurfaceTone(kind, isError) : undefined,
        className,
      )}
    >
      <CardContent className={cn(compact ? "p-0" : "space-y-3 p-4", contentClassName)}>{children}</CardContent>
    </Card>
  );
}

function renderChatCard(event: MessageEvent, kind: "user" | "assistant", options: MarkdownRenderOptions) {
  return <ChatMessageCard event={event} kind={kind} options={options} />;
}

function shouldAllowFuzzyAskUserMatch(messages: MessageEvent[], index: number) {
  const event = messages[index];
  if (!isUnresolvedAskUserEvent(event)) return true;
  const signature = askUserHistorySignature(event);
  if (!signature) return true;
  for (let cursor = index + 1; cursor < messages.length; cursor += 1) {
    const candidate = messages[cursor];
    if (!isUnresolvedAskUserEvent(candidate)) continue;
    if (askUserHistorySignature(candidate) === signature) {
      return false;
    }
  }
  return true;
}

function renderAskUserCard(
  event: MessageEvent,
  sessionId: string | undefined,
  options: MarkdownRenderOptions,
  allowFuzzyLiveMatch: boolean,
  allowLegacyFallback: boolean,
) {
  return (
    <AskUserCard
      event={event}
      sessionId={sessionId}
      allowFuzzyLiveMatch={allowFuzzyLiveMatch}
      allowLegacyFallback={allowLegacyFallback}
      renderRichText={(value, className) => renderRichText(value, className, options)}
    />
  );
}

function renderReasoningCard(event: MessageEvent, options: MarkdownRenderOptions) {
  const summary = firstNonEmptyText(event.summary);
  const body = firstNonEmptyText(event.text, summary);

  return (
    <MessageSurface kind="reasoning">
      {renderCardHeader("reasoning", undefined, summary && summary !== body ? summary : undefined, event.ts)}
      {body ? <ExpandableRichText key={body} value={body} options={options} /> : null}
    </MessageSurface>
  );
}

function CompactToolSurface({
  kind,
  title,
  summary,
  ts,
  isError = false,
  children,
}: {
  kind: "tool" | "tool_result";
  title: string;
  summary: string;
  ts?: number;
  isError?: boolean;
  children?: ComponentChildren;
}) {
  const [expanded, setExpanded] = useState(false);
  const showTimestamp = isDisplayableEpochTs(ts);

  return (
    <MessageSurface kind={kind} isError={isError} compact>
      <div className="messageToolBlock space-y-2">
        <div
          className={cn(
            "messageToolRow flex items-center gap-3 rounded-xl border px-3 py-2",
            kind === "tool"
              ? "border-sky-200/80 bg-sky-50/80"
              : isError
                ? "border-red-200/80 bg-red-50/80"
                : "border-emerald-200/80 bg-emerald-50/80",
          )}
        >
          <Badge variant={surfaceBadgeVariant(kind)} className="shrink-0">{eventLabel(kind)}</Badge>
          <div className="min-w-0 flex-1 flex items-center gap-2 overflow-hidden">
            <span className="shrink-0 text-sm font-semibold text-foreground">{title}</span>
            {summary ? <span className="messageToolSummary min-w-0 flex-1 truncate text-sm text-muted-foreground">{summary}</span> : null}
          </div>
          {showTimestamp ? (
            <time className="shrink-0 text-xs text-muted-foreground" dateTime={new Date(ts * 1000).toISOString()}>
              {formatMessageTimestamp(ts)}
            </time>
          ) : null}
          <button
            type="button"
            className="messageToolToggle inline-flex items-center rounded-full border border-border/70 px-3 py-1 text-xs font-medium text-muted-foreground transition hover:bg-accent hover:text-accent-foreground"
            aria-expanded={expanded ? "true" : "false"}
            onClick={() => setExpanded((current) => !current)}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
        </div>
        {expanded ? <div className="messageToolDetails rounded-xl border border-border/60 bg-background/75 p-3">{children}</div> : null}
      </div>
    </MessageSurface>
  );
}

function renderToolCard(event: MessageEvent, options: MarkdownRenderOptions) {
  const body = firstNonEmptyText(event.text, event.summary, event.context);
  const summary = toolSummaryText(event);

  return (
    <CompactToolSurface kind="tool" title={firstNonEmptyText(event.name, "Unnamed tool")} summary={summary} ts={event.ts}>
      {body ? renderRichText(body, "messageBody", options) : <div className="messageCardFooterText text-sm text-muted-foreground">No additional tool input.</div>}
    </CompactToolSurface>
  );
}

function renderToolResultCard(event: MessageEvent, options: MarkdownRenderOptions) {
  const body = firstNonEmptyText(event.text, detailsSummary(event.details));
  const detailsText = !event.text && event.details ? JSON.stringify(event.details, null, 2) : "";
  const summary = toolSummaryText(event) || compactSingleLine(detailsText);

  return (
    <CompactToolSurface kind="tool_result" title={firstNonEmptyText(event.name, "Tool result")} summary={summary} ts={event.ts} isError={Boolean(event.is_error)}>
      {body ? renderRichText(body, "messageBody", options) : null}
      {detailsText ? <pre className="messageCardPre overflow-x-auto rounded-xl bg-background/80 p-3 text-sm">{detailsText}</pre> : null}
    </CompactToolSurface>
  );
}

function renderSubagentCard(event: MessageEvent, options: MarkdownRenderOptions) {
  const output = firstNonEmptyText(event.output, event.text);
  const pending = !output;

  return (
    <MessageSurface kind="subagent">
      {renderCardHeader("subagent", firstNonEmptyText(event.agent, "subagent"), firstNonEmptyText(event.task), event.ts)}
      <div className="messageMetaList flex flex-col gap-2">
        <div className="grid grid-cols-2 gap-2">
          {event.agent ? (
            <div className="messageMetaItem rounded-xl bg-background/70 p-3 text-sm">
              <span className="block text-xs uppercase tracking-wide text-muted-foreground">Agent</span>
              <strong>{event.agent}</strong>
            </div>
          ) : null}
          <div className={cn("messageMetaItem rounded-xl bg-background/70 p-3 text-sm", !event.agent && "col-span-2")}>
            <span className="block text-xs uppercase tracking-wide text-muted-foreground">Status</span>
            <strong>{pending ? "Running" : "Completed"}</strong>
          </div>
        </div>
        {event.task ? (
          <div className="messageMetaItem rounded-xl bg-background/70 p-3 text-sm">
            <span className="block text-xs uppercase tracking-wide text-muted-foreground">Task</span>
            <strong>{event.task}</strong>
          </div>
        ) : null}
      </div>
      {output ? renderRichText(output, "messageBody", options) : <div className="messageCardFooterText text-sm text-muted-foreground">Waiting for subagent output...</div>}
    </MessageSurface>
  );
}

function renderTodoSnapshotCard(event: MessageEvent, options: MarkdownRenderOptions) {
  const items = Array.isArray(event.items) ? event.items.slice(0, 3) : [];

  return (
    <MessageSurface kind="todo_snapshot">
      {renderCardHeader("todo_snapshot", firstNonEmptyText(event.progress_text, "Todo snapshot"), firstNonEmptyText(event.operation), event.ts)}
      {items.length ? (
        <ul className="messageTodoList space-y-2">
          {items.map((item, index) => (
            <li key={`${item.title || "todo"}-${index}`} className="messageTodoItem flex items-start gap-3 rounded-xl bg-background/70 px-3 py-2 text-sm">
              <span className={cn("messageTodoStatus rounded-full px-2 py-0.5 text-xs font-semibold uppercase tracking-wide", typeof item.status === "string" ? item.status : "unknown")}>{item.status || "unknown"}</span>
              <span>{item.title || item.description || "Untitled item"}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {event.text ? renderRichText(event.text, "messageBody", options) : null}
    </MessageSurface>
  );
}

function renderCustomMessageCard(event: MessageEvent, options: MarkdownRenderOptions) {
  const customType = typeof event.custom_type === "string" ? event.custom_type : "";

  if (customType === "claude-todo-v2-task-assignment") {
    return (
      <MessageSurface kind="custom_message">
        {renderCardHeader("custom_message", firstNonEmptyText(event.text, "Task assignment"), "Claude Todo V2", event.ts)}
        <div className="space-y-3">
          {event.subject ? (
            <div className="messageMetaItem rounded-xl bg-background/70 p-3 text-sm">
              <span className="block text-xs uppercase tracking-wide text-muted-foreground">Subject</span>
              <strong>{event.subject}</strong>
            </div>
          ) : null}
          <div className="grid grid-cols-2 gap-2">
            {event.owner ? (
              <div className="messageMetaItem rounded-xl bg-background/70 p-3 text-sm">
                <span className="block text-xs uppercase tracking-wide text-muted-foreground">Owner</span>
                <strong>{event.owner}</strong>
              </div>
            ) : null}
            {event.assigned_by ? (
              <div className="messageMetaItem rounded-xl bg-background/70 p-3 text-sm">
                <span className="block text-xs uppercase tracking-wide text-muted-foreground">Assigned By</span>
                <strong>{event.assigned_by}</strong>
              </div>
            ) : null}
          </div>
          {event.description ? renderRichText(event.description, "messageBody", options) : null}
        </div>
      </MessageSurface>
    );
  }

  return (
    <MessageSurface kind="custom_message">
      {renderCardHeader("custom_message", firstNonEmptyText(event.text, event.summary, event.name), customType || undefined, event.ts)}
      {event.description ? renderRichText(event.description, "messageBody", options) : null}
    </MessageSurface>
  );
}

function renderSystemCard(event: MessageEvent, kind: string, options: MarkdownRenderOptions) {
  const title = firstNonEmptyText(event.summary, event.name);
  const body = firstNonEmptyText(event.text, event.context, event.question, title === event.summary ? "" : event.summary);

  return (
    <MessageSurface kind={kind}>
      {renderCardHeader(kind, title || undefined, undefined, event.ts)}
      {body ? renderRichText(body, "messageBody", options) : null}
    </MessageSurface>
  );
}

function renderConversationEvent(
  event: MessageEvent,
  kind: string,
  sessionId: string | undefined,
  options: MarkdownRenderOptions,
  allowFuzzyLiveMatch = true,
  allowLegacyFallback = false,
) {
  switch (kind) {
    case "user":
    case "assistant":
      return renderChatCard(event, kind, options);
    case "ask_user":
      return renderAskUserCard(event, sessionId, options, allowFuzzyLiveMatch, allowLegacyFallback);
    case "reasoning":
      return renderReasoningCard(event, options);
    case "tool":
      return renderToolCard(event, options);
    case "tool_result":
      return renderToolResultCard(event, options);
    case "subagent":
      return renderSubagentCard(event, options);
    case "todo_snapshot":
      return renderTodoSnapshotCard(event, options);
    case "custom_message":
      return renderCustomMessageCard(event, options);
    case "pi_session":
    case "pi_model_change":
    case "pi_thinking_level_change":
    case "pi_event":
    case "event":
      return renderSystemCard(event, kind, options);
    default:
      return renderSystemCard(event, kind, options);
  }
}

function renderLoadingCards() {
  return (
    <div className="messageList flex flex-col gap-3">
      {Array.from({ length: 3 }, (_value, index) => (
        <div key={index} className={cn("messageRow flex", index === 0 ? "assistant" : index === 1 ? "tool" : "assistant")}>
          <Card data-testid="message-surface" data-kind="loading" className="messageSurface max-w-4xl rounded-[1.35rem] border border-border/60 bg-card/90 shadow-sm">
            <CardContent className="space-y-3 p-4">
              <div className="flex items-center gap-2">
                <Skeleton className="h-5 w-20 rounded-full" />
                <Skeleton className="h-4 w-36" />
              </div>
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
              {index === 1 ? <Skeleton className="h-16 w-full rounded-xl" /> : null}
            </CardContent>
          </Card>
        </div>
      ))}
    </div>
  );
}

function WorkingIndicator() {
  return (
    <div className="messageRow assistant workingIndicator flex px-1 py-1">
      <div className="flex items-center gap-2 rounded-2xl border border-border/40 bg-muted/30 px-3 py-2 text-muted-foreground/70 shadow-sm">
        <div className="flex gap-1">
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.3s]" />
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.15s]" />
          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current" />
        </div>
        <span className="text-xs font-medium uppercase tracking-wider">Working</span>
      </div>
    </div>
  );
}

const messageRowIds = new WeakMap<object, string>();
let nextMessageRowId = 0;

function messageRowKey(event: MessageEvent, kind: string, index: number) {
  const row = event as Record<string, unknown>;
  const messageId = typeof row.message_id === "string" ? row.message_id.trim() : "";
  if (messageId) {
    return `${kind}:${messageId}`;
  }
  if (event && typeof event === "object") {
    let objectKey = messageRowIds.get(event as object);
    if (!objectKey) {
      nextMessageRowId += 1;
      objectKey = `row-${nextMessageRowId}`;
      messageRowIds.set(event as object, objectKey);
    }
    return `${kind}:${objectKey}`;
  }
  return `${kind}:fallback-${index}`;
}

function scrollPaneToBottom(element: HTMLElement) {
  if (typeof element.scrollTo === "function") {
    element.scrollTo({ top: element.scrollHeight });
    return;
  }
  element.scrollTop = element.scrollHeight;
}

const PREVIOUS_USER_BUTTON_SCROLL_THRESHOLD = 320;
const PREVIOUS_USER_BUTTON_VIEWPORT_RATIO = 0.5;
const PREVIOUS_USER_TARGET_TOLERANCE = 24;
const PREVIOUS_USER_SCROLL_TOP_PADDING = 16;

function ArrowUpTurnIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 6 8 3l3 3" />
      <path d="M8 3v7" />
      <path d="M8 10h3.5A2.5 2.5 0 0 1 14 12.5" />
    </svg>
  );
}

function ArrowDownIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 3v10" />
      <path d="m5 10 3 3 3-3" />
    </svg>
  );
}

function findPreviousUserRow(pane: HTMLElement): HTMLElement | null {
  const visibilityThreshold = pane.clientHeight > 0
    ? pane.clientHeight * PREVIOUS_USER_BUTTON_VIEWPORT_RATIO
    : PREVIOUS_USER_BUTTON_SCROLL_THRESHOLD;
  if (pane.scrollTop < visibilityThreshold) {
    return null;
  }

  const threshold = pane.scrollTop - PREVIOUS_USER_TARGET_TOLERANCE;

  const rows = Array.from(pane.querySelectorAll<HTMLElement>(".messageRow.user"));
  let candidate: HTMLElement | null = null;
  for (const row of rows) {
    if (row.offsetTop <= threshold) {
      candidate = row;
      continue;
    }
    break;
  }

  return candidate;
}

function shouldShowScrollToBottom(pane: HTMLElement): boolean {
  const rows = Array.from(pane.querySelectorAll<HTMLElement>(".messageRow"));
  const lastRow = rows[rows.length - 1] ?? null;
  const fallbackContentBottom = lastRow ? lastRow.offsetTop + lastRow.offsetHeight : 0;
  const contentBottom = Math.max(pane.scrollHeight, fallbackContentBottom);
  const visibleHeight = pane.clientHeight > 0 ? pane.clientHeight : 0;
  const distanceFromBottom = contentBottom - (pane.scrollTop + visibleHeight);
  const threshold = visibleHeight > 0 ? Math.max(160, Math.round(visibleHeight * 0.5)) : 180;
  return distanceFromBottom > threshold;
}

function scrollPaneToPosition(element: HTMLElement, top: number) {
  const nextTop = Math.max(0, top);
  if (typeof element.scrollTo === "function") {
    element.scrollTo({ top: nextTop, behavior: "smooth" });
    return;
  }
  element.scrollTop = nextTop;
}

interface ConversationPaneProps {
  onOpenFilePath?: (path: string, line?: number | null) => void;
}

export function ConversationPane({ onOpenFilePath }: ConversationPaneProps) {
  const { activeSessionId, items } = useSessionsStore();
  const { busyBySessionId } = useLiveSessionStore();
  const composerState = useComposerStore();
  const composerStoreApi = useComposerStoreApi();
  const pendingBySessionId = composerState.pendingBySessionId ?? {};
  const messagesState = useMessagesStore();
  const bySessionId = messagesState.bySessionId;
  const hasOlderBySessionId = messagesState.hasOlderBySessionId ?? {};
  const olderBeforeBySessionId = messagesState.olderBeforeBySessionId ?? {};
  const loadingOlderBySessionId = messagesState.loadingOlderBySessionId ?? {};
  const loadingBySessionId = (messagesState as { loadingBySessionId?: Record<string, boolean> }).loadingBySessionId ?? {};
  const loadedBySessionId = (messagesState as { loadedBySessionId?: Record<string, boolean> }).loadedBySessionId ?? {};
  const messagesStoreApi = useMessagesStoreApi();
  const liveSessionStoreApi = useLiveSessionStoreApi();
  const activeSession = items.find((session) => session.session_id === activeSessionId) ?? null;
  const activeSessionIsHistoricalPi = activeSession?.historical === true && activeSession?.agent_backend === "pi";
  const allowLegacyAskUserFallback = Boolean(activeSession?.agent_backend === "pi" && activeSession?.transport !== "pi-rpc");
  const isBusy = Boolean(
    (activeSessionId && busyBySessionId[activeSessionId] === true)
    || activeSession?.busy === true,
  );
  const persistedMessages = activeSessionId ? bySessionId[activeSessionId] ?? [] : [];
  const pendingMessages = activeSessionId ? pendingBySessionId[activeSessionId] ?? [] : [];
  const rawMessages = [...persistedMessages, ...pendingMessages];
  const messages = rawMessages.filter(shouldRenderInMainConversation);
  const sectionRef = useRef<HTMLElement | null>(null);
  const historyAnchorRef = useRef<{ key: string; top: number } | null>(null);
  const scrollModeRef = useRef<"bottom" | "preserve" | null>(null);
  const [showPreviousUserJump, setShowPreviousUserJump] = useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const hasOlder = activeSessionId ? hasOlderBySessionId[activeSessionId] === true : false;
  const olderCursor = activeSessionId ? olderBeforeBySessionId[activeSessionId] ?? 0 : 0;
  const olderLoading = activeSessionId ? loadingOlderBySessionId[activeSessionId] === true : false;
  const activeSessionLoading = activeSessionId ? loadingBySessionId[activeSessionId] === true : false;
  const activeSessionLoaded = activeSessionId ? loadedBySessionId[activeSessionId] === true : false;
  const showHistoryControls = Boolean(activeSessionId && messages.length && (hasOlder || olderCursor > 0 || olderLoading));
  const waitingForInitialHistoricalReplay = activeSessionIsHistoricalPi && messages.length === 0 && !activeSessionLoaded;
  const showLoadingState = (activeSessionLoading || waitingForInitialHistoricalReplay) && messages.length === 0 && !activeSessionLoaded;
  const markdownOptions: MarkdownRenderOptions = {
    sessionId: activeSessionId || undefined,
    cwd: activeSession?.cwd,
    onOpenLocalFile: onOpenFilePath,
  };

  useEffect(() => {
    if (!activeSessionId) return;
    if ((pendingBySessionId[activeSessionId] ?? []).length === 0) return;
    const clearAcknowledgedPending = (composerStoreApi as { clearAcknowledgedPending?: (sessionId: string, events: MessageEvent[]) => void }).clearAcknowledgedPending;
    if (typeof clearAcknowledgedPending !== "function") return;
    clearAcknowledgedPending(activeSessionId, persistedMessages);
  }, [activeSessionId, persistedMessages, pendingBySessionId, composerStoreApi]);

  const recomputeFloatingNavigation = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane || !activeSessionId) {
      setShowPreviousUserJump(false);
      setShowScrollToBottom(false);
      return;
    }
    setShowPreviousUserJump(Boolean(findPreviousUserRow(pane)));
    setShowScrollToBottom(shouldShowScrollToBottom(pane));
  };

  useLayoutEffect(() => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane || (!messages.length && !isBusy)) {
      setShowPreviousUserJump(false);
      setShowScrollToBottom(false);
      return;
    }

    if (scrollModeRef.current === "preserve") {
      const anchor = historyAnchorRef.current;
      if (anchor) {
        const anchorRow = pane.querySelector(`[data-row-key="${anchor.key}"]`) as HTMLElement | null;
        if (anchorRow) {
          pane.scrollTop = Math.max(0, anchorRow.offsetTop - anchor.top);
        }
      }
      historyAnchorRef.current = null;
      scrollModeRef.current = null;
      recomputeFloatingNavigation();
      return;
    }

    scrollPaneToBottom(pane);
    scrollModeRef.current = null;
    recomputeFloatingNavigation();
  }, [messages.length, activeSessionId, isBusy]);

  useEffect(() => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane || !activeSessionId) {
      return undefined;
    }

    const onScroll = () => {
      if (pane.scrollTop <= 12 && !olderLoading && (hasOlder || olderCursor > 0)) {
        void handleLoadOlder();
      }
      setShowPreviousUserJump(Boolean(findPreviousUserRow(pane)));
      setShowScrollToBottom(shouldShowScrollToBottom(pane));
    };

    pane.addEventListener("scroll", onScroll);
    return () => pane.removeEventListener("scroll", onScroll);
  }, [activeSessionId, hasOlder, olderCursor, olderLoading]);

  const handleLoadOlder = async () => {
    if (!activeSessionId) return;
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    const firstRow = pane?.querySelector("[data-row-key]") as HTMLElement | null;
    if (pane && firstRow) {
      historyAnchorRef.current = {
        key: String(firstRow.dataset.rowKey || ""),
        top: firstRow.offsetTop - pane.scrollTop,
      };
      scrollModeRef.current = "preserve";
    }
    await messagesStoreApi.loadOlder(activeSessionId);
  };

  useEffect(() => {
    if (!activeSessionId || !activeSessionIsHistoricalPi) {
      return;
    }
    if (activeSessionLoaded || activeSessionLoading) {
      return;
    }
    void messagesStoreApi.loadInitial(activeSessionId);
  }, [activeSessionId, activeSessionIsHistoricalPi, activeSessionLoaded, activeSessionLoading, messagesStoreApi]);

  const handleJumpToLatest = async () => {
    if (!activeSessionId) return;
    historyAnchorRef.current = null;
    scrollModeRef.current = "bottom";
    if (activeSessionIsHistoricalPi) {
      await messagesStoreApi.loadInitial(activeSessionId);
      return;
    }
    await liveSessionStoreApi.loadInitial(activeSessionId);
  };

  const handleJumpToPreviousUser = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane) return;
    const target = findPreviousUserRow(pane);
    if (!target) {
      setShowPreviousUserJump(false);
      return;
    }
    scrollPaneToPosition(pane, target.offsetTop - PREVIOUS_USER_SCROLL_TOP_PADDING);
  };

  const handleScrollToBottom = () => {
    const pane = sectionRef.current?.querySelector(".conversationPane") as HTMLElement | null;
    if (!pane) return;
    scrollPaneToPosition(pane, pane.scrollHeight);
  };

  return (
    <section ref={sectionRef} className="conversationTimeline relative flex min-h-0 flex-1">
      <ScrollArea className={cn("conversationPane conversationScrollArea min-h-0 flex-1 px-3 py-4", !activeSessionId && "emptyState")}>
        {showLoadingState ? (
          renderLoadingCards()
        ) : (
          <div className="messageList flex flex-col gap-3">
            {showHistoryControls ? (
              <div className="historyControls flex flex-wrap items-center justify-between gap-2 rounded-[1.1rem] border border-border/60 bg-background/70 px-3 py-2 text-sm text-muted-foreground">
                <span>{hasOlder || olderCursor > 0 ? "Older conversation history is available." : "You are viewing older history."}</span>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    className="inline-flex items-center rounded-full border border-border/70 px-3 py-1.5 font-medium text-foreground transition hover:bg-accent hover:text-accent-foreground disabled:cursor-not-allowed disabled:opacity-60"
                    onClick={() => void handleLoadOlder()}
                    disabled={olderLoading || !activeSessionId || (!hasOlder && olderCursor <= 0)}
                  >
                    {olderLoading ? "Loading..." : "Load older"}
                  </button>
                  <button
                    type="button"
                    className="inline-flex items-center rounded-full border border-border/70 px-3 py-1.5 font-medium text-foreground transition hover:bg-accent hover:text-accent-foreground disabled:cursor-not-allowed disabled:opacity-60"
                    onClick={() => void handleJumpToLatest()}
                    disabled={!activeSessionId}
                  >
                    Jump to latest
                  </button>
                </div>
              </div>
            ) : null}
            {messages.length ? (
              messages.map((message, index) => {
                const kind = eventKind(message);
                const prevKind = index > 0 ? eventKind(messages[index - 1]) : null;
                const grouped = prevKind === kind && canGroupEvent(kind);
                const rowKey = messageRowKey(message, kind, index);
                const ts = typeof message.ts === "number" ? message.ts : null;
                const prevTs = index > 0 && typeof messages[index - 1]?.ts === "number" ? messages[index - 1].ts as number : null;
                const showDaySeparator = isDisplayableEpochTs(ts) && (!isDisplayableEpochTs(prevTs) || messageDayKey(prevTs) !== messageDayKey(ts));
                return (
                  <Fragment key={rowKey}>
                    {showDaySeparator ? (
                      <div className="daySeparator flex items-center gap-3 py-1 text-xs uppercase tracking-[0.16em] text-muted-foreground">
                        <span className="h-px flex-1 bg-border/60" />
                        <span>{formatDaySeparator(ts)}</span>
                        <span className="h-px flex-1 bg-border/60" />
                      </div>
                    ) : null}
                    <div data-row-key={rowKey} className={cn("messageRow flex", kind, grouped && "grouped")}>
                      {renderConversationEvent(
                        message,
                        kind,
                        activeSessionId || undefined,
                        markdownOptions,
                        kind === "ask_user" ? shouldAllowFuzzyAskUserMatch(messages, index) : true,
                        kind === "ask_user" ? allowLegacyAskUserFallback : false,
                      )}
                    </div>
                  </Fragment>
                );
              })
            ) : (
              <Card className="rounded-[1.35rem] border-dashed border-border/60 bg-muted/20 shadow-none">
                <CardContent className="p-6 text-sm text-muted-foreground">
                  {activeSessionId ? "No conversation events yet." : "Select a session to see its conversation timeline."}
                </CardContent>
              </Card>
            )}
            {isBusy && <WorkingIndicator />}
          </div>
        )}
      </ScrollArea>
      {showPreviousUserJump || showScrollToBottom ? (
        <div className="conversationNavButtons">
          {showPreviousUserJump ? (
            <Button
              data-testid="jump-to-previous-user"
              type="button"
              variant="secondary"
              size="icon"
              className="conversationJumpButton shadow-lg"
              onClick={handleJumpToPreviousUser}
              aria-label="Jump to previous user message"
            >
              <ArrowUpTurnIcon />
            </Button>
          ) : null}
          {showScrollToBottom ? (
            <Button
              data-testid="scroll-to-bottom"
              type="button"
              variant="secondary"
              size="icon"
              className="conversationJumpButton shadow-lg"
              onClick={handleScrollToBottom}
              aria-label="Scroll to conversation bottom"
            >
              <ArrowDownIcon />
            </Button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
