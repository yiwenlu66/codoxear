import type { ComponentChildren, JSX } from "preact";
import { useRef, useState } from "preact/hooks";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

import { useLiveSessionStore, useLiveSessionStoreApi, useSessionUiStore, useSessionUiStoreApi } from "../../app/providers";
import { api } from "../../lib/api";
import type { SessionUiRequest, TodoSnapshot, TodoSnapshotItem } from "../../lib/types";

type DraftValue = string | string[];
type OptionInput = { label?: string; value?: string; title?: string; description?: string } | string;
type AskUserBridgeQuestion = {
  header: string;
  question: string;
  options: Array<{ label: string; description?: string; preview?: string }>;
  multiSelect?: boolean;
};
type AskUserBridgeRequest = {
  questions: AskUserBridgeQuestion[];
  metadata?: Record<string, unknown>;
};
type AskUserBridgeAnswers = Record<string, string | string[]>;

const ASK_USER_BRIDGE_PREFIX = "__codoxear_ask_user_bridge_v1__";

function normalizeOption(option: OptionInput, index: number) {
  if (typeof option === "string") {
    return { label: option, description: "", value: option, key: option || String(index) };
  }

  const label = option.label ?? option.title ?? option.value ?? `Option ${index + 1}`;
  const value = String(option.value ?? option.title ?? label ?? "");

  return {
    label,
    description: option.description ?? "",
    value,
    key: value || String(index),
  };
}

function parseAskUserBridgeRequest(request: SessionUiRequest): AskUserBridgeRequest | null {
  if (request.method !== "editor") {
    return null;
  }

  const prefill = typeof request.prefill === "string" ? request.prefill : "";
  if (!prefill.startsWith(`${ASK_USER_BRIDGE_PREFIX}\n`)) {
    return null;
  }

  try {
    const parsed = JSON.parse(prefill.slice(ASK_USER_BRIDGE_PREFIX.length + 1)) as {
      questions?: unknown;
      metadata?: Record<string, unknown>;
    };
    if (!Array.isArray(parsed.questions) || !parsed.questions.length) {
      return null;
    }

    const questions: AskUserBridgeQuestion[] = [];
    for (const question of parsed.questions) {
      if (!question || typeof question !== "object") {
        continue;
      }
      const row = question as Record<string, unknown>;
      const header = typeof row.header === "string" ? row.header.trim() : "";
      const prompt = typeof row.question === "string" ? row.question.trim() : "";
      const options: AskUserBridgeQuestion["options"] = [];
      if (Array.isArray(row.options)) {
        for (const option of row.options) {
          if (!option || typeof option !== "object") {
            continue;
          }
          const value = option as Record<string, unknown>;
          const label = typeof value.label === "string" ? value.label.trim() : "";
          if (!label) {
            continue;
          }
          options.push({
            label,
            description: typeof value.description === "string" ? value.description.trim() : undefined,
            preview: typeof value.preview === "string" ? value.preview : undefined,
          });
        }
      }

      if (!header || !prompt || !options.length) {
        continue;
      }

      questions.push({
        header,
        question: prompt,
        options,
        multiSelect: row.multiSelect === true,
      });
    }

    if (!questions.length) {
      return null;
    }

    return {
      questions,
      metadata: parsed.metadata,
    };
  } catch {
    return null;
  }
}

function encodeAskUserBridgeResponse(answers: AskUserBridgeAnswers) {
  return `${ASK_USER_BRIDGE_PREFIX}\n${JSON.stringify({ action: "answered", answers })}`;
}

function getInitialDraftValue(request: SessionUiRequest): DraftValue {
  if (Array.isArray(request.value)) {
    return request.value.filter((item): item is string => typeof item === "string");
  }
  if (typeof request.value === "string") {
    return request.value;
  }
  if (request.method === "select" && Array.isArray(request.options) && request.options.length > 0 && !request.allow_multiple) {
    return normalizeOption(request.options[0], 0).value;
  }
  return request.allow_multiple ? [] : "";
}

function normalizeRequestValue(request: SessionUiRequest, draftValue: DraftValue): string | string[] | undefined {
  if (request.method === "confirm") {
    return undefined;
  }
  if (request.allow_multiple) {
    return Array.isArray(draftValue) ? draftValue : draftValue ? [draftValue] : [];
  }
  return Array.isArray(draftValue) ? draftValue[0] ?? "" : draftValue;
}

function getRequestHeading(request: SessionUiRequest): string {
  return request.title || request.label || request.question || request.method || "Request";
}

function getRequestBody(request: SessionUiRequest): string {
  return request.message || request.context || "";
}

function entriesFromRecord(value: Record<string, unknown> | null) {
  return value ? Object.entries(value) : [];
}

function queueItemsFromValue(queue: Record<string, unknown> | null) {
  const rawItems = queue?.items;
  if (!Array.isArray(rawItems)) {
    return [];
  }
  return rawItems.map((item) => {
    if (item && typeof item === "object" && "text" in item) {
      return String((item as { text?: unknown }).text ?? "");
    }
    return String(item);
  });
}

function normalizeTodoItem(value: unknown): TodoSnapshotItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const item = value as Record<string, unknown>;
  return {
    id: typeof item.id === "number" || typeof item.id === "string" ? item.id : undefined,
    title: typeof item.title === "string" ? item.title : undefined,
    status: typeof item.status === "string" ? item.status : undefined,
    description: typeof item.description === "string" ? item.description : undefined,
  };
}

function normalizeTodoSnapshot(snapshot: unknown): TodoSnapshot {
  if (!snapshot || typeof snapshot !== "object") {
    return { available: false, error: false, items: [] };
  }
  const raw = snapshot as Record<string, unknown>;
  return {
    available: raw.available === true,
    error: raw.error === true,
    progress_text: typeof raw.progress_text === "string" ? raw.progress_text : undefined,
    items: Array.isArray(raw.items)
      ? raw.items.map(normalizeTodoItem).filter((item): item is TodoSnapshotItem => Boolean(item))
      : [],
  };
}

function formatDiagnosticLabel(key: string): string {
  switch (key) {
    case "log_path":
      return "Log";
    case "session_file_path":
      return "Session file";
    case "updated_ts":
      return "Updated";
    case "cwd":
      return "Working directory";
    case "queue_len":
      return "Queue";
    default:
      return key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }
}

function formatDiagnosticValue(key: string, value: unknown): string {
  if (key === "updated_ts" && typeof value === "number" && Number.isFinite(value) && value > 1_000_000_000) {
    return new Date(value * 1000).toLocaleString();
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function renderTodoSnapshotSection(snapshot: TodoSnapshot) {
  return (
    <div className="space-y-3 rounded-2xl border border-border/60 bg-card/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-foreground">Todo list</p>
          {snapshot.progress_text ? <p className="text-sm text-muted-foreground">{snapshot.progress_text}</p> : null}
        </div>
        <Badge variant="outline">{snapshot.available ? `${snapshot.items.length}` : "0"}</Badge>
      </div>
      {!snapshot.available ? (
        <p className="text-sm text-muted-foreground">{snapshot.error ? "Todo list unavailable" : "No todo list yet"}</p>
      ) : (
        <div className="space-y-2">
          {snapshot.items.map((item, index) => (
            <article key={`${item.title || "todo"}-${index}`} className="rounded-xl border border-border/60 bg-background/70 px-3 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <strong className="text-sm text-foreground">{item.title || "Untitled todo"}</strong>
                <Badge variant="secondary">{item.status || "unknown"}</Badge>
              </div>
              {item.description ? <p className="mt-2 text-sm text-muted-foreground">{item.description}</p> : null}
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function mergeFreeformValue(request: SessionUiRequest, normalizedValue: string | string[] | undefined, freeformValue: string) {
  const trimmedFreeform = freeformValue.trim();

  if (!trimmedFreeform) {
    return normalizedValue;
  }

  if (request.allow_multiple) {
    const existingValues = Array.isArray(normalizedValue)
      ? normalizedValue
      : normalizedValue
        ? [normalizedValue]
        : [];
    return [...existingValues, trimmedFreeform];
  }

  return trimmedFreeform;
}

function SelectField(props: JSX.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
      {...props}
    />
  );
}

function WorkspaceSection({
  title,
  badge,
  children,
}: {
  title: string;
  badge?: string;
  children: ComponentChildren;
}) {
  return (
    <section className="workspaceSurface space-y-3 rounded-[1.2rem] border border-border/70 bg-background/75 p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {badge ? <Badge variant="outline">{badge}</Badge> : null}
      </div>
      {children}
    </section>
  );
}

interface SessionWorkspaceProps {
  mode?: "default" | "details";
}

export function SessionWorkspace({ mode = "default" }: SessionWorkspaceProps) {
  const sessionUiState = useSessionUiStore() as {
    sessionId: string | null;
    runtimeId: string | null;
    diagnostics: Record<string, unknown> | null;
    queue: Record<string, unknown> | null;
    loading: boolean;
    requests?: SessionUiRequest[];
    files?: string[];
  };
  const { sessionId, runtimeId, diagnostics, queue, loading } = sessionUiState;
  const liveSessionState = useLiveSessionStore();
  const liveSessionStoreApi = useLiveSessionStoreApi();
  const liveRequests = sessionId ? liveSessionState.requestsBySessionId[sessionId] ?? [] : [];
  const requests = liveRequests.length ? liveRequests : Array.isArray(sessionUiState.requests) ? sessionUiState.requests : [];
  const files = Array.isArray(sessionUiState.files) ? sessionUiState.files : [];
  const sessionUiStoreApi = useSessionUiStoreApi();
  const [drafts, setDrafts] = useState<Record<string, DraftValue>>({});
  const [freeformDrafts, setFreeformDrafts] = useState<Record<string, string>>({});
  const [askUserBridgeDrafts, setAskUserBridgeDrafts] = useState<Record<string, AskUserBridgeAnswers>>({});
  const [requestSubmittingById, setRequestSubmittingById] = useState<Record<string, boolean>>({});
  const [requestErrorById, setRequestErrorById] = useState<Record<string, string>>({});
  const requestSubmittingIdsRef = useRef(new Set<string>());
  const diagnosticsEntries = entriesFromRecord(diagnostics);
  const todoSnapshot = normalizeTodoSnapshot(diagnostics && typeof diagnostics === "object" ? (diagnostics as { todo_snapshot?: unknown }).todo_snapshot : null);
  const detailEntries = diagnosticsEntries.filter(([key]) => key !== "todo_snapshot");
  const prioritizedDetailKeys = new Set(["session_file_path", "log_path", "updated_ts"]);
  const priorityDetailEntries = detailEntries.filter(([key]) => prioritizedDetailKeys.has(key));
  const genericDetailEntries = detailEntries.filter(([key]) => !prioritizedDetailKeys.has(key));
  const queueItems = queueItemsFromValue(queue);
  const showDetails = mode === "details";
  const hasWorkspaceData = diagnosticsEntries.length > 0 || queueItems.length > 0;
  const showTabs = showDetails || hasWorkspaceData || requests.length > 0;
  const defaultTab = showDetails
    ? "overview"
    : requests.length > 0
      ? "requests"
      : diagnosticsEntries.length > 0
        ? "diagnostics"
        : queueItems.length > 0
          ? "queue"
          : "requests";

  const submitRequestResponse = async (requestId: string, payload: Record<string, unknown>) => {
    if (!sessionId || requestSubmittingIdsRef.current.has(requestId)) {
      return;
    }

    requestSubmittingIdsRef.current.add(requestId);
    setRequestSubmittingById((current) => ({ ...current, [requestId]: true }));
    setRequestErrorById((current) => ({ ...current, [requestId]: "" }));

    try {
      await (runtimeId ? api.submitUiResponse(sessionId, payload, runtimeId) : api.submitUiResponse(sessionId, payload));
      await Promise.all([
        runtimeId ? liveSessionStoreApi.loadInitial(sessionId, runtimeId) : liveSessionStoreApi.loadInitial(sessionId),
        runtimeId ? sessionUiStoreApi.refresh(sessionId, { agentBackend: "pi", runtimeId }) : sessionUiStoreApi.refresh(sessionId, { agentBackend: "pi" }),
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to submit response";
      setRequestErrorById((current) => ({ ...current, [requestId]: message }));
    } finally {
      requestSubmittingIdsRef.current.delete(requestId);
      setRequestSubmittingById((current) => ({ ...current, [requestId]: false }));
    }
  };

  return (
    <aside className="workspacePane">
      <Card
        data-testid="workspace-card"
        className="workspaceCard flex h-full min-h-0 flex-col rounded-[1.5rem] border-border/70 bg-card/95 shadow-lg shadow-primary/5"
      >
        <CardHeader className="space-y-4 pb-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="text-base">Workspace</CardTitle>
              <p className="text-sm text-muted-foreground">
                {requests.length ? `${requests.length} pending UI request${requests.length === 1 ? "" : "s"}` : "No pending UI requests"}
              </p>
            </div>
            <Badge variant={loading ? "default" : hasWorkspaceData ? "secondary" : "outline"}>
              {loading ? "Refreshing" : hasWorkspaceData ? "Live context" : "Quiet"}
            </Badge>
          </div>
          {showTabs ? (
            <Tabs defaultValue={defaultTab} className="min-h-0 flex-1">
              <TabsList className="workspaceTabsList flex h-auto flex-wrap items-center gap-2 rounded-2xl bg-muted/60 p-1">
                {showDetails ? <TabsTrigger value="overview">Overview</TabsTrigger> : null}
                <TabsTrigger value="requests">UI Requests</TabsTrigger>
                <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
                <TabsTrigger value="queue">Queue</TabsTrigger>
                {files.length ? <TabsTrigger value="files">Files</TabsTrigger> : null}
              </TabsList>
              <Separator className="bg-border/70" />
              <CardContent className="flex min-h-0 flex-1 flex-col p-0 pt-4">
                {showDetails ? (
                  <TabsContent value="overview" className="min-h-0">
                    <ScrollArea className="workspaceScroll h-full pr-1">
                      <div className="workspacePanelGrid grid gap-4 lg:grid-cols-2">
                        <WorkspaceSection title="Diagnostics" badge={diagnosticsEntries.length ? `${diagnosticsEntries.length}` : undefined}>
                          {detailEntries.length || todoSnapshot.available || todoSnapshot.error ? (
                            <div className="space-y-4">
                              {priorityDetailEntries.length ? (
                                <dl className="space-y-3">
                                  {priorityDetailEntries.map(([key, value]) => (
                                    <div key={key} className="grid gap-1 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
                                      <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{formatDiagnosticLabel(key)}</dt>
                                      <dd className="m-0 break-all font-mono text-sm text-foreground">{formatDiagnosticValue(key, value)}</dd>
                                    </div>
                                  ))}
                                </dl>
                              ) : null}
                              {todoSnapshot.available || todoSnapshot.error ? renderTodoSnapshotSection(todoSnapshot) : null}
                              {genericDetailEntries.length ? (
                                <dl className="space-y-3">
                                  {genericDetailEntries.map(([key, value]) => (
                                    <div key={key} className="grid gap-1 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
                                      <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{formatDiagnosticLabel(key)}</dt>
                                      <dd className="m-0 text-sm text-foreground">{formatDiagnosticValue(key, value)}</dd>
                                    </div>
                                  ))}
                                </dl>
                              ) : null}
                            </div>
                          ) : (
                            <p className="text-sm text-muted-foreground">No diagnostics available.</p>
                          )}
                        </WorkspaceSection>
                        <WorkspaceSection title="Queue" badge={queueItems.length ? `${queueItems.length}` : undefined}>
                          {queueItems.length ? (
                            <ul className="workspaceCollection space-y-2 text-sm text-foreground">
                              {queueItems.map((item, index) => (
                                <li key={`${item}-${index}`} className="rounded-xl border border-border/60 bg-card/60 px-3 py-2">
                                  {item}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-sm text-muted-foreground">No queued items.</p>
                          )}
                        </WorkspaceSection>
                        {files.length ? (
                          <WorkspaceSection title="Files" badge={`${files.length}`}>
                            <ul className="workspaceCollection space-y-2 text-sm text-foreground">
                              {files.map((file) => (
                                <li key={file} className="rounded-xl border border-border/60 bg-card/60 px-3 py-2 font-mono text-xs sm:text-sm">
                                  {file}
                                </li>
                              ))}
                            </ul>
                          </WorkspaceSection>
                        ) : null}
                        <WorkspaceSection title="UI Requests" badge={requests.length ? `${requests.length}` : undefined}>
                          <p className="text-sm text-muted-foreground">
                            {requests.length ? "Review and respond in the dedicated tab." : "No pending requests."}
                          </p>
                        </WorkspaceSection>
                      </div>
                    </ScrollArea>
                  </TabsContent>
                ) : null}
                <TabsContent value="requests" className="min-h-0">
                  <ScrollArea className="workspaceScroll h-full pr-1">
                    <div className="space-y-4">
                      {requests.length ? (
                        requests.map((request, index) => {
                          const requestId = String(request.id ?? index);
                          const askUserBridge = parseAskUserBridgeRequest(request);
                          const draftValue = drafts[requestId] ?? getInitialDraftValue(request);
                          const freeformValue = freeformDrafts[requestId] ?? "";
                          const options = Array.isArray(request.options) ? request.options : [];
                          const bodyText = getRequestBody(request);
                          const selectedValues = Array.isArray(draftValue) ? draftValue : [];
                          const askUserBridgeAnswers = askUserBridgeDrafts[requestId] ?? {};
                          const askUserBridgeReady = Boolean(
                            askUserBridge && askUserBridge.questions.every((question) => {
                              const answer = askUserBridgeAnswers[question.question];
                              return Array.isArray(answer) ? answer.length > 0 : typeof answer === "string" && answer.trim().length > 0;
                            })
                          );

                          return (
                            <Card key={requestId} className="rounded-[1.2rem] border-border/70 bg-background/75 shadow-sm">
                              <CardContent className="space-y-4 p-4">
                                <div className="space-y-2">
                                  <div className="flex flex-wrap items-center gap-2">
                                    <Badge variant="secondary">{request.method || "request"}</Badge>
                                    {!askUserBridge && request.allow_multiple ? <Badge variant="outline">multi-select</Badge> : null}
                                    {!askUserBridge && request.allow_freeform ? <Badge variant="outline">freeform</Badge> : null}
                                  </div>
                                  <div>
                                    <h3 className="text-sm font-semibold text-foreground">{askUserBridge ? "AskUserQuestion" : getRequestHeading(request)}</h3>
                                    {bodyText ? <p className="mt-1 text-sm text-muted-foreground">{bodyText}</p> : null}
                                  </div>
                                </div>

                                {askUserBridge ? (
                                  <div className="space-y-4">
                                    {askUserBridge.questions.map((question) => {
                                      const currentAnswer = askUserBridgeAnswers[question.question];
                                      const selectedAnswer = Array.isArray(currentAnswer)
                                        ? currentAnswer
                                        : typeof currentAnswer === "string"
                                          ? [currentAnswer]
                                          : [];
                                      return (
                                        <section key={question.question} className="space-y-3 rounded-2xl border border-border/60 bg-card/60 p-3">
                                          <div>
                                            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{question.header}</p>
                                            <h4 className="text-sm font-semibold text-foreground">{question.question}</h4>
                                          </div>
                                          <div className="flex flex-wrap gap-2">
                                            {question.options.map((option) => {
                                              const isSelected = selectedAnswer.includes(option.label);
                                              return (
                                                <Button
                                                  key={`${question.question}-${option.label}`}
                                                  type="button"
                                                  variant={isSelected ? "default" : "outline"}
                                                  className="h-auto min-h-10 rounded-full px-4 py-2 text-left"
                                                  onClick={() => {
                                                    setAskUserBridgeDrafts((current) => {
                                                      const existing = current[requestId] ?? {};
                                                      const previous = existing[question.question];
                                                      const previousValues = Array.isArray(previous)
                                                        ? previous
                                                        : typeof previous === "string"
                                                          ? [previous]
                                                          : [];
                                                      const nextValue = question.multiSelect
                                                        ? previousValues.includes(option.label)
                                                          ? previousValues.filter((value) => value !== option.label)
                                                          : [...previousValues, option.label]
                                                        : option.label;
                                                      return {
                                                        ...current,
                                                        [requestId]: {
                                                          ...existing,
                                                          [question.question]: nextValue,
                                                        },
                                                      };
                                                    });
                                                  }}
                                                >
                                                  <span className="flex flex-col items-start gap-1">
                                                    <span>{option.label}</span>
                                                    {option.description ? <span className="text-xs font-normal text-muted-foreground">{option.description}</span> : null}
                                                  </span>
                                                </Button>
                                              );
                                            })}
                                          </div>
                                        </section>
                                      );
                                    })}
                                  </div>
                                ) : null}

                                {!askUserBridge && (request.method === "confirm" ? null : options.length ? (
                                  request.allow_multiple ? (
                                    <div className="space-y-2">
                                      {options.map((option, optionIndex) => {
                                        const normalized = normalizeOption(option, optionIndex);
                                        return (
                                          <label key={normalized.key} className="workspaceOption flex cursor-pointer gap-3 rounded-2xl border border-border/60 bg-card/60 px-3 py-3 text-sm">
                                            <input
                                              type="checkbox"
                                              checked={selectedValues.includes(normalized.value)}
                                              onInput={(event) => {
                                                const checked = event.currentTarget.checked;
                                                setDrafts((current) => {
                                                  const existing = Array.isArray(current[requestId]) ? current[requestId] : [];
                                                  const next = checked
                                                    ? [...existing, normalized.value]
                                                    : existing.filter((value) => value !== normalized.value);
                                                  return { ...current, [requestId]: next };
                                                });
                                              }}
                                            />
                                            <span className="space-y-1">
                                              <span className="block font-medium text-foreground">{normalized.label}</span>
                                              {normalized.description ? (
                                                <span className="block text-muted-foreground">{normalized.description}</span>
                                              ) : null}
                                            </span>
                                          </label>
                                        );
                                      })}
                                    </div>
                                  ) : (
                                    <SelectField
                                      value={Array.isArray(draftValue) ? draftValue[0] ?? "" : draftValue}
                                      onInput={(event) => setDrafts((current) => ({ ...current, [requestId]: event.currentTarget.value }))}
                                    >
                                      {options.map((option, optionIndex) => {
                                        const normalized = normalizeOption(option, optionIndex);
                                        return (
                                          <option key={normalized.key} value={normalized.value}>
                                            {normalized.label}
                                          </option>
                                        );
                                      })}
                                    </SelectField>
                                  )
                                ) : (
                                  <Textarea
                                    value={Array.isArray(draftValue) ? draftValue.join("\n") : draftValue}
                                    className="min-h-[8rem] rounded-2xl border-border/70 bg-background/80"
                                    onInput={(event) => setDrafts((current) => ({ ...current, [requestId]: event.currentTarget.value }))}
                                  />
                                ))}

                                {!askUserBridge && request.allow_freeform ? (
                                  <div className="space-y-2">
                                    <label className="text-sm font-medium text-foreground">Other response</label>
                                    <Textarea
                                      value={freeformValue}
                                      placeholder="Other response"
                                      className="min-h-[6rem] rounded-2xl border-border/70 bg-background/80"
                                      onInput={(event) =>
                                        setFreeformDrafts((current) => ({ ...current, [requestId]: event.currentTarget.value }))
                                      }
                                    />
                                  </div>
                                ) : null}

                                {sessionId ? (
                                  <div className="space-y-2">
                                    <div className="formActions gap-2">
                                      <Button
                                        type="button"
                                        disabled={Boolean(requestSubmittingById[requestId]) || Boolean(askUserBridge && !askUserBridgeReady)}
                                        onClick={() => {
                                          const payload =
                                            askUserBridge
                                              ? {
                                                  id: request.id,
                                                  value: encodeAskUserBridgeResponse(askUserBridgeAnswers),
                                                }
                                              : request.method === "confirm"
                                              ? { id: request.id, confirmed: true }
                                              : {
                                                  id: request.id,
                                                  value: mergeFreeformValue(request, normalizeRequestValue(request, draftValue), freeformValue),
                                                };
                                          void submitRequestResponse(requestId, payload);
                                        }}
                                      >
                                        {requestSubmittingById[requestId] ? "Submitting..." : "Confirm"}
                                      </Button>
                                      <Button
                                        type="button"
                                        variant="outline"
                                        disabled={Boolean(requestSubmittingById[requestId])}
                                        onClick={() => {
                                          void submitRequestResponse(requestId, {
                                            id: request.id,
                                            cancelled: true,
                                          });
                                        }}
                                      >
                                        Cancel
                                      </Button>
                                    </div>
                                    {requestErrorById[requestId] ? (
                                      <p className="text-sm font-medium text-destructive">{requestErrorById[requestId]}</p>
                                    ) : null}
                                  </div>
                                ) : null}
                              </CardContent>
                            </Card>
                          );
                        })
                      ) : (
                        <WorkspaceSection title="UI Requests">
                          <p className="text-sm text-muted-foreground">No pending requests.</p>
                        </WorkspaceSection>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>
                <TabsContent value="diagnostics" className="min-h-0">
                  <ScrollArea className="workspaceScroll h-full pr-1">
                        <WorkspaceSection title="Diagnostics" badge={diagnosticsEntries.length ? `${diagnosticsEntries.length}` : undefined}>
                          {detailEntries.length || todoSnapshot.available || todoSnapshot.error ? (
                            <div className="space-y-4">
                              {priorityDetailEntries.length ? (
                                <dl className="space-y-3">
                                  {priorityDetailEntries.map(([key, value]) => (
                                    <div key={key} className="grid gap-1 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
                                      <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{formatDiagnosticLabel(key)}</dt>
                                      <dd className="m-0 break-all font-mono text-sm text-foreground">{formatDiagnosticValue(key, value)}</dd>
                                    </div>
                                  ))}
                                </dl>
                              ) : null}
                              {todoSnapshot.available || todoSnapshot.error ? renderTodoSnapshotSection(todoSnapshot) : null}
                              {genericDetailEntries.length ? (
                                <dl className="space-y-3">
                                  {genericDetailEntries.map(([key, value]) => (
                                    <div key={key} className="grid gap-1 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
                                      <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{formatDiagnosticLabel(key)}</dt>
                                      <dd className="m-0 text-sm text-foreground">{formatDiagnosticValue(key, value)}</dd>
                                    </div>
                                  ))}
                                </dl>
                              ) : null}
                            </div>
                          ) : (
                            <p className="text-sm text-muted-foreground">No diagnostics available.</p>
                          )}
                        </WorkspaceSection>
                  </ScrollArea>
                </TabsContent>
                <TabsContent value="queue" className="min-h-0">
                  <ScrollArea className="workspaceScroll h-full pr-1">
                    <WorkspaceSection title="Queue" badge={queueItems.length ? `${queueItems.length}` : undefined}>
                      {queueItems.length ? (
                        <ul className="workspaceCollection space-y-2 text-sm text-foreground">
                          {queueItems.map((item, index) => (
                            <li key={`${item}-${index}`} className="rounded-xl border border-border/60 bg-card/60 px-3 py-2">
                              {item}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-muted-foreground">No queued items.</p>
                      )}
                    </WorkspaceSection>
                  </ScrollArea>
                </TabsContent>
                {files.length ? (
                  <TabsContent value="files" className="min-h-0">
                    <ScrollArea className="workspaceScroll h-full pr-1">
                      <WorkspaceSection title="Files" badge={`${files.length}`}>
                        <ul className="workspaceCollection space-y-2 text-sm text-foreground">
                          {files.map((file) => (
                            <li key={file} className="rounded-xl border border-border/60 bg-card/60 px-3 py-2 font-mono text-xs sm:text-sm">
                              {file}
                            </li>
                          ))}
                        </ul>
                      </WorkspaceSection>
                    </ScrollArea>
                  </TabsContent>
                ) : null}
              </CardContent>
            </Tabs>
          ) : (
            <>
              <Separator className="bg-border/70" />
              <CardContent className="pt-4">
                <WorkspaceSection title="UI Requests">
                  <p className="text-sm text-muted-foreground">No pending requests.</p>
                </WorkspaceSection>
              </CardContent>
            </>
          )}
        </CardHeader>
      </Card>
    </aside>
  );
}
