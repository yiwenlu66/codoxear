import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

import {
  useComposerStore,
  useComposerStoreApi,
  useLiveSessionStore,
  useLiveSessionStoreApi,
  useSessionUiStore,
  useSessionUiStoreApi,
  useSessionsStore,
  useSessionsStoreApi,
} from "../../app/providers";
import { api } from "../../lib/api";
import { getSessionRuntimeId } from "../../lib/session-identity";
import type { SessionCommand } from "../../lib/types";
import { getDisplayableTodoSnapshot, TodoComposerPanel } from "./TodoComposerPanel";

function enterToSendEnabled() {
  return window.localStorage.getItem("codoxear.enterToSend") === "1";
}

function getSlashDraftQuery(draft: string) {
  if (!draft.startsWith("/") || /\s/.test(draft.slice(1))) {
    return null;
  }

  return draft.slice(1).toLowerCase();
}

function formatSlashCommandValue(commandName: string) {
  const normalized = commandName.startsWith("/") ? commandName : `/${commandName}`;

  return `${normalized} `;
}

const MOBILE_COMPOSER_QUERY = "(max-width: 880px)";
const MOBILE_COMPOSER_MIN_HEIGHT_PX = 56;
const MOBILE_COMPOSER_MAX_HEIGHT_PX = 176;
const POST_SEND_REFRESH_DELAYS_MS = [1500, 4000, 8000];

function shouldUseMobileComposerAutosize() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }

  return window.matchMedia(MOBILE_COMPOSER_QUERY).matches;
}

function syncComposerTextareaHeight(textarea: HTMLTextAreaElement | null, enabled: boolean) {
  if (!textarea) {
    return;
  }

  if (!enabled) {
    textarea.style.height = "";
    textarea.style.minHeight = "";
    textarea.style.maxHeight = "";
    textarea.style.overflowY = "";
    return;
  }

  textarea.style.minHeight = `${MOBILE_COMPOSER_MIN_HEIGHT_PX}px`;
  textarea.style.maxHeight = `${MOBILE_COMPOSER_MAX_HEIGHT_PX}px`;
  textarea.style.height = "auto";

  const nextHeight = Math.min(
    Math.max(textarea.scrollHeight, MOBILE_COMPOSER_MIN_HEIGHT_PX),
    MOBILE_COMPOSER_MAX_HEIGHT_PX,
  );

  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > MOBILE_COMPOSER_MAX_HEIGHT_PX ? "auto" : "hidden";
}

const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;

function formatContextK(value: number) {
  const normalized = Math.max(0, Math.round(value));
  if (normalized <= 0) {
    return "0";
  }
  return `${Math.round(normalized / 1000)}K`;
}

function getContextUsageLabel(contextUsage: { used_tokens?: number; total_tokens?: number; percent_used?: number } | null | undefined) {
  if (!contextUsage) {
    return null;
  }
  const totalTokens = typeof contextUsage.total_tokens === "number" && Number.isFinite(contextUsage.total_tokens)
    ? Math.max(0, Math.round(contextUsage.total_tokens))
    : 0;
  if (totalTokens <= 0) {
    return null;
  }
  const usedTokens = typeof contextUsage.used_tokens === "number" && Number.isFinite(contextUsage.used_tokens)
    ? Math.min(totalTokens, Math.max(0, Math.round(contextUsage.used_tokens)))
    : 0;
  const percentUsed = typeof contextUsage.percent_used === "number" && Number.isFinite(contextUsage.percent_used)
    ? Math.min(100, Math.max(0, Math.round(contextUsage.percent_used)))
    : Math.min(100, Math.max(0, Math.round((usedTokens / totalTokens) * 100)));
  return `${formatContextK(usedTokens)}/${formatContextK(totalTokens)} ${percentUsed}%`;
}

function safeStem(name: string) {
  const base = String(name || "file").split("/").pop() || "file";
  const dot = base.lastIndexOf(".");

  return (dot > 0 ? base.slice(0, dot) : base).replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80) || "file";
}

function extLower(name: string) {
  const dot = String(name || "").lastIndexOf(".");

  return dot >= 0 ? String(name).slice(dot + 1).toLowerCase() : "";
}

function isLikelyHeic(file: File) {
  const type = String(file.type || "").toLowerCase();
  const ext = extLower(file.name);

  return type.includes("heic") || type.includes("heif") || ext === "heic" || ext === "heif";
}

function looksLikeImage(file: File) {
  const type = String(file.type || "").toLowerCase();
  const ext = extLower(file.name);

  return type.startsWith("image/") || ["png", "jpg", "jpeg", "webp", "gif", "bmp", "svg", "avif", "heic", "heif"].includes(ext);
}

function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  const chunkSize = 0x8000;

  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }

  return btoa(binary);
}

async function blobToArrayBuffer(blob: Blob) {
  if (typeof blob.arrayBuffer === "function") {
    return blob.arrayBuffer();
  }

  return new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(reader.result);
        return;
      }

      reject(new Error("read failed"));
    };
    reader.readAsArrayBuffer(blob);
  });
}

async function toJpegBlob(file: File, options: { maxDim: number; quality: number }) {
  const url = URL.createObjectURL(file);

  try {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    if (image.decode) {
      await image.decode();
    } else {
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error("decode failed"));
      });
    }

    const naturalWidth = image.naturalWidth || image.width || 0;
    const naturalHeight = image.naturalHeight || image.height || 0;
    if (!naturalWidth || !naturalHeight) {
      throw new Error("invalid image dimensions");
    }

    const scale = Math.min(1, options.maxDim / Math.max(naturalWidth, naturalHeight));
    const width = Math.max(1, Math.round(naturalWidth * scale));
    const height = Math.max(1, Math.round(naturalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) {
      throw new Error("no canvas");
    }
    context.drawImage(image, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", options.quality));
    if (!blob) {
      throw new Error("jpeg encode failed");
    }

    return blob;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function Composer() {
  const { activeSessionId, items } = useSessionsStore();
  const { busyBySessionId = {}, contextUsageBySessionId = {} } = useLiveSessionStore() as {
    busyBySessionId?: Record<string, boolean>;
    contextUsageBySessionId?: Record<string, { used_tokens?: number; total_tokens?: number; percent_used?: number } | null>;
  };
  const composerState = useComposerStore();
  const sending = composerState.sending;
  const { sessionId: sessionUiSessionId, diagnostics } = useSessionUiStore();
  const sessionsStoreApi = useSessionsStoreApi();
  const composerStoreApi = useComposerStoreApi();
  const liveSessionStoreApi = useLiveSessionStoreApi();
  const sessionUiStoreApi = useSessionUiStoreApi();
  const [todoExpandedBySessionId, setTodoExpandedBySessionId] = useState<Record<string, boolean>>({});
  const [commandsBySessionId, setCommandsBySessionId] = useState<Record<string, SessionCommand[]>>({});
  const [commandsLoadedBySessionId, setCommandsLoadedBySessionId] = useState<Record<string, boolean>>({});
  const [commandsLoadingBySessionId, setCommandsLoadingBySessionId] = useState<Record<string, boolean>>({});
  const [highlightedCommandIndex, setHighlightedCommandIndex] = useState(0);
  const [slashMenuDismissed, setSlashMenuDismissed] = useState(false);
  const [attachedFilesBySessionId, setAttachedFilesBySessionId] = useState<Record<string, number>>({});
  const [attachmentUploading, setAttachmentUploading] = useState(false);
  const [mobileComposerAutosize, setMobileComposerAutosize] = useState(() => shouldUseMobileComposerAutosize());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const postSendRefreshTimeoutsRef = useRef<number[]>([]);
  const activeSession = items.find((session) => session.session_id === activeSessionId) ?? null;
  const activeSessionRuntimeId = getSessionRuntimeId(activeSession);
  const draft = activeSessionId ? composerState.draftBySessionId?.[activeSessionId] ?? "" : "";
  const activeSessionPending = activeSession?.pending_startup === true;
  const activeSessionBusy = Boolean(activeSession && (activeSession.busy || (activeSessionId ? busyBySessionId[activeSessionId] === true : false)));
  const activeSessionIsPi = activeSession?.agent_backend === "pi";
  const activeSessionIsHistoricalPi = activeSessionIsPi && activeSession?.historical === true;
  const activeAttachmentCount = activeSessionId ? attachedFilesBySessionId[activeSessionId] ?? 0 : 0;
  const attachmentsSupported = Boolean(activeSessionId && activeSession?.agent_backend !== "pi");
  const slashQuery = getSlashDraftQuery(draft);
  const todoSnapshot = useMemo(() => {
    if (!activeSessionId || activeSession?.agent_backend !== "pi") {
      return null;
    }

    if (sessionUiSessionId !== activeSessionId) {
      return null;
    }

    if (!diagnostics || typeof diagnostics !== "object") {
      return null;
    }

    const snapshot = (diagnostics as { todo_snapshot?: unknown }).todo_snapshot;

    return getDisplayableTodoSnapshot(snapshot);
  }, [activeSession?.agent_backend, activeSessionId, diagnostics, sessionUiSessionId]);

  const composerContextUsageLabel = useMemo(() => {
    if (!activeSessionId || !activeSessionIsPi) {
      return null;
    }
    return getContextUsageLabel(contextUsageBySessionId[activeSessionId] ?? null);
  }, [activeSessionId, activeSessionIsPi, contextUsageBySessionId]);

  const visibleTodoExpanded = activeSessionId ? Boolean(todoExpandedBySessionId[activeSessionId]) : false;
  const visibleCommands = useMemo(() => {
    if (!activeSessionId || !activeSessionIsPi || slashQuery === null) {
      return [];
    }

    const commands = commandsBySessionId[activeSessionId] ?? [];

    return commands
      .filter((command) => command.name.toLowerCase().startsWith(slashQuery))
      .slice(0, 8);
  }, [activeSessionId, activeSessionIsPi, commandsBySessionId, slashQuery]);
  const commandsLoaded = activeSessionId ? Boolean(commandsLoadedBySessionId[activeSessionId]) : false;
  const commandsLoading = activeSessionId ? Boolean(commandsLoadingBySessionId[activeSessionId]) : false;
  const commandMenuOpen = Boolean(
    activeSessionIsPi && slashQuery !== null && !slashMenuDismissed && (commandsLoading || visibleCommands.length > 0),
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia(MOBILE_COMPOSER_QUERY);
    const update = () => {
      setMobileComposerAutosize(mediaQuery.matches);
    };

    update();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", update);
    } else {
      mediaQuery.addListener?.(update);
    }

    return () => {
      if (typeof mediaQuery.removeEventListener === "function") {
        mediaQuery.removeEventListener("change", update);
      } else {
        mediaQuery.removeListener?.(update);
      }
    };
  }, []);

  useEffect(() => () => {
    for (const timeoutId of postSendRefreshTimeoutsRef.current) {
      window.clearTimeout(timeoutId);
    }
    postSendRefreshTimeoutsRef.current = [];
  }, []);

  useLayoutEffect(() => {
    syncComposerTextareaHeight(textareaRef.current, mobileComposerAutosize);
  }, [draft, mobileComposerAutosize]);

  useEffect(() => {
    if (!activeSessionId || !activeSessionIsPi || activeSessionPending || slashQuery === null) {
      return;
    }
    if (commandsLoaded || commandsLoading) {
      return;
    }

    let cancelled = false;

    setCommandsLoadingBySessionId((value) => ({
      ...value,
      [activeSessionId]: true,
    }));

    (activeSessionRuntimeId
      ? api.getSessionCommands(activeSessionId, undefined, activeSessionRuntimeId)
      : api.getSessionCommands(activeSessionId))
      .then((response) => {
        if (cancelled) {
          return;
        }

        setCommandsBySessionId((value) => ({
          ...value,
          [activeSessionId]: Array.isArray(response.commands) ? response.commands : [],
        }));
        setCommandsLoadedBySessionId((value) => ({
          ...value,
          [activeSessionId]: true,
        }));
      })
      .catch(() => {
        if (cancelled) {
          return;
        }

        setCommandsBySessionId((value) => ({
          ...value,
          [activeSessionId]: [],
        }));
        setCommandsLoadedBySessionId((value) => ({
          ...value,
          [activeSessionId]: true,
        }));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }

        setCommandsLoadingBySessionId((value) => ({
          ...value,
          [activeSessionId]: false,
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [activeSessionId, activeSessionIsPi, activeSessionPending, activeSessionRuntimeId, slashQuery]);

  useEffect(() => {
    setHighlightedCommandIndex(0);
  }, [activeSessionId, slashQuery]);

  useEffect(() => {
    setSlashMenuDismissed(false);
  }, [activeSessionId, slashQuery]);

  useEffect(() => {
    if (!visibleCommands.length) {
      setHighlightedCommandIndex(0);
      return;
    }

    setHighlightedCommandIndex((value) => Math.min(value, visibleCommands.length - 1));
  }, [visibleCommands.length]);

  const applySlashCommand = (command: SessionCommand | undefined) => {
    if (!command) {
      return;
    }

    composerStoreApi.setDraft(activeSessionId, formatSlashCommandValue(command.name));
    setHighlightedCommandIndex(0);
  };

  const clearAttachmentCount = (sessionId: string) => {
    setAttachedFilesBySessionId((value) => {
      if (!value[sessionId]) {
        return value;
      }

      return {
        ...value,
        [sessionId]: 0,
      };
    });
  };

  const refreshSessionAfterSend = (sessionId: string, runtimeId?: string | null, agentBackend?: string) => {
    const refreshNow = () => Promise.allSettled([
      runtimeId ? liveSessionStoreApi.loadInitial(sessionId, runtimeId) : liveSessionStoreApi.loadInitial(sessionId),
      runtimeId ? sessionUiStoreApi.refresh(sessionId, { agentBackend, runtimeId }) : sessionUiStoreApi.refresh(sessionId, { agentBackend }),
      sessionsStoreApi.refresh(),
    ]);

    for (const timeoutId of postSendRefreshTimeoutsRef.current) {
      window.clearTimeout(timeoutId);
    }
    postSendRefreshTimeoutsRef.current = [];

    for (const delayMs of POST_SEND_REFRESH_DELAYS_MS) {
      const timeoutId = window.setTimeout(() => {
        void Promise.allSettled([
          runtimeId ? liveSessionStoreApi.poll(sessionId, runtimeId) : liveSessionStoreApi.poll(sessionId),
          runtimeId ? sessionUiStoreApi.refresh(sessionId, { agentBackend, runtimeId }) : sessionUiStoreApi.refresh(sessionId, { agentBackend }),
          sessionsStoreApi.refresh(),
        ]);
      }, delayMs);
      postSendRefreshTimeoutsRef.current.push(timeoutId);
    }

    return refreshNow();
  };

  const resolvePostSendSessionIdentity = async (response: unknown, fallbackSessionId: string, fallbackRuntimeId?: string | null) => {
    const nextSessionId = response && typeof response === "object"
      ? String((response as { session_id?: unknown }).session_id || "").trim()
      : "";
    const nextRuntimeId = response && typeof response === "object"
      ? String((response as { runtime_id?: unknown }).runtime_id || "").trim()
      : "";
    if (!nextSessionId || nextSessionId === fallbackSessionId) {
      return { sessionId: fallbackSessionId, runtimeId: fallbackRuntimeId || null };
    }

    await sessionsStoreApi.refresh();
    sessionsStoreApi.select(nextSessionId);
    return { sessionId: nextSessionId, runtimeId: nextRuntimeId || null };
  };

  const submitCurrentDraft = () => {
    if (!activeSessionId || !draft.trim() || sending || activeSessionPending) {
      return;
    }

    (activeSessionRuntimeId
      ? composerStoreApi.submit(activeSessionId, activeSessionRuntimeId)
      : composerStoreApi.submit(activeSessionId))
      .then(async (response) => {
        clearAttachmentCount(activeSessionId);
        const target = await resolvePostSendSessionIdentity(response, activeSessionId, activeSessionRuntimeId);
        await refreshSessionAfterSend(target.sessionId, target.runtimeId, activeSession?.agent_backend);
      })
      .catch(() => undefined);
  };

  const queueCurrentDraft = () => {
    if (!activeSessionId || !draft.trim() || sending) {
      return;
    }

    const queuedText = draft;
    composerStoreApi.setDraft(activeSessionId, "");
    (activeSessionRuntimeId
      ? api.enqueueMessage(activeSessionId, queuedText, activeSessionRuntimeId)
      : api.enqueueMessage(activeSessionId, queuedText))
      .then(async (response) => {
        clearAttachmentCount(activeSessionId);
        const target = await resolvePostSendSessionIdentity(response, activeSessionId, activeSessionRuntimeId);
        if (activeSessionIsHistoricalPi && target.sessionId !== activeSessionId) {
          return refreshSessionAfterSend(target.sessionId, target.runtimeId, activeSession?.agent_backend);
        }
        return target.runtimeId
          ? sessionUiStoreApi.refresh(target.sessionId, { agentBackend: activeSession?.agent_backend, runtimeId: target.runtimeId })
          : sessionUiStoreApi.refresh(target.sessionId, { agentBackend: activeSession?.agent_backend });
      })
      .catch(() => {
        composerStoreApi.setDraft(activeSessionId, queuedText);
      });
  };

  const interruptCurrentLoop = () => {
    if (!activeSessionId || !activeSessionBusy) {
      return;
    }

    const interruptRequest = activeSessionRuntimeId
      ? api.interruptSession(activeSessionId, activeSessionRuntimeId)
      : api.interruptSession(activeSessionId);

    interruptRequest
      .then(() => Promise.allSettled([
        sessionsStoreApi.refresh(),
        activeSessionRuntimeId
          ? liveSessionStoreApi.loadInitial(activeSessionId, activeSessionRuntimeId)
          : liveSessionStoreApi.loadInitial(activeSessionId),
        activeSessionRuntimeId
          ? sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSession?.agent_backend, runtimeId: activeSessionRuntimeId })
          : sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSession?.agent_backend }),
      ]))
      .catch(() => undefined);
  };

  const handleAttachClick = () => {
    if (!attachmentsSupported || attachmentUploading || sending) {
      return;
    }

    const input = fileInputRef.current;
    if (!input) {
      return;
    }

    input.value = "";
    input.click();
  };

  const handleAttachChange = async (event: Event) => {
    if (!attachmentsSupported || !activeSessionId || attachmentUploading || sending) {
      return;
    }

    const input = event.currentTarget as HTMLInputElement | null;
    const file = input?.files?.[0];
    if (input) {
      input.value = "";
    }
    if (!file) {
      return;
    }

    const attachmentIndex = activeAttachmentCount + 1;
    setAttachmentUploading(true);

    try {
      let uploadBlob: Blob = file;
      let uploadName = file.name || "file";

      if (looksLikeImage(file) && (file.size > MAX_ATTACHMENT_BYTES || isLikelyHeic(file))) {
        uploadName = `${safeStem(file.name)}.jpg`;
        const attempts = [
          { maxDim: 2048, quality: 0.86 },
          { maxDim: 1600, quality: 0.82 },
          { maxDim: 1600, quality: 0.72 },
          { maxDim: 1280, quality: 0.68 },
          { maxDim: 1280, quality: 0.58 },
        ];

        let compressedBlob: Blob | null = null;
        for (const attempt of attempts) {
          compressedBlob = await toJpegBlob(file, attempt);
          if (compressedBlob.size <= MAX_ATTACHMENT_BYTES) {
            break;
          }
        }
        if (!compressedBlob || compressedBlob.size > MAX_ATTACHMENT_BYTES) {
          throw new Error("image too large");
        }

        uploadBlob = compressedBlob;
      }

      const buffer = await blobToArrayBuffer(uploadBlob);
      if (buffer.byteLength > MAX_ATTACHMENT_BYTES) {
        throw new Error("file too large");
      }

      if (activeSessionRuntimeId) {
        await api.attachSessionFile(activeSessionId, {
          filename: uploadName,
          data_b64: bytesToBase64(new Uint8Array(buffer)),
          attachment_index: attachmentIndex,
        }, activeSessionRuntimeId);
      } else {
        await api.attachSessionFile(activeSessionId, {
          filename: uploadName,
          data_b64: bytesToBase64(new Uint8Array(buffer)),
          attachment_index: attachmentIndex,
        });
      }

      setAttachedFilesBySessionId((value) => ({
        ...value,
        [activeSessionId]: attachmentIndex,
      }));
    } catch (error) {
      console.error("Failed to attach file", error);
    } finally {
      setAttachmentUploading(false);
    }
  };

  const attachButtonTitle = !activeSessionId
    ? "Select a session first"
    : activeSessionIsPi
      ? "Attachments are not available for Pi sessions"
      : attachmentUploading
        ? "Uploading attachment..."
        : "Attach file";

  return (
    <div className="composerStack space-y-3">
      {todoSnapshot ? (
        <TodoComposerPanel
          snapshot={todoSnapshot}
          expanded={visibleTodoExpanded}
          onToggle={() => {
            const currentSessionId = sessionsStoreApi.getState().activeSessionId;

            if (!currentSessionId) {
              return;
            }

            setTodoExpandedBySessionId((value) => ({
              ...value,
              [currentSessionId]: !value[currentSessionId],
            }));
          }}
        />
      ) : null}
      <Card
        data-testid="composer-card"
        className="composerCard rounded-[1.5rem] border-border/70 bg-card/95 shadow-lg shadow-primary/5 backdrop-blur-sm"
      >
        <CardContent className="p-3 sm:p-4">
          <form
            className={cn("composer composerShell flex items-end gap-2 border-t-0", draft.includes("\n") && "multiline")}
            onSubmit={(event) => {
              event.preventDefault();
              submitCurrentDraft();
            }}
          >
            <div className="composerInputWrap flex-1">
              <Textarea
                textareaRef={textareaRef}
                value={draft}
                rows={mobileComposerAutosize ? 2 : undefined}
                placeholder="Enter your instructions here"
                className="composerTextarea"
                onInput={(event) => {
                  syncComposerTextareaHeight(event.currentTarget, mobileComposerAutosize);
                  composerStoreApi.setDraft(activeSessionId, event.currentTarget.value);
                }}
                onKeyDown={(event) => {
                  if (commandMenuOpen) {
                    if (event.key === "ArrowDown") {
                      event.preventDefault();
                      setHighlightedCommandIndex((value) => (visibleCommands.length ? (value + 1) % visibleCommands.length : 0));
                      return;
                    }
                    if (event.key === "ArrowUp") {
                      event.preventDefault();
                      setHighlightedCommandIndex((value) => {
                        if (!visibleCommands.length) {
                          return 0;
                        }

                        return value <= 0 ? visibleCommands.length - 1 : value - 1;
                      });
                      return;
                    }
                    if (event.key === "Escape") {
                      event.preventDefault();
                      setSlashMenuDismissed(true);
                      return;
                    }
                    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && visibleCommands.length) {
                      event.preventDefault();
                      applySlashCommand(visibleCommands[highlightedCommandIndex] ?? visibleCommands[0]);
                      return;
                    }
                  }
                  if (event.key !== "Enter" || event.isComposing) {
                    return;
                  }
                  if (event.shiftKey) {
                    return;
                  }
                  if (!enterToSendEnabled() && !event.ctrlKey && !event.metaKey) {
                    return;
                  }
                  if (!activeSessionId) {
                    return;
                  }
                  event.preventDefault();
                  submitCurrentDraft();
                }}
                disabled={sending}
              />
              {commandMenuOpen ? (
                <div className="composerCommandMenu" data-testid="composer-command-menu">
                  {commandsLoading ? <div className="composerCommandHint">Loading Pi commands...</div> : null}
                  {!commandsLoading
                    ? visibleCommands.map((command, index) => (
                      <button
                        key={command.name}
                        type="button"
                        className={cn("composerCommandItem", index === highlightedCommandIndex && "is-active")}
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => applySlashCommand(command)}
                      >
                        <span className="composerCommandName">/{command.name}</span>
                        {command.description ? <span className="composerCommandDescription">{command.description}</span> : null}
                      </button>
                    ))
                    : null}
                </div>
              ) : null}
            </div>
            <div className="composerControlsColumn">
              {composerContextUsageLabel ? <div className="composerContextUsage">{composerContextUsageLabel}</div> : null}
              <div className="composerControlsRow">
                <input ref={fileInputRef} type="file" hidden tabIndex={-1} onChange={handleAttachChange} />
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="composerAttachButton"
                  aria-label="Attach file"
                  title={attachButtonTitle}
                  disabled={!attachmentsSupported || attachmentUploading || sending}
                  onClick={handleAttachClick}
                >
                  <span className="buttonGlyph">📎</span>
                  {activeAttachmentCount > 0 ? <span className="composerAttachBadge">{activeAttachmentCount}</span> : null}
                  <span className="visuallyHidden">Attach file</span>
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="composerQueueButton"
                  aria-label="Queued messages"
                  disabled={sending || !draft.trim()}
                  onClick={queueCurrentDraft}
                >
                  Queue
                </Button>
                {activeSessionBusy ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="composerInterruptButton"
                    aria-label="Cancel current loop"
                    onClick={interruptCurrentLoop}
                  >
                    Cancel loop
                  </Button>
                ) : null}
                <Button
                  type="submit"
                  className="sendButton"
                  aria-label={sending ? "Sending" : "Send"}
                  disabled={sending || activeSessionPending || !draft.trim()}
                >
                  <span className="buttonGlyph">➤</span>
                  <span className="visuallyHidden">{sending ? "Sending..." : "Send"}</span>
                </Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
