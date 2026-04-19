import type { JSX } from "preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

import { useSessionsStore, useSessionsStoreApi } from "../../app/providers";
import { api } from "../../lib/api";
import { providerChoiceToSettings } from "../../lib/launch";
import { getSessionDisplayName } from "../../lib/session-display";
import type { LaunchBackendDefaults, SessionResumeCandidate, SessionResumeCandidatesResponse } from "../../lib/types";

interface NewSessionDialogProps {
  open: boolean;
  onClose: () => void;
}

interface SessionCwdInfo {
  exists: boolean;
  willCreate: boolean;
  gitRepo: boolean;
  gitRoot: string;
  gitBranch: string;
}

type LaunchSettingField = "backend" | "model" | "providerChoice" | "reasoningEffort" | "createInTmux" | "fastMode";
type NewSessionSurfaceTab = "launch" | "focus";

function baseName(value: string) {
  const trimmed = value.trim().replace(/[\\/]+$/, "");
  if (!trimmed) return "";
  const parts = trimmed.split(/[\\/]+/);
  return parts[parts.length - 1] || "";
}

function uniqueStrings(values: Array<string | null | undefined>) {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    result.push(trimmed);
  }
  return result;
}

function initialCwdForDialog(activeSessionCwd: string | null | undefined, recentCwds: string[]) {
  const active = activeSessionCwd?.trim();
  if (active) return active;
  return recentCwds.find((cwd) => typeof cwd === "string" && cwd.trim())?.trim() || "";
}

function providerChoicesForDefaults(defaults: LaunchBackendDefaults) {
  return uniqueStrings([...(defaults.provider_choices ?? []), defaults.provider_choice, ...(defaults.model_providers ?? [])]);
}

function reasoningChoicesForDefaults(defaults: LaunchBackendDefaults) {
  return uniqueStrings([...(defaults.reasoning_efforts ?? []), defaults.reasoning_effort]);
}

function modelChoicesForDefaults(defaults: LaunchBackendDefaults, backend: string, providerChoice: string) {
  if (backend === "pi") {
    const scopedModels = defaults.provider_models?.[providerChoice] ?? [];
    const configuredModel = defaults.provider_choice === providerChoice ? defaults.model : undefined;
    return uniqueStrings([...scopedModels, configuredModel]);
  }
  return uniqueStrings([...(defaults.models ?? []), defaults.model]);
}

function defaultPiModelForProvider(defaults: LaunchBackendDefaults, providerChoice: string) {
  const scopedModels = uniqueStrings(defaults.provider_models?.[providerChoice] ?? []);
  if (defaults.provider_choice === providerChoice) {
    const configuredModel = defaults.model?.trim();
    if (configuredModel) {
      return configuredModel;
    }
  }
  return scopedModels[0] || "";
}

const RESUME_PAGE_SIZE = 20;

function resumeOptionLabel(item: SessionResumeCandidate) {
  const title = item.title?.trim() || item.alias?.trim() || item.first_user_message?.trim() || item.session_id.slice(0, 8);
  const branch = item.git_branch?.trim();
  return branch ? `${title} (${branch})` : title;
}

function SelectField(props: JSX.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
      {...props}
    />
  );
}

function ToggleField({
  label,
  name,
  checked,
  disabled,
  description,
  onChange,
}: {
  label: string;
  name?: string;
  checked: boolean;
  disabled?: boolean;
  description?: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className={cn("toggleOption flex cursor-pointer items-start gap-3 rounded-2xl border border-border/70 bg-background/80 px-3 py-3 text-sm", disabled && "cursor-not-allowed opacity-60")}>
      <input
        type="checkbox"
        name={name}
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.currentTarget.checked)}
      />
      <span className="space-y-1">
        <span className="block font-medium text-foreground">{label}</span>
        {description ? <span className="block text-muted-foreground">{description}</span> : null}
      </span>
    </label>
  );
}

export function NewSessionDialog({ open, onClose }: NewSessionDialogProps) {
  const { activeSessionId, bootstrapLoaded, items, newSessionDefaults, recentCwds, tmuxAvailable } = useSessionsStore();
  const sessionsStoreApi = useSessionsStoreApi();
  const [cwd, setCwd] = useState("");
  const [backend, setBackend] = useState("pi");
  const [sessionName, setSessionName] = useState("");
  const [surfaceTab, setSurfaceTab] = useState<NewSessionSurfaceTab>("launch");
  const [model, setModel] = useState("");
  const [providerChoice, setProviderChoice] = useState("");
  const [reasoningEffort, setReasoningEffort] = useState("");
  const [createInTmux, setCreateInTmux] = useState(false);
  const [fastMode, setFastMode] = useState(false);
  const [resumeSessionId, setResumeSessionId] = useState("");
  const [resumeCandidates, setResumeCandidates] = useState<SessionResumeCandidate[]>([]);
  const [resumeOffset, setResumeOffset] = useState(0);
  const [resumeRemaining, setResumeRemaining] = useState(0);
  const [resumeLoading, setResumeLoading] = useState(false);
  const [useWorktree, setUseWorktree] = useState(false);
  const [worktreeBranch, setWorktreeBranch] = useState("");
  const [cwdInfo, setCwdInfo] = useState<SessionCwdInfo>({
    exists: false,
    willCreate: false,
    gitRepo: false,
    gitRoot: "",
    gitBranch: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [lookupError, setLookupError] = useState("");
  const wasOpenRef = useRef(false);
  const hydratedDefaultsRef = useRef(false);
  const touchedLaunchSettingsRef = useRef<Record<LaunchSettingField, boolean>>({
    backend: false,
    model: false,
    providerChoice: false,
    reasoningEffort: false,
    createInTmux: false,
    fastMode: false,
  });
  const submittingRef = useRef(false);

  const backendNames = useMemo(
    () => Object.keys(newSessionDefaults?.backends || { codex: {}, pi: {} }),
    [newSessionDefaults?.backends],
  );
  const backendDefaults = newSessionDefaults?.backends?.[backend] || {};
  const providerChoices = useMemo(() => providerChoicesForDefaults(backendDefaults), [backendDefaults]);
  const reasoningChoices = useMemo(() => reasoningChoicesForDefaults(backendDefaults), [backendDefaults]);
  const modelChoices = useMemo(() => modelChoicesForDefaults(backendDefaults, backend, providerChoice), [backendDefaults, backend, providerChoice]);
  const supportsFast = !!backendDefaults.supports_fast;
  const supportsTmux = tmuxAvailable;
  const supportsWorktree = backend === "codex";
  const activeSession = useMemo(
    () => items.find((session) => session.session_id === activeSessionId) ?? null,
    [activeSessionId, items],
  );
  const focusedSessions = useMemo(
    () => items.filter((session) => session.focused === true && session.historical !== true),
    [items],
  );
  const sessionNamePlaceholder = baseName(cwd) || "session-name";
  const dialogTitleId = "new-session-dialog-title";

  useEffect(() => {
    if (!open) {
      wasOpenRef.current = false;
      hydratedDefaultsRef.current = false;
      touchedLaunchSettingsRef.current = {
        backend: false,
        model: false,
        providerChoice: false,
        reasoningEffort: false,
        createInTmux: false,
        fastMode: false,
      };
      return;
    }
    if (wasOpenRef.current) {
      return;
    }
    wasOpenRef.current = true;
    const initialBackend = newSessionDefaults?.default_backend || "pi";
    const initialDefaults = newSessionDefaults?.backends?.[initialBackend] || {};
    const initialProviders = providerChoicesForDefaults(initialDefaults);
    const initialReasoning = reasoningChoicesForDefaults(initialDefaults);
    hydratedDefaultsRef.current = Object.keys(newSessionDefaults?.backends || {}).length > 0;
    setCwd(initialCwdForDialog(activeSession?.cwd, recentCwds));
    setBackend(initialBackend);
    setSessionName("");
    setSurfaceTab("launch");
    setModel(initialDefaults.model?.trim() || "");
    setProviderChoice(initialDefaults.provider_choice?.trim() || initialProviders[0] || "");
    setReasoningEffort(initialDefaults.reasoning_effort?.trim() || initialReasoning[0] || "high");
    setCreateInTmux(Boolean(tmuxAvailable));
    setFastMode(String(initialDefaults.service_tier || "").trim().toLowerCase() === "fast");
    setResumeSessionId("");
    setResumeCandidates([]);
    setResumeOffset(0);
    setResumeRemaining(0);
    setResumeLoading(false);
    setUseWorktree(false);
    setWorktreeBranch("");
    setCwdInfo({ exists: false, willCreate: false, gitRepo: false, gitRoot: "", gitBranch: "" });
    setSubmitting(false);
    setError("");
    setLookupError("");
  }, [activeSession?.cwd, newSessionDefaults, open, recentCwds, tmuxAvailable]);

  useEffect(() => {
    if (!open || hydratedDefaultsRef.current) {
      return;
    }

    const backendNames = Object.keys(newSessionDefaults?.backends || {});
    if (!backendNames.length) {
      return;
    }

    hydratedDefaultsRef.current = true;
    const defaultBackend = newSessionDefaults?.default_backend || backendNames[0] || "pi";
    const selectedBackend = touchedLaunchSettingsRef.current.backend && backend ? backend : defaultBackend;
    const defaultValues = newSessionDefaults?.backends?.[selectedBackend] || {};
    const providerDefaults = providerChoicesForDefaults(defaultValues);
    const reasoningDefaults = reasoningChoicesForDefaults(defaultValues);

    if (!touchedLaunchSettingsRef.current.backend) {
      setBackend(selectedBackend);
    }
    if (!touchedLaunchSettingsRef.current.model) {
      setModel(defaultValues.model?.trim() || "");
    }
    if (!touchedLaunchSettingsRef.current.providerChoice) {
      setProviderChoice(defaultValues.provider_choice?.trim() || providerDefaults[0] || "");
    }
    if (!touchedLaunchSettingsRef.current.reasoningEffort) {
      setReasoningEffort(defaultValues.reasoning_effort?.trim() || reasoningDefaults[0] || "high");
    }
    if (!touchedLaunchSettingsRef.current.createInTmux) {
      setCreateInTmux(Boolean(tmuxAvailable));
    }
    if (!touchedLaunchSettingsRef.current.fastMode) {
      setFastMode(String(defaultValues.service_tier || "").trim().toLowerCase() === "fast");
    }
  }, [backend, open, newSessionDefaults, tmuxAvailable]);

  useEffect(() => {
    if (!supportsFast && fastMode) {
      setFastMode(false);
    }
  }, [fastMode, supportsFast]);

  useEffect(() => {
    if (!supportsTmux && createInTmux) {
      setCreateInTmux(false);
    }
  }, [createInTmux, supportsTmux]);

  useEffect(() => {
    if (!supportsWorktree) {
      setUseWorktree(false);
      setWorktreeBranch("");
    }
  }, [supportsWorktree]);

  useEffect(() => {
    if (!open || bootstrapLoaded) {
      return;
    }
    sessionsStoreApi.refreshBootstrap().catch(() => undefined);
  }, [bootstrapLoaded, open, sessionsStoreApi]);

  useEffect(() => {
    if (!open) return;
    setResumeOffset(0);
  }, [backend, cwd, open]);

  useEffect(() => {
    if (!open) return;
    const rawCwd = cwd.trim();
    if (!rawCwd) {
      setResumeCandidates([]);
      setResumeSessionId("");
      setResumeRemaining(0);
      setResumeLoading(false);
      setLookupError("");
      setCwdInfo({ exists: false, willCreate: false, gitRepo: false, gitRoot: "", gitBranch: "" });
      return;
    }

    let cancelled = false;
    setResumeLoading(true);
    const timeoutId = window.setTimeout(async () => {
      try {
        const result: SessionResumeCandidatesResponse = await api.getSessionResumeCandidates(rawCwd, backend, { offset: resumeOffset, limit: RESUME_PAGE_SIZE });
        if (cancelled) return;
        setResumeCandidates(Array.isArray(result.sessions) ? result.sessions : []);
        setResumeRemaining(Math.max(0, Number(result.remaining || 0)));
        setResumeSessionId((current) => {
          if (!current) return "";
          return (result.sessions || []).some((item) => item.session_id === current) ? current : "";
        });
        setCwdInfo({
          exists: !!result.exists,
          willCreate: !!result.will_create,
          gitRepo: !!result.git_repo,
          gitRoot: result.git_root || "",
          gitBranch: result.git_branch || "",
        });
        setLookupError("");
      } catch (loadError) {
        if (cancelled) return;
        setResumeCandidates([]);
        setResumeSessionId("");
        setResumeRemaining(0);
        setCwdInfo({ exists: false, willCreate: false, gitRepo: false, gitRoot: "", gitBranch: "" });
        setLookupError(loadError instanceof Error ? loadError.message : "Failed to inspect working directory");
      } finally {
        if (!cancelled) {
          setResumeLoading(false);
        }
      }
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [backend, cwd, open, resumeOffset]);

  if (!open) return null;

  const cwdHint = cwd.trim()
    ? resumeLoading
      ? "Inspecting directory and looking for resumable sessions..."
      : cwdInfo.gitRepo
        ? `Git repo${cwdInfo.gitBranch ? ` · ${cwdInfo.gitBranch}` : ""}${cwdInfo.gitRoot ? ` · ${cwdInfo.gitRoot}` : ""}`
        : cwdInfo.exists
          ? "Directory exists and will start fresh unless you choose a previous session."
          : cwdInfo.willCreate
            ? "Directory does not exist yet. The backend will create it on launch."
            : ""
    : "";

  const markLaunchSettingTouched = (field: LaunchSettingField) => {
    touchedLaunchSettingsRef.current[field] = true;
  };

  const applyBackend = (nextBackend: string) => {
    touchedLaunchSettingsRef.current.backend = true;
    const nextDefaults = newSessionDefaults?.backends?.[nextBackend] || {};
    const nextProviders = providerChoicesForDefaults(nextDefaults);
    const nextReasoning = reasoningChoicesForDefaults(nextDefaults);
    const hasLaunchDefaults = Boolean(
      nextDefaults.model
      || nextDefaults.provider_choice
      || nextDefaults.reasoning_effort
      || nextDefaults.service_tier
      || nextProviders.length
      || nextReasoning.length,
    );

    if (hasLaunchDefaults) {
      touchedLaunchSettingsRef.current.model = false;
      touchedLaunchSettingsRef.current.providerChoice = false;
      touchedLaunchSettingsRef.current.reasoningEffort = false;
      touchedLaunchSettingsRef.current.createInTmux = false;
      touchedLaunchSettingsRef.current.fastMode = false;
    }
    setBackend(nextBackend);
    setProviderChoice(nextDefaults.provider_choice?.trim() || nextProviders[0] || "");
    setModel(nextDefaults.model?.trim() || "");
    setReasoningEffort(nextDefaults.reasoning_effort?.trim() || nextReasoning[0] || "high");
    setFastMode(String(nextDefaults.service_tier || "").trim().toLowerCase() === "fast");
    setCreateInTmux(Boolean(tmuxAvailable));
    setResumeSessionId("");
    setResumeOffset(0);
    setResumeRemaining(0);
    setUseWorktree(false);
    setWorktreeBranch("");
    setError("");
  };

  const applyProviderChoice = (nextProviderChoice: string) => {
    markLaunchSettingTouched("providerChoice");
    setProviderChoice(nextProviderChoice);
    if (backend !== "pi" || touchedLaunchSettingsRef.current.model) {
      return;
    }
    setModel(defaultPiModelForProvider(backendDefaults, nextProviderChoice));
  };

  return (
    <Dialog open={open}>
      <div data-testid="new-session-dialog" className="w-full max-w-3xl">
        <DialogContent titleId={dialogTitleId} className="newSessionDialog max-h-[88dvh] overflow-hidden border-border/70 bg-card/95 p-0 shadow-2xl shadow-primary/10">
          <DialogHeader className="space-y-4 p-6 pb-5">
            <div className="newSessionHeaderLead">
              <div className="space-y-3">
                <div className="space-y-1">
                  <DialogTitle id={dialogTitleId}>New session</DialogTitle>
                  <p className="text-sm text-muted-foreground">Launch a backend in a project directory.</p>
                </div>
                <div className="newSessionMeta flex flex-wrap gap-2">
                  <Badge variant="secondary" className="capitalize">{backend}</Badge>
                  {supportsFast ? <Badge variant="outline">Fast available</Badge> : null}
                  {supportsTmux ? <Badge variant="outline">tmux ready</Badge> : null}
                  {supportsWorktree ? <Badge variant="outline">worktree support</Badge> : null}
                  {focusedSessions.length ? <Badge variant="outline">{focusedSessions.length} Focus</Badge> : null}
                </div>
              </div>
              <div className="agentBackendTabs grid min-w-[14rem] grid-cols-2 gap-2 rounded-2xl bg-muted/60 p-1">
                {backendNames.map((backendName) => (
                  <Button
                    key={backendName}
                    type="button"
                    variant={backend === backendName ? "default" : "ghost"}
                    data-testid={`backend-tab-${backendName}`}
                    className="backendOptionButton h-11 rounded-[1rem] capitalize"
                    onClick={() => applyBackend(backendName)}
                  >
                    {backendName}
                  </Button>
                ))}
              </div>
            </div>
            <div className="newSessionSurfaceTabs grid w-full max-w-sm grid-cols-2 gap-2 rounded-2xl bg-muted/60 p-1">
              <Button
                type="button"
                variant={surfaceTab === "launch" ? "default" : "ghost"}
                className="h-10 rounded-[1rem]"
                onClick={() => setSurfaceTab("launch")}
              >
                Launch
              </Button>
              <Button
                type="button"
                variant={surfaceTab === "focus" ? "default" : "ghost"}
                className="h-10 rounded-[1rem]"
                onClick={() => setSurfaceTab("focus")}
              >
                Focus
              </Button>
            </div>
          </DialogHeader>

          <Separator className="bg-border/70" />

          <form
            className="newSessionForm flex max-h-[calc(88dvh-8.5rem)] flex-col"
            onSubmit={async (event) => {
              event.preventDefault();
              const trimmedCwd = cwd.trim();
              const trimmedWorktreeBranch = worktreeBranch.trim();
              if (!trimmedCwd) {
                setError("Working directory is required.");
                return;
              }
              if (supportsWorktree && useWorktree && !resumeSessionId && !trimmedWorktreeBranch) {
                setError("Branch name is required.");
                return;
              }

              if (submittingRef.current) {
                return;
              }

              submittingRef.current = true;
              setSubmitting(true);
              setError("");
              try {
                const providerSettings = providerChoiceToSettings(providerChoice || backendDefaults.provider_choice || "", backend);
                const response = await api.createSession({
                  cwd: trimmedCwd,
                  backend,
                  resume_session_id: resumeSessionId || undefined,
                  worktree_branch: supportsWorktree && useWorktree && !resumeSessionId ? trimmedWorktreeBranch : undefined,
                  create_in_tmux: supportsTmux ? createInTmux : undefined,
                  model: model.trim() || undefined,
                  model_provider: providerSettings.model_provider,
                  preferred_auth_method: providerSettings.preferred_auth_method,
                  reasoning_effort: reasoningEffort.trim() || undefined,
                  service_tier: supportsFast && fastMode ? "fast" : undefined,
                });
                await sessionsStoreApi.refresh();
                const returnedSessionId = String(response.session_id || "").trim();
                const renamedSessionId = returnedSessionId || "";
                let createdSessionId = renamedSessionId;
                if (!createdSessionId) {
                  const brokerPid = typeof response.broker_pid === "number" ? response.broker_pid : null;
                  const matched = brokerPid === null
                    ? undefined
                    : sessionsStoreApi.getState().items.find((session) => session.broker_pid === brokerPid);
                  createdSessionId = matched?.session_id || "";
                }
                await sessionsStoreApi.refreshBootstrap();
                if (sessionName.trim() && createdSessionId) {
                  try {
                    await api.renameSession(createdSessionId, sessionName.trim());
                    await sessionsStoreApi.refresh();
                  } catch (renameError) {
                    // Launch succeeded; keep the new session selected even if post-create rename fails.
                    console.warn("Failed to rename new session", renameError);
                  }
                }
                if (createdSessionId) {
                  sessionsStoreApi.select(createdSessionId);
                }
                onClose();
              } catch (submitError) {
                setError(submitError instanceof Error ? submitError.message : "Failed to create session");
              } finally {
                submittingRef.current = false;
                setSubmitting(false);
              }
            }}
          >
            <div className="newSessionFormBody space-y-5 overflow-y-auto px-6 py-5">
              {surfaceTab === "focus" ? (
                <section className="dialogSection space-y-4">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">Focus</h3>
                    <p className="mt-1 text-sm text-muted-foreground">Jump straight to the sessions you manually shortlisted.</p>
                  </div>
                  {focusedSessions.length ? (
                    <div className="focusSessionList">
                      {focusedSessions.map((session) => (
                        <button
                          key={session.session_id}
                          type="button"
                          className="focusSessionItem"
                          onClick={() => {
                            sessionsStoreApi.select(session.session_id);
                            onClose();
                          }}
                        >
                          <span className="focusSessionTitle">{getSessionDisplayName(session)}</span>
                          <span className="focusSessionMeta">
                            {session.agent_backend || "codex"}
                            {session.cwd?.trim() ? ` · ${session.cwd.trim()}` : ""}
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="focusSessionEmpty">No live sessions are in Focus yet. Use the star button in the left rail first.</div>
                  )}
                </section>
              ) : (
                <>
              <section className="dialogSection space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">Project target</h3>
                    <p className="mt-1 text-sm text-muted-foreground">Pick the working directory, then choose whether to resume or branch off.</p>
                  </div>
                  {resumeCandidates.length ? <Badge variant="outline">{resumeCandidates.length} resumable</Badge> : null}
                </div>
                <label className="fieldBlock space-y-2">
                  <span className="fieldLabel">Working directory</span>
                  <Input
                    name="cwd"
                    value={cwd}
                    onInput={(event) => setCwd(event.currentTarget.value)}
                    onChange={(event) => setCwd(event.currentTarget.value)}
                    placeholder="/path/to/project"
                    list="new-session-recent-cwds"
                  />
                </label>
                {recentCwds.length ? (
                  <datalist id="new-session-recent-cwds">
                    {recentCwds.map((recentCwd) => (
                      <option key={recentCwd} value={recentCwd} />
                    ))}
                  </datalist>
                ) : null}
                {cwdHint ? <p className="fieldHint text-sm text-muted-foreground">{cwdHint}</p> : null}
                {lookupError ? <p className="errorText text-sm font-medium">{lookupError}</p> : null}
                <div className="fieldGrid twoCol gap-3">
                  <label className="fieldBlock space-y-2">
                    <span className="fieldLabel">Session name</span>
                    <Input
                      name="sessionName"
                      value={sessionName}
                      onInput={(event) => setSessionName(event.currentTarget.value)}
                      onChange={(event) => setSessionName(event.currentTarget.value)}
                      placeholder={sessionNamePlaceholder}
                    />
                  </label>
                  <label className="fieldBlock space-y-2">
                    <span className="fieldLabel">Resume conversation</span>
                    <SelectField
                      name="resumeSessionId"
                      value={resumeSessionId}
                      onInput={(event) => setResumeSessionId(event.currentTarget.value)}
                      onChange={(event) => setResumeSessionId(event.currentTarget.value)}
                    >
                      <option value="">Start fresh</option>
                      {resumeCandidates.map((item) => (
                        <option key={item.session_id} value={item.session_id}>
                          {resumeOptionLabel(item)}
                        </option>
                      ))}
                    </SelectField>
                    {resumeCandidates.length || resumeOffset > 0 || resumeRemaining > 0 ? (
                      <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                        <span>
                          {resumeCandidates.length
                            ? `Showing ${resumeOffset + 1}-${resumeOffset + resumeCandidates.length}`
                            : `Showing ${resumeOffset + 1}-${resumeOffset}`}
                          {resumeRemaining > 0 ? `, ${resumeRemaining} older` : ""}
                        </span>
                        <div className="flex items-center gap-2">
                          <Button
                            type="button"
                            variant="ghost"
                            className="h-8 px-2"
                            disabled={resumeLoading || resumeOffset <= 0}
                            onClick={() => setResumeOffset((current) => Math.max(0, current - RESUME_PAGE_SIZE))}
                          >
                            Newer
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            className="h-8 px-2"
                            disabled={resumeLoading || resumeRemaining <= 0}
                            onClick={() => setResumeOffset((current) => current + RESUME_PAGE_SIZE)}
                          >
                            Older
                          </Button>
                        </div>
                      </div>
                    ) : null}
                  </label>
                </div>
              </section>

              <Separator className="bg-border/70" />

              <section className="dialogSection space-y-3">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">Model settings</h3>
                  <p className="mt-1 text-sm text-muted-foreground">Tune provider, model, and reasoning without leaving the launch flow.</p>
                </div>
                <div className="fieldGrid threeCol gap-3">
                  <label className="fieldBlock space-y-2">
                    <span className="fieldLabel">Model</span>
                    <Input
                      name="model"
                      value={model}
                      onInput={(event) => {
                        markLaunchSettingTouched("model");
                        setModel(event.currentTarget.value);
                      }}
                      onChange={(event) => {
                        markLaunchSettingTouched("model");
                        setModel(event.currentTarget.value);
                      }}
                      placeholder="default"
                      list="new-session-models"
                    />
                  </label>
                  {modelChoices.length ? (
                    <datalist id="new-session-models">
                      {modelChoices.map((modelOption) => (
                        <option key={modelOption} value={modelOption} />
                      ))}
                    </datalist>
                  ) : null}
                  <label className="fieldBlock space-y-2">
                    <span className="fieldLabel">Reasoning effort</span>
                    <SelectField
                      name="reasoningEffort"
                      value={reasoningEffort}
                      onInput={(event) => {
                        markLaunchSettingTouched("reasoningEffort");
                        setReasoningEffort(event.currentTarget.value);
                      }}
                      onChange={(event) => {
                        markLaunchSettingTouched("reasoningEffort");
                        setReasoningEffort(event.currentTarget.value);
                      }}
                    >
                      {reasoningChoices.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </SelectField>
                  </label>
                  <div className="fieldBlock space-y-2">
                    <span className="fieldLabel">Speed</span>
                    <ToggleField
                      label="Fast"
                      name="fastMode"
                      checked={fastMode}
                      disabled={!supportsFast}
                      description={supportsFast ? "Use the backend's faster service tier when available." : "This backend does not expose a fast tier."}
                      onChange={(checked) => {
                        markLaunchSettingTouched("fastMode");
                        setFastMode(checked);
                      }}
                    />
                  </div>
                </div>
                <div className="fieldGrid twoCol gap-3">
                  <label className="fieldBlock space-y-2">
                    <span className="fieldLabel">Provider</span>
                    <SelectField
                      name="providerChoice"
                      value={providerChoice}
                      onInput={(event) => {
                        applyProviderChoice(event.currentTarget.value);
                      }}
                      onChange={(event) => {
                        applyProviderChoice(event.currentTarget.value);
                      }}
                    >
                      {providerChoices.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </SelectField>
                  </label>
                  <div className="fieldBlock space-y-2">
                    <span className="fieldLabel">Launch mode</span>
                    <ToggleField
                      label={supportsTmux ? "Create in tmux" : "tmux unavailable"}
                      name="createInTmux"
                      checked={createInTmux}
                      disabled={!supportsTmux}
                      description={supportsTmux ? (backend === "pi" ? "Host the new Pi session in tmux while pi-rpc handles web control." : "Keep the new session attached to a tmux pane.") : "tmux is unavailable on this host."}
                      onChange={(checked) => {
                        markLaunchSettingTouched("createInTmux");
                        setCreateInTmux(checked);
                      }}
                    />
                  </div>
                </div>
              </section>

              {supportsWorktree ? (
                <>
                  <Separator className="bg-border/70" />
                  <section className="dialogSection space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h3 className="text-sm font-semibold text-foreground">Git worktree branch</h3>
                        <p className="mt-1 text-sm text-muted-foreground">Spin up an isolated branch when you want clean diffs for a new task.</p>
                      </div>
                      <Badge variant="outline">Codex only</Badge>
                    </div>
                    <ToggleField
                      label={resumeSessionId ? "Unavailable while resuming a session" : "Create a new worktree for this session"}
                      name="useWorktree"
                      checked={useWorktree}
                      disabled={!!resumeSessionId}
                      description={resumeSessionId ? "Resume uses the existing working tree." : "A new worktree keeps the session isolated from your current checkout."}
                      onChange={setUseWorktree}
                    />
                    <Input
                      name="worktreeBranch"
                      value={worktreeBranch}
                      onInput={(event) => setWorktreeBranch(event.currentTarget.value)}
                      onChange={(event) => setWorktreeBranch(event.currentTarget.value)}
                      placeholder="feature/my-branch"
                      disabled={!useWorktree || !!resumeSessionId}
                    />
                  </section>
                </>
              ) : null}

                </>
              )}
              {error ? <p className="errorText text-sm font-medium">{error}</p> : null}
            </div>

            <Separator className="bg-border/70" />

            <div className="newSessionFooter flex items-center justify-end gap-3 px-6 py-4">
              <Button type="button" variant="outline" onClick={onClose} disabled={submitting}>
                Cancel
              </Button>
              {surfaceTab === "launch" ? (
                <Button type="submit" disabled={submitting || !cwd.trim()}>
                  {submitting ? "Launching..." : "Start session"}
                </Button>
              ) : null}
            </div>
          </form>
        </DialogContent>
      </div>
    </Dialog>
  );
}
