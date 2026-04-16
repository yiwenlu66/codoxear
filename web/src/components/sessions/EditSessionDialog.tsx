import { useMemo, useState } from "preact/hooks";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";

import { api } from "../../lib/api";
import type { SessionSummary } from "../../lib/types";

interface EditSessionDialogProps {
  open: boolean;
  session: SessionSummary | null;
  sessions: SessionSummary[];
  onClose: () => void;
  onSaved: () => Promise<void> | void;
}

type SnoozeMode = "none" | "4h" | "tomorrow" | "custom";

function shortSessionId(sessionId: string) {
  const match = sessionId.match(/^([0-9a-f]{8})[0-9a-f-]{20,}$/i);
  return match ? match[1] : sessionId.slice(0, 8);
}

function sessionLabel(session: SessionSummary) {
  return session.alias || session.first_user_message || session.title || shortSessionId(session.session_id);
}

function formatPriorityOffset(value: number) {
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(2)}`;
}

function tomorrowSnoozeSeconds() {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  date.setHours(9, 0, 0, 0);
  return Math.floor(date.getTime() / 1000);
}

function fillCustomSnoozeFields(tsSeconds: number) {
  const ts = Number(tsSeconds);
  const date = Number.isFinite(ts) && ts > 0 ? new Date(ts * 1000) : new Date(Date.now() + 24 * 3600 * 1000);
  const yyyy = String(date.getFullYear()).padStart(4, "0");
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const mi = String(date.getMinutes()).padStart(2, "0");
  return {
    customDate: `${yyyy}-${mm}-${dd}`,
    customTime: `${hh}:${mi}`,
  };
}

function initialFormState(session: SessionSummary | null) {
  const snoozeUntil = Number(session?.snooze_until || 0);
  const nextCustom = fillCustomSnoozeFields(snoozeUntil > Date.now() / 1000 ? snoozeUntil : tomorrowSnoozeSeconds());
  return {
    sessionName: String(session?.alias || ""),
    priorityOffset: Number(session?.priority_offset || 0),
    snoozeMode: (snoozeUntil > Date.now() / 1000 ? "custom" : "none") as SnoozeMode,
    customDate: nextCustom.customDate,
    customTime: nextCustom.customTime,
    dependencySessionId: String(session?.dependency_session_id || ""),
  };
}

export function EditSessionDialog({ open, session, sessions, onClose, onSaved }: EditSessionDialogProps) {
  const [sessionName, setSessionName] = useState(() => initialFormState(session).sessionName);
  const [priorityOffset, setPriorityOffset] = useState(() => initialFormState(session).priorityOffset);
  const [snoozeMode, setSnoozeMode] = useState<SnoozeMode>(() => initialFormState(session).snoozeMode);
  const [customDate, setCustomDate] = useState(() => initialFormState(session).customDate);
  const [customTime, setCustomTime] = useState(() => initialFormState(session).customTime);
  const [dependencySessionId, setDependencySessionId] = useState(() => initialFormState(session).dependencySessionId);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const dependencyOptions = useMemo(
    () => sessions.filter((item) => item.session_id !== session?.session_id),
    [session?.session_id, sessions],
  );

  if (!open || !session) {
    return null;
  }

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => {
      if (!nextOpen && !saving) {
        onClose();
      }
    }}>
      <DialogContent titleId="edit-session-dialog-title" className="max-w-2xl border-border/70 bg-card/95 p-0 shadow-2xl shadow-primary/10">
        <DialogHeader className="space-y-3 p-6 pb-5">
          <div className="space-y-1">
            <DialogTitle id="edit-session-dialog-title">Edit conversation</DialogTitle>
            <p className="text-sm text-muted-foreground">Restore the legacy session metadata controls from the old sidebar card.</p>
          </div>
        </DialogHeader>

        <Separator className="bg-border/70" />

        <div className="space-y-5 px-6 py-5">
          <label className="block space-y-2">
            <span className="text-sm font-medium text-foreground">Conversation name</span>
            <Input
              name="sessionName"
              value={sessionName}
              maxLength={80}
              placeholder={sessionLabel(session)}
              onInput={(event) => setSessionName(event.currentTarget.value)}
              onChange={(event) => setSessionName(event.currentTarget.value)}
            />
          </label>

          <label className="block space-y-2">
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-medium text-foreground">Priority offset</span>
              <span className="text-sm font-medium text-muted-foreground">{formatPriorityOffset(priorityOffset)}</span>
            </div>
            <div className="flex items-center gap-3">
              <input
                name="priorityOffset"
                type="range"
                min="-1"
                max="1"
                step="0.05"
                value={String(priorityOffset)}
                className="w-full"
                onInput={(event) => setPriorityOffset(Number(event.currentTarget.value || 0))}
                onChange={(event) => setPriorityOffset(Number(event.currentTarget.value || 0))}
              />
              <Button type="button" variant="ghost" size="sm" onClick={() => setPriorityOffset(0)}>Reset</Button>
            </div>
          </label>

          <label className="block space-y-2">
            <span className="text-sm font-medium text-foreground">Snooze</span>
            <select
              name="snoozeMode"
              value={snoozeMode}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              onChange={(event) => {
                const nextMode = event.currentTarget.value as SnoozeMode;
                setSnoozeMode(nextMode);
                if (nextMode === "tomorrow") {
                  const next = fillCustomSnoozeFields(tomorrowSnoozeSeconds());
                  setCustomDate(next.customDate);
                  setCustomTime(next.customTime);
                }
                if (nextMode === "4h") {
                  const next = fillCustomSnoozeFields(Math.floor(Date.now() / 1000) + 4 * 3600);
                  setCustomDate(next.customDate);
                  setCustomTime(next.customTime);
                }
              }}
            >
              <option value="none">No snooze</option>
              <option value="4h">4 hours</option>
              <option value="tomorrow">Tomorrow 09:00</option>
              <option value="custom">Custom</option>
            </select>
            {snoozeMode === "custom" ? (
              <div className="grid gap-3 sm:grid-cols-2">
                <Input name="customSnoozeDate" type="date" value={customDate} onInput={(event) => setCustomDate(event.currentTarget.value)} onChange={(event) => setCustomDate(event.currentTarget.value)} />
                <Input name="customSnoozeTime" type="time" step={60} value={customTime} onInput={(event) => setCustomTime(event.currentTarget.value)} onChange={(event) => setCustomTime(event.currentTarget.value)} />
              </div>
            ) : null}
          </label>

          <label className="block space-y-2">
            <span className="text-sm font-medium text-foreground">Depends on</span>
            <select
              name="dependencySessionId"
              value={dependencySessionId}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              onChange={(event) => setDependencySessionId(event.currentTarget.value)}
            >
              <option value="">No dependency</option>
              {dependencyOptions.map((item) => (
                <option key={item.session_id} value={item.session_id}>{sessionLabel(item)}</option>
              ))}
            </select>
          </label>

          {error ? <p className="text-sm font-medium text-red-600">{error}</p> : null}
        </div>

        <Separator className="bg-border/70" />

        <div className="flex items-center justify-end gap-3 px-6 py-4">
          <Button type="button" variant="ghost" onClick={() => !saving && onClose()} disabled={saving}>Cancel</Button>
          <Button
            type="button"
            disabled={saving}
            onClick={async () => {
              let snoozeUntil: number | null = null;
              if (snoozeMode === "4h") {
                snoozeUntil = Math.floor(Date.now() / 1000) + 4 * 3600;
              } else if (snoozeMode === "tomorrow") {
                snoozeUntil = tomorrowSnoozeSeconds();
              } else if (snoozeMode === "custom") {
                if (!customDate || !customTime) {
                  setError("Choose both a custom date and time.");
                  return;
                }
                const parsed = Date.parse(`${customDate}T${customTime}`);
                if (!Number.isFinite(parsed)) {
                  setError("Invalid snooze time.");
                  return;
                }
                snoozeUntil = Math.floor(parsed / 1000);
              }

              setSaving(true);
              setError("");
              try {
                await api.editSession(session.session_id, {
                  name: sessionName,
                  priority_offset: Number(priorityOffset.toFixed(2)),
                  snooze_until: snoozeUntil,
                  dependency_session_id: dependencySessionId || null,
                });
                await onSaved();
                onClose();
              } catch (saveError) {
                setError(saveError instanceof Error ? saveError.message : "Failed to update session");
              } finally {
                setSaving(false);
              }
            }}
          >
            {saving ? "Saving..." : "Save changes"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
