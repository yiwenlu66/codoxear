import { useEffect, useState } from "preact/hooks";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

import { api } from "../../lib/api";

interface HarnessDialogProps {
  open: boolean;
  sessionId: string | null;
  onClose: () => void;
}

export function HarnessDialog({ open, sessionId, onClose }: HarnessDialogProps) {
  const [enabled, setEnabled] = useState(false);
  const [request, setRequest] = useState("");
  const [cooldownMinutes, setCooldownMinutes] = useState("30");
  const [remainingInjections, setRemainingInjections] = useState("3");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");

  useEffect(() => {
    if (!open || !sessionId) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    setStatus("");
    api.getHarness(sessionId)
      .then((response) => {
        if (cancelled) return;
        setEnabled(response.enabled === true);
        setRequest(typeof response.request === "string" ? response.request : "");
        setCooldownMinutes(String(response.cooldown_minutes ?? 30));
        setRemainingInjections(String(response.remaining_injections ?? 3));
      })
      .catch((error) => {
        if (cancelled) return;
        setStatus(error instanceof Error ? error.message : "Unable to load harness settings");
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [open, sessionId]);

  const save = async () => {
    if (!sessionId || saving) {
      return;
    }
    setSaving(true);
    setStatus("Saving…");
    try {
      await api.saveHarness(sessionId, {
        enabled,
        request,
        cooldown_minutes: Number(cooldownMinutes || 0),
        remaining_injections: Number(remainingInjections || 0),
      });
      setStatus("Saved");
      window.setTimeout(() => {
        onClose();
      }, 150);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to save harness settings");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open}>
      <DialogContent className="max-w-2xl" titleId="harness-dialog-title">
        <DialogHeader>
          <div className="flex items-start justify-between gap-3">
            <div>
              <DialogTitle id="harness-dialog-title">Harness mode</DialogTitle>
              <p className="text-sm text-muted-foreground">Configure automatic follow-up guidance for this session.</p>
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>Close</Button>
          </div>
        </DialogHeader>
        <div className="space-y-4 px-6 pb-6 pt-2">
          <label className="toggleOption flex items-start gap-3 rounded-2xl border border-border/70 bg-background/80 px-3 py-3 text-sm">
            <input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.currentTarget.checked)} />
            <div>
              <strong className="block text-foreground">Enabled</strong>
              <span className="text-muted-foreground">Allow Codoxear to inject the configured request during harness sweeps.</span>
            </div>
          </label>
          <label className="fieldBlock">
            <span className="fieldLabel">Additional request</span>
            <Textarea value={request} onInput={(event) => setRequest(event.currentTarget.value)} rows={6} placeholder="Summarize the next blocker and suggest the next concrete step." />
          </label>
          <div className="fieldGrid twoCol">
            <label className="fieldBlock">
              <span className="fieldLabel">Cooldown (minutes)</span>
              <Input type="number" min={1} value={cooldownMinutes} onInput={(event) => setCooldownMinutes(event.currentTarget.value)} />
            </label>
            <label className="fieldBlock">
              <span className="fieldLabel">Remaining injections</span>
              <Input type="number" min={0} value={remainingInjections} onInput={(event) => setRemainingInjections(event.currentTarget.value)} />
            </label>
          </div>
          {loading ? <p className="text-sm text-muted-foreground">Loading harness settings…</p> : null}
          {status ? <p className="text-sm text-muted-foreground">{status}</p> : null}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
            <Button type="button" onClick={() => void save()} disabled={saving || loading || !sessionId}>Save</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
