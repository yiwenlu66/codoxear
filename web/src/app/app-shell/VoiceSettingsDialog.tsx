import { Button } from "@/components/ui/button";

interface VoiceSettingsDialogProps {
  audioMeta: {
    enabledDevices: number;
    lastError: string;
    listeners: number;
    queue: number;
    segments: number;
    totalDevices: number;
  };
  enterToSendDraft: boolean;
  narrationEnabledDraft: boolean;
  open: boolean;
  replySoundEnabled: boolean;
  status: string;
  voiceApiKeyDraft: string;
  voiceBaseUrlDraft: string;
  onChangeEnterToSend(value: boolean): void;
  onChangeNarrationEnabled(value: boolean): void;
  onChangeReplySoundEnabled(value: boolean): void;
  onChangeVoiceApiKey(value: string): void;
  onChangeVoiceBaseUrl(value: string): void;
  onClose(): void;
  onSave(): void;
  onTriggerTestPush(): void;
}

export function VoiceSettingsDialog({
  audioMeta,
  enterToSendDraft,
  narrationEnabledDraft,
  open,
  replySoundEnabled,
  status,
  voiceApiKeyDraft,
  voiceBaseUrlDraft,
  onChangeEnterToSend,
  onChangeNarrationEnabled,
  onChangeReplySoundEnabled,
  onChangeVoiceApiKey,
  onChangeVoiceBaseUrl,
  onClose,
  onSave,
  onTriggerTestPush,
}: VoiceSettingsDialogProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="dialogBackdrop" onClick={onClose}>
      <section className="dialogCard legacyDialog voiceSettingsDialog" onClick={(event) => event.stopPropagation()}>
        <header className="dialogHeader">
          <h2>Settings</h2>
          <p>Configure announcements and notification delivery.</p>
        </header>
        <div className="newSessionForm">
          {status ? <p className="fieldHint voiceSettingsStatus">{status}</p> : null}
          <label className="fieldBlock">
            <span className="fieldLabel">OpenAI-compatible API base URL</span>
            <input
              value={voiceBaseUrlDraft}
              onInput={(event) => onChangeVoiceBaseUrl(event.currentTarget.value)}
              onChange={(event) => onChangeVoiceBaseUrl(event.currentTarget.value)}
              placeholder="https://api.openai.com/v1"
            />
          </label>
          <label className="fieldBlock">
            <span className="fieldLabel">OpenAI-compatible API key</span>
            <input
              value={voiceApiKeyDraft}
              onInput={(event) => onChangeVoiceApiKey(event.currentTarget.value)}
              onChange={(event) => onChangeVoiceApiKey(event.currentTarget.value)}
              placeholder="sk-..."
              type="password"
            />
          </label>
          <div className="fieldBlock toggleField">
            <span className="fieldLabel">Announcements</span>
            <label className="checkField">
              <input
                type="checkbox"
                checked={narrationEnabledDraft}
                onChange={(event) => onChangeNarrationEnabled(event.currentTarget.checked)}
              />
              <span>Announce narration messages</span>
            </label>
          </div>
          <div className="fieldBlock toggleField">
            <span className="fieldLabel">Composer</span>
            <label className="checkField">
              <input
                type="checkbox"
                checked={enterToSendDraft}
                onChange={(event) => onChangeEnterToSend(event.currentTarget.checked)}
              />
              <span>Press Enter to send</span>
            </label>
          </div>
          <div className="fieldBlock toggleField">
            <span className="fieldLabel">Reply sound</span>
            <label className="checkField">
              <input
                type="checkbox"
                checked={replySoundEnabled}
                onChange={(event) => onChangeReplySoundEnabled(event.currentTarget.checked)}
              />
              <span>Play a short beep when the assistant finishes a reply</span>
            </label>
          </div>
          <div className="voiceSettingsMeta fieldHint">
            <span>Listeners: {audioMeta.listeners}</span>
            <span>Queue: {audioMeta.queue}</span>
            <span>Segments: {audioMeta.segments}</span>
            <span>Mobile notifications: {audioMeta.enabledDevices}/{audioMeta.totalDevices}</span>
          </div>
          {audioMeta.lastError ? (
            <p className="fieldHint voiceSettingsStatus">Audio error: {audioMeta.lastError}</p>
          ) : null}
          <div className="formActions dialogFormActions">
            <Button type="button" variant="outline" onClick={onTriggerTestPush}>Test Push</Button>
            <div className="flex-1" />
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
            <Button type="button" onClick={onSave}>Save</Button>
          </div>
        </div>
      </section>
    </div>
  );
}
