import { render } from "preact";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VoiceSettingsDialog } from "./VoiceSettingsDialog";

describe("VoiceSettingsDialog", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("uses the shared dialog action button styles instead of legacy button classes", () => {
    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <VoiceSettingsDialog
        audioMeta={{ enabledDevices: 0, lastError: "", listeners: 0, queue: 0, segments: 0, totalDevices: 0 }}
        enterToSendDraft={false}
        narrationEnabledDraft={false}
        open
        replySoundEnabled={false}
        status=""
        voiceApiKeyDraft=""
        voiceBaseUrlDraft=""
        onChangeEnterToSend={() => undefined}
        onChangeNarrationEnabled={() => undefined}
        onChangeReplySoundEnabled={() => undefined}
        onChangeVoiceApiKey={() => undefined}
        onChangeVoiceBaseUrl={() => undefined}
        onClose={() => undefined}
        onSave={() => undefined}
        onTriggerTestPush={() => undefined}
      />,
      root,
    );

    const testPushButton = root.querySelector('button[type="button"]');
    const buttons = Array.from(root.querySelectorAll("button"));
    const cancelButton = buttons.find((button) => button.textContent === "Cancel");
    const saveButton = buttons.find((button) => button.textContent === "Save");

    expect(testPushButton?.className).toContain("border");
    expect(testPushButton?.className).toContain("bg-background");
    expect(testPushButton?.className).not.toContain("secondaryButton");

    expect(cancelButton?.className).toContain("border");
    expect(cancelButton?.className).toContain("bg-background");
    expect(cancelButton?.className).not.toContain("secondaryButton");

    expect(saveButton?.className).toContain("bg-primary");
    expect(saveButton?.className).toContain("text-primary-foreground");
    expect(saveButton?.className).not.toContain("primaryButton");
  });

  it("wires action callbacks through the footer controls", () => {
    const onClose = vi.fn();
    const onSave = vi.fn();
    const onTriggerTestPush = vi.fn();

    root = document.createElement("div");
    document.body.appendChild(root);

    render(
      <VoiceSettingsDialog
        audioMeta={{ enabledDevices: 0, lastError: "", listeners: 0, queue: 0, segments: 0, totalDevices: 0 }}
        enterToSendDraft={false}
        narrationEnabledDraft={false}
        open
        replySoundEnabled={false}
        status=""
        voiceApiKeyDraft=""
        voiceBaseUrlDraft=""
        onChangeEnterToSend={() => undefined}
        onChangeNarrationEnabled={() => undefined}
        onChangeReplySoundEnabled={() => undefined}
        onChangeVoiceApiKey={() => undefined}
        onChangeVoiceBaseUrl={() => undefined}
        onClose={onClose}
        onSave={onSave}
        onTriggerTestPush={onTriggerTestPush}
      />,
      root,
    );

    const buttons = Array.from(root.querySelectorAll("button"));
    buttons.find((button) => button.textContent === "Test Push")?.click();
    buttons.find((button) => button.textContent === "Cancel")?.click();
    buttons.find((button) => button.textContent === "Save")?.click();

    expect(onTriggerTestPush).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledTimes(1);
  });
});
