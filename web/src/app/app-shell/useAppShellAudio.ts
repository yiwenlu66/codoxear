import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api } from "../../lib/api";
import type { VoiceSettingsResponse } from "../../lib/types";
import {
  DEFAULT_VOICE_SETTINGS,
  getAnnouncementClientId,
  mergeVoiceSettings,
  readLocalToggle,
  shouldPreferNativeHlsPlayback,
  writeLocalToggle,
} from "./utils";

interface StartPlaybackOptions {
  force?: boolean;
  resetSource?: boolean;
}

export function useAppShellAudio() {
  const [announcementEnabled, setAnnouncementEnabled] = useState(() => readLocalToggle("codoxear.announcementEnabled"));
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettingsResponse>(DEFAULT_VOICE_SETTINGS);
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(false);
  const [voiceSettingsStatus, setVoiceSettingsStatus] = useState("");
  const [voiceBaseUrlDraft, setVoiceBaseUrlDraft] = useState("");
  const [voiceApiKeyDraft, setVoiceApiKeyDraft] = useState("");
  const [narrationEnabledDraft, setNarrationEnabledDraft] = useState(false);
  const [enterToSendDraft, setEnterToSendDraft] = useState(() => readLocalToggle("codoxear.enterToSend"));
  const liveAudioRef = useRef<HTMLAudioElement | null>(null);
  const audioRetryTimerRef = useRef<number | null>(null);
  const hlsRef = useRef<any>(null);
  const announcementClientId = useMemo(() => getAnnouncementClientId(), []);

  useEffect(() => {
    let cancelled = false;
    api.getVoiceSettings().then((response) => {
      if (cancelled) return;
      const nextSettings = mergeVoiceSettings(response);
      setVoiceSettings(nextSettings);
      setVoiceBaseUrlDraft(String(nextSettings.tts_base_url || ""));
      setVoiceApiKeyDraft(String(nextSettings.tts_api_key || ""));
      setNarrationEnabledDraft(!!nextSettings.tts_enabled_for_narration);
      if (announcementEnabled) {
        queueMicrotask(() => startAnnouncementPlayback(nextSettings));
      }
    }).catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    writeLocalToggle("codoxear.announcementEnabled", announcementEnabled);
  }, [announcementEnabled]);

  const startAnnouncementPlayback = (settings: VoiceSettingsResponse, { resetSource = false, force = false }: StartPlaybackOptions = {}) => {
    const audio = liveAudioRef.current;
    if (!audio) return;
    if (!force && !announcementEnabled) return;
    const streamUrl = String(settings.audio?.stream_url || "").trim();
    const hasSegments = Number(settings.audio?.segment_count || 0) > 0;

    const browserPrefersNativeHls = shouldPreferNativeHlsPlayback();
    const nativeHls = browserPrefersNativeHls && ["application/vnd.apple.mpegurl", "audio/mpegurl"].some((kind) => {
      const result = audio.canPlayType(kind);
      return result === "probably" || result === "maybe";
    });

    const Hls = (window as any).Hls;
    const canUseHlsJs = Hls && Hls.isSupported();

    if (!streamUrl || !hasSegments || (!nativeHls && !canUseHlsJs)) {
      return;
    }

    if (nativeHls) {
      if (resetSource || audio.src !== streamUrl) {
        console.log("[Audio] Using native HLS playback", streamUrl);
        audio.src = streamUrl;
      }
    } else if (canUseHlsJs) {
      if (!hlsRef.current) {
        console.log("[Audio] Initializing hls.js");
        const hls = new Hls({
          enableWorker: true,
          lowLatencyMode: true,
          backBufferLength: 30,
          manifestLoadingMaxRetry: 10,
          levelLoadingMaxRetry: 10,
        });
        hls.on(Hls.Events.ERROR, (_event: any, data: any) => {
          console.warn("[Audio] hls.js error:", data);
          if (data.fatal) {
            switch (data.type) {
              case Hls.ErrorTypes.NETWORK_ERROR:
                hls.startLoad();
                break;
              case Hls.ErrorTypes.MEDIA_ERROR:
                hls.recoverMediaError();
                break;
              default:
                void startAnnouncementPlayback(settings, { resetSource: true });
                break;
            }
          }
        });
        hls.attachMedia(audio);
        hlsRef.current = hls;
      }
      const hls = hlsRef.current;
      if (resetSource || audio.getAttribute("data-hls-url") !== streamUrl) {
        console.log("[Audio] Loading HLS source via hls.js", streamUrl);
        audio.setAttribute("data-hls-url", streamUrl);
        hls.loadSource(streamUrl);
      }
    }

    const playPromise = audio.play();
    if (playPromise !== undefined) {
      playPromise.then(() => {
        if (resetSource) console.log("[Audio] Playback started/resumed");
      }).catch((err) => {
        console.warn("[Audio] Playback failed, will retry:", err);
        if (audioRetryTimerRef.current !== null) {
          window.clearTimeout(audioRetryTimerRef.current);
        }
        audioRetryTimerRef.current = window.setTimeout(() => {
          audioRetryTimerRef.current = null;
          startAnnouncementPlayback(settings, { resetSource: true });
        }, 1200);
      });
    }
  };

  useEffect(() => {
    let intervalId: number | undefined;
    api.setAudioListener(announcementClientId, announcementEnabled).catch(() => undefined);
    if (announcementEnabled) {
      intervalId = window.setInterval(() => {
        api.setAudioListener(announcementClientId, true).catch(() => undefined);
      }, 15000);
    }
    return () => {
      if (typeof intervalId === "number") {
        window.clearInterval(intervalId);
      }
      api.setAudioListener(announcementClientId, false).catch(() => undefined);
    };
  }, [announcementClientId, announcementEnabled]);

  useEffect(() => {
    const audio = liveAudioRef.current;
    if (!audio) return;
    if (!announcementEnabled) {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
      audio.removeAttribute("src");
      audio.removeAttribute("data-hls-url");
      if (audio.src) {
        audio.src = "";
      }
      if (audioRetryTimerRef.current !== null) {
        window.clearTimeout(audioRetryTimerRef.current);
        audioRetryTimerRef.current = null;
      }
      return;
    }
    startAnnouncementPlayback(voiceSettings);
  }, [announcementEnabled, voiceSettings.audio?.segment_count, voiceSettings.audio?.stream_url]);

  useEffect(() => {
    const audio = liveAudioRef.current;
    if (!audio) return;
    const retry = () => startAnnouncementPlayback(voiceSettings, { resetSource: true });
    audio.addEventListener("ended", retry);
    audio.addEventListener("error", retry);
    return () => {
      audio.removeEventListener("ended", retry);
      audio.removeEventListener("error", retry);
    };
  }, [announcementEnabled, voiceSettings]);

  const openVoiceSettings = (status = "") => {
    setVoiceSettingsStatus(status);
    setVoiceBaseUrlDraft(String(voiceSettings.tts_base_url || ""));
    setVoiceApiKeyDraft(String(voiceSettings.tts_api_key || ""));
    setNarrationEnabledDraft(!!voiceSettings.tts_enabled_for_narration);
    setEnterToSendDraft(readLocalToggle("codoxear.enterToSend"));
    setVoiceSettingsOpen(true);
  };

  const closeVoiceSettings = () => {
    setVoiceSettingsOpen(false);
    setVoiceSettingsStatus("");
  };

  const hasAnnouncementCredentials = Boolean(String(voiceSettings.tts_base_url || "").trim() && String(voiceSettings.tts_api_key || "").trim());
  const announcementLabel = announcementEnabled ? "Announcements on" : "Announcements off";

  const toggleAnnouncements = async () => {
    const next = !announcementEnabled;
    if (next && !hasAnnouncementCredentials) {
      openVoiceSettings("Set the OpenAI-compatible API base URL and API key before enabling announcements.");
      return;
    }
    setAnnouncementEnabled(next);
  };

  const saveVoiceSettings = async () => {
    setVoiceSettingsStatus("Saving...");
    try {
      const payload = {
        tts_enabled_for_narration: narrationEnabledDraft,
        tts_enabled_for_final_response: true,
        tts_base_url: voiceBaseUrlDraft.trim(),
        tts_api_key: voiceApiKeyDraft.trim(),
      };
      const response = await api.saveVoiceSettings(payload);
      const nextSettings = mergeVoiceSettings(response);
      setVoiceSettings(nextSettings);
      writeLocalToggle("codoxear.enterToSend", enterToSendDraft);
      setVoiceSettingsStatus("");
      setVoiceSettingsOpen(false);
    } catch (error) {
      setVoiceSettingsStatus(error instanceof Error ? `save error: ${error.message}` : "save error: unknown error");
    }
  };

  const playTestSound = () => {
    try {
      const AudioContext = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!AudioContext) {
        setVoiceSettingsStatus("Audio Context not supported in this browser.");
        return;
      }
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.1);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.8);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.8);
      setVoiceSettingsStatus("Playing test sound (beep)...");
      window.setTimeout(() => setVoiceSettingsStatus(""), 1500);
    } catch (error) {
      setVoiceSettingsStatus(error instanceof Error ? `test error: ${error.message}` : "test error: unknown error");
    }
  };

  const triggerTestAnnouncement = async () => {
    setVoiceSettingsStatus("Requesting test announcement...");
    try {
      await api.triggerTestAnnouncement();
      window.setTimeout(() => {
        api.getVoiceSettings()
          .then((response) => {
            const nextSettings = mergeVoiceSettings(response);
            setVoiceSettings(nextSettings);
            const lastError = String(nextSettings.audio?.last_error || "").trim();
            if (lastError) {
              setVoiceSettingsStatus(`Test announcement failed: ${lastError}`);
              return;
            }
            setVoiceSettingsStatus("Test announcement queued. If you still hear nothing, send me the Console [Audio] logs and the audio status below.");
          })
          .catch(() => {
            setVoiceSettingsStatus("Test announcement queued. If you still hear nothing, send me the Console [Audio] logs and the audio status below.");
          });
      }, 1600);
    } catch (error) {
      setVoiceSettingsStatus(error instanceof Error ? `test announcement error: ${error.message}` : "test announcement error: unknown error");
    }
  };

  return {
    announcementEnabled,
    announcementLabel,
    closeVoiceSettings,
    enterToSendDraft,
    hasAnnouncementCredentials,
    liveAudioRef,
    narrationEnabledDraft,
    openVoiceSettings,
    playTestSound,
    saveVoiceSettings,
    setAnnouncementEnabled,
    setEnterToSendDraft,
    setNarrationEnabledDraft,
    setVoiceSettingsStatus,
    setVoiceApiKeyDraft,
    setVoiceBaseUrlDraft,
    startAnnouncementPlayback,
    toggleAnnouncements,
    triggerTestAnnouncement,
    voiceApiKeyDraft,
    voiceBaseUrlDraft,
    voiceSettings,
    voiceSettingsOpen,
    voiceSettingsStatus,
  };
}
