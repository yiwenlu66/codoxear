import json
import subprocess
import textwrap
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.voice_openai_client import OpenAICompatibleClient
from codoxear.voice_push import VoicePushCoordinator


ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "codoxear" / "static" / "app_voice.js"
HELPERS = ROOT / "codoxear" / "static" / "app_voice_helpers.js"
MODAL = ROOT / "codoxear" / "static" / "app_modal.js"


def eval_desktop_notification() -> dict:
    sources = [path.read_text(encoding="utf-8") for path in (MODAL, HELPERS, VOICE)]
    script = textwrap.dedent(
        f"""
        const vm=require("vm"), notifications=[], focused=[];
        class HTMLElement {{}}
        class Notification {{ constructor(title,options) {{this.title=title;this.options=options;notifications.push(this)}} close(){{this.closed=true}} }} Notification.permission="granted";
        function node(){{return {{style:{{}},value:"",checked:false,textContent:"",classList:{{add(){{}},remove(){{}},toggle(){{}}}},setAttribute(){{}},addEventListener(){{}},matches:()=>false,focus(){{}}}}}}
        const ctx={{HTMLElement,Notification,window:{{isSecureContext:true,focus:()=>focused.push("window")}},navigator:{{userAgent:"X11"}},document:{{activeElement:null,contains:()=>true}},requestAnimationFrame:f=>f(),setTimeout:()=>1,clearTimeout(){{}},setInterval:()=>1,clearInterval(){{}}}};
        vm.createContext(ctx);for(const source of {json.dumps(sources)})vm.runInContext(source,ctx);
        const d={{announceBtn:node(),notificationBtn:node(),liveAudio:node(),voiceSettingsBackdrop:node(),voiceSettingsCloseBtn:node(),voiceSettingsStatus:node(),voiceBaseUrlInput:node(),voiceApiKeyInput:node(),voiceClearApiKeyToggle:node(),narrationSettingToggle:node(),voiceSettingsViewer:node(),voiceSettingsCancelBtn:node(),voiceSettingsSaveBtn:node(),isAppDisposed:()=>false,api:async u=>String(u).includes("feed")?{{items:[{{message_id:"m1",session_display_name:"Repo",notification_text:" hello   world ",session_id:"s1",updated_ts:1}}]}}:{{}},setToast(){{}},handleAppAuthLoss(){{}},prepareModalOpen(){{}},afterModalVisibilityChanged(){{}},resolveAppUrl:x=>x,versionedShellAssetPath:x=>x,storageGetItem:k=>k==="codoxear.notificationEnabled"?"1":k==="codoxear.desktopNotificationsEnabled"?"1":null,storageSetItem(){{}},storageRemoveItem(){{}},focusSessionFromNotification:s=>focused.push(s),requestFrame:f=>f(),setTimeout:()=>1,clearTimeout(){{}},setInterval:()=>1,clearInterval(){{}}}};
        (async()=>{{const c=ctx.window.CodoxearVoice.createVoiceController(d);await c.syncNotificationState({{subscriptions:[]}});await c.pollNotificationFeed();notifications[0].onclick({{preventDefault(){{}}}});process.stdout.write(JSON.stringify({{title:notifications[0].title,body:notifications[0].options.body,tag:notifications[0].options.tag,focused,closed:notifications[0].closed}}))}})().catch(e=>{{console.error(e);process.exit(1)}})
        """
    )
    proc = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


class TestVoicePushBehavior(unittest.TestCase):
    def test_summary_request_uses_a_bounded_compression_prompt(self) -> None:
        client = OpenAICompatibleClient(timeout_seconds=1)
        captured = {}
        client._request_json = lambda **kwargs: captured.update(kwargs) or {"choices": [{"message": {"content": "done"}}]}  # type: ignore[method-assign]
        self.assertEqual(client.summarize(base_url="https://api.openai.com/v1", api_key="key", model="gpt", session_name="repo", source_label="Final response", text="long response", target_words=30), "done")
        self.assertEqual(captured["payload"]["max_completion_tokens"], 72)
        self.assertIn("Aim for about 30 words", captured["payload"]["messages"][0]["content"])

    def test_browser_settings_projection_redacts_persisted_api_key(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = VoicePushCoordinator(app_dir=root, stop_event=threading.Event(), settings_path=root / "settings.json", subscriptions_path=root / "subscriptions.json", delivery_ledger_path=root / "ledger.json", vapid_private_key_path=root / "vapid.pem")
            coordinator.set_settings({"tts_api_key": "secret", "tts_base_url": "https://api.openai.com/v1"})
            snapshot = coordinator.settings_snapshot(redact_secrets=True)
            self.assertEqual(snapshot["tts_api_key"], "")
            self.assertTrue(snapshot["has_tts_api_key"])

    def test_desktop_notification_click_focuses_its_origin_session(self) -> None:
        result = eval_desktop_notification()
        self.assertEqual((result["title"], result["body"], result["tag"]), ("Repo", "hello world", "m1"))
        self.assertEqual(result["focused"], ["window", "s1"])
        self.assertTrue(result["closed"])


if __name__ == "__main__":
    unittest.main()
