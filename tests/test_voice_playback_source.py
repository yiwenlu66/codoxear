import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "codoxear" / "static" / "app_voice_helpers.js"


def eval_playback_support() -> dict:
    source = HELPERS.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm=require("vm"),ctx={{window:{{}}}};vm.createContext(ctx);vm.runInContext({json.dumps(source)},ctx);
        const h=ctx.window.CodoxearVoiceHelpers, playable={{canPlayType:k=>k==="application/vnd.apple.mpegurl"?"probably":""}}, maybe={{canPlayType:()=>"maybe"}}, none={{canPlayType:()=>""}};
        process.stdout.write(JSON.stringify({{native:h.browserSupportsNativeLiveAudioPlayback(playable),maybe:h.browserSupportsNativeLiveAudioPlayback(maybe),unsupported:h.browserSupportsLiveAudioPlayback(none,{{Hls:{{isSupported:()=>false}}}}),mse:h.browserSupportsLiveAudioPlayback(none,{{Hls:{{isSupported:()=>true}}}}),apple:h.shouldPreferNativeLiveAudioPlayback(maybe,{{vendor:"Apple Computer, Inc.",userAgent:"Safari"}}),chrome:h.shouldPreferNativeLiveAudioPlayback(maybe,{{vendor:"",userAgent:"AppleWebKit Chrome"}}),decoded:String.fromCharCode(...h.base64UrlToUint8Array("SGVsbG8td29ybGQ",v=>Buffer.from(v,"base64").toString("binary"))),mobile:h.notificationDeviceClass({{userAgent:"Mozilla Android Mobile"}})}}));
        """
    )
    proc = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


class TestVoicePlaybackBehavior(unittest.TestCase):
    def test_live_playback_selects_native_or_mse_and_rejects_unsupported_browsers(self) -> None:
        result = eval_playback_support()
        self.assertTrue(result["native"])
        self.assertTrue(result["maybe"])
        self.assertFalse(result["unsupported"])
        self.assertTrue(result["mse"])
        self.assertTrue(result["apple"])
        self.assertFalse(result["chrome"])

    def test_push_key_decoding_and_device_classification_are_executable(self) -> None:
        result = eval_playback_support()
        self.assertEqual(result["decoded"], "Hello-world")
        self.assertEqual(result["mobile"], "mobile")


if __name__ == "__main__":
    unittest.main()
