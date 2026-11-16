"""A file reply in another app version's shape names the fix, not the field."""
import unittest

from test_file_preview_default import CONTROLLER_HARNESS, run_vm


MESSAGE = "page and server versions don't match \u2014 reload the page"


class TestFileReplyVersionMismatch(unittest.TestCase):
    def test_reply_missing_current_fields_asks_for_reload(self):
        # The video reply is the pre-media_url shape a service worker served
        # from a cache written before the rename deploy.
        result = run_vm(CONTROLLER_HARNESS + r"""
const controller = buildController();
const request = controller.startFileOpenRequest('clip.mp4', {}).request;
const replies = {
  video: { kind: 'video', content_type: 'video/mp4', size: 1, video_url: '/blob' },
  audio: { kind: 'audio', content_type: 'audio/mpeg', size: 1 },
  image: { kind: 'image', size: 1 },
  pdf: { kind: 'pdf', size: 1 },
  text: { kind: 'text' },
  nokind: { ok: true },
};
const messages = {};
for (const [name, reply] of Object.entries(replies)) {
  try { controller.prepareFileLoadResult('clip.mp4', reply, request); messages[name] = null; }
  catch (error) { messages[name] = error.message; }
}
process.stdout.write(JSON.stringify(messages));
""")
        self.assertEqual(result, {name: MESSAGE for name in ("video", "audio", "image", "pdf", "text", "nokind")})


if __name__ == "__main__":
    unittest.main()
