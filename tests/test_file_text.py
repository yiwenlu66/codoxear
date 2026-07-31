import hashlib
import tempfile
import unittest
from pathlib import Path

from codoxear.file_text import read_text_file_for_client
from codoxear.file_text import read_text_file_for_write
from codoxear.file_text import write_new_text_file_atomic
from codoxear.file_text import write_text_file_atomic


class TestFileTextModuleBehavior(unittest.TestCase):
    def test_text_read_write_helpers_preserve_versions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.txt"

            size, version = write_new_text_file_atomic(path, text="hello\n")
            self.assertEqual(size, len("hello\n".encode("utf-8")))
            self.assertEqual(version, hashlib.sha256(b"hello\n").hexdigest())

            text, read_size, read_version = read_text_file_for_write(path, max_bytes=1024)
            self.assertEqual(text, "hello\n")
            self.assertEqual(read_size, size)
            self.assertEqual(read_version, version)

            client_text, client_size, client_editable, client_version = read_text_file_for_client(path, max_bytes=1024)
            self.assertEqual(client_text, "hello\n")
            self.assertEqual(client_size, size)
            self.assertTrue(client_editable)
            self.assertEqual(client_version, version)

            new_size, new_version = write_text_file_atomic(path, text="goodbye\n")
            self.assertEqual(new_size, len("goodbye\n".encode("utf-8")))
            self.assertEqual(new_version, hashlib.sha256(b"goodbye\n").hexdigest())
            self.assertEqual(path.read_text(encoding="utf-8"), "goodbye\n")


if __name__ == "__main__":
    unittest.main()
