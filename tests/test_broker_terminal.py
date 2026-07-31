import unittest

from codoxear.broker_terminal import _reply_to_terminal_queries


class TestBrokerTerminalQueries(unittest.TestCase):
    def test_replies_to_terminal_queries_and_consumes_buffer(self) -> None:
        writes: list[tuple[int, bytes]] = []
        buf = _reply_to_terminal_queries(
            term_query_buf=b"prefix",
            fd=7,
            chunk=b"\x1b[5n middle \x1b[c",
            write_all=lambda fd, data: writes.append((fd, data)),
        )

        self.assertEqual(writes, [(7, b"\x1b[0n"), (7, b"\x1b[?1;2c")])
        self.assertNotIn(b"\x1b[5n", buf)
        self.assertNotIn(b"\x1b[c", buf)

    def test_matches_queries_across_chunk_boundary(self) -> None:
        writes: list[bytes] = []
        buf = _reply_to_terminal_queries(
            term_query_buf=b"abc\x1b[",
            fd=3,
            chunk=b"6n",
            write_all=lambda _fd, data: writes.append(data),
        )

        self.assertEqual(writes, [b"\x1b[1;1R"])
        self.assertNotIn(b"\x1b[6n", buf)


if __name__ == "__main__":
    unittest.main()
