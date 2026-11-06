"""Unit tests for codoxear.federation_proto framing and token auth."""

from __future__ import annotations

import json
import socket
import struct
import threading

import pytest

from codoxear import federation_proto


def _socketpair():
    return socket.socketpair()


def test_frame_round_trip():
    left, right = _socketpair()
    try:
        payload = {"type": "hello", "name": "node-1", "nested": {"list": [1, 2, 3]}, "text": "héllo"}
        federation_proto.send_frame(left, payload)
        assert federation_proto.recv_frame(right) == payload
        # Two frames back to back decode in order.
        federation_proto.send_frame(left, {"type": "register", "session": {"session_id": "a"}})
        federation_proto.send_frame(left, {"type": "unregister", "session_id": "a"})
        assert federation_proto.recv_frame(right)["type"] == "register"
        assert federation_proto.recv_frame(right)["type"] == "unregister"
    finally:
        left.close()
        right.close()


def test_recv_frame_rejects_zero_length():
    left, right = _socketpair()
    try:
        left.sendall(struct.pack(">I", 0))
        with pytest.raises(federation_proto.FederationProtocolError):
            federation_proto.recv_frame(right)
    finally:
        left.close()
        right.close()


def test_recv_frame_rejects_oversized_length():
    left, right = _socketpair()
    try:
        left.sendall(struct.pack(">I", federation_proto.MAX_FRAME_BYTES + 1))
        with pytest.raises(federation_proto.FederationProtocolError):
            federation_proto.recv_frame(right)
    finally:
        left.close()
        right.close()


def test_recv_frame_rejects_non_object_payload():
    left, right = _socketpair()
    try:
        body = json.dumps([1, 2, 3]).encode()
        left.sendall(struct.pack(">I", len(body)) + body)
        with pytest.raises(federation_proto.FederationProtocolError):
            federation_proto.recv_frame(right)
    finally:
        left.close()
        right.close()


def test_recv_frame_rejects_invalid_json():
    left, right = _socketpair()
    try:
        body = b"{not json"
        left.sendall(struct.pack(">I", len(body)) + body)
        with pytest.raises(federation_proto.FederationProtocolError):
            federation_proto.recv_frame(right)
    finally:
        left.close()
        right.close()


def test_recv_frame_raises_connection_error_on_eof():
    left, right = _socketpair()
    try:
        left.close()
        with pytest.raises(ConnectionError):
            federation_proto.recv_frame(right)
    finally:
        right.close()


def test_recv_frame_raises_connection_error_on_mid_frame_eof():
    left, right = _socketpair()
    try:
        left.sendall(struct.pack(">I", 100) + b"{}")
        left.close()
        with pytest.raises(ConnectionError):
            federation_proto.recv_frame(right)
    finally:
        right.close()


def test_token_accept_and_reject():
    secret = "shared-secret"
    token = federation_proto.federation_token(secret, "node-b")
    assert federation_proto.token_matches(secret, "node-b", token)
    # Wrong node name, wrong secret, and malformed tokens all reject.
    assert not federation_proto.token_matches(secret, "node-a", token)
    assert not federation_proto.token_matches("other-secret", "node-b", token)
    assert not federation_proto.token_matches(secret, "node-b", "")
    assert not federation_proto.token_matches(secret, "node-b", None)
    assert not federation_proto.token_matches(secret, "node-b", token[:-1] + ("0" if token[-1] != "0" else "1"))


def test_hello_message_carries_advertised_http_port():
    msg = federation_proto.hello_message(node_name="n1", secret="s", http_port=9999)
    assert msg["type"] == federation_proto.HELLO
    assert msg["http_port"] == 9999
    assert msg["version"] == federation_proto.PROTOCOL_VERSION
    assert federation_proto.token_matches("s", "n1", msg["token"])


def test_hello_welcome_handshake_round_trip():
    """A minimal hub-side accept: hello with valid token gets a welcome."""
    left, right = _socketpair()
    secret = "s3cret"
    try:
        received: dict = {}

        def hub_side():
            hello = federation_proto.recv_frame(right)
            received.update(hello)
            if federation_proto.token_matches(secret, hello["name"], hello.get("token")):
                federation_proto.send_frame(right, {"type": federation_proto.WELCOME})

        thread = threading.Thread(target=hub_side)
        thread.start()
        federation_proto.send_frame(left, federation_proto.hello_message(node_name="node-x", secret=secret, http_port=8743))
        reply = federation_proto.recv_frame(left)
        thread.join(timeout=5)
        assert reply["type"] == federation_proto.WELCOME
        assert received["name"] == "node-x"
    finally:
        left.close()
        right.close()
