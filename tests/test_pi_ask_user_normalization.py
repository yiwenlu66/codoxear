from codoxear import pi_messages


def test_ask_user_tool_call_and_result_emit_single_event() -> None:
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "ask-1",
                        "name": "ask_user",
                        "arguments": {
                            "context": "Pick where todo should appear.",
                            "question": "Where should the todo live?",
                            "options": ["Details", "Sidebar"],
                            "allowFreeform": True,
                        },
                    }
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "ask-1",
                "toolName": "ask_user",
                "details": {
                    "answer": "Details",
                    "cancelled": False,
                    "wasCustom": False,
                },
                "content": [{"type": "text", "text": "User answered: Details"}],
            },
        },
    ]

    events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)

    assert len(events) == 1
    assert events[0]["type"] == "ask_user"
    assert events[0]["tool_call_id"] == "ask-1"
    assert events[0]["question"] == "Where should the todo live?"
    assert events[0]["context"] == "Pick where todo should appear."
    assert events[0]["options"] == ["Details", "Sidebar"]
    assert events[0]["allow_freeform"] is True
    assert events[0]["answer"] == "Details"
    assert events[0]["cancelled"] is False
    assert events[0]["was_custom"] is False
    assert events[0]["resolved"] is True
    assert events[0]["ts"] == 0.0


def test_ask_user_without_result_stays_unresolved() -> None:
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "ask-2",
                        "name": "ask_user",
                        "arguments": {"question": "Pick one", "options": ["A", "B"]},
                    }
                ],
            },
        }
    ]

    events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)

    assert len(events) == 1
    assert events[0]["type"] == "ask_user"
    assert events[0]["tool_call_id"] == "ask-2"
    assert events[0]["question"] == "Pick one"
    assert events[0]["context"] == ""
    assert events[0]["options"] == ["A", "B"]
    assert events[0]["allow_freeform"] is True
    assert events[0]["resolved"] is False
    assert events[0]["ts"] == 0.0
    assert events[0].get("answer") is None


def test_ask_user_snake_case_flags_are_respected() -> None:
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "ask-2b",
                        "name": "ask_user",
                        "arguments": {
                            "question": "Pick destinations",
                            "options": ["A", "B"],
                            "allow_freeform": False,
                            "allow_multiple": True,
                        },
                    }
                ],
            },
        }
    ]

    events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)

    assert len(events) == 1
    assert events[0]["type"] == "ask_user"
    assert events[0]["tool_call_id"] == "ask-2b"
    assert events[0]["allow_freeform"] is False
    assert events[0]["allow_multiple"] is True
    assert events[0]["resolved"] is False


def test_ask_user_multiselect_result_preserves_answer_list() -> None:
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "ask-3",
                        "name": "ask_user",
                        "arguments": {
                            "question": "Pick destinations",
                            "options": ["Inbox", "Sidebar", "Details"],
                            "allowMultiple": True,
                        },
                    }
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "ask-3",
                "toolName": "ask_user",
                "details": {
                    "answer": ["Inbox", "Sidebar"],
                    "cancelled": False,
                    "wasCustom": False,
                },
            },
        },
    ]

    events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)

    assert len(events) == 1
    assert events[0]["type"] == "ask_user"
    assert events[0]["tool_call_id"] == "ask-3"
    assert events[0]["allow_multiple"] is True
    assert events[0]["answer"] == ["Inbox", "Sidebar"]
    assert events[0]["resolved"] is True


def test_ask_user_without_usable_id_degrades_to_request_event() -> None:
    entries = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "name": "ask_user",
                        "arguments": {
                            "context": "Need a placement decision.",
                            "question": "Where should it go?",
                            "options": ["Sidebar", "Panel"],
                            "allowMultiple": True,
                        },
                    }
                ],
            },
        }
    ]

    events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries, include_system=True)

    assert len(events) == 1
    assert events[0]["type"] == "ask_user"
    assert events[0]["tool_call_id"] is None
    assert events[0]["question"] == "Where should it go?"
    assert events[0]["context"] == "Need a placement decision."
    assert events[0]["options"] == ["Sidebar", "Panel"]
    assert events[0]["allow_multiple"] is True
    assert events[0]["resolved"] is False
    assert events[0]["ts"] == 0.0
