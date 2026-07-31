from __future__ import annotations

from dataclasses import dataclass

from .voice_push_state import AnnouncementTask
from .voice_push_state import DEFAULT_VOICES
from .voice_push_state import _sha256_hex


@dataclass(frozen=True)
class AnnouncementQueueUpdate:
    queue: list[AnnouncementTask]
    replaced_tasks: tuple[AnnouncementTask, ...]


def voice_for_session(session_id: str) -> str:
    token = _sha256_hex(session_id)
    return DEFAULT_VOICES[int(token[:8], 16) % len(DEFAULT_VOICES)]


def merge_narration_tasks(older: AnnouncementTask, newer: AnnouncementTask) -> AnnouncementTask:
    source_message_ids = tuple(dict.fromkeys((*older.source_message_ids, *newer.source_message_ids)))
    parts = [part for part in (older.source_text, newer.source_text) if part]
    return AnnouncementTask(
        message_id=newer.message_id,
        source_message_ids=source_message_ids,
        session_id=newer.session_id,
        session_display_name=newer.session_display_name,
        message_class="narration",
        source_text="\n\n".join(parts),
        spoken_text="",
        notification_text="",
        voice=newer.voice,
        ts=newer.ts if newer.ts is not None else older.ts,
        summary_word_target=newer.summary_word_target,
        listener_epoch=newer.listener_epoch,
    )


def enqueue_announcement_task(queue: list[AnnouncementTask], new_task: AnnouncementTask) -> AnnouncementQueueUpdate:
    if not queue:
        return AnnouncementQueueUpdate(queue=[new_task], replaced_tasks=())
    kept: list[AnnouncementTask] = []
    replaced: list[AnnouncementTask] = []
    insert_index: int | None = None
    task_to_enqueue = new_task
    for queued in queue:
        same_slot = queued.session_id == new_task.session_id and queued.message_class == new_task.message_class
        if not same_slot:
            kept.append(queued)
            continue
        if insert_index is None:
            insert_index = len(kept)
        if new_task.message_class == "narration":
            task_to_enqueue = merge_narration_tasks(queued, task_to_enqueue)
        else:
            replaced.append(queued)
    if insert_index is None:
        kept.append(task_to_enqueue)
    else:
        kept.insert(insert_index, task_to_enqueue)
    return AnnouncementQueueUpdate(queue=kept, replaced_tasks=tuple(replaced))
