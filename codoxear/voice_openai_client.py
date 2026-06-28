from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleClient:
    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self._timeout_seconds = float(timeout_seconds)

    def _request_json(self, *, base_url: str, api_key: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not api_key:
            raise ValueError("tts_api_key is required")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + route,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{route} failed with {e.code}: {detail}") from e
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"{route} returned non-object json")
        return obj

    def _request_bytes(self, *, base_url: str, api_key: str, route: str, payload: dict[str, Any]) -> bytes:
        if not api_key:
            raise ValueError("tts_api_key is required")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + route,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{route} failed with {e.code}: {detail}") from e

    def summarize(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        session_name: str,
        source_label: str,
        text: str,
        target_words: int,
    ) -> str:
        max_words = 15 if int(target_words) <= 15 else 30
        if max_words <= 15:
            system_content = (
                "You compress assistant progress narration for spoken mobile notifications. "
                "Return exactly one plain sentence with only the concrete progress fact. "
                "Aim for about 15 words, roughly 12 to 18 words. "
                "Use at most 15 words. If the source is already 15 words or fewer, do not expand it. "
                "Compression only: never add filler, politeness, waiting language, stage directions, or meta-commentary. "
                "No markdown, no quotes, no prefixes."
            )
        else:
            system_content = (
                "You compress assistant final responses for spoken mobile notifications. "
                "Return exactly one plain sentence with only the main result. "
                "Aim for about 30 words, roughly 24 to 36 words. "
                "Use at most 30 words. Prefer compression over paraphrase. "
                "Never add filler, politeness, stage directions, or meta-commentary, and never invent details not present in the source. "
                "No markdown, no quotes, no prefixes."
            )
        obj = self._request_json(
            base_url=base_url,
            api_key=api_key,
            route="/chat/completions",
            payload={
                "model": model,
                "temperature": 0.0,
                "max_completion_tokens": 48 if max_words <= 15 else 72,
                "messages": [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": f"Session name: {session_name}\n{source_label}:\n{text}",
                    },
                ],
            },
        )
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat completions response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("chat completions response missing message")
        content = message.get("content")
        if isinstance(content, str):
            summary = " ".join(content.split()).strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            summary = " ".join("".join(parts).split()).strip()
        else:
            raise ValueError("chat completions response missing content")
        if not summary:
            raise ValueError("empty summary response")
        summary_word_count = len(summary.split())
        if summary_word_count > max_words:
            raise ValueError(f"summary exceeded {max_words} words")
        return summary

    def synthesize(self, *, base_url: str, api_key: str, model: str, voice: str, text: str) -> bytes:
        audio = self._request_bytes(
            base_url=base_url,
            api_key=api_key,
            route="/audio/speech",
            payload={
                "model": model,
                "voice": voice,
                "input": text,
                "response_format": "aac",
            },
        )
        if not audio:
            raise ValueError("audio/speech returned empty body")
        return audio
