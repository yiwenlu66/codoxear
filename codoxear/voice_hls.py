from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


HLS_TARGET_DURATION_SECONDS = 12
HLS_MAX_SEGMENTS = 18
HLS_KEEPALIVE_SECONDS = 6.0
HLS_SILENCE_SECONDS = 6.0


class MergedHLSStream:
    def __init__(self, *, root_dir: Path) -> None:
        self._root_dir = Path(root_dir)
        self._segments_dir = self._root_dir / "segments"
        self._playlist_path = self._root_dir / "live.m3u8"
        self._lock = threading.Lock()
        self._segments: list[dict[str, Any]] = []
        self._next_seq = 1
        self._last_error = ""
        self._last_append_ts = 0.0
        os.makedirs(self._segments_dir, exist_ok=True)
        self._rewrite_playlist()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "segment_count": len(self._segments),
                "last_error": self._last_error,
                "media_sequence": self._segments[0]["seq"] if self._segments else self._next_seq,
            }

    def playlist_bytes(self) -> bytes:
        with self._lock:
            return self._playlist_path.read_bytes() if self._playlist_path.exists() else b"#EXTM3U\n"

    def segment_path(self, segment_name: str) -> Path:
        name = Path(segment_name).name
        if name != segment_name or not name.endswith(".ts"):
            raise FileNotFoundError(segment_name)
        path = (self._segments_dir / name).resolve()
        if not str(path).startswith(str(self._segments_dir.resolve())) or not path.exists():
            raise FileNotFoundError(segment_name)
        return path

    def append_audio(self, *, message_id: str, audio_bytes: bytes) -> float:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe are required for merged HLS output")
        os.makedirs(self._segments_dir, exist_ok=True)
        input_path = self._segments_dir / f"{message_id[:12] or 'audio'}.aac"
        input_path.write_bytes(audio_bytes)
        tmp_pattern = self._segments_dir / f"{message_id[:12] or 'audio'}-part-%03d.ts"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-f",
                    "segment",
                    "-segment_time",
                    "6",
                    "-segment_format",
                    "mpegts",
                    "-reset_timestamps",
                    "1",
                    str(tmp_pattern),
                ],
                check=True,
                capture_output=True,
            )
            total_duration = 0.0
            chunk_paths = sorted(self._segments_dir.glob(f"{message_id[:12] or 'audio'}-part-*.ts"))
            if not chunk_paths:
                raise RuntimeError("ffmpeg produced no HLS segments")
            for chunk_path in chunk_paths:
                try:
                    duration = self._segment_duration_seconds(chunk_path)
                except RuntimeError as e:
                    if "invalid ffprobe duration: N/A" not in str(e):
                        raise
                    try:
                        chunk_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                seq, segment_name, segment_path = self._reserve_segment(f"{message_id[:12] or 'audio'}")
                chunk_path.replace(segment_path)
                total_duration += duration
                self._store_segment(seq=seq, segment_name=segment_name, segment_path=segment_path, duration=duration)
            if total_duration <= 0.0:
                raise RuntimeError("ffmpeg produced no valid HLS segments")
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed: {detail}") from e
        finally:
            try:
                input_path.unlink()
            except FileNotFoundError:
                pass
            for chunk_path in self._segments_dir.glob(f"{message_id[:12] or 'audio'}-part-*.ts"):
                try:
                    chunk_path.unlink()
                except FileNotFoundError:
                    pass

        return total_duration

    def append_silence(self, *, force: bool = False) -> bool:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe are required for merged HLS output")
        with self._lock:
            if (not force) and self._last_append_ts and (time.time() - self._last_append_ts) < HLS_KEEPALIVE_SECONDS:
                return False
        os.makedirs(self._segments_dir, exist_ok=True)
        seq, segment_name, segment_path = self._reserve_segment("silence")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=24000:cl=mono",
                    "-t",
                    str(HLS_SILENCE_SECONDS),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "32k",
                    "-f",
                    "mpegts",
                    str(segment_path),
                ],
                check=True,
                capture_output=True,
            )
            duration = self._segment_duration_seconds(segment_path)
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed: {detail}") from e
        self._store_segment(seq=seq, segment_name=segment_name, segment_path=segment_path, duration=duration)
        return True

    def reset(self) -> None:
        with self._lock:
            old_paths = [Path(item["path"]) for item in self._segments]
            self._segments = []
            self._last_error = ""
            self._last_append_ts = 0.0
            self._rewrite_playlist()
        for path in old_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _reserve_segment(self, prefix: str) -> tuple[int, str, Path]:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        segment_name = f"{seq:06d}-{prefix[:12]}.ts"
        segment_path = self._segments_dir / segment_name
        return seq, segment_name, segment_path

    def _store_segment(self, *, seq: int, segment_name: str, segment_path: Path, duration: float) -> None:
        with self._lock:
            self._segments.append({"seq": seq, "name": segment_name, "duration": duration, "path": segment_path})
            self._segments.sort(key=lambda item: int(item["seq"]))
            while len(self._segments) > HLS_MAX_SEGMENTS:
                old = self._segments.pop(0)
                try:
                    Path(old["path"]).unlink()
                except FileNotFoundError:
                    pass
            self._last_append_ts = time.time()
            self._rewrite_playlist()

    def set_last_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message or "").strip()

    def _segment_duration_seconds(self, segment_path: Path) -> float:
        try:
            raw = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(segment_path),
                ],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e}") from e
        try:
            value = float(raw)
        except ValueError as e:
            raise RuntimeError(f"invalid ffprobe duration: {raw}") from e
        return max(0.2, value)

    def _rewrite_playlist(self) -> None:
        target_duration = max(
            HLS_TARGET_DURATION_SECONDS,
            int(math.ceil(max((float(item["duration"]) for item in self._segments), default=0.0))),
        )
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            f"#EXT-X-MEDIA-SEQUENCE:{self._segments[0]['seq'] if self._segments else self._next_seq}",
        ]
        for item in self._segments:
            lines.append(f"#EXTINF:{item['duration']:.3f},")
            lines.append(f"segments/{item['name']}")
        self._playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
