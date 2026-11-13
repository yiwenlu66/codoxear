# OPS — file viewer audio

Append-only. Timestamps are local (Asia/Shanghai unless noted).

## 2026-11-15

- Read the video feature end to end: `codoxear/video_preview.py`,
  `file_types.py`, `file_view.py`, `file_get_routes.py`, `file_global_routes.py`,
  `server_config.py`, `server.py`, `server_route_deps.py`,
  `static/app_file_video.js`, `app_file_render_surface.js`,
  `app_file_viewer_operations.js`, `app_file_viewer.js`, `app_file_mode.js`,
  `app_file_ops.js`, `app_wiring.js`, `app_shell.js`.
- Established that `resetFileViewerPanel()` (called by `openFilePath` and by the
  viewer `hide()` path) reaches `resetRenderSurface()` → `reset()` → pauses and
  unloads the `<video>`; the same funnel must clear `<audio>`.
- Established `send_inline_file_response` already implements single-range 206
  responses, so audio seeking needs no new transport work.
- Established `ffmpeg` on the host has `libmp3lame` (encoder list checked).
- Established the service worker bypasses
  `/api/(sessions/<id>/file|files)/(office|video)_preview` and every Range
  request, so a renamed `media_preview` route must be added to that regex.
- Web check: Safari/iOS does not natively decode Ogg Vorbis or Opus; MP3, AAC
  in MP4, WAV and FLAC are broadly supported. Decision recorded in EPISTEMIC.md.
- Grep census before editing: 173 references to `video_preview`/`videoPreview`
  outside build artifacts, spanning 12 python/js/sh files plus tests.

## 2026-11-15 (later)

- Concurrency note: another pi session (`codox-tip`, cwd /home/yiwen/codoxear)
  is running its own executor that edited `tests/fixtures/upload_fault_proxy.py`
  (adds `delay_upload`) in this same working tree. Announced my file set over
  intercom and asked them not to rebuild `dist/app.bundle.js` or commit while
  my client change is in flight. Commit rule for this task: stage explicit
  paths only, never the unrelated fixture edit.
- Validated the audio transcode argv locally: ogg (libvorbis) -> mp3
  (libmp3lame, `-q:a 2`) yields an audio-only `stream,mp3` ffprobe result, so
  the planned encoder flags are real on the target ffmpeg build.
- Confirmed `docker/verify.Dockerfile` installs ffmpeg (fixtures can be
  generated in-container) and that `codoxear-iso3:latest` exists locally, and
  that `agent-browser` supports `--args` for the autoplay-policy launch flag.
- Wrote `scripts/docker_audio_preview_verify.sh` (fixture media + A/B/C/D
  checks: mp3 direct play, ogg conversion, close stops playback, mkv video
  regression).
