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

## 2026-11-15 (integration)

- Server half: 189 targeted tests green including `test_audio_preview_real_ffmpeg`
  (WAV -> MP3, ffprobe reports one audio stream with codec mp3).
- Client half: wiring guard passed, bundle rebuilt, VM behavior tests green.
- Integration fixes I made myself: `tests/test_frontend_toast_routing.py` still
  loaded `app_file_video.js`; `tests/test_frontend_file_helpers_source.py` still
  called `fileVideoPreviewErrorText` (and expected the old default text); the
  PDF/fallback seam `clearFileVideo` -> `clearFileMedia`; `TestVideoPreviewCache`
  -> `TestMediaPreviewCache`; `_generate_media_preview` built the video argv then
  overwrote it for audio (now one per-kind branch); added a VM test for the
  error-recovery path (trusted type errors once -> converts, converted file
  errors -> reported, no loop); added a `libmp3lame` probe to `codoxear-doctor`.
- Full suite: 2975 passed, 1 failed — `test_hint_mode_coverage` (`draftUndoBtn`),
  which fails on HEAD from the concurrent session's committed draft-undo work,
  not from this change. The concurrent session's uncommitted `app.css`
  attachment-chip hunk and `scripts/docker_upload_pill_geometry_verify.*` were
  left unstaged; `.fileAudio` was staged as a single hunk from `app.css`.
- Commits: 83ff7dfc (server pipeline), 71564572 (client media surface),
  76877f20 (docs + Docker audio verifier). Pushed to origin/main.
- Armed `scripts/docker_audio_preview_verify.sh HEAD` detached for the
  real-browser run (pid 3742631, log /tmp/audio-verify-watch.log).

## 2026-11-15 (browser verification)

- Run 1 (commit 76877f20) failed in the harness: the picker's only row is
  `filePickerOption-0`, and the script hardcoded `-1`. Fixed by clicking the
  menu's first row and pinning the opened fixture by its status line
  (af04757f).
- Run 2 (commit af04757f) collected every browser observation with PASS, but
  the report block had unclosed `bool(` expressions, so it died writing
  report.json and never printed the verdict. Re-running that block against the
  recorded artifacts produced pass=true; fixed the script (ac6d922e) and
  re-ran so the verifier itself emits PASS.
- Run 2 observations (raw, in /tmp/codoxear-audio-verify.el76R2):
  - MP3: `#fileAudio` visible, `#fileVideo` hidden, src = `/file/blob?path=tone.mp3`
    (no conversion), `tone.mp3 - audio - 5.66 KB`, currentTime 0.451/1.2 s,
    paused false, readyState 4, mediaError null, control titled
    "Use compatible MP3 preview".
  - Ogg Vorbis: src = `/file/media_preview?path=speech.ogg`, status
    `speech.ogg - compatible audio preview - 5.79 KB`, currentTime 0.655/1.2 s,
    readyState 4, no error; the derivative answers 206 with
    `content-type: audio/mpeg`, `accept-ranges: bytes`,
    `content-range: bytes 0-0/5924`.
  - Close: src null, paused true, element hidden, viewer closed.
  - MKV: src = `/file/media_preview?path=clip.mkv`, status
    `clip.mkv - compatible video preview - 13.1 KB`, videoWidth 160,
    currentTime 0.639/0.928 s — the video path still converts and plays.
  - pageErrors [], console [] (no fatal lines).
- Screenshots kept in this task directory: audio-mp3.png,
  audio-ogg-converted.png.
