# Epistemic model — file viewer audio

## Phenomenon to explain

An audio file in a session workspace is currently either unopenable (no audio
kind exists) or, for `.ogg`, silently treated as video and rendered as a
320px-tall black player. The feature must make audio play like video does.

## Live mechanisms

- **The video pipeline is a media pipeline with a video-shaped name.** The
  decision content is: original URL + optional converted URL + a per-kind set
  of types the browser is trusted to decode. Nothing in it is video-specific
  except the target element and the encoder flags. So the smallest model is one
  media state machine parameterized by kind, not a parallel audio copy — a copy
  would duplicate the fallback token/preparing/used state machine and the
  header control for no behavioral difference.
- **Conversion is load-bearing, not decorative.** iOS Safari (the primary
  client) cannot decode Ogg Vorbis/Opus, and agent-produced speech is commonly
  `.ogg`/`.opus`. Without conversion those files fail on the phone with no
  recovery path. Hence the MP3 conversion exists for the same reason MP4 does
  for `.mkv`: a codec the client cannot read.
- **The safe-set is a bet, the error handler is the recovery.** Claiming a MIME
  type is safe only decides whether the raw file is tried first; an element
  `error` triggers conversion automatically. That is why the audio safe set can
  be narrow (MP3/MP4/WAV/FLAC) without costing desktop Chrome users a dead end.

## Ruled out

- Separate `/file/audio_preview` route + duplicated client fallback machinery:
  duplicates the token guard, the preparing state, and the header control.
- Leaving `.ogg` classified as video: it renders an audio-only file in a video
  box and misroutes the conversion (MP4 with no video track).
- Reusing `<video>` for audio: a `<video>` element on an audio-only file paints
  a large empty box; the native audio control strip is the correct surface.

## Live mechanisms (confirmed while building)

- **The original `OggS` sniff had a two-sided defect.** `video_content_type`
  claimed every Ogg stream as `video/ogg`, so a Vorbis file opened in the video
  stage. The first attempt at a fix then made the mirror mistake (extension
  checked before the codec, so Theora-in-`.ogg` became audio). The stable rule
  is three ordered checks: a declared video extension keeps its video identity;
  inside `OggS` the theora identification header decides; only then do the
  audio extension map and the remaining magic bytes apply. Verified by running
  `file_kind` on all three shapes.
- **The video payload key rename is safe because the API has one client.** The
  only readers of `video_url`/`video_preview_url` were the file viewer runtime
  and the python tests that pin the payload.

## Anomalies / open questions

- Whether iOS Safari fires `error` on an undecodable `<audio>` with
  `preload="metadata"` before the user presses play. If it does not, the
  conversion must be reached through the header control instead. This is why
  Ogg is excluded from the trusted set: the conversion is taken *first*, not
  only after an error.
- Answered: no other consumer reads `video_url`/`VIDEO_PREVIEW_DIR`; the grep
  census in OPS.md covers python, JS, shell and docker scripts.

## Current claim

Audio should land as a *media* generalization of the video path: `file_kind`
gains `audio`, the preview cache becomes format-parameterized, and the client
gains a second element inside one media fallback state. Evidence for this is
that every video-specific branch either reads the kind or sets the encoder,
both of which are one-line parameterizations.

## What would most change the model

A real-browser measurement showing that the raw-blob `<audio>` element fails on
a format we declared trusted (e.g. FLAC on some iOS version) — that would move
the item out of the safe set.
