# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

"Hey Claude" is a voice assistant in `assistant.py` plus the optional visuals in `viz.py` (no package, no tests, no linter configured). Setup and user-facing docs are in `README.md`.

## Commands

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="sk-ant-..."   # required; Brain exits without it

python assistant.py --text    # typed questions, spoken answers: no mic/STT/wake word needed, best for testing
python assistant.py --ptt     # press Enter instead of wake word
python assistant.py           # wake-word mode
python assistant.py --list-voices
python assistant.py --viz pygame     # or --viz terminal: live waveforms and spectrum
```

Other flags: `--announce`, `--no-web`, `--voice NAME`. Python 3.10+. Linux needs `portaudio19-dev espeak-ng`.

## Architecture

Pipeline: wake word -> record -> transcribe -> route -> answer -> speak. The pieces in `assistant.py`:

- **`Ears`**: a `sounddevice` input stream pushes 100 ms int16 chunks into a queue. It calibrates an RMS silence threshold at startup (`_calibrate`). Wake word uses a Vosk recognizer with a grammar restricted to `WAKE_PHRASES` + `[unk]`. `record_question` uses RMS-based endpointing (`SILENCE_TO_STOP`, `MAX_QUESTION_SECONDS`). Transcription is local faster-whisper (`WHISPER_SIZE`). Vosk and Whisper models are imported lazily; the Vosk model is downloaded to `~/.hey-claude/` on first use.
- **`Brain`**: two-stage Claude calls. `route()` forces a `route` tool call on Haiku, returning `{model, needs_web, reason}`; it falls back to sonnet on API error. `answer()` calls the chosen tier (`MODELS` dict) with `ANSWER_SYSTEM`, optionally with the server-side `web_search` tool. `_ask` loops on `pause_turn` (up to 4 times) and retries without web search on `BadRequestError`. It joins all text blocks, since web search splits the answer across several. History is kept for `MEMORY_MINUTES` / `MEMORY_TURNS`; only the final text is stored, not tool blocks.
- **`Voice`**: `--tts system` (default) is macOS `say`, otherwise `pyttsx3`; `--tts kokoro` is local neural TTS via `kokoro-onnx` (model files auto-download to `~/.hey-claude/`). After the wake word, `main` speaks a greeting ("Yes, <name>." from `--name` / `HEY_CLAUDE_NAME`) before the beep; with kokoro it is pre-rendered at startup. `--stream-tts` makes `Voice._say_stream` use kokoro's `create_stream` (a producer thread feeds a bounded queue; one `sd.OutputStream` plays the chunks; the greeting still uses the pre-rendered cache path). `speakable()` strips markdown, URLs and code before TTS.
- **`handle` / `speak` / `main`**: `speak` mutes the mic while TTS or beeps play so the assistant doesn't hear itself. `--text` mode passes `ears=None` and skips all audio input. `handle` returns True when the spoken reply ends with a question (`asks_back`) and wasn't stopped or an error; the main loop then skips the wake word / Enter and greeting and records the reply with `record_question(wait=FOLLOWUP_WAIT_SECONDS)`, falling back to the wake word on silence. A follow-up reply that `is_stop_reply` matches (the whole utterance is one of `STOP_PHRASES`, or just thanks, with `STOP_FILLERS` trimmed from the ends) ends the conversation with a low beep and no API call or log entry; it doesn't apply to wake-word questions. Log entries carry `"followup": true` for those turns.

## Visualizations (`viz.py`)

`--viz terminal|pygame` (default none). `viz.hub` is a thread-safe feed that `assistant.py` pushes into via `VIZ`: `Ears._on_audio` (`push_mic`, zeros while muted), `Voice._say` (`voice_start(samples, rate)` / `voice_chunk` when streaming / `voice_end`; the answer caption set by `set_answer` only appears at the first of these; kokoro passes its samples so the view slices them by playback time, other engines get a simulated voice), and `handle` / the main loop (`set_stage`, `begin_turn`, `set_route`, `set_answer`). The feed methods are near no-ops until `hub.enabled`. Renderers (`TerminalView` plain ANSI on the alternate screen with stdout/stderr teed into `hub.log`, with Leo as a 20x20 pixel sprite drawn in 24-bit color half blocks (`_LION_HALF` mirrored, per-pose row overrides in `LION_POSES`, same `POSE_FOR_STAGE` mapping) and the captions beside him when the terminal is at least 60x26; `PygameView`) only read `hub.frame()`. With a viz on, `main` runs `session(args)` (everything including `Voice`/`Ears` creation) in a worker thread via `viz.run`, keeping the renderer on the main thread; without it `session` runs directly. Keep new events going through `VIZ` rather than into the renderers. `PygameView` draws Leo inside the orb's glow, in a pose picked by stage (`POSE_FOR_STAGE`: portrait when idle, poses from the `assets/leo-poses.webp` sheet for listening / thinking / speaking), crossfading over `POSE_FADE`. Each pose is a circle cut from an image (`AVATARS`, in that image's pixels), cached per size; missing files fall back to the portrait, then to the plain orb; `assets/leo-full.webp` is the full-body version used in the README. The window icon is the portrait circle; on Windows, `_windows_app_id` gives the process its own taskbar entry (so the icon isn't python.exe's) and `_windows_title_bar` colors the title bar via DWM (both best effort, silently skipped elsewhere).

## Stopping speech

The mic is muted while Leo talks (to avoid hearing itself), so stopping is a global hotkey (`STOP_HOTKEY`, via `pynput`) that sets the `_stop` event. `Voice.say()` polls it and returns True when cut short; `speak()`, `handle()` and the main loop use that to skip the rest of the turn. `_stop` is cleared at the start of each turn, not inside `say()`, so a press while Leo is still thinking cancels the spoken reply.

## Conversation log

Separate from `--debug`, every question is appended as one JSON object per line to `~/.hey-claude/conversations/YYYY-MM.jsonl` (monthly files keep each one small), unless `--no-log`. `Brain` collects per-call stats in `brain.trace` (reset by `handle()`), and `handle()` builds the record and calls `log_turn()` on every exit path. The entry has a `"v": 1` schema version; bump it if fields change meaning.

## Debugging

`--debug` turns on `dbg()` (prints `[debug] ...` and appends to `~/.hey-claude/debug.log`). New instrumentation should go through `dbg()` so it stays silent by default.

## Tuning

Behavior knobs (`ROUTER_SYSTEM`, `ANSWER_SYSTEM`, `WAKE_PHRASES`, `WHISPER_SIZE`, `SILENCE_TO_STOP`, `MODELS`) are constants at the top of `assistant.py`. `ANSWER_SYSTEM` is written for text-to-speech (no markdown, short answers), so keep answers speakable if you edit it. Its length is set by the wording and the example Q/A pairs, not a word cap (a stated maximum became the target); `ANSWER_MAX_TOKENS` is only a safety cap, and must stay generous: these models think by default and thinking tokens count toward it (a 400 cap produced empty answers). Thinking depth per tier is `EFFORT`, sent as `output_config.effort`; `Brain.answer` replaces an empty answer with a spoken apology instead of going silent. Since a trailing question opens the mic, the prompt also limits when Leo asks back. Check changes against `answer_words` in the conversation log.
