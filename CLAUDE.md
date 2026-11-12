# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

"Hey Claude" is a voice assistant in a single file, `assistant.py` (no package, no tests, no linter configured). Setup and user-facing docs are in `README.md`.

## Commands

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="sk-ant-..."   # required; Brain exits without it

python assistant.py --text    # typed questions, spoken answers: no mic/STT/wake word needed, best for testing
python assistant.py --ptt     # press Enter instead of wake word
python assistant.py           # wake-word mode
python assistant.py --list-voices
```

Other flags: `--announce`, `--no-web`, `--voice NAME`. Python 3.10+. Linux needs `portaudio19-dev espeak-ng`.

## Architecture

Pipeline: wake word -> record -> transcribe -> route -> answer -> speak. The pieces in `assistant.py`:

- **`Ears`**: a `sounddevice` input stream pushes 100 ms int16 chunks into a queue. It calibrates an RMS silence threshold at startup (`_calibrate`). Wake word uses a Vosk recognizer with a grammar restricted to `WAKE_PHRASES` + `[unk]`. `record_question` uses RMS-based endpointing (`SILENCE_TO_STOP`, `MAX_QUESTION_SECONDS`). Transcription is local faster-whisper (`WHISPER_SIZE`). Vosk and Whisper models are imported lazily; the Vosk model is downloaded to `~/.hey-claude/` on first use.
- **`Brain`**: two-stage Claude calls. `route()` forces a `route` tool call on Haiku, returning `{model, needs_web, reason}`; it falls back to sonnet on API error. `answer()` calls the chosen tier (`MODELS` dict) with `ANSWER_SYSTEM`, optionally with the server-side `web_search` tool. `_ask` loops on `pause_turn` (up to 4 times) and retries without web search on `BadRequestError`. It joins all text blocks, since web search splits the answer across several. History is kept for `MEMORY_MINUTES` / `MEMORY_TURNS`; only the final text is stored, not tool blocks.
- **`Voice`**: `--tts system` (default) is macOS `say`, otherwise `pyttsx3`; `--tts kokoro` is local neural TTS via `kokoro-onnx` (model files auto-download to `~/.hey-claude/`). After the wake word, `main` speaks a greeting ("Yes, <name>." from `--name` / `HEY_CLAUDE_NAME`) before the beep; with kokoro it is pre-rendered at startup. `speakable()` strips markdown, URLs and code before TTS.
- **`handle` / `speak` / `main`**: `speak` mutes the mic while TTS or beeps play so the assistant doesn't hear itself. `--text` mode passes `ears=None` and skips all audio input.

## Debugging

`--debug` turns on `dbg()` (prints `[debug] ...` and appends to `~/.hey-claude/debug.log`). New instrumentation should go through `dbg()` so it stays silent by default.

## Tuning

Behavior knobs (`ROUTER_SYSTEM`, `ANSWER_SYSTEM`, `WAKE_PHRASES`, `WHISPER_SIZE`, `SILENCE_TO_STOP`, `MODELS`) are constants at the top of `assistant.py`. `ANSWER_SYSTEM` is written for text-to-speech (no markdown, short answers), so keep answers speakable if you edit it.
