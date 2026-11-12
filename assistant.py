#!/usr/bin/env python3
"""
Hey Leo — a voice assistant for your computer.

  1. Waits for you to say "Hey Leo" (offline wake word, nothing leaves your machine).
  2. Records your question until you stop talking, and transcribes it locally.
  3. Claude Haiku looks at the question and picks the right model for it:
       Haiku  -> quick facts, lookups, small talk
       Sonnet -> explanations, advice, comparisons
       Opus   -> hard reasoning, math, tricky planning
  4. That model answers (it can search the web for current info),
     and the answer is read back to you out loud.

Usage:
  python assistant.py            # wake word mode: say "Hey Leo"
  python assistant.py --ptt      # push-to-talk: press Enter, then speak
  python assistant.py --text     # type questions, hear answers (good for testing)

Options:
  --announce     say which model is answering ("Let me think harder...")
  --no-web       don't let Claude search the web
  --voice NAME   pick a text-to-speech voice (run with --list-voices to see them)
  --tts kokoro   natural-sounding local voice instead of the system voices
  --name NAME    after the wake word it replies "Yes, NAME" (or set HEY_CLAUDE_NAME)
  --debug        show timings, token use, wake-word and web-search details (also logged to a file)
"""

import argparse
import json
import os
import platform
import queue
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import anthropic

# ---------------------------------------------------------------- settings ---

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
ROUTER_MODEL = MODELS["haiku"]

BOT_NAME = "Leo"
WAKE_PHRASES = ["hey leo", "okay leo"]
WHISPER_SIZE = "base.en"          # "tiny.en" is faster, "small.en" is more accurate
SAMPLE_RATE = 16000
CHUNK = 1600                      # 100 ms of audio per chunk
SILENCE_TO_STOP = 1.2             # seconds of quiet that ends your question
MAX_QUESTION_SECONDS = 20
WAIT_FOR_SPEECH_SECONDS = 6
MEMORY_MINUTES = 5                # follow-up questions remember context this long
MEMORY_TURNS = 6

APP_DIR = Path.home() / ".hey-claude"
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_DIR = APP_DIR / "vosk-model-small-en-us-0.15"

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ["kokoro-v1.0.int8.onnx", "voices-v1.0.bin"]   # about 120 MB total
KOKORO_DEFAULT_VOICE = "am_michael"   # others to try: am_onyx, am_liam, bm_daniel

DEBUG = False
LOG_FILE = APP_DIR / "debug.log"


def dbg(msg):
    """Debug output: printed to the terminal and appended to LOG_FILE. Only active with --debug."""
    if not DEBUG:
        return
    print(f"  [debug] {msg}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def enable_debug():
    global DEBUG
    DEBUG = True
    APP_DIR.mkdir(parents=True, exist_ok=True)
    dbg(f"--- debug session started (log file: {LOG_FILE}) ---")


def usage_str(resp):
    u = getattr(resp, "usage", None)
    return f"{getattr(u, 'input_tokens', '?')} in / {getattr(u, 'output_tokens', '?')} out tokens"


ANSWER_SYSTEM = """You are Leo, a voice assistant. Your answer will be read aloud by a \
text-to-speech engine, so:
- Speak naturally, like a knowledgeable friend. Lead with the answer.
- Keep it short: usually one to three sentences. Go longer only if the question \
truly needs it, and even then stay under about 150 words.
- No markdown, bullet points, headings, tables, emoji, URLs or code blocks. \
Write numbers and units the way a person would say them.
- If you search the web, don't read out sources or links; just give the answer.
- If the question is ambiguous, give the most likely answer rather than asking back, \
unless you really can't.
The current date and time is {now}."""

ROUTER_SYSTEM = """You route spoken questions to the right AI model. Pick the \
cheapest model that will answer well.

haiku: quick facts, definitions, unit conversions, simple math, weather, sports \
scores, news headlines, trivia, small talk, simple how-to, recipes, spelling.
sonnet: questions that need a real explanation or judgment: "why" and "how does it \
work" questions, advice, comparisons, recommendations, summaries, writing help, \
moderate multi-step reasoning.
opus: genuinely hard problems: tricky math or logic puzzles, multi-constraint \
planning, deep analysis, debugging, questions where subtle reasoning or expertise \
really matters and a mistake would be costly.

Also decide if answering needs current information from the web (anything about \
today, recent events, prices, scores, weather, hours, or things that change)."""

ROUTER_TOOL = {
    "name": "route",
    "description": "Choose which model answers the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "model": {"type": "string", "enum": ["haiku", "sonnet", "opus"]},
            "needs_web": {"type": "boolean"},
            "reason": {"type": "string", "description": "A few words on why."},
        },
        "required": ["model", "needs_web", "reason"],
    },
}

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

ANNOUNCEMENTS = {"haiku": None, "sonnet": "Let me think.", "opus": "Let me think harder about that."}


# ------------------------------------------------------------------ claude ---

class Brain:
    def __init__(self, allow_web=True):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first (see README.md).")
        self.client = anthropic.Anthropic()
        self.allow_web = allow_web
        self.history = []
        self.last_turn = 0.0

    def route(self, question):
        try:
            t = time.time()
            resp = self.client.messages.create(
                model=ROUTER_MODEL,
                max_tokens=200,
                system=ROUTER_SYSTEM,
                tools=[ROUTER_TOOL],
                tool_choice={"type": "tool", "name": "route"},
                messages=[{"role": "user", "content": question}],
            )
            for block in resp.content:
                if block.type == "tool_use":
                    dbg(f"router ({ROUTER_MODEL}): {time.time() - t:.2f}s, {usage_str(resp)}, result={block.input}")
                    return block.input
        except anthropic.APIError as e:
            if is_fatal(e):          # no point falling back; the answer call would fail the same way
                raise
            print(f"  (router error, defaulting to sonnet: {e})")
        return {"model": "sonnet", "needs_web": False, "reason": "fallback"}

    def answer(self, question, tier, needs_web):
        if time.time() - self.last_turn > MEMORY_MINUTES * 60:
            self.history = []
        messages = self.history + [{"role": "user", "content": question}]
        system = ANSWER_SYSTEM.format(now=time.strftime("%A, %B %d, %Y, %I:%M %p"))
        use_web = self.allow_web and needs_web

        text = self._ask(MODELS[tier], system, messages, use_web)

        self.history = (messages + [{"role": "assistant", "content": text}])[-MEMORY_TURNS * 2:]
        self.last_turn = time.time()
        return text

    def _ask(self, model, system, messages, use_web):
        kwargs = dict(model=model, max_tokens=1024, system=system)
        if use_web:
            kwargs["tools"] = [WEB_SEARCH_TOOL]
        convo = list(messages)
        try:
            for _ in range(4):  # web search can pause a long turn; continue it
                t = time.time()
                resp = self.client.messages.create(messages=convo, **kwargs)
                dbg(f"{model}: {time.time() - t:.2f}s, stop={resp.stop_reason}, {usage_str(resp)}")
                for b in resp.content:
                    if b.type == "server_tool_use":
                        dbg(f"web search: {getattr(b, 'input', {}).get('query')!r}")
                if resp.stop_reason != "pause_turn":
                    break
                convo = convo + [{"role": "assistant", "content": resp.content}]
        except anthropic.BadRequestError as e:
            if use_web and not is_fatal(e):
                print(f"  (web search unavailable, answering without it: {e.message})")
                return self._ask(model, system, messages, use_web=False)
            raise
        # When web search is used, the answer arrives in several text blocks; join them.
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def is_out_of_credits(e):
    return isinstance(e, anthropic.BadRequestError) and "credit balance" in str(e).lower()


def is_fatal(e):
    """Errors that retrying or falling back can't fix."""
    return is_out_of_credits(e) or isinstance(e, anthropic.AuthenticationError)


def spoken_error(e):
    """What Leo says out loud when a Claude call fails."""
    if is_out_of_credits(e):
        return ("I can't answer because your Anthropic account is out of credits. "
                "Add more at console dot anthropic dot com, under Billing.")
    if isinstance(e, anthropic.AuthenticationError):
        return "Your Anthropic API key was rejected. Check that it's set correctly."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "Your Anthropic API key doesn't have permission to use that model."
    if isinstance(e, anthropic.RateLimitError):
        return "I'm being rate limited right now. Try again in a moment."
    if isinstance(e, anthropic.APIConnectionError):
        return "I couldn't reach Claude. Check your internet connection."
    return "Sorry, I couldn't reach Claude just now."


def speakable(text):
    """Strip anything that sounds bad when read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)     # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>|]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------- voice ---

class Voice:
    """Text-to-speech.

    engine="kokoro": local neural voice (natural sounding, ~120 MB one-time download).
    engine="system": macOS `say` on Mac, pyttsx3 (Windows/Linux system voices) elsewhere.
    """

    def __init__(self, voice_name=None, engine="system"):
        self.kokoro_mode = engine == "kokoro"
        self.mac = platform.system() == "Darwin"
        self.voice_name = voice_name
        self._cache = {}                      # short phrases (the greeting) are synthesized once
        if self.kokoro_mode:
            self.voice_name = voice_name or KOKORO_DEFAULT_VOICE
            self._load_kokoro()
        elif not self.mac:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 185)
            if voice_name:
                for v in self.engine.getProperty("voices"):
                    if voice_name.lower() in v.name.lower():
                        self.engine.setProperty("voice", v.id)
                        break

    def _load_kokoro(self):
        from kokoro_onnx import Kokoro
        APP_DIR.mkdir(parents=True, exist_ok=True)
        for name in KOKORO_FILES:
            path = APP_DIR / name
            if not path.exists():
                print(f"Downloading voice model {name} (one time only)...")
                tmp = path.with_suffix(path.suffix + ".part")
                urlretrieve(KOKORO_BASE + name, tmp)
                tmp.replace(path)
        self.kokoro = Kokoro(str(APP_DIR / KOKORO_FILES[0]), str(APP_DIR / KOKORO_FILES[1]))

    def _synthesize(self, text):
        return self.kokoro.create(text, voice=self.voice_name, speed=1.05, lang="en-us")

    def say(self, text, cache=False):
        if not text:
            return
        if self.kokoro_mode:
            import sounddevice as sd
            if cache and text in self._cache:
                samples, rate = self._cache[text]
            else:
                samples, rate = self._synthesize(text)
                if cache:
                    self._cache[text] = (samples, rate)
            sd.play(samples, rate)
            sd.wait()
        elif self.mac:
            cmd = ["say"] + (["-v", self.voice_name] if self.voice_name else []) + [text]
            subprocess.run(cmd)
        else:
            self.engine.say(text)
            self.engine.runAndWait()

    def list_voices(self):
        if self.kokoro_mode:
            print("\n".join(sorted(self.kokoro.get_voices())))
        elif self.mac:
            subprocess.run(["say", "-v", "?"])
        else:
            for v in self.engine.getProperty("voices"):
                print(v.name)


# --------------------------------------------------------------- listening ---

class Ears:
    def __init__(self):
        import numpy as np
        import sounddevice as sd
        self.np, self.sd = np, sd
        self.q = queue.Queue()
        self.muted = False
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=CHUNK, dtype="int16",
            channels=1, callback=self._on_audio,
        )
        self.stream.start()
        self.threshold = self._calibrate()

        print("Loading speech-to-text model (first run downloads it)...")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    def _on_audio(self, data, frames, t, status):
        if not self.muted:
            self.q.put(bytes(data))

    def _rms(self, chunk):
        a = self.np.frombuffer(chunk, dtype=self.np.int16).astype(self.np.float32)
        return float(self.np.sqrt(self.np.mean(a * a))) if a.size else 0.0

    def _calibrate(self):
        print("Measuring background noise, stay quiet for a second...")
        levels = [self._rms(self.q.get()) for _ in range(10)]
        ambient = sum(levels) / len(levels)
        threshold = max(ambient * 2.5, 300.0)
        dbg(f"background noise level {ambient:.0f}, speech threshold {threshold:.0f}")
        return threshold

    def flush(self):
        while not self.q.empty():
            self.q.get_nowait()

    def mute(self, on):
        self.muted = on
        if not on:
            self.flush()

    def beep(self, freq=880, ms=120):
        t = self.np.linspace(0, ms / 1000, int(22050 * ms / 1000), False)
        tone = 0.25 * self.np.sin(2 * self.np.pi * freq * t)
        fade = self.np.minimum(1, self.np.minimum(t, t[::-1]) * 200)
        self.mute(True)
        self.sd.play((tone * fade).astype(self.np.float32), 22050)
        self.sd.wait()
        self.mute(False)

    def wait_for_wake_word(self):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        if not hasattr(self, "vosk"):
            self.vosk = Model(str(ensure_vosk_model()))
        rec = KaldiRecognizer(self.vosk, SAMPLE_RATE, json.dumps(WAKE_PHRASES + ["[unk]"]))
        self.flush()
        while True:
            chunk = self.q.get()
            if rec.AcceptWaveform(chunk):
                heard = json.loads(rec.Result()).get("text", "")
            else:
                heard = json.loads(rec.PartialResult()).get("partial", "")
            if any(p in heard for p in WAKE_PHRASES):
                dbg(f"wake word heard: {heard!r}")
                return

    def record_question(self):
        """Record from now until the speaker goes quiet. Returns raw audio bytes or None."""
        self.flush()
        frames, started, quiet, elapsed = [], False, 0.0, 0.0
        step = CHUNK / SAMPLE_RATE
        while elapsed < MAX_QUESTION_SECONDS:
            chunk = self.q.get()
            elapsed += step
            loud = self._rms(chunk) > self.threshold
            if not started:
                if loud:
                    started = True
                    frames.append(chunk)
                elif elapsed > WAIT_FOR_SPEECH_SECONDS:
                    dbg(f"no speech within {WAIT_FOR_SPEECH_SECONDS}s (threshold {self.threshold:.0f})")
                    return None
                continue
            frames.append(chunk)
            quiet = 0.0 if loud else quiet + step
            if quiet >= SILENCE_TO_STOP:
                break
        dbg(f"recorded {len(frames) * step:.1f}s of audio, ended by "
            f"{'silence' if quiet >= SILENCE_TO_STOP else 'the time limit'}")
        return b"".join(frames)

    def transcribe(self, audio):
        a = self.np.frombuffer(audio, dtype=self.np.int16).astype(self.np.float32) / 32768.0
        t = time.time()
        segments, _ = self.whisper.transcribe(a, language="en", beam_size=1, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()
        dbg(f"transcribed in {time.time() - t:.2f}s: {text!r}")
        return text


def ensure_vosk_model():
    if VOSK_DIR.exists():
        return VOSK_DIR
    APP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = APP_DIR / "vosk-model.zip"
    print("Downloading wake-word model (about 40 MB, one time only)...")
    urlretrieve(VOSK_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(APP_DIR)
    zip_path.unlink()
    return VOSK_DIR


# -------------------------------------------------------------------- main ---

def handle(question, brain, voice, ears, announce):
    print(f"\nYou: {question}")
    started = time.time()
    try:
        route = brain.route(question)
        tier = route.get("model", "sonnet")
        if tier not in MODELS:
            tier = "sonnet"
        web = " + web" if route.get("needs_web") and brain.allow_web else ""
        print(f"  -> {tier}{web}  ({route.get('reason', '')})")

        if announce and ANNOUNCEMENTS[tier]:
            speak(ANNOUNCEMENTS[tier], voice, ears)

        started = time.time()
        reply = brain.answer(question, tier, route.get("needs_web", False))
    except anthropic.APIError as e:
        print(f"  error: {e}")
        dbg(f"API error: {type(e).__name__}, status={getattr(e, 'status_code', None)}, "
            f"request_id={getattr(e, 'request_id', None)}")
        reply = spoken_error(e)
    print(f"Claude ({time.time() - started:.1f}s): {reply}")
    t = time.time()
    speak(speakable(reply), voice, ears)
    dbg(f"spoken in {time.time() - t:.1f}s")


def speak(text, voice, ears, cache=False):
    if ears:
        ears.mute(True)
    voice.say(text, cache=cache)
    if ears:
        time.sleep(0.2)
        ears.mute(False)


def main():
    p = argparse.ArgumentParser(description="Hey Leo voice assistant")
    p.add_argument("--ptt", action="store_true", help="push-to-talk instead of wake word")
    p.add_argument("--text", action="store_true", help="type questions instead of speaking")
    p.add_argument("--announce", action="store_true", help="say when a bigger model is thinking")
    p.add_argument("--no-web", action="store_true", help="disable web search")
    p.add_argument("--voice", help="text-to-speech voice name (kokoro: e.g. af_heart, am_michael)")
    p.add_argument("--tts", choices=["system", "kokoro"], default="system",
                   help="speech engine: system voices, or kokoro (natural, local)")
    p.add_argument("--name", default=os.environ.get("HEY_CLAUDE_NAME"),
                   help='your name; after the wake word it says "Yes, <name>" (or set HEY_CLAUDE_NAME)')
    p.add_argument("--list-voices", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help=f"show timings, token use, wake-word and web-search details; also logs to {LOG_FILE}")
    p.add_argument("--test-voice", action="store_true",
                   help="just speak a sample and exit (no API key or microphone needed)")
    args = p.parse_args()

    if args.debug:
        enable_debug()
        dbg(f"python {platform.python_version()} on {platform.system()}; "
            f"mode={'text' if args.text else 'ptt' if args.ptt else 'wake word'}, tts={args.tts}, "
            f"voice={args.voice}, web={'off' if args.no_web else 'on'}, whisper={WHISPER_SIZE}")
    voice = Voice(args.voice, args.tts)
    if args.list_voices:
        voice.list_voices()
        return
    if args.test_voice:
        who = f", {args.name}" if args.name else ""
        for line in (f"Yes{who}.",
                     f"Hi, I'm {BOT_NAME}. This is how I sound when I read an answer out loud.",
                     "Cats purr by rapidly twitching the muscles in their voice box. "
                     "It's a bit like a tiny motor, and it usually means they're content."):
            print(line)
            voice.say(line)
        return
    brain = Brain(allow_web=not args.no_web)

    if args.text:
        print("Type a question (Ctrl+C to quit).")
        while True:
            q = input("\n> ").strip()
            if q:
                handle(q, brain, voice, None, args.announce)

    greeting = f"Yes, {args.name}." if args.name else "Yes?"
    if voice.kokoro_mode:
        voice._cache[greeting] = voice._synthesize(greeting)   # pre-render so the reply is instant
    ears = Ears()
    if not args.ptt:
        ensure_vosk_model()
    print("\nReady." + (" Press Enter, then ask." if args.ptt else ' Say "Hey Leo".'))
    print("Ctrl+C to quit.")

    while True:
        if args.ptt:
            input()
        else:
            ears.wait_for_wake_word()
            speak(greeting, voice, ears, cache=True)
        ears.beep(880)
        print("Listening...")
        audio = ears.record_question()
        ears.beep(660, 80)
        if not audio:
            print("  (didn't hear anything)")
            continue
        question = ears.transcribe(audio)
        if len(question) < 2:
            print("  (couldn't make that out)")
            continue
        handle(question, brain, voice, ears, args.announce)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception:
        dbg("crashed:\n" + traceback.format_exc())
        raise
