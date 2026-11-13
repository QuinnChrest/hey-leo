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
  --no-log       don't save questions and answers (normally kept in ~/.hey-claude/conversations/)
  --debug        show timings, token use, wake-word and web-search details (also logged to a file)
"""

import argparse
import asyncio
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import anthropic

import viz

# ---------------------------------------------------------------- settings ---

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
ROUTER_MODEL = MODELS["haiku"]

BOT_NAME = "Leo"
WAKE_PHRASES = ["hey leo"]
# Similar-sounding phrases. The wake-word recognizer can only output words it is given, so
# without these it forces "hello" or "hey you" into "hey leo". Add any phrase that triggers it by mistake.
WAKE_DECOYS = ["hello", "hey", "okay", "leo", "hey lee", "hey leon", "hey you", "hey there",
               "hey google", "hey claude", "hey siri", "hey alexa", "hello leo", "yo leo"]
STOP_HOTKEY = "<ctrl>+<alt>+s"   # global: works from any window. Stops Leo mid-sentence.
GREETING_PAUSE = 0.3             # seconds between hearing the wake word and replying "Yes, Quinn"
WHISPER_SIZE = "base.en"          # "tiny.en" is faster, "small.en" is more accurate
SAMPLE_RATE = 16000
CHUNK = 1600                      # 100 ms of audio per chunk
SILENCE_TO_STOP = 1.2             # seconds of quiet that ends your question
MAX_QUESTION_SECONDS = 20
WAIT_FOR_SPEECH_SECONDS = 6
MEMORY_MINUTES = 5                # follow-up questions remember context this long
MEMORY_TURNS = 6
# Answers think before they speak, and thinking counts toward max_tokens, so leave plenty of
# room: a low cap gives an empty answer. ANSWER_SYSTEM is what keeps the spoken part short.
ANSWER_MAX_TOKENS = 4096
# How hard each tier thinks (low | medium | high | xhigh | max). Lower is faster to answer.
EFFORT = {"haiku": "low", "sonnet": "low", "opus": "medium"}
FOLLOWUP_WAIT_SECONDS = 5         # after Leo asks you something, how long the mic waits for a reply
# Saying one of these as your whole question (after "Hey Leo") or reply (to a follow-up question)
# cancels without asking Claude. Only the whole utterance counts, so "no, I have chicken" or
# "how do I stop a leak" is still answered.
STOP_PHRASES = ["stop", "stop listening", "cancel", "never mind", "nevermind", "forget it",
                "no", "nope", "nah", "no thanks", "no thank you", "nothing", "that's all", "that's it",
                "i'm done", "i'm good", "we're done", "all done", "i'm all set", "goodbye", "bye"]
STOP_FILLERS = ["leo", "please", "thanks", "thank you", "okay", "ok"]   # ignored around a stop phrase

APP_DIR = Path.home() / ".hey-claude"
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_DIR = APP_DIR / "vosk-model-small-en-us-0.15"

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ["kokoro-v1.0.int8.onnx", "voices-v1.0.bin"]   # about 120 MB total
KOKORO_DEFAULT_VOICE = "am_michael"   # others to try: am_onyx, am_liam, bm_daniel

VIZ = viz.hub                     # fed by Ears, Voice and handle; drawn by --viz terminal|pygame

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


_stop = threading.Event()        # set by the stop hotkey; cleared at the start of each turn


def _on_stop():
    _stop.set()
    VIZ.flash_stop()


def start_stop_hotkey():
    """Listen for STOP_HOTKEY from any window (the mic is muted while Leo talks, so
    stopping by voice isn't possible)."""
    try:
        from pynput import keyboard
        hk = keyboard.GlobalHotKeys({STOP_HOTKEY: _on_stop})
        hk.daemon = True
        hk.start()
        key = STOP_HOTKEY.replace("<", "").replace(">", "").replace("+", " + ").title()
        print(f"Press {key} to stop Leo if it speaks by mistake.")
    except Exception as e:
        print(f"  (stop hotkey unavailable: {e})")


CONV_DIR = APP_DIR / "conversations"   # one JSON Lines file per month: 2026-11.jsonl
LOG_CONVERSATIONS = True               # turned off by --no-log
RUN_MODE = "wake"                      # "wake", "ptt" or "text"; recorded in each log entry


def new_trace():
    return {"calls": [], "searches": [], "history_turns": None}


def log_turn(rec, brain):
    """Append one finished question/answer to this month's JSON Lines file."""
    if not LOG_CONVERSATIONS:
        return
    calls = brain.trace["calls"]
    rec["answer_words"] = len(rec["answer"].split()) if rec.get("answer") else None
    rec["web_searches"] = brain.trace["searches"]
    rec["history_turns"] = brain.trace["history_turns"]
    rec["calls"] = calls
    rec["totals"] = {
        "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
        "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
    }
    rec["truncated"] = any(c["stop_reason"] == "max_tokens" for c in calls)   # answer cut off by the limit
    try:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        path = CONV_DIR / f"{time.strftime('%Y-%m')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"  (couldn't save conversation log: {e})")


def usage_str(resp):
    u = getattr(resp, "usage", None)
    return f"{getattr(u, 'input_tokens', '?')} in / {getattr(u, 'output_tokens', '?')} out tokens"


ANSWER_SYSTEM = """You are Leo, a voice assistant. Everything you write is read aloud, \
and a listener can't skim or skip ahead, so every extra sentence costs them time.

How much to say:
- Default to one or two sentences, about ten seconds of speech. Lead with the answer.
- Answer exactly what was asked. Leave out background, caveats, alternatives and tips \
nobody asked for.
- For broad questions (advice, how-to, recommendations, "tell me about"), give only the \
one or two things that matter most, then offer more in a few words, like "Want the steps?" \
You remember the conversation, so they can simply ask for the rest.
- Go longer, up to about sixty words, only when they ask for detail or a short answer \
would be wrong.
- No warm-up or wrap-up: don't praise the question, restate it, sum up, or say \
"let me know if".

Keeping it a conversation:
- Talk like a friend in the room, not a reference book.
- When you end with a question, the mic opens right away so they can answer without \
the wake word. Most answers should just end, with no question. Ask back only when you \
need something from them to answer well, or when you held back a part they'd clearly \
want, like the steps of a recipe. Never tack "want to know more?" onto an answer that \
is already complete, and ask at most one question.
- If the question is ambiguous but one reading is clearly most likely, answer that. \
Ask back only when the answer really depends on something you don't know.
- If they decline or say thanks, reply in a few words and don't ask anything.

Format: plain spoken sentences. No markdown, lists, headings, emoji, URLs or code. \
Write numbers and units the way a person says them. If you search the web, don't \
mention sources; just give the answer.

Examples of the right length:
Q: What's the capital of Australia?
A: Canberra.
Q: Why is the sky blue?
A: Air scatters blue light much more than red, so blue reaches your eyes from every \
part of the sky.
Q: How do I brew the perfect cup of coffee?
A: Grind fresh beans right before brewing, use about one gram of coffee for every \
sixteen of water, and pour water just off the boil. Want tips for your brewer?
Q: How can my buddy get a girlfriend?
A: Get him somewhere he'll see the same people every week, like a rec league or a \
class, since that's how most couples meet. Is he shy, or just not getting out much?
Q: No, that's all, thanks.
A: Anytime.

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

Also decide if answering needs the web. Say yes for current information (anything \
about today, recent events, prices, scores, weather, hours, or things that change), \
and for specific details about a particular video game, TV show, movie, book, app or \
product (bosses, unlocks, builds, plot points, episode details, settings, specs), \
which models often misremember. Say no for general knowledge that is well known and \
stable."""

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
        self.trace = new_trace()     # per-question stats, reset by handle() and saved by log_turn()

    def _record_call(self, role, model, started, resp):
        u = getattr(resp, "usage", None)
        self.trace["calls"].append({
            "role": role,
            "model": model,
            "seconds": round(time.time() - started, 2),
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "stop_reason": resp.stop_reason,
        })

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
            self._record_call("router", ROUTER_MODEL, t, resp)
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
        self.trace["history_turns"] = len(self.history) // 2
        messages = self.history + [{"role": "user", "content": question}]
        system = ANSWER_SYSTEM.format(now=time.strftime("%A, %B %d, %Y, %I:%M %p"))
        use_web = self.allow_web and needs_web

        text = self._ask(MODELS[tier], system, messages, use_web, EFFORT[tier])
        if not text:     # e.g. it spent the whole token cap thinking; don't go silent or save an empty turn
            dbg(f"empty answer from {tier}")
            return "Sorry, I lost my train of thought. Could you ask that again?"

        self.history = (messages + [{"role": "assistant", "content": text}])[-MEMORY_TURNS * 2:]
        self.last_turn = time.time()
        return text

    def _ask(self, model, system, messages, use_web, effort):
        kwargs = dict(model=model, max_tokens=ANSWER_MAX_TOKENS, system=system,
                      output_config={"effort": effort})
        if use_web:
            kwargs["tools"] = [WEB_SEARCH_TOOL]
        convo = list(messages)
        try:
            for _ in range(4):  # web search can pause a long turn; continue it
                t = time.time()
                resp = self.client.messages.create(messages=convo, **kwargs)
                dbg(f"{model}: {time.time() - t:.2f}s, stop={resp.stop_reason}, {usage_str(resp)}")
                self._record_call("answer", model, t, resp)
                for b in resp.content:
                    if b.type == "server_tool_use":
                        query = getattr(b, "input", {}).get("query")
                        self.trace["searches"].append(query)
                        dbg(f"web search: {query!r}")
                if resp.stop_reason != "pause_turn":
                    break
                convo = convo + [{"role": "assistant", "content": resp.content}]
        except anthropic.BadRequestError as e:
            if use_web and not is_fatal(e):
                print(f"  (web search unavailable, answering without it: {e.message})")
                return self._ask(model, system, messages, False, effort)
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

    def __init__(self, voice_name=None, engine="system", stream=False):
        self.kokoro_mode = engine == "kokoro"
        self.stream_tts = stream and self.kokoro_mode   # speak sentence by sentence while the rest is synthesized
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
            self.engine.connect("started-word", lambda name, location, length:
                                self.engine.stop() if _stop.is_set() else None)
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
        """Speak text. Returns True if the stop hotkey cut it short (or had already been pressed)."""
        try:
            return self._say(text, cache)
        finally:
            VIZ.voice_end()

    def _say(self, text, cache):
        if not text:
            return False
        if _stop.is_set():
            return True
        if self.kokoro_mode:
            if self.stream_tts and not cache:
                return self._say_stream(text)
            import sounddevice as sd
            if cache and text in self._cache:
                samples, rate = self._cache[text]
            else:
                samples, rate = self._synthesize(text)
                if cache:
                    self._cache[text] = (samples, rate)
            if _stop.is_set():
                return True
            VIZ.voice_start(samples, rate)
            sd.play(samples, rate)
            end = time.time() + len(samples) / rate
            while time.time() < end + 0.1:
                if _stop.is_set():
                    sd.stop()
                    return True
                time.sleep(0.03)
            sd.wait()
        elif self.mac:
            VIZ.voice_start()
            cmd = ["say"] + (["-v", self.voice_name] if self.voice_name else []) + [text]
            proc = subprocess.Popen(cmd)
            while proc.poll() is None:
                if _stop.is_set():
                    proc.terminate()
                    return True
                time.sleep(0.03)
        else:
            VIZ.voice_start()
            self.engine.say(text)
            self.engine.runAndWait()      # the started-word callback stops it if _stop is set
        return _stop.is_set()

    def _say_stream(self, text):
        """Kokoro, streamed: a background thread synthesizes sentence by sentence while the
        previous sentence plays, so speech starts after the first one instead of after all of them."""
        import sounddevice as sd
        chunks = queue.Queue(maxsize=2)
        cancel = threading.Event()

        def put(item):
            while not cancel.is_set():
                try:
                    chunks.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        async def synthesize():
            gen = self.kokoro.create_stream(text, voice=self.voice_name, speed=1.05, lang="en-us")
            try:
                async for samples, rate in gen:
                    put((samples, rate))
                    if cancel.is_set():
                        break
            finally:
                await gen.aclose()

        def produce():
            try:
                asyncio.run(synthesize())
            except Exception as e:
                put(e)
            put(None)

        threading.Thread(target=produce, daemon=True).start()
        stream, scheduled_end, t_start = None, 0.0, time.time()
        try:
            while True:
                try:
                    item = chunks.get(timeout=0.05)
                except queue.Empty:
                    if _stop.is_set():
                        return True
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                samples, rate = item
                if stream is None:
                    stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
                    stream.start()
                    dbg(f"first sentence ready after {time.time() - t_start:.2f}s")
                starts = max(scheduled_end, time.time() + (stream.latency if not scheduled_end else 0))
                scheduled_end = starts + len(samples) / rate
                VIZ.voice_chunk(samples, rate, starts)
                step = max(1, rate // 20)
                for i in range(0, len(samples), step):
                    if _stop.is_set():
                        stream.abort()
                        return True
                    stream.write(samples[i:i + step].reshape(-1, 1))
            if stream is not None:
                stream.stop()           # waits for the tail to finish playing
            return _stop.is_set()
        finally:
            cancel.set()
            if stream is not None:
                stream.close()

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
        VIZ.push_mic(data, SAMPLE_RATE, self.muted)
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
        rec = KaldiRecognizer(self.vosk, SAMPLE_RATE, json.dumps(WAKE_PHRASES + WAKE_DECOYS + ["[unk]"]))
        self.flush()
        while True:
            chunk = self.q.get()
            if not rec.AcceptWaveform(chunk):
                continue    # judge finished utterances only; partial results flicker and cause false triggers
            heard = json.loads(rec.Result()).get("text", "")
            if is_wake(heard):
                dbg(f"wake word heard: {heard!r}")
                return
            if heard and heard != "[unk]":
                dbg(f"ignored (not the wake phrase): {heard!r}")

    def record_question(self, wait=WAIT_FOR_SPEECH_SECONDS):
        """Record from now until the speaker goes quiet. Returns raw audio bytes, or None if
        nobody starts talking within `wait` seconds."""
        self.flush()
        frames, started, quiet, elapsed = [], False, 0.0, 0.0
        step = CHUNK / SAMPLE_RATE
        while elapsed < MAX_QUESTION_SECONDS:
            chunk = self.q.get()
            if _stop.is_set():
                return None
            elapsed += step
            loud = self._rms(chunk) > self.threshold
            if not started:
                if loud:
                    started = True
                    frames.append(chunk)
                elif elapsed > wait:
                    dbg(f"no speech within {wait}s (threshold {self.threshold:.0f})")
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


def is_stop_reply(text):
    """True if a reply to Leo's follow-up question means "we're done": a stop phrase, or just
    thanks, with optional fillers like "okay" or "Leo" around it. "Okay" alone means yes, so
    it doesn't count."""
    t = re.sub(r"[^a-z' ]", " ", text.lower().replace("’", "'"))
    t = " ".join(t.split())
    fillers = sorted(STOP_FILLERS, key=len, reverse=True)    # "thank you" before "thanks"
    trimmed = True
    while trimmed and t:
        trimmed = False
        for f in fillers:
            if t == f:
                t, trimmed = "", True
            elif t.startswith(f + " "):
                t, trimmed = t[len(f) + 1:], True
            elif t.endswith(" " + f):
                t, trimmed = t[:-len(f) - 1], True
    if not t:
        return "thank" in text.lower()
    return t in STOP_PHRASES


def is_wake(heard):
    """True if the utterance starts with the wake phrase. Speech before it (a conversation
    in the room) doesn't count, and neither does a longer word like "hey leon"."""
    return any((heard + " ").startswith(p + " ") for p in WAKE_PHRASES)


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

def asks_back(reply):
    """True if Leo's answer ends with a question, so the mic should open for the reply."""
    return speakable(reply).rstrip(" \"'”’)").endswith("?")


def handle(question, brain, voice, ears, announce, followup=False):
    """Answer one question. Returns True if Leo asked something back and finished saying it,
    so the caller should listen for a reply without the wake word."""
    print(f"\nYou: {question}")
    VIZ.begin_turn(question)
    brain.trace = new_trace()
    rec = {"v": 1, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": RUN_MODE, "question": question,
           "followup": followup, "route": None, "tier": None, "answer": None, "stopped": False,
           "error": None, "seconds": {}}
    if _stop.is_set():
        print("  (stopped)")
        rec["stopped"] = True
        log_turn(rec, brain)
        return False
    started = time.time()
    try:
        t = time.time()
        route = brain.route(question)
        rec["seconds"]["route"] = round(time.time() - t, 2)
        rec["route"] = route
        tier = route.get("model", "sonnet")
        if tier not in MODELS:
            tier = "sonnet"
        rec["tier"] = tier
        web = " + web" if route.get("needs_web") and brain.allow_web else ""
        print(f"  -> {tier}{web}  ({route.get('reason', '')})")
        VIZ.set_route(tier, bool(web))
        VIZ.set_stage("answering")

        if announce and ANNOUNCEMENTS[tier]:
            speak(ANNOUNCEMENTS[tier], voice, ears)
        if _stop.is_set():          # stopped while routing: don't spend credits on an answer
            print("  (stopped)")
            rec["stopped"] = True
            log_turn(rec, brain)
            return False

        started = time.time()
        reply = brain.answer(question, tier, route.get("needs_web", False))
        rec["seconds"]["answer"] = round(time.time() - started, 2)
        rec["answer"] = reply
    except anthropic.APIError as e:
        print(f"  error: {e}")
        dbg(f"API error: {type(e).__name__}, status={getattr(e, 'status_code', None)}, "
            f"request_id={getattr(e, 'request_id', None)}")
        rec["error"] = {"type": type(e).__name__, "status": getattr(e, "status_code", None),
                        "message": str(e)[:300]}
        reply = spoken_error(e)
    print(f"Claude ({time.time() - started:.1f}s): {reply}")
    t = time.time()
    VIZ.set_answer(speakable(reply))
    rec["stopped"] = speak(speakable(reply), voice, ears)
    if rec["stopped"]:
        print("  (stopped)")
    rec["seconds"]["speak"] = round(time.time() - t, 2)
    dbg(f"spoken in {rec['seconds']['speak']:.1f}s")
    log_turn(rec, brain)
    return rec["error"] is None and not rec["stopped"] and asks_back(reply)


def speak(text, voice, ears, cache=False):
    """Speak with the mic muted. Returns True if the user stopped it."""
    if ears:
        ears.mute(True)
    stopped = voice.say(text, cache=cache)
    if ears:
        time.sleep(0.2)
        ears.mute(False)
    return stopped


def main():
    global LOG_CONVERSATIONS, RUN_MODE
    p = argparse.ArgumentParser(description="Hey Leo voice assistant")
    p.add_argument("--ptt", action="store_true", help="push-to-talk instead of wake word")
    p.add_argument("--text", action="store_true", help="type questions instead of speaking")
    p.add_argument("--announce", action="store_true", help="say when a bigger model is thinking")
    p.add_argument("--no-web", action="store_true", help="disable web search")
    p.add_argument("--voice", help="text-to-speech voice name (kokoro: e.g. af_heart, am_michael)")
    p.add_argument("--tts", choices=["system", "kokoro"], default="system",
                   help="speech engine: system voices, or kokoro (natural, local)")
    p.add_argument("--stream-tts", action="store_true",
                   help="with --tts kokoro: start speaking after the first sentence instead of waiting "
                        "for the whole answer to be synthesized")
    p.add_argument("--name", default=os.environ.get("HEY_CLAUDE_NAME"),
                   help='your name; after the wake word it says "Yes, <name>" (or set HEY_CLAUDE_NAME)')
    p.add_argument("--list-voices", action="store_true")
    p.add_argument("--no-log", action="store_true",
                   help=f"don't save questions and answers to {CONV_DIR}")
    p.add_argument("--debug", action="store_true",
                   help=f"show timings, token use, wake-word and web-search details; also logs to {LOG_FILE}")
    p.add_argument("--viz", choices=["none", "terminal", "pygame"], default="none",
                   help="live visuals: waveforms of you and Leo, spectrum, pipeline stage "
                        "(terminal: in this console; pygame: its own window)")
    p.add_argument("--tray", action="store_true",
                   help="live in the system tray; when you say the wake word Leo goes fullscreen "
                        "(pygame visuals) and hides again when the conversation is over")
    p.add_argument("--monitor", type=int, default=1,
                   help="with --tray: which monitor Leo appears on, 1 = first (also changeable from the tray menu)")
    p.add_argument("--hide-after", type=float, default=viz.TRAY_LINGER, metavar="SECONDS",
                   help=f"with --tray: how long Leo stays on screen after the conversation ends "
                        f"(default {viz.TRAY_LINGER:g})")
    p.add_argument("--test-voice", action="store_true",
                   help="just speak a sample and exit (no API key or microphone needed)")
    args = p.parse_args()
    if args.stream_tts and args.tts != "kokoro":
        p.error("--stream-tts needs --tts kokoro")
    if args.tray:
        if args.viz == "terminal":
            p.error("--tray uses the pygame window, so it can't be combined with --viz terminal")
        if args.monitor < 1:
            p.error("--monitor counts from 1")
        if args.hide_after < 0:
            p.error("--hide-after can't be negative")
        args.viz = "pygame"
        if sys.stdout is None:          # pythonw: no console, so keep the output in a file
            APP_DIR.mkdir(parents=True, exist_ok=True)
            sys.stdout = sys.stderr = open(APP_DIR / "tray.log", "a", encoding="utf-8", buffering=1)
    elif args.monitor != 1 or args.hide_after != viz.TRAY_LINGER:
        p.error("--monitor and --hide-after need --tray")
    if args.viz == "terminal" and args.text:
        p.error("--viz terminal takes over the screen, so it can't be combined with --text; use --viz pygame")

    LOG_CONVERSATIONS = not args.no_log
    RUN_MODE = "text" if args.text else "ptt" if args.ptt else "wake"
    if args.debug:
        enable_debug()
        dbg(f"python {platform.python_version()} on {platform.system()}; "
            f"mode={'text' if args.text else 'ptt' if args.ptt else 'wake word'}, tts={args.tts}, "
            f"voice={args.voice}, web={'off' if args.no_web else 'on'}, whisper={WHISPER_SIZE}")
    if args.viz == "none" or args.list_voices or args.test_voice:
        session(args)
    else:
        # The assistant runs in a worker thread so the visuals can own the main thread.
        viz.run(args.viz, lambda: session(args), tray_monitor=args.monitor - 1 if args.tray else None,
                tray_linger=args.hide_after)


def session(args):
    voice = Voice(args.voice, args.tts, stream=args.stream_tts)
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
    start_stop_hotkey()

    if args.text:
        print("Type a question (Ctrl+C to quit).")
        while True:
            q = input("\n> ").strip()
            if q:
                _stop.clear()
                handle(q, brain, voice, None, args.announce)
                VIZ.set_stage("idle")

    greeting = f"Yes, {args.name}." if args.name else "Yes?"
    if voice.kokoro_mode:
        voice._cache[greeting] = voice._synthesize(greeting)   # pre-render so the reply is instant
    ears = Ears()
    if not args.ptt:
        ensure_vosk_model()
    print("\nReady." + (" Press Enter, then ask." if args.ptt else ' Say "Hey Leo".'))
    print("Ctrl+C to quit.")

    followup = False     # Leo just asked something: listen for the reply, no wake word needed
    while True:
        if followup:
            _stop.clear()
            ears.beep(880)
            print("Listening for your reply...")
            VIZ.set_stage("listening")
            audio = ears.record_question(wait=FOLLOWUP_WAIT_SECONDS)
            if not audio:          # no reply: quietly go back to waiting for the wake word
                dbg("no reply to the follow-up question")
                followup = False
                continue
        else:
            VIZ.set_stage("idle")
            if args.ptt:
                input()
                _stop.clear()
            else:
                ears.wait_for_wake_word()
                _stop.clear()
                time.sleep(GREETING_PAUSE)
                if speak(greeting, voice, ears, cache=True):
                    print("  (stopped)")
                    continue
            ears.beep(880)
            print("Listening...")
            VIZ.set_stage("listening")
            audio = ears.record_question()
        was_followup, followup = followup, False
        VIZ.set_stage("transcribing")
        ears.beep(660, 80)
        if _stop.is_set():
            print("  (stopped)")
            continue
        if not audio:
            print("  (didn't hear anything)")
            continue
        question = ears.transcribe(audio)
        if len(question) < 2:
            print("  (couldn't make that out)")
            continue
        if is_stop_reply(question):     # "stop", "never mind", "no thanks"... after the wake word or a follow-up
            print(f"\nYou: {question}\n  ({'conversation ended' if was_followup else 'cancelled'})")
            dbg(f"stop phrase instead of a question: {question!r}")
            ears.beep(440, 120)                           # lower tone: done, back to the wake word
            continue
        followup = handle(question, brain, voice, ears, args.announce, followup=was_followup)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception:
        dbg("crashed:\n" + traceback.format_exc())
        raise
