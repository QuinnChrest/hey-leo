# Hey Leo

<img src="assets/leo-full.webp" alt="Leo, a lion professor in a tweed jacket, denim shirt, orange tie and teal glasses" width="260" align="right">

A voice assistant for your computer, like "Hey Google" but powered by Claude.

You say **"Hey Leo"** and it beeps. Then you ask your question. Claude Haiku decides which model should answer, and the answer is read out loud:

| Your question sounds like…                         | Who answers |
|----------------------------------------------------|-------------|
| "What's 15% of 80?" "Who won the game last night?"  | Haiku (fast, cheapest) |
| "Why is the sky blue?" "Should I lease or buy?"     | Sonnet |
| "Plan a 3-city trip under $2,000 with these rules…" | Opus (slowest, smartest) |

The wake word and speech-to-text both run on your computer, so only the text of your question goes to Claude. When a question needs current information (weather, scores, news), Claude searches the web.

## 1. Get an API key (5 minutes)

Your Claude.ai subscription doesn't cover this. The API is billed separately, per use.

1. Go to **console.anthropic.com** and sign up or log in.
2. Under **Billing**, add a payment method and buy some credits. $5 lasts a long time: most questions cost a fraction of a cent, and Opus questions cost a few cents.
3. Under **API Keys**, click **Create Key** and copy it. It starts with `sk-ant-`.
4. Optional: under **Settings → Privacy**, check that web search is turned on.

## 2. Install

You need **Python 3.10 or newer** (python.org).

**Mac**
```bash
cd voice-claude
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-...your key..."
```
The first time it runs, macOS will ask whether Terminal can use the microphone. Click **Allow**.

**Windows** (in PowerShell)
```powershell
cd voice-claude
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="sk-ant-...your key..."
```

**Linux**
```bash
sudo apt install portaudio19-dev espeak-ng
cd voice-claude && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-...your key..."
```

To keep the key set permanently, add the `export` line to `~/.zshrc` on a Mac or `~/.bashrc` on Linux. On Windows, run `setx ANTHROPIC_API_KEY "sk-ant-..."` once.

## 3. Run it

```bash
python assistant.py --text    # type questions first, to check your key works
python assistant.py           # then the real thing: say "Hey Leo"
```

The first voice run downloads two small speech models (about 190 MB in total), one time only.

**Other ways to run it**
- `--ptt`: press Enter instead of saying the wake word. Use this if the wake word is unreliable in a noisy room.
- `--announce`: says "Let me think harder…" when Opus is answering, so you know why it's taking longer.
- `--stream-tts` (with `--tts kokoro`): starts speaking after the first sentence while the rest is still being generated, so long answers begin sooner. Try it and drop the flag if it sounds choppy.
- `--no-web`: turns off web searches.
- `--viz terminal` or `--viz pygame`: live visuals. Scrolling waveforms of you (cyan) and Leo (orange), a frequency spectrum, the pipeline stage (wake, listen, transcribe, route, answer, speak), the model tier chosen, and captions. `terminal` draws in this console with no extra install (not available with `--text`); `pygame` opens its own window (Esc or Q closes it) and works with every mode, with Leo in the middle, ringed in the color of the current stage: he looks over while listening, rests his chin on his hand while thinking, and talks with his hands while speaking. With `--tts kokoro` Leo's waveform is his real audio; with the system voice it is a simulated one, since that audio can't be read.
- `--debug`: shows what's happening behind the scenes: when the wake word is heard, the noise threshold, how long recording, transcription, each Claude call and speaking take, token usage, and what Claude searched for. It's also saved to `~/.hey-claude/debug.log` (this file contains your questions and answers, and stays on your computer).
- `--tts kokoro`: a much more natural-sounding voice that runs on your computer. The first run downloads about 120 MB, one time only. Pick a voice with `--voice` (e.g. `af_heart`, `af_bella`, `am_michael`; `--tts kokoro --list-voices` shows all).
- `--name Quinn`: after "Hey Leo" it says "Yes, Quinn" before the beep. You can also set the `HEY_CLAUDE_NAME` environment variable. Without a name it says "Yes?".
- `--voice Samantha`: chooses a voice. Run `--list-voices` to see what's installed. On a Mac, better voices can be downloaded under System Settings → Accessibility → Spoken Content.

The terminal shows which model was picked and why:
```
You: why do cats purr
  -> sonnet  (needs a real explanation)
Claude (2.1s): Cats purr by rapidly twitching the muscles in their voice box...
```

**Stopping Leo:** if it wakes up by mistake or starts talking and you don't want it to, press **Ctrl + Alt + S**. It works from any window, even if you're in a game. Leo stops talking right away, skips the rest of that turn (no answer is requested if it hasn't been yet), and goes back to waiting for "Hey Leo". Change the key combination with `STOP_HOTKEY` at the top of `assistant.py`. On a Mac, you'll need to allow Terminal under System Settings → Privacy & Security → Input Monitoring.

Follow-up questions work: "How about dogs?" remembers what you just asked, for up to 5 minutes.

## Conversation log

Every question and answer is saved on your computer, one file per month, in `~/.hey-claude/conversations/` (for example `2026-11.jsonl`). Each line is one question as JSON: what you asked, which model Haiku picked and why, the answer and its word count, the web searches used, input and output tokens for every call, how long each step took, and whether the answer was stopped by you or cut off by the length limit (`"truncated": true`). Nothing is uploaded. Use `--no-log` to turn it off, and delete the files any time.

To see how long answers actually are, in PowerShell:
```powershell
Get-Content ~\.hey-claude\conversations\*.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Select-Object time, tier, answer_words, truncated, question
```

## Tweaking

Everything you'd want to change is at the top of `assistant.py`:

- `ROUTER_SYSTEM`: the rules Haiku uses to pick a model. Edit these if it sends too much to Opus, or too little.
- `ANSWER_SYSTEM`: how answers should sound (length, style).
- `WAKE_PHRASES`: add your own wake phrases.
- `WAKE_DECOYS`: similar-sounding phrases ("hello", "hey you"…) that the wake-word recognizer should recognize as *not* the wake phrase. If it wakes up to something by mistake, add that phrase here. Run with `--debug` to see what it ignored.
- `GREETING_PAUSE`: the pause between hearing "Hey Leo" and the "Yes, Quinn" reply.
- `WHISPER_SIZE`: `tiny.en` is faster, `small.en` is more accurate.
- `SILENCE_TO_STOP`: raise it if it cuts you off mid-sentence.

## Troubleshooting

- **It doesn't react to "Hey Leo".** Speak clearly and pause briefly after the wake word. Use `--ptt` in loud rooms.
- **It cuts me off, or keeps listening forever.** It measures background noise at startup, so start it in a quiet moment. Raise `SILENCE_TO_STOP` if it cuts you off.
- **"Set ANTHROPIC_API_KEY first."** The key isn't set in this terminal window. Run the `export` / `$env:` line again.
- **"web search unavailable"** It still answers, just without the web. Turn web search on in the Console (step 1.4).
