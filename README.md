# Hey Leo

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
- `--no-web`: turns off web searches.
- `--tts kokoro`: a much more natural-sounding voice that runs on your computer. The first run downloads about 120 MB, one time only. Pick a voice with `--voice` (e.g. `af_heart`, `af_bella`, `am_michael`; `--tts kokoro --list-voices` shows all).
- `--name Quinn`: after "Hey Leo" it says "Yes, Quinn" before the beep. You can also set the `HEY_CLAUDE_NAME` environment variable. Without a name it says "Yes?".
- `--voice Samantha`: chooses a voice. Run `--list-voices` to see what's installed. On a Mac, better voices can be downloaded under System Settings → Accessibility → Spoken Content.

The terminal shows which model was picked and why:
```
You: why do cats purr
  -> sonnet  (needs a real explanation)
Claude (2.1s): Cats purr by rapidly twitching the muscles in their voice box...
```

Follow-up questions work: "How about dogs?" remembers what you just asked, for up to 5 minutes.

## Tweaking

Everything you'd want to change is at the top of `assistant.py`:

- `ROUTER_SYSTEM`: the rules Haiku uses to pick a model. Edit these if it sends too much to Opus, or too little.
- `ANSWER_SYSTEM`: how answers should sound (length, style).
- `WAKE_PHRASES`: add your own wake phrases.
- `WHISPER_SIZE`: `tiny.en` is faster, `small.en` is more accurate.
- `SILENCE_TO_STOP`: raise it if it cuts you off mid-sentence.

## Troubleshooting

- **It doesn't react to "Hey Leo".** Speak clearly and pause briefly after the wake word. Use `--ptt` in loud rooms.
- **It cuts me off, or keeps listening forever.** It measures background noise at startup, so start it in a quiet moment. Raise `SILENCE_TO_STOP` if it cuts you off.
- **"Set ANTHROPIC_API_KEY first."** The key isn't set in this terminal window. Run the `export` / `$env:` line again.
- **"web search unavailable"** It still answers, just without the web. Turn web search on in the Console (step 1.4).
