"""Visualizations for Hey Leo.

`hub` is a thread-safe object that assistant.py feeds (mic audio, what Leo is playing,
pipeline stage, captions). It costs almost nothing when no renderer is attached.

Two renderers read from it: TerminalView (plain ANSI, no extra dependency) and PygameView.
`run(mode, target)` starts the assistant (`target`) in a worker thread and keeps the
renderer on the main thread, which pygame requires on some platforms.
"""
import collections
import math
import os
import shutil
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import numpy as np

PIPELINE = [("idle", "WAKE"), ("listening", "LISTEN"), ("transcribing", "TRANSCRIBE"),
            ("routing", "ROUTE"), ("answering", "ANSWER"), ("speaking", "SPEAK")]
LABELS = dict(PIPELINE)

YOU_COLOR = (80, 220, 255)
LEO_COLOR = (255, 150, 70)
STATE_COLOR = {
    "idle": (100, 120, 175),
    "listening": YOU_COLOR,
    "transcribing": (255, 210, 90),
    "routing": (255, 210, 90),
    "answering": (170, 120, 255),
    "speaking": LEO_COLOR,
}
THINKING = ("transcribing", "routing", "answering")


# --------------------------------------------------------------------- hub ---

class Hub:
    MIC_N = 8192            # samples of mic history kept
    WINDOW = 2048           # samples handed to renderers per frame

    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False            # set by run(); until then the feed methods do nothing
        self.stage = "idle"             # idle, listening, transcribing, routing, answering
        self.speaking = False           # Leo is playing audio right now
        self.tier = None
        self.web = False
        self.question = ""
        self.answer = ""
        self.stop_at = 0.0
        self.log = collections.deque(maxlen=300)    # captured console output (terminal view)
        self.mic = np.zeros(self.MIC_N, np.float32)
        self.mic_rate = 16000
        self._voice = []                # [(samples, rate, start time)] when we know the audio (kokoro)
        self._pending_answer = None     # caption shown once Leo actually starts speaking
        self._voice_t0 = 0.0            # start of speech when we don't (system TTS)

    # -- fed by the assistant --
    def push_mic(self, data, rate, muted=False):
        """One chunk of raw int16 mic audio. While muted the trace goes flat."""
        if not self.enabled:
            return
        a = (np.zeros(len(data) // 2, np.float32) if muted
             else np.frombuffer(data, np.int16).astype(np.float32) / 32768.0)
        with self.lock:
            self.mic_rate = rate
            self.mic = np.concatenate((self.mic[len(a):], a))[-self.MIC_N:]

    def voice_start(self, samples=None, rate=None):
        """Leo starts playing. Pass the samples when known (kokoro); otherwise the view fakes a voice."""
        with self.lock:
            self.speaking = True
            self._voice_t0 = time.time()
            self._voice = []
            if self._pending_answer is not None:
                self.answer, self._pending_answer = self._pending_answer, None
            if self.enabled and samples is not None:
                self._voice = [(np.asarray(samples, np.float32).reshape(-1), rate, time.time())]

    def voice_chunk(self, samples, rate, start):
        """Streamed speech: another piece of audio that will start playing at time `start`."""
        with self.lock:
            if not self.speaking:
                self.speaking, self._voice_t0 = True, start
                if self._pending_answer is not None:
                    self.answer, self._pending_answer = self._pending_answer, None
            if self.enabled:
                self._voice.append((np.asarray(samples, np.float32).reshape(-1), rate, start))

    def voice_end(self):
        with self.lock:
            self.speaking = False
            self._voice = []

    def set_stage(self, stage):
        self.stage = stage

    def begin_turn(self, question):
        with self.lock:
            self.question, self.answer, self.tier, self.web = question, "", None, False
            self._pending_answer = None
            self.stage = "routing"

    def set_route(self, tier, web):
        self.tier, self.web = tier, web

    def set_answer(self, text):
        """The caption appears when Leo starts speaking, not while the audio is still being generated."""
        self._pending_answer = text

    def flash_stop(self):
        self.stop_at = time.time()

    # -- read by the renderers --
    def frame(self):
        now = time.time()
        with self.lock:
            mic = self.mic[-self.WINDOW:].copy()
            speaking, voice, t0 = self.speaking, list(self._voice), self._voice_t0
            rate = self.mic_rate
        v_rate = 16000
        wave = np.zeros(self.WINDOW, np.float32)
        if speaking:
            playing = [v for v in voice if v[2] <= now]
            if playing:
                samples, v_rate, started = playing[-1]
                pos = int((now - started) * v_rate)
                seg = samples[max(0, pos - self.WINDOW):max(0, pos)]
                wave[self.WINDOW - len(seg):] = seg
            elif not voice:
                wave = _fake_voice(now - t0, self.WINDOW, v_rate)
        return SimpleNamespace(
            stage=self.stage, speaking=speaking, tier=self.tier, web=self.web,
            question=self.question, answer=self.answer, stop_age=now - self.stop_at,
            mic=mic, mic_rate=rate, voice=wave, voice_rate=v_rate)

    def log_tail(self, n):
        return list(self.log)[-n:]


hub = Hub()


def _fake_voice(t_now, n, rate):
    """A speech-looking signal for engines whose audio we can't see (system TTS)."""
    t = t_now - n / rate + np.arange(n) / rate
    env = np.clip(0.55 + 0.45 * np.sin(2 * np.pi * 3.3 * t) * np.sin(2 * np.pi * 1.1 * t + 1.0), 0, 1)
    env *= np.clip(t * 8, 0, 1)
    ph = 2 * np.pi * (130 * t - (25 / (2 * np.pi * 0.7)) * np.cos(2 * np.pi * 0.7 * t))
    return (env * 0.17 * (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph))).astype(np.float32)


# ----------------------------------------------------------------- helpers ---

def level(w):
    """0..1 loudness of a window, shaped to look good for speech."""
    if len(w) == 0:
        return 0.0
    return min(1.0, (float(np.sqrt(np.mean(w * w))) * 5.0) ** 0.7)


_hann = {}


def spectrum(w, rate, bands=48, lo=80.0, hi=6000.0):
    """Log-spaced frequency bands, each 0..1 (speech range)."""
    n = len(w)
    out = np.zeros(bands, np.float32)
    if n < 64 or not w.any():
        return out
    if n not in _hann:
        _hann[n] = np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(w * _hann[n])) / n
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    idx = np.searchsorted(freqs, np.geomspace(lo, min(hi, rate / 2 - 1), bands + 1))
    for i in range(bands):
        seg = mag[idx[i]:max(idx[i + 1], idx[i] + 1)]
        if seg.size:
            out[i] = seg.max()
    db = 20 * np.log10(out + 1e-6)
    return np.clip((db + 80) / 50 + np.linspace(0, 0.15, bands), 0, 1).astype(np.float32)


def ease(prev, new, attack=0.6, decay=0.15):
    """Move prev toward new: fast up, slow down. Works on floats and numpy arrays."""
    rate = np.where(np.asarray(new) > np.asarray(prev), attack, decay)
    out = prev + (new - prev) * rate
    return out if isinstance(out, np.ndarray) else float(out)


def lerp_color(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def active_stage(f):
    return "speaking" if f.speaking else f.stage


def badge(f):
    if not f.tier:
        return ""
    return f.tier.upper() + (" + WEB" if f.web else "")


# ------------------------------------------------------------ terminal view ---

class _Tee:
    """Replaces stdout/stderr while the terminal view owns the screen: output goes to hub.log."""
    encoding = "utf-8"
    errors = "replace"

    def __init__(self, log):
        self.log, self.buf = log, ""

    def write(self, s):
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            self.log.append(line.split("\r")[-1])
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("not a real file")


BLOCKS = " ▁▂▃▄▅▆▇█"


def _fg(rgb, text):
    r, g, b = (int(c) for c in rgb)
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


def _bars(values, rows):
    """Vertical bars, one column per value in 0..1, `rows` characters tall. Returns `rows` strings."""
    units = [v * rows * 8 for v in values]
    lines = []
    for r in range(rows):
        k = rows - 1 - r
        chars = []
        for u in units:
            u -= k * 8
            chars.append(BLOCKS[min(8, math.ceil(u))] if u > 0 else " ")
        lines.append("".join(chars))
    return lines


# Leo as a 20x20 pixel sprite, drawn two pixels per character cell with half blocks.
LION_COLORS = {
    "M": (150, 55, 25), "m": (215, 95, 35), "o": (245, 140, 60), "S": (225, 215, 200),
    "F": (230, 150, 70), "f": (190, 110, 45), "C": (245, 225, 190), "W": (240, 235, 225),
    "B": (60, 35, 25), "G": (40, 160, 170), "E": (240, 170, 40), "P": (30, 20, 15),
    "N": (90, 45, 35), "K": (60, 20, 20), "T": (250, 250, 245),
    "J": (80, 58, 46), "D": (60, 100, 160), "O": (235, 105, 40),
}                                   # "." is transparent
_LION_HALF = (                      # left half of the head-on pose; the right half is its mirror
    "......Mmmm", "....MmmoMm", "..MmmoommM", ".MmSSmFFFF", "MmSSmFFFFF",
    "MmmFBBBFFF", "MmmGGGGGGG", "MmmGEPEGFF", "MmmFGGGFFF", "MmmfFFFFNN",
    "MmmfFCCCCN", "MmmmfCCCCC", "MmmmfCCCKK", ".MmmmfWWWW", ".MmmmmWWWW",
    "..MmmmmWWW", "..MmmmmmWW", "JJJJMmmmDW", "JJJJJJDDDO", "JJJJJJJDDO",
)
LION_POSES = {                      # rows that differ from the head-on portrait
    "portrait": {},
    "listening": {5: "MmmFFBBBFFFFFBBBFmmM",            # glancing over to the side
                  7: "MmmGEEPGFFFFGEEPGmmM"},
    "thinking": {7: "MmmGBEPGFFFFGBEPGmmM",             # narrowed eyes, paw on chin
                 12: "MmmmfCCCKKKCCfFFfmmM", 13: ".MmmmfWWWWWfFfFfFfM.",
                 14: ".MmmmmWWWWWFFFFFFfM.", 15: "..MmmmmWWWWfFFFFfmM.",
                 16: "..MmmmmmWWWJJFFJJmM.", 17: "JJJJMmmmDWWDJJJJJJJJ"},
    "speaking": {12: "MmmmfCCKTTTTKCCfmmmM", 13: ".MmmmfWWKKKKWWfmmmM."},
}


def _lion_art(pose):
    rows = [h + h[::-1] for h in _LION_HALF]
    for i, row in LION_POSES[pose].items():
        rows[i] = row
    lines = []
    for y in range(0, len(rows), 2):
        cells = []
        for top, bot in zip(rows[y], rows[y + 1]):
            t, b = LION_COLORS.get(top), LION_COLORS.get(bot)
            if t and b:
                cells.append("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm▀" % (*t, *b))
            elif t or b:
                cells.append("\x1b[49m\x1b[38;2;%d;%d;%dm%s" % (*(t or b), "▀" if t else "▄"))
            else:
                cells.append("\x1b[0m ")
        lines.append("".join(cells) + "\x1b[0m")
    return lines


LION_ART = {pose: _lion_art(pose) for pose in LION_POSES}
LION_W = len(_LION_HALF[0]) * 2


class TerminalView:
    FPS = 24

    def __init__(self, hub):
        self.hub = hub
        self.lm = self.lv = 0.0
        self.spec = np.zeros(48, np.float32)
        self.hist_mic = collections.deque(maxlen=400)
        self.hist_voice = collections.deque(maxlen=400)

    def open(self):
        os.system("")                       # switches on ANSI escape handling in Windows consoles
        self.out = sys.stdout
        try:
            self.out.reconfigure(errors="replace")
        except Exception:
            pass
        self.saved = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _Tee(self.hub.log)
        self.out.write("\x1b[?1049h\x1b[?25l\x1b[2J")      # alternate screen, hide cursor
        self.out.flush()

    def close(self):
        sys.stdout, sys.stderr = self.saved
        self.out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        self.out.flush()
        for line in self.hub.log_tail(25):                  # keep the session output in the scrollback
            print(line)

    def loop(self, done):
        while not done():
            t = time.time()
            self.out.write(self.render(self.hub.frame()))
            self.out.flush()
            time.sleep(max(0.0, 1.0 / self.FPS - (time.time() - t)))

    def render(self, f):
        W, H = shutil.get_terminal_size((80, 24))
        W, H = max(40, W), max(12, H)
        state = active_stage(f)
        color = STATE_COLOR[state]
        self.lm = ease(self.lm, level(f.mic))
        self.lv = ease(self.lv, level(f.voice))
        self.hist_mic.append(self.lm)
        self.hist_voice.append(self.lv)
        wave, rate = (f.voice, f.voice_rate) if f.speaking else (f.mic, f.mic_rate)
        self.spec = ease(self.spec, spectrum(wave, rate, 48))

        inner = W - 2
        art = LION_ART["portrait"] if W >= 60 and H >= 26 else None    # Leo, with captions beside him
        h = H - len(art) if art else H
        lane_rows = 4 if h >= 30 else 3 if h >= 24 else 2 if h >= 17 else 1
        lines = []

        # header and pipeline
        right = badge(f)
        left = f" HEY LEO   ● {LABELS[state]}"
        lines.append("\x1b[1m" + _fg(color, left) + "\x1b[0m" + " " * max(1, W - len(left) - len(right) - 1)
                     + _fg(LEO_COLOR, right))
        strip = []
        for key, label in PIPELINE:
            if key == state:
                strip.append(f"\x1b[1;7m{_fg(color, ' ' + label + ' ')}")
            else:
                strip.append(_fg((90, 95, 110), " " + label + " "))
        lines.append(" " + _fg((90, 95, 110), "›").join(strip))
        lines.append("")

        if art:
            pose = POSE_FOR_STAGE.get(state, "portrait")
            if pose == "speaking" and self.lv < 0.2:
                pose = "portrait"       # mouth closes between syllables
            caps = [""]
            for tag, text, tcol, n in (("You  ", f.question, YOU_COLOR, 2), ("Leo  ", f.answer, LEO_COLOR, 6)):
                for i, c in enumerate(textwrap.wrap(text or "", W - LION_W - 10)[:n]):
                    caps.append((_fg(tcol, tag) if i == 0 else "     ") + c)
                if text:
                    caps.append("")
            for i, row in enumerate(LION_ART[pose]):
                lines.append(" " + row + "   " + (caps[i] if i < len(caps) else ""))
            lines.append("")

        # waveform lanes
        for name, hist, col in (("YOU", self.hist_mic, YOU_COLOR), ("LEO", self.hist_voice, LEO_COLOR)):
            vals = list(hist)[-inner:]
            vals = [0.0] * (inner - len(vals)) + vals
            lines.append(" " + _fg(col, name))
            lines.extend(" " + _fg(col, row) for row in _bars(vals, lane_rows))

        # spectrum of whoever is talking
        sp = np.interp(np.linspace(0, len(self.spec) - 1, inner), np.arange(len(self.spec)), self.spec)
        col = LEO_COLOR if f.speaking else YOU_COLOR
        lines.append(" " + _fg((130, 135, 150), "SPECTRUM"))
        lines.append(" " + _fg(col, "".join(BLOCKS[min(8, math.ceil(v * 8))] for v in sp)))

        # captions, then the console log
        room = H - len(lines) - 1
        if room >= 3 and not art:
            lines.append("")
            for tag, text, tcol in (("You  ", f.question, YOU_COLOR), ("Leo  ", f.answer, LEO_COLOR)):
                if not text:
                    continue
                cap = textwrap.wrap(text, inner - 5)[:2 if tag == "You  " else 3]
                for i, c in enumerate(cap):
                    lines.append(" " + (_fg(tcol, tag) if i == 0 else "     ") + c)
        room = H - len(lines) - 1
        if room >= 3:
            lines.append(_fg((70, 75, 90), " " + "─" * inner))
            for entry in self.hub.log_tail(min(5, room - 1)):
                lines.append(_fg((120, 125, 140), " " + entry[:inner]))

        return "\x1b[H" + "\x1b[K\n".join(lines[:H]) + "\x1b[K\x1b[J"


# ------------------------------------------------------------- pygame view ---

def _wrap(font, text, width, max_lines):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if font.size(trial)[0] <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + "…"
    return lines


ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# Leo's poses: image file and the circle (center x, center y, radius, in that image's pixels) cut out of it
AVATARS = {
    "portrait":  ("leo-portrait.webp", (681, 480, 360)),
    "listening": ("leo-poses.webp", (320, 315, 215)),
    "thinking":  ("leo-poses.webp", (895, 330, 225)),
    "speaking":  ("leo-poses.webp", (1495, 315, 215)),
}
POSE_FOR_STAGE = {"listening": "listening", "transcribing": "thinking", "routing": "thinking",
                  "answering": "thinking", "speaking": "speaking"}     # anything else: portrait
POSE_FADE = 0.35                    # seconds to crossfade between poses
ORB_Y = 0.4                         # height of Leo's orb, as a fraction of the window; captions sit below it
TRAY_LINGER = 5.0                   # --tray: default seconds the window stays up after the conversation ends
APPEAR = 0.45                       # --tray: seconds to fade in (and pop Leo up) when he's called
VANISH = 0.35                       # --tray: seconds to fade out when he's done


def _ease_out(x):
    return 1 - (1 - x) ** 3


def _ease_out_back(x, k=1.7):
    """0 to 1 with a small overshoot past 1 before settling."""
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def _windows_app_id():
    """Give the window its own taskbar entry (and so Leo's icon) instead of grouping it under python.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("HeyLeo.Viz")
    except Exception:
        pass


def _windows_title_bar(hwnd, bg, fg):
    """Dark title bar (Windows 10 20H1+), colored to match the window on Windows 11. Best effort."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        import ctypes
        from ctypes import wintypes
        dwm = ctypes.windll.dwmapi

        def attr(key, value):
            v = ctypes.c_uint(value)
            dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), key, ctypes.byref(v), ctypes.sizeof(v))

        colorref = lambda c: int(c[0]) | int(c[1]) << 8 | int(c[2]) << 16
        attr(20, 1)                     # DWMWA_USE_IMMERSIVE_DARK_MODE
        attr(35, colorref(bg))          # DWMWA_CAPTION_COLOR (Windows 11)
        attr(34, colorref(bg))          # DWMWA_BORDER_COLOR (Windows 11)
        attr(36, colorref(fg))          # DWMWA_TEXT_COLOR (Windows 11)
    except Exception:
        pass


def _windows_on_top(hwnd):
    """Keep the borderless --tray window above everything, taskbar included, and try to focus it."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND] + [ctypes.c_int] * 4 + [ctypes.c_uint]
        user32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)  # TOPMOST, NOSIZE|NOMOVE|SHOW
        user32.SetForegroundWindow(wintypes.HWND(hwnd))
    except Exception:
        pass


def _tray_image():
    """Leo's portrait circle for the tray icon, or a plain orange dot without it."""
    from PIL import Image, ImageDraw
    size = 64
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    try:
        file, (x, y, r) = AVATARS["portrait"]
        img = Image.open(os.path.join(ASSETS, file)).convert("RGBA").crop((x - r, y - r, x + r, y + r))
        img = img.resize((size, size), Image.LANCZOS)
    except OSError:
        img = Image.new("RGBA", (size, size), LEO_COLOR + (255,))
    img.putalpha(mask)
    return img


class PygameView:
    BG = (11, 13, 20)

    def __init__(self, hub, tray_monitor=None, tray_linger=TRAY_LINGER):
        try:
            import pygame
        except ImportError:
            sys.exit("--viz pygame needs pygame: pip install pygame")
        self.pg = pygame
        self.hub = hub
        # --tray: no window until Leo is woken, then borderless fullscreen on this display (0-based)
        self.tray = tray_monitor is not None
        self.monitor = tray_monitor or 0
        self.linger = tray_linger      # --hide-after
        self.visible = False
        self.pinned = False             # "Show Leo" in the tray menu: stay up even when idle
        self.dismissed = False          # Esc during a conversation: stay hidden until it ends
        self.last_busy = 0.0
        self.reopen = False             # monitor changed from the tray menu while showing
        self.quit = False
        self.icon = None
        self.window = None              # SDL window handle, for its opacity
        self.fade = 1.0                 # 0..1 how far the --tray window has faded in (always 1 otherwise)
        self.t_step = time.time()
        self.u = 1.0                    # UI scale: fonts and pipeline pills grow on big fullscreen windows
        self.lm = self.lv = 0.0
        self.spec = np.zeros(32, np.float32)
        self.col = np.array(STATE_COLOR["idle"], np.float64)
        rng = np.random.default_rng(7)
        self.particles = rng.uniform(0, 2 * math.pi, (40, 3))   # phase, speed, wobble

    def open(self):
        pg = self.pg
        pg.init()
        self.poses, self.pose_imgs = {}, {}
        files = {}
        for name, (file, circle) in AVATARS.items():
            try:
                if file not in files:
                    files[file] = pg.image.load(os.path.join(ASSETS, file))
                self.poses[name] = (files[file], circle)
            except (OSError, pg.error):
                pass                    # missing pose: falls back to the portrait, or the plain orb
        self.pose, self.prev_pose, self.pose_since = "portrait", None, 0.0

        # window: Leo as the icon, and on Windows his own taskbar entry and a title bar in the viz colors
        _windows_app_id()
        self.fonts(1.0)
        if self.tray:
            self.sizes = pg.display.get_desktop_sizes()
            self.monitor = min(self.monitor, len(self.sizes) - 1)
            pg.display.quit()           # no window until he's called
            self.start_tray()
            return
        if "portrait" in self.poses:
            pg.display.set_icon(self.circle("portrait", 32))
        pg.display.set_caption("Hey Leo")
        pg.display.set_mode((1000, 660), pg.RESIZABLE)
        _windows_title_bar(pg.display.get_wm_info().get("window"), self.BG, (225, 228, 240))

    def fonts(self, u):
        pg = self.pg
        names = "segoeui,helveticaneue,arial,dejavusans"
        self.u = u
        self.f_big = pg.font.SysFont(names, int(26 * u), bold=True)
        self.f_mid = pg.font.SysFont(names, int(20 * u))
        self.f_small = pg.font.SysFont(names, int(15 * u), bold=True)

    # -- --tray: an icon in the system tray, and a fullscreen window only while Leo is in a conversation --
    def start_tray(self):
        try:
            import pystray
        except ImportError:
            sys.exit("--tray needs pystray and Pillow: pip install pystray pillow")
        item, Menu = pystray.MenuItem, pystray.Menu

        def pick(n):
            def choose(icon, it):
                self.monitor, self.reopen = n, True
            return choose

        def toggle(icon, it):
            self.pinned = not self.pinned
            self.dismissed = False

        def quit_(icon, it):
            self.quit = True

        screens = [item(f"{n + 1}: {w}\u00d7{h}", pick(n), checked=lambda it, n=n: self.monitor == n, radio=True)
                   for n, (w, h) in enumerate(self.sizes)]
        menu = Menu(item("Show Leo", toggle, checked=lambda it: self.pinned, default=True),
                    item("Monitor", Menu(*screens)),
                    Menu.SEPARATOR,
                    item("Quit", quit_))
        self.icon = pystray.Icon("HeyLeo", _tray_image(), "Hey Leo", menu)
        threading.Thread(target=self.icon.run, daemon=True).start()
        print("Monitors: " + ", ".join(f"{n + 1}: {w}x{h}" for n, (w, h) in enumerate(self.sizes))
              + f". Leo shows up on {self.monitor + 1} (change it from the tray icon).")

    def show(self):
        pg = self.pg
        pg.display.init()
        sizes = pg.display.get_desktop_sizes()
        n = min(self.monitor, len(sizes) - 1)
        if "portrait" in self.poses:
            pg.display.set_icon(self.circle("portrait", 32))
        pg.display.set_caption("Hey Leo")
        # created hidden and fully transparent, so it fades in instead of flashing up black
        pg.display.set_mode(sizes[n], pg.NOFRAME | pg.HIDDEN, display=n)
        self.window = None
        try:
            from pygame._sdl2.video import Window
            self.window = Window.from_display_module()
            self.window.opacity = 0.0
            self.window.show()
        except Exception:               # no opacity control: just appear (the pop-in still plays)
            pg.display.set_mode(sizes[n], pg.NOFRAME, display=n)
        _windows_on_top(pg.display.get_wm_info().get("window"))
        self.pose_imgs.clear()
        self.fonts(max(1.0, sizes[n][1] / 760))
        self.fade = 0.0
        self.hold = 2                   # frames drawn at opacity 0 first: the window and Leo's images get built
        self.visible = True

    def hide(self):
        self.pg.display.quit()
        self.window = None
        self.visible = False

    def tray_step(self):
        """Show, hide or fade the window for what Leo is doing. Returns True when it is up."""
        f = self.hub.frame()
        now = time.time()
        dt, self.t_step = min(0.1, now - self.t_step), now
        busy = active_stage(f) != "idle" or f.stop_age < 0.7
        if not busy:
            self.dismissed = False
        elif not self.dismissed:
            self.last_busy = now
        want = (self.pinned or now - self.last_busy < self.linger) and not self.dismissed
        if self.visible and self.reopen:
            self.hide()                 # moving to another monitor: no fade, it reappears there at once
        self.reopen = False
        if want and not self.visible:
            self.show()
        if self.visible and self.hold:
            self.hold -= 1
        elif self.visible:
            self.fade = min(1.0, self.fade + dt / APPEAR) if want else max(0.0, self.fade - dt / VANISH)
            if self.fade == 0.0 and not want:
                self.hide()
            elif self.window:
                self.window.opacity = _ease_out(self.fade)
        return self.visible

    def dismiss(self):
        """Esc, Q or closing the window in --tray mode: fade out until Leo is called again."""
        self.pinned, self.dismissed, self.last_busy = False, True, 0.0

    def circle(self, name, radius):
        """Pose `name` cut to a circle of `radius`, as a new surface with transparent corners."""
        pg = self.pg
        src, (x, y, r) = self.poses[name]
        iw, ih = src.get_size()
        r = min(r, x, y, iw - x, ih - y)
        img = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
        img.blit(pg.transform.smoothscale(src.subsurface(pg.Rect(x - r, y - r, 2 * r, 2 * r)), img.get_size()), (0, 0))
        mask = pg.Surface(img.get_size(), pg.SRCALPHA)
        pg.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)
        img.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MIN)
        return img

    def avatar(self, name, radius):
        """`circle()` for drawing, rebuilt only when the size changes."""
        img = self.pose_imgs.get(name)
        if img is None or img.get_width() != radius * 2:
            img = self.pose_imgs[name] = self.circle(name, radius).convert_alpha()
        return img

    def draw_avatar(self, s, cx, cy, radius, state, t):
        """Leo in the pose for this stage, crossfading from the previous one."""
        want = POSE_FOR_STAGE.get(state, "portrait")
        if want not in self.poses:
            want = "portrait" if "portrait" in self.poses else next(iter(self.poses))
        if want != self.pose:
            self.prev_pose, self.pose, self.pose_since = self.pose, want, t
        fade = min(1.0, (t - self.pose_since) / POSE_FADE)
        # set_alpha(None) would switch off blending and blit the cut-away corners as a square,
        # so every blit sets an explicit alpha instead
        if fade < 1 and self.prev_pose in self.poses:
            prev = self.avatar(self.prev_pose, radius)
            prev.set_alpha(255)
            s.blit(prev, (cx - radius, cy - radius))
        img = self.avatar(self.pose, radius)
        img.set_alpha(int(255 * fade))
        s.blit(img, (cx - radius, cy - radius))

    def close(self):
        if self.icon:
            self.icon.stop()
        self.pg.quit()

    def loop(self, done):
        pg = self.pg
        clock = pg.time.Clock()
        while not done() and not self.quit:
            if self.tray and not self.tray_step():
                time.sleep(0.05)
                continue
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                    if not self.tray:
                        return
                    self.dismiss()
                    break
            if not pg.display.get_init():
                continue
            self.draw(pg.display.get_surface(), self.hub.frame())
            pg.display.flip()
            clock.tick(60)

    # -- drawing --
    def draw(self, s, f):
        pg = self.pg
        W, H = s.get_size()
        t = time.time()
        s.fill(self.BG)

        state = active_stage(f)
        self.lm = ease(self.lm, level(f.mic))
        self.lv = ease(self.lv, level(f.voice))
        active = self.lv if f.speaking else self.lm
        self.col += (np.array(STATE_COLOR[state]) - self.col) * 0.08
        col = tuple(self.col)
        wave, rate = (f.voice, f.voice_rate) if f.speaking else (f.mic, f.mic_rate)
        self.spec = ease(self.spec, spectrum(wave, rate, 32))

        self.draw_pipeline(s, W, state, col)
        self.draw_orb(s, W, H, t, state, active, col)
        self.draw_captions(s, W, H, f)

        b = badge(f)
        if b:
            img = self.f_small.render(b, True, LEO_COLOR)
            s.blit(img, (W - img.get_width() - int(20 * self.u), int(22 * self.u)))
        if f.stop_age < 0.7:
            veil = pg.Surface((W, H), pg.SRCALPHA)
            veil.fill((255, 60, 60, int(110 * (1 - f.stop_age / 0.7))))
            s.blit(veil, (0, 0))
            img = self.f_big.render("STOPPED", True, (255, 255, 255))
            s.blit(img, img.get_rect(center=(W // 2, int(H * ORB_Y))))

    def draw_pipeline(self, s, W, state, col):
        pg = self.pg
        u = self.u
        pads, gap, h = int(14 * u), int(8 * u), int(28 * u)
        top = int(16 * u - (1 - _ease_out(self.fade)) * 50 * u)     # slides down from above as the window fades in
        imgs = [(key, self.f_small.render(label, True, (255, 255, 255))) for key, label in PIPELINE]
        total = sum(i.get_width() + 2 * pads for _, i in imgs) + gap * (len(imgs) - 1)
        x = (W - total) // 2
        for key, img in imgs:
            w = img.get_width() + 2 * pads
            r = pg.Rect(x, top, w, h)
            ty = r.centery - img.get_height() // 2
            if key == state:
                pg.draw.rect(s, tuple(int(c) for c in col), r, border_radius=h // 2)
                s.blit(self.f_small.render(LABELS[key], True, (12, 14, 22)), (x + pads, ty))
            else:
                pg.draw.rect(s, (45, 50, 66), r, width=1, border_radius=h // 2)
                s.blit(self.f_small.render(LABELS[key], True, (95, 102, 125)), (x + pads, ty))
            x += w + gap

    def draw_orb(self, s, W, H, t, state, active, col):
        pg = self.pg
        cx, cy = W // 2, int(H * ORB_Y)
        r0 = min(W * 0.12, H * 0.1) * (0.55 + 0.45 * _ease_out_back(self.fade))   # pops up as the window fades in
        breath = 0.05 * math.sin(t * 1.6) if state == "idle" else 0.0
        R = r0 * (1 + breath + 0.45 * active)
        ra = int(r0 * 1.35) if self.poses else 0   # Leo's portrait radius (fixed)
        if ra:
            R = max(R, ra * 0.85)       # keep the glow outside the portrait

        # frequency ring, mirrored left/right, lows at the top
        nb = len(self.spec)
        rot = t * 0.6 if state in THINKING else 0.0
        inner = max(R * 1.3, ra + 12)
        for i in range(nb * 2):
            idx = i if i < nb else 2 * nb - 1 - i
            ang = -math.pi / 2 + (i + 0.5) * math.pi / nb + rot
            length = 5 + float(self.spec[idx]) * r0 * 1.1
            c = lerp_color(col, (255, 255, 255), 0.25 * float(self.spec[idx]))
            p1 = (cx + math.cos(ang) * inner, cy + math.sin(ang) * inner)
            p2 = (cx + math.cos(ang) * (inner + length), cy + math.sin(ang) * (inner + length))
            pg.draw.line(s, tuple(int(v) for v in c), p1, p2, 4)

        # glow and orb body
        gs = int(R * 2.7)
        glow = pg.Surface((gs * 2, gs * 2), pg.SRCALPHA)
        for k in range(8, 0, -1):
            pg.draw.circle(glow, (*[int(c) for c in col], 6 + (9 - k) * 3), (gs, gs), int(R * (1 + 0.2 * k)))
        if not ra:
            for i in range(12):
                frac = i / 11
                c = lerp_color(tuple(c * 0.55 for c in col), lerp_color(col, (255, 255, 255), 0.55), frac)
                pg.draw.circle(glow, (*[int(v) for v in c], 255), (gs, gs), max(1, int(R * (1 - frac * 0.85))))
        s.blit(glow, (cx - gs, cy - gs))

        # Leo's portrait, framed by a ring in the stage color that brightens with the sound
        if ra:
            self.draw_avatar(s, cx, cy, ra, state, t)
            ring = lerp_color(col, (255, 255, 255), min(0.6, active))
            pg.draw.circle(s, tuple(int(v) for v in ring), (cx, cy), ra + 2, 3 + int(5 * active))

        # thinking: particles orbiting the orb
        if state in THINKING:
            for ph, sp, wob in self.particles:
                ang = ph + t * (0.6 + sp * 0.15)
                rad = R * (1.9 + 0.35 * math.sin(t * 1.3 + wob)) + 20
                pg.draw.circle(s, tuple(int(c) for c in col), (cx + math.cos(ang) * rad, cy + math.sin(ang) * rad * 0.8), 3)

    def draw_captions(self, s, W, H, f):
        x, width = int(W * 0.12), int(W * 0.76)
        y = int(H * 0.7 + (1 - _ease_out(self.fade)) * 40 * self.u)     # rises into place as the window fades in
        for tag, text, color, n in (("YOU", f.question, YOU_COLOR, 1), ("LEO", f.answer, LEO_COLOR, 3)):
            if not text:
                continue
            s.blit(self.f_small.render(tag, True, color), (x - int(46 * self.u), y + int(4 * self.u)))
            for line in _wrap(self.f_mid, text, width, n):
                s.blit(self.f_mid.render(line, True, (225, 228, 240)), (x, y))
                y += self.f_mid.get_linesize()
            y += 4


# --------------------------------------------------------------------- run ---

def run(mode, target, tray_monitor=None, tray_linger=TRAY_LINGER):
    """Run `target()` (the assistant) in a worker thread, with a renderer on the main thread.
    `tray_monitor` (pygame only): live in the system tray and go fullscreen on that display when called,
    hiding again `tray_linger` seconds after the conversation ends."""
    view = TerminalView(hub) if mode == "terminal" else PygameView(hub, tray_monitor, tray_linger)
    errors = []

    def work():
        try:
            target()
        except BaseException as e:      # SystemExit included: re-raised below, after the screen is restored
            errors.append(e)

    hub.enabled = True
    view.open()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    interrupted = False
    try:
        view.loop(lambda: not worker.is_alive())
    except KeyboardInterrupt:
        interrupted = True
    finally:
        view.close()
    if interrupted:
        raise KeyboardInterrupt
    if errors:
        raise errors[0]
