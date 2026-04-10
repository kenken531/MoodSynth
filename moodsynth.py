"""
MoodSynth  —  Windows Edition
===============================
Type a mood description in plain English.
The local LLM translates it into synthesis parameters.
A synthesis engine generates and plays matching ambient sound live.

Prerequisites:
    1. Install ollama:        https://ollama.com/download
    2. Start ollama server:   ollama serve   (separate terminal)
    3. Pull a model:          ollama pull qwen2.5:3b
    4. Install Python deps:   pip install ollama sounddevice numpy

Usage:
    python moodsynth.py
    python moodsynth.py --model llama3.2:3b
"""

import argparse
import json
import sys
import os
import time
import threading
import re
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:3b"
SAMPLE_RATE   = 44100
BLOCK_SIZE    = 2048       # audio buffer size — increase if you hear crackling
CHANNELS      = 2          # stereo

# How many seconds of audio to generate per synthesis cycle
# Longer = smoother crossfade but slightly more latency on mood change
CYCLE_SECS    = 4.0

# Master volume — reduce if audio clips or is too loud
MASTER_VOL    = 0.35

# Fade duration in seconds when switching between moods
FADE_SECS     = 1.5

# ── Default synthesis parameters (used as fallback) ───────────────────────────

DEFAULT_PARAMS = {
    "tempo":        60,          # BPM — controls LFO and pulse rates
    "base_freq":    220.0,       # Hz — root frequency of the soundscape
    "waveform":     "sine",      # sine | saw | square | noise | pad
    "reverb_depth": 0.4,         # 0.0 (dry) → 1.0 (very wet)
    "brightness":   0.5,         # 0.0 (dark/muffled) → 1.0 (bright/sharp)
    "density":      0.5,         # 0.0 (sparse) → 1.0 (dense/layered)
    "rhythm":       0.0,         # 0.0 (no pulse) → 1.0 (strong rhythmic pulse)
    "description":  "ambient"    # LLM's own label for the mood
}

# ── Colours ───────────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def c(text, color): return f"{color}{text}{RESET}"

# ── LLM parameter extraction ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a synthesis parameter generator for an ambient sound engine.

When given a mood description, respond with ONLY a valid JSON object — no explanation,
no markdown, no code fences, no extra text. Just raw JSON.

The JSON must have exactly these keys:
{
  "tempo":        <integer 20-160, BPM, slow moods=20-60, energetic=80-160>,
  "base_freq":    <float 40.0-880.0, Hz, dark moods=40-120, bright=400-880>,
  "waveform":     <string: "sine" | "saw" | "square" | "noise" | "pad">,
  "reverb_depth": <float 0.0-1.0, dry=0.0, cavernous=1.0>,
  "brightness":   <float 0.0-1.0, dark=0.0, bright=1.0>,
  "density":      <float 0.0-1.0, sparse=0.0, dense=1.0>,
  "rhythm":       <float 0.0-1.0, no pulse=0.0, strong pulse=1.0>,
  "description":  <string, your 3-5 word poetic label for this mood>
}

Examples:
- "calm rainy night" -> low tempo (~40), low base_freq (~80), pad waveform, high reverb (~0.8), low brightness (~0.2), medium density (~0.5), no rhythm (0.1)
- "tense thriller" -> high tempo (~120), mid freq (~300), saw waveform, medium reverb (~0.4), high brightness (~0.8), high density (~0.9), strong rhythm (~0.8)
- "peaceful forest" -> slow tempo (~35), mid-low freq (~160), sine waveform, medium reverb (~0.6), medium brightness (~0.5), low density (~0.3), no rhythm (~0.05)

Respond with ONLY the JSON object. Nothing else."""


def query_llm(model: str, mood: str) -> dict:
    """
    Ask the LLM to translate a mood description into synthesis parameters.
    Returns a dict of parameters, falling back to defaults on any failure.
    """
    print(f"\n  {c('Querying LLM...', DIM)}", end="", flush=True)
    t0 = time.time()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Mood: {mood}"},
            ],
            stream=False,
        )
        raw = response["message"]["content"].strip()
    except Exception as e:
        print(f"\n  {c('LLM error:', RED)} {e}")
        return dict(DEFAULT_PARAMS)

    elapsed = time.time() - t0
    print(f" {c(f'done ({elapsed:.1f}s)', DIM)}")

    # ── Parse JSON — robust extraction ──
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the first { ... } block
    brace_start = raw.find("{")
    brace_end   = raw.rfind("}")
    if brace_start != -1 and brace_end != -1:
        raw = raw[brace_start : brace_end + 1]

    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  {c('Could not parse JSON — using defaults.', YELLOW)}")
        print(f"  {c('Raw response: ' + repr(raw[:120]), DIM)}")
        return dict(DEFAULT_PARAMS)

    # Merge with defaults so missing keys don't crash synthesis
    merged = dict(DEFAULT_PARAMS)
    merged.update(params)

    # Clamp all numeric values to safe ranges
    merged["tempo"]        = int(np.clip(merged["tempo"],        20,    160))
    merged["base_freq"]    = float(np.clip(merged["base_freq"],  40.0,  880.0))
    merged["reverb_depth"] = float(np.clip(merged["reverb_depth"], 0.0, 1.0))
    merged["brightness"]   = float(np.clip(merged["brightness"], 0.0,   1.0))
    merged["density"]      = float(np.clip(merged["density"],    0.0,   1.0))
    merged["rhythm"]       = float(np.clip(merged["rhythm"],     0.0,   1.0))
    if merged["waveform"] not in ("sine", "saw", "square", "noise", "pad"):
        merged["waveform"] = "sine"

    return merged

# ── Synthesis engine ──────────────────────────────────────────────────────────

def generate_waveform(freq: float, n_samples: int, waveform: str,
                       phase_offset: float = 0.0) -> np.ndarray:
    """Generate one cycle of a basic waveform at the given frequency."""
    t = np.linspace(phase_offset,
                    phase_offset + n_samples / SAMPLE_RATE,
                    n_samples, endpoint=False)

    if waveform == "sine":
        wave = np.sin(2 * np.pi * freq * t)

    elif waveform == "saw":
        # Sawtooth: ramp from -1 to 1 per cycle
        wave = 2.0 * (freq * t - np.floor(freq * t + 0.5))

    elif waveform == "square":
        # Square: ±1 based on sine sign, with soft edge via tanh
        wave = np.tanh(10.0 * np.sin(2 * np.pi * freq * t))

    elif waveform == "noise":
        # Band-limited noise centred around freq
        noise = np.random.randn(n_samples).astype(np.float32)
        # Simple IIR low-pass to shape the noise spectrum
        alpha = np.clip(freq / (SAMPLE_RATE / 2), 0.001, 0.999)
        for i in range(1, n_samples):
            noise[i] = alpha * noise[i] + (1 - alpha) * noise[i - 1]
        wave = noise

    elif waveform == "pad":
        # Rich pad: sum of slightly detuned sines (chorus effect)
        wave  = np.sin(2 * np.pi * freq * t)
        wave += 0.6 * np.sin(2 * np.pi * freq * 1.007 * t)
        wave += 0.4 * np.sin(2 * np.pi * freq * 0.993 * t)
        wave += 0.3 * np.sin(2 * np.pi * freq * 2.0   * t)   # octave
        wave += 0.2 * np.sin(2 * np.pi * freq * 3.0   * t)   # fifth
        wave /= 2.5

    else:
        wave = np.sin(2 * np.pi * freq * t)

    return wave.astype(np.float32)


def apply_reverb(signal: np.ndarray, depth: float) -> np.ndarray:
    """
    Simple comb-filter reverb. Depth 0 = dry, 1 = very wet.
    Uses multiple delay lines summed together.
    """
    if depth < 0.01:
        return signal

    output   = signal.copy()
    delays   = [int(SAMPLE_RATE * d) for d in [0.029, 0.037, 0.041, 0.053]]
    decays   = [0.7, 0.65, 0.6, 0.55]

    for delay_samples, decay in zip(delays, decays):
        if delay_samples >= len(signal):
            continue
        delayed = np.zeros_like(signal)
        delayed[delay_samples:] = signal[:-delay_samples] * decay
        output += delayed * depth * 0.25

    # Normalise to prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output /= peak

    return output.astype(np.float32)


def apply_brightness_filter(signal: np.ndarray, brightness: float) -> np.ndarray:
    """
    Vectorised first-order IIR low-pass filter using scipy.signal.lfilter.
    brightness=0 → heavy low-pass (dark/muffled)
    brightness=1 → light filtering (bright/sharp)
    Vectorised to avoid Python for-loop overhead inside the audio callback.
    """
    from scipy.signal import lfilter
    alpha = 0.05 + brightness * 0.9
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    return lfilter(b, a, signal).astype(np.float32)


def synthesize(params: dict, n_samples: int, phase: float = 0.0) -> tuple[np.ndarray, float]:
    """
    Generate n_samples of audio from the given synthesis parameters.
    Returns (stereo_audio, new_phase) where new_phase continues the waveform.
    """
    freq      = params["base_freq"]
    waveform  = params["waveform"]
    depth     = params["reverb_depth"]
    bright    = params["brightness"]
    density   = params["density"]
    rhythm    = params["rhythm"]
    tempo     = params["tempo"]

    # Base layer
    wave = generate_waveform(freq, n_samples, waveform, phase)
    new_phase = phase + n_samples / SAMPLE_RATE

    # Density layers: add harmonics/sub-octaves based on density
    if density > 0.2:
        # Sub-octave
        sub = generate_waveform(freq * 0.5, n_samples, "sine", new_phase) * density * 0.5
        wave += sub
    if density > 0.5:
        # Fifth harmony
        fifth = generate_waveform(freq * 1.5, n_samples, "sine", new_phase) * density * 0.3
        wave += fifth
    if density > 0.75:
        # Texture layer
        texture = generate_waveform(freq * 2.0, n_samples, "noise", new_phase) * (density - 0.5) * 0.3
        wave += texture

    # LFO amplitude modulation (slow tremolo tied to tempo)
    lfo_freq = tempo / 60.0 * 0.25   # quarter of beat frequency
    t_lfo    = np.linspace(new_phase, new_phase + n_samples / SAMPLE_RATE, n_samples)
    lfo      = 0.5 + 0.5 * np.sin(2 * np.pi * lfo_freq * t_lfo)
    lfo      = 0.7 + 0.3 * lfo       # scale to [0.7, 1.0] so it never goes silent

    wave *= lfo

    # Rhythmic pulse (if rhythm > 0)
    if rhythm > 0.05:
        beat_freq  = tempo / 60.0
        beat_env   = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t_lfo)
        beat_env   = np.power(np.clip(beat_env, 0, 1), 2)   # sharper attack
        wave      *= (1.0 - rhythm * 0.6 + rhythm * 0.6 * beat_env)

    # Apply brightness filter
    wave = apply_brightness_filter(wave, bright)

    # Apply reverb
    wave = apply_reverb(wave, depth)

    # Normalise
    peak = np.max(np.abs(wave))
    if peak > 0.01:
        wave /= peak

    wave *= MASTER_VOL

    # Stereo: slight pan difference for width
    left  = wave * 0.9
    right = wave * 0.85
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    return stereo, new_phase

# ── Audio engine (shared state) ───────────────────────────────────────────────

_current_params  = dict(DEFAULT_PARAMS)
_target_params   = dict(DEFAULT_PARAMS)
_params_lock     = threading.Lock()
_phase           = 0.0
_fade_progress   = 1.0   # 1.0 = fully on target, <1.0 = fading in
_prev_audio      = None  # last block of previous params for crossfade
_audio_stream    = None
_stop_event      = threading.Event()


def audio_callback(outdata: np.ndarray, frames: int, time_info, status):
    """Called by sounddevice for each audio block. Must be fast and non-blocking."""
    global _phase, _fade_progress, _prev_audio

    with _params_lock:
        params = dict(_current_params)

    audio, new_phase = synthesize(params, frames, _phase)
    _phase = new_phase % 1000.0   # prevent float drift over time

    outdata[:] = audio


def start_audio():
    """Open the sounddevice output stream."""
    global _audio_stream
    _audio_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    _audio_stream.start()


def set_params(new_params: dict):
    """Thread-safe update of synthesis parameters."""
    with _params_lock:
        _current_params.update(new_params)

# ── Display ───────────────────────────────────────────────────────────────────

def params_bar(label: str, value: float, lo: float, hi: float,
               width: int = 16, color: str = CYAN) -> str:
    frac   = float(np.clip((value - lo) / (hi - lo), 0, 1))
    filled = int(frac * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"  {label:<14} {c(bar, color)}  {c(str(round(value, 2)), BOLD)}"


def display_params(params: dict, mood: str):
    waveform_icons = {
        "sine":   "∿",
        "saw":    "⊿",
        "square": "⊓",
        "noise":  "⋯",
        "pad":    "≋",
    }
    icon = waveform_icons.get(params["waveform"], "?")

    print(f"\n  {c('─' * 52, DIM)}")
    print(f"  {c('Mood     :', DIM)} {c(mood, YELLOW)}")
    print(f"  {c('Sound    :', DIM)} {c(params.get('description', ''), MAGENTA)}")
    print(f"  {c('Waveform :', DIM)} {c(icon + '  ' + params['waveform'], CYAN)}")
    print()
    print(params_bar("Tempo",      params["tempo"],        20,  160, color=GREEN))
    print(params_bar("Base Freq",  params["base_freq"],    40,  880, color=CYAN))
    print(params_bar("Reverb",     params["reverb_depth"], 0.0, 1.0, color=MAGENTA))
    print(params_bar("Brightness", params["brightness"],   0.0, 1.0, color=YELLOW))
    print(params_bar("Density",    params["density"],      0.0, 1.0, color=GREEN))
    print(params_bar("Rhythm",     params["rhythm"],       0.0, 1.0, color=RED))
    print(f"  {c('─' * 52, DIM)}")
    print(f"  {c('Audio is playing. Type a new mood or press Ctrl+C to quit.', DIM)}")
    print()

# ── Banner ────────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner(model: str):
    print(f"{BOLD}{MAGENTA}")
    print("  ███╗   ███╗ ██████╗  ██████╗ ██████╗ ")
    print("  ████╗ ████║██╔═══██╗██╔═══██╗██╔══██╗")
    print("  ██╔████╔██║██║   ██║██║   ██║██║  ██║")
    print("  ██║╚██╔╝██║██║   ██║██║   ██║██║  ██║")
    print("  ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██████╔╝")
    print("  ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝ ")
    print(f"  {DIM}S Y N T H{RESET}{BOLD}{MAGENTA}              Day 11 — BUILDCORED ORCAS{RESET}")
    print()
    print(f"  {BOLD}Model   :{RESET} {c(model, YELLOW)}")
    print(f"  {BOLD}Engine  :{RESET} {c('numpy synthesis — no external audio libs', DIM)}")
    print(f"  {BOLD}Output  :{RESET} {c('stereo 44100Hz via sounddevice', DIM)}")
    print()
    print(f"  {DIM}Type any mood description and press Enter.")
    print(f"  The LLM will translate it into sound parameters.")
    print(f"  Examples:{RESET}")
    print(f"    {c('calm rainy night', CYAN)}")
    print(f"    {c('tense thriller scene', CYAN)}")
    print(f"    {c('peaceful forest at dawn', CYAN)}")
    print(f"    {c('deep space exploration', CYAN)}")
    print(f"    {c('busy city morning', CYAN)}")
    print()
    print(f"  {c('─' * 52, DIM)}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MoodSynth — type a mood, hear it as ambient sound"
    )
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    clear()
    banner(args.model)

    # Verify ollama
    try:
        ollama.list()
    except Exception as e:
        print(f"  {c('ERROR:', RED)} Cannot reach ollama server.")
        print(f"  Run {c('ollama serve', GREEN)} in a separate terminal first.")
        print(f"  Details: {e}\n")
        sys.exit(1)

    # Start audio
    try:
        start_audio()
    except Exception as e:
        print(f"  {c('ERROR: Could not open audio output:', RED)} {e}")
        print(f"  Try running: python -c \"import sounddevice; print(sounddevice.query_devices())\"")
        print(f"  to check available devices.\n")
        sys.exit(1)

    print(f"  {c('Audio engine started.', GREEN)} Silence until you enter a mood.\n")

    # Main input loop
    while True:
        try:
            mood = input(f"  {c('Mood', MAGENTA)}{c(':', DIM)} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {c('Goodbye!', MAGENTA)}\n")
            break

        if not mood:
            continue

        if mood.lower() in ("exit", "quit", "q"):
            print(f"\n  {c('Goodbye!', MAGENTA)}\n")
            break

        # Query LLM for parameters
        params = query_llm(args.model, mood)

        # Apply to audio engine
        set_params(params)

        # Display the parameters
        display_params(params, mood)

    if _audio_stream:
        _audio_stream.stop()
        _audio_stream.close()


if __name__ == "__main__":
    main()