# MoodSynth 🎵

MoodSynth is a **text-to-sound ambient synthesizer**: type any mood description in plain English, and a local LLM translates it into synthesis parameters — tempo, base frequency, waveform type, reverb depth — which a pure-numpy audio engine turns into matching ambient sound played live. It's built for the **BUILDCORED ORCAS — Day 11** challenge.

## How it works

- Uses **ollama** to run a local LLM that reads your mood description and returns a structured JSON object with synthesis parameters.
- A **pure-numpy synthesis engine** (no external audio libraries needed) generates the waveform — supporting sine, sawtooth, square, noise, and pad waveforms — with density layering, LFO tremolo, rhythmic pulse, and reverb.
- Audio is streamed live to your speakers via **sounddevice** in real time.
- **Different moods produce audibly different sounds** — a calm mood yields a slow, reverb-heavy pad at low frequency; a tense mood yields a fast, bright sawtooth with strong rhythm.
- Switch moods at any time by typing a new description — the audio updates instantly.

## Requirements

- Python 3.10.x
- [ollama](https://ollama.com/download) installed and running
- The model pulled locally (see Setup)
- Working speaker or headphone output

## Python packages:

```bash
pip install ollama sounddevice numpy scipy
```

## Setup

1. Download and install ollama from [ollama.com/download](https://ollama.com/download).
2. In a **separate terminal**, start the ollama server:
```
ollama serve
```
3. Pull the model:
```
ollama pull qwen2.5:3b
```
4. Install the Python packages (see above or run:
```
pip install -r requirements.txt
```
after downloading `requirements.txt`)

## Usage

From the project folder:

```bash
python moodsynth.py
```

Then type any mood description and press Enter:

```
Mood: calm rainy night
Mood: tense thriller scene
Mood: peaceful forest at dawn
Mood: deep space exploration
Mood: busy city morning
```

- The LLM responds with synthesis parameters displayed as labelled bars in the terminal.
- Audio starts playing immediately and continues until you type a new mood.
- Type `exit` or press `Ctrl+C` to quit.

## Synthesis parameters

| Parameter | Range | Effect |
|---|---|---|
| `tempo` | 20–160 BPM | Controls LFO tremolo speed and rhythmic pulse rate |
| `base_freq` | 40–880 Hz | Root pitch of the soundscape |
| `waveform` | sine / saw / square / noise / pad | Tonal character of the sound |
| `reverb_depth` | 0.0–1.0 | Dry to cavernous space |
| `brightness` | 0.0–1.0 | Muffled/dark to sharp/bright |
| `density` | 0.0–1.0 | Sparse to layered harmonics |
| `rhythm` | 0.0–1.0 | No pulse to strong rhythmic beat |

## Common fixes

**Audio is silent** — run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices and check your default output is correct.

**Sound is harsh or clipping** — lower `MASTER_VOL` near the top of the script (default `0.35`). Increase `BLOCK_SIZE` if you hear crackling.

**LLM returns broken JSON** — this occasionally happens with smaller models. Just re-enter your mood description. The parser falls back to safe defaults automatically.

**ollama not running** — open a separate terminal and run `ollama serve` before launching MoodSynth.

**Very slow LLM response** — switch to a smaller model with `--model qwen2.5:1.5b`, or close other heavy applications.

## Hardware concept

MoodSynth mirrors a DAC (Digital-to-Analog Converter): text becomes numbers, numbers become waveform samples, waveform samples become voltage changes, voltage changes move a speaker cone. The LLM is the encoding step — it maps a high-dimensional semantic space (mood descriptions) to a low-dimensional parameter space (7 floats). The synthesis engine is the DAC — it maps those parameters to physical air pressure waves. Same signal chain as every speaker ever built, just at a different abstraction layer.

## Credits

- Local LLM inference: [ollama](https://ollama.com)
- Audio output: [sounddevice](https://python-sounddevice.readthedocs.io)
- Signal processing: [NumPy](https://numpy.org) + [SciPy](https://scipy.org)
- Default model: [Qwen2.5 3B](https://ollama.com/library/qwen2.5)

Built as part of the **BUILDCORED ORCAS — Day 11: MoodSynth** challenge.
