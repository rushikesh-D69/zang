"""Speech prosody analysis (numpy only): loudness, pitch, stressed syllables, pauses, phrases.

Timing in co-speech gesture comes from prosody, not from words: a beat gesture's apex lands on
the stressed syllable (pitch accent). So we take timing from the *audio that is actually played*
(the TTS output) and meaning from the text. This keeps motion in sync even when the TTS gives
no word timestamps.
"""
import shutil
import subprocess
import wave

import numpy as np

SR = 16000
HOP = 0.01  # 10 ms analysis grid


def load_audio(path):
    """Mono float32 at 16 kHz. WAV via the stdlib; anything else via ffmpeg if installed."""
    if path.lower().endswith(".wav"):
        with wave.open(path) as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
        x = np.frombuffer(raw, dt).astype(np.float32)
        x = (x - 128) / 128 if sw == 1 else x / float(np.iinfo(dt).max)
        x = x.reshape(-1, ch).mean(1)
        return resample(x, sr, SR)
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path}: only .wav can be read without ffmpeg (sudo apt install ffmpeg), "
                           "or drop the file in the twin page, which decodes it in the browser")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def resample(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n) * sr_in / sr_out, np.arange(len(x)), x).astype(np.float32)


def _frames(x, win, hop):
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _smooth(y, n):
    if n <= 1:
        return y
    k = np.hanning(n + 2)[1:-1]
    return np.convolve(y, k / k.sum(), mode="same")


def _pitch(x, hop, floor_db, db):
    """Autocorrelation pitch, 75..500 Hz, 40 ms window. Returns f0 Hz (0 = unvoiced)."""
    win = int(0.04 * SR)
    F = _frames(np.pad(x, (win // 2, win)), win, hop)[:len(db)]
    F = (F - F.mean(1, keepdims=True)) * np.hanning(win)
    lo, hi = SR // 500, SR // 75
    f0 = np.zeros(len(F))
    for s in range(0, len(F), 1000):  # chunk to bound memory
        blk = F[s:s + 1000]
        spec = np.fft.rfft(blk, 2 * win)
        r = np.fft.irfft(np.abs(spec) ** 2)[:, :win]
        r /= r[:, :1] + 1e-12
        lag = lo + np.argmax(r[:, lo:hi], 1)
        peak = r[np.arange(len(r)), lag]
        voiced = (peak > 0.45) & (db[s:s + 1000] > floor_db)
        f0[s:s + 1000] = np.where(voiced, SR / lag, 0)
    # remove octave blips: median filter over 5 frames on voiced runs
    g = f0.copy()
    for i in range(2, len(f0) - 2):
        w = f0[i - 2:i + 3]
        w = w[w > 0]
        if f0[i] > 0 and len(w) >= 3:
            g[i] = np.median(w)
    return g


FRAME_S = 0.025            # loudness analysis window
MIN_PAUSE_S = 0.2          # silence at least this long is a pause
MIN_PHRASE_S = 0.12        # speech shorter than this is a click, not a phrase
MIN_SYLLABLE_GAP_S = 0.12
MIN_SYLLABLE_PROMINENCE_DB = 2.5


def _energy(x):
    """Per-frame loudness (dB) and speech-band (300-2500 Hz) energy, on the 10 ms grid. Syllable nuclei come
    from the band energy. Returns (frame times, db, band_db)."""
    hop = int(HOP * SR)
    win = int(FRAME_S * SR)
    frames = _frames(np.pad(x, (win // 2, win)), win, hop)[:int(np.ceil(len(x) / hop))]
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), 512, axis=1)) ** 2
    freqs = np.fft.rfftfreq(512, 1 / SR)
    band = 10 * np.log10(spec[:, (freqs >= 300) & (freqs <= 2500)].sum(1) + 1e-12)
    return np.arange(len(frames)) * HOP, db, band


def _pauses_and_phrases(speech, t, duration):
    """Runs of silence >= 200 ms are pauses; runs of speech >= 120 ms are phrases, merged across short gaps."""
    pauses, phrases = [], []
    i = 0
    while i < len(speech):
        j = i
        while j < len(speech) and speech[j] == speech[i]:
            j += 1
        run = (t[i], t[min(j, len(t) - 1)] if j < len(t) else duration)
        if not speech[i] and run[1] - run[0] >= MIN_PAUSE_S:
            pauses.append(run)
        elif speech[i] and run[1] - run[0] >= MIN_PHRASE_S:
            phrases.append(run)
        i = j
    merged = []
    for phrase in phrases:
        if merged and phrase[0] - merged[-1][1] < MIN_PAUSE_S:
            merged[-1] = (merged[-1][0], phrase[1])
        else:
            merged.append(phrase)
    return pauses, merged


def _syllable_nuclei(band, speech, db, floor_db):
    """Syllable nuclei = peaks of the smoothed speech-band envelope, >= 120 ms apart. Returns [(frame, prominence dB)]."""
    env = _smooth(band, 7)
    candidates = [k for k in range(1, len(env) - 1)
                  if env[k] >= env[k - 1] and env[k] > env[k + 1] and speech[k] and db[k] > floor_db + 4]
    nuclei = []
    for k in candidates:
        a, b = max(0, k - 15), min(len(env), k + 16)
        prominence_db = env[k] - max(env[a:k + 1].min(), env[k:b].min())
        if prominence_db < MIN_SYLLABLE_PROMINENCE_DB:
            continue
        if nuclei and (k - nuclei[-1][0]) * HOP < MIN_SYLLABLE_GAP_S:
            if env[k] > env[nuclei[-1][0]]:
                nuclei[-1] = (k, prominence_db)
            continue
        nuclei.append((k, prominence_db))
    return nuclei


def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def _stressed_syllables(nuclei, phrases, db, f0, semi, p95):
    """Stress (prominence) per syllable: louder, higher or moving pitch, phrase-initial."""
    if not nuclei:
        return []
    frames = np.array([k for k, _ in nuclei])
    pitch_dev = np.array([np.abs(semi[max(0, k - 5):k + 6]).max() for k in frames])
    pitch_move = np.array([np.ptp(semi[max(0, k - 8):k + 9][f0[max(0, k - 8):k + 9] > 0])
                           if (f0[max(0, k - 8):k + 9] > 0).sum() > 2 else 0.0 for k in frames])
    local = np.array([prominence_db for _, prominence_db in nuclei])
    prominence = 0.45 * _zscore(db[frames]) + 0.2 * _zscore(pitch_dev) + 0.2 * _zscore(pitch_move) + 0.15 * _zscore(local)
    phrase_starts = np.array([phrase[0] for phrase in phrases])
    syllables = []
    for k, prom in zip(frames, prominence):
        t = k * HOP
        phrase_initial = len(phrase_starts) and np.min(np.abs(t - phrase_starts)) < 0.25
        syllables.append({"t": round(t, 3), "prom": round(float(prom + (0.5 if phrase_initial else 0)), 3),
                          "db": round(float(db[k] - p95), 1), "f0": round(float(f0[k]), 1)})
    return syllables


def _label_phrases(phrases, t, f0, semi):
    """Phrase dicts with the terminal pitch rise (end vs. the stretch just before, semitones): rising = question."""
    labelled = []
    for start, end in phrases:
        last = (t >= end - 0.12) & (t <= end) & (f0 > 0)
        before = (t >= end - 0.6) & (t < end - 0.2) & (f0 > 0)
        rise = float(semi[last].mean() - semi[before].mean()) if last.sum() > 2 and before.sum() > 2 else 0.0
        labelled.append({"t0": round(float(start), 3), "t1": round(float(end), 3), "rise": round(rise, 2),
                         "question": rise > 1.5})
    return labelled


def analyse(x, sr=SR):
    """Return a dict of prosodic features and events. All times in seconds."""
    x = resample(np.asarray(x, np.float32), sr, SR)
    hop = int(HOP * SR)
    t, db, band = _energy(x)

    p10, p95 = np.percentile(db, [10, 95])
    floor_db = max(p10 + 8, p95 - 35)          # speech/silence threshold
    speech = _smooth((db > floor_db).astype(float), 9) > 0.5

    f0 = _pitch(x, hop, floor_db, db)
    voiced_f0 = f0[f0 > 0]
    f0_median = float(np.median(voiced_f0)) if len(voiced_f0) else 0.0
    semitones = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1) / max(f0_median, 1)), 0.0)

    pauses, phrases = _pauses_and_phrases(speech, t, len(x) / SR)
    nuclei = _syllable_nuclei(band, speech, db, floor_db)
    syllables = _stressed_syllables(nuclei, phrases, db, f0, semitones, p95)
    phrases = _label_phrases(phrases, t, f0, semitones)

    loud_norm = np.clip((db - floor_db) / max(p95 - floor_db, 1e-3), 0, 1)
    speech_time = sum(p["t1"] - p["t0"] for p in phrases)
    return {
        "duration": round(len(x) / SR, 3), "hop": HOP, "t": t, "loud": loud_norm, "db": db, "f0": f0,
        "speech": speech, "f0_median": round(f0_median, 1), "syllables": syllables,
        "pauses": [[round(float(v), 3) for v in p] for p in pauses],
        "phrases": phrases, "speech_ratio": round(float(speech.mean()), 3),
        "syllable_rate": round(len(syllables) / max(speech_time, 1e-3), 2),
    }


def summary(a):
    """JSON-safe subset (drops the per-frame arrays except a decimated loudness curve)."""
    keep = {k: v for k, v in a.items() if k not in ("t", "loud", "db", "f0", "speech")}
    keep["loud_curve"] = [round(float(v), 3) for v in a["loud"][::2]]  # 20 ms
    keep["f0_curve"] = [round(float(v), 1) for v in a["f0"][::2]]
    return keep
