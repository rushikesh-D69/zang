"""Word-level alignment of any speech audio to the reply text (general: any text, voice, length, source).

Cascade, best first; the result says which method produced it:
  1. "engine"   - word times supplied by the TTS engine itself (e.g. ElevenLabs/Azure), used as-is
  2. "whisper"  - faster-whisper word timestamps on the actual audio, matched to our words with a
                  sequence alignment (handles numbers like "2026" spoken as four words, contractions,
                  dropped/extra words), then each sentence is corrected onto its measured voiced span
  3. "sentence" - only sentence start/end are known (we synthesised sentence by sentence): words are
                  spread inside each sentence by the text-prosody model
  4. "text"     - no audio: pure text-prosody prediction
Unmatched words are interpolated between matched neighbours by character length, so every word gets
a time; `coverage` = share of words timed directly from the audio.
"""
import difflib
import os
import re

import numpy as np

from ..paths import ROOT

WHISPER_DIR = os.path.join(ROOT, "models", "whisper")
_model = None


def whisper_available():
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(WHISPER_DIR) and bool(os.listdir(WHISPER_DIR))


def _whisper(size="base"):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(size, device="cpu", compute_type="int8", download_root=WHISPER_DIR)
    return _model


def norm(w):
    return re.sub(r"[^\w]", "", w.lower().replace("’", "'"))


def whisper_words(audio, sr, language=None):
    """-> [(token, t0, t1)] from faster-whisper (16 kHz mono float)."""
    from .audio import resample
    x = resample(np.asarray(audio, np.float32), sr, 16000)
    segs, _ = _whisper().transcribe(x, word_timestamps=True, beam_size=1, language=language,
                                    condition_on_previous_text=False, vad_filter=False)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in s.words if w.word.strip()]


def match(words, asr):
    """Monotonic alignment of our words to ASR tokens -> list of (t0, t1) or None per word."""
    a = [norm(w) for w in words]
    b = [norm(t) for t, _, _ in asr]
    out = [None] * len(words)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = (asr[j1 + k][1], asr[j1 + k][2])
        elif tag == "replace" and j2 > j1:
            # e.g. our "2026" vs asr "twenty twenty six": spread the ASR span over our words by length
            t0, t1 = asr[j1][1], asr[j2 - 1][2]
            lens = np.array([max(len(a[i]), 1) for i in range(i1, i2)], float)
            edges = t0 + (t1 - t0) * np.concatenate([[0], np.cumsum(lens) / lens.sum()])
            for k in range(i2 - i1):
                out[i1 + k] = (float(edges[k]), float(edges[k + 1]))
    return out


def fill(words, times, t_start, t_end):
    """Interpolate missing word times between timed neighbours (by character length)."""
    n = len(words)
    res = list(times)
    i = 0
    while i < n:
        if res[i] is not None:
            i += 1
            continue
        j = i
        while j < n and res[j] is None:
            j += 1
        a = res[i - 1][1] if i > 0 else t_start
        b = res[j][0] if j < n else t_end
        lens = np.array([max(len(norm(words[k])), 1) for k in range(i, j)], float)
        edges = a + (b - a) * np.concatenate([[0], np.cumsum(lens) / lens.sum()])
        for k in range(i, j):
            res[k] = (float(edges[k - i]), float(edges[k - i + 1]))
        i = j
    return res


def sentence_ranges(words):
    """[(first_word_idx, last_word_idx)] split at . ! ? ; :"""
    out, s = [], 0
    for i, w in enumerate(words):
        if re.search(r"[.!?;:।॥]$", w) or i == len(words) - 1:
            out.append((s, i))
            s = i + 1
    return out


def correct_to_sentences(words, times, sentences):
    """Linear correction per sentence so first/last word land on the measured voiced span."""
    rngs = sentence_ranges(words)
    if not sentences or len(rngs) != len(sentences):
        return times
    res = list(times)
    for (i0, i1), s in zip(rngs, sentences):
        a, b = res[i0][0], res[i1][1]
        if b - a < 1e-3:
            continue
        k = (s["t1"] - s["t0"]) / (b - a)
        for i in range(i0, i1 + 1):
            res[i] = (s["t0"] + (res[i][0] - a) * k, s["t0"] + (res[i][1] - a) * k)
    return res


def snap_onsets(times, audio, sr, search=0.3):
    """ASR word starts drift ~0.1-0.2 s at phrase edges. A word that follows silence (or opens the clip)
    is moved to the first voiced sample in [t0 - 0.05, t0 + search]."""
    env = np.convolve(np.abs(audio), np.ones(int(0.01 * sr)) / (0.01 * sr), "same")
    voiced = env > env.max() * 10 ** (-35 / 20)
    out = list(times)
    for i, (t0, t1) in enumerate(out):
        a = int(max(0, t0 - 0.05) * sr)
        gap_before = i == 0 or t0 - out[i - 1][1] > 0.12 or not voiced[max(0, a - int(0.08 * sr)):a].any()
        if not gap_before:
            continue
        seg = voiced[a:int(min(len(audio), (t0 + search) * sr))]
        if seg.any():
            nt0 = (a + int(np.argmax(seg))) / sr
            if nt0 < t1 - 0.03:
                out[i] = (nt0, t1)
    return out


def align(words, audio=None, sr=None, sentences=None, engine_times=None, language=None, use_whisper=True):
    """-> dict(words=[{w,t0,t1}], method, coverage, notes)."""
    n = len(words)
    if engine_times and len(engine_times) == n:
        return {"words": [{"w": w, "t0": t[0], "t1": t[1]} for w, t in zip(words, engine_times)],
                "method": "engine", "coverage": 1.0, "notes": "word times from the TTS engine"}
    if audio is not None and use_whisper and whisper_available():
        asr = whisper_words(audio, sr, language)
        m = match(words, asr)
        cov = sum(t is not None for t in m) / max(n, 1)
        if cov >= 0.5:
            dur = len(audio) / sr
            t = fill(words, m, sentences[0]["t0"] if sentences else 0.0, sentences[-1]["t1"] if sentences else dur)
            t = correct_to_sentences(words, t, sentences) if sentences else snap_onsets(t, np.asarray(audio, np.float32), sr)
            return {"words": [{"w": w, "t0": round(a, 3), "t1": round(b, 3)} for w, (a, b) in zip(words, t)],
                    "method": "whisper", "coverage": round(cov, 3),
                    "notes": f"{sum(x is not None for x in m)}/{n} words matched to the ASR transcript"}
    if sentences:
        return {"words": None, "method": "sentence", "coverage": 0.0, "notes": "sentence boundaries only"}
    return {"words": None, "method": "text", "coverage": 0.0, "notes": "text prosody model only"}
