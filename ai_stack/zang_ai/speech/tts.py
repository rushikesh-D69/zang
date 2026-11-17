"""Text -> speech audio for Zang (offline Piper neural TTS, or OpenAI TTS when a key is set).

Output matches the MP3s already on Zang's SD card (MPEG-1 layer III, 44.1 kHz, stereo, 64 kbps), encoded
with LAME (lameenc), so the firmware's decoder gets exactly what it already plays.

Timing: Piper voices are patched in memory to output phoneme alignments (samples per phoneme), so every
word's start/end and its stressed vowel (espeak's 'ˈ' mark) are known exactly from the TTS itself, at no
extra cost (no ASR needed). Sentences are synthesised one by one with our own pauses, which also makes
streaming natural (see stream.py).
"""
import glob
import json
import os
import re
import urllib.request

import numpy as np

from ..paths import ROOT

VOICE_DIR = os.path.join(ROOT, "models", "piper")
OUT_SR = 44100
PAUSE = {".": 0.35, "!": 0.35, "?": 0.45, ";": 0.25, ":": 0.25, "।": 0.35, "॥": 0.45}
_voices = {}


def voices():
    out = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(VOICE_DIR, "*.onnx")))]
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai:alloy")
    return out


ABBREV = r"(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|No|Fig)\."


def split_sentences(text):
    """Sentence chunks for synthesis; abbreviations (Dr., Mr., e.g.) and decimals (3.5) do not end a sentence."""
    protected = re.sub(ABBREV, lambda m: m.group(0).replace(".", "\x00"), text)
    protected = re.sub(r"(\d)\.(\d)", "\\1\x00\\2", protected)
    out = [s.strip().replace("\x00", ".") for s in re.findall(r"[^.!?;:।॥]+[.!?;:।॥]*", protected)]
    return [s for s in out if re.search(r"\w", s)]


VOWELS = set("aeiouyɪʊɛɔæʌɑɒəɚɜɝɐøɨʉɯɤœɶ")
_word_groups = {}


def _piper(name):
    if name not in _voices:
        from piper import PiperVoice
        try:
            _voices[name] = PiperVoice.load(os.path.join(VOICE_DIR, name + ".onnx"), include_alignments=True)
        except Exception:  # no onnx package: audio only, timing falls back to the aligner
            _voices[name] = PiperVoice.load(os.path.join(VOICE_DIR, name + ".onnx"))
    return _voices[name]


def warm(voice=None):
    """Load (and alignment-patch) voices up front so the first reply is not slow."""
    for v in ([voice] if voice else [x for x in voices() if not x.startswith("openai:")]):
        _piper(v)


BOUNDARY = re.compile(r"[ .,!?;:—\-^$_।॥]")


def _word_phonemes(v, word):
    """Phonemes of a written word on its own (stress marks dropped), cached."""
    key = (id(v), word.lower())
    if key not in _word_groups:
        ph = [p for sent in v.phonemize(word) for p in sent]
        _word_groups[key] = [p for p in ph if not BOUNDARY.fullmatch(p) and p not in "ˈˌ"]
    return _word_groups[key]


def _piper_sentence(v, text, rate):
    """-> audio (float32 @ voice sr), sr, per-word (t0, t1) and stressed-vowel time (s from chunk start) or None.

    General word timing: each written word is phonemized alone and its phoneme string is aligned to the
    sentence's phoneme stream (which carries exact sample counts). Handles espeak merging words ("I am" ->
    aɪɐm), expanding them ("2026" -> two thousand twenty six) and any text."""
    import difflib
    from piper import SynthesisConfig
    chunks = list(v.synthesize(text, SynthesisConfig(length_scale=1.0 / max(rate, 0.3)), include_alignments=True))
    sr = chunks[0].sample_rate
    x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    al = [(a.phoneme, int(a.num_samples)) for c in chunks for a in (c.phoneme_alignments or [])]
    if not al:
        return x, sr, None, None
    stream, pos, stress_mark = [], 0, False  # (phoneme, start, end, stressed)
    for ph, n in al:
        if ph in "ˈˌ":
            stress_mark = stress_mark or ph == "ˈ"
        elif not BOUNDARY.fullmatch(ph):
            stream.append((ph, pos, pos + n, stress_mark and ph in VOWELS))
            if ph in VOWELS:
                stress_mark = False
        pos += n
    ours = re.findall(r"[\w'’̀-ͯऀ-ॣ०-෿]+", text)
    seq, owner = [], []
    for wi, w in enumerate(ours):
        for ph in _word_phonemes(v, w):
            seq.append(ph)
            owner.append(wi)
    sm = difflib.SequenceMatcher(None, seq, [p[0] for p in stream], autojunk=False)
    hit = [[] for _ in ours]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(j2 - j1):  # spread stream phonemes over the written word(s) they replace
                i = i1 + min(k * max(i2 - i1, 1) // max(j2 - j1, 1), max(i2 - i1 - 1, 0))
                if i < len(owner):
                    hit[owner[i]].append(j1 + k)
    if sum(1 for h in hit if h) < 0.6 * len(ours):
        return x, sr, None, None
    spans, stresses = [], []
    for h in hit:
        if not h:
            spans.append(None)
            stresses.append(None)
            continue
        seg = stream[min(h):max(h) + 1]
        spans.append((seg[0][1] / sr, seg[-1][2] / sr))
        st = [(p[1] + p[2]) / 2 / sr for p in seg if p[3]] or [(p[1] + p[2]) / 2 / sr for p in seg if p[0] in VOWELS]
        stresses.append(st[0] if st else None)
    # words the matcher missed: interpolate between neighbours
    for i, sp in enumerate(spans):
        if sp is None:
            a = spans[i - 1][1] if i and spans[i - 1] else 0.0
            b = next((q[0] for q in spans[i + 1:] if q), pos / sr)
            spans[i] = (a, max(a + 0.05, b))
    return x, sr, spans, stresses


def _voiced_span(x, sr):
    """First/last sample above -38 dB of the sentence peak (trims Piper's lead-in/out silence)."""
    env = np.convolve(np.abs(x), np.ones(int(0.01 * sr)) / (0.01 * sr), "same")
    thr = env.max() * 10 ** (-38 / 20)
    idx = np.where(env > thr)[0]
    return (idx[0], idx[-1]) if len(idx) else (0, len(x))


def synthesize(text, voice=None, rate=1.0, lead_in=0.15):
    """-> dict(audio float32 mono @44.1k, sr, duration, sentences=[{text, t0, t1}] (voiced spans, s), voice,
    word_times=[(t0, t1)] per word (exact, from TTS alignments) or None, stress_times=[t or None] per word)."""
    voice = voice or (voices()[0] if voices() else None)
    if voice is None:
        raise RuntimeError("no TTS voice installed (models/piper) and no OPENAI_API_KEY")
    parts, sents, t = [np.zeros(int(lead_in * OUT_SR), np.float32)], [], lead_in
    word_times, stress_times, exact = [], [], True
    for s in split_sentences(text):
        spans = stresses = None
        if voice.startswith("openai:"):
            x, sr = _openai(s, voice.split(":", 1)[1], rate)
        else:
            x, sr, spans, stresses = _piper_sentence(_piper(voice), s, rate)
        x = _resample(x, sr, OUT_SR)
        a, b = _voiced_span(x, OUT_SR)
        a0 = max(0, a - int(0.02 * OUT_SR))
        x = x[a0: b + int(0.03 * OUT_SR)]
        shift = t - a0 / OUT_SR  # chunk-local seconds -> output timeline
        if spans:
            word_times += [(round(w0 + shift, 3), round(w1 + shift, 3)) for w0, w1 in spans]
            stress_times += [None if st is None else round(st + shift, 3) for st in stresses]
        else:
            exact = False
        sents.append({"text": s, "t0": round(t + 0.02, 3), "t1": round(t + len(x) / OUT_SR - 0.03, 3)})
        parts.append(x)
        gap = PAUSE.get(s[-1], 0.2)
        parts.append(np.zeros(int(gap * OUT_SR), np.float32))
        t += len(x) / OUT_SR + gap
    audio = np.concatenate(parts)
    audio = 0.89 * audio / max(np.abs(audio).max(), 1e-6)
    return {"audio": audio, "sr": OUT_SR, "duration": round(len(audio) / OUT_SR, 3), "sentences": sents, "voice": voice,
            "word_times": word_times if exact and word_times else None,
            "stress_times": stress_times if exact and word_times else None}


def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n) * sr_in / sr_out, np.arange(len(x)), x).astype(np.float32)


def _openai(text, voice, rate):
    """OpenAI speech endpoint -> PCM (24 kHz, 16-bit mono)."""
    body = json.dumps({"model": os.environ.get("ZANG_TTS_MODEL", "gpt-4o-mini-tts"), "voice": voice, "input": text,
                       "response_format": "pcm", "speed": rate}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/audio/speech", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    pcm = urllib.request.urlopen(req, timeout=60).read()
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768, 24000


def to_mp3(audio, sr=OUT_SR, kbps=64, stereo=True):
    """MPEG-1 layer III like the files already on Zang's SD card."""
    import lameenc
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    if stereo:
        pcm = np.repeat(pcm, 2)
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(sr)
    enc.set_out_sample_rate(sr)  # keep 44.1 kHz (MPEG-1), LAME would drop to 24 kHz at 64 kbps
    enc.set_channels(2 if stereo else 1)
    enc.set_quality(2)
    return bytes(enc.encode(pcm.tobytes()) + enc.flush())
