"""Listening: voice activity (Silero VAD), end of turn (Pipecat Smart Turn v3.2) and the "Hello Zang" wake phrase.

All local ONNX/CPU, measured on this Ryzen 3:  VAD 0.16 ms per 32 ms frame, Smart Turn 90 ms per check,
wake phrase (Whisper tiny.en on the short utterance) ~0.7 s.

Models (models/voice):  silero_vad.onnx  (onnx-community/silero-vad, MIT)
                        smart-turn-v3.2-int8.onnx  (soniqo/Smart-Turn-v3.2-ONNX, BSD-2; 16 kHz audio in,
                        last 8 s, probability that the speaker has finished)
"""
import difflib
import os
import re

import numpy as np

from ..paths import ROOT

VOICE = os.path.join(ROOT, "models", "voice")
SR = 16000
FRAME = 512  # 32 ms


def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class SileroVAD:
    """Streaming speech probability per 32 ms frame, with hysteresis -> speech start / end events."""

    def __init__(self, on=0.5, off=0.35, start_frames=3, path=os.path.join(VOICE, "silero_vad.onnx")):
        self.s = _session(path)
        self.on, self.off, self.start_frames = on, off, start_frames
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)
        self.speaking = False
        self.run_on = 0
        self.silence_frames = 0
        self.p = 0.0

    def frame(self, x):
        """x: 512 float32 samples @16 kHz -> (prob, event) where event in {None, 'start', 'end'}."""
        x = np.asarray(x, np.float32)
        out, self.state = self.s.run(None, {"input": x[None], "state": self.state, "sr": np.array(SR, np.int64)})
        self.p = float(out[0, 0])
        ev = None
        if not self.speaking:
            self.run_on = self.run_on + 1 if self.p >= self.on else 0
            if self.run_on >= self.start_frames:
                self.speaking, self.silence_frames, ev = True, 0, "start"
        else:
            self.silence_frames = self.silence_frames + 1 if self.p < self.off else 0
            if self.silence_frames == 1:
                ev = "pause"  # first silent frame after speech (end-of-turn checks start counting here)
        return self.p, ev

    def end(self):
        self.speaking, self.run_on, self.silence_frames = False, 0, 0

    @property
    def silence_ms(self):
        return self.silence_frames * FRAME * 1000 / SR if self.speaking else 0.0


class SmartTurn:
    """P(turn complete) from the last 8 s of the user's audio (intonation + wording, not just silence)."""

    def __init__(self, path=os.path.join(VOICE, "smart-turn-v3.2-int8.onnx")):
        self.s = _session(path)

    def __call__(self, audio16k):
        a = np.asarray(audio16k, np.float32)[-128000:]
        buf = np.zeros(128000, np.float32)
        buf[-len(a):] = a
        return float(self.s.run(None, {"audio": buf[None]})[0][0, 0])


class EndOfTurn:
    """Combine VAD silence with Smart Turn, like production voice agents do:
       reply when silence >= min_ms and the model says 'done'; always after max_ms; wait up to hold_ms
       when the model says the speaker is mid-thought ("so I was thinking, um,")."""

    def __init__(self, min_ms=250, max_ms=1200, hold_ms=2000, threshold=0.5):
        self.min_ms, self.max_ms, self.hold_ms, self.thr = min_ms, max_ms, hold_ms, threshold
        self.turn = SmartTurn() if os.path.exists(os.path.join(VOICE, "smart-turn-v3.2-int8.onnx")) else None
        self.reset()

    def reset(self):
        self.p, self.checked_at = None, -1

    def decide(self, silence_ms, utterance):
        if silence_ms < self.min_ms:
            return False
        if self.turn is not None and silence_ms - self.checked_at >= 200:  # re-check every ~200 ms of silence
            self.p, self.checked_at = self.turn(utterance), silence_ms
        if self.p is None:
            return silence_ms >= self.max_ms
        if self.p >= self.thr:
            return True
        limit = self.hold_ms if self.p < 0.1 else self.max_ms
        return silence_ms >= limit


WAKE = ["hellozang", "heyzang", "hizang", "okzang", "okayzang", "namastezang", "hellosang", "hellozong",
        "hellotang", "heysang", "hellojang", "hizang", "hellozhang", "yozang", "hellozan"]
SLEEP = re.compile(r"\b(bye|goodbye|good night|see you|that's all|stop listening|go to sleep|alvida)\b", re.I)


def wake_match(text):
    """Fuzzy 'Hello Zang' detector on an STT transcript. -> (matched, remainder_text)."""
    t = re.sub(r"[^a-z ]", "", text.lower())
    words = t.split()
    for i in range(len(words)):
        for n in (1, 2, 3):
            chunk = "".join(words[i:i + n])
            if not chunk:
                continue
            best = max(difflib.SequenceMatcher(None, chunk, w).ratio() for w in WAKE)
            if best >= 0.8 or (n == 1 and chunk in ("zang", "zhang")):
                return True, " ".join(text.split()[i + n:]).strip(" ,.!?")
    return False, ""
