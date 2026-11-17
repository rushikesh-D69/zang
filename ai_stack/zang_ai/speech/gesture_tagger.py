"""Text -> gesture spans with a pretrained Hugging Face model (runs offline on CPU).

Model: qfrodicio/roberta-finetuned-gesture-prediction-21-classes (MIT, RoBERTa-base token
classifier trained to predict social-robot gestures from utterance text, F1 0.83 on its eval set).
Downloaded into models/hf/ by:  HF_HOME=models/hf .venv/bin/python -c "from transformers import
pipeline; pipeline('token-classification', model='qfrodicio/roberta-finetuned-gesture-prediction-21-classes')"

Its classes are mapped onto Zang's gesture set. It is a fallback/enricher: explicit LLM tags win,
the keyword lexicon fills what the model is unsure about (score < MIN_SCORE).
"""
import os
import re
import threading

from ..paths import ROOT

MODEL = "qfrodicio/roberta-finetuned-gesture-prediction-21-classes"
MIN_SCORE = 0.55

def _hf_map():
    """Model class -> Zang gesture, from config/gestures.json ("hf" lists)."""
    from .text import VOCAB
    return {c: g for g, v in VOCAB.items() for c in v.get("hf", [])}


MAP = _hf_map()
_pipe = None
_lock = threading.Lock()  # the server warms this model up in a background thread while requests can already arrive:
                          # two threads importing transformers at once fail with "cannot import name 'pipeline'"


def available():
    return os.path.isdir(os.path.join(ROOT, "models", "hf"))


def _load():
    global _pipe
    with _lock:
        if _pipe is None:
            os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            from transformers import pipeline
            _pipe = pipeline("token-classification", model=MODEL, aggregation_strategy="simple", device=-1)
        return _pipe


def tag(words):
    """words (tag-free tokens) -> [(word_index, gesture, score, label)] at the first word of each span."""
    text = " ".join(words)
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    out = []
    for sent in re.finditer(r"[^.!?]+[.!?]*", text):  # sentence by sentence (model was trained on utterances)
        for sp in _load()(sent.group(0)):
            g = MAP.get(sp["entity_group"])
            if not g or float(sp["score"]) < MIN_SCORE:
                continue
            c0 = sent.start() + sp["start"]
            wi = max(i for i, s in enumerate(starts) if s <= c0 + 1) if starts else 0
            out.append((wi, g, round(float(sp["score"]), 3), sp["entity_group"]))
    return out
