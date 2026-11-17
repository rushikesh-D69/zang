"""Meaning layer: LLM gesture tags + keyword lexicon -> semantic gesture events in audio time.

The LLM is asked to put tags in its reply, e.g.
    "Hi there! <wave> I'm Zang. <self> Do you want to build a robot <point> with me?"
Tags may be written <wave>, [wave] or {wave}. Words without tags still trigger gestures through
the lexicon below (English + Hinglish), so a plain reply also works.

Word timing comes from (best first): Whisper / TTS word timestamps if given, else the words
are spread over the detected phrases by character count (good to ~150 ms, enough to pick the
right stressed syllable, which then gives the exact timing).
"""
import re

import json
import os

from ..paths import ROOT

TAG_RE = re.compile(r"[<\[{]\s*([a-zA-Z_][\w:\-]*)\s*[>\]}]")
GESTURE_FILE = os.path.join(ROOT, "config", "gestures.json")


def load_gestures():
    """Movement vocabulary from config/gestures.json (edit that file to add gestures / trigger words)."""
    return json.load(open(GESTURE_FILE, encoding="utf-8"))["gestures"]


VOCAB = load_gestures()
GESTURES = tuple(VOCAB)
LEXICON = {g: v.get("words", "") for g, v in VOCAB.items() if v.get("words")}
_WORD2G = {w: g for g, ws in LEXICON.items() for w in ws.split()}


def llm_tag_help(library_names=()):
    """Tag list for the LLM system prompt, generated from config/gestures.json (+ library actions)."""
    shown = [g for g, v in VOCAB.items() if not v.get("head_only") or g in ("no", "yes")]
    out = ("Gesture tags (one per sentence; a long rhyme or story line may have two on different words. Put each right after the words it belongs to). Choose the MOST "
           "SPECIFIC tag for what the sentence means (size -> <big>/<small>, steps -> <count>, direction -> <up>/<there>, "
           "not knowing -> <shrug>, thinking -> <think>, success -> <celebrate>, yes/no answers -> <yes>/<no>). "
           "Danger or a warning -> <stop>. <wave> only when greeting or saying goodbye. Keep <excited> for real "
           "excitement. A question back to the person needs no tag, or <point>. Not every sentence needs a tag; never add "
           "one just to fill. Don't use the same tag in two sentences in a row. Use only these names:\n"
           + "\n".join(f"<{g}> {VOCAB[g].get('means', '')}" for g in shown))
    acts = sorted({n for n in library_names if n.isupper() and len(n) <= 12})[:20]
    if acts:
        out += "\nYou may also replay one of Zang's own recorded actions with <act:NAME>, e.g. " + ", ".join(f"<act:{a}>" for a in acts[:12])
    return out


# words LLMs write instead of our tag names (seen in real Groq replies: <curious>, <ask>)
ALIASES = {"curious": "think", "wonder": "think", "idea": "think", "hmm": "think", "ask": "question", "asking": "question",
           "hello": "wave", "hi": "wave", "bye": "wave", "goodbye": "wave", "greet": "wave", "hug": "heart",
           "love": "heart", "happy": "excited", "joy": "excited", "yay": "celebrate", "cheer": "celebrate",
           "clap": "clap", "win": "celebrate", "nod": "yes", "agree": "yes", "shake": "no", "disagree": "no",
           "cry": "sad", "upset": "sad", "embarrassed": "shy", "blush": "shy", "wow": "surprise", "shock": "surprise",
           "giggle": "laugh", "lol": "laugh", "funny": "laugh", "strong": "flex", "muscle": "flex", "relax": "calm",
           "wait": "stop", "careful": "stop", "hear": "listen", "me": "self", "you": "point", "look": "there",
           "huge": "big", "tiny": "small", "steps": "count", "dunno": "shrug", "thank": "thanks", "thankyou": "thanks",
           "apologize": "sorry", "earth": "down", "ground": "down", "floor": "down", "downward": "down", "shoo": "away",
           "leave": "away", "rainy": "rain", "drops": "rain", "snow": "rain", "sunny": "sun", "sunshine": "sun", "wheel": "round",
           "spin": "round", "bird": "fly", "wings": "fly", "bed": "sleep", "dream": "sleep", "yawn": "sleep", "applause": "clap",
           "jump": "run", "walk": "run", "hop": "run", "sing": "music", "song": "music", "together": "open", "welcome": "open", "dance": "music", "pray": "namaste"}


def resolve(name):
    """Tag -> (gesture, variant): 'think' -> ('think', None), 'think2' / 'think:2' -> ('think', 1) (1-based in tags),
    'think:scratch' -> ('think', 'scratch'), a common synonym ('curious') -> its gesture. Unknown -> (None, None)."""
    if name in VOCAB:
        return name, None
    if name in ALIASES:
        return ALIASES[name], None
    m = re.fullmatch(r"([a-z_]+?):?(\d+)", name) or re.fullmatch(r"([a-z_]+):([a-z_]+)", name)
    if m and m.group(1) in VOCAB:
        v, n = m.group(2), len(VOCAB[m.group(1)].get("variants", []))
        if v.isdigit():
            return m.group(1), (int(v) - 1 if 1 <= int(v) <= n else None)
        if any(x.get("id") == v for x in VOCAB[m.group(1)].get("variants", [])):
            return m.group(1), v
    return None, None


def is_gesture(name):
    """Known vocabulary entry (optionally a specific variant: think2, think:scratch), or a library action like act:HEYTHERE."""
    return name.startswith("act:") or resolve(name)[0] is not None


def parse(text):
    """-> (words, tags) where tags = [(word_index, gesture)].

    A tag belongs to the clause it follows ("Hi there! <wave>" waves on "Hi"), or to the
    next word when it opens the text / a sentence ("<wave> Hi there").
    """
    words, tags = [], []
    pos = 0
    for m in TAG_RE.finditer(text):
        words += _split(text[pos:m.start()])
        g = m.group(1)
        g = g if g.lower().startswith("act:") else g.lower()
        if g.lower().startswith("act:"):
            g = "act:" + g.split(":", 1)[1]
        if is_gesture(g):
            i = len(words) - 1
            if i < 0 or not text[pos:m.start()].strip():
                i = len(words)  # tag opens the text or follows another tag -> next word
            else:
                while i > 0 and not re.search(r"[.!?,;:]$", words[i - 1]) and len(words) - i < 4:
                    i -= 1  # back to the start of the clause (max 4 words)
            tags.append((i, g))
        pos = m.end()
    words += _split(text[pos:])
    tags = [(min(i, max(len(words) - 1, 0)), g) for i, g in tags]
    return words, tags


def split_tagged_sentences(text):
    """Split a tagged reply into sentences; a tag stays with the sentence it follows ("Hi! <wave> I am..." ->
    ["Hi! <wave>", "I am..."]), so each part keeps its own gestures."""
    out, cur = [], []
    for tok in re.findall(r"[<\[{][^>\]}]*[>\]}]|[^\s]+", text):
        last = _last_word(" ".join(cur))
        if not TAG_RE.fullmatch(tok) and re.search(r"[.!?।॥]$", last) and not re.fullmatch(r"(?i)(dr|mr|mrs|ms|prof|sr|jr|st|vs|etc|e\.g|i\.e)\.", last):
            out.append(" ".join(cur))
            cur = []
        cur.append(tok)
    if cur:
        out.append(" ".join(cur))
    return out


def _last_word(s):
    ws = [w for w in s.split() if not TAG_RE.fullmatch(w)]
    return ws[-1] if ws else ""


def _split(s):
    return [w for w in re.findall(r"[\w'’̀-ͯऀ-ॣ०-෿]+[?!.,;:।॥]*", s)]


def align(words, analysis, word_times=None):
    """-> [{w, t0, t1}] in seconds. word_times: optional [{w,t0,t1}] (Whisper JSON, ms or s)."""
    if word_times:
        scale = 0.001 if max(w["t1"] for w in word_times) > analysis["duration"] * 3 else 1.0
        return [{"w": w["w"], "t0": w["t0"] * scale, "t1": w["t1"] * scale} for w in word_times]
    phrases = analysis["phrases"] or [{"t0": 0.0, "t1": analysis["duration"]}]
    total_t = sum(p["t1"] - p["t0"] for p in phrases)
    chars = [len(re.sub(r"\W", "", w)) + 1 for w in words]
    total_c = sum(chars) or 1
    out, c_acc = [], 0
    for w, c in zip(words, chars):
        out.append({"w": w, "t0": _speech_time(phrases, total_t * c_acc / total_c),
                    "t1": _speech_time(phrases, total_t * (c_acc + c) / total_c)})
        c_acc += c
    return out


def _speech_time(phrases, s):
    """Map 'seconds of speech' to clock time by skipping pauses."""
    for p in phrases:
        d = p["t1"] - p["t0"]
        if s <= d:
            return p["t0"] + s
        s -= d
    return phrases[-1]["t1"]


TAG_GAP = 1.2  # s: minimum spacing of two explicit LLM tags (rhymes / songs tag every line); other cues use 1/max_rate


def semantic_events(words, tags, timed, max_rate=0.6, model_tags=None):
    """-> [{t, gesture, word, source}] sorted, thinned so semantic gestures are >= 1/max_rate s apart
    (explicit LLM tags: TAG_GAP).

    Explicit tags always win; lexicon hits fill gaps. Punctuation adds question / emphasis cues.
    """
    ev = []
    for i, g in tags:
        ev.append({"t": timed[i]["t0"], "gesture": g, "word": timed[i]["w"], "source": "tag", "prio": 2})
    for i, g, score, label in model_tags or []:
        ev.append({"t": timed[i]["t0"], "gesture": g, "word": timed[i]["w"], "source": f"model:{label}:{score}", "prio": 0.6 + 0.7 * score})
    for i, w in enumerate(words):
        base = re.sub(r"[^\w']", "", w.lower().replace("’", "'"))
        g = _WORD2G.get(base)
        if g:
            # pronoun / greeting / yes-no words are near-certain cues; topic words are weaker
            ev.append({"t": timed[i]["t0"], "gesture": g, "word": w, "source": "lexicon",
                       "prio": 1.2 if g in ("wave", "self", "point", "no", "yes") else 1.0})
        if w.endswith("?"):
            ev.append({"t": timed[i]["t1"], "gesture": "question", "word": w, "source": "punct", "prio": 1})
        if w.endswith("!"):
            ev.append({"t": timed[i]["t0"], "gesture": "emphasis", "word": w, "source": "punct", "prio": 0})
    ev.sort(key=lambda e: (e["t"], -e["prio"]))
    kept = []
    min_gap = 1.0 / max_rate
    for e in sorted(ev, key=lambda e: -e["prio"]):
        light = VOCAB.get(e["gesture"], {}).get("head_only", False)  # head/accent only, may overlap
        if light or all(abs(e["t"] - k["t"]) >= (TAG_GAP if e["source"] == k["source"] == "tag" else min_gap)
                        or VOCAB.get(k["gesture"], {}).get("head_only", False) for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e["t"])
    for e in kept:
        e.pop("prio")
    return kept


VAD_LINE = re.compile(r"\{[^{}]*\"valence\"[^{}]*\}\s*$")


def split_llm_reply(reply, default_vad=(0.6, 0.5, 0.5)):
    """LLM reply (see config/llm_system_prompt.md) -> (tts_text, tagged_text, vad).

    tts_text has tags and the trailing emotion JSON removed (send this to TTS);
    tagged_text keeps the tags (send this to the motion planner).
    """
    import json
    vad = default_vad
    m = VAD_LINE.search(reply.strip())
    if m:
        try:
            e = json.loads(m.group(0))
            vad = tuple(float(min(max(e.get(k, d), 0), 1)) for k, d in zip(("valence", "arousal", "dominance"), default_vad))
        except ValueError:
            pass
        reply = reply.strip()[:m.start()].strip()
    tts = re.sub(r"\s{2,}", " ", TAG_RE.sub(" ", reply)).strip()
    return tts, reply, vad
