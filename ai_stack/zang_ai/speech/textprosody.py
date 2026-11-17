"""Text -> speech timeline, without audio.

Predicts when every word and syllable will be spoken, and which syllables carry the accent,
directly from the LLM's reply text. This is what lets the gesture planner work *before* (or
without) the TTS audio, and it is all deterministic arithmetic:

  syllables      vowel-group syllabifier (silent-e, -le, -ed, -es rules; works for romanised Hinglish)
  lexical stress English suffix/prefix rules, else first syllable; function words unstressed
  phrasing       punctuation -> intonation phrases; '?' marks a question
  accents        nuclear accent = last content word of the phrase (strongest), onset accent = first
                 content word, emphasis from '!', ALL CAPS, *word* and intensifiers
  durations      d_syl = (60 / rate_spm) * k_stress * k_function * k_final        (Klatt-style factors)
                 pause after , ; : . ! ? = 0.22 0.30 0.30 0.45 0.45 0.50 s, scaled by speaking rate
  fit            optional: scale to the real TTS duration, or warp to TTS word timestamps

Returns the same `analysis` dict the planner/reward consume (syllables with prominence, phrases,
pauses, a synthetic loudness envelope) plus timed words.
"""
import re

import numpy as np

FUNCTION = set("""a an the and or but if so as of to in on at by for from with into onto than then
is am are was were be been being do does did have has had will would can could shall should may might must
i me my we us our you your he him his she her it its they them their this that these those there here
not no yes oh um uh just also very too well
hai ho hain ka ki ke ko se me mein par aur ya bhi to toh na""".split())
INTENSIFIERS = set("really very so super totally absolutely amazing awesome never always huge everything everyone bilkul bahut".split())
PAUSE = {",": 0.22, ";": 0.30, ":": 0.30, ".": 0.45, "!": 0.45, "?": 0.50, "—": 0.25, "-": 0.0, "।": 0.45, "॥": 0.50}
SUFFIX_PENULT = ("tion", "sion", "ic", "ical", "ity", "ian", "ial", "ious", "eous", "ient", "ency")
PREFIX_SECOND = ("be", "re", "de", "a", "ex", "com", "con", "pro", "pre", "un", "in", "en", "for")
VOWELS = "aeiouy"
# frequent conversational words the rules get wrong: word -> (syllables, stressed syllable index)
LEXICON = {w: (int(n), int(k)) for w, n, k in (x.split(":") for x in """
something:2:0 someone:2:0 sometimes:2:0 everyone:3:0 everything:3:0 everybody:4:0 every:2:0 anything:3:0
anyone:3:0 nothing:2:0 maybe:2:0 people:2:0 business:2:0 different:3:0 evening:2:0 family:3:0 interesting:3:0
amazing:3:1 another:3:1 about:2:1 around:2:1 again:2:1 awesome:2:0 together:3:1 tomorrow:3:1 today:2:1
hello:2:1 okay:2:1 ok:2:1 robot:2:0 robots:2:0 idea:3:1 ideas:3:1 area:3:0 really:2:0 actually:4:0 usually:4:0
favourite:3:0 favorite:3:0 beautiful:3:0 wonderful:3:0 fantastic:3:1 excited:3:1 exciting:3:1 begin:2:1
before:2:1 because:2:1 believe:2:1 remember:3:1 forget:2:1 understand:3:2 computer:3:1 electronics:4:2
namaste:3:1 dosti:2:0 accha:2:0 theek:1:0 bahut:2:1 zang:1:0 yudu:2:0 science:2:0 create:2:1 machine:2:1
""".split())}


def syllables(word):
    """Count syllables and return approximate syllable boundaries (character spans)."""
    digits = re.sub(r"\D", "", word)
    if digits and not re.search(r"[a-zA-Z]", word):  # "2026" is spoken "twenty twenty-six"
        return max(1, round(1.5 * len(digits)))
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1                                   # silent e: make, time
    if w.endswith("ed") and not w.endswith(("ted", "ded")) and n > 1:
        n -= 1                                   # walked
    if w.endswith("es") and not w.endswith(("ses", "zes", "ces", "ges", "xes")) and n > 1:
        n -= 1                                   # makes
    return max(1, n)


def lexical_stress(word, n):
    """Index of the stressed syllable."""
    w = word.lower()
    if n <= 1:
        return 0
    if w.endswith(SUFFIX_PENULT):
        return max(0, n - 2)
    if w.endswith(("ee", "eer", "ese", "ique", "oon")):
        return n - 1
    if n == 2 and w.startswith(PREFIX_SECOND) and len(w) > 4 and w not in ("people", "robot", "other", "into"):
        return 1
    return 0


def tokenize(text):
    """-> list of (word, trailing_punct, emphasis_flag)."""
    out = []
    for m in re.finditer(r"(\*?)([\w'’̀-ͯऀ-ॣ०-෿]+)(\*?)([,;:.!?—।॥]*)", text):
        star, w, _, punct = m.groups()
        caps = len(w) > 1 and w.isupper()
        out.append((w, punct, bool(star) or caps))
    return out


PHRASE_END = ",;:.!?—।॥"
SYLLABLES_PER_WORD = 1.45      # conversational English
REFERENCE_WPM = 150.0          # the pause table is calibrated at this rate
NUCLEUS_FRACTION = 0.35        # the vowel sits about 35% into a syllable
LOUDNESS_HOP_S = 0.01


def _split_phrases(toks):
    """Token indices grouped into intonation phrases: a phrase ends at punctuation or at the last word."""
    phrases, current = [], []
    for i, (_, punct, _) in enumerate(toks):
        current.append(i)
        if any(c in punct for c in PHRASE_END) or i == len(toks) - 1:
            phrases.append(current)
            current = []
    return phrases


def _word_syllables(word):
    """(syllable count, index of the stressed syllable): lexicon first, else the rules."""
    letters = re.sub(r"[^a-z]", "", word.lower())
    n, stressed_index = LEXICON.get(letters) or (syllables(word), None)
    if stressed_index is None:
        stressed_index = lexical_stress(word, n)
    return n, stressed_index


def _prominence(word, is_function, stressed, is_nuclear, is_onset, emphasised, exclaim):
    """Accent strength of one syllable: unstressed function words lowest, the phrase's nuclear accent highest."""
    if is_function:
        return -0.8
    if not stressed:
        return -0.6
    prom = 0.6
    if is_nuclear:
        prom += 0.7
    elif is_onset:
        prom += 0.3
    if emphasised or word.lower() in INTENSIFIERS:
        prom += 0.5
    if exclaim:
        prom += 0.25
    return prom


def _time_phrases(toks, phrases, syl_dur, rate_scale, lead_in):
    """Walk the phrases and give every syllable a time and prominence.
    Returns (words, syllables, phrases, pauses, time at the end of the speech)."""
    words, syls, phrase_spans, pauses = [], [], [], []
    t = lead_in
    for phrase in phrases:
        content = [i for i in phrase if toks[i][0].lower() not in FUNCTION]
        nuclear = content[-1] if content else phrase[-1]
        onset = content[0] if content else None
        last_punct = toks[phrase[-1]][1]
        exclaim = "!" in last_punct
        phrase_start = t
        for i in phrase:
            word, punct, emphasised = toks[i]
            n, stressed_index = _word_syllables(word)
            is_function = word.lower() in FUNCTION
            word_start = t
            for k in range(n):
                stressed = (k == stressed_index) and not is_function
                d = syl_dur * (1.3 if stressed else 0.85) * (0.8 if is_function else 1.0)
                if i == phrase[-1] and k == n - 1:
                    d *= 1.4                      # phrase-final lengthening
                prom = _prominence(word, is_function, stressed, i == nuclear, i == onset, emphasised, exclaim)
                syls.append({"t": round(t + NUCLEUS_FRACTION * d, 3), "prom": round(prom, 3), "word": i,
                             "stressed": stressed})
                t += d
            words.append({"w": word + punct, "t0": round(word_start, 3), "t1": round(t, 3)})
        pause = max(PAUSE.get(c, 0) for c in last_punct) if last_punct else 0.0
        phrase_spans.append({"t0": round(phrase_start, 3), "t1": round(t, 3), "question": "?" in last_punct,
                             "rise": 3.0 if "?" in last_punct else -2.0})
        if pause:
            pauses.append([round(t, 3), round(t + pause * rate_scale, 3)])
            t += pause * rate_scale
    return words, syls, phrase_spans, pauses, t


def _time_warp(words, duration, lead_in, fit_duration, word_times, anchors):
    """Function mapping predicted times to real ones, or None. Priority: anchors, word times, total duration."""
    if anchors:  # [(predicted_t, real_t)] e.g. sentence starts/ends measured in the TTS audio
        src = np.array([a for a, _ in anchors], float)
        dst = np.array([b for _, b in anchors], float)
        order = np.argsort(src)
        src, dst = src[order], dst[order]
        return lambda x: np.interp(x, src, dst)
    if word_times and len(word_times) == len(words):
        src = np.array([w["t0"] for w in words] + [words[-1]["t1"]])
        dst = np.array([w["t0"] for w in word_times] + [word_times[-1]["t1"]], float)
        if dst.max() > duration * 5:
            dst = dst / 1000.0  # ms
        return lambda x: np.interp(x, src, dst, left=None, right=None)
    if fit_duration:
        scale = (fit_duration - lead_in - 0.3) / max(duration - lead_in - 0.3, 1e-3)
        return lambda x: lead_in + (np.asarray(x) - lead_in) * scale
    return None


def _loudness_and_speech(syls, phrase_spans, duration):
    """Synthetic loudness envelope (a bump per syllable, height by prominence) and the speech mask."""
    tt = np.arange(0, duration, LOUDNESS_HOP_S)
    loud = np.zeros_like(tt)
    for s in syls:
        loud += (0.45 + 0.25 * np.clip(s["prom"], -1, 1.5)) * np.exp(-0.5 * ((tt - s["t"]) / 0.06) ** 2)
    loud = np.clip(loud / max(loud.max(), 1e-6), 0, 1)
    speech = np.zeros_like(tt, bool)
    for span in phrase_spans:
        speech |= (tt >= span["t0"]) & (tt <= span["t1"])
    return tt, loud, speech


def predict(text, rate_wpm=150.0, lead_in=0.15, fit_duration=None, word_times=None, anchors=None, duration_override=None):
    toks = tokenize(text)
    if not toks:
        return None
    syl_dur = 60.0 / (rate_wpm * SYLLABLES_PER_WORD)
    rate_scale = REFERENCE_WPM / rate_wpm
    words, syls, phrase_spans, pauses, t_end = _time_phrases(toks, _split_phrases(toks), syl_dur, rate_scale, lead_in)
    duration = t_end + 0.3

    warp = _time_warp(words, duration, lead_in, fit_duration, word_times, anchors)
    if warp:
        rounded = lambda x: round(float(warp(x)), 3)  # noqa: E731
        for s in syls:
            s["t"] = rounded(s["t"])
        for w in words:
            w["t0"], w["t1"] = rounded(w["t0"]), rounded(w["t1"])
        for span in phrase_spans:
            span["t0"], span["t1"] = rounded(span["t0"]), rounded(span["t1"])
        pauses = [[rounded(a), rounded(b)] for a, b in pauses]
        duration = duration_override or fit_duration or (words[-1]["t1"] + 0.3)

    tt, loud, speech = _loudness_and_speech(syls, phrase_spans, duration)
    speech_time = sum(span["t1"] - span["t0"] for span in phrase_spans)
    return {
        "source": "text", "duration": round(float(duration), 3), "hop": LOUDNESS_HOP_S, "t": tt, "loud": loud,
        "db": 20 * np.log10(loud + 1e-3), "f0": np.zeros_like(tt), "speech": speech, "f0_median": 0.0,
        "syllables": [{"t": s["t"], "prom": s["prom"], "db": 0.0, "f0": 0.0, "word": s["word"],
                       "stressed": s["stressed"]} for s in syls],
        "pauses": pauses, "phrases": phrase_spans, "speech_ratio": round(float(speech.mean()), 3),
        "syllable_rate": round(len(syls) / max(speech_time, 1e-3), 2), "words": words,
        "rate_wpm": rate_wpm,
    }
