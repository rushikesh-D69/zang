"""Face director: mood + co-speech gestures -> face keyframes for the chest TFT, on the motion's clock.

    keys = face_track(result["events"], vad, duration)
    -> [{"t", "emotion", "id", "intensity", "blend_ms"}]   (t in seconds; id = ZLink FACE_SET / FACE_TRACK emotion id)

Layers (config/faces.json): a base expression from the LLM's mood (valence/arousal), then a gesture-specific
expression over each semantic gesture (onset 0.25 s before the stroke apex, held ~1.3 s), then back to the base.
Talking (mouth) and blinking are animated by the renderers themselves, not keyed here.
"""
import json

from .. import paths

LIB = json.load(open(paths.config("faces.json"), encoding="utf-8"))
EMOTIONS = [e["name"] for e in LIB["emotions"]]
LEAD, HOLD = 0.25, 1.3


def mood_emotion(vad):
    v, a, d = (list(vad) + [0.5, 0.5, 0.5])[:3]
    env = {"valence": v, "arousal": a, "dominance": d, "true": True}
    for rule in LIB["from_mood"]:
        # rules are tiny boolean expressions over valence/arousal/dominance from our own config file
        if eval(rule["if"], {"__builtins__": {}}, env):  # noqa: S307
            return rule["emotion"]
    return "neutral"


def _key(t, emotion, intensity, blend_ms):
    return {"t": round(max(0.0, t), 3), "emotion": emotion, "id": EMOTIONS.index(emotion),
            "intensity": round(float(min(max(intensity, 0.0), 1.0)), 3), "blend_ms": int(blend_ms)}


def face_track(events, vad=(0.6, 0.5, 0.5), duration=None, final=True):
    v, a = vad[0], vad[1]
    base = mood_emotion(vad)
    base_i = 0.55 + 0.45 * min(1.0, abs(v - 0.5) * 2 + max(0.0, a - 0.5))
    keys = [_key(0.0, base, base_i, 300)]
    spans = []  # accepted (start, end, emotion)
    # semantic gestures claim the face first; short head cues (question, emphasis, nod) only fill the gaps
    for kind, hold in (("semantic", HOLD), ("head", 0.8)):
        for e in sorted((e for e in events if e.get("type") == kind), key=lambda e: e["t"]):
            emo = LIB["from_gesture"].get(e.get("gesture"))
            if not emo or emo == base:
                continue
            t_on, t_off = e["t"] - LEAD, e["t"] - LEAD + hold
            if any(t_on < b - 0.3 and t_off > a_ + 0.3 for a_, b, _ in spans):
                continue
            spans.append((t_on, t_off, emo))
    spans.sort()
    for n, (t_on, t_off, emo) in enumerate(spans):
        nxt = spans[n + 1][0] if n + 1 < len(spans) else None
        keys.append(_key(t_on, emo, 0.7 + 0.3 * a, 180))
        if nxt is None or nxt > t_off + 0.05:  # touching spans hand over directly, no dip to the base face
            keys.append(_key(t_off, base, base_i, 400))
    keys.sort(key=lambda k: k["t"])
    out = []
    for k in keys:  # a later key at the same time wins; drop redundant repeats
        if out and abs(out[-1]["t"] - k["t"]) < 1e-3:
            out[-1] = k
        elif not out or (out[-1]["emotion"], out[-1]["intensity"]) != (k["emotion"], k["intensity"]):
            out.append(k)
    if final and duration:
        out.append(_key(max(duration - 0.4, out[-1]["t"] + 0.1), "neutral", 1.0, 500))
    return out
