"""Text (LLM reply with gesture tags) -> co-speech motion for Zang, verified before it can move the robot.

    reply text ─► words + tags ─► timing (text prosody, or exact TTS word/stress times, or any audio + aligner)
              ─► semantic gestures (LLM tags > high-precision words > HF model > lexicon)
              ─► planner (anatomical, 50 Hz) ─► retarget to joints ─► SAFETY STAGE ─► physics twin ─► result

The speech/timing half is ported from ZingAIStack (zingmotion/pipeline.py) unchanged in behaviour.
"""
import re

import numpy as np

from .motion.planner import FPS, Params, Planner
from .robot import ANAT, RobotModel
from .safety.stage import SafetyStage
from .speech import text as T


class Composer:
    def __init__(self, robot=None):
        self.r = robot or RobotModel()
        self.safety = SafetyStage(self.r)
        self.planner = Planner(self.r)
        self.planner.feasible = self._feasible_arm
        self._twin = None

    @property
    def twin(self):
        if self._twin is None:
            from .sim.twin import PhysicsTwin
            self._twin = PhysicsTwin(self.r)
        return self._twin

    def _feasible_arm(self, side, pose):
        """Planner hook: (side, [pitch, roll, elbow] deg) -> nearest collision-free arm pose (other arm at rest)."""
        A = np.zeros(len(ANAT))
        A[[ANAT.index(f"{side}_{j}") for j in ("pitch", "roll", "elbow")]] = pose
        A = np.clip(A, self.r.anat_lo, self.r.anat_hi)
        B = self.safety.col.project_arm(A, side)
        return pose if B is None else B[[ANAT.index(f"{side}_{j}") for j in ("pitch", "roll", "elbow")]]

    # ---- timing context (ported) -------------------------------------------------------------------------
    def context_text(self, text, vad=(0.6, 0.5, 0.5), rate_wpm=150.0, fit_duration=None, word_times=None,
                     use_model=True, model_tags=None):
        """Reply text -> context with no audio: timing from the text prosody model."""
        from .speech import gesture_tagger
        from .speech.textprosody import predict
        words, tags = T.parse(text or "")
        model_tags = list(model_tags or [])
        if not model_tags and use_model and gesture_tagger.available() and words:
            try:
                model_tags = gesture_tagger.tag(words)
            except Exception as e:  # missing torch/transformers -> lexicon only
                print("gesture model unavailable:", e)
        an = predict(" ".join(words), rate_wpm, fit_duration=fit_duration, word_times=word_times)
        if an is None:
            raise ValueError("empty text")
        timed = an["words"] if len(an["words"]) == len(words) else T.align(words, an)
        sem = T.semantic_events(words, tags, timed, model_tags=model_tags)
        return {"analysis": an, "words": timed, "semantic": sem, "vad": tuple(vad), "model_tags": model_tags,
                "tags": tags}

    def context_voice(self, text, vad=(0.6, 0.5, 0.5), voice=None, rate=1.0, use_model=True):
        """Synthesise the reply with the offline TTS and time the motion to that audio. -> (ctx, speech)."""
        from .speech import tts
        words, _ = T.parse(text or "")
        speech = tts.synthesize(" ".join(words), voice, rate)
        lang = "hi" if (speech["voice"] or "").startswith("hi_") else None
        ctx = self.context_speech(text, speech["audio"], speech["sr"], vad, speech["sentences"],
                                  engine_times=[tuple(w) for w in speech["word_times"]] if speech.get("word_times") else None,
                                  stress_times=speech.get("stress_times"), language=lang, use_model=use_model)
        ctx["tts"] = {k: v for k, v in speech.items() if k != "audio"}
        return ctx, speech

    def context_speech(self, text, audio, sr, vad=(0.6, 0.5, 0.5), sentences=None, engine_times=None,
                       language=None, use_model=True, stress_times=None, model_tags=None):
        """Reply text + ANY speech audio of it (our TTS, cloud TTS, a recording)."""
        from .speech import align as AL
        from .speech import audio as A
        from .speech.textprosody import predict
        words, tags = T.parse(text or "")
        clean = " ".join(words)
        dur = len(audio) / sr
        ctx = self.context_text(text, vad, 150.0, use_model=use_model, model_tags=model_tags)
        al = AL.align(words, audio, sr, sentences, engine_times, language)
        if al["words"]:
            wt = al["words"]
            voiced = sum(w["t1"] - w["t0"] for w in wt)
            wpm = 60 * len(words) / max(voiced * 1.15, 1e-3)
            an = predict(clean, wpm, word_times=wt, duration_override=dur)
            if stress_times and len(stress_times) == len(words):
                for syl in an["syllables"]:
                    st = stress_times[syl["word"]] if syl["word"] < len(stress_times) else None
                    if syl.get("stressed") and st is not None:
                        syl["t"] = round(float(st), 3)
        else:
            pw = ctx["analysis"]["words"]
            anchors, wi = [], 0
            for snt in sentences or []:
                n = len(re.findall(r"[\w'’̀-ͯऀ-ॣ०-෿]+", snt["text"]))
                if n == 0 or wi + n > len(pw):
                    break
                anchors += [(pw[wi]["t0"], snt["t0"]), (pw[wi + n - 1]["t1"], snt["t1"])]
                wi += n
            an = predict(clean, 150.0, anchors=anchors or None, fit_duration=None if anchors else dur,
                         duration_override=dur)
        real = A.analyse(audio, sr)
        an["t"], an["loud"] = real["t"], real["loud"]
        an["duration"] = round(dur, 3)
        sem = T.semantic_events(words, tags, an["words"], model_tags=ctx.get("model_tags"))
        ctx.update(analysis=an, words=an["words"], semantic=sem,
                   alignment={k: al[k] for k in ("method", "coverage", "notes")})
        return ctx

    # ---- motion -------------------------------------------------------------------------------------------
    @staticmethod
    def preroll_needed(ctx, wind_up=1.0):
        """Seconds the motion must start before the audio so the first gesture is not late (a wave on 'Hi')."""
        arm = [e["t"] for e in ctx["semantic"] if not T.VOCAB.get(T.resolve(e["gesture"])[0] or "", {}).get("head_only")]
        if not arm:
            return 0.0
        return round(float(np.clip(wind_up - min(arm), 0.0, wind_up)) / 0.02) * 0.02

    @staticmethod
    def shift_context(ctx, dt):
        """Delay the whole speech timeline by dt seconds (leading silence)."""
        if dt <= 0:
            return ctx
        an = dict(ctx["analysis"])
        hop = an.get("hop", 0.01)
        n = int(round(dt / hop))
        if "t" in an:
            an["t"] = np.concatenate([np.arange(n) * hop, np.asarray(an["t"]) + n * hop])
            for k in ("loud", "db", "f0", "speech"):
                if k in an and hasattr(an[k], "__len__"):
                    v = np.asarray(an[k])
                    an[k] = np.concatenate([np.full(n, v[0] if k != "speech" else False, dtype=v.dtype), v])
        an["syllables"] = [{**s, "t": s["t"] + dt} for s in an["syllables"]]
        an["phrases"] = [{**p, "t0": p["t0"] + dt, "t1": p["t1"] + dt} for p in an["phrases"]]
        an["pauses"] = [[0.0, dt]] + [[a + dt, b + dt] for a, b in an["pauses"]]
        an["words"] = [{**w, "t0": w["t0"] + dt, "t1": w["t1"] + dt} for w in an.get("words", [])]
        an["duration"] = an["duration"] + dt
        out = dict(ctx, analysis=an, words=[{**w, "t0": w["t0"] + dt, "t1": w["t1"] + dt} for w in ctx["words"]],
                   semantic=[{**e, "t": e["t"] + dt} for e in ctx["semantic"]])
        return out

    def compose(self, ctx, params=None, physics=True, state=None, final=True, q_start=None, log=print, preroll=None):
        """Context -> planned, safe, (physics-checked) motion. Physics failure shrinks and re-plans (<= 2x).

        preroll: seconds of motion before the audio starts (None = automatic for the first sentence of a reply,
        0 when continuing a stream). The result's 'audio_offset' tells players where the audio begins."""
        if preroll is None:
            preroll = self.preroll_needed(ctx) if state is None else 0.0
        ctx = self.shift_context(ctx, preroll)
        res = self._compose(ctx, params, physics, state, final, q_start, log)
        res["audio_offset"] = preroll
        res["words"] = ctx["words"]
        from .face.expr import face_track
        # faces can show what the arms could not fit: every explicit script tag (the arm planner thins tags closer
        # than 1.2 s apart; a face has no such limit), timed on its word, unless the planner already placed it
        placed = [(e["t"], e["gesture"]) for e in res["events"] if e["type"] == "semantic"]
        extra = []
        for i, g in ctx.get("tags", []):
            name = T.resolve(g)[0]
            if not name or T.VOCAB.get(name, {}).get("head_only") or i >= len(ctx["words"]):
                continue
            t = ctx["words"][i]["t0"] + 0.3  # roughly where the stroke apex would be
            if not any(pg == name and abs(pt - t) < 1.0 for pt, pg in placed):
                extra.append({"type": "semantic", "t": t, "gesture": name})
        res["face"] = face_track(res["events"] + extra, ctx["vad"], float(res["t"][-1]), final)
        return res

    def _compose(self, ctx, params, physics, state, final, q_start, log):
        p = params or Params()
        for attempt in range(3):
            plan = self.planner.plan(ctx["analysis"], ctx["semantic"], ctx["vad"], p, state=state, final=final)
            Q, safety = self.safety.run(plan["anat"], FPS, q_start)
            res = {"t": plan["t"], "anat": plan["anat"], "q": Q, "events": plan["events"], "state": plan["state"],
                   "params": p.as_dict(), "safety": safety, "physics": None, "sim": None}
            if not physics:
                return res
            from .sim.twin import report
            sim = self.twin.simulate(Q, FPS)
            res["sim"], res["physics"] = sim, report(self.twin, sim)
            res["physics"]["attempt"] = attempt
            if log:
                log(f"compose attempt {attempt}: safety {'PASS' if safety['pass'] else 'FAIL'}, "
                    f"physics {'PASS' if res['physics']['pass'] else 'FAIL'} {res['physics']['checks']}")
            if res["physics"]["pass"]:
                return res
            p = Params(**{**p.as_dict(), "amp": p.amp * 0.85, "sway_gain": p.sway_gain * 0.6})
        return res

    def compose_text(self, text, vad=(0.6, 0.5, 0.5), voice=None, physics=True, use_model=True, rate_wpm=150.0):
        """Convenience: text -> (result, speech or None). voice=None: no TTS, timing from text prosody."""
        if voice:
            ctx, speech = self.context_voice(text, vad, voice, use_model=use_model)
        else:
            ctx, speech = self.context_text(text, vad, rate_wpm, use_model=use_model), None
        res = self.compose(ctx, physics=physics)
        res["ctx"] = ctx
        return res, speech
