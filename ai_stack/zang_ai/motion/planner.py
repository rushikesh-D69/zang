"""Co-speech gesture planner: prosody + semantic events + emotion -> ANATOMICAL trajectories (deg, 50 Hz).

Ported from ZingAIStack (zingmotion/planner.py). Zang changes: output is the anatomical array of
zang_ai.robot.ANAT (arms, gaze, sway), retargeted to joints by RobotModel.anat_to_q; the head layer drives
a waist gaze turn (Zang has no head); Zing library-clip replay and the ankle-pitch feed-forward are removed
(Zang has no recorded library and no ankle-pitch servo).

Layers (each is how human co-speech motion is organised, after Kendon/McNeill):
  1. Gesture units  - per phrase the hands rise into 'gesture space' and only return to rest at
                      real pauses, so the robot does not bob up/down on every beat.
  2. Beats          - stroke apex lands on stressed syllables (command is sent `lead` early to
                      cover servo lag; the optimiser tunes `lead` against the servo model).
  3. Semantic       - ~35 gestures (config/gestures.json) from tags + lexicon + the HF model, each with
                      several visually different variants; a variant is never used twice in a row.
  4. Head           - gaze shift at phrase starts, accents on stresses, question turn, no-shake.
  5. Torso sway     - slow weight shift per phrase, lagging the arms (biological inertia).
  6. Idle           - breathing, so the robot is never frozen.
Every key-to-key move uses a minimum-jerk profile; a move that would exceed joint speed is
started earlier (look-ahead) instead of arriving late.
"""
from dataclasses import asdict, dataclass

import numpy as np

from ..robot import ANAT
from ..speech.text import VOCAB, resolve

FPS = 50

# Arm poses are (shoulder pitch, shoulder roll, elbow) in anatomical degrees:
# pitch + = arm forward/up, roll + = arm outward, elbow + = flexed (the forearm swings toward the
# midline: inferred from the library's Waving/Salute poses). Rest = home pose.
REST = np.array([0.0, 0.0, 0.0])  # home pose = URDF zero: arms straight down
READY = np.array([36.0, 12.0, 55.0])    # hands up in front of the body, between beats
COMPANION = np.array([26.0, 8.0, 30.0])  # the non-gesturing hand while the other one talks

# beat families: (prep offset, apex offset) relative to READY, scaled by beat size
# (all collision-checked against the body in MuJoCo for sizes 0.6..1.3: the elbow bends inward, so
#  the forearm only clears the chest when the upper arm is pitched forward enough)
BEATS = {
    "baton":     (np.array([14, 0, 8]), np.array([-8, 0, -10])),
    "palm_open": (np.array([8, -2, 6]), np.array([-4, 12, -10])),
    "chop":      (np.array([18, 0, 10]), np.array([-10, 0, -6])),
    "precision": (np.array([6, -2, 6]), np.array([-2, 2, -4])),
}

# semantic gestures come from config/gestures.json (see text.VOCAB). Each gesture is a list of variants: pose
# gestures (+ optional oscillation / path / head pulses / weight-shift wiggle) and library-action gestures
# (Zing only; dropped from the Zang vocabulary)
_INHERIT = ("arms", "pose", "pose_l", "hold", "prep", "head", "head_osc", "osc", "path", "sway", "action", "head_only")


def _variant_table():
    from ..speech.text import VOCAB
    out = {}
    for g, v in VOCAB.items():
        base = {k: v[k] for k in _INHERIT if k in v}
        out[g] = [{**base, **x} for x in v.get("variants", [])] or [{"id": g, **base}]
    return out


VARIANTS = _variant_table()


# torso sway: -1..1, retargeted by RobotModel to counter-rotating hip_roll / ankle_roll (feet stay flat);
# amplitude in degrees at sway = 1 is config/motion.json sway.deg.


@dataclass
class Params:
    amp: float = 1.0          # overall gesture size
    stress_thr: float = 0.45  # syllable prominence (z) needed for a beat
    min_gap: float = 0.5      # s between beats
    lead: float = 0.06        # s, command the apex this much before the syllable (servo lag)
    prep: float = 0.32        # s, preparation move
    stroke: float = 0.18      # s, stroke move (prep pose -> apex)
    hold: float = 0.12        # s, post-stroke hold
    rest_pause: float = 0.6   # pauses longer than this send the hands back to rest
    both_ratio: float = 0.2   # share of strongest beats done with both hands
    head_gain: float = 1.0
    sway_gain: float = 0.6
    energy: float = 0.5       # persona: > 0.5 prefers big gesture variants, < 0.5 small ones
    seed: int = 0

    BOUNDS = {"amp": (0.6, 1.25), "stress_thr": (0.1, 1.2), "min_gap": (0.35, 0.8), "lead": (0.0, 0.16),
              "prep": (0.2, 0.45), "stroke": (0.12, 0.28), "hold": (0.05, 0.25), "rest_pause": (0.35, 1.0),
              "both_ratio": (0.0, 0.45), "head_gain": (0.5, 1.5), "sway_gain": (0.0, 1.0)}

    def as_dict(self):
        return asdict(self)


class Track:
    """Keyframes for a group of DOFs. Segments are quintic Hermite curves: a normal key is a stop
    (minimum-jerk), a `via` key is passed through with a Catmull-Rom velocity (no stop), which is
    how a preparation flows into its stroke."""

    def __init__(self, pose, vmax):
        self.t = [0.0]
        self.v = [np.asarray(pose, float)]
        self.via = [False]
        self.vmax = np.asarray(vmax, float)

    def hold_until(self, t):
        if t > self.t[-1] + 0.02:
            self.t.append(t)
            self.v.append(self.v[-1].copy())
            self.via.append(False)

    def min_dur(self, a, b):
        return 1.875 * float(np.max(np.abs(np.asarray(b) - a) / self.vmax)) * 1.1

    def move(self, t, pose, natural, via=False):
        """Arrive at `pose` at time t. Starts `natural` seconds before, or earlier if speed demands it."""
        pose = np.asarray(pose, float)
        need = max(natural, self.min_dur(self.v[-1], pose))
        self.hold_until(t - need)
        if t <= self.t[-1] + 0.02:  # no time left: arrive as soon as possible (safety + reward handle it)
            t = self.t[-1] + max(0.04, self.min_dur(self.v[-1], pose))
        self.t.append(t)
        self.v.append(pose)
        self.via.append(via)
        return t

    def sample(self, times):
        T = np.asarray(self.t)
        V = np.asarray(self.v)
        vel = np.zeros_like(V)
        for k in range(1, len(T) - 1):
            if self.via[k]:
                vel[k] = 0.8 * 0.5 * ((V[k] - V[k - 1]) / (T[k] - T[k - 1]) + (V[k + 1] - V[k]) / (T[k + 1] - T[k]))
        i = np.clip(np.searchsorted(T, times, side="right") - 1, 0, len(T) - 1)
        j = np.minimum(i + 1, len(T) - 1)
        h = np.where(j > i, T[j] - T[i], 1.0)[:, None]
        u = np.clip((times - T[i]) / h[:, 0], 0, 1)[:, None]
        u3, u4, u5 = u ** 3, u ** 4, u ** 5
        h0 = 1 - 10 * u3 + 15 * u4 - 6 * u5
        h1 = u - 6 * u3 + 8 * u4 - 3 * u5
        h4 = -4 * u3 + 7 * u4 - 3 * u5
        h5 = 10 * u3 - 15 * u4 + 6 * u5
        return h0 * V[i] + h1 * vel[i] * h + h4 * vel[j] * h + h5 * V[j]


def _size(v):
    """How big a variant looks, ~0 (small, close to rest) .. 1 (large, fast, both arms)."""
    if v.get("action"):
        return 0.8
    if "pose" not in v:
        return 0.2
    reach = np.abs(np.array(v["pose"], float) - REST).sum() / 200.0
    return float(np.clip(reach * (1.3 if v.get("arms") == "both" else 1.0) + 0.2 * ("osc" in v) + 0.1 * bool(v.get("sway")), 0, 1))


def _bump(times, t0, width, amp):
    """Smooth additive pulse (raised cosine) centred at t0."""
    u = (times - t0) / width
    return np.where(np.abs(u) < 1, amp * 0.5 * (1 + np.cos(np.pi * u)), 0.0)


BREATH_HZ = 0.23  # resting breathing rate


class _PlanRun:
    """Everything one Planner.plan() call builds up; the planning steps read and extend it in order."""

    def __init__(self, analysis, semantic, vad, p, state, final):
        self.analysis, self.semantic, self.p = analysis, semantic, p
        self.vad, self.state, self.final = vad, state, final
        self.valence, self.arousal, self.dominance = vad
        self.rng = np.random.default_rng(p.seed)
        self.duration = analysis["duration"] + (0.8 if final else 0.0)
        times = np.arange(0, self.duration, 1.0 / FPS)
        if self.duration - times[-1] > 1e-6:
            times = np.append(times, self.duration)  # end exactly on the chunk boundary: no drift across sentences
        self.times = times
        self.t_off = state["t"] if state else 0.0
        # emotion -> style
        self.size = p.amp * (0.55 + 0.6 * self.arousal)
        self.tempo = 1.15 - 0.3 * self.arousal
        self.ready = READY + np.array([8 * (self.dominance - 0.5) * 2, 6 * (self.valence - 0.5) * 2, 0])
        # filled by the planning steps
        self.events = []
        self.busy = {"l": [], "r": []}            # time spans each arm is already committed to
        self.sem_strokes, self.head_cues, self.clips = [], [], []
        self.recent = {g: list(v) for g, v in (state or {}).get("recent", {}).items()}
        self.last_hand = (state or {}).get("hand", "l")
        self.beats, self.beat_strokes = [], []
        self.arm_traj = None


class Planner:
    """robot: zang_ai.robot.RobotModel. Works in anatomical degrees; see robot.ANAT for the column order."""

    def __init__(self, robot):
        self.r = robot
        gi = robot.idx
        arm_j = [gi["left_shoulder_pitch"], gi["left_shoulder_roll"], gi["left_elbow"]]
        self.vmax_arm = np.degrees(robot.vmax[arm_j])          # deg/s, anatomical (pitch, roll, elbow)
        self.vmax_gaze = float(np.degrees(robot.vmax[gi["waist_yaw"]]))
        self.gaze_gain = robot.motion["gaze"]["gain"]
        self.feasible = None  # optional (side, pose) -> collision-free pose, set by the pipeline

    def plan(self, analysis, semantic, vad=(0.6, 0.5, 0.5), p=None, state=None, final=True):
        """state: continue from a previous sentence (streaming): arm/head/sway values and breathing phase at
        the end of the previous chunk. final=False: more speech follows, so hands stay in gesture space at the
        end and there is no settle tail (chunk motion = chunk audio, exactly).

        The steps run in a fixed order because they share one random generator."""
        run = _PlanRun(analysis, semantic, vad, p or Params(), state, final)
        self._place_semantic_gestures(run)
        self._place_beats(run)
        self._compile_arms(run)
        head_traj, head_end = self._plan_head(run)
        sway = self._plan_sway(run)
        return self._compose(run, head_traj, head_end, sway)

    # ---- semantic gestures claim the arms ---------------------------------------------------------
    def _place_semantic_gestures(self, run):
        """Explicit LLM tags claim the arms first, then model / lexicon cues fill the gaps."""
        for e in sorted(run.semantic, key=lambda e: (e.get("source") != "tag", e["t"])):
            if e["gesture"].startswith("act:"):
                continue  # Zing library clips: Zang has no recorded action library yet
            gesture, want = resolve(e["gesture"])
            if gesture is None:
                continue
            spec = VOCAB.get(gesture, {})
            if run.arousal < spec.get("min_arousal", 0):
                gesture, want = spec.get("fallback", gesture), None
            variant = self._pick(gesture, want, run.recent, run.rng, run.p.energy)
            if variant is None or variant.get("action"):
                continue
            variant_id = variant["id"]
            t_apex = self._apex_time(run, e)
            if "pose" not in variant:  # head-only gesture (no / yes / question / emphasis)
                run.head_cues.append((e["t"], gesture, variant, 1))
                self._used(run.recent, gesture, variant_id)
            else:
                self._place_pose_gesture(run, e, gesture, variant, t_apex)

    @staticmethod
    def _apex_time(run, cue):
        """The cue's stroke lands on its most prominent nearby syllable, `lead` early to cover servo lag."""
        near = [s for s in run.analysis["syllables"] if -0.15 <= s["t"] - cue["t"] <= 0.45]
        syllable_t = max(near, key=lambda s: s["prom"])["t"] if near else cue["t"] + 0.1
        return syllable_t - run.p.lead

    @staticmethod
    def _overlaps(spans, start, end, margin=0.0):
        return any(not (end + margin < s0 or start - margin > s1) for s0, s1 in spans)

    def _place_pose_gesture(self, run, cue, gesture, variant, t_apex):
        """Reach a pose with one or both hands on the stressed syllable (skipped if no hand is free)."""
        p = run.p
        hold = float(variant.get("hold", 0.45))
        extra = (variant["osc"]["period"] * variant["osc"]["cycles"] if "osc" in variant else 0) + \
                (variant["path"]["period"] * len(variant["path"]["poses"]) if "path" in variant else 0)
        if cue.get("source") == "tag":  # explicit LLM tag (rhymes tag every line): reserve only the real wind-up + settle
            span = (t_apex - variant.get("prep", max(p.prep, 0.35)) - 0.1, t_apex + hold + extra + 0.15)
        else:
            span = (t_apex - 0.8, t_apex + hold + extra + 0.3)
        free = [side for side in "lr" if not self._overlaps(run.busy[side], *span)]
        if variant["arms"] == "either":  # whichever hand is free, alternating, so it is not always the right one
            sides = ("l" if run.last_hand == "r" else "r") if len(free) == 2 else (free[0] if free else None)
        else:
            sides = "lr" if variant["arms"] == "both" else variant["arms"]
        if not sides or any(side not in free for side in sides):
            return
        for side in sides:
            run.busy[side].append(span)
        if len(sides) == 1:
            run.last_hand = sides
        poses = {"r": np.array(variant["pose"], float), "l": np.array(variant.get("pose_l", variant["pose"]), float)}
        head_turn = -1 if sides == "l" else 1  # head turns follow the hand used
        run.sem_strokes.append((t_apex, gesture, sides, poses, hold, variant))
        run.head_cues.append((cue["t"], gesture, variant, head_turn))
        self._used(run.recent, gesture, variant["id"])
        run.events.append({"type": "semantic", "t": round(t_apex + p.lead, 3), "gesture": gesture,
                           "variant": variant["id"], "word": cue["word"], "arms": sides,
                           "apex": {side: poses[side].tolist() for side in sides} if "osc" not in variant else None})

    # ---- beats on stressed syllables ------------------------------------------------------------
    def _pick_beats(self, run):
        """The most prominent stressed syllables, at least min_gap apart, in time order."""
        p = run.p
        stressed = [s for s in run.analysis["syllables"] if s["prom"] >= p.stress_thr]
        beats = []
        for s in sorted(stressed, key=lambda s: -s["prom"]):
            if all(abs(s["t"] - b["t"]) >= p.min_gap for b in beats):
                beats.append(s)
        beats.sort(key=lambda s: s["t"])
        return beats

    def _place_beats(self, run):
        p, rng, ready = run.p, run.rng, run.ready
        beats = run.beats = self._pick_beats(run)
        proms = np.array([b["prom"] for b in beats]) if beats else np.zeros(0)
        both_cut = np.quantile(proms, 1 - p.both_ratio) if len(proms) and p.both_ratio > 0 else np.inf
        emphasis_times = [e["t"] for e in run.semantic if e["gesture"].startswith("emphasis")]
        last_side, last_t = "l", -9
        phrase_shape = {}
        for beat in beats:
            t_apex = beat["t"] - p.lead
            phrase = next((i for i, q in enumerate(run.analysis["phrases"])
                           if q["t0"] - 0.3 <= beat["t"] <= q["t1"] + 0.3), -1)
            if phrase not in phrase_shape:  # one or two beat forms per gesture unit ('catchment')
                weights = np.array([0.5 + run.dominance, 0.5 + run.valence, 0.3 + run.dominance, 1.3 - run.arousal])
                phrase_shape[phrase] = list(rng.choice(list(BEATS), size=2, p=weights / weights.sum()))
            shape = phrase_shape[phrase][int(rng.random() < 0.3)]
            emphatic = any(0 <= beat["t"] - t <= 0.6 for t in emphasis_times)
            want_both = beat["prom"] >= both_cut or emphatic
            free = [side for side in "lr"
                    if not any(b0 - 0.3 <= t_apex <= b1 + 0.3 for b0, b1 in run.busy[side])]
            if not free:
                continue
            if want_both and len(free) == 2:
                sides = "lr"
            else:
                preferred = "r" if rng.random() < 0.6 else "l"
                if t_apex - last_t < 0.7 and len(free) == 2:
                    preferred = "l" if last_side == "r" else "r"
                sides = preferred if preferred in free else free[0]
            size = run.size * float(np.clip(0.75 + 0.25 * beat["prom"], 0.6, 1.3)) * (1.3 if emphatic else 1.0)
            run.beat_strokes.append((t_apex, shape, sides, size))
            last_side, last_t = sides[-1], t_apex
            run.events.append({"type": "beat", "t": beat["t"], "shape": shape, "arms": sides, "prom": beat["prom"],
                               "size": round(size, 2),
                               "apex": {side: (ready + BEATS[shape][1] * size).round(2).tolist() for side in sides}})

    # ---- arm keyframes -------------------------------------------------------------------------
    def _collect_strokes(self, run):
        """Per arm: (kind, t_apex, name, arg, hold) for every semantic stroke, beat and companion movement."""
        p, phrases = run.p, run.analysis["phrases"]
        strokes = {"l": [], "r": []}
        for t_apex, _, sides, poses, hold, variant in run.sem_strokes:
            for side in sides:
                strokes[side].append(("sem", t_apex, variant, poses[side].copy(), hold))
        for t_apex, shape, sides, size in run.beat_strokes:
            for side in sides:
                strokes[side].append(("beat", t_apex, shape, size, p.hold))
        # the quiet hand keeps company in long phrases where the other hand beats a lot
        for side, other in (("l", "r"), ("r", "l")):
            for q in phrases:
                inside = lambda stroke: q["t0"] - 0.5 <= stroke[1] <= q["t1"] + 0.5  # noqa: E731
                if q["t1"] - q["t0"] >= 1.5 and not any(map(inside, strokes[side])) \
                        and sum(map(inside, strokes[other])) >= 2:
                    strokes[side].append(("companion", q["t0"] + 0.2, None, q, None))
        return strokes

    def _compile_arms(self, run):
        """Turn strokes into minimum-jerk arm tracks, then sample them at 50 Hz into run.arm_traj."""
        p = run.p
        strokes = self._collect_strokes(run)
        pause_spans = [(a, b) for a, b in run.analysis["pauses"] if b - a >= p.rest_pause]
        vmax_arm = self.vmax_arm
        tracks = {side: Track(run.state["arms"][side] if run.state else REST, vmax_arm) for side in "lr"}
        for side in "lr":
            self._play_arm(run, side, tracks[side], sorted(strokes[side], key=lambda x: x[1]), pause_spans)
        run.arm_traj = {side: tracks[side].sample(run.times) for side in "lr"}

    def _play_arm(self, run, side, track, strokes, pause_spans):
        if not strokes and np.abs(track.v[0] - REST).max() > 2:  # arm left raised by the previous sentence: relax it
            track.move(1.0, REST, 0.8)
        for n, stroke in enumerate(strokes):
            t_end, end_pose = self._play_stroke(run, side, track, stroke)
            next_apex = strokes[n + 1][1] if n + 1 < len(strokes) else None
            self._settle_after(run, track, t_end, end_pose, next_apex, pause_spans)

    def _play_stroke(self, run, side, track, stroke):
        """Move the arm through one stroke. Returns (time the stroke ends, pose it ends in)."""
        p, ready = run.p, run.ready
        kind, t_apex, name, arg, hold = stroke
        if kind == "beat":
            prep_offset, apex_offset = BEATS[name]
            prep_pose = self._ok(side, ready + prep_offset * arg)
            apex_pose = self._ok(side, ready + apex_offset * arg)
            stroke_s = max(p.stroke * run.tempo, track.min_dur(prep_pose, apex_pose))
            track.move(t_apex - stroke_s, prep_pose, p.prep * run.tempo, via=True)
            t_end = track.move(t_apex, apex_pose, stroke_s)
            track.hold_until(t_end + hold)
            return t_end, apex_pose
        if kind == "companion":
            track.move(max(t_apex, track.t[-1] + 0.3), REST + (COMPANION - REST) * min(1.0, run.size), 0.5)
            track.hold_until(arg["t1"])
            return track.t[-1], track.v[-1]
        t_end = self._semantic(track, side, name, t_apex, self._ok(side, arg), hold, p, run.tempo)
        return t_end, track.v[-1]

    def _settle_after(self, run, track, t_end, end_pose, next_apex, pause_spans):
        """After a stroke: stay in gesture space if another stroke is soon, else retract to rest."""
        p, ready, tempo = run.p, run.ready, run.tempo
        gap_has_pause = next_apex is not None and any(t_end < a and b < next_apex for a, b in pause_spans)
        if next_apex is None and not run.final:  # more sentences follow: stay in gesture space for the next one
            track.hold_until(track.t[-1] + 0.15)
            track.move(min(t_end + 0.6, max(track.t[-1] + 0.3, run.duration - 0.05)), 0.5 * (end_pose + ready), 0.4)
        elif next_apex is None or gap_has_pause or next_apex - t_end > p.rest_pause + 1.2:
            track.hold_until(track.t[-1] + 0.15)
            track.move(track.t[-1] + 0.55 * tempo, REST, 0.55 * tempo)
        elif next_apex - t_end > 0.9:
            track.move(min(t_end + 0.6, next_apex - 0.6), 0.5 * (end_pose + ready), 0.4)

    # ---- head ----------------------------------------------------------------------------------
    def _head_keys(self, run):
        """Head targets (t, target, natural, hold, prio): phrase gaze shifts, then gesture head poses which win."""
        phrases, rng, gain = run.analysis["phrases"], run.rng, run.p.head_gain
        side = 1 if rng.random() < 0.5 else -1
        keys = []
        for q in phrases:
            side = -side
            keys.append((q["t0"] - 0.05, side * rng.uniform(4, 10) * gain, 0.2, 0, 0))
            if q.get("question"):
                keys.append((q["t1"] - 0.15, 9 * gain, 0.3, 0.5, 1))
            next_pause = [a for a, b in run.analysis["pauses"] if a >= q["t1"] - 0.05 and b - a > 1.0]
            if next_pause:
                keys.append((next_pause[0] + 0.7, 0.0, 0.6, 0, 0))
        for t, gesture, variant, turn in run.head_cues:
            if variant.get("head"):
                if gesture == "question" and any(q.get("question") and abs(q["t1"] - t) < 0.6 for q in phrases):
                    continue
                keys.append((t - 0.1, turn * variant["head"] * gain, 0.3, max(0.5, float(variant.get("hold", 0.5))), 1))
            if variant.get("head") or variant.get("head_osc"):
                run.events.append({"type": "head", "t": round(t, 3), "gesture": gesture, "variant": variant["id"]})
        return keys

    def _plan_head(self, run):
        """Head pan track in degrees at 50 Hz, and its final value (for the next streamed sentence)."""
        p, times, gain = run.p, run.times, run.p.head_gain
        head = Track([run.state["head"] if run.state else 0.0], [self.vmax_gaze / max(self.gaze_gain, 1e-3)])
        busy_until = -1.0
        for t, target, natural, hold, priority in sorted(self._head_keys(run), key=lambda k: k[0]):
            if priority == 0 and t < busy_until:
                continue
            t_arrive = head.move(max(t, head.t[-1] + 0.05), [target], natural)
            if hold:
                head.hold_until(t_arrive + hold)
                busy_until = t_arrive + hold
        head_traj = head.sample(times)[:, 0]
        for beat in run.beats:  # small accents toward the current gaze direction
            gaze = np.interp(beat["t"], times, head_traj)
            accent = -np.sign(gaze or 1) * 2.5 * gain * min(1.3, 0.7 + 0.3 * beat["prom"])
            head_traj += _bump(times, beat["t"] - p.lead, 0.22, accent)
        for t, _, variant, turn in run.head_cues:  # shakes, nods, giggles: additive pan pulses
            for dt, amplitude in variant.get("head_osc", []):
                head_traj += _bump(times, t + dt, 0.16, turn * amplitude * gain)
        return head_traj, float(head.sample(times[-1:])[0, 0])

    # ---- torso sway ----------------------------------------------------------------------------
    def _plan_sway(self, run):
        """Slow weight shift per phrase (lagging, damped) in -1..1, plus quick wiggles for shy / happy gestures."""
        p, times, pauses = run.p, run.times, run.analysis["pauses"]
        sway = Track([run.state["sway"] if run.state else 0.0], [0.4])
        direction = 1 if run.rng.random() < 0.5 else -1
        for q in run.analysis["phrases"]:
            direction = -direction
            amplitude = direction * (0.35 + 0.4 * run.arousal) * p.sway_gain
            sway.move(max(q["t0"] + 0.25 + 1.2, sway.t[-1] + 0.3), [amplitude], 1.2)
            long_pause = [a for a, b in pauses if a >= q["t1"] - 0.05 and b - a > 1.2]
            if long_pause:
                sway.move(long_pause[0] + 1.5, [0.0], 1.5)
        shift = sway.sample(times)[:, 0]
        window = np.hanning(int(0.6 * FPS))
        shift = np.convolve(np.pad(shift, (len(window), len(window)), mode="edge"), window / window.sum(),
                            "same")[len(window):-len(window)]
        for t_apex, _, _, _, _, variant in run.sem_strokes:
            for n, amplitude in enumerate(variant.get("sway", [])):
                shift += _bump(times, t_apex + 0.1 + 0.4 * n, 0.35, 0.6 * amplitude)
        return shift

    # ---- compose -------------------------------------------------------------------------------
    def _compose(self, run, head_traj, head_end, sway):
        """Arms + gaze + sway + breathing -> anatomical array (N, 8) in robot.ANAT order, plus the plan dict."""
        times, rng = run.times, run.rng
        A = np.zeros((len(times), len(ANAT)))
        breath_phase = run.state["ph0"] if run.state else rng.uniform(0, 2 * np.pi)
        t_continuous = times + run.t_off  # continuous breathing across streamed sentences
        breath = np.sin(2 * np.pi * BREATH_HZ * t_continuous + breath_phase)
        for side in "lr":
            c = [ANAT.index(f"{side}_{j}") for j in ("pitch", "roll", "elbow")]
            A[:, c] = run.arm_traj[side]
            A[:, c[0]] += 1.8 * breath
            A[:, c[2]] += 0.75 * (1 + np.sin(2 * np.pi * BREATH_HZ * t_continuous + breath_phase + 0.6))
        A[:, ANAT.index("gaze")] = self.gaze_gain * head_traj
        A[:, ANAT.index("sway")] = sway
        for q in run.analysis["phrases"]:
            run.events.append({"type": "phrase", "t0": q["t0"], "t1": q["t1"], "question": q.get("question", False)})
        new_state = {"t": run.t_off + run.analysis["duration"], "arms": {a: run.arm_traj[a][-1].copy() for a in "lr"},
                     "head": head_end, "sway": float(sway[-1]), "ph0": breath_phase,
                     "recent": run.recent, "hand": run.last_hand}
        return {"t": times, "anat": A, "events": run.events, "params": run.p.as_dict(),
                "vad": list(run.vad), "state": new_state}

    def _ok(self, side, pose):
        return self.feasible(side, pose) if self.feasible else pose

    @staticmethod
    def _pick(g, want, recent, rng, energy=0.5):
        """Choose a variant of gesture g: the one asked for (think2 / think:scratch), else a random one that
        was not among the last few used, so repeated tags never look the same twice in a row. The persona's
        energy tilts the odds toward big (energetic) or small (calm, shy) variants."""
        vs = VARIANTS.get(g)
        if not vs:
            return None
        if isinstance(want, int):
            return vs[want]
        if isinstance(want, str):
            return next(x for x in vs if x["id"] == want)
        last = recent.get(g, [])[-(len(vs) - 1):] if len(vs) > 1 else []
        pool = [x for x in vs if x["id"] not in last] or vs
        w = np.array([float(x.get("weight", 1.0)) * np.exp(3.0 * (energy - 0.5) * (_size(x) - 0.5)) for x in pool])
        return pool[int(rng.choice(len(pool), p=w / w.sum()))]

    @staticmethod
    def _used(recent, g, vid):
        recent[g] = (recent.get(g, []) + [vid])[-3:]

    def _semantic(self, tr, side, v, t_apex, pose, hold, p, tempo):
        """Reach the variant's pose on the stressed syllable, then oscillate one joint (wave, beckon, giggle)
        and/or flow through its extra path poses (tap, sweep, count), then hold."""
        tr.move(t_apex - (0.05 if "osc" in v else 0), pose, v.get("prep", max(p.prep, 0.35)) * tempo)
        t = t_apex - (0.05 if "osc" in v else 0)
        osc = v.get("osc")
        if osc:
            j = {"pitch": 0, "roll": 1, "elbow": 2}[osc["joint"]]
            for k in range(osc["cycles"]):
                w = pose.copy()
                w[j] = osc["values"][k % len(osc["values"])]
                t = tr.move(t + osc["period"], self._ok(side, w), osc["period"])
        path = v.get("path")
        if path:
            for q in path["poses"]:
                t = tr.move(t + path["period"] * tempo, self._ok(side, np.array(q, float)), path["period"] * tempo)
        tr.hold_until(t + hold)
        return t + hold
