# Zang AI stack

This stack turns what Zang says into upper-body co-speech motion. It checks that motion against the real robot's
geometry and physics, then plays it in a digital twin and streams it to the robot or exports it to the SD card.

```
make setup && make models     # once
make up                       # studio + emulator in the background -> http://localhost:8790
make status | make logs       # what is running, who holds the ports | last log lines
make stop                     # kill everything this stack started and free ports 8790, 4210-4211, 6006, 9333
make studio                   # or: foreground studio (make app = standalone desktop window)
make test                     # make help lists everything; make rl-help for the RL targets
```

## Pipeline

```
LLM reply with tags ─► words + tags ─► timing ─► semantic gestures ─► planner ─► retarget ─► SAFETY STAGE ─► physics twin
 "Hi! <wave> I'm Zang"   (text prosody, or     (LLM tag > word cue    (anatomical,  (signs from  (limits, collision,  (MuJoCo, servo
                          exact Piper TTS        > HF model > lexicon)   50 Hz, min-   the MJCF)    balance, rate)       latency + speed)
                          word/stress times)                             jerk)
                                                                                         ─► studio │ .zact + .mp3 │ ZLink stream
```

| Stage | Module | Notes |
|---|---|---|
| Speech, timing, gesture choice | `zang_ai/speech/` | Ported from ZingAIStack unchanged in behaviour: Piper TTS, faster-whisper, RoBERTa gesture tagger, text prosody, LLM client (Groq/Mistral/OpenAI) |
| Gesture vocabulary | `config/gestures.json` | 43 gestures, ~100 variants (Zing's set; library-clip variants removed) |
| Planner | `zang_ai/motion/planner.py` | Zing's 6-layer co-speech planner. Output is anatomical degrees; the head layer becomes a waist **gaze** turn (Zang has no head) |
| Robot model | `zang_ai/robot.py` | `config/robot.json` (joint contract) + `zang_description` MJCF. Anatomical ↔ joint signs are **derived from the MJCF** at load |
| Safety stage | `zang_ai/safety/` | see below |
| Physics twin | `zang_ai/sim/twin.py` | floating base, closed parallelogram loops, firmware-style linear ramps, servo latency and speed limit |
| Pre-roll | `pipeline.Composer.compose` | if the first gesture needs wind-up (a wave on "Hi"), motion starts before the audio. `audio_offset` says by how much |
| Export | `zang_ai/motion/export.py` | `.zact` (no frame limit) + MP3, spec in `firmware/docs/ZLINK.md` |
| Robot link | `zang_ai/link/` | ZLink codec, UDP client (streams chunks 0.6 s ahead), software emulator |

## Faces (`config/faces.json`, `zang_ai/face/`)

One parametric face model (eyes, lids, gaze, brows, mouth, blush and symbols) is drawn by one signed-distance
algorithm. It exists three times: `face/render.py` (reference), `studio/web/face.js` (studio and twin) and
`firmware/.../display/face_sdf.c` (robot). `make faces` writes the golden images the C port is tested against.
`face/expr.py` turns each reply into a face track. The base emotion comes from the LLM mood. A gesture expression is
held over each gesture (from the planner's placed gestures plus every script tag), and touching spans hand over
directly. The track plays on the motion clock (ZLink FACE_TRACK); the robot animates the talking mouth and blinks.

## Sim-to-real

`docs/SIM2REAL.md` audits every link from CAD to servo and gives the bring-up order. The tools are `make servo-id`
(fits latency, speed, t63, overshoot, deadband and error on one servo over ZLink) and `make sim2real` (plays the same
commands on the robot and in the twin, and reports per-joint real-vs-sim error). Both run against `make emulator`
today.

## Safety stage (`zang_ai/safety/stage.py`)

Every motion, whether it comes from the planner, the Pose sliders or a policy, passes through these steps in order:
1. **Envelope.** Anatomical co-speech range (`config/motion.json`), then the joint limits. Legs are held to ±4° and
   15°/s during co-speech.
2. **Collision.** Arms against the body on the CAD collision meshes, with 4 mm clearance. Offending poses are projected
   to the nearest clear pose (forward / outward / unfold elbow) with 3 mm extra headroom for servo lag. Corrections are
   held and smoothed so the arm doesn't twitch.
3. **Balance.** The static CoM must stay inside the support polygon of both soles by 8 mm; if not, the motion shrinks
   toward home.
4. **Rate.** Per-joint velocity and acceleration limits with look-ahead (start early, never arrive late).
5. **Verify.** Collision and balance are re-checked on the final trajectory, with up to 3 repair rounds. The report
   always shows what is left.

The physics twin then checks the dynamic result: tilt, ZMP margin, foot slip, servo load, tracking and self-contact. On
a FAIL the composer shrinks and re-plans (≤ 2×). The firmware repeats limits and rate as the last line of defence.

## Motion Studio (`zang_ai/studio/`)

A stdlib HTTP server plus a three.js page (vendored, works offline). The 3D view never does its own kinematics: it
shows MuJoCo body poses, so the picture is exactly the model that was checked.
- **Compose**: Ask Zang (LLM), reply script with tags, mood (valence/arousal/dominance), voice or text-only timing,
  physics on/off, HF gesture model on/off.
- **Gestures**: preview every variant in physics.
- **Pose**: 15 joint sliders with live safety projection; can stream to the robot.
- **Device**: ZLink connect, torque, home, hold, **E-STOP**, play the current motion on the robot, live servo readback.
- **Timeline**: words, gestures, beats, arm and gaze curves, audio-start marker. Click to seek; Space plays.
- **Inspector**: safety and physics reports. Export `.zact` + `.mp3` for the SD card.
- **Display modes**: **WHITE** and **DARK**. The choice is remembered per browser.
- **Policies tab** lists every RL run found in `rl/logs/` and `rl/runs/` (`rl/runs_index.py`, also usable from the
  terminal): trainer, task, IMU, iterations, learning curves, push-recovery results from `rl/logs/push_compare.txt`,
  export status, and a verdict against the hold-the-home-pose baseline. **Replay in 3D viewer** runs a rollout of the
  policy (push strength, length, physics, seed) and plays it on the twin, falls included, with push markers on the
  timeline. It exports the policy first if only a checkpoint exists. Unzip a Colab results zip into `rl/runs/<name>/`
  (it needs `policy.onnx` + `policy_meta.json`, or `policy_brax.npz`) to replay it here too.
- **Display tab** controls the chest TFT from the PC: 21 emotions (live thumbnails), intensity, blend time, gaze,
  talking/hold, a designer for all 23 face parameters (copy the result as a new `faces.json` emotion), text, image
  upload (converted to a 240×240 JPEG), brightness and the robot camera preview. Everything previews on the twin and is
  sent over ZLink v1.1 when a robot (or `make emulator`) is connected.
- **Viewport instruments**: a readout (time, frame, tilt, ZMP margin, servo load), camera presets (ISO, FRONT, SIDE,
  TOP), and a balance overlay showing the support polygon and the live CoP/ZMP from the physics twin.
- **Chest TFT**: the CAD chest plate has a 40 mm round window but no display part. The server finds the window by
  ray-casting the chest meshes and the twin mounts a virtual round 240×240 screen there (eyes blink, mouth follows
  speech).
- **Rendering**: CAD meshes get creased normals (smooth within 30°, hard edges kept) and double-sided materials.
- URL hooks for demos: `#theme=night&compose&seek=1.2&view=front`.
- `tools/screenshot.py` captures it headless, after the meshes and the compose are done.

## Numbers (this laptop, CPU)

An 11 s reply composes in ~2–4 s with physics (warm). Physics runs ~13× real time. The safety stage takes about 1 s per
10 s of motion. All sample replies pass safety and physics on the first attempt.

## Placeholders (replace with measurements)

| What | Where | How to measure |
|---|---|---|
| Joint limits ±90° | `config/robot.json` | mechanical stops per joint |
| Servo model (latency, speed, kp, force) | `config/motion.json`, MJCF | DS-R003G bench identification (see `rl/README.md`) |
| Masses (1.34 kg CAD) | `zang_description` | weigh the robot and parts |
| Servo IDs, signs, zero offsets | firmware NVS calibration | per-servo calibration (CALIB_SET over ZLink) |

## Layout

```
config/      robot.json (joint contract, canonical), motion.json, gestures.json, LLM prompts
zang_ai/     robot.py · pipeline.py · speech/ · motion/ · safety/ · sim/ · link/ · studio/
rl/          MuJoCo Playground balance/walk environments (future learned policies), see rl/README.md
tools/       gen_zlink_vectors.py (firmware golden vectors), screenshot.py
tests/       pytest
models/      symlinks to the speech models (not in git)
```
API keys: `config/secrets.env` (`GROQ_API_KEY=...`, not in git). You can copy yours from ZingAIStack's config.
