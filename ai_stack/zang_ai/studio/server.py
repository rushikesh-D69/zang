"""Zang Motion Studio: HTTP server (stdlib only) for the digital-twin web UI.

    .venv/bin/python -m zang_ai.studio            -> http://localhost:8790
    .venv/bin/python -m zang_ai.studio --app      -> same, opened as a standalone desktop window (Chrome/Chromium app mode)

API (JSON unless noted)
  GET  /api/model                     joints, anatomical channels, bodies, visual geoms, home pose
  GET  /api/mesh/<id>                 binary mesh: u32 nv, u32 nf, f32 xyz[nv], u32 tri[nf]
  GET  /api/gestures                  gesture vocabulary
  GET  /api/status                    voices, LLM providers, robot link
  POST /api/fk          {q, safe}     body poses (+ the nearest safe pose and its report when safe=true)
  POST /api/compose     {text, vad, voice, physics, use_model, rate_wpm}  -> motion result (see _result_json)
  POST /api/gesture     {name, variant}                                  -> preview of one gesture
  POST /api/llm         {prompt, history}                                -> {reply, text, vad}
  POST /api/export      {id, name}    out/NAME.zact (+ NAME.mp3)
  GET  /api/file?path=out/...         exported / generated audio
  POST /api/robot/connect {host, port} | play {id} | hold | estop | torque | home | pose {q}
  GET  /api/robot/state            (+ v1.1 tail: buffered_ms, stream_pos_ms, display_mode, emotion)
  GET  /api/faces                  face library (config/faces.json)
  GET  /api/policies[?refresh]     every RL run on disk: settings, curves, tests (rl/runs_index.py)
  POST /api/policies/replay {run, push, seconds, seed, config, vx}   one rollout as viewer body poses (rl/replay_policy.py)
  POST /api/display/face|params|text|image|mode|brightness   chest TFT (preview always, sent when connected)
  POST /api/robot/camera {on, fps, size, quality | snapshot}   GET /api/robot/camera.jpg  latest frame
"""
import argparse
import io
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import traceback
import uuid
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np

from .. import paths
from ..robot import ANAT

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".svg": "image/svg+xml", ".png": "image/png"}


class Studio:
    """Holds the models (one composer, one lock: MuJoCo data is not thread-safe) and recent results."""

    def __init__(self):
        from ..pipeline import Composer
        self.comp = Composer()
        self.r = self.comp.r
        self.lock = threading.Lock()
        self.results = {}
        self.link = None
        self._mesh_cache = {}
        os.makedirs(paths.OUT, exist_ok=True)

    # ---- model ------------------------------------------------------------------------------------------
    def model_json(self):
        m, r = self.r.m, self.r
        geoms = []
        for g in range(m.ngeom):
            if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            geoms.append({"body": int(m.geom_bodyid[g]), "mesh": int(m.geom_dataid[g]),
                          "pos": m.geom_pos[g].round(6).tolist(), "quat": m.geom_quat[g].round(6).tolist(),
                          "rgba": np.asarray(rgba).round(3).tolist()})
        with self.lock:
            home = r.body_poses(r.home)[0]
            r.kinematics(r.home)
            support = r.support(r.d)
            window = display_window(r)
        return {
            "display": window, "support": np.round(support, 5).tolist(),
            "joints": [{"name": n, "lower": float(r.lower[i]), "upper": float(r.upper[i]), "home": float(r.home[i]),
                        "group": r.group[i]} for i, n in enumerate(r.names)],
            "anat": {"names": ANAT, "lo": r.anat_lo.tolist(), "hi": r.anat_hi.tolist()},
            "bodies": [m.body(b).name for b in range(m.nbody)],
            "geoms": geoms, "home_bodies": home.round(6).tolist(), "fps": 50,
        }

    def mesh_bin(self, mid):
        if mid not in self._mesh_cache:
            m = self.r.m
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            v = m.mesh_vert[va:va + vn].astype("<f4")
            f = m.mesh_face[fa:fa + fn].astype("<u4")
            self._mesh_cache[mid] = struct.pack("<II", vn, fn) + v.tobytes() + f.tobytes()
        return self._mesh_cache[mid]

    # ---- motion results ----------------------------------------------------------------------------------
    def _store(self, res, speech, label):
        rid = uuid.uuid4().hex[:8]
        audio_url = None
        if speech is not None:
            pre = np.zeros(int(round(res["audio_offset"] * speech["sr"])), np.float32)
            audio = np.concatenate([pre, speech["audio"]])
            res["audio"], res["sr"] = audio, speech["sr"]
            path = os.path.join(paths.OUT, f"_{rid}.wav")
            _write_wav(path, audio, speech["sr"])
            audio_url = f"/api/file?path=out/_{rid}.wav"
        res["label"] = label
        self.results[rid] = res
        for old in list(self.results)[:-12]:  # keep the last 12
            self.results.pop(old, None)
        return self._result_json(rid, res, audio_url)

    def _result_json(self, rid, res, audio_url):
        r = self.r
        sim = res.get("sim")
        shown_q = sim["q"] if sim is not None else res["q"]
        base = sim["base"] if sim is not None else None
        with self.lock:
            bodies = r.body_poses(shown_q, base)
        r4 = lambda a, n=4: np.round(np.nan_to_num(np.asarray(a, float)), n).tolist()  # noqa: E731
        out = {"id": rid, "label": res.get("label"), "fps": 50, "t": r4(res["t"], 3),
               "duration": float(res["t"][-1]), "audio": audio_url, "audio_offset": res.get("audio_offset", 0.0),
               "q_cmd": r4(np.degrees(res["q"]), 2), "q_sim": r4(np.degrees(shown_q), 2) if sim is not None else None,
               "anat": r4(res["anat"], 2), "bodies": r4(bodies, 5), "events": _jsonable(res["events"]),
               "words": _jsonable(res.get("words", [])), "safety": _jsonable(res["safety"]),
               "physics": _jsonable(res["physics"]), "params": _jsonable(res.get("params")),
               "face": _jsonable(res.get("face", []))}
        if sim is not None:
            out["sim"] = {"tilt": r4(sim["tilt"], 2), "margin": r4(sim["margin"], 1),
                          "load_pct": r4(sim["load"].max(1) * 100, 1), "cop": r4(sim["cop"] * 1000, 1)}
        return out

    def compose(self, b):
        text = (b.get("text") or "").strip()
        if not text:
            raise ValueError("empty text")
        vad = tuple(b.get("vad") or (0.6, 0.5, 0.5))
        voice = b.get("voice") or None
        with self.lock:
            res, speech = self.comp.compose_text(text, vad, voice, physics=bool(b.get("physics", True)),
                                                 use_model=bool(b.get("use_model", True)),
                                                 rate_wpm=float(b.get("rate_wpm", 150)))
        return self._store(res, speech, text[:60])

    def gesture(self, b):
        name, variant = b["name"], b.get("variant")
        tag = f"<{name}:{variant}>" if variant else f"<{name}>"
        text = f"{tag} This is {name}."
        with self.lock:
            ctx = self.comp.context_text(text, use_model=False)
            res = self.comp.compose(ctx, physics=bool(b.get("physics", True)), log=None)
        res["words"] = res.get("words", ctx["words"])
        return self._store(res, None, f"gesture {name}{':' + variant if variant else ''}")

    def fk(self, b):
        q = np.asarray(b["q"], float)
        rep = None
        with self.lock:
            if b.get("safe"):
                q, rep = self.comp.safety.safe_pose(q)
            bodies = self.r.body_poses(q)[0]
        return {"q": q.tolist(), "bodies": np.round(bodies, 5).tolist(), "report": rep}

    def export(self, b):
        from ..motion.export import write_zact
        from ..speech.tts import to_mp3
        res = self.results[b["id"]]
        name = "".join(c for c in (b.get("name") or "ACTION").upper() if c.isalnum() or c in "_-")[:24] or "ACTION"
        audio_name = None
        if res.get("audio") is not None:
            audio_name = name + ".mp3"
            with open(os.path.join(paths.OUT, audio_name), "wb") as fh:
                fh.write(to_mp3(res["audio"], res["sr"]))
        path = write_zact(os.path.join(paths.OUT, name + ".zact"), res["t"], res["q"], audio_name)
        return {"zact": os.path.relpath(path, paths.ROOT), "audio": audio_name and f"out/{audio_name}",
                "frames": len(res["t"])}

    def llm(self, b):
        from ..speech import llm as L
        from ..speech import text as T
        sys_prompt = open(paths.config("llm_conversation_prompt.md"), encoding="utf-8").read() + "\n\n" + T.llm_tag_help()
        msgs = [{"role": "system", "content": sys_prompt}] + list(b.get("history") or [])[-8:] + \
               [{"role": "user", "content": b["prompt"]}]
        reply = "".join(x for kind, x in L.stream_chat(msgs) if kind == "text")
        reply, vad = _split_mood(reply)
        return {"reply": reply, "vad": vad}

    # ---- robot -------------------------------------------------------------------------------------------
    def robot(self, action, b):
        from ..link.client import RobotLink
        if action == "connect":
            if self.link:
                self.link.close()
            self.link = RobotLink(b.get("host", "127.0.0.1"), int(b.get("port", 4210)))
            info = self.link.connect()
            return {"info": info, "rtt_ms": self.link.rtt_ms}
        if self.link is None:
            raise RuntimeError("not connected")
        if action == "play":
            res = self.results[b["id"]]
            self.link.play(np.asarray(res["t"]) - res["t"][0], res["q"], face=res.get("face"))
        elif action == "hold":
            self.link.hold()
        elif action == "estop":
            self.link.estop()
        elif action == "torque":
            self.link.torque_on()
        elif action == "home":
            self.link.go_home(1500)
        elif action == "pose":
            self.link.targets(np.asarray(b["q"], float), int(b.get("move_ms", 300)))
        return {"ok": True}

    # ---- chest display (works without a robot: the studio previews; with one, it is sent too) ----------------------
    def display(self, action, b):
        from ..face.render import EMOTIONS, PARAMS
        from ..link import zlink as Z
        sent = self.link is not None
        if action == "face":
            emo = b.get("emotion", "neutral")
            eid = EMOTIONS.index(emo) if isinstance(emo, str) else int(emo)
            if sent:
                self.link.face(eid, float(b.get("intensity", 1.0)), int(b.get("blend_ms", 250)),
                               tuple(b.get("gaze", (0, 0))), bool(b.get("talking")), bool(b.get("hold")))
        elif action == "params":
            by_id = {PARAMS.index(k): float(v) for k, v in (b.get("params") or {}).items() if k in PARAMS}
            for i in range(0, len(by_id), 40):  # stay well under one frame
                if sent:
                    self.link.face_params(dict(list(by_id.items())[i:i + 40]), int(b.get("blend_ms", 120)))
        elif action == "text":
            fg = Z.rgb565(*bytes.fromhex(b.get("fg", "#ffffff").lstrip("#")))
            bgc = Z.rgb565(*bytes.fromhex(b.get("bg", "#000000").lstrip("#")))
            if sent:
                self.link.display_text(b.get("text", ""), int(b.get("size", 2)), fg, bgc, int(b.get("duration_ms", 0)))
        elif action == "image":
            jpeg = panel_jpeg(b["data_url"], b.get("fit", "cover"))
            image_id = int(b.get("image_id", 1))
            if sent:
                self.link.display_image(image_id, jpeg, show=True, store=bool(b.get("store")))
            import base64
            return {"sent": sent, "bytes": len(jpeg), "preview": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}
        elif action == "mode":
            if sent:
                self.link.display_mode(int(b.get("mode", 0)), int(b.get("image_id", 0)))
        elif action == "brightness":
            if sent:
                self.link.set_brightness(int(b.get("percent", 80)))
        else:
            raise ValueError(f"unknown display action {action}")
        return {"sent": sent}

    def camera(self, b):
        if self.link is None:
            raise RuntimeError("not connected")
        if b.get("snapshot"):
            self.link.snapshot()
        else:
            self.link.camera_stream(bool(b.get("on", True)), int(b.get("size", 0)), int(b.get("fps", 10)),
                                    int(b.get("quality", 60)))
        return {"ok": True}

    # ---- RL policies: index built by rl/runs_index.py (it needs the RL venv: tensorboard) ---------------------------
    def policies(self, refresh=False):
        import subprocess
        import sys
        cache = getattr(self, "_pol", None)
        if cache and not refresh and time.time() - cache[0] < 30:
            return cache[1]
        rl = os.path.join(paths.ROOT, "rl")
        py = next((p for p in (os.path.join(rl, ".venv", "bin", "python"), sys.executable) if os.path.exists(p)), sys.executable)
        r = subprocess.run([py, os.path.join(rl, "runs_index.py"), "--json"], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError("runs_index.py failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
        data = json.loads(r.stdout)
        self._pol = (time.time(), data)
        return data

    def policy_replay(self, b):
        """Run one rollout of an RL policy (rl/replay_policy.py, in the RL venv: CPU MuJoCo, same loop as sim2sim) and
        return it in the viewer's format: MuJoCo body poses for every control step, so the twin plays what the policy did."""
        import subprocess
        import sys
        rl = os.path.join(paths.ROOT, "rl")
        run = str(b.get("run", "home"))
        if run != "home":
            full = os.path.realpath(os.path.join(rl, run))
            if not full.startswith(os.path.realpath(rl) + os.sep) or not os.path.isdir(full):
                raise ValueError("unknown run: " + run)
        py = next((p for p in (os.path.join(rl, ".venv", "bin", "python"), sys.executable) if os.path.exists(p)), sys.executable)
        cmd = [py, os.path.join(rl, "replay_policy.py"), "--run", run, "--config", str(b.get("config", "train")),
               "--push", str(float(b.get("push", 0.45))), "--seconds", str(min(30.0, max(2.0, float(b.get("seconds", 10))))),
               "--seed", str(int(b.get("seed", 0))), "--command", str(float(b.get("vx", 0.1))), "0", "0"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeError("replay failed: " + (proc.stderr.strip().splitlines() or ["no output"])[-1])
        if out.get("error"):
            raise ValueError(out["error"])
        Q, base = np.asarray(out["q"], float), np.asarray(out["base"], float)
        fps = out["fps"]
        with self.lock:
            bodies = self.r.body_poses(Q, base)
            anat = self.r.q_to_anat(Q)
        rid = uuid.uuid4().hex[:8]
        fell = out["fell_at"]
        label = ("hold home pose" if run == "home" else os.path.basename(run.rstrip("/"))) + f" · push {out['push']} m/s"
        res = {"id": rid, "kind": "policy", "label": label, "fps": fps, "t": (np.arange(len(Q)) / fps).round(3).tolist(),
               "duration": (len(Q) - 1) / fps, "audio": None, "audio_offset": 0.0, "anat": np.round(anat, 2).tolist(),
               "bodies": np.round(bodies, 5).tolist(), "words": [], "face": [], "safety": None, "physics": None,
               "events": [{"type": "push", "t": t} for t in out["pushes"]],
               "sim": {"tilt": out["tilt"], "margin": [], "load_pct": [], "cop": []},
               "policy": {"run": run, "task": out["task"], "imu": out["imu"], "config": out["config"], "push": out["push"],
                          "fell_at": fell, "note": out.get("note"), "metrics": out["metrics"]}}
        self.results[rid] = {"t": np.arange(len(Q)) / fps, "q": Q, "anat": anat, "events": [], "safety": {}, "physics": None,
                             "label": label}
        return res

    def robot_state(self):
        if not self.link or not self.link.state:
            return {"connected": bool(self.link), "state": None}
        s = self.link.state
        return {"connected": True, "rtt_ms": self.link.rtt_ms, "info": self.link.info,
                "state": {"t_ms": s.t_robot_ms, "flags": {f: s.flag(f) for f in s.FLAGS},
                          "q_meas_deg": np.round(np.degrees(np.nan_to_num(s.q_meas)), 2).tolist(),
                          "q_cmd_deg": np.round(np.degrees(s.q_cmd), 2).tolist(), "vbat_mv": s.vbat_mv,
                          "temp_max_c": s.temp_max_c, "buffered_ms": s.stream_buffered_ms,
                          "stream_pos_ms": s.stream_pos_ms, "display_mode": s.display_mode, "emotion": s.emotion},
                "camera_frame": self.link.camera["frame_id"] if self.link.camera else None}

    def status(self):
        from ..speech import llm as L
        from ..speech import tts
        return {"voices": tts.voices(), "llm": L.status(), "robot": bool(self.link)}


def panel_jpeg(data_url, fit="cover", size=240, quality=85):
    """Any uploaded image (data URL) -> 240x240 JPEG for the round chest TFT (ZLink DISPLAY_IMAGE).
    cover = crop to fill the circle, contain = letterbox on black."""
    import base64
    import io

    from PIL import Image, ImageOps
    raw = base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if fit == "contain":
        im.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        im = canvas
    else:
        im = ImageOps.fit(im, (size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def display_window(r, step=0.0015):
    """Find the TFT window in the chest plate (the CAD has no display part) by ray-casting the chest's visual
    meshes front to back. -> {body, center [x y z], size [w h]} in the chest frame, or None.
    The studio puts a virtual screen there, slightly larger than the hole and just behind the plate."""
    m, d = r.m, r.d
    c = m.body("chest").id
    r.kinematics(r.home)
    R, p = d.xmat[c].reshape(3, 3), d.xpos[c]
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == c and m.geom_group[g] == 2:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            pts.append((d.geom_xpos[g] + m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T - p) @ R)
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    ys, zs = np.arange(lo[1], hi[1], step), np.arange(lo[2], hi[2], step)
    group = np.array([0, 0, 1, 0, 0, 0], np.uint8)
    gid = np.zeros(1, np.int32)
    x0 = hi[0] + 0.05
    depth = np.full((len(ys), len(zs)), np.nan)
    for i, y in enumerate(ys):
        for j, z in enumerate(zs):
            dist = mujoco.mj_ray(m, d, p + R @ np.array([x0, y, z]), R @ np.array([-1.0, 0, 0]), group, 1, -1, gid)
            if dist >= 0 and m.geom_bodyid[gid[0]] == c:
                depth[i, j] = x0 - dist
    front = np.nanmax(depth)
    deep = np.isnan(depth) | (depth < front - 0.008)
    # window = deep cells NOT connected to the border (flood fill from the outside)
    outside = np.zeros_like(deep)
    stack = [(i, j) for i in range(len(ys)) for j in (0, len(zs) - 1)] + [(i, j) for j in range(len(zs)) for i in (0, len(ys) - 1)]
    while stack:
        i, j = stack.pop()
        if 0 <= i < len(ys) and 0 <= j < len(zs) and deep[i, j] and not outside[i, j]:
            outside[i, j] = True
            stack += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    win = deep & ~outside
    # the largest enclosed opening is the window (small enclosed ones are screw holes)
    label = np.zeros(win.shape, int)
    best, best_n = 0, 0
    for i0, j0 in zip(*np.nonzero(win)):
        if label[i0, j0]:
            continue
        lab = label.max() + 1
        stack, n = [(i0, j0)], 0
        while stack:
            i, j = stack.pop()
            if 0 <= i < len(ys) and 0 <= j < len(zs) and win[i, j] and not label[i, j]:
                label[i, j] = lab
                n += 1
                stack += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        if n > best_n:
            best, best_n = lab, n
    if best_n < 20:
        return None
    ii, jj = np.nonzero(label == best)
    y0, y1, z0, z1 = ys[ii.min()], ys[ii.max()], zs[jj.min()], zs[jj.max()]
    w, h = y1 - y0 + step, z1 - z0 + step
    area = best_n * step * step
    round_ = abs(w - h) / max(w, h) < 0.2  # the CAD plate has a circular window; rays near its rim can leak
    plate_cells = ~deep[max(ii.min() - 4, 0):ii.max() + 5, max(jj.min() - 4, 0):jj.max() + 5]
    plate = np.nanmedian(depth[max(ii.min() - 4, 0):ii.max() + 5, max(jj.min() - 4, 0):jj.max() + 5][plate_cells])
    if round_:  # centroid + equal-area diameter: robust to a few leaked rays at the rim
        cy, cz = float(ys[ii].mean()), float(zs[jj].mean())
        dia = 2 * np.sqrt(area / np.pi)
        size = [dia + 0.004, dia + 0.004]
        opening = [round(dia * 1000, 1)] * 2
    else:
        cy, cz = float((y0 + y1) / 2), float((z0 + z1) / 2)
        size = [w + 0.004, h + 0.004]
        opening = [round(w * 1000, 1), round(h * 1000, 1)]
    return {"body": int(c), "shape": "round" if round_ else "rect",
            "center": [round(float(plate) - 0.004, 5), round(cy, 5), round(cz, 5)],
            "size": [round(float(x), 5) for x in size], "opening_mm": [float(x) for x in opening]}


def _split_mood(reply):
    """Mood JSON as the first line (conversation prompt) or the last line (Zing's older prompt)."""
    from ..speech import text as T
    lines = reply.strip().splitlines()
    vad = (0.6, 0.5, 0.5)
    if lines and lines[0].strip().startswith("{") and "valence" in lines[0]:
        try:
            e = json.loads(lines[0])
            vad = tuple(float(min(max(e.get(k, d), 0), 1)) for k, d in zip(("valence", "arousal", "dominance"), vad))
            lines = lines[1:]
        except ValueError:
            pass
    _, tagged, vad2 = T.split_llm_reply("\n".join(lines), vad)
    return tagged.strip(), list(vad2)


def _write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def make_handler(studio):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if "/api/robot/state" not in (args[0] if args else ""):
                print(time.strftime("%H:%M:%S"), fmt % args)

        def _send(self, code, body, ctype="application/json", cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            url = urlparse(self.path)
            p = url.path
            try:
                if p in ("/", "/index.html"):
                    return self._static("index.html")
                if p.startswith("/web/"):
                    return self._static(p[5:])
                if p == "/api/model":
                    return self._json(studio.model_json())
                if p.startswith("/api/mesh/"):
                    return self._send(200, studio.mesh_bin(int(p.rsplit("/", 1)[1])), "application/octet-stream", True)
                if p == "/api/gestures":
                    from ..speech.text import VOCAB
                    return self._json({g: {"means": v.get("means", ""), "head_only": v.get("head_only", False),
                                           "variants": [x.get("id") for x in v.get("variants", [])]}
                                       for g, v in VOCAB.items()})
                if p == "/api/status":
                    return self._json(studio.status())
                if p == "/api/policies":
                    return self._json(studio.policies("refresh" in parse_qs(url.query)))
                if p == "/api/faces":
                    return self._send(200, open(paths.config("faces.json"), "rb").read(), "application/json")
                if p == "/api/robot/camera.jpg":
                    f = studio.link.camera if studio.link else None
                    if not f:
                        return self._json({"error": "no camera frame"}, 404)
                    return self._send(200, f["data"], "image/jpeg")
                if p == "/api/robot/state":
                    return self._json(studio.robot_state())
                if p == "/api/file":
                    rel = parse_qs(url.query).get("path", [""])[0]
                    full = os.path.realpath(os.path.join(paths.ROOT, rel))
                    if not full.startswith(os.path.realpath(paths.OUT) + os.sep) or not os.path.isfile(full):
                        return self._json({"error": "not found"}, 404)
                    return self._send(200, open(full, "rb").read(), MIME.get(os.path.splitext(full)[1], "application/octet-stream"))
                return self._json({"error": "unknown route"}, 404)
            except Exception as e:
                traceback.print_exc()
                return self._json({"error": str(e)}, 500)

        def do_POST(self):
            p = urlparse(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            try:
                b = json.loads(self.rfile.read(n) or b"{}")
                if p == "/api/clientlog":  # browser errors -> this terminal (the user may not have devtools open)
                    print(f"CLIENT ERROR [{b.get('where')}] {b.get('message')}\n  {b.get('stack', '')[:600]}\n  {b.get('agent', '')}")
                    return self._json({"ok": True})
                routes = {"/api/fk": studio.fk, "/api/compose": studio.compose, "/api/gesture": studio.gesture,
                          "/api/export": studio.export, "/api/llm": studio.llm}
                if p in routes:
                    return self._json(routes[p](b))
                if p == "/api/policies/replay":
                    return self._json(studio.policy_replay(b))
                if p.startswith("/api/display/"):
                    return self._json(studio.display(p.rsplit("/", 1)[1], b))
                if p == "/api/robot/camera":
                    return self._json(studio.camera(b))
                if p.startswith("/api/robot/"):
                    return self._json(studio.robot(p.rsplit("/", 1)[1], b))
                return self._json({"error": "unknown route"}, 404)
            except Exception as e:
                traceback.print_exc()
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _static(self, rel):
            full = os.path.realpath(os.path.join(WEB, rel))
            if not full.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            self._send(200, open(full, "rb").read(), MIME.get(os.path.splitext(full)[1], "application/octet-stream"),
                       cache="/vendor/" in full)

    return Handler


def open_app_window(url):
    """Standalone desktop window: a Chromium-family browser in --app mode (no tabs / address bar)."""
    for exe in ("google-chrome", "chromium", "chromium-browser", "brave-browser", "brave", "microsoft-edge"):
        path = shutil.which(exe)
        if path:
            # use the real GPU even if the driver is on Chrome's blocklist; no silent software fallback (the status bar
            # shows the renderer and warns if it is software)
            subprocess.Popen([path, f"--app={url}", "--window-size=1500,920", "--ignore-gpu-blocklist",
                              "--use-gl=angle", "--use-angle=gl", "--enable-gpu-rasterization"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return exe
    webbrowser.open(url)
    return "default browser"


def main():
    ap = argparse.ArgumentParser(description="Zang Motion Studio")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--app", action="store_true", help="open as a standalone desktop window")
    a = ap.parse_args()
    t0 = time.time()
    studio = Studio()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(studio))
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"Zang Motion Studio ready in {time.time() - t0:.1f}s -> {url}")
    if a.app:
        print("window:", open_app_window(url))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
