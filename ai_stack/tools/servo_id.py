"""Servo identification over ZLink: replaces the placeholder servo model with measured numbers.

    .venv/bin/python tools/servo_id.py --host zang.local --joint left_elbow [--amp 30] [--write]
    (works against `make emulator` too: --host 127.0.0.1)

Bench setup: the joint moves freely or carries a known load (e.g. a pendulum arm of known mass and length); the rest
of the robot is held (or on a stand). The tool sends step and ramp targets on ONE joint with JOINT_TARGETS, records
STATE (q_meas) and fits:
  latency_ms   effective command -> motion lag (velocity cross-correlation; includes the servo's own lag)
  vmax_dps     the servo's own speed limit (steepest measured slope on large steps)
  t63_large_step_s  time to 63 % of a large step (includes the speed-limited phase)
  overshoot    of large steps, deg
  deadband     smallest step that produces motion, deg
  error_deg    steady-state error after each step
--write merges latency_ms and vmax_dps into config/motion.json servo_model and saves the full report to
config/servo_identified.json (which the RL environment can read via rl/assets).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zang_ai import paths  # noqa: E402
from zang_ai.link.client import RobotLink  # noqa: E402
from zang_ai.link.recorder import StateRecorder, estimate_delay  # noqa: E402
from zang_ai.robot import RobotModel  # noqa: E402


def run(host, port, joint, amp_deg, settle=1.2):
    r = RobotModel()
    j = r.idx[joint]
    link = RobotLink(host, port)
    link.connect()
    link.torque_on()
    q0 = r.home.copy()
    link.targets(q0, 800)
    time.sleep(1.2)
    steps = [amp_deg, -amp_deg, amp_deg / 3, -amp_deg / 3, 2, -2, 1, -1, 0.5, -0.5, 0]
    marks = []
    with StateRecorder(link) as rec:
        t_start = time.perf_counter()
        for s in steps:  # move_ms = 0: the servo's own speed limit governs the motion
            q = q0.copy()
            q[j] += np.radians(s)
            marks.append((time.perf_counter() - t_start, s))
            link.targets(q, 0)
            time.sleep(settle)
        t, _, qm, qc = rec.arrays(t_start)
    link.go_home(800)
    link.close()
    meas = np.degrees(qm[:, j] - q0[j])
    cmd = np.zeros_like(t)
    for (tm, s) in marks:
        cmd[t >= tm] = s
    rep = analyse(t, cmd, meas, marks, settle)
    rep.update(joint=joint, host=host, amp_deg=amp_deg, samples=int(len(t)),
               state_rate_hz=round(float(len(t) / max(t[-1] - t[0], 1e-6)), 1))
    return rep


def analyse(t, cmd, meas, marks, settle):
    ok = ~np.isnan(meas)
    t, cmd, meas = t[ok], cmd[ok], meas[ok]
    delay = estimate_delay(t, cmd, meas)
    vel = np.gradient(meas, t)
    vmax = float(np.percentile(np.abs(vel), 99.5))
    per_step, prev = [], 0.0
    for (tm, s) in marks:
        sel = (t >= tm) & (t < tm + settle)
        if sel.sum() < 5:
            prev = s
            continue
        y, tt = meas[sel], t[sel] - tm
        final = float(np.median(y[-max(3, len(y) // 5):]))
        moved = abs(final - prev) > 0.25 * max(abs(s - prev), 0.2)
        span = final - prev
        over = float(np.max((y - final) * np.sign(span))) if abs(span) > 1 else 0.0
        k63 = np.argmax(np.abs(y - prev) >= 0.63 * abs(span)) if abs(span) > 0.2 else 0
        per_step.append({"target_deg": s, "final_deg": round(final, 3), "error_deg": round(final - s, 3),
                         "moved": bool(moved), "t63_s": round(float(tt[k63]), 3), "overshoot_deg": round(over, 3)})
        prev = final
    small = [p for p in per_step if abs(p["target_deg"]) <= 2]
    deadband = min((abs(p["target_deg"]) for p in small if p["moved"]), default=None)
    big = [p for p in per_step if abs(p["target_deg"]) >= 10]
    return {"latency_ms": round(delay * 1000, 1), "vmax_dps": round(vmax, 1),
            "t63_large_step_s": round(float(np.median([p["t63_s"] for p in big])), 3) if big else None,
            "overshoot_deg": round(float(np.max([p["overshoot_deg"] for p in big])), 2) if big else None,
            "deadband_deg": deadband,
            "steady_error_deg_rms": round(float(np.sqrt(np.mean([p["error_deg"] ** 2 for p in per_step]))), 3),
            "steps": per_step}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4210)
    ap.add_argument("--joint", default="left_elbow")
    ap.add_argument("--amp", type=float, default=30.0)
    ap.add_argument("--write", action="store_true", help="update config/motion.json servo_model + save the report")
    a = ap.parse_args()
    rep = run(a.host, a.port, a.joint, a.amp)
    print(json.dumps({k: v for k, v in rep.items() if k != "steps"}, indent=1))
    if a.write:
        with open(paths.config("servo_identified.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
        mo_path = paths.config("motion.json")
        mo = json.load(open(mo_path))
        mo["servo_model"].update(latency_ms=rep["latency_ms"], vmax_dps=rep["vmax_dps"],
                                 _doc=f"MEASURED by tools/servo_id.py on {rep['joint']} ({time.strftime('%Y-%m-%d')}).")
        with open(mo_path, "w") as fh:
            json.dump(mo, fh, indent=2)
        print("wrote config/servo_identified.json and updated config/motion.json servo_model")


if __name__ == "__main__":
    main()
