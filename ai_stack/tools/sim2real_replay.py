"""Sim-to-real replay: play the SAME commands on the robot and in the physics twin, then compare.

    .venv/bin/python tools/sim2real_replay.py --host zang.local --text "Hi! <wave> I am Zang."
    .venv/bin/python tools/sim2real_replay.py --host zang.local --zact out/HELLO.zact
    (works against `make emulator` too: --host 127.0.0.1)

Reports per joint: RMS of (real - sim) and of (real - command), the real latency (velocity cross-correlation) and
the peak error, plus the STATE rate and any stream drops. Writes out/sim2real_<name>.json (and .csv with the time
series) so runs can be compared before / after a calibration or servo-model change.
Interpretation: real ≈ sim but both ≠ command → the servo model is right (that's the servo's real lag); real ≠ sim →
fix the twin (servo model, masses, calibration) using tools/servo_id.py and the calibration table.
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


def load_motion(a, comp):
    if a.zact:
        from zang_ai.motion.export import read_zact
        z = read_zact(a.zact)
        return z["t"] - z["t"][0], z["q"], os.path.splitext(os.path.basename(a.zact))[0]
    res, _ = comp.compose_text(a.text, use_model=False, physics=False)
    return res["t"] - res["t"][0], res["q"], "text"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4210)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--text", default="Hi there! <wave> I am Zang. Let me think about it. <think>")
    g.add_argument("--zact")
    a = ap.parse_args()

    from zang_ai.pipeline import Composer
    from zang_ai.sim.twin import PhysicsTwin
    comp = Composer()
    r = comp.r
    t, Q, name = load_motion(a, comp)
    # 1. predicted: the twin plays the exact commands (50 Hz resampled)
    tt = np.arange(0, t[-1], 0.02)
    Qs = np.stack([np.interp(tt, t, Q[:, j]) for j in range(r.n)], 1)
    sim = PhysicsTwin(r).simulate(Qs, 50, settle=0.5)
    # 2. real: the same commands streamed to the robot, STATE recorded
    link = RobotLink(a.host, a.port)
    info = link.connect()
    link.torque_on()
    link.targets(Qs[0], 800)
    time.sleep(1.0)
    with StateRecorder(link) as rec:
        t0 = time.perf_counter()
        link.play(tt, Qs, block=True)
        time.sleep(0.8)
        tr, robot_ms, qm, qc = rec.arrays(t0)
    link.close()
    # align: the stream starts LOOKAHEAD after STREAM_START on the robot clock; use the robot's own clock where possible
    real = np.stack([np.interp(tt, tr, qm[:, j]) for j in range(r.n)], 1)
    rep = {"name": name, "robot": info, "duration_s": round(float(tt[-1]), 2), "state_samples": int(len(tr)),
           "state_rate_hz": round(len(tr) / max(tr[-1] - tr[0], 1e-6), 1) if len(tr) > 1 else 0, "joints": {}}
    err_rs, err_rc = [], []
    for j, n in enumerate(r.names):
        moving = np.ptp(Qs[:, j]) > np.radians(1.0)
        e_rs = np.degrees(real[:, j] - sim["q"][:, j])
        e_rc = np.degrees(real[:, j] - Qs[:, j])
        d = estimate_delay(tt, np.degrees(Qs[:, j]), np.degrees(real[:, j])) if moving else None
        rep["joints"][n] = {"moving": bool(moving), "rms_real_vs_sim_deg": round(float(np.sqrt(np.nanmean(e_rs ** 2))), 2),
                            "rms_real_vs_cmd_deg": round(float(np.sqrt(np.nanmean(e_rc ** 2))), 2),
                            "peak_real_vs_sim_deg": round(float(np.nanmax(np.abs(e_rs))), 2),
                            "latency_ms": None if d is None else round(d * 1000, 1)}
        if moving:
            err_rs.append(rep["joints"][n]["rms_real_vs_sim_deg"])
            err_rc.append(rep["joints"][n]["rms_real_vs_cmd_deg"])
    rep["summary"] = {"rms_real_vs_sim_deg": round(float(np.mean(err_rs)), 2) if err_rs else None,
                      "rms_real_vs_cmd_deg": round(float(np.mean(err_rc)), 2) if err_rc else None,
                      "worst_joint": max(rep["joints"], key=lambda k: rep["joints"][k]["rms_real_vs_sim_deg"])}
    os.makedirs(paths.OUT, exist_ok=True)
    out = os.path.join(paths.OUT, f"sim2real_{name}")
    with open(out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(out + ".csv", "w") as fh:
        fh.write("t," + ",".join(f"cmd_{n},sim_{n},real_{n}" for n in r.names) + "\n")
        for k in range(len(tt)):
            fh.write(f"{tt[k]:.3f}," + ",".join(f"{np.degrees(Qs[k, j]):.2f},{np.degrees(sim['q'][k, j]):.2f},"
                                               f"{np.degrees(real[k, j]):.2f}" for j in range(r.n)) + "\n")
    print(json.dumps(rep["summary"], indent=1), f"\nfull report: {out}.json / .csv")


if __name__ == "__main__":
    main()
