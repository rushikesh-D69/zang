"""Replay one policy rollout and print it as JSON (base pose + 15 servo angles per control step), for the studio's 3D viewer.

    .venv/bin/python replay_policy.py --run logs/ZangStand/20261117-152022 --push 0.45 --seconds 10
    .venv/bin/python replay_policy.py --run home                # baseline: hold the home pose
    .venv/bin/python replay_policy.py --run runs/colab_stand    # an unzipped Colab result (needs policy.onnx or policy_brax.npz)

--run is a run folder (relative to rl/) or 'home'. The policy is policy.onnx in that folder; if only a checkpoint exists
(rsl_rl model_*.pt, SB3 final_model.zip, brax policy_brax.npz) it is exported first with export_policy.py. The rollout
is one episode of sim2sim.py's loop with the same options (--config train|sim2sim|euler, --push, --seconds, --seed,
--command vx vy wz for walking policies); after a fall it keeps 0.6 s so the fall can be watched.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def resolve_policy(run):
    if run == "home":
        return "home", None
    d = os.path.join(ROOT, run) if not os.path.isabs(run) else run
    onnx = os.path.join(d, "policy.onnx")
    if os.path.exists(onnx):
        return onnx, None
    cands = sorted(glob.glob(os.path.join(d, "model_*.pt")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1]))
    ck = (cands[-1] if cands else next(iter(glob.glob(os.path.join(d, "final_model.zip")) + glob.glob(os.path.join(d, "policy_brax.npz"))), None))
    if ck is None:
        raise SystemExit(f"{run}: no policy.onnx and no checkpoint (model_*.pt, final_model.zip, policy_brax.npz) to export")
    import export_policy
    export_policy.export(ck, d)
    if not os.path.exists(onnx):
        raise SystemExit(f"export of {ck} did not produce {onnx}")
    return onnx, f"exported {os.path.basename(ck)} to policy.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="train", choices=["train", "sim2sim", "euler"])
    ap.add_argument("--push", type=float, default=0.45)
    ap.add_argument("--push-interval", type=float, default=2.5)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--no-noise", dest="noise", action="store_false")
    a0 = ap.parse_args()
    policy_path, note = resolve_policy(a0.run)

    import sim2sim as S
    from zang_rl.config import NoiseCfg, ServoCfg
    argv = ["--policy", policy_path, "--episodes", "1", "--seconds", str(a0.seconds), "--push", str(a0.push),
            "--push-interval", str(a0.push_interval), "--seed", str(a0.seed), "--command", *map(str, a0.command)]
    if not a0.noise:
        argv.append("--no-noise")
    a = S.parse(argv)
    if policy_path != "home":  # a policy exported before an observation-layout change cannot run: say so, don't crash
        import onnxruntime as ort
        n_in = ort.InferenceSession(policy_path).get_inputs()[0].shape[-1]
        z3 = np.zeros(3)
        n_exp = len(S.O.build_actor_obs(np.zeros(15), np.zeros(15), np.zeros(15), z3, np.zeros(2),
                                        *((z3, np.array([0.0, 0.0, -1.0])) if a.imu else (None, None))))
        if n_in != n_exp:
            print(json.dumps({"error": f"this policy expects {n_in} observation values but the current layout has {n_exp} "
                                       f"({'with' if a.imu else 'no'} IMU). It was exported before the observation layout "
                                       "changed: re-train it or re-export from its checkpoint."}))
            sys.exit(2)
    a.record = {"base": [], "q": [], "tilt": [], "pushes": [], "fell_at": None}
    row = S.run_config(a0.config, S.Policy(policy_path), a, ServoCfg.load(), NoiseCfg())
    rec = a.record
    meta_path = os.path.join(os.path.dirname(os.path.abspath(policy_path)), "policy_meta.json") if policy_path != "home" else None
    meta = json.load(open(meta_path)) if meta_path and os.path.exists(meta_path) else {}
    out = {
        "fps": round(1.0 / a.ctrl_dt), "run": a0.run, "note": note, "config": a0.config,
        "task": meta.get("task", "stand"), "imu": bool(meta.get("imu", False)),
        "q": np.round(np.array(rec["q"]), 5).tolist(), "base": np.round(np.array(rec["base"]), 5).tolist(),
        "tilt": np.round(rec["tilt"], 2).tolist(), "pushes": rec["pushes"], "fell_at": rec["fell_at"],
        "push": a0.push, "metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()},
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
