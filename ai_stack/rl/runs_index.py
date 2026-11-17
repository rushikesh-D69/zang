"""Index every RL run found on disk, for the studio's Policies tab (and for the command line).

    .venv/bin/python runs_index.py            # table of runs
    .venv/bin/python runs_index.py --json     # machine-readable (what the studio calls)

A run is any folder under logs/ (rsl_rl: logs/<Task>/<run>, SB3: logs/sb3/<Task>/<run>) or runs/ (e.g. an unzipped
Colab result) that holds a policy_meta.json or TensorBoard events. For each run: trainer, task, IMU, size of the
training, checkpoints, whether it was exported (ONNX / P4 header), learning curves downsampled from the TensorBoard
events, and the sim2sim / push-recovery results recorded in logs/push_compare.txt (baseline '.' = hold the home pose).
"""
import argparse
import glob
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH = [os.path.join(ROOT, "logs"), os.path.join(ROOT, "runs")]
# curve name -> TensorBoard tags to try, in order (rsl_rl, SB3)
CURVES = {
    "reward": ["Train/mean_reward", "rollout/ep_rew_mean"],
    "episode_s": ["Train/mean_episode_length", "rollout/ep_len_mean"],
    "fall_rate": ["/termination/fall_rate", "termination/fall_rate"],
    "upright": ["/episode_reward/upright"],
    "pose": ["/episode_reward/pose"],
    "std": ["Policy/mean_std"],
}
MAX_POINTS = 120


def _scalars(run):
    files = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
    if not files:
        return {}, 0
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out, last_step = {}, 0
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))
        for name, cands in CURVES.items():
            tag = next((t for t in cands if t in tags), None)
            if tag is None or name in out:
                continue
            ev = ea.Scalars(tag)
            if not ev:
                continue
            step = max(1, len(ev) // MAX_POINTS)
            pts = [[e.step, round(e.value, 4)] for e in ev[::step]]
            if pts[-1][0] != ev[-1].step:
                pts.append([ev[-1].step, round(ev[-1].value, 4)])
            out[name] = pts
            last_step = max(last_step, ev[-1].step)
    return out, last_step


def _push_results():
    """logs/push_compare.txt lines -> {run: {push: {mode: {...}}}} (run '.' = home-pose hold baseline)."""
    path = os.path.join(ROOT, "logs", "push_compare.txt")
    res = {}
    if not os.path.exists(path):
        return res
    pat = re.compile(r"push\s+([\d.]+)\s+(\S+)\s+(train|sim2sim)\s+fall\s+([\d.]+)\s+mean_fall_s\s+(\S+)\s+track_rms\s+([\d.]+)"
                     r"\s+tilt_mean\s+([\d.]+)\s+tilt_max\s+([\d.]+)")
    for line in open(path, encoding="utf-8"):
        m = pat.match(line.strip())
        if m:
            push, run, mode, fall, fs, rms, tm, tx = m.groups()
            res.setdefault(run, {}).setdefault(push, {})[mode] = {
                "fall": float(fall), "fall_s": None if fs == "nan" else float(fs), "track_rms_deg": float(rms),
                "tilt_mean_deg": float(tm), "tilt_max_deg": float(tx)}
    return res


def _find_runs():
    runs = []
    for base in SEARCH:
        for root, dirs, files in os.walk(base):
            if "policy_meta.json" in files or any(f.startswith("events.out.tfevents") for f in files):
                runs.append(root)
                dirs[:] = [d for d in dirs if d not in ("p4", "checkpoints")]
    return sorted(set(runs))


def _checkpoints(run):
    pts = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1]))
    its = [int(re.findall(r"\d+", os.path.basename(p))[-1]) for p in pts]
    n_sb3 = len(glob.glob(os.path.join(run, "checkpoints", "*")))
    return {"count": len(pts) + n_sb3, "latest_iter": its[-1] if its else None,
            "final": os.path.exists(os.path.join(run, "final_model.zip"))}


def index():
    push = _push_results()
    runs = []
    for r in _find_runs():
        rel = os.path.relpath(r, ROOT)
        name = os.path.basename(r)
        meta = {}
        mp = os.path.join(r, "policy_meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
        curves, last_step = _scalars(r)
        trainer = meta.get("trainer") or ("sb3" if "/sb3/" in r.replace("\\", "/") else "rsl_rl")
        task = meta.get("task") or os.path.basename(os.path.dirname(r))
        ck = _checkpoints(r)
        stamp = re.match(r"(\d{8})-(\d{6})", name)
        created = (time.strftime("%Y-%m-%d %H:%M", time.strptime(stamp.group(1) + stamp.group(2), "%Y%m%d%H%M%S"))
                   if stamp else time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(r))))
        env_cfg = {}
        if os.path.exists(os.path.join(r, "env_cfg.json")):
            env_cfg = json.load(open(os.path.join(r, "env_cfg.json"), encoding="utf-8"))
        runs.append({
            "id": rel, "name": name, "trainer": trainer, "task": task,
            "imu": bool(meta.get("imu", "IMU" in rel)), "obs_dim": meta.get("obs_dim"), "act_dim": meta.get("act_dim"),
            "created": created, "iterations": ck["latest_iter"] if ck["latest_iter"] is not None else last_step,
            "progress_unit": "steps" if trainer == "sb3" else "iters",
            "checkpoints": ck["count"], "final_model": ck["final"],
            "exported": {"meta": bool(meta), "onnx": os.path.exists(os.path.join(r, "policy.onnx")),
                         "p4_header": os.path.exists(os.path.join(r, "p4", "policy_weights.h"))},
            "mlp_macs": meta.get("mlp_macs"),
            "last": {k: v[-1][1] for k, v in curves.items()}, "curves": curves,
            "tests": push.get(name, {}), "env": {k: env_cfg[k] for k in list(env_cfg)[:0]},
        })
    runs.sort(key=lambda x: x["created"], reverse=True)
    return {"runs": runs, "baseline": push.get(".", {}), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "baseline = hold the home pose (zero action). Fall rates are over a few episodes: treat as a first signal."}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = index()
    if a.json:
        print(json.dumps(data))
        return
    b = data["baseline"].get("0.45", {}).get("sim2sim", {})
    print(f"{len(data['runs'])} runs; baseline (hold home pose) fall rate at 0.45 m/s push: {b.get('fall', 'n/a')}")
    print(f"{'created':17s} {'trainer':7s} {'task':8s} {'imu':3s} {'iters':>6s} {'reward':>8s} {'fall@.45':>9s}  run")
    for r in data["runs"]:
        f = r["tests"].get("0.45", {}).get("sim2sim", {}).get("fall")
        print(f"{r['created']:17s} {r['trainer']:7s} {r['task']:8s} {'yes' if r['imu'] else 'no':3s} {str(r['iterations']):>6s} "
              f"{r['last'].get('reward', float('nan')):8.2f} {('%.2f' % f) if f is not None else '-':>9s}  {r['id']}")


if __name__ == "__main__":
    main()
