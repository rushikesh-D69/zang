"""Export a trained policy for deployment: rsl_rl checkpoint .pt, SB3 model .zip + vecnormalize.pkl, or the brax
(GPU / Colab notebook) actor policy_brax.npz written by zang_rl/brax_train.py.

  .venv/bin/python export_policy.py logs/ZangStand/<run>/model_3000.pt
  .venv/bin/python export_policy.py logs/sb3/<run>/final_model.zip        # vecnormalize.pkl next to it
  python export_policy.py <run>/policy_brax.npz                           # env_cfg.json next to it

Writes into DIR (default: the checkpoint's folder):
  policy.onnx            actor: observation normalisation folded in, output = action `a` in the shared convention
                         (joint target = home + action_scale * a, wire order)
  policy_meta.json       observation layout, action scale, home pose, joint order, control period
  p4/policy_weights.h    float32 weights + normaliser for export/p4/zang_policy.c (the ESP32-P4 C MLP)
and checks that the C MLP (compiled with gcc on this PC) matches onnxruntime on random observations.
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P4_SRC = os.path.join(HERE, "export", "p4")


def _env_cfg_from_run(run_dir):
    from zang_rl.config import EnvCfg
    cfg = EnvCfg(task="stand", nworld=1, nthread=1)
    p = os.path.join(run_dir, "env_cfg.json")
    if os.path.exists(p):
        raw = json.load(open(p))
        for k in ("task", "action_scale", "ctrl_dt", "imu_body", "imu", "gym_action_gain"):
            if k in raw:
                setattr(cfg, k, raw[k])
    return cfg


# ================================================================================================ actor modules
class _Actor(torch.nn.Module):
    """normalise -> Linear/ELU stack -> optional clip -> gain. Mirrors export/p4/zang_policy.c exactly."""

    def __init__(self, mean, inv_std, linears, clip=0.0, gain=1.0):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("inv_std", torch.as_tensor(np.asarray(inv_std), dtype=torch.float32))
        self.linears = torch.nn.ModuleList(linears)
        self.clip, self.gain = float(clip), float(gain)

    def forward(self, obs):
        x = (obs - self.mean) * self.inv_std
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = torch.nn.functional.elu(x)
        if self.clip > 0:
            x = torch.clamp(x, -self.clip, self.clip)
        return x * self.gain


def _linears_checked(seq):
    lins = [m for m in seq if isinstance(m, torch.nn.Linear)]
    acts = [type(m).__name__ for m in seq if not isinstance(m, torch.nn.Linear)]
    assert all(a == "ELU" for a in acts), f"the C MLP supports ELU only, got {acts}"
    return lins


def actor_from_rsl(ckpt):
    from rsl_rl.runners import OnPolicyRunner

    from zang_rl.env import ZangEnv
    from zang_rl.registry import ZangRunnerCfg
    run_dir = os.path.dirname(os.path.abspath(ckpt))
    cfg = _env_cfg_from_run(run_dir)
    rc = os.path.join(run_dir, "runner_cfg.json")
    runner_cfg = json.load(open(rc)) if os.path.exists(rc) else dataclasses.asdict(ZangRunnerCfg())
    runner_cfg["nworld"] = 1
    runner = OnPolicyRunner(ZangEnv(cfg), runner_cfg, log_dir=None, device="cpu")
    runner.load(ckpt, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
                map_location="cpu")
    pol = runner.alg.get_policy()
    norm = pol.obs_normalizer
    lins = _linears_checked(pol.mlp)
    n = lins[0].in_features
    if isinstance(norm, torch.nn.Identity):
        mean, inv_std = np.zeros(n), np.ones(n)
    else:
        mean = norm._mean.squeeze(0).numpy()
        inv_std = 1.0 / (norm._std.squeeze(0).numpy() + norm.eps)
    return _Actor(mean, inv_std, lins), cfg, dict(clip=0.0, gain=1.0)


def actor_from_sb3(model_zip):
    import pickle

    from stable_baselines3 import PPO
    run_dir = os.path.dirname(os.path.abspath(model_zip))
    cfg = _env_cfg_from_run(run_dir)
    model = PPO.load(model_zip, device="cpu")
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    mean, var, eps = vn.obs_rms.mean, vn.obs_rms.var, vn.epsilon
    lins = _linears_checked(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]
    for m in model.policy.mlp_extractor.policy_net:
        if not isinstance(m, torch.nn.Linear):
            assert type(m).__name__ == "ELU", "train SB3 with activation_fn=nn.ELU for the C MLP"
    return _Actor(mean, 1.0 / np.sqrt(var + eps), lins, clip=1.0, gain=cfg.gym_action_gain), cfg, \
        dict(clip=1.0, gain=cfg.gym_action_gain)


def actor_from_brax(npz_path):
    """Actor of a brax PPO run (zang_rl/brax_train.py): normalise -> Dense/ELU stack -> clip(+-1) -> gain. Same
    post-processing as the SB3 path: the MJX env applies clip(a, -1, 1) * gym_action_gain to the raw output."""
    d = np.load(npz_path)
    cfg = _env_cfg_from_run(os.path.dirname(os.path.abspath(npz_path)))
    lins = []
    for i in range(int(d["n_layers"])):
        w, b = d[f"W{i}"], d[f"b{i}"]                      # flax kernel is (in, out); torch weight is (out, in)
        lin = torch.nn.Linear(w.shape[0], w.shape[1])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w.T)))
            lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)))
        lins.append(lin)
    inv_std = 1.0 / d["std"]
    return _Actor(d["mean"], inv_std, lins, clip=1.0, gain=cfg.gym_action_gain), cfg, \
        dict(clip=1.0, gain=cfg.gym_action_gain)


# =================================================================================================== writers
def _c_array(name, a):
    a = np.asarray(a, np.float32).ravel()
    body = ",\n    ".join(", ".join(f"{v:.9e}f" for v in a[i:i + 8]) for i in range(0, len(a), 8))
    return f"static const float {name}[{len(a)}] = {{\n    {body}\n}};\n"


def write_c_header(actor: _Actor, cfg, zm, path):
    lins = list(actor.linears)
    widths = [l.in_features for l in lins] + [lins[-1].out_features]
    out = ["/* GENERATED by ai_stack/rl/export_policy.py - do not edit. */", "#pragma once", "",
           f"#define ZP_OBS_DIM {lins[0].in_features}", f"#define ZP_ACT_DIM {lins[-1].out_features}",
           f"#define ZP_N_LAYERS {len(lins)}", f"#define ZP_MAX_WIDTH {max(widths)}",
           f"#define ZP_ACTION_SCALE {cfg.action_scale:.9e}f   /* target = home + scale * action */",
           f"#define ZP_HAS_ACTION_CLIP {1 if actor.clip > 0 else 0}",
           f"#define ZP_ACTION_CLIP {actor.clip:.9e}f          /* raw MLP output clamp (SB3 policies) */",
           f"#define ZP_ACTION_GAIN {actor.gain:.9e}f          /* applied after the clamp */",
           f"#define ZP_CTRL_PERIOD_MS {int(round(cfg.ctrl_dt * 1000))}",
           f"#define ZP_HAS_IMU {1 if cfg.imu else 0}   /* 1: obs starts with gyro(3), gravity(3) */", "",
           "/* joint order (wire order): " + ", ".join(zm.joint_names) + " */",
           _c_array("zp_home_q", zm.home_q), _c_array("zp_obs_mean", actor.mean.numpy()),
           _c_array("zp_obs_inv_std", actor.inv_std.numpy())]
    for i, l in enumerate(lins):
        out.append(_c_array(f"zp_w{i}", l.weight.detach().numpy()))      # row-major [out][in]
        out.append(_c_array(f"zp_b{i}", l.bias.detach().numpy()))
    n = len(lins)
    out.append("static const int zp_layer_in[ZP_N_LAYERS] = {" + ", ".join(str(l.in_features) for l in lins) + "};")
    out.append("static const int zp_layer_out[ZP_N_LAYERS] = {" + ", ".join(str(l.out_features) for l in lins) + "};")
    out.append("static const float *const zp_w[ZP_N_LAYERS] = {" + ", ".join(f"zp_w{i}" for i in range(n)) + "};")
    out.append("static const float *const zp_b[ZP_N_LAYERS] = {" + ", ".join(f"zp_b{i}" for i in range(n)) + "};")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return sum(l.in_features * l.out_features for l in lins)


def check_c_vs_onnx(onnx_path, header_dir, n_obs, n=200):
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, (n, n_obs)).astype(np.float32)
    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name
    ref = np.concatenate([sess.run(None, {name: o[None]})[0] for o in obs])
    exe = os.path.join(header_dir, "test_policy_host")
    subprocess.run(["gcc", "-O2", "-std=c99", f"-I{P4_SRC}", f"-I{header_dir}", os.path.join(P4_SRC, "zang_policy.c"),
                    os.path.join(P4_SRC, "test_policy_host.c"), "-lm", "-o", exe], check=True)
    fi, fo = os.path.join(header_dir, "obs.f32"), os.path.join(header_dir, "act.f32")
    obs.tofile(fi)
    subprocess.run([exe, fi, fo], check=True)
    got = np.fromfile(fo, np.float32).reshape(n, -1)
    for p in (fi, fo, exe):
        os.remove(p)
    return float(np.abs(got - ref).max())


def export(ckpt: str, out_dir: str | None = None):
    from zang_rl import obs as O
    from zang_rl import robot
    out_dir = out_dir or os.path.dirname(os.path.abspath(ckpt))
    trainer = "brax" if ckpt.endswith(".npz") else "sb3" if ckpt.endswith(".zip") else "rsl_rl"
    actor, cfg, post = {"brax": actor_from_brax, "sb3": actor_from_sb3, "rsl_rl": actor_from_rsl}[trainer](ckpt)
    actor.eval()
    zm = robot.build(imu_body=cfg.imu_body)
    n_obs = actor.linears[0].in_features
    assert n_obs == O.obs_dim(cfg.imu), f"policy input {n_obs} != layout {O.obs_dim(cfg.imu)} (imu={cfg.imu})"
    onnx_path = os.path.join(out_dir, "policy.onnx")
    kw = dict(input_names=["obs"], output_names=["action"], opset_version=18)
    try:
        torch.onnx.export(actor, (torch.zeros(1, n_obs),), onnx_path, external_data=False, **kw)
    except TypeError:                                      # older torch (e.g. a Colab image): no external_data argument
        torch.onnx.export(actor, (torch.zeros(1, n_obs),), onnx_path, **kw)
    macs = write_c_header(actor, cfg, zm, os.path.join(out_dir, "p4", "policy_weights.h"))
    meta = {
        "checkpoint": os.path.abspath(ckpt), "trainer": trainer,
        "task": cfg.task, "imu": cfg.imu, "obs_dim": n_obs, "act_dim": zm.nj,
        "obs_layout": {k: [s.start, s.stop] for k, s in O.layout(cfg.imu).items()},
        "joint_order": zm.joint_names, "home_q": zm.home_q.tolist(),
        "joint_lower": zm.lower.tolist(), "joint_upper": zm.upper.tolist(),
        "action_scale": cfg.action_scale, "action_output_clip": post["clip"], "action_output_gain": post["gain"],
        "ctrl_dt": cfg.ctrl_dt, "imu_body": cfg.imu_body, "mlp_macs": macs,
        "command_path": "target = home + action_scale * action; ZLink JOINT_TARGETS (0x10), move_ms = ctrl period",
    }
    with open(os.path.join(out_dir, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    err = check_c_vs_onnx(onnx_path, os.path.join(out_dir, "p4"), n_obs)
    print(f"exported {onnx_path}, policy_meta.json, p4/policy_weights.h ({macs} MACs/step); "
          f"C MLP vs onnxruntime max |diff| = {err:.2e}")
    if err > 1e-4:
        sys.exit("C MLP does not match ONNX")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="rsl_rl model_*.pt, SB3 *.zip (with vecnormalize.pkl beside it) or brax policy_brax.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    export(a.checkpoint, a.out)
