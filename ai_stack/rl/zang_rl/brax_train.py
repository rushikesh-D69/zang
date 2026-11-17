"""Helpers for training Zang with brax PPO on the MJX env (the Colab GPU notebook, train_colab_gpu.ipynb).

Only glue lives here; the task is zang_rl/core.py and the env is zang_rl/mjx_env.py.

  network_factory()        policy and value networks for brax's ppo.train (small ELU actor for the ESP32-P4)
  save_checkpoint()        numpy checkpoint, written atomically (safe on a Google Drive mount), newest few kept
  latest_checkpoint()      newest checkpoint in a folder, for resuming
  load_checkpoint()
  export_actor_npz()       actor + observation normaliser as plain numpy, read by export_policy.py (no jax needed)

Why a numpy pickle instead of brax's orbax checkpoint: orbax writes many small files and renames directories,
which is slow and fragile on the Drive FUSE mount, and `ppo.train(restore_params=...)` takes the very same
(normalizer, policy, value) tuple. Optimizer state and the step counter are not restored: Adam restarts with
fresh moments and the step count is carried in the checkpoint file instead.

Why the 'normal' action distribution and not brax's default tanh-normal: the policy output goes through the same
`clip(-1, 1) * gym_action_gain` as the Gymnasium/SB3 path, which is exactly what the P4 C MLP implements
(zang_policy.c has no tanh). `noise_std_type='log'` keeps the exploration std positive (exp of a parameter).

Checked against the installed brax 0.14.2 source (training/networks.py, agents/ppo/networks.py, train.py).
"""
from __future__ import annotations

import functools
import glob
import os
import pickle
import tempfile

import numpy as np

OBS_KEY = "state"                      # actor observation (IMU-free, 50 values by default)
VALUE_OBS_KEY = "privileged_state"     # critic observation (training only)


def network_factory(policy_hidden=(128, 128), value_hidden=(256, 256, 128), init_noise_std=0.8):
    """brax `network_factory` for ppo.train. The actor is obs -> 128 -> 128 -> 15 with ELU: about 25 k MACs,
    the size the ESP32-P4 C MLP (export/p4) runs in well under 1 ms."""
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from flax import linen

    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(policy_hidden),
        value_hidden_layer_sizes=tuple(value_hidden),
        activation=linen.elu,
        policy_obs_key=OBS_KEY,
        value_obs_key=VALUE_OBS_KEY,
        distribution_type="normal",
        noise_std_type="log",
        init_noise_std=init_noise_std,
        # small initial actions: start close to the home pose
        mean_kernel_init_fn=jax.nn.initializers.variance_scaling,
        mean_kernel_init_kwargs=dict(scale=1e-2, mode="fan_in", distribution="uniform"),
    )


# ================================================================================================ checkpoints
def _ckpt_name(steps: int) -> str:
    return f"ckpt_{int(steps):012d}.pkl"


def save_checkpoint(directory: str, steps: int, params, meta: dict | None = None, keep: int = 3) -> str:
    """params = brax (normalizer_params, policy_params, value_params). Returns the file path."""
    import jax

    os.makedirs(directory, exist_ok=True)
    blob = {"steps": int(steps), "params": jax.tree.map(np.asarray, params), "meta": meta or {}}
    path = os.path.join(directory, _ckpt_name(steps))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_ckpt_")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                      # atomic: a half-written file never looks like a checkpoint
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    for old in sorted(glob.glob(os.path.join(directory, "ckpt_*.pkl")))[:-keep]:
        os.remove(old)
    return path


def latest_checkpoint(directory: str) -> str | None:
    files = sorted(glob.glob(os.path.join(directory, "ckpt_*.pkl")))
    return files[-1] if files else None


def load_checkpoint(path: str) -> dict:
    """{'steps', 'params', 'meta'}. Only load files you wrote yourself (pickle)."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ===================================================================================== actor for export_policy
def actor_arrays(params, obs_key: str = OBS_KEY) -> dict:
    """Plain-numpy view of the actor: observation mean/std and the Dense layers (flax kernels are (in, out))."""
    norm, policy = params[0], params[1]
    mean = np.asarray(norm.mean[obs_key], np.float32)
    std = np.asarray(norm.std[obs_key], np.float32)
    p = policy["params"]
    mlp = [k for k in p if k.startswith("MLP")]
    assert len(mlp) == 1 and "Dense_0" in p, f"unexpected policy parameter layout: {list(p)}"
    assert "Dense_1" not in p, "state-dependent std is not supported by the P4 C MLP"
    hidden = sorted(p[mlp[0]], key=lambda k: int(k.split("_")[1]))
    layers = [p[mlp[0]][k] for k in hidden] + [p["Dense_0"]]
    out = {"mean": mean, "std": std, "n_layers": np.int64(len(layers))}
    for i, l in enumerate(layers):
        out[f"W{i}"] = np.asarray(l["kernel"], np.float32)
        out[f"b{i}"] = np.asarray(l["bias"], np.float32)
    return out


def export_actor_npz(params, path: str, obs_key: str = OBS_KEY) -> str:
    np.savez(path, **actor_arrays(params, obs_key))
    return path
