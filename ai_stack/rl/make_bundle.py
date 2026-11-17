"""Build the upload zip for the Colab GPU notebook (train_colab_gpu.ipynb): only what that notebook needs.

  .venv/bin/python make_bundle.py            # -> zang_rl_colab_bundle.zip (next to this file)
  .venv/bin/python make_bundle.py --check    # also verify the zip: MANIFEST hashes, clean unzip, build the models from it

Zip layout. The tree under Zang/ mirrors the repository, so the relative paths in zang_rl/robot.py keep working:
  MANIFEST.json                                   sha256 + size of every other file (the notebook verifies them)
  requirements_colab.txt                          pinned dependencies for the Colab runtime
  Zang/ai_stack/config/robot.json                 joint order contract
  Zang/ai_stack/rl/zang_rl/*.py                   the task (core), the MJX env, the Gymnasium env (parity check), helpers
  Zang/ai_stack/rl/assets/servo_ds_r003g.json     servo parameters (PLACEHOLDERS until identified)
  Zang/ai_stack/rl/export_policy.py, sim2sim.py   export (ONNX + C header) and sim-to-sim evaluation
  Zang/ai_stack/rl/export/p4/*                    the P4 C MLP that export_policy.py compiles to check the header
  Zang/zang_description/mjcf/{scene,zang}.xml     the robot
  Zang/zang_description/meshes/collision/*.stl    only sim2sim.py needs these (full-contact physics)

Not included on purpose: the rsl_rl env (env.py, registry.py, rsl_cfg.py; need torch + rsl_rl), tests, trainers, logs,
the visual meshes (23 MB). zang_rl.robot.build(strip_visuals=True) deletes the visual geoms and their meshes before
compiling, and with box feet and feet-only collisions no mesh remains at all, so training reads no STL file.

The zip is reproducible: entries are sorted and carry a fixed timestamp.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

RL = os.path.dirname(os.path.abspath(__file__))
ZANG = os.path.dirname(os.path.dirname(RL))
DEFAULT_OUT = os.path.join(RL, "zang_rl_colab_bundle.zip")
FIXED_TIME = (2026, 1, 1, 0, 0, 0)

ZANG_RL_FILES = ["__init__.py", "config.py", "core.py", "obs.py", "robot.py", "gym_env.py", "mjx_env.py",
                 "brax_train.py"]


def collect() -> dict:
    """zip path -> source path, for everything in the bundle except MANIFEST.json."""
    out = {"requirements_colab.txt": os.path.join(RL, "requirements_colab.txt"),
           "Zang/ai_stack/config/robot.json": os.path.join(ZANG, "ai_stack", "config", "robot.json")}
    rl = "Zang/ai_stack/rl/"
    for n in ZANG_RL_FILES:
        out[f"{rl}zang_rl/{n}"] = os.path.join(RL, "zang_rl", n)
    out[f"{rl}assets/servo_ds_r003g.json"] = os.path.join(RL, "assets", "servo_ds_r003g.json")
    for n in ("export_policy.py", "sim2sim.py"):
        out[rl + n] = os.path.join(RL, n)
    for n in ("zang_policy.c", "zang_policy.h", "test_policy_host.c"):
        out[f"{rl}export/p4/{n}"] = os.path.join(RL, "export", "p4", n)
    desc = os.path.join(ZANG, "zang_description")
    for n in ("scene.xml", "zang.xml"):
        out[f"Zang/zang_description/mjcf/{n}"] = os.path.join(desc, "mjcf", n)
    mesh_dir = os.path.join(desc, "meshes", "collision")
    for n in sorted(os.listdir(mesh_dir)):
        out[f"Zang/zang_description/meshes/collision/{n}"] = os.path.join(mesh_dir, n)
    missing = [s for s in out.values() if not os.path.isfile(s)]
    if missing:
        sys.exit(f"missing source files: {missing}")
    return dict(sorted(out.items()))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out: str) -> str:
    files = collect()
    blobs = {}
    for arc, src in files.items():
        with open(src, "rb") as f:
            blobs[arc] = f.read()
    manifest = {
        "bundle": "zang_rl_colab_bundle", "format": 1,
        "note": "sha256 of every file in the zip except this one; train_colab_gpu.ipynb verifies all of them",
        "n_files": len(blobs),
        "files": [{"path": p, "sha256": sha256(b), "size": len(b)} for p, b in blobs.items()],
    }
    blobs["MANIFEST.json"] = (json.dumps(manifest, indent=1) + "\n").encode()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc in ["MANIFEST.json"] + list(files):
            zi = zipfile.ZipInfo(arc, date_time=FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, blobs[arc], compresslevel=9)
    raw = sum(len(b) for b in blobs.values())
    print(f"{out}: {len(blobs)} files ({len(files)} listed in MANIFEST.json), "
          f"{os.path.getsize(out) / 1e6:.2f} MB zipped, {raw / 1e6:.2f} MB unpacked")
    return out


def check(zip_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            assert z.testzip() is None, "corrupt zip"
            z.extractall(tmp)
        man = json.load(open(os.path.join(tmp, "MANIFEST.json")))
        listed = {f["path"] for f in man["files"]}
        on_disk = {os.path.relpath(os.path.join(r, n), tmp).replace(os.sep, "/")
                   for r, _, ns in os.walk(tmp) for n in ns} - {"MANIFEST.json"}
        assert listed == on_disk, f"manifest / zip mismatch: {listed ^ on_disk}"
        for f in man["files"]:
            with open(os.path.join(tmp, f["path"]), "rb") as fh:
                data = fh.read()
            assert sha256(data) == f["sha256"] and len(data) == f["size"], f"bad hash {f['path']}"
        print(f"manifest OK: {len(listed)} files hashed")
        rl = os.path.join(tmp, "Zang", "ai_stack", "rl")
        # compile-only checks from the clean unzip (no stepping): the training model, the sim2sim models that
        # use the collision meshes, and the Gymnasium env
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import gymnasium as gym, zang_rl; from zang_rl import robot, obs; "
            "zm = robot.build(); n_geom = zm.model.ngeom; "
            "za = robot.build(collisions='all', timestep=0.001); "
            "env = gym.make('ZangStand-v0').unwrapped; "
            "assert env.observation_space.shape[0] == obs.obs_dim(False) == 50; "
            "print('bundle OK: training model nq=%d ngeom=%d (no meshes), sim2sim model ngeom=%d, obs %d' % "
            "(zm.model.nq, n_geom, za.model.ngeom, env.observation_space.shape[0]))")
        subprocess.run([sys.executable, "-I", "-c", code, rl], cwd=tmp, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = build(a.out)
    if a.check:
        check(path)
