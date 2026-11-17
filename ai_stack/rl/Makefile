# Zang RL: MuJoCo Playground-style stand / walk policies.  `make help` lists the targets.
PY      := .venv/bin/python
BASE    := /home/rushikesh/ZingAIStack/.venv-playground/lib/python3.12/site-packages
RUN     ?= $(lastword $(sort $(wildcard logs/ZangStand/*/)))
CKPT    ?= $(lastword $(sort $(wildcard $(RUN)model_*.pt)))

.PHONY: help setup check check-gym smoke train-stand train-stand-imu train-joystick train-sb3 export sim2sim bundle tensorboard

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## create .venv layered on the Playground venv (no second torch)
	python3.12 -m venv .venv
	echo "$(BASE)" > .venv/lib/python3.12/site-packages/zz_playground_base.pth
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install --no-deps rsl-rl-lib==5.4.2 stable-baselines3==2.9.2

check:  ## physics sanity: loop closure, sensors, throughput
	$(PY) check_env.py

check-gym:  ## gymnasium env_checker on ZangStand-v0 and ZangWalk-v0
	$(PY) -c "import gymnasium as gym, zang_rl; from gymnasium.utils.env_checker import check_env; [check_env(gym.make(i).unwrapped) or print(i, 'OK') for i in ('ZangStand-v0', 'ZangWalk-v0')]"

smoke:  ## 5-iteration training + export, proves the pipeline (~1 min)
	$(PY) train.py --env ZangStand --iters 5 --nworld 16 --steps-per-env 24

train-stand:  ## full stand/push-recovery training, no IMU (hardware v0.0.1); hours on 4 CPU cores
	$(PY) train.py --env ZangStand

train-stand-imu:  ## same with an IMU in the observation (future board revision)
	$(PY) train.py --env ZangStandIMU

train-joystick:  ## velocity-tracking walk, initialised from a stand policy: make train-joystick CKPT=...
	$(PY) train.py --env ZangJoystick $(if $(CKPT),--resume $(CKPT))

train-sb3:  ## SB3 PPO on the Gymnasium env ZangStand-v0 (4 subprocess workers, ~2.8 GB RAM)
	$(PY) train_sb3.py --env ZangStand-v0 --steps 5000000 --n-envs 4

export:  ## ONNX + P4 C header for CKPT (default: newest ZangStand checkpoint)
	$(PY) export_policy.py $(CKPT)

sim2sim:  ## run RUN's policy.onnx in different physics (dt 1 ms, RK4, full mesh contacts)
	$(PY) sim2sim.py --policy $(RUN)policy.onnx --push 0.5

bundle:  ## zang_rl_colab_bundle.zip for train_colab_gpu.ipynb, verified (manifest + clean unzip)
	$(PY) make_bundle.py --check

tensorboard:  ## training curves on http://localhost:6006
	$(PY) -m tensorboard.main --logdir logs
