# Zang AI stack. `make` lists the targets.
SHELL := /bin/bash
PY := .venv/bin/python
# studio HTTP port
PORT ?= 8790
# robot / emulator ZLink UDP port (TCP bulk channel = ROBOT_PORT + 1)
ROBOT_PORT ?= 4210
# robot address and joint for servo-id / sim2real
HOST ?= 127.0.0.1
JOINT ?= left_elbow
# pid files + logs of the background services
RUN := out/run
# every port this stack can hold: studio, ZLink UDP + TCP, TensorBoard (rl), headless Chrome (screenshot)
PORTS := $(PORT)/tcp $(ROBOT_PORT)/udp $(shell echo $$(($(ROBOT_PORT) + 1)))/tcp 6006/tcp 9333/tcp
# process patterns `make stop` kills; passed through the environment, so the pattern never appears in a command
# line that pgrep could match (e.g. the recipe's own shell)
export ZANG_STOP_PATTERNS := zang_ai.studio|zang_ai.link.emulator|tensorboard --logdir|zang-shot-

.DEFAULT_GOAL := help
.PHONY: help setup models studio app emulator emulator-physics up _bg restart status logs stop test vectors faces \
        goldens servo-id sim2real screenshot clean

help:  ## list targets
	@grep -E '^[a-z0-9%-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-18s %s\n", $$1, $$2}'

# ---- setup ---------------------------------------------------------------------------------------------------------
setup:  ## create .venv and install everything (CPU only)
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu

models:  ## link the voice / whisper / gesture models (reuses ~/ZingAIStack/models if present)
	@mkdir -p models
	@for d in piper whisper voice hf; do \
	  if [ ! -e models/$$d ] && [ -d $$HOME/ZingAIStack/models/$$d ]; then ln -s $$HOME/ZingAIStack/models/$$d models/$$d; fi; \
	done; ls -l models

# ---- run in the foreground (Ctrl+C to stop) ---------------------------------------------------------------------------
studio:  ## motion studio in the browser -> http://localhost:8790
	$(PY) -m zang_ai.studio --port $(PORT)

app:  ## motion studio as a standalone desktop window (Chrome/Chromium app mode)
	$(PY) -m zang_ai.studio --port $(PORT) --app

emulator:  ## software robot speaking ZLink on udp 4210 + tcp 4211
	$(PY) -m zang_ai.link.emulator --port $(ROBOT_PORT)

emulator-physics:  ## same, servos simulated by the MuJoCo twin
	$(PY) -m zang_ai.link.emulator --port $(ROBOT_PORT) --physics

# ---- run in the background ------------------------------------------------------------------------------------------
up:  ## studio + emulator in the background (logs in out/run/), ready at http://localhost:8790
	@mkdir -p $(RUN)
	@$(MAKE) -s --no-print-directory _bg NAME=emulator CMD="$(PY) -m zang_ai.link.emulator --port $(ROBOT_PORT)"
	@$(MAKE) -s --no-print-directory _bg NAME=studio CMD="$(PY) -m zang_ai.studio --port $(PORT)"
	@for i in $$(seq 1 90); do grep -q "ready" $(RUN)/studio.log 2>/dev/null && break; sleep 0.5; done
	@grep -h "ready" $(RUN)/studio.log || { echo "studio did not start, see $(RUN)/studio.log"; exit 1; }

_bg:
	@if [ -f $(RUN)/$(NAME).pid ] && kill -0 $$(cat $(RUN)/$(NAME).pid) 2>/dev/null; then \
	  echo "$(NAME) already running (pid $$(cat $(RUN)/$(NAME).pid))"; \
	else \
	  PYTHONUNBUFFERED=1 setsid nohup $(CMD) > $(RUN)/$(NAME).log 2>&1 < /dev/null & echo $$! > $(RUN)/$(NAME).pid; \
	  echo "$(NAME) started (pid $$(cat $(RUN)/$(NAME).pid), log $(RUN)/$(NAME).log)"; \
	fi

restart: stop up  ## stop everything, then start studio + emulator again

status:  ## background services, and who holds the stack's ports
	@for n in studio emulator; do \
	  if [ -f $(RUN)/$$n.pid ] && kill -0 $$(cat $(RUN)/$$n.pid) 2>/dev/null; then echo "$$n: running (pid $$(cat $(RUN)/$$n.pid))"; \
	  else echo "$$n: stopped"; fi; done
	@for p in $(PORTS); do pids=$$(fuser $$p 2>/dev/null | xargs); \
	  if [ -n "$$pids" ]; then echo "port $$p: used by $$(ps -o pid=,comm= -p $${pids// /,} | xargs)"; else echo "port $$p: free"; fi; done

logs:  ## last lines of the background services' logs
	@tail -n 25 $(RUN)/*.log 2>/dev/null || echo "no logs in $(RUN)"

stop:  ## kill all studio / emulator / TensorBoard / screenshot processes and free ports 8790, 4210, 4211, 6006, 9333
	@for n in studio emulator; do \
	  if [ -f $(RUN)/$$n.pid ]; then kill $$(cat $(RUN)/$$n.pid) 2>/dev/null && echo "stopped $$n (pid $$(cat $(RUN)/$$n.pid))"; rm -f $(RUN)/$$n.pid; fi; done
	@pids=$$(pgrep -f "$$ZANG_STOP_PATTERNS" | xargs); \
	  if [ -n "$$pids" ]; then echo "killing:"; ps -o pid=,args= -p $${pids// /,} | cut -c1-100; kill $$pids 2>/dev/null; sleep 0.5; fi
	@for p in $(PORTS); do pids=$$(fuser $$p 2>/dev/null | xargs); \
	  if [ -n "$$pids" ]; then echo "port $$p held by $$pids -> kill"; fuser -k -TERM $$p >/dev/null 2>&1; sleep 0.3; fuser -k -KILL $$p >/dev/null 2>&1; fi; done
	@left=""; for p in $(PORTS); do [ -n "$$(fuser $$p 2>/dev/null)" ] && left="$$left $$p"; done; \
	  if [ -n "$$left" ]; then echo "still busy:$$left"; exit 1; else echo "all ports free: $(PORTS)"; fi

# ---- checks + contracts ---------------------------------------------------------------------------------------------
test:  ## unit + integration tests
	$(PY) -m pytest -q

vectors:  ## regenerate the ZLink golden vectors for the firmware host tests
	$(PY) tools/gen_zlink_vectors.py

faces:  ## regenerate the face golden images for the firmware's C renderer test
	$(PY) tools/gen_face_goldens.py

goldens: vectors faces  ## both (after changing ZLINK.md / zlink.py or faces.json / face/render.py)

# ---- robot tools ----------------------------------------------------------------------------------------------------
servo-id:  ## identify one servo over ZLink (HOST=, JOINT=; WRITE=1 updates config/motion.json)
	$(PY) tools/servo_id.py --host $(HOST) --port $(ROBOT_PORT) --joint $(JOINT) $(if $(WRITE),--write,)

sim2real:  ## play the same motion on the robot and in the twin, then compare (HOST=)
	$(PY) tools/sim2real_replay.py --host $(HOST) --port $(ROBOT_PORT)

screenshot:  ## screenshot the running studio after a compose -> out/studio.png
	$(PY) tools/screenshot.py out/studio.png "http://localhost:$(PORT)/#compose&seek=1.2"

# ---- misc -----------------------------------------------------------------------------------------------------------
rl-%:  ## run a target of rl/Makefile, e.g. make rl-help, make rl-bundle
	$(MAKE) -C rl $*

clean:  ## remove generated audio, emulator output and service logs
	rm -f out/_*.wav out/emulator_display.jpg
	rm -rf $(RUN)
