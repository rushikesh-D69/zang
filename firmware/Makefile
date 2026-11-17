# Zang firmware: ESP32-P4 main app + ESP32-C2 Wi-Fi co-processor. `make` lists the targets.
# Every IDF target sources the IDF environment itself, so no `source export.sh` is needed first.
SHELL := /bin/bash
IDF_ACTIVATE ?= $(HOME)/.espressif/tools/activate_idf_v5.5.6.sh
IDF := source $(IDF_ACTIVATE) >/dev/null &&
# parallel compile jobs (15 GB RAM: keep low)
JOBS ?= 2
# P4 USB-Serial-JTAG (GPIO24/25)
PORT ?= /dev/ttyACM0
# USB-UART adapter on the C2's U0TXD/U0RXD (the C2 is not reachable over the P4's USB)
C2_PORT ?= /dev/ttyUSB0
BAUD ?= 460800
P4 := zang_p4
C2 := zang_c2_hosted
TEST_BUILD := $(P4)/test_host/build
# processes that hold serial ports or JTAG, never the compilers (so `make stop` cannot break a running build).
# Passed through the environment, so the pattern never appears in a command line pgrep could match.
export ZANG_STOP_PATTERNS := idf_monitor|idf.py .*monitor|esptool|openocd

.DEFAULT_GOAL := help
.PHONY: help gen gen-check build test p4 p4-build p4-flash p4-monitor p4-flash-monitor p4-menuconfig p4-size \
        p4-erase p4-clean c2 c2-build c2-flash c2-menuconfig c2-clean goldens datasheets ports stop clean

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-18s %s\n", $$1, $$2}'
	@echo "  variables: PORT=$(PORT) C2_PORT=$(C2_PORT) JOBS=$(JOBS) IDF_ACTIVATE=$(IDF_ACTIVATE)"

# ---- generated sources (contracts live in ai_stack/config) --------------------------------------------------------
gen:  ## regenerate robot_joints, board pins and the face library from their sources
	python3 tools/gen_robot_joints.py
	python3 tools/gen_board_pins.py
	python3 tools/gen_faces.py

gen-check:  ## fail if any generated file is stale (CI)
	python3 tools/gen_robot_joints.py --check
	python3 tools/gen_board_pins.py --check
	python3 tools/gen_faces.py --check

goldens:  ## regenerate ZLink golden vectors + face goldens from the Python side (ai_stack)
	$(MAKE) -C ../ai_stack goldens

# ---- everything ----------------------------------------------------------------------------------------------------
build: p4-build c2-build  ## build the P4 app and the C2 co-processor

test: gen-check  ## host unit tests (ZLink vectors, servo packets, limiter, faces) with gcc + ctest
	cmake -S $(P4)/test_host -B $(TEST_BUILD)
	cmake --build $(TEST_BUILD) -j$(JOBS)
	ctest --test-dir $(TEST_BUILD) --output-on-failure

# ---- ESP32-P4 ----------------------------------------------------------------------------------------------------
p4: p4-build  ## = p4-build

p4-build: gen-check  ## build the P4 firmware
	cd $(P4) && $(IDF) idf.py build -- -j$(JOBS)

p4-flash:  ## flash the P4 over USB (PORT=)
	cd $(P4) && $(IDF) idf.py -p $(PORT) -b $(BAUD) flash

p4-monitor:  ## serial monitor of the P4 (Ctrl+] to quit)
	cd $(P4) && $(IDF) idf.py -p $(PORT) monitor

p4-flash-monitor:  ## flash, then monitor
	cd $(P4) && $(IDF) idf.py -p $(PORT) -b $(BAUD) flash monitor

p4-menuconfig:  ## board pins, display panel, network, ... (menuconfig)
	cd $(P4) && $(IDF) idf.py menuconfig

p4-size:  ## firmware size breakdown
	cd $(P4) && $(IDF) idf.py size

p4-erase:  ## erase the whole P4 flash (NVS calibration too!)
	cd $(P4) && $(IDF) idf.py -p $(PORT) erase-flash

p4-clean:  ## remove the P4 build directory
	cd $(P4) && $(IDF) idf.py fullclean

# ---- ESP32-C2 (ESP-Hosted slave) ---------------------------------------------------------------------------------
c2: c2-build  ## = c2-build

c2-build:  ## build the C2 ESP-Hosted firmware
	cd $(C2) && $(IDF) idf.py build -- -j$(JOBS)

c2-flash:  ## flash the C2 through a USB-UART adapter on its UART0 (C2_PORT=); hold BOOT_C2 low or let the P4 do it
	cd $(C2) && $(IDF) idf.py -p $(C2_PORT) -b $(BAUD) flash

c2-menuconfig:  ## ESP-Hosted transport settings (UART pins, baud)
	cd $(C2) && $(IDF) idf.py menuconfig

c2-clean:  ## remove the C2 build directory
	cd $(C2) && $(IDF) idf.py fullclean

# ---- tools -----------------------------------------------------------------------------------------------------------
datasheets:  ## re-download the datasheets listed in docs/datasheets/INDEX.md
	tools/fetch_datasheets.sh

ports:  ## list serial ports and which process holds each
	@found=0; for p in /dev/ttyACM* /dev/ttyUSB*; do [ -e "$$p" ] || continue; found=1; \
	  pids=$$(fuser $$p 2>/dev/null | xargs); \
	  if [ -n "$$pids" ]; then echo "$$p: held by $$(ps -o pid=,comm= -p $${pids// /,} | xargs)"; else echo "$$p: free"; fi; \
	done; [ $$found = 1 ] || echo "no serial ports"

stop:  ## kill monitors / esptool / openocd and free the serial ports (builds are left alone)
	@pids=$$(pgrep -f "$$ZANG_STOP_PATTERNS" | xargs); \
	  if [ -n "$$pids" ]; then echo "killing: $$(ps -o pid=,args= -p $${pids// /,} | cut -c1-90)"; kill $$pids 2>/dev/null; sleep 0.5; fi
	@for p in $$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null); do \
	  if [ -n "$$(fuser $$p 2>/dev/null)" ]; then echo "$$p held by $$(fuser $$p 2>/dev/null | xargs) -> kill"; fuser -k $$p >/dev/null 2>&1; fi; done
	@echo "serial ports free"

clean: p4-clean c2-clean  ## remove all build directories (P4, C2, host tests)
	rm -rf $(TEST_BUILD)
