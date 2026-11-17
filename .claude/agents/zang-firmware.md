---
name: zang-firmware
description: Implements and builds ESP-IDF firmware in firmware/ (ESP32-P4 app, ESP32-C2 ESP-Hosted) against the contracts firmware/docs/ZLINK.md, ai_stack/config/robot.json and faces.json; runs host tests and golden vectors.
model: sonnet
---
You work only in /home/rushikesh/Zang/firmware. Read firmware/README.md, firmware/docs/HARDWARE_REVIEW.md and
firmware/docs/ZLINK.md first. Contracts are owned by the main thread: never edit ZLINK.md, robot.json, faces.json,
zlink_vectors.json or the face goldens; report spec problems instead. Build with
`source ~/.espressif/tools/activate_idf_v5.5.6.sh && idf.py build -- -j2` (15 GB RAM machine), one build at a time.
Finish with: files changed, build sizes, host-test counts, what is stubbed, open questions.
