# Zang humanoid: digital twin workspace

- `Zang Assembly v1.0.2.step` / `.f3z`: source CAD from Fusion. It has **no joints**; they are defined in code.
- `zang_description/`: generated robot model.
  - `zang.urdf` for Gazebo/ROS.
  - `mjcf/scene.xml` for MuJoCo, with keyframe `home`.
  - `meshes/`
  - `tools/`: the STEP → URDF/MJCF pipeline. `build.py` holds the joint definitions and the placeholder masses and limits.
- `IK/`: kinematics library, maths, figures and tests. **Read `IK/CLAUDE.md` first** for conventions and pitfalls.
- `.venv/`: Python environment (cadquery/OCP, trimesh, mujoco, scipy, matplotlib, yourdfpy). Use `.venv/bin/python`.
- `firmware/`: ESP-IDF v5.5.6. `zang_p4/` is the ESP32-P4 main app; `zang_c2_hosted/` is the ESP32-C2 Wi-Fi co-processor
  (ESP-Hosted over UART). `docs/` holds the board schematic, `HARDWARE_REVIEW.md` (pin map and board bugs), `ZLINK.md`
  and `datasheets/`. See `firmware/README.md`.
- `ai_stack/`: its own venv `ai_stack/.venv`. It holds co-speech motion, the safety stage, the physics twin, the Motion
  Studio (`make studio`), the ZLink client and emulator, and `rl/` (MuJoCo Playground). See `ai_stack/README.md`.

Contracts between firmware and ai_stack: `ai_stack/config/robot.json` (joint order and IDs; the firmware header is
generated from it) and `firmware/docs/ZLINK.md` (wire protocol; golden vectors via `ai_stack/tools/gen_zlink_vectors.py`).
Change the contract first, then both sides.

Hardware v0.0.1 has **no IMU**, and its servo bus is not wired to the P4 (rework: GPIO47/48). Details are in
`firmware/docs/HARDWARE_REVIEW.md`.

Key fact: each half-leg is a parallelogram linkage. That makes the robot a series-parallel hybrid. The lower leg servo (`knee_pitch`) is what the user calls the knee motor.
