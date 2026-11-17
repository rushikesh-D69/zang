# IK: kinematics for the Zang humanoid

This folder holds the forward kinematics, analytic inverse kinematics and whole-body IK for Zang, a 1.34 kg biped with 15 DS-R003G servos. It is meant for future control work: gait generation, ROS nodes and the digital twin. The maths is in [IK_MATH.md](IK_MATH.md); read it before changing the solvers.

## Files

| file | what it is |
|---|---|
| `zang_ik.py` | The library, class `Zang`. It needs only numpy and reads `../zang_description/zang.urdf` at load. |
| `IK_MATH.md` | Derivations, frames, sign conventions, pin geometry, validation results. |
| `joints.csv`, `links.csv` | Machine-readable joint table (origin, axis, limits, home value, mimic) and link table (mass, CoM, inertia). Regenerate with `python zang_ik.py`. |
| `test_ik.py` | Self-checks: FK vs MuJoCo, IK round trips, whole-body, balance, numeric IK, HOME vs keyframe. Needs mujoco. |
| `demo_mujoco.py` | Drives the MuJoCo sim through IK. Options: `--view` (viewer), `--render out.png` (contact sheet). |
| `verify_linkage.py` | Checks the series-parallel leg model against the true closed four-bar loops, both an exact solve and a closed-chain MuJoCo sim. Results are in IK_MATH.md §3.4. |
| `make_figures.py`, `figures/` | Kinematic graph (root body, active, passive and loop joints), leg mechanism side view, whole-robot skeleton. Drawn from the live model. |

Run everything with the venv at `/home/rushikesh/Zang/.venv`, for example `../.venv/bin/python test_ik.py`.

## API at a glance

```python
from zang_ik import Zang, HOME
z = Zang()
F = z.fk(q, base_pos, base_rot)          # {link: 4x4 world transform}; q = dict of servo angles, pins auto-filled
z.sole_pos(F, "left"); z.hand_pos(F, "left"); z.com(F)
z.leg_ik("left", sole_pos_in_pelvis, foot_roll=0.0)        # -> 4 leg servo angles
z.arm_ik("left", hand_pos_in_chest)                        # -> 3 arm servo angles
z.whole_body_ik(pelvis_pos, pelvis_rpy, feet={side: (sole_world, roll)}, hands={side: pos_world}, waist_yaw=0)
z.balance_ik(com_xy, pelvis_z, feet)                       # moves the pelvis so the CoM sits over com_xy
z.numeric_ik({link: (point_in_link, world_target)}, joints=[...])
```

The solvers raise `IKError` when a target is unreachable within joint limits.

## Facts that are easy to get wrong

- **Each half-leg is a parallelogram linkage, not a rigid shank.** The knee block translates without rotating, and the foot always keeps the pelvis's pitch and yaw. The 8 passive pins are mimic joints:
  - `knee_upper = -hip_pitch`
  - `thigh_shell = +hip_pitch`
  - `knee_lower = -knee_pitch`
  - `shin_shell = -knee_pitch`
- **The mimic relations assume ideal parallelograms.** The CAD pin spacings are slightly off (60.43 vs 60.00 mm), which costs up to 1.4° of foot pitch in the working range. `verify_linkage.py` quantifies this. The user has not yet measured the real parts.
- **`knee_pitch` is the lower leg servo** (the user calls it the "downside motor"). There is no separate ankle-pitch servo. Each leg has 4 servos: hip_roll, hip_pitch, knee_pitch, ankle_roll.
- **No hip yaw.** With flat feet, the pelvis pitch must be 0 and the feet share the pelvis's yaw. The robot cannot turn in place without sliding a foot.
- **Rest pose:** `HOME` = hip_pitch = knee_pitch = -0.5 on both legs (knees bent forward), which is the MJCF keyframe `home`. If you change it, update both `HOME` here and `HOME` in `../zang_description/tools/build.py`; `test_ik.py` checks they match.
- **Axis directions are not mirrored.** Positive rotation is right-handed about +x/+y/+z of the robot frame for both sides. Firmware needs per-servo sign and offset mapping; IK_MATH.md §1 has the table.
- **Dynamics are placeholders.** Servo mass, torque, speed and limits are guesses, defined in `../zang_description/tools/build.py`.

## Where the model comes from

`../zang_description/` is generated from `/home/rushikesh/Zang/Zang Assembly v1.0.2.step`. The CAD has no joints, so `build.py` defines them. To regenerate after a CAD change:

```
cd ../zang_description
../.venv/bin/python -I tools/extract.py "../Zang Assembly v1.0.2.step" build/parts.pkl
../.venv/bin/python -I tools/contacts.py build/parts.pkl
../.venv/bin/python -I tools/holes.py "../Zang Assembly v1.0.2.step" build/holes.pkl "Thigh:1" "Thigh:2" "Thigh:3" "Thigh:4" "Knee:1" "Knee:2" "Leg Link:1" "Leg Link:2" "Leg Link:3" "Leg Link:4" "Leg Link:5" "Leg Link:6" "Leg Link:7" "Leg Link:8" "Hip Link 2:1" "Hip Link 1:1" "Hip Link 1:2" "Hip Link 2:2"
../.venv/bin/python -I tools/build.py && ../.venv/bin/python -I tools/check_mujoco.py
```

Then rerun `test_ik.py` here. `zang_ik.py` asserts the joint topology it was derived for (`_check_structure`), so a structural change fails loudly instead of returning wrong angles.
